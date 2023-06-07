import os
import numpy as np
import cv2
import open3d as o3d
from plyfile import PlyData, PlyElement
import sys
import pdb

def filter_mesh_noise(mesh_path, out_mesh_path):
    mesh = o3d.io.read_triangle_mesh(mesh_path)
    # remove noise in the mesh by keeping only the biggest cluster
    idxs, count, _ = mesh.cluster_connected_triangles()
    max_cluster_idx = np.argmax(count)
    triangles_to_remove = [i for i in range(len(mesh.triangles)) if idxs[i] != max_cluster_idx]
    mesh.remove_triangles_by_index(triangles_to_remove)
    mesh.remove_unreferenced_vertices()
    print(f'Mesh has {len(mesh.vertices)/1e6:.2f} M vertices and {len(mesh.triangles)/1e6:.2f} M faces.')
    # get vertices & triangles
    vertices_ = np.asarray(mesh.vertices).astype(np.float32)
    vertices_color_ = np.clip(np.asarray(mesh.vertex_colors) * 255.0, 0, 255).astype(np.uint8)
    triangles = np.asarray(mesh.triangles)
    # out
    vertices_.dtype = [('x', 'f4'), ('y', 'f4'), ('z', 'f4')]
    vertices_color_.dtype = [('red', 'u1'), ('green', 'u1'), ('blue', 'u1')]
    
    N_vertices = len(vertices_)
    vertex_all = np.empty(N_vertices, vertices_.dtype.descr+vertices_color_.dtype.descr)
    for prop in vertices_.dtype.names:
        vertex_all[prop] = vertices_[prop][:, 0]
    for prop in vertices_color_.dtype.names:
        vertex_all[prop] = vertices_color_[prop][:, 0]
        
    face = np.empty(len(triangles), dtype=[('vertex_indices', 'i4', (3,))])
    face['vertex_indices'] = triangles

    dirpath = os.path.dirname(out_mesh_path)
    if not os.path.exists(dirpath):
        os.makedirs(dirpath)

    PlyData([PlyElement.describe(vertex_all, 'vertex'), 
             PlyElement.describe(face, 'face')]).write(out_mesh_path)

    print('Done!')


if __name__ == '__main__':
    print(sys.argv)
    filter_mesh_noise(sys.argv[-2], sys.argv[-1])
