import os
os.environ["PYOPENGL_PLATFORM"] = "egl"
import numpy as np
import matplotlib.pyplot as plt
import cv2
import open3d as o3d
from plyfile import PlyData, PlyElement
import torch
import trimesh
import pyrender
import time
import math
from tqdm import tqdm
import pdb


def render_mesh(mesh_path, cam_extr, H, W, F, K=None, near=0.01, far=9000.0):
    """
    @param:
    cam_extr: 4*4
    """

    mesh_nerf = trimesh.load(mesh_path)

    # scene setting
    scene = pyrender.Scene()
    scene.add(pyrender.Mesh.from_trimesh(mesh_nerf, smooth=True), pose=np.eye(4))

    _cam_extr = np.eye(4)
    _cam_extr[:3, :4] = cam_extr[:3, :4]

    if K is None:
        yfov = math.atan((H/2.)/F)*2.
        camera = pyrender.PerspectiveCamera(yfov=yfov)
    else:
        fx, fy, cx, cy = K[0, 0], K[1, 1], K[0, 2], K[1, 2]
        camera = pyrender.IntrinsicsCamera(fx=fx, fy=fy, cx=cx, cy=cy, znear=near, zfar=far)
    nc = pyrender.Node(camera=camera, matrix=_cam_extr)
    scene.add_node(nc)

    r = pyrender.OffscreenRenderer(W, H)
    color, depth = r.render(scene, flags=pyrender.constants.RenderFlags.FLAT)
    r.delete()

    return color, depth


def render_dataset(dataset, mesh_path):
    W, H = dataset.W, dataset.H
    Ks = dataset.intrinsics_all.cpu().numpy()[:,:3,:3]
    print('render ...')
    depths = []
    data_num = len(dataset.images_lis)
    for idx in tqdm(range(data_num)):
        ## read the camera to world relative pose
        P_c2w = dataset.llff_poses[idx]
        # using `pyrender` for depth 
        _, depth_src = render_mesh(mesh_path, P_c2w, H, W, None, K=Ks[idx])

        depths.append(depth_src)
    return depths

def fusion_color(dataset, mesh_path, dist_thresh=5.0):
    # load mesh
    mesh = o3d.io.read_triangle_mesh(mesh_path)
    # get vertices & triangles
    vertices_ = np.asarray(mesh.vertices).astype(np.float32)
    triangles = np.asarray(mesh.triangles)

    # perform color prediction
    # Step 0. define constants (image width, height and intrinsics)
    W, H = dataset.W, dataset.H
    Ks = dataset.intrinsics_all.cpu().numpy()[:,:3,:3]
    # Step 1. transform vertices into world coordinate
    N_vertices = len(vertices_)
    vertices_homo = np.concatenate([vertices_, np.ones((N_vertices, 1))], 1) # (N, 4)
    ## use my color average method. see README_mesh.md
    ## buffers to store the final averaged color
    non_occluded_sum = np.zeros((N_vertices, 1))
    v_color_sum = np.zeros((N_vertices, 3))

    # Step 2. project the vertices onto each training image to infer the color
    print('Fusing colors ...')

    data_num = len(dataset.images_lis)
    for idx in tqdm(range(data_num)):
        
        image = cv2.imread(dataset.images_lis[idx])[:,:,::-1]
        ## read the camera to world relative pose
        P_c2w = dataset.llff_poses[idx]
        P_w2c = np.linalg.inv(P_c2w)[:3]
        K = Ks[idx]
        ## project vertices from world coordinate to camera coordinate
        vertices_cam = (P_w2c @ vertices_homo.T) # (3, N) in "right up back"
        # opengl coordinator to opencv coordinator
        vertices_cam[1:] *= -1
        ## project vertices from camera coordinate to pixel coordinate
        vertices_image = (K @ vertices_cam).T # (N, 3)
        depth = vertices_image[:, -1:]+1e-5 # the depth of the vertices, used as far plane
        vertices_image = vertices_image[:, :2]/depth
        vertices_image = vertices_image.astype(np.float32)
        vertices_image[:, 0] = np.clip(vertices_image[:, 0], 0, W-1)
        vertices_image[:, 1] = np.clip(vertices_image[:, 1], 0, H-1)

        ## compute the color on these projected pixel coordinates
        ## using bilinear interpolation.
        ## NOTE: opencv's implementation has a size limit of 32768 pixels per side,
        ## so we split the input into chunks.
        colors = []
        remap_chunk = int(3e4)
        for i in range(0, N_vertices, remap_chunk):
            colors += [cv2.remap(image, 
                                vertices_image[i:i+remap_chunk, 0],
                                vertices_image[i:i+remap_chunk, 1],
                                interpolation=cv2.INTER_LINEAR)[:, 0]]
        colors = np.vstack(colors) # (N_vertices, 3)
        # using `pyrender` for depth 
        _, depth_src = render_mesh(mesh_path, P_c2w, H, W, None, K=K)

        depths = []
        for i in range(0, N_vertices, remap_chunk):
            depths += [cv2.remap(depth_src, 
                                vertices_image[i:i+remap_chunk, 0],
                                vertices_image[i:i+remap_chunk, 1],
                                interpolation=cv2.INTER_LINEAR)]
        depths = np.vstack(depths) # (N_vertices, 3)

        non_occluded = abs(depth-depths) < dist_thresh

        v_color_sum += colors * non_occluded
        non_occluded_sum += non_occluded

    # Step 3. combine the output and write to file
    print(f'non_occluded_sum: {non_occluded_sum.max()}')
    v_colors = v_color_sum/non_occluded_sum
    v_colors = v_colors.astype(np.uint8)
    v_colors.dtype = [('red', 'u1'), ('green', 'u1'), ('blue', 'u1')]
    vertices_.dtype = [('x', 'f4'), ('y', 'f4'), ('z', 'f4')]
    vertex_all = np.empty(N_vertices, vertices_.dtype.descr+v_colors.dtype.descr)
    for prop in vertices_.dtype.names:
        vertex_all[prop] = vertices_[prop][:, 0]
    for prop in v_colors.dtype.names:
        vertex_all[prop] = v_colors[prop][:, 0]
        
    face = np.empty(len(triangles), dtype=[('vertex_indices', 'i4', (3,))])
    face['vertex_indices'] = triangles

    PlyData([PlyElement.describe(vertex_all, 'vertex'), 
             PlyElement.describe(face, 'face')]).write(mesh_path)

    textured_mesh = o3d.io.read_triangle_mesh(mesh_path)
    obj_mesh_path = mesh_path.replace('.ply',  '.obj')
    o3d.io.write_triangle_mesh(obj_mesh_path, textured_mesh)
    print('Done!')
