#  NeuralWarp  All rights reseved to Thales LAS and ENPC.
#
#  This code is freely available for academic use only and Provided “as is” without any warranty.
#
#  Modification are allowed for academic research provided that the following conditions are met :
#    * Redistributions of source code or any format must retain the above copyright notice and this list of conditions.
#    * Neither the name of Thales LAS and ENPC nor the names of its contributors may be used to endorse or promote products derived from this software without specific prior written permission.

from . import mesh_filtering
from tqdm import tqdm
import torch
import numpy as np
from skimage import measure
import trimesh



def get_grid_uniform(resolution, bbox_size):
    x = np.linspace(-bbox_size, bbox_size, resolution)
    y = x
    z = x

    xx, yy, zz = np.meshgrid(x, y, z)
    grid_points = torch.tensor(np.vstack([xx.ravel(), yy.ravel(), zz.ravel()]).T, dtype=torch.float)

    return {"grid_points": grid_points.cuda(),
            "shortest_axis_length": 2.0,
            "xyz": [x, y, z],
            "shortest_axis_index": 0}


def get_grid(points, resolution):
    eps = 0.2
    input_min = torch.min(points, dim=0)[0].squeeze().numpy()
    input_max = torch.max(points, dim=0)[0].squeeze().numpy()

    bounding_box = input_max - input_min
    shortest_axis = np.argmin(bounding_box)
    if (shortest_axis == 0):
        x = np.linspace(input_min[shortest_axis] - eps,
                        input_max[shortest_axis] + eps, resolution)
        length = np.max(x) - np.min(x)
        y = np.arange(input_min[1] - eps, input_max[1] + length / (x.shape[0] - 1) + eps, length / (x.shape[0] - 1))
        z = np.arange(input_min[2] - eps, input_max[2] + length / (x.shape[0] - 1) + eps, length / (x.shape[0] - 1))
    elif (shortest_axis == 1):
        y = np.linspace(input_min[shortest_axis] - eps,
                        input_max[shortest_axis] + eps, resolution)
        length = np.max(y) - np.min(y)
        x = np.arange(input_min[0] - eps, input_max[0] + length / (y.shape[0] - 1) + eps, length / (y.shape[0] - 1))
        z = np.arange(input_min[2] - eps, input_max[2] + length / (y.shape[0] - 1) + eps, length / (y.shape[0] - 1))
    elif (shortest_axis == 2):
        z = np.linspace(input_min[shortest_axis] - eps,
                        input_max[shortest_axis] + eps, resolution)
        length = np.max(z) - np.min(z)
        x = np.arange(input_min[0] - eps, input_max[0] + length / (z.shape[0] - 1) + eps, length / (z.shape[0] - 1))
        y = np.arange(input_min[1] - eps, input_max[1] + length / (z.shape[0] - 1) + eps, length / (z.shape[0] - 1))

    xx, yy, zz = np.meshgrid(x, y, z)
    grid_points = torch.tensor(np.vstack([xx.ravel(), yy.ravel(), zz.ravel()]).T, dtype=torch.float)
    return {"grid_points": grid_points,
            "shortest_axis_length": length,
            "xyz": [x, y, z],
            "shortest_axis_index": shortest_axis}


def get_surface_high_res_mesh(sdf, resolution=100, refine_bb=True, cams=None, masks=None,
                              bbox_size=1.):
    # get low res mesh to sample point cloud
    grid = get_grid_uniform(100 if refine_bb else resolution, bbox_size)
    z = []
    points = grid['grid_points']

    for i, pnts in enumerate(tqdm((torch.split(points, 100000, dim=0)), desc="Low resolution evaluation")):
        z.append(sdf(pnts).detach().cpu().numpy())
    z = np.concatenate(z, axis=0)

    z = z.astype(np.float32)

    verts, faces, normals, values = measure.marching_cubes(
        volume=z.reshape(grid['xyz'][1].shape[0], grid['xyz'][0].shape[0],
                         grid['xyz'][2].shape[0]).transpose([1, 0, 2]),
        level=0,
        gradient_direction="ascent",
        spacing=(grid['xyz'][0][2] - grid['xyz'][0][1],
                 grid['xyz'][0][2] - grid['xyz'][0][1],
                 grid['xyz'][0][2] - grid['xyz'][0][1]))

    verts = verts + np.array([grid['xyz'][0][0], grid['xyz'][1][0], grid['xyz'][2][0]])

    mesh_low_res = trimesh.Trimesh(verts, faces, vertex_normals=normals)

    if not refine_bb:
        return mesh_low_res

    # remove from low res mesh triangles with that never projects or outside of visual hull
    visual_hull = mesh_filtering.visual_hull_mask(torch.from_numpy(mesh_low_res.vertices).float().cuda(),
                                                  cams, masks, nb_visible=1)
    face_mask = visual_hull[mesh_low_res.faces].any(axis=1).cpu().numpy()
    face_mask = face_mask & mesh_filtering.visible_mask(mesh_low_res, cams, nb_visible=1).cpu().numpy()
    mesh_low_res.update_faces(face_mask)

    components = mesh_low_res.split(only_watertight=False)
    areas = np.array([c.area for c in components], dtype=np.float)
    mesh_low_res = components[areas.argmax()]

    recon_pc = trimesh.sample.sample_surface(mesh_low_res, 10000)[0]
    recon_pc = torch.from_numpy(recon_pc).float().cuda()

    # Center the pc
    s_mean = recon_pc.mean(dim=0)
    vecs = torch.eye(3).cuda()
    helper = recon_pc - s_mean

    # high resolution grid
    grid_aligned = get_grid(helper.cpu(), resolution)

    points = grid_aligned['grid_points']
    chunk_size = 5000
    nb_pts = points.shape[0]
    z = []

    for i, pnts in enumerate(tqdm(torch.split(points, chunk_size, dim=0), desc="High resolution evaluation")):
        pnts = pnts.cuda()
        tmp = (vecs.unsqueeze(0).expand(pnts.shape[0], -1, -1).transpose(1, 2) @ pnts.unsqueeze(-1)).squeeze() + s_mean
        points[chunk_size * i:min(chunk_size * (i+1), nb_pts)] = tmp
        sdf_vals = sdf(tmp).detach()
        z.append(sdf_vals.cpu().numpy())

    z = np.concatenate(z, axis=0)

    if not (np.min(z) > 0 or np.max(z) < 0):
        z = z.astype(np.float32)

        verts, faces, normals, values = measure.marching_cubes(
            volume=z.reshape(grid_aligned['xyz'][1].shape[0], grid_aligned['xyz'][0].shape[0],
                             grid_aligned['xyz'][2].shape[0]).transpose([1, 0, 2]),
            level=0,
            gradient_direction="ascent",
            spacing=(grid_aligned['xyz'][0][2] - grid_aligned['xyz'][0][1],
                     grid_aligned['xyz'][0][2] - grid_aligned['xyz'][0][1],
                     grid_aligned['xyz'][0][2] - grid_aligned['xyz'][0][1]))

        verts = torch.from_numpy(verts).float().cuda()
        verts = torch.bmm(vecs.unsqueeze(0).repeat(verts.shape[0], 1, 1).transpose(1, 2),
                   verts.unsqueeze(-1)).squeeze()
        verts = (verts + points[0].cuda())

        mesh = trimesh.Trimesh(verts.cpu().numpy(), faces, vertex_normals=normals)

        return mesh
