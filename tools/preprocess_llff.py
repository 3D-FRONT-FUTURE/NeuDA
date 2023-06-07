import sys
import os
import numpy as np
import trimesh

import colmap_read_model as read_model


def load_colmap_data(realdir):

    camerasfile = os.path.join(realdir, 'cameras.bin')
    camdata = read_model.read_cameras_binary(camerasfile)

    list_of_keys = list(camdata.keys())
    cam = camdata[list_of_keys[0]]
    print( 'Cameras', len(cam))

    h, w, f = cam.height, cam.width, cam.params[0]
    hwf = np.array([h,w,f]).reshape([3,1])

    imagesfile = os.path.join(realdir, 'images.bin')
    imdata = read_model.read_images_binary(imagesfile)

    w2c_mats = []
    bottom = np.array([0,0,0,1.]).reshape([1,4])

    names = [imdata[k].name for k in imdata]
    print( 'Images #', len(names))
    perm = np.argsort(names)
    for k in imdata:
        im = imdata[k]
        R = im.qvec2rotmat()
        t = im.tvec.reshape([3,1])
        m = np.concatenate([np.concatenate([R, t], 1), bottom], 0)
        w2c_mats.append(m)

    w2c_mats = np.stack(w2c_mats, 0)
    c2w_mats = np.linalg.inv(w2c_mats)

    poses = c2w_mats[:, :3, :4].transpose([1,2,0])
    poses = np.concatenate([poses, np.tile(hwf[..., np.newaxis], [1,1,poses.shape[-1]])], 1)

    points3dfile = os.path.join(realdir, 'points3D.bin')
    pts3d = read_model.read_points3d_binary(points3dfile)

    # must switch to [-u, r, -t] from [r, -u, t], NOT [r, u, -t]
    poses = np.concatenate([poses[:, 1:2, :], poses[:, 0:1, :], -poses[:, 2:3, :], poses[:, 3:4, :], poses[:, 4:5, :]], 1)

    return poses, pts3d, perm


def preprocess_llff_pose(basedir):
    # load sparse PCD
    points3dfile = os.path.join(basedir, 'sparse/points3D.bin')
    pts3d = read_model.read_points3d_binary(points3dfile)
    pts_arr = [pts3d[k].xyz for k in pts3d]
    pts = np.stack(pts_arr, axis=0)

    pcd = trimesh.PointCloud(pts)
    pcd.export(os.path.join(basedir, 'sparse_points.ply'))

    vertices = pcd.vertices
    bbox_max = np.max(vertices, axis=0)
    bbox_min = np.min(vertices, axis=0)
    center = (bbox_max + bbox_min) * 0.5
    radius = np.linalg.norm(vertices - center, ord=2, axis=-1).max() * 1.5
    scale_mat = np.diag([radius, radius, radius, 1.0]).astype(np.float32)
    scale_mat[:3, 3] = center

    poses_hwf = np.load(os.path.join(basedir, 'poses_bounds.npy')) # n_images, 3, 5
    poses_hwf = poses_hwf[:, :15].reshape(-1, 3, 5)
    poses_raw = poses_hwf[:, :, :4]
    hwf = poses_hwf[:, :, 4]
    pose = np.diag([1.0, 1.0, 1.0, 1.0])
    pose[:3, :4] = poses_raw[0]

    cam_dict = dict()
    n_images = len(poses_raw)

    # Convert space
    convert_mat = np.zeros([4, 4], dtype=np.float32)
    convert_mat[0, 1] = 1.0
    convert_mat[1, 0] = 1.0
    convert_mat[2, 2] =-1.0
    convert_mat[3, 3] = 1.0

    for i in range(n_images):
        pose = np.diag([1.0, 1.0, 1.0, 1.0]).astype(np.float32)
        pose[:3, :4] = poses_raw[i]
        pose = pose @ convert_mat
        h, w, f = hwf[i, 0], hwf[i, 1], hwf[i, 2]
        intrinsic = np.diag([f, f, 1.0, 1.0]).astype(np.float32)
        intrinsic[0, 2] = (w - 1) * 0.5
        intrinsic[1, 2] = (h - 1) * 0.5
        w2c = np.linalg.inv(pose)
        world_mat = intrinsic @ w2c
        world_mat = world_mat.astype(np.float32)
        cam_dict['camera_mat_{}'.format(i)] = intrinsic
        cam_dict['camera_mat_inv_{}'.format(i)] = np.linalg.inv(intrinsic)
        cam_dict['world_mat_{}'.format(i)] = world_mat
        cam_dict['world_mat_inv_{}'.format(i)] = np.linalg.inv(world_mat)

    for i in range(n_images):
        cam_dict['scale_mat_{}'.format(i)] = scale_mat
        cam_dict['scale_mat_inv_{}'.format(i)] = np.linalg.inv(scale_mat)

    np.savez(os.path.join(basedir, 'cameras_sphere.npz'), **cam_dict)
    print('Process done!')


if __name__ == '__main__':
    print(sys.argv)
    basedir = sys.argv[1]
    preprocess_llff_pose(basedir)