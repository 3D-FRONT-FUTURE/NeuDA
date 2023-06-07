import torch
import cv2 as cv
import numpy as np
import os
from glob import glob
from scipy.spatial.transform import Rotation as Rot
from scipy.spatial.transform import Slerp
import pdb


# This function is borrowed from IDR: https://github.com/lioryariv/idr
def load_K_Rt_from_P(filename, P=None):
    if P is None:
        lines = open(filename).read().splitlines()
        if len(lines) == 4:
            lines = lines[1:]
        lines = [[x[0], x[1], x[2], x[3]] for x in (x.split(" ") for x in lines)]
        P = np.asarray(lines).astype(np.float32).squeeze()

    out = cv.decomposeProjectionMatrix(P)
    K = out[0]
    R = out[1]
    t = out[2]

    K = K / K[2, 2]
    intrinsics = np.eye(4)
    intrinsics[:3, :3] = K

    pose = np.eye(4, dtype=np.float32)
    pose[:3, :3] = R.transpose()
    pose[:3, 3] = (t[:3] / t[3])[:, 0]

    return intrinsics, pose


class Dataset:
    def __init__(self, conf, img_step=1):
        super(Dataset, self).__init__()
        print('Load data: Begin')
        self.device = torch.device('cuda')
        self.conf = conf

        self.data_dir = conf.get_string('data_dir')
        self.render_cameras_name = conf.get_string('render_cameras_name')
        self.object_cameras_name = conf.get_string('object_cameras_name')

        self.camera_outside_sphere = conf.get_bool('camera_outside_sphere', default=True)
        self.scale_mat_scale = conf.get_float('scale_mat_scale', default=1.1)
        self.loader_mode = conf.get_string('loader_mode', default='image')

        camera_dict = np.load(os.path.join(self.data_dir, self.render_cameras_name))
        self.camera_dict = camera_dict

        self.images_lis = sorted(glob(os.path.join(self.data_dir, 'image/*.png')))
        n_all_images = len(self.images_lis)
        self.images_lis = self.images_lis[::img_step]

        self.n_images = len(self.images_lis)
        self.images_np = np.stack([cv.imread(im_name) for im_name in self.images_lis]) / 256.0
        self.masks_lis = sorted(glob(os.path.join(self.data_dir, 'mask/*.png')))[::img_step]
        
        if len(self.masks_lis) == 0:
            self.masks_np = np.stack([np.ones_like(im) for im in self.images_np])
        else:
            self.masks_np = np.stack([cv.imread(im_name) for im_name in self.masks_lis]) / 256.0

        # world_mat is a projection matrix from world to image
        self.world_mats_np = [camera_dict['world_mat_%d' % (idx)].astype(np.float32) for idx in range(0, n_all_images, img_step)]

        self.scale_mats_np = []

        # scale_mat: used for coordinate normalization, we assume the scene to render is inside a unit sphere at origin.
        self.scale_mats_np = [camera_dict['scale_mat_%d' % (idx)].astype(np.float32) for idx in range(0, n_all_images, img_step)]

        self.intrinsics_all = []
        self.pose_all = []
        llff_poses = []

        for scale_mat, world_mat in zip(self.scale_mats_np, self.world_mats_np):
            P = world_mat @ scale_mat
            P = P[:3, :4]
            intrinsics, pose = load_K_Rt_from_P(None, P)
            self.intrinsics_all.append(torch.from_numpy(intrinsics).float())
            self.pose_all.append(torch.from_numpy(pose).float())
            # c2w
            _intrinsics, _c2w = load_K_Rt_from_P(None, world_mat[:3, :4])
            llff_poses.append(_c2w)

        llff_poses = np.stack(llff_poses).astype(np.float32)
        self.llff_poses = np.concatenate([llff_poses[..., 0:1], -llff_poses[..., 1:2], -llff_poses[..., 2:3], llff_poses[..., 3:4]], -1)

        self.images = torch.from_numpy(self.images_np.astype(np.float32)).cpu()  # [n_images, H, W, 3]
        self.masks  = torch.from_numpy(self.masks_np.astype(np.float32)).cpu()   # [n_images, H, W, 3]
        self.intrinsics_all = torch.stack(self.intrinsics_all).to(self.device)   # [n_images, 4, 4]
        self.intrinsics_all_inv = torch.inverse(self.intrinsics_all)  # [n_images, 4, 4]
        self.focal = self.intrinsics_all[0][0, 0]
        self.pose_all = torch.stack(self.pose_all).to(self.device)  # [n_images, 4, 4]
        self.H, self.W = self.images.shape[1], self.images.shape[2]
        self.image_pixels = self.H * self.W

        object_bbox_min = np.array([-1.01, -1.01, -1.01, 1.0])
        object_bbox_max = np.array([ 1.01,  1.01,  1.01, 1.0])
        # Object scale mat: region of interest to **extract mesh**
        object_scale_mat = np.load(os.path.join(self.data_dir, self.object_cameras_name))['scale_mat_0']
        object_bbox_min = np.linalg.inv(self.scale_mats_np[0]) @ object_scale_mat @ object_bbox_min[:, None]
        object_bbox_max = np.linalg.inv(self.scale_mats_np[0]) @ object_scale_mat @ object_bbox_max[:, None]
        self.object_bbox_min = object_bbox_min[:3, 0]
        self.object_bbox_max = object_bbox_max[:3, 0]
        print(self.object_bbox_min, self.object_bbox_max)
        # rays sampling variable
        self.rays_count_list = []
        self.valid_xyxy = np.zeros((self.n_images, 4), dtype=np.int32)
        self.n_rays = 0
        if self.loader_mode == 'ray':
            self.init_all_rays()

        print('Load data: End')

    def gen_rays_at(self, img_idx, resolution_level=1):
        """
        Generate rays at world space from one camera.
        """
        l = resolution_level
        tx = torch.linspace(0, self.W - 1, self.W // l)
        ty = torch.linspace(0, self.H - 1, self.H // l)
        pixels_x, pixels_y = torch.meshgrid(tx, ty)
        p = torch.stack([pixels_x, pixels_y, torch.ones_like(pixels_y)], dim=-1) # W, H, 3
        p = torch.matmul(self.intrinsics_all_inv[img_idx, None, None, :3, :3], p[:, :, :, None]).squeeze()  # W, H, 3
        rays_v = p / torch.linalg.norm(p, ord=2, dim=-1, keepdim=True)  # W, H, 3
        rays_v = torch.matmul(self.pose_all[img_idx, None, None, :3, :3], rays_v[:, :, :, None]).squeeze()  # W, H, 3
        rays_o = self.pose_all[img_idx, None, None, :3, 3].expand(rays_v.shape)  # W, H, 3
        return rays_o.transpose(0, 1), rays_v.transpose(0, 1)


    def _sample_rays_from_bbox(self, img_idx, pad_ratio=0.1):
        if any(self.valid_xyxy[img_idx] != 0):
            return self.valid_xyxy[img_idx].tolist()

        mask = self.masks[img_idx] # H x W x 3
        valid_mask = mask[:,:,0] > 0.5

        # get bbox
        _valid_mask = valid_mask.view(self.H, self.W).numpy().astype(np.uint8)
        valid_ij = np.where(_valid_mask == 1)
        # bbox padding
        _v_h_p = (valid_ij[0].max() - valid_ij[0].min()) * pad_ratio
        _v_w_p = (valid_ij[1].max() - valid_ij[1].min()) * pad_ratio

        valid_i_min = max(int(valid_ij[0].min() - _v_h_p ), 0) # y_min
        valid_i_max = min(int(valid_ij[0].max() + _v_h_p ), self.H-1) # y_max
        valid_j_min = max(int(valid_ij[1].min() - _v_w_p ), 0) # x_min
        valid_j_max = min(int(valid_ij[1].max() + _v_w_p ), self.W-1) # x_max
        
        self.rays_count_list += [
            (valid_j_max - valid_j_min + 1) * (valid_i_max - valid_i_min + 1)
        ]
        self.valid_xyxy[img_idx] = np.array([valid_j_min, valid_i_min, valid_j_max, valid_i_max], dtype=np.int32)

        return self.valid_xyxy[img_idx].tolist() # (x0, y0, x1, y1)


    def gen_random_rays_at_image(self, img_idx, batch_size, precrop_ratio=1., precrop_step=0, iter_step=0, use_valid_bbox=True):
        """
        Generate random rays at world space from one camera.
        """
        if precrop_ratio == 1. or precrop_step == 0 or iter_step >= precrop_step:
            pixels_x = torch.randint(low=0, high=self.W, size=[batch_size])
            pixels_y = torch.randint(low=0, high=self.H, size=[batch_size])
        else:
            ratio = precrop_ratio * (1. - iter_step / precrop_step) + 1. * (iter_step / precrop_step)
            ratio = (1 - ratio) * 0.5
            start_x, end_x = int(ratio * self.W), int((1 - ratio) * self.W)
            start_y, end_y = int(ratio * self.H), int((1 - ratio) * self.H)
            pixels_x = torch.randint(low=start_x, high=end_x, size=[batch_size])
            pixels_y = torch.randint(low=start_y, high=end_y, size=[batch_size])

        if use_valid_bbox:
            start_x, start_y, end_x, end_y = self._sample_rays_from_bbox(img_idx)
            pixels_x = torch.randint(low=start_x, high=end_x, size=[batch_size])
            pixels_y = torch.randint(low=start_y, high=end_y, size=[batch_size])

        color = self.images[img_idx][(pixels_y, pixels_x)]    # batch_size, 3
        mask = self.masks[img_idx][(pixels_y, pixels_x)]      # batch_size, 3
        p = torch.stack([pixels_x, pixels_y, torch.ones_like(pixels_y)], dim=-1).float()  # batch_size, 3
        p = torch.matmul(self.intrinsics_all_inv[img_idx, None, :3, :3], p[:, :, None]).squeeze() # batch_size, 3
        rays_v = p / torch.linalg.norm(p, ord=2, dim=-1, keepdim=True)    # batch_size, 3
        rays_v = torch.matmul(self.pose_all[img_idx, None, :3, :3], rays_v[:, :, None]).squeeze()  # batch_size, 3
        rays_o = self.pose_all[img_idx, None, :3, 3].expand(rays_v.shape) # batch_size, 3
        return torch.cat([rays_o.cpu(), rays_v.cpu(), color, mask[:, :1]], dim=-1).cuda()    # batch_size, 10

    def init_all_rays(self, use_valid_bbox=True):        
        self.all_rays = []
        for img_idx in range(self.n_images):
            if use_valid_bbox:
                start_x, start_y, end_x, end_y = self._sample_rays_from_bbox(img_idx)
            else:
                start_x, start_y, end_x, end_y = 0, 0, self.W, self.H
            pixels_x = torch.arange(start=start_x, end=end_x)
            pixels_y = torch.arange(start=start_y, end=end_y)
            color = self.images[img_idx][(pixels_y, pixels_x)]    # batch_size, 3
            mask = self.masks[img_idx][(pixels_y, pixels_x)]      # batch_size, 3
            p = torch.stack([pixels_x, pixels_y, torch.ones_like(pixels_y)], dim=-1).float()  # batch_size, 3
            p = torch.matmul(self.intrinsics_all_inv[img_idx, None, :3, :3], p[:, :, None]).squeeze() # batch_size, 3
            rays_v = p / torch.linalg.norm(p, ord=2, dim=-1, keepdim=True)    # batch_size, 3
            rays_v = torch.matmul(self.pose_all[img_idx, None, :3, :3], rays_v[:, :, None]).squeeze()  # batch_size, 3
            rays_o = self.pose_all[img_idx, None, :3, 3].expand(rays_v.shape) # batch_size, 3
            image_data = torch.cat([rays_o.cpu(), rays_v.cpu(), color, mask[:, :1]], dim=-1).cuda()    # batch_size, 10
            all_rays += [image_data]
        self.all_rays = torch.cat(all_rays, dim=0) # image_data, 10
        self.n_rays = len(self.all_rays)

    def get_all_rays_perm(self, batch_size):
        rays_ids = torch.randperm(self.n_rays)
        self.rays_groups = rays_ids.split(batch_size)

    def gen_random_rays_at_allrays(self, group_idx):
        """
        Generate random rays at world space from one camera.
        """
        rays_ids = self.rays_group[group_idx]
        rays_batch = self.all_rays[rays_ids]    # batch_size, 3
        return rays_batch.cuda()    # batch_size, 10

    def gen_random_rays_at(self, idx, batch_size, precrop_ratio=1., precrop_step=0, iter_step=0, use_valid_bbox=True):
        if self.loader_mode == 'image':
            data = self.gen_random_rays_at_image(idx, batch_size, precrop_ratio, precrop_step, iter_step, use_valid_bbox)
        elif self.loader_mode == 'ray':
            data = self.gen_random_rays_at_allrays(idx)
        else:
            assert False, f'[ERROR] Unknown loader mode: {self.loader_mode} in dataset.py'
        return data

    def gen_rays_between(self, idx_0, idx_1, ratio, resolution_level=1):
        """
        Interpolate pose between two cameras.
        """
        l = resolution_level
        tx = torch.linspace(0, self.W - 1, self.W // l)
        ty = torch.linspace(0, self.H - 1, self.H // l)
        pixels_x, pixels_y = torch.meshgrid(tx, ty)
        p = torch.stack([pixels_x, pixels_y, torch.ones_like(pixels_y)], dim=-1)  # W, H, 3
        p = torch.matmul(self.intrinsics_all_inv[0, None, None, :3, :3], p[:, :, :, None]).squeeze()  # W, H, 3
        rays_v = p / torch.linalg.norm(p, ord=2, dim=-1, keepdim=True)  # W, H, 3
        trans = self.pose_all[idx_0, :3, 3] * (1.0 - ratio) + self.pose_all[idx_1, :3, 3] * ratio
        pose_0 = self.pose_all[idx_0].detach().cpu().numpy()
        pose_1 = self.pose_all[idx_1].detach().cpu().numpy()
        pose_0 = np.linalg.inv(pose_0)
        pose_1 = np.linalg.inv(pose_1)
        rot_0 = pose_0[:3, :3]
        rot_1 = pose_1[:3, :3]
        rots = Rot.from_matrix(np.stack([rot_0, rot_1]))
        key_times = [0, 1]
        slerp = Slerp(key_times, rots)
        rot = slerp(ratio)
        pose = np.diag([1.0, 1.0, 1.0, 1.0])
        pose = pose.astype(np.float32)
        pose[:3, :3] = rot.as_matrix()
        pose[:3, 3] = ((1.0 - ratio) * pose_0 + ratio * pose_1)[:3, 3]
        pose = np.linalg.inv(pose)
        rot = torch.from_numpy(pose[:3, :3]).cuda()
        trans = torch.from_numpy(pose[:3, 3]).cuda()
        rays_v = torch.matmul(rot[None, None, :3, :3], rays_v[:, :, :, None]).squeeze()  # W, H, 3
        rays_o = trans[None, None, :3].expand(rays_v.shape)  # W, H, 3
        return rays_o.transpose(0, 1), rays_v.transpose(0, 1)

    def near_far_from_sphere(self, rays_o, rays_d):
        a = torch.sum(rays_d**2, dim=-1, keepdim=True)
        b = 2.0 * torch.sum(rays_o * rays_d, dim=-1, keepdim=True)
        mid = 0.5 * (-b) / a
        near = mid - 1.0
        far = mid + 1.0
        return near, far

    def image_at(self, idx, resolution_level):
        img = cv.imread(self.images_lis[idx])
        return (cv.resize(img, (self.W // resolution_level, self.H // resolution_level))).clip(0, 255)

    def org_image_at(self, idx):
        img = cv.imread(self.images_lis[idx])
        return img

    def mask_at(self, idx):
        return self.masks[idx]

