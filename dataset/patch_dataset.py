import torch
import torch.nn.functional as F
from torchvision import transforms as T
import cv2 as cv
import time
import numpy as np
import os
from glob import glob
import traceback
from models.texture.texture_mesh import MeshTexture
from models.utils import render_dataset
from .collate import default_collate
from tqdm import tqdm
import shutil
import pickle
import pdb



class DataProvider():
    def __init__(self, dataset, batch_size, **kw):
        self.args = kw
        self.dataset = dataset
        self.data_len = len(self.dataset)
        self.epoch = 0
        self.iteration = 0
        self.batch_size = batch_size
        self.data_ids = None
        self.cur_idx = 0
        self.build()
    
    def build(self):
        self.iteration = 0
        self.data_ids = np.random.permutation(self.data_len)

    def __next__(self):
        if self.iteration < self.data_len and self.iteration + self.batch_size >= self.data_len:
            batch_size = self.data_len - self.iteration
        elif self.iteration + self.batch_size < self.data_len:
            batch_size = self.batch_size
        else:
            self.epoch += 1
            self.iteration = 0
            self.build()
            raise StopIteration

        batch_data = []
        # get data
        for _ in range(batch_size):
            data_item = self.__next_item__()
            batch_data.append(data_item)

        return default_collate(batch_data)
        
    def __next_item__(self):
        if self.data_ids is None:
            self.build()
        
        idx = self.data_ids[self.iteration]
        batch = self.dataset.__getitem__(idx)
        # img, label = batch
        self.iteration += 1
        return batch
    
    next = __next__

    def __iter__(self):
        return self


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

####################################################
#
#           Adversarial Texture
#
####################################################

class TextureDataset(torch.utils.data.Dataset):
    def __init__(self, conf, tex_dim=1024, patch_size=256, val_interval=30, mode='train'):
        super(TextureDataset, self).__init__()
        print('Load data: Begin')
        self.device = torch.device('cuda')
        self.mode = mode
        self.conf = conf

        self.data_dir = conf.get_string('data_dir')
        self.render_cameras_name = conf.get_string('render_cameras_name')
        self.object_cameras_name = conf.get_string('object_cameras_name')

        self.camera_outside_sphere = conf.get_bool('camera_outside_sphere', default=True)
        self.scale_mat_scale = conf.get_float('scale_mat_scale', default=1.1)

        camera_dict = np.load(os.path.join(self.data_dir, self.render_cameras_name))
        self.camera_dict = camera_dict
        self.images_all_lis = sorted(glob(os.path.join(self.data_dir, 'image/*.png')))
        self.images_lis = self.images_all_lis
        # ------ data partition --------
        # training list
        self.train_idx = [_ for _ in range(0, len(self.images_all_lis))]
        self.train_lis = [self.images_all_lis[_] for _ in self.train_idx]
        if mode == 'train':
            self.images_lis = self.train_lis
            self.idx_lis = self.train_idx
        # val using some training data for a simple testing
        elif mode == 'val':
            self.images_lis = self.train_lis[::val_interval]
            self.idx_lis = self.train_idx[::val_interval]
        else:
            assert False, f'Unknwon dataset mode:{mode}'
        self.n_images = len(self.images_lis)
        self.images_np = np.stack([cv.imread(im_name) for im_name in self.images_lis]) / 255.0
        self.masks_lis = sorted(glob(os.path.join(self.data_dir, 'mask/*.png')))
        self.masks_lis = [self.masks_lis[_] for _ in self.idx_lis]
        if len(self.masks_lis) == 0:
            self.masks_np = np.stack([np.ones_like(im) for im in self.images_np])
        else:
            self.masks_np = np.stack([cv.imread(im_name) for im_name in self.masks_lis]) / 255.0

        # world_mat is a projection matrix from world to image
        self.world_mats_np = [camera_dict['world_mat_%d' % (idx)].astype(np.float32) for idx in self.idx_lis]

        self.scale_mats_np = []

        # scale_mat: used for coordinate normalization, we assume the scene to render is inside a unit sphere at origin.
        self.scale_mats_np = [camera_dict['scale_mat_%d' % (idx)].astype(np.float32) for idx in self.idx_lis]

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

        llff_poses = np.stack(llff_poses).astype(np.float32)[:, :3, :4]
        self.llff_poses = np.concatenate([llff_poses[..., 0:1], -llff_poses[..., 1:2], -llff_poses[..., 2:3], llff_poses[..., 3:4]], -1)

        self.images = torch.from_numpy(self.images_np.astype(np.float32))  # [n_images, H, W, 3]
        self.masks  = torch.from_numpy(self.masks_np.astype(np.float32))   # [n_images, H, W, 3]
        self.intrinsics_all = torch.stack(self.intrinsics_all)  # [n_images, 4, 4]
        self.intrinsics_all_inv = torch.inverse(self.intrinsics_all)  # [n_images, 4, 4]
        self.focal = self.intrinsics_all[0][0, 0]
        self.pose_all = torch.stack(self.pose_all)  # [n_images, 4, 4]
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
        

        # ------ mesh texture initialization --------
        # 
        self.object_path = conf.get_string('object_path')
        self.object_dir = os.path.dirname(self.object_path)
        self.texture_cache_dir = os.path.join(os.path.dirname(self.object_dir), 'cache')
        case_name = conf.get_string('case_name')
        intrinsic = self.intrinsics_all[0].cpu().numpy()
        self.mesh_texture = MeshTexture(obj_path=self.object_path, name=case_name, intrinsic=intrinsic, tex_dim=tex_dim, img_hw=[self.H, self.W])
        # init texture data
        cam2worlds = np.array([
            np.concatenate([P_c2w, np.array([0, 0, 0, 1]).reshape(1, 4)], 0) 
            for P_c2w in self.llff_poses
        ]) # list
        self.cam2worlds = torch.from_numpy(cam2worlds)
        _npy_num = len(list(glob(os.path.join(self.texture_cache_dir, '0*.npy')))) if os.path.exists(self.texture_cache_dir) else 0
        if not os.path.exists(self.texture_cache_dir):
            print("generate texture data & all view uv&mask...")
            init_texture, depths = self.generate_texture_data(cam2worlds)
            self.refine_texture_path = os.path.join(self.texture_cache_dir, f'{case_name}_init_tex.png')
            # render all view data
            all_uv, all_mask = self.generate_all_view(cam2worlds)
            self._save_render_data(all_uv, all_mask, depths)
            cv.imwrite(self.refine_texture_path, init_texture)
        else:
            print("load texture data & all view uv&mask...")
            all_uv, all_mask, depths = [], [], []
            try:
                val_interval = 1 if mode == 'train' else val_interval
                for idx in range(0, len(self.train_lis), val_interval):
                    uv, mask, depth = self._load_render_data(idx)
                    all_uv.append(uv)
                    all_mask.append(mask)
                    depths.append(depth)
                all_uv = np.stack(all_uv, axis=0)
                all_mask = np.stack(all_mask, axis=0)
                depths = np.stack(depths, axis=0)
            except Exception as e:
                print(e)
                print(traceback.print_exc)
                assert False, "[ERROR] _load_render_data failed."
                
        self.all_uv = torch.from_numpy(all_uv)
        self.all_mask = torch.from_numpy(all_mask)
        self.depths = torch.from_numpy(depths)

        # construct pairs
        if mode == 'train':
            view_pair_data_path = os.path.join(self.texture_cache_dir, 'view_pair_data.npy')
            if not os.path.exists(view_pair_data_path):
                print("generate view-pair data...")
                self.view_pair = self.get_view_pair(cam2worlds)
            else:
                print("load view-pair data...")
                _view_pair_data = np.load(view_pair_data_path, allow_pickle=True).item()
                self.view_pair = _view_pair_data['view_pair']
        print('Load data: End')
        # current_points
        self.rays_count_list = []
        self.valid_xyxy = np.zeros((self.n_images, 4), dtype=np.int32)
        self.init_valid_bbox(use_valid_bbox=True)
        self.patch_size = patch_size
        
    
    def _save_render_data(self, all_uv, all_mask, depths):
        assert all_uv.shape[0] == all_mask.shape[0] and all_uv.shape[0] == depths.shape[0], f"[ERROR] data not match: all_uv.shape: {all_uv.shape}, all_mask: {all_mask.shape}, depths: {depths}"
        if not os.path.exists(self.texture_cache_dir):
            os.makedirs(self.texture_cache_dir)
        _count = 0
        for idx, (_uv, _mask, _depth) in enumerate(zip(all_uv, all_mask, depths)):
            _cache_path = os.path.join(self.texture_cache_dir, '{:08d}.npy'.format(idx))
            np.save(_cache_path, {'uv': _uv, 'mask': _mask, 'depth': _depth})
            _count += 1
        print(f"[INFO] save render data done. npy-files: {_count}")
        return True

    def _load_render_data(self, idx):
        cache_path = os.path.join(self.texture_cache_dir, '{:08d}.npy'.format(idx))
        data = np.load(cache_path, allow_pickle=True).item()
        return data['uv'], data['mask'], data['depth']


    def generate_texture_data(self, cam2worlds):
        depths = render_dataset(self, self.object_path)
        colors = (self.images_np * 255).astype(np.uint8) # [n_images, H, W, 3]
        init_texture = self.mesh_texture.get_texture_map(colors, depths, cam2worlds, is_save=False)
        depths = np.stack(depths, axis=0)
        return init_texture, depths

    def generate_all_view(self, cam2worlds):
        all_uv = []
        all_mask = []
        for i in tqdm(range(len(cam2worlds))):
            world2cam = np.linalg.inv(cam2worlds[i]).astype('float32')
            uv, _depth, mask = self.mesh_texture.get_view_data(world2cam, is_save=False)
            all_uv.append(uv)
            all_mask.append(mask)
        all_uv = np.stack(all_uv, axis=0)
        all_mask = np.stack(all_mask, axis=0)
        return all_uv, all_mask

    def _get_bbox_from_image(self, img_idx, pad_ratio=0.1):
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

        valid_ids = np.array([
            _i * self.W + _j
            for _i in range(valid_i_min, valid_i_max + 1) 
            for _j in range(valid_j_min, valid_j_max + 1)
        ])
        
        self.rays_count_list += [
            (valid_j_max - valid_j_min + 1) * (valid_i_max - valid_i_min + 1)
        ]
        return valid_j_min, valid_i_min, valid_j_max, valid_i_max #(x0, y0, x1, y1)

    def init_valid_bbox(self, use_valid_bbox=True):
        for img_idx in range(self.n_images):
            if use_valid_bbox:
                start_x, start_y, end_x, end_y = self._get_bbox_from_image(img_idx)
            else:
                start_x, start_y, end_x, end_y = 0, 0, self.W, self.H
            self.valid_xyxy[img_idx] = np.array([start_x, start_y, end_x, end_y], dtype=np.int32)
        return True

    def get_view_pair(self, cam2worlds, view_interval=15.0):
        view_pairs = []
        min_len = 1000
        for i in range(cam2worlds.shape[0]):
            a = []
            for j in range(cam2worlds.shape[0]):
                angle = np.dot(cam2worlds[i,:,2], cam2worlds[j,:,2])
                if angle > np.cos(view_interval / 180.0 * np.pi):
                    a.append(j)
            if len(a) < min_len:
                min_len = len(a)
            view_pairs.append(a)
        # post process
        for i in range(len(view_pairs)):
            if type(view_pairs[i]) == type([]):
                p = view_pairs[i].copy()
            else:
                p = view_pairs[i].tolist()
            p.append(i)
            view_pairs[i] = np.array(p,dtype='int32')
        return view_pairs

    def reprojection_from_BtoA(self, idx, b_idx=None):
        # query data if exists _pairs_color
        if hasattr(self, '_pairs_color'):
            rindex = np.random.choice(len(self.view_pair[idx]))
            color_btoa = self._pairs_color[idx][rindex]
            return color_btoa

        # prepare target reprojection data
        depth_a = self.depths[idx].to(self.device)
        color_a = self.images[idx].to(self.device)
        
        rindex = np.random.choice(self.view_pair[idx]) if b_idx is None else b_idx
        if rindex != idx:
            color_b, uv_b, depth_b, mask_b, cam2world_b\
                = self.images[rindex].to(self.device), self.all_uv[rindex].to(self.device), self.depths[rindex].to(self.device), self.all_mask[rindex].to(self.device), self.cam2worlds[rindex].to(self.device)
            
            cam2world_a = self.cam2worlds[idx].to(self.device)
            world2cam_b = torch.linalg.inv(cam2world_b)
            a2b = torch.transpose(torch.matmul(world2cam_b, cam2world_a), 0, 1).float()

            y = torch.linspace(0, self.H - 1, self.H)
            x = torch.linspace(0, self.W - 1, self.W)
            yy, xx = torch.meshgrid(y,x)

            intrinsic = self.intrinsics_all[0].to(self.device)
            intrinsic_inv = torch.linalg.inv(intrinsic)

            coords = torch.stack([xx, yy, torch.ones_like(xx), torch.ones_like(xx)], dim=2)
            coords = torch.matmul(coords, torch.transpose(intrinsic_inv, 0, 1))
            coords[:,:,:2] *= depth_a[:,:,None]
            coords[:,:,1] *= -1
            coords[:,:,2] = -depth_a
            coords = torch.matmul(coords, a2b)
            coords[:,:,1] *= -1
            coords[:,:,2] *= -1
            z_tar = coords[:,:,2].clone()
            coords[:,:,:-1] /= coords[:,:,2:3] + 1e-8
            coords_ = torch.matmul(coords, torch.transpose(intrinsic, 0, 1))
            x = coords_[:,:,0]
            y = coords_[:,:,1]

            mask0 = (depth_a == 0)
            mask1 = (x < 0) + (y < 0) + (x >= self.W-1) + (y >= self.H-1)
            lx = torch.floor(x) / self.W * 2.0 - 1.0 
            ly = torch.floor(y) / self.H * 2.0 - 1.0 
            rx = (lx + 1) / self.W * 2.0 - 1.0  
            ry = (ly + 1) / self.H * 2.0 - 1.0 

            depth_b = depth_b.view(1,1,depth_b.shape[0],depth_b.shape[1])
            sample_z1 = torch.abs(z_tar\
                - F.grid_sample(depth_b, torch.stack([lx, ly], dim=-1).unsqueeze(0), align_corners=True))
            sample_z2 = torch.abs(z_tar\
                - F.grid_sample(depth_b, torch.stack([lx, ry], dim=-1).unsqueeze(0), align_corners=True))
            sample_z3 = torch.abs(z_tar\
                - F.grid_sample(depth_b, torch.stack([rx, ly], dim=-1).unsqueeze(0), align_corners=True))
            sample_z4 = torch.abs(z_tar\
                - F.grid_sample(depth_b, torch.stack([rx, ry], dim=-1).unsqueeze(0), align_corners=True))
            mask2 = torch.minimum(torch.minimum(sample_z1, sample_z2),\
                torch.minimum(sample_z3, sample_z4)) > 5.0 # if visible surface

            mask_remap = (1 - (mask0 + mask1 + mask2 > 0).int()).float() 

            map_xy = torch.stack([x / self.W * 2.0 - 1.0, y / self.H * 2.0 - 1.0], dim=-1).unsqueeze(0)

            color_b = color_b.permute(2,0,1).unsqueeze(0)
            color_btoa = F.grid_sample(color_b, map_xy, align_corners=True)
            color_btoa = color_btoa.permute(0,2,3,1).squeeze()
            mask_b = mask_b.view(1,1,mask_b.shape[0],mask_b.shape[1]).float()
            mask = (F.grid_sample(mask_b, map_xy, align_corners=True) > 0.99)\
                * mask_remap
            mask = mask.permute(0,2,3,1).squeeze()
            mask = torch.repeat_interleave(mask[:,:,None], 3, dim=-1).bool() 
            color_btoa[~mask] = 255/255.

        else:
            color_btoa = color_a.clone()
        return color_btoa

    def _sample_data_from_bbox(self, idx):        
        start_x, start_y, end_x, end_y = self.valid_xyxy[idx]
        color = self.images[idx][start_y:end_y, start_x:end_x].to(self.device)    # H x W x 3
        mask = self.all_mask[idx][start_y:end_y, start_x:end_x].to(self.device)     # H x W x 3
        depth = self.depths[idx][start_y:end_y, start_x:end_x].to(self.device)      # H x W 
        uv = self.all_uv[idx][start_y:end_y, start_x:end_x].to(self.device)      # H x W 

        if 'train' == self.mode:
            color_btoa = self.reprojection_from_BtoA(idx)
            color_btoa = color_btoa[start_y:end_y, start_x:end_x] 
        else:
            color_btoa = color
        
        return color, mask, depth, uv, color_btoa

    def _transform_uv(self, uv):
        uv[:,:,1] = 1 - uv[:,:,1]
        uv = uv * 2 - 1.0
        return uv

    def __len__(self):
        return self.n_images


    def __getitem__(self, idx):

        color, mask, depth, uv, color_btoa = self._sample_data_from_bbox(idx)
        uv = self._transform_uv(uv)    

        _crop_data = torch.cat([color, mask.unsqueeze(-1), depth.unsqueeze(-1), uv, color_btoa], dim=-1)
        _crop_data = _crop_data.permute(2, 0, 1)
        if 'train' == self.mode:  # bs,H,W,C
            hh, ww = color.shape[0], color.shape[1]
            crop_s = min(
                min(hh, ww),
                self.patch_size
            )
            crop_data = T.RandomCrop((crop_s, crop_s))(_crop_data)
            batch_data = torch.zeros(crop_data.shape[0], self.patch_size, self.patch_size)
            batch_data[:, :crop_s, :crop_s] = crop_data
        elif 'val' == self.mode:
            hh, ww = color.shape[0], color.shape[1]
            crop_s = min(hh, ww)
            crop_data = T.CenterCrop((crop_s, crop_s))(_crop_data)
            batch_data = crop_data

        color, mask, depth, uv, color_btoa = batch_data[:3], batch_data[3:4], batch_data[4:5], batch_data[5:7], batch_data[7:10]
        color = color * 2.0 - 1.0
        color_btoa = color_btoa * 2.0 - 1.0
        mask = (mask > 0.5).float()
        
        data = {
            'color': color,
            'mask': mask,
            'uv': uv,
            'depth': depth,
            'color_btoa': color_btoa,
        }
        return data

