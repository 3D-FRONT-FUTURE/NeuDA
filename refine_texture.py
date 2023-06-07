import os
import logging
import argparse
import numpy as np
import cv2 as cv
import trimesh
import torch
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter
from torch.optim.lr_scheduler import MultiStepLR
from shutil import copyfile
from tqdm import tqdm
from pyhocon import ConfigFactory
from dataset.patch_dataset import TextureDataset, DataProvider
from models.texture.UVTexture import UVTexture
from models.fields import Discriminator_UNet, init_weights
from models.loss import GANLoss, VGGPerceptualLoss
import pdb


class Runner:
    def __init__(self, conf_path, mode='train', case='CASE_NAME', is_continue=False, expname='', exproot=''):
        self.device = torch.device('cuda')

        # Configuration
        self.conf_path = conf_path
        f = open(self.conf_path)
        conf_text = f.read()
        conf_text = conf_text.replace('CASE_NAME', case)
        f.close()

        self.conf = ConfigFactory.parse_string(conf_text)
        
        if len(exproot) == 0:
            exproot = self.conf['general.base_exp_dir']


        if len(expname) != 0:
            self.base_exp_dir = exproot + '_' + expname
        else:
            self.base_exp_dir = exproot
        os.makedirs(self.base_exp_dir, exist_ok=True)
        
        # Training parameters
        self.end_iter = self.conf.get_int('train.end_iter')
        self.save_freq = self.conf.get_int('train.save_freq')
        self.report_freq = self.conf.get_int('train.report_freq')
        self.val_freq = self.conf.get_int('train.val_freq')
        self.batch_size = self.conf.get_int('train.batch_size')
        num_workers = self.conf.get_int('train.num_workers')
        self.lr_G = self.conf.get_float('train.lr_G')
        self.lr_D = self.conf.get_float('train.lr_D')
        self.learning_rate_alpha = self.conf.get_float('train.learning_rate_alpha')
        learning_decay = self.conf.get_list('train.learning_decay')
        decay_gamma = self.conf.get_float('train.decay_gamma')
        self.warmup_epoch = self.conf.get_int('train.warmup_epoch', default=0)

        # # Weights
        self.rgb_weight = self.conf.get_float('train.rgb_weight')
        self.adv_weight = self.conf.get_float('train.adv_weight')
        self.is_continue = is_continue
        self.mode = mode
        self.model_list = []
        self.writer = None

        # dataset & dataloader
        self.dataset = TextureDataset(self.conf['dataset'], mode='train')
        self.loader_train = DataProvider(
            dataset=self.dataset,
            batch_size = self.batch_size
        )
        self.val_dataset = TextureDataset(self.conf['dataset'], mode='val')
        self.loader_val = torch.utils.data.DataLoader(
            dataset=self.val_dataset,
            batch_size = 1
        )
        self.iter_step = 0

        # UVTexture
        #texture_path = self.conf.get_string('model.texture')
        _object_path = self.conf.get_string('dataset.object_path')
        texture_path = os.path.join(os.path.dirname(_object_path), f'../cache/{case}_init_tex.png')
        self.uvtex_generator = UVTexture(texture_path, tex_dim=1024).to(self.device)
        self.model_list.append(self.uvtex_generator)
        # Networks
        self.netD = Discriminator_UNet(input_nc=6).to(self.device)
        init_weights(self.netD, init_type='orthogonal', init_bn_type='uniform', gain=0.2)
        self.model_list.append(self.netD)

        # optimizer
        params_to_train_d = list(self.netD.parameters())
        params_to_train_g = list(self.uvtex_generator.parameters())
        self.optimizer_d = torch.optim.Adam(params_to_train_d, lr=self.lr_D, betas=(0.5, 0.999))
        self.optimizer_g = torch.optim.Adam(params_to_train_g, lr=self.lr_G, betas=(0.5, 0.999))

        # scheduler
        self.scheduler_d = MultiStepLR(self.optimizer_d, milestones=learning_decay, 
                                gamma=decay_gamma)
        self.scheduler_g = MultiStepLR(self.optimizer_g, milestones=learning_decay, 
                                gamma=decay_gamma)
        # Loss
        self.l1_loss = torch.nn.L1Loss()
        self.adv_loss = GANLoss('gan') 
        # self.perceptual_loss = VGGPerceptualLoss()

        # Load checkpoint
        latest_model_name = None
        if is_continue:
            model_list_raw = os.listdir(os.path.join(self.base_exp_dir, 'checkpoints_texture'))
            model_list = []
            for model_name in model_list_raw:
                if model_name[-3:] == 'pth' and int(model_name[5:-4]) <= self.end_iter:
                    model_list.append(model_name)
            model_list.sort()
            latest_model_name = model_list[-1]

        if latest_model_name is not None:
            logging.info('Find checkpoint: {}'.format(latest_model_name))
            self.load_checkpoint(latest_model_name)

        # Backup codes and configs for debug
        if self.mode[:5] == 'train':
            self.file_backup()

    def decode_batch(self, data):
        uv = data['uv'].to(self.device)
        color = data['color'].to(self.device)#.cpu().numpy()
        color_btoa = data['color_btoa'].to(self.device)#.cpu().numpy()
        mask = data['mask'].to(self.device)#.cpu().numpy()
        depth = data['depth'].to(self.device)#.cpu().numpy()
        return uv, color, color_btoa, mask, depth

    def train_epoch(self, loader_train, use_gan=False):
        for iter_i, data in enumerate(loader_train):
            uv, color, color_btoa, mask, depth = self.decode_batch(data)

            uv = uv.permute(0, 2, 3, 1)
            view = self.uvtex_generator(uv)
            # view = torch.cat([F.grid_sample(self.texture, uv[_].unsqueeze(0)) for _ in range(uv.shape[0])], dim=0)
            #mask_ = torch.repeat_interleave(mask, repeats=3, dim=1).bool()
            
            mask_ = torch.repeat_interleave(mask, repeats=3, dim=1).bool()
            color[~mask_] = 0.
            color_btoa[~mask_] = 0.
            view[~mask_] = 0.
            
            ### Optimize Generator
            for p in self.netD.parameters():
                p.requires_grad = False
            rgb_loss = self.l1_loss(view[mask_], color_btoa[mask_])
            # perceptual_loss = self.perceptual_loss(rgb_light, rgb)
            # lpips_loss = self.loss_fn_vgg(2.0*(rgb_light) - 1.0, 2.0*(rgb) - 1.0)

            x_fake = torch.cat([color, view - color], dim=1)
            g_loss = self.adv_loss(self.netD(x_fake), True)

            # USE_GAN = torch.sum(valid_mask.float()) >= 1.5*torch.sum(1.0 - valid_mask.float()) and self._hparams.use_gan

            if not use_gan:
                loss_g = rgb_loss * self.rgb_weight
            else:
                loss_g = rgb_loss * self.rgb_weight + self.adv_weight * g_loss
            
            self.optimizer_g.zero_grad(set_to_none=True)
            loss_g.backward()
            self.optimizer_g.step()

            for p in self.netD.parameters():
                p.requires_grad = True
            
            ### Optimize Discriminator
            x_real = torch.cat([color, color_btoa - color], dim=1)

            real_loss = self.adv_loss(self.netD(x_real), True)
            fake_loss = self.adv_loss(self.netD(x_fake.detach()), False)
            loss_d = 0.5 * (real_loss + fake_loss)
            # don't update discriminator if generator is obviously weaker than discriminator
            use_gan = False if self.adv_weight > loss_d * 2.0 else use_gan

            if use_gan:
                self.optimizer_d.zero_grad(set_to_none=True)
                loss_d.backward()
                self.optimizer_d.step()
            else:
                self.optimizer_d.zero_grad(set_to_none=True)


            self.iter_step += 1

            self.writer.add_scalar('Loss/loss_g', loss_g, self.iter_step)
            self.writer.add_scalar('Loss/loss_d', loss_d, self.iter_step)
            self.writer.add_scalar('Loss/rgb_loss', rgb_loss, self.iter_step)
            self.writer.add_scalar('Loss/g_loss', g_loss, self.iter_step)

            if self.iter_step % self.report_freq == 0:
                print(self.base_exp_dir)
                print('iter:{:8>d} rgb_loss = {} loss_g = {} loss_d = {} lr_g={} lr_d={}'.format(self.iter_step, rgb_loss, loss_g, loss_d, self.optimizer_g.param_groups[0]['lr'], self.optimizer_d.param_groups[0]['lr']))
            


    def train(self):
        self.writer = SummaryWriter(log_dir=os.path.join(self.base_exp_dir, 'logs'))
        self.update_learning_rate()
        res_epochs = self.end_iter - self.iter_step

        for epoch_i in tqdm(range(res_epochs)):
            if epoch_i % self.val_freq == 0:
                _loader_val = iter(self.loader_val)
                self.validate_epoch(_loader_val, epoch_i)
                self.save_texture(epoch_i)

            _loader_train = iter(self.loader_train)
            self.train_epoch(self.loader_train, use_gan=(epoch_i >= self.warmup_epoch))

            if epoch_i % self.save_freq == 0:
                self.save_checkpoint()

            if epoch_i >= self.warmup_epoch:
                self.update_learning_rate()

        # save final result
        _loader_val = iter(self.loader_val)
        self.validate_epoch(_loader_val, epoch_i)
        self.save_texture('_final')
        self.save_checkpoint()

    def get_view_comparison(self, color, color_btoa, view, mask, suffix_name):
        texture_cmp_dir = os.path.join(self.base_exp_dir, 'texture_validate')
        if not os.path.exists(texture_cmp_dir):
            os.makedirs(texture_cmp_dir)
        mask_ = torch.repeat_interleave(mask, repeats=3, dim=1).bool()
        color[~mask_] = 1.
        color_btoa[~mask_] = 1.
        view[~mask_] = 1.
        color_np = ((color[0] + 1.) * 0.5 * 255).permute(1,2,0).detach().cpu().numpy()
        color_btoa_np = ((color_btoa[0] + 1.) * 0.5 * 255).permute(1,2,0).detach().cpu().numpy()
        view = ((view[0] + 1.) * 0.5 * 255).permute(1,2,0).detach().cpu().numpy()
        comparison = np.concatenate([color_np.astype(np.uint8), color_btoa_np.astype(np.uint8), view], axis=1)
        cv.imwrite(f'{texture_cmp_dir}/view_cmp_{suffix_name}.jpg', comparison)

    def save_texture(self, idx):
        out_path = os.path.join(self.base_exp_dir, 'texture_validate', f'texture_epoch_{idx}.png')
        self.uvtex_generator.save_texture(out_path)
            
    def validate_epoch(self, loader_val, epoch_i):
        for iter_i, data in enumerate(loader_val):
            uv, color, color_btoa, mask, depth = self.decode_batch(data)
            uv = uv.permute(0, 2, 3, 1)
            with torch.no_grad():
                view = self.uvtex_generator(uv)
                mask_ = torch.repeat_interleave(mask, repeats=3, dim=1).bool()
                rgb_loss = self.l1_loss(view[mask_], color[mask_])
                print('Validate - iter:{:8>d} loss_rgb = {}'.format(self.iter_step, rgb_loss))
                suffix_name = 'ep{:04d}_id{:04d}'.format(epoch_i, iter_i)
                self.get_view_comparison(color, color_btoa, view, mask, suffix_name)
                

    def update_learning_rate(self):
        self.scheduler_d.step()
        self.scheduler_g.step()


    def file_backup(self):
        dir_lis = self.conf['general.recording']
        os.makedirs(os.path.join(self.base_exp_dir, 'recording'), exist_ok=True)
        for dir_name in dir_lis:
            cur_dir = os.path.join(self.base_exp_dir, 'recording', dir_name)
            os.makedirs(cur_dir, exist_ok=True)
            files = os.listdir(dir_name)
            for f_name in files:
                if f_name[-3:] == '.py':
                    copyfile(os.path.join(dir_name, f_name), os.path.join(cur_dir, f_name))

        copyfile(self.conf_path, os.path.join(self.base_exp_dir, 'recording', 'config.conf'))

    def load_checkpoint(self, checkpoint_name):
        checkpoint = torch.load(os.path.join(self.base_exp_dir, 'checkpoints_texture', checkpoint_name), map_location=self.device)
        self.uvtex_generator.load_state_dict(checkpoint['texture_generator'])
        self.netD.load_state_dict(checkpoint['discriminator'])
        #self.optimizer_d.load_state_dict(checkpoint['optimizer_d'])
        #self.optimizer_g.load_state_dict(checkpoint['optimizer_g'])
        #self.iter_step = checkpoint['iter_step']

        logging.info('End')

    def save_checkpoint(self):
        checkpoint = {
            'texture_generator': self.uvtex_generator.state_dict(),
            'discriminator': self.netD.state_dict(),
            'optimizer_d': self.optimizer_d.state_dict(),
            'optimizer_g': self.optimizer_g.state_dict(),
            'iter_step': self.iter_step,
        }

        os.makedirs(os.path.join(self.base_exp_dir, 'checkpoints_texture'), exist_ok=True)
        torch.save(checkpoint, os.path.join(self.base_exp_dir, 'checkpoints_texture', 'ckpt_{:0>6d}.pth'.format(self.iter_step)))




if __name__ == '__main__':
    print('Hello Wooden')

    torch.set_default_tensor_type('torch.cuda.FloatTensor')

    FORMAT = "[%(filename)s:%(lineno)s - %(funcName)20s() ] %(message)s"
    logging.basicConfig(level=logging.DEBUG, format=FORMAT)

    parser = argparse.ArgumentParser()
    parser.add_argument('--conf', type=str, default='./confs/texture_opt.conf')
    parser.add_argument('--mode', type=str, default='train')
    parser.add_argument('--is_continue', default=False, action="store_true")
    parser.add_argument('--gpu', type=int, default=0)
    parser.add_argument('--case', type=str, default='')
    parser.add_argument('--expname', type=str, default='')
    parser.add_argument('--exproot', type=str, default='')
    parser.add_argument('--img', type=int, default=-1)

    args = parser.parse_args()

    # training
    torch.cuda.set_device(args.gpu)
    runner = Runner(args.conf, args.mode, args.case, args.is_continue, args.expname, args.exproot)

    if args.mode == 'train':
        runner.train()