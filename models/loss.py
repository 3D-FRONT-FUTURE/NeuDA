import torch
from torch import nn
import torch.nn.functional as F
import torchvision
from torch import autograd as autograd
import numpy as np
from math import exp
import pdb


# --------------------------------------------
# Perceptual loss
# --------------------------------------------
class VGGFeatureExtractor(nn.Module):
    def __init__(self, feature_layer=[2,7,16,25,34], use_input_norm=True, use_range_norm=False):
        super(VGGFeatureExtractor, self).__init__()
        '''
        use_input_norm: If True, x: [0, 1] --> (x - mean) / std
        use_range_norm: If True, x: [0, 1] --> x: [-1, 1]
        '''
        model = torchvision.models.vgg19(pretrained=True)
        self.use_input_norm = use_input_norm
        self.use_range_norm = use_range_norm
        if self.use_input_norm:
            mean = torch.Tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
            std = torch.Tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)
            self.register_buffer('mean', mean)
            self.register_buffer('std', std)
        self.list_outputs = isinstance(feature_layer, list)
        if self.list_outputs:
            self.features = nn.Sequential()
            feature_layer = [-1] + feature_layer
            for i in range(len(feature_layer)-1):
                self.features.add_module('child'+str(i), nn.Sequential(*list(model.features.children())[(feature_layer[i]+1):(feature_layer[i+1]+1)]))
        else:
            self.features = nn.Sequential(*list(model.features.children())[:(feature_layer + 1)])
        print(self.features)

        # No need to BP to variable
        for k, v in self.features.named_parameters():
            v.requires_grad = False

    def forward(self, x):
        if self.use_range_norm:
            x = (x + 1.0) / 2.0
        if self.use_input_norm:
            x = (x - self.mean) / self.std
        if self.list_outputs:
            output = []
            for child_model in self.features.children():
                x = child_model(x)
                output.append(x.clone())
            return output
        else:
            return self.features(x)

class VGGPerceptualLoss(torch.nn.Module):
    
    def __init__(self, resize=True):
        super(VGGPerceptualLoss, self).__init__()
        blocks = []
        blocks.append(torchvision.models.vgg16(pretrained=True).features[:4].eval())
        blocks.append(torchvision.models.vgg16(pretrained=True).features[4:9].eval())
        blocks.append(torchvision.models.vgg16(pretrained=True).features[9:16].eval())
        blocks.append(torchvision.models.vgg16(pretrained=True).features[16:23].eval())
        for bl in blocks:
            for p in bl:
                p.requires_grad = False
        self.blocks = torch.nn.ModuleList(blocks)
        self.transform = torch.nn.functional.interpolate
        self.resize = resize
        self.register_buffer("mean", torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1))
        self.register_buffer("std", torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1))

    def forward(self, input, target, feature_layers=[0, 1, 2, 3], style_layers=[]):
        #input = input.permute(0, 3, 1, 2)
        #target = target.permute(0, 3, 1, 2)
        input = (input-self.mean) / self.std
        target = (target-self.mean) / self.std
        if self.resize:
            input = self.transform(input, mode='bilinear', size=(224, 224), align_corners=False)
            target = self.transform(target, mode='bilinear', size=(224, 224), align_corners=False)
        loss = 0.0
        x = input
        y = target
        for i, block in enumerate(self.blocks):
            x = block(x)
            with torch.no_grad():
                y = block(y)
            if i in feature_layers:
                loss += torch.nn.functional.l1_loss(x, y)
            if i in style_layers:
                act_x = x.reshape(x.shape[0], x.shape[1], -1)
                act_y = y.reshape(y.shape[0], y.shape[1], -1)
                gram_x = act_x @ act_x.permute(0, 2, 1)
                gram_y = act_y @ act_y.permute(0, 2, 1)
                loss += torch.nn.functional.l1_loss(gram_x, gram_y) 
        return loss



class PerceptualLoss(nn.Module):
    """VGG Perceptual loss
    """
    def __init__(self, feature_layer=[2,7,16,25,34], weights=[0.1,0.1,1.0,1.0,1.0], lossfn_type='l1', use_input_norm=True, use_range_norm=False):
        super(PerceptualLoss, self).__init__()
        self.vgg = VGGFeatureExtractor(feature_layer=feature_layer, use_input_norm=use_input_norm, use_range_norm=use_range_norm)
        self.lossfn_type = lossfn_type
        self.weights = weights
        if self.lossfn_type == 'l1':
            self.lossfn = nn.L1Loss()
        else:
            self.lossfn = nn.MSELoss()
        print(f'feature_layer: {feature_layer}  with weights: {weights}')

    def forward(self, x, gt):
        """Forward function.
        Args:
            x (Tensor): Input tensor with shape (n, c, h, w).
            gt (Tensor): Ground-truth tensor with shape (n, c, h, w).
        Returns:
            Tensor: Forward results.
        """
        x_vgg, gt_vgg = self.vgg(x), self.vgg(gt.detach())
        loss = 0.0
        if isinstance(x_vgg, list):
            n = len(x_vgg)
            for i in range(n):
                loss += self.weights[i] * self.lossfn(x_vgg[i], gt_vgg[i])
        else:
            loss += self.lossfn(x_vgg, gt_vgg.detach())
        return loss

# --------------------------------------------
# GAN loss: gan, ragan
# --------------------------------------------
class GANLoss(nn.Module):
    def __init__(self, gan_type, real_label_val=1.0, fake_label_val=0.0):
        super(GANLoss, self).__init__()
        self.gan_type = gan_type.lower()
        self.real_label_val = real_label_val
        self.fake_label_val = fake_label_val

        if self.gan_type == 'gan' or self.gan_type == 'ragan':
            self.loss = nn.BCEWithLogitsLoss()
        elif self.gan_type == 'lsgan':
            self.loss = nn.MSELoss()
        elif self.gan_type == 'wgan':
            def wgan_loss(input, target):
                # target is boolean
                return -1 * input.mean() if target else input.mean()

            self.loss = wgan_loss
        elif self.gan_type == 'softplusgan':
            def softplusgan_loss(input, target):
                # target is boolean
                return F.softplus(-input).mean() if target else F.softplus(input).mean()

            self.loss = softplusgan_loss
        else:
            raise NotImplementedError('GAN type [{:s}] is not found'.format(self.gan_type))

    def get_target_label(self, input, target_is_real):
        if self.gan_type in ['wgan', 'softplusgan']:
            return target_is_real
        if target_is_real:
            return torch.empty_like(input).fill_(self.real_label_val)
        else:
            return torch.empty_like(input).fill_(self.fake_label_val)

    def forward(self, input, target_is_real):
        target_label = self.get_target_label(input, target_is_real)
        loss = self.loss(input, target_label)
        return loss
