
import os
from . import loader
from . import rasterizer
from . import painter
import collections
import numpy as np
import skimage.io as sio
import cv2
from ctypes import *

TexMesh = collections.namedtuple("TexMesh", "V,F,VT,FT,VN,FN")

libpath = os.path.dirname(os.path.abspath(__file__))
Render = cdll.LoadLibrary(libpath + '/CudaRender/libRender.so')

class MeshTexture(object):
    
    def __init__(
        self, obj_path, 
        name,
        intrinsic, 
        tex_dim=1024, img_hw=(480, 640)
    ) -> None:
        # init mesh
        V, F, VT, FT, VN, FN = loader.LoadOBJ(obj_path)
        self.tex_mesh = TexMesh(V=V, F=F, VT=VT, FT=FT, VN=VN, FN=FN)
        self.context = self.set_mesh(V, F)
        #init pose
        self.tex_dim = tex_dim
        self.img_hw = img_hw
        self.intrinsic = intrinsic
        self.name = name
        
        # initialize render
        Render.InitializeCamera(
            img_hw[1], img_hw[0], 
            c_float(intrinsic[0,0]), c_float(intrinsic[1,1]), c_float(intrinsic[0,2]), c_float(intrinsic[1,2])
        )

    def get_texture_map(self, colors, depths, cam2worlds, is_save=True):
        print('RasterizeTexture...')
        vweights, findices = rasterizer.RasterizeTexture(self.tex_mesh.VT, self.tex_mesh.FT, self.tex_dim, self.tex_dim)
        print('generate textiles...')
        points, normals, coords = rasterizer.GeneratePoints(self.tex_mesh.V, self.tex_mesh.F, self.tex_mesh.VN, self.tex_mesh.FN, vweights, findices)
        print('paint colors...')
        point_colors = np.zeros((points.shape[0], 4), dtype='float32')

        for i in range(colors.shape[0]):
            painter.ProjectPaint(points, normals, point_colors, colors[i], depths[i], np.linalg.inv(cam2worlds[i]).astype('float32'), self.intrinsic)

        print('prepare final texture')
        original_texture = np.zeros((self.tex_dim, self.tex_dim, 3), dtype='uint8')
        painter.PaintToTexturemap(original_texture, point_colors, coords)

        if is_save:
            if not os.path.exists('./tmp'):
                os.makedirs('./tmp')
            sio.imsave(f'./tmp/{self.name}_texture.png', original_texture)
            np.savetxt(f'./tmp/{self.name}_intrinsic.txt', self.intrinsic)

        return original_texture


    def get_view_data(self, world2cam, is_save=True):
        # render
        self.render(self.context, world2cam)
        depth = self.get_depth()
        vindices, vweights, findices = self.get_vmap(self.context)
        uv = np.zeros((findices.shape[0], findices.shape[1], 3), dtype='float32')
        for j in range(2):
            uv[:,:,j] = 0
            for k in range(3):
                vind = self.tex_mesh.FT[findices][:,:,k]
                uv_ind = self.tex_mesh.VT[vind][:,:,j]
                uv[:,:,j] += vweights[:,:,k] * uv_ind
        mask = (findices != -1)
        for j in range(3):
            uv[:,:,j] *= mask

        depth *= mask
        mask = (mask*255).astype('uint8')
        if is_save:
            sio.imsave(f'./tmp/{self.name}_mask.png', mask)
            np.savez_compressed(f'./tmp/{self.name}_depth.npz', depth)
            np.savez_compressed(f'./tmp/{self.name}_uv.npz', uv[:,:,:2])
            np.savetxt(f'./tmp/{self.name}_pose.txt', world2cam)
        return uv[:,:,:2], depth, mask 

    def set_mesh(self, V, F):
        handle = Render.SetMesh(c_void_p(V.ctypes.data), c_void_p(F.ctypes.data), V.shape[0], F.shape[0])
        return handle

    def render(self, handle, world2cam):
        Render.SetTransform(handle, c_void_p(world2cam.ctypes.data))
        Render.Render(handle)

    def get_depth(self, ):
        depth = np.zeros(self.img_hw, dtype='float32')
        Render.GetDepth(c_void_p(depth.ctypes.data))

        return depth

    def get_vmap(self, handle):
        imsize_3c = list(self.img_hw) + [3]
        vindices = np.zeros(imsize_3c, dtype='int32')
        vweights = np.zeros(imsize_3c, dtype='float32')
        findices = np.zeros(self.img_hw, dtype='int32')

        Render.GetVMap(handle, c_void_p(vindices.ctypes.data), c_void_p(vweights.ctypes.data), c_void_p(findices.ctypes.data))

        return vindices, vweights, findices

    def __project(self, info, world2cam, point):
        points = np.transpose(np.dot(world2cam[0:3,0:3], np.transpose(point)))
        points[:,0] += world2cam[0,3]
        points[:,1] += world2cam[1,3]
        points[:,2] += world2cam[2,3]
        x = points[:,0] / points[:,2] * info[0,0] + info[0,2]
        y = points[:,1] / points[:,2] * info[1,1] + info[1,2]
        return np.array([x,y,points[:,2]])


if __name__ == '__main__':
    obj_path = './test_case/chair00.obj'
    name = 'chair00'
    intrinsic = None
    tex_dim = 1024
    img_hw = (1200, 1200)
    mesh_texture = MeshTexture(obj_path, name, intrinsic, tex_dim, img_hw)
    # generate texture map
    colors = None
    depths = None
    cam2worlds = None
    mesh_texture.get_texture_map(colors, depths, cam2worlds, is_save=True)
    # generate view_data i
    for i in len(cam2worlds):
        world2cam = np.linalg.inv(cam2worlds[i]).astype('float32')
        uv, depth, mask = mesh_texture.get_view_data(world2cam, is_save=True)
