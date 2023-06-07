import argparse
import os
import logging
from unittest import case
from pyhocon import ConfigFactory
import torch
import numpy as np
from tqdm import tqdm
from skimage import morphology as morph
from evaluation import get_mesh
from dataset.rays_dataset import Dataset
from models.fields import SDFNetwork
from evaluation import mesh_filtering, dtu_eval
import pdb

FORMAT = "[%(filename)s:%(lineno)s - %(funcName)20s() ] %(message)s"
logging.basicConfig(level=logging.DEBUG, format=FORMAT)

def mkdir_ifnotexists(directory):
    if not os.path.exists(directory):
        os.mkdir(directory)

def extract_mesh(args):
    torch.set_default_dtype(torch.float32)

    # Configuration
    f = open(args.conf)
    conf_text = f.read()
    conf_text = conf_text.replace('CASE_NAME', args.case)
    f.close()

    conf = ConfigFactory.parse_string(conf_text)
    base_exp_dir = conf['general.base_exp_dir']

    end_iter = conf.get_int('train.end_iter')
    # dataset
    if 'DTU' in conf['dataset.data_dir']:
        _dataset = 'DTU'
    else:
        assert False, '[ERROR] Not support now.'

    evals_folder_name = "evals"
    evaldir = os.path.join(base_exp_dir, evals_folder_name)
    mkdir_ifnotexists(evaldir)

    # ------------------
    device = torch.device('cuda')
    sdf_network = SDFNetwork(**conf['model.sdf_network']).to(device)
    if torch.cuda.is_available():
        sdf_network.cuda()

    dataset_conf = conf.get_config('dataset')

    # Load checkpoint
    if len(args.checkpoint) == 0:
        latest_model_name = None
        model_list_raw = os.listdir(os.path.join(base_exp_dir, 'checkpoints'))
        model_list = []
        for model_name in model_list_raw:
            if model_name[-3:] == 'pth' and int(model_name[5:-4]) <= end_iter:
                model_list.append(model_name)
        model_list.sort()
        latest_model_name = model_list[-1]
    else:
        latest_model_name = args.checkpoint

    if latest_model_name is not None:
        logging.info('Find checkpoint: {}'.format(latest_model_name))
        checkpoint = torch.load(os.path.join(base_exp_dir, 'checkpoints', latest_model_name), map_location=device)
        sdf_network.load_state_dict(checkpoint['sdf_network_fine'])
    
    # load dataset
    eval_dataset = Dataset(dataset_conf)
    scale_mat = eval_dataset.scale_mats_np[0] 
    num_images = eval_dataset.n_images 
    K = eval_dataset.intrinsics_all
    pose = eval_dataset.pose_all
    masks = (eval_dataset.masks > (127.5/256))[...,0].bool() # N x H x W
    img_res = [masks.shape[2], masks.shape[1]]

    print("dilation...")
    dilated_masks = list()

    for m in tqdm(masks, desc="Mask dilation"):
        if args.no_masks:
            dilated_masks.append(torch.ones_like(m, device="cuda"))
        else:
            struct_elem = morph.disk(args.dilation_radius)
            dilated_masks.append(torch.from_numpy(morph.binary_dilation(m.numpy(), struct_elem)))
    masks = torch.stack(dilated_masks).cuda()

    sdf_network.eval()

    with torch.no_grad():
        size = img_res
        pose = pose.cuda()
        cams = [
            K[:, :3, :3].cuda(),
            pose[:, :3, :3].transpose(2, 1),
            - pose[:, :3, :3].transpose(2, 1) @ pose[:, :3, 3:],
            torch.tensor([size for i in range(num_images)]).cuda().float()
        ]

        mesh = get_mesh.get_surface_high_res_mesh(
            sdf=lambda x: -sdf_network.sdf(x)[:, 0], refine_bb=not args.no_refine_bb,
            resolution=args.resolution, cams=cams, masks=masks, bbox_size=args.bbox_size
        )

        mesh_filtering.mesh_filter(args, mesh, masks, cams)  # inplace filtering

    if args.one_cc: # Taking the biggest connected component
        components = mesh.split(only_watertight=False)
        areas = np.array([c.area for c in components], dtype=float)
        mesh = components[areas.argmax()]

    # Transform to world coordinates
    mesh.apply_transform(scale_mat)
    mesh.export(f'{evaldir}/output_mesh{args.suffix}.ply', 'ply')

    # evaluation metrics
    if args.eval_metric:
        eval_mesh_path = f'{evaldir}/output_mesh{args.suffix}.ply'
        scene = args.case
        if _dataset == 'DTU':
            scene = scene.split('_scan')[-1]
            dtu_eval.eval(eval_mesh_path, int(scene), "data/dtu_eval", evaldir, args.suffix)
        else:
            print("not GT data, skip evaluation")


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--conf', type=str)
    parser.add_argument('--timestamp', default='', type=str, help='The experiment timestamp to test.')
    parser.add_argument('--checkpoint', default='',type=str,help='The trained model checkpoint to test')
    parser.add_argument('--case', type=str, default=None, help='If set, taken to be the scan id.')
    parser.add_argument('--resolution', default=512, type=int, help='Grid resolution for marching cube')
    parser.add_argument('--no_refine_bb', action="store_true", help='Skip bounding box refinement')
    parser.add_argument("--bbox_size", default=1., type=float, help="Size of the bounding volume to querry")
    parser.add_argument("--one_cc", action="store_true", default=True,
                        help="Keep only the biggest connected component or all")
    parser.add_argument("--no_one_cc", action="store_false", dest="one_cc")
    parser.add_argument("--filter_visible_triangles", action="store_true",
                        help="Whether to remove triangles that have no projection in images (uses mesh rasterization)")
    parser.add_argument('--min_nb_visible', type=int, default=2, help="Minimum number of images used for visual hull"
                                                                      "filtering and triangle visibility filtering")
    parser.add_argument("--no_masks", action="store_true", help="Ignore the visual hull masks")
    parser.add_argument("--dilation_radius", type=int, default=12)
    parser.add_argument("--suffix", default="")
    parser.add_argument("--eval_metric", action="store_true", help="whether evaluate the reconstructed mesh, only support DTU dataset now.")
    args = parser.parse_args()

    extract_mesh(args)
