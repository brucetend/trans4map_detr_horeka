import os
import sys
import json
import yaml
import h5py
import torch
import numpy as np
from tqdm import tqdm
import argparse
from torch.utils.data import DataLoader
import cv2


BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = BASE_DIR
sys.path.append(os.path.join(ROOT_DIR, '../'))

# from utils.habitat_utils import HabitatUtils
import torchvision.transforms as transforms
from Backbone.segformer import Segformer

from lib2.config import config, update_config, infer_exp_id
from lib2 import dataset



if __name__ == '__main__':

    output_dir = '/home/zteng/Trans4Map/precompute_training_inputs/data/training'
    os.makedirs(output_dir, exist_ok=True)

    device = torch.device('cuda')

    model = Segformer()
    model = model.to(device)

    print('Loading pre-trained weights: ')
    # model_path = './runs/gru_fullrez_lastlayer_m256/20434/smnet_mp3d_best_model.pkl'
    #model_path = '/cvhci/data/VisLoc/smnet_mp3d_best_model.pkl'
    model_path = '/cvhci/data/VisLoc/weights_ssmap/smnet_mp3d_best_model_b2_4points.pkl'
    # model_path = '/home/zteng/Trans4Map/checkpoints/gru_fullrez_lastlayer_m256/25070/ckpt_model.pkl'
    model_path = '/home/zteng/Trans4Map/checkpoints/mit_b2.pth'


    state = torch.load(model_path)
    print('state:', state.keys())

    # model_state = state['model_state']
    #
    # ########################################################################################################################
    weights={}
    # for k, v in model_state.items():
    #     if k.startswith('module.encoder'):
    #         print("key_:", k)
    #         k = '.'.join(k.split('.')[2:])
    #         print("key_key_key:", k)
    #         weights[k] = v

    for k, v in state.items():
        print('key_:', k)
        weights[k] = v


    model.load_state_dict(weights, strict=False)
    model.eval()

    ########################################################################################################################

    normalize = transforms.Compose([transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                    std=[0.229, 0.224, 0.225])])

    depth_normalize = transforms.Normalize(mean=[0.213], std=[0.285])

    ########################################################################################################################
    # Parse args & config
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    parser.add_argument('--cfg', required=True)
    # parser.add_argument('--pth')
    # arser.add_argument('--out')
    # parser.add_argument('--vis_dir')

    # parser.add_argument('--y', action='store_true')
    # parser.add_argument('--test_hw', type=int, nargs='*')

    parser.add_argument('opts',
                        help='Modify config options using the command-line',
                        default=None, nargs=argparse.REMAINDER)
    args = parser.parse_args()
    update_config(config, args)


    device = 'cuda' if config.cuda else 'cpu'
    if config.cuda and config.cuda_benchmark:
        torch.backends.cudnn.benchmark = False

    #####################################################################################################################

    # -- -- Load json
    # paths = json.load(open('data/paths.json', 'r'))

    # envs_splits = json.load(open('data/envs_splits.json', 'r'))
    # test_envs = envs_splits['test_envs']
    # test_envs = [x for x in test_envs if x in paths]
    # test_envs.sort()

    # input_dir = ""
    # test_envs = os.listdir(input_dir).sort()

    print("************** start... ****************")
    # if os.path.isfile(os.path.join(output_dir, env+'.h5')): continue
    # -- instantiate Habitat

    # house, level = env.split('_')
    # scene = 'data/mp3d/{}/{}.glb'.format(house, house)
    # habitat = HabitatUtils(scene, int(level))
    #path = paths[env]

    #N = len(path['positions'])

    ########################################### inital Dataset #########################################################
    # Init dataset
    DatasetClass = getattr(dataset, config.dataset.name)
    config.dataset.valid_kwargs.update(config.dataset.common_kwargs)
    print("config.dataset.valid_kwargs:", config.dataset.valid_kwargs)


    valid_dataset = DatasetClass(**config.dataset.valid_kwargs)

    valid_loader = DataLoader(valid_dataset, 1,
                              num_workers=config.num_workers,
                              pin_memory=config.cuda)

    # features_lastlayer = np.zeros((N,64,120,160), dtype=np.float32)


    with torch.no_grad():
        for batch in tqdm(valid_loader, position=1, total = len(valid_loader)):
            color = batch['x'].to(device)
            # sem = batch['sem'].to(device)

            print('color_size:', color.size())
            rgb = color[:,:3,:,:].long()
            print('rgb_size:', rgb.size())
            #rgb = color.astype(np.float32)
            rgb = rgb / 255.0
            # rgb = torch.FloatTensor(rgb).permute(2,0,1)  # ? 通道在前面么？
            rgb = normalize(rgb)
            rgb = rgb.unsqueeze(0).to(device)

            # depth_enc = habitat.render(mode='depth')
            # depth_enc = depth_enc[:,:,0]
            # depth_enc = depth_enc.astype(np.float32)
            # depth_enc = torch.FloatTensor(depth_enc).unsqueeze(0)
            # depth_enc = depth_normalize(depth_enc)
            # depth_enc = depth_enc.unsqueeze(0).to(device)

            semfeat_lastlayer = model(rgb)


            semfeat_lastlayer = semfeat_lastlayer[0].cpu().numpy()

            semfeat_lastlayer = semfeat_lastlayer.astype(np.float32)
            print("semfeat_lastlayer:", semfeat_lastlayer.shape)


    ########################################################################################################################
        filename = os.path.join(output_dir, +"haha"+ '.h5')
        with h5py.File(filename, 'w') as f:
            f.create_dataset('features_lastlayer', data=semfeat_lastlayer, dtype=np.float32)
    # del habitat