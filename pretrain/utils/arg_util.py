# Copyright (c) ByteDance, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import json
import os
import sys

from tap import Tap

import dist


class Args(Tap):
    # environment
    #exp_name: str = 'Pretrain DAN IAM on page level'
    #exp_dir: str = 'IAM_line_40perc_random_full_mask_L2'   # will be created if not exists
    seed: int = 1
    exp_name: str = 'Pretrain DAN on RIMES dataset L2 seed 1 dropout=0 resnet mean/std'
    #exp_dir: str = f'READ_pretrain_L2_224_25perc_full_patches_set_encoder_A{seed}'   # will be created if not exists
    exp_dir: str = f'RIMES_200epochs_lr_pretrain_L2_224_40perc_random_full_mask_L2_seed_{seed}'
    #data_path: str = '/home/michel/dev/python/formatted/IAM_page/train'  # path to training data
    #data_path: str = '/home/michel/dev/python/formatted/IAM_224/train'
    #data_path = "/home/michel/dev/python/formatted/RD_BAUTZEN_224/classes"
    #data_path: str = '/home/michel/dev/python/formatted/IAM_non_syn_line/train'
    data_path: str = '/home/michel/dev/python/formatted/RIMES_224/train'
    samples = None, # 500 # None # for debug; set samples to an integer number to use only part of the dataset for training
    init_weight: str = ''   # use some checkpoint as model weight initialization; ONLY load model weights
    resume_from: str = f'/home/michel/dev/python/SparK/{exp_dir}/DAN_encoder_withdecoder_1kpretrained_spark_style.pth'   # resume the experiment from some checkpoint.pth; load model weights, optimizer states, and last epoch
    # resume_from: str = '/home/michel/dev/python/SparK/Pre_DAN_IAM_line_64h_16x16_40perc_garbagel1/DAN_encoder_1kpretrained_timm_style.pth'     
    # SparK hyperparameters
    #resume_from: str = f'/home/michel/dev/python/SparK/{exp_dir}/DAN_encoder_withdecoder_1kpretrained_spark_style_27.pth'
    mask: float = 0.40   # mask ratio, should be in (0, 1)
    mask_type = 'random' # {'random', 'block', 'line', 'grid', 'diagonal' }
    mask_area = 'full' # full, patches, area: full is full page, patches is only patches that contain text, area is the area with patches containing text

    
    # encoder hyperparameters
    #model: str = 'DAN_encoder'
    model: str  = 'DAN_encoder'
    #input_size = (128, 1248)#(864, 616)#
    #input_size = (1760, 1216)
    input_size = (224,224)
    sbn: bool = False # sync batch norm; if the model is convnext or resnet, sbn would be set to False automatically
    
    # data hyperparameters
    #bs: int = 40 # for line
    bs: int = 64 # for page
    dataloader_workers: int = 8
    
    # pre-training hyperparameters
    dp: float = 0.0
    base_lr: float = 2e-4
    #wd: float = 0.04
    wd: float = 0.02
    wde: float = 0.02
    ep: int = 200
    wp_ep: int = 20
    clip: int = 1.
    opt: str = 'lamb'
    #opt: str = 'AdamW'
    ada: float = 0.
    
    # NO NEED TO SPECIFIED; each of these args would be updated in runtime automatically
    lr: float = None
    batch_size_per_gpu: int = 0
    glb_batch_size: int = 0
    densify_norm: str = ''
    device: str = 'cuda'
    local_rank: int = 0
    cmd: str = ' '.join(sys.argv[1:])
    commit_id: str = os.popen(f'git rev-parse HEAD').read().strip() or '[unknown]'
    commit_msg: str = (os.popen(f'git log -1').read().strip().splitlines() or ['[unknown]'])[-1].strip()
    last_loss: float = 0.
    cur_ep: str = ''
    remain_time: str = ''
    finish_time: str = ''
    first_logging: bool = True
    log_txt_name: str = '{args.exp_dir}/pretrain_log.txt'
    tb_lg_dir: str = ''     # tensorboard log directory
    
    @property
    def is_convnext(self):
        return 'convnext' in self.model or 'cnx' in self.model
    
    @property
    def is_resnet(self):
        return 'resnet' in self.model
    
    def log_epoch(self):
        if not dist.is_local_master():
            return
        
        if self.first_logging:
            self.first_logging = False
            with open(self.log_txt_name, 'w') as fp:
                json.dump({
                    'name': self.exp_name, 'cmd': self.cmd, 'git_commit_id': self.commit_id, 'git_commit_msg': self.commit_msg,
                    'model': self.model,
                }, fp)
                fp.write('\n\n')
        
        with open(self.log_txt_name, 'a') as fp:
            json.dump({
                'cur_ep': self.cur_ep,
                'last_L': self.last_loss,
                'rema': self.remain_time, 'fini': self.finish_time,
            }, fp)
            fp.write('\n')


def init_dist_and_get_args():
    from utils import misc
    
    # initialize
    args = Args(explicit_bool=True).parse_args()
    e = os.path.abspath(args.exp_dir)
    d, e = os.path.dirname(e), os.path.basename(e)
    e = ''.join(ch if (ch.isalnum() or ch == '-') else '_' for ch in e)
    args.exp_dir = os.path.join(d, e)
    
    os.makedirs(args.exp_dir, exist_ok=True)
    args.log_txt_name = os.path.join(args.exp_dir, 'pretrain_log.txt')
    args.tb_lg_dir = args.tb_lg_dir or os.path.join(args.exp_dir, 'tensorboard_log')
    try:
        os.makedirs(args.tb_lg_dir, exist_ok=True)
    except:
        pass
    
    misc.init_distributed_environ(exp_dir=args.exp_dir)
    
    # update args
    if not dist.initialized():
        args.sbn = False
    args.first_logging = True
    args.device = dist.get_device()
    args.batch_size_per_gpu = args.bs // dist.get_world_size()
    args.glb_batch_size = args.batch_size_per_gpu * dist.get_world_size()
    
    if args.is_resnet:
        args.ada = args.ada or 0.95
        args.densify_norm = 'bn'
    
    if args.is_convnext:
        args.ada = args.ada or 0.999
        args.densify_norm = 'ln'
    
    args.opt = args.opt.lower()
    args.lr = args.base_lr * args.glb_batch_size / 256
    args.wde = args.wde or args.wd
    
    return args
