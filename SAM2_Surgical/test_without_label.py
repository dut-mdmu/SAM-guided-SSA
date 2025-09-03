# set up environment
import numpy as np
import random 
import matplotlib.pyplot as plt
import os
join = os.path.join
from tqdm import tqdm
from torch.backends import cudnn
import torch
import torch.nn as nn
import torch.distributed as dist
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import argparse
from torch.cuda import amp
import torch.multiprocessing as mp
from dataloader import get_dataset_without_label
from train_utils import DiceLoss, build_model
import nibabel as nib  
import cv2
from torch.nn import CrossEntropyLoss

import warnings
warnings.filterwarnings("ignore") 
warnings.filterwarnings("ignore", category=UserWarning)


parser = argparse.ArgumentParser()
parser.add_argument('--work_dir', type=str, default='work_dir')
parser.add_argument('--task_name', type=str, default='surgical_test1')
#load data
parser.add_argument("--data_root", type = str, default='datasets/test/Knot_Tying/image/Knot_Tying_B002_capture2')
parser.add_argument('--image_size', type=int, default=1024)
parser.add_argument('--slice_length', type=int, default=15)
parser.add_argument('--mode', type = str, default='test')
parser.add_argument('--batch_size', type=int, default=1)
parser.add_argument('--num_workers', type=int, default=0)
#load model
parser.add_argument("--model_type", type = str, default='sam2')
parser.add_argument("--model_cfg", type = str, default='sam2_hiera_t.yaml')
parser.add_argument("--sam2_ckpt", type = str, default='checkpoints/sam2_hiera_tiny.pt')
# train
parser.add_argument('--pretrain_path', type=str, default='work_dir/surgical_seg/sam_model_dice_best.pth')
parser.add_argument('--device', type=str, default='cuda')
parser.add_argument('--seed', default=0, type=int)
parser.add_argument('--gpu_ids', type=int, nargs='+', default=[0])
parser.add_argument('--multi_gpu', action='store_true', default=False)
parser.add_argument('--lr', type=float, default=1e-4)
parser.add_argument('--weight_decay', type=float, default=1e-5)
parser.add_argument('--port', type=int, default=11364)
parser.add_argument('--dist', dest='dist', type=bool, default=False, help='distributed training or not')


args = parser.parse_args()
os.environ["CUDA_VISIBLE_DEVICES"] = ','.join([str(i) for i in args.gpu_ids])

device = args.device
MODEL_SAVE_PATH = join(args.work_dir, args.task_name, args.data_root.split('/')[-1].split('\\')[-1])
os.makedirs(MODEL_SAVE_PATH, exist_ok=True)


class BaseTester:
    def __init__(self, model, dataloaders, args):
        self.model = model
        self.dataloaders = dataloaders
        self.args = args
        self.model = self.model.module if self.args.multi_gpu else self.model

        if args.pretrain_path is not None:
            self.load_checkpoint(args.pretrain_path)

    def load_checkpoint(self, ckp_path):
        last_ckpt = None
        if os.path.exists(ckp_path):
            if self.args.multi_gpu:
                dist.barrier()
            last_ckpt = torch.load(ckp_path, map_location=self.args.device, weights_only=False)
        if last_ckpt:
            try:
                self.model.load_state_dict(last_ckpt['model_state_dict'])
            except Exception as e:
                print(f"Failed to load model state dict: {e}")
                self.model.load_state_dict(last_ckpt['model_state_dict'], False)


    # #保存为：掩码部分两色
    # def save_mask(self, mask, ori_size, image_name_list):
    #     ori_size = ori_size[::-1]
    #     ori_mask = F.interpolate(mask, size=ori_size, mode='bilinear', align_corners=False)
    #     ori_mask = (torch.sigmoid(ori_mask) > 0.5).cpu().numpy()
    #     color1 = [255, 0, 0]  # 红色
    #     color2 = [0, 255, 0]  # 绿色
    #     for idx, image_name in enumerate(image_name_list):
    #         visualization = np.zeros((ori_size[0], ori_size[1], 3), dtype=np.uint8)
    #         pre_mask = ori_mask[:,idx,...]
    #         visualization[pre_mask[0] == 1] = color1  # 第一种掩码用蓝色
    #         visualization[pre_mask[1] == 1] = color2  # 第二种掩码用绿色
    #         save_name = os.path.join(MODEL_SAVE_PATH, image_name)
    #         cv2.imwrite(save_name, visualization)

    #保存为：掩码部分原图像素，空白部分纯黑色
    def save_mask(self, mask, ori_size, image_name_list):
        target_size = ori_size[::-1]
        ori_mask = F.interpolate(mask, size=target_size, mode='bilinear', align_corners=False)
        binary_mask = (torch.sigmoid(ori_mask) > 0.5).cpu().numpy()
        combined_mask = np.zeros(binary_mask.shape[1:], dtype=bool)
        for obj in range(binary_mask.shape[0]):
            combined_mask = np.logical_or(combined_mask, binary_mask[obj])

        image_dir = self.args.data_root  # 用self传入的路径作为原图目录

        for idx, image_name in enumerate(image_name_list):
            image_path = os.path.join(image_dir, image_name)  # 拼接完整路径
            if not os.path.exists(image_path):
                print(f"[ERROR] Not found: {image_path}")
                continue

            original = cv2.imread(image_path)
            if original is None:
                print(f"[ERROR] Cannot load image: {image_path}")
                continue

            original_resized = cv2.resize(original, (target_size[1], target_size[0]))
            output = np.zeros_like(original_resized)
            frame_mask = combined_mask[idx]
            output[frame_mask] = original_resized[frame_mask]
            save_name = os.path.join(MODEL_SAVE_PATH, image_name)
            cv2.imwrite(save_name, output)



    def test(self):
        self.model.eval()
        l = len(self.dataloaders)
        tbar = tqdm(self.dataloaders)
        for step, batch_input in enumerate(tbar): 
            interval_prompts = batch_input["pre_interval_prompt"]
            interval_images = batch_input["pre_interval_image"]
            pre_interval_name = batch_input["pre_interval_name"]
            ori_size = batch_input["ori_size"]
   
            for interval in range(interval_images.shape[0]):
                images = interval_images[interval].to(device)
                prompts = interval_prompts[interval]
                image_name_list = pre_interval_name[interval]
                train_state = self.model.train_init_state(images)

                obj_list = list(prompts.keys())
                obj_segments = {}
                output_dict = {obj_id: [] for obj_id in obj_list} 

                with torch.no_grad():
                    for obj_id, obj_data in prompts.items():
                        for slice_idx, text_embed in obj_data['text_prompt'].items():
                            _, _, out_mask_logits = self.model.train_add_new_text(
                                inference_state=train_state, frame_idx=slice_idx, obj_id=obj_id,
                                points=None, labels=None, clear_old_points=False, text= text_embed)
                    
                    start_slice = min(slice_idx for slice_idx in prompts[next(iter(prompts))]["text_prompt"].keys())
                    for direction in [False, True]:  # 正向和反向  
                        for out_frame_idx, out_obj_ids, out_mask_logits in self.model.train_propagate_in_video(  
                            train_state, start_frame_idx=start_slice, reverse=direction):  
                            obj_segments[out_frame_idx] = {out_obj_id: out_mask_logits[i] for i, out_obj_id in enumerate(out_obj_ids)}  

                    for out_frame_idx in range(images.shape[0]):  
                        for out_obj_id, out_mask in obj_segments[out_frame_idx].items():  
                            output_dict[out_obj_id].append(out_mask)  

                outputs_ = [torch.cat(masks, dim=0) for masks in output_dict.values()]  
                mask = torch.stack(outputs_, dim=0)  
                self.save_mask(mask, ori_size, image_name_list)
                self.model.reset_state(train_state)  
                

def init_seeds(seed=0, cuda_deterministic=True):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if cuda_deterministic:  
        cudnn.deterministic = True
        cudnn.benchmark = False
    else: 
        cudnn.deterministic = False
        cudnn.benchmark = True

def device_config(args):
    try:
        if not args.multi_gpu:
            # Single GPU
            # args.multi_gpu = False
            if args.device == 'mps':
                args.device = torch.device('mps')
            else:
                args.device = torch.device(f"cuda:{args.gpu_ids[0]}")
        else:
            # args.multi_gpu = True
            args.nodes = 1
            args.ngpus_per_node = len(args.gpu_ids)
            args.world_size = args.nodes * args.ngpus_per_node
    except RuntimeError as e:
        print(e)

def main():
    for key, value in vars(args).items():
        print(key + ': ' + str(value))
    mp.set_sharing_strategy('file_system')
    device_config(args)
    if args.multi_gpu:
        mp.spawn(main_worker, nprocs=args.world_size, args=(args, ))
    else:
        random.seed(42)
        np.random.seed(42)
        torch.manual_seed(42)
        # Load datasets
        dataloaders = get_dataset_without_label(args)
        # Build model
        model = build_model(args)
        # Create trainer
        tester = BaseTester(model, dataloaders, args)
        # Train
        tester.test()

def main_worker(rank, args):
    setup(rank, args.world_size)
    torch.cuda.set_device(rank)
    args.device = torch.device(f"cuda:{rank}")
    args.rank = rank
    args.gpu_info = {"gpu_count":args.world_size, 'gpu_name':rank}
    init_seeds(2025 + rank)
    dataloaders = get_dataset_without_label(args)
    model = build_model(args)
    tester = BaseTester(model, dataloaders, args)
    tester.test()
    cleanup()


def setup(rank, world_size):
    # initialize the process group
    os.environ['MASTER_ADDR'] = '127.0.0.1'
    os.environ['MASTER_PORT'] = f'{args.port}'
    dist.init_process_group(backend='NCCL', init_method='env://', rank=rank, world_size=world_size)

def cleanup():
    dist.destroy_process_group()

if __name__ == '__main__':
    main()
