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
from dataloader import get_dataset
from train_utils import DiceLoss, build_model, get_logger
import nibabel as nib  
import cv2
from torch.nn import CrossEntropyLoss

import warnings
warnings.filterwarnings("ignore") 
warnings.filterwarnings("ignore", category=UserWarning)


parser = argparse.ArgumentParser()
parser.add_argument('--work_dir', type=str, default='work_dir')
parser.add_argument('--task_name', type=str, default='surgical_seg')
#load data
parser.add_argument("--data_root", type = str, default='datasets/train')
parser.add_argument('--image_size', type=int, default=1024)
parser.add_argument('--slice_length', type=int, default=8)
parser.add_argument('--mode', type = str, default='train')
parser.add_argument('--batch_size', type=int, default=1)
parser.add_argument('--num_objs', type=int, default=2)
parser.add_argument('--num_workers', type=int, default=0)
#load model
parser.add_argument("--model_type", type = str, default='sam2')
parser.add_argument("--model_cfg", type = str, default='sam2_hiera_t.yaml')
parser.add_argument("--sam2_ckpt", type = str, default='checkpoints/sam2_hiera_tiny.pt')
# train
parser.add_argument('--pretrain_path', type=str, default=None)
parser.add_argument('--resume', action='store_true', default=False)
parser.add_argument('--device', type=str, default='cuda')
parser.add_argument('--seed', default=0, type=int)
parser.add_argument('--num_epochs', type=int, default=100)
parser.add_argument('--gpu_ids', type=int, nargs='+', default=[0])
parser.add_argument('--multi_gpu', action='store_true', default=False)
parser.add_argument('--lr_scheduler', type=str, default='multisteplr')
parser.add_argument('--step_size', type=list, default=[40, 70]) 
parser.add_argument('--gamma', type=float, default=0.5)
parser.add_argument('--lr', type=float, default=1e-4)
parser.add_argument('--weight_decay', type=float, default=1e-5)
parser.add_argument('--port', type=int, default=11364)
parser.add_argument('--dist', dest='dist', type=bool, default=False, help='distributed training or not')


args = parser.parse_args()
os.environ["CUDA_VISIBLE_DEVICES"] = ','.join([str(i) for i in args.gpu_ids])

device = args.device
MODEL_SAVE_PATH = join(args.work_dir, args.task_name)
os.makedirs(MODEL_SAVE_PATH, exist_ok=True)


class BaseTrainer:
    def __init__(self, model, dataloaders, args):
        self.model = model
        self.dataloaders = dataloaders
        self.args = args
        self.best_loss = np.inf
        self.best_dice = 0.0
        self.best_iou = 0.0
        self.step_best_dice = 0.0
        self.losses = []
        self.dices = []
        self.ious = []
        self.set_loss_fn()
        self.set_optimizer()
        self.set_lr_scheduler()

        self.model = self.model.module if self.args.multi_gpu else self.model

        if args.pretrain_path is not None:
            self.load_checkpoint(args.pretrain_path, args.resume)
        else:
            self.start_epoch = 0

    def set_loss_fn(self):
        self.seg_loss = DiceLoss()
        self.ce_loss = CrossEntropyLoss()

    def set_optimizer(self):
        self.optimizer = torch.optim.AdamW(self.model.parameters(), lr=self.args.lr, weight_decay=self.args.weight_decay) #

    def set_lr_scheduler(self):
        if self.args.lr_scheduler == "multisteplr":
            self.lr_scheduler = torch.optim.lr_scheduler.MultiStepLR(self.optimizer,
                                                                self.args.step_size,
                                                                self.args.gamma)
        elif self.args.lr_scheduler == "steplr":
            self.lr_scheduler = torch.optim.lr_scheduler.StepLR(self.optimizer,
                                                                self.args.step_size[0],
                                                                self.args.gamma)
        elif self.args.lr_scheduler == 'coswarm':
            # TODO: add args for coswarm
            self.lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(self.optimizer)
        elif self.args.lr_scheduler == 'linearlr':
            self.lr_scheduler = torch.optim.lr_scheduler.LinearLR(self.optimizer, 
                                              end_factor=0.0, 
                                              total_iters=self.args.num_epochs * len(self.dataloaders))
        else:
            self.lr_scheduler = None

    def load_checkpoint(self, ckp_path, resume):
        last_ckpt = None
        if os.path.exists(ckp_path):
            if self.args.multi_gpu:
                dist.barrier()
            if 'step_' in ckp_path:
                self.start_step = int(ckp_path.split('step_')[-1].split('.pth')[0])
            else:
                self.start_step = 0

            last_ckpt = torch.load(ckp_path, map_location=self.args.device)

        if last_ckpt:
            try:
                self.model.load_state_dict(last_ckpt['model_state_dict'])
            except Exception as e:
                print(f"Failed to load model state dict: {e}")
                self.model.load_state_dict(last_ckpt['model_state_dict'], False)

            if resume:
                self.start_epoch = last_ckpt['epoch']
                self.optimizer.load_state_dict(last_ckpt['optimizer_state_dict'])
                self.lr_scheduler.load_state_dict(last_ckpt['lr_scheduler_state_dict'])
                self.losses = last_ckpt['losses']
                self.dices = last_ckpt['dices']
                self.ious = last_ckpt['ious']
                self.best_loss = last_ckpt['best_loss']
                self.best_dice = last_ckpt['best_dice']
            else:
                self.start_epoch = 0
            print(f"Loaded checkpoint from {ckp_path} (epoch {self.start_epoch}, step: {self.start_step})")
            
        else:
            self.start_epoch, self.start_step = 0, 0
            print(f"No checkpoint found at {ckp_path}, start training from scratch")
    
    def save_checkpoint(self, epoch, state_dict, describe="last"):
        torch.save({
            "epoch": epoch + 1,
            "model_state_dict": state_dict,
            "optimizer_state_dict": self.optimizer.state_dict(),
            "lr_scheduler_state_dict": self.lr_scheduler.state_dict(),
            "losses": self.losses,
            "ious": self.ious,
            "dices": self.dices,
            "best_loss": self.best_loss,
            "best_iou": self.best_iou,
            "best_dice": self.best_dice,
            "args": self.args,
        }, join(MODEL_SAVE_PATH, f"sam_model_{describe}.pth"))


    def get_iou_and_dice(self, pred, label):
        assert pred.shape == label.shape
        pred = (torch.sigmoid(pred) > 0.5)
        label = (label > 0)
        intersection = torch.logical_and(pred, label).sum(dim=(1,2,3)) 
        union = torch.logical_or(pred, label).sum(dim=(1,2,3))  
        iou = intersection.float() / (union.float() + 1e-8) 
        dice = (2 * intersection.float()) / (pred.sum(dim=(1,2,3)) + label.sum(dim=(1,2,3)) + 1e-8) 
        return iou.mean().item(), dice.mean().item()

    def plot_result(self, plot_data, description, save_name):
        plt.plot(plot_data)
        plt.title(description)
        plt.xlabel('Epoch')
        plt.ylabel(f'{save_name}')
        plt.savefig(join(MODEL_SAVE_PATH, f'{save_name}.png'))
        plt.close()


    def train_epoch(self, epoch):
        self.model.train()
        l = len(self.dataloaders)
        tbar = tqdm(self.dataloaders, desc=f'Epoch {epoch+1} / {self.args.num_epochs}')
        epoch_loss, epoch_iou, epoch_dice = 0, 0, 0
        for step, batch_input in enumerate(tbar): 
            batch_loss, batch_iou, batch_dice = [], [], []
            interval_prompts = batch_input["pre_interval_prompt"]
            interval_images, interval_labels = batch_input["pre_interval_image"], batch_input["pre_interval_label"]
            
            for interval in range(interval_images.shape[0]):
                images = interval_images[interval].to(device)
                labels = interval_labels[interval]
                prompts = interval_prompts[interval]

                train_state = self.model.train_init_state(images)

                obj_list = list(labels.keys())
                obj_segments = {}
                prompt_labels = []
                output_dict = {obj_id: [] for obj_id in obj_list} 

                for obj_id, obj_data in prompts.items():
                    obj_label = labels[obj_id].to(device).type(torch.long)  
        
                    for slice_idx, text_embed in obj_data['text_prompt'].items():
                        _, _, out_mask_logits = self.model.train_add_new_text(
                            inference_state=train_state, frame_idx=slice_idx, obj_id=obj_id,
                            points=None, labels=None, clear_old_points=False, text= text_embed
                            )
       
                        prompt_labels.append(obj_label[slice_idx]) 
                      
                prompt_label = torch.stack(prompt_labels, dim=0)  
                prompt_loss = self.seg_loss(out_mask_logits, prompt_label.unsqueeze(1))  
                prompt_iou, prompt_dice = self.get_iou_and_dice(out_mask_logits, prompt_label.unsqueeze(1))  
                
         
                if prompt_dice < 0.45:  #0.6
                    self.optimizer.zero_grad()  
                    self.scaler.scale(prompt_loss).backward()  
                    self.scaler.step(self.optimizer)  
                    self.scaler.update()  
                    self.model.reset_state(train_state) 

                    batch_loss.append(prompt_loss.item())  
                    batch_iou.append(prompt_iou)  
                    batch_dice.append(prompt_dice)  
                    continue

                start_slice = min(slice_idx for slice_idx in prompts[next(iter(prompts))]["text_prompt"].keys())
                for direction in [False, True]:  # 正向和反向  
                    for out_frame_idx, out_obj_ids, out_mask_logits in self.model.train_propagate_in_video(  
                        train_state, start_frame_idx=start_slice, reverse=direction):  
                        obj_segments[out_frame_idx] = {out_obj_id: out_mask_logits[i] for i, out_obj_id in enumerate(out_obj_ids)}  

                for out_frame_idx in range(images.shape[0]):  
                    for out_obj_id, out_mask in obj_segments[out_frame_idx].items():  
                        output_dict[out_obj_id].append(out_mask)  


                outputs_ = [torch.cat(masks, dim=0) for masks in output_dict.values()]  
                labels_ = [labels[obj_id].to(device).type(torch.long) for obj_id in output_dict.keys()]  
                mask = torch.stack(outputs_, dim=0)  
                label = torch.stack(labels_, dim=0)  
                total_loss = self.seg_loss(mask, label) + prompt_loss  

                self.optimizer.zero_grad()  
                self.scaler.scale(total_loss).backward()  
                self.scaler.step(self.optimizer)  
                self.scaler.update()  
        
                self.model.reset_state(train_state)  
                iou, dice = self.get_iou_and_dice(mask, label) 
                batch_loss.append(total_loss.item())  
                batch_iou.append(iou)  
                batch_dice.append(dice)  

                print(f'Metrics, IoU: {iou:.4f}, Dice: {dice:.4f}')


            if not self.args.multi_gpu or (self.args.multi_gpu and self.args.rank == 0):
                if (step+1) % 50 == 0:
                    self.args.logger.info(f'Epoch: {epoch+1}, Step: {step+1}, loss: {np.mean(batch_loss):.4f}, iou: {np.mean(batch_iou):.4f}, dice: {np.mean(batch_dice):.4f}')
                    state_dict = self.model.state_dict()
                    #save latest checkpoint
                    self.save_checkpoint(epoch, state_dict, describe='setp')

            epoch_loss += np.mean(batch_loss)
            epoch_iou += np.mean(batch_iou)
            epoch_dice += np.mean(batch_dice)
        
        avg_loss, avg_iou, avg_dice = epoch_loss / l, epoch_iou / l, epoch_dice / l
        return avg_loss, avg_iou, avg_dice

    def train(self):
        self.scaler = amp.GradScaler()
        for epoch in range(self.start_epoch, self.args.num_epochs):
            torch.cuda.empty_cache()
            if not self.args.multi_gpu or (self.args.multi_gpu and self.args.rank == 0):
                print(f'Epoch: {epoch+1}/{self.args.num_epochs}')

            if self.args.multi_gpu:
                # dist.barrier()
                self.dataloaders.sampler.set_epoch(epoch)

            avg_loss, avg_iou, avg_dice = self.train_epoch(epoch)
            
            if self.lr_scheduler is not None:
                self.lr_scheduler.step()

            if not self.args.multi_gpu or (self.args.multi_gpu and self.args.rank == 0):
                self.losses.append(avg_loss)
                self.ious.append(avg_iou)
                self.dices.append(avg_dice)

                self.args.logger.info(f'Epoch\t {epoch+1}\t LR\t {self.lr_scheduler.get_last_lr()}\t: loss: {avg_loss:.5f}, iou: {avg_iou:.5f}, dice: {avg_dice:.5f}')

                state_dict = self.model.state_dict()
                 #save latest checkpoint
                self.save_checkpoint(epoch, state_dict, describe='latest')

                #save train loss best checkpoint
                if avg_loss < self.best_loss: 
                    self.best_loss = avg_loss
                    # self.save_checkpoint(epoch, state_dict, describe='loss_best')
                
                if avg_iou > self.best_iou: 
                    self.best_iou = avg_iou
                    # self.save_checkpoint(epoch, state_dict, describe='iou_best')

                # save train dice best checkpoint
                if avg_dice > self.best_dice: 
                    self.best_dice = avg_dice
                    self.save_checkpoint(epoch, state_dict, describe='dice_best')

                self.plot_result(self.losses, 'Loss', 'Loss')
                self.plot_result(self.dices, 'Dice', 'Dice')
                self.plot_result(self.ious, 'IoU', 'IoU')
      
        if not self.args.multi_gpu or (self.args.multi_gpu and self.args.rank == 0):
            self.args.logger.info('=====================================================================')
            for key, value in vars(self.args).items():
                self.args.logger.info(key + ': ' + str(value))
            self.args.logger.info(f'Best loss: {self.best_loss}, Best iou: {self.best_iou}, Best dice: {self.best_dice}')
            self.args.logger.info('=====================================================================')


########################################## Trainer ##########################################
def init_seeds(seed=0, cuda_deterministic=True):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    # Speed-reproducibility tradeoff https://pytorch.org/docs/stable/notes/randomness.html
    if cuda_deterministic:  # slower, more reproducible
        cudnn.deterministic = True
        cudnn.benchmark = False
    else:  # faster, less reproducible
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
        logger = get_logger(args)
        args.logger = logger
        random.seed(42)
        np.random.seed(42)
        torch.manual_seed(42)
        # Load datasets
        dataloaders = get_dataset(args)
        # Build model
        model = build_model(args)
        # Create trainer
        trainer = BaseTrainer(model, dataloaders, args)
        # Train
        trainer.train()

def main_worker(rank, args):
    setup(rank, args.world_size)
    torch.cuda.set_device(rank)
    args.device = torch.device(f"cuda:{rank}")
    args.rank = rank
    args.gpu_info = {"gpu_count":args.world_size, 'gpu_name':rank}
    init_seeds(2024 + rank)
    if rank == 0:
        logger = get_logger(args)
        args.logger = logger
    dataloaders = get_dataset(args)
    model = build_model(args)
    trainer = BaseTrainer(model, dataloaders, args)
    trainer.train()
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
