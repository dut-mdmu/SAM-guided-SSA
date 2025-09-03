from torch.nn.parallel import DistributedDataParallel as DDP
import torch.nn as nn
import torch
import logging
import datetime
import os
from sam2.build_sam import build_sam2_video_predictor
join = os.path.join

def build_model(args):
    model_cfg = args.model_cfg
    sam2_ckpt = args.sam2_ckpt
    sam2 = build_sam2_video_predictor(model_cfg, sam2_ckpt, device=args.device)
    for name, param in sam2.named_parameters():  
        if 'image_encoder.' in name:
            param.requires_grad = False
    if args.multi_gpu:
        sam2 = DDP(sam2, device_ids=[args.rank], output_device=args.rank)
    return sam2

# 自定义过滤器  
class IgnoreWarningsFilter(logging.Filter):  
    def filter(self, record):  
        # 假设我们不希望记录包含"pixdim"的警告  
        return not ("pixdim" in record.msg)

def get_logger(args):
    logger = logging.getLogger(__name__)
    logger.setLevel(logging.DEBUG)
    LOG_OUT_DIR = join(args.work_dir, args.task_name)
    os.makedirs(LOG_OUT_DIR, exist_ok=True)
    cur_time = datetime.datetime.now().strftime('%Y-%m-%d-%H-%M-%S')
    log_file_path = os.path.join(LOG_OUT_DIR, f'output_{cur_time}.log')

    file_handler = logging.FileHandler(log_file_path)  
    file_handler.setLevel(logging.INFO)  
    file_handler.addFilter(IgnoreWarningsFilter())   
    file_handler.setFormatter(logging.Formatter('[%(asctime)s] - %(message)s'))  
    logger.addHandler(file_handler)  
    return logger



class FocalLoss(nn.Module):
    def __init__(self, gamma=2.0, alpha=0.25):
        super(FocalLoss, self).__init__()
        self.gamma = gamma
        self.alpha = alpha

    def forward(self, pred, mask):
        """
        pred: [B, 1, H, W]
        mask: [B, 1, H, W]
        """
        assert pred.shape == mask.shape, "pred and mask should have the same shape."
        pred = torch.sigmoid(pred)
        num_pos = torch.sum(mask)
        num_neg = mask.numel() - num_pos
        w_pos = (1 - pred) ** self.gamma
        w_neg = pred ** self.gamma

        loss_pos = -self.alpha * mask * w_pos * torch.log(pred + 1e-12)
        loss_neg = -(1 - self.alpha) * (1 - mask) * w_neg * torch.log(1 - pred + 1e-12)

        loss = (torch.sum(loss_pos) + torch.sum(loss_neg)) / (num_pos + num_neg + 1e-12)
        # if torch.isnan(loss).any():  
        #     raise AssertionError("Loss is NaN. Check your loss calculation.") 
        return loss


class DiceLoss(nn.Module):
    def __init__(self, smooth=1.0):
        super(DiceLoss, self).__init__()
        self.smooth = smooth

    def forward(self, pred, mask):
        """
        pred: [B, 1, H, W]
        mask: [B, 1, H, W]
        """
        assert pred.shape == mask.shape, "pred and mask should have the same shape."
        pred = torch.sigmoid(pred)
        intersection = torch.sum(pred * mask)
        union = torch.sum(pred) + torch.sum(mask)
        dice_loss = (2.0 * intersection + self.smooth) / (union + self.smooth)
        # if torch.isnan(dice_loss).any():  
        #     raise AssertionError("Loss is NaN. Check your loss calculation.") 
        return 1 - dice_loss


class MaskMSE(nn.Module):
    def __init__(self, ):
        super(MaskMSE, self).__init__()

    def forward(self, pred, mask, pred_iou):
        """
        pred: [B, 1, H, W]
        mask: [B, 1, H, W]
        pred_iou: [B, 1]
        """
        assert pred.shape == mask.shape, "pred and mask should have the same shape."

        pred = torch.sigmoid(pred)
        intersection = torch.sum(pred * mask)
        union = torch.sum(pred) + torch.sum(mask) - intersection
        iou = (intersection + 1e-7) / (union + 1e-7)
        mse = torch.mean((iou - pred_iou) ** 2)
        return mse


class FocalDice_MSELoss(nn.Module):
    def __init__(self, weight=20.0, iou_scale=1.0):
        super(FocalDice_MSELoss, self).__init__()
        self.weight = weight
        self.iou_scale = iou_scale
        self.focal_loss = FocalLoss()
        self.dice_loss = DiceLoss()
        self.maskiou_mse = MaskMSE()
    def forward(self, pred, mask, pred_iou):
        """
        pred: [B, 1, H, W]
        mask: [B, 1, H, W]
        """
        assert pred.shape == mask.shape, "pred and mask should have the same shape."

        focal_loss = self.focal_loss(pred, mask)
        dice_loss =self.dice_loss(pred, mask)
        loss1 = self.weight * focal_loss + dice_loss
        loss2 = self.maskiou_mse(pred, mask, pred_iou)
        loss = loss1 + loss2 * self.iou_scale
        return loss

class Seg_loss(nn.Module):
    def __init__(self, weight=20.0):
        super(Seg_loss, self).__init__()
        self.weight = weight
        self.focal_loss = FocalLoss()
        self.dice_loss = DiceLoss()

    def forward(self, pred, mask):
        """
        pred: [B, 1, H, W]
        mask: [B, 1, H, W]
        """
        assert pred.shape == mask.shape, "pred and mask should have the same shape."

        focal_loss = self.focal_loss(pred, mask)
        dice_loss =self.dice_loss(pred, mask)
        loss = self.weight * focal_loss + dice_loss
        return loss



if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_cfg", type = str, default='sam2_configs/sam_med2.yaml')
    parser.add_argument("--sam_med2_ckpt", type = str, default='checkpoints/sam_med2_x1024_b.pt')
    parser.add_argument('--multi_gpu', action='store_true', default=False)
    parser.add_argument('--device', type=str, default='cuda')
    args = parser.parse_args()

    model = build_model(args)