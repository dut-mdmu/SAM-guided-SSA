import argparse
import os
import random

import cv2
import numpy as np
from PIL import Image
import torch
from monai import transforms
from dataloader import Resize, Normalization
from train_utils import build_model
import torch.nn.functional as F


parser = argparse.ArgumentParser()
parser.add_argument('--work_dir', type=str, default='work_dir')
parser.add_argument('--task_name', type=str, default='surgical_test1')
#load data
parser.add_argument('--data_root', type = str,
                    default='/media/zyj/data/zyj/2025/医学评价模型/项目整合/Surgical-Skill-Assessment-via-Video-Semantic-Aggregation/data/jigsaws/Suturing/Suturing_160x120')
parser.add_argument('--result_root', type = str,
                    default='/media/zyj/data/zyj/2025/医学评价模型/项目整合/Surgical-Skill-Assessment-via-Video-Semantic-Aggregation/data/jigsaws/Suturing/Suturing_160x120-seg')

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

transform_2d = transforms.Compose(
                [
                    Resize(keys=["image"], target_size=(args.image_size, args.image_size), num_class=2),  #
                    transforms.ToTensord(keys=["image"]),
                    Normalization(keys=["image"]),
                ])
def process_frame_with_prompts(image_farme_path):
    start_slice = random.randint(0, args.slice_length - 1)
    select_obj = [0, 1]
    all_image = torch.zeros(len(image_farme_path), 3, args.image_size, args.image_size)
    all_image_name = []
    all_prompt = {obj: {'text_prompt': {}} for obj in select_obj}
    for slice_index, image_path in enumerate(image_farme_path):
        image2d = Image.open(image_path).convert('RGB')
        all_image_name.append(image_path.split('/')[-1].split('\\')[-1])
        slice_item = transform_2d({"image": image2d})
        all_image[slice_index] = slice_item["image"]

        for idx, obj in enumerate(select_obj):
            # if slice_index == start_slice:
            all_prompt[obj]['text_prompt'][slice_index] = surgery_feat[obj]

    return {'image': all_image, 'prompt': all_prompt, 'image_name': all_image_name, 'ori_size': image2d.size}


def collate_without_label(batch):
    # assert len(batch) == 1, 'Please set batch size to 1 when testing mode'
    pre_interval_image = batch["image"].unsqueeze(0)
    return {
        "pre_interval_image": pre_interval_image,
        "pre_interval_prompt":{0: batch["prompt"]},
        "pre_interval_name":{0: batch["image_name"]},
        "ori_size":batch["ori_size"],
    }


def run(interval_images, interval_prompts, pre_interval_name, ori_size):
    # for interval in range(interval_images.shape[0]):
    interval = 0
    images = interval_images[interval].to(args.device)
    prompts = interval_prompts[interval]
    train_state = model.train_init_state(images)

    obj_list = list(prompts.keys())
    obj_segments = {}
    output_dict = {obj_id: [] for obj_id in obj_list}
    with torch.no_grad():
        for obj_id, obj_data in prompts.items():
            for slice_idx, text_embed in obj_data['text_prompt'].items():
                _, _, out_mask_logits = model.train_add_new_text(
                    inference_state=train_state, frame_idx=slice_idx, obj_id=obj_id,
                    points=None, labels=None, clear_old_points=False, text=text_embed)
        # print([slice_idx for slice_idx in prompts[next(iter(prompts))]["text_prompt"].keys()])
        start_slice = min(slice_idx for slice_idx in prompts[next(iter(prompts))]["text_prompt"].keys())
        # start_slice = 0

        for direction in [False, True]:  # 正向和反向
            for out_frame_idx, out_obj_ids, out_mask_logits in model.train_propagate_in_video(
                    train_state, start_frame_idx=start_slice, reverse=direction):
                obj_segments[out_frame_idx] = {out_obj_id: out_mask_logits[i] for i, out_obj_id in
                                               enumerate(out_obj_ids)}

        for out_frame_idx in range(images.shape[0]):
            for out_obj_id, out_mask in obj_segments[out_frame_idx].items():
                output_dict[out_obj_id].append(out_mask)

    outputs_ = [torch.cat(masks, dim=0) for masks in output_dict.values()]
    mask = torch.stack(outputs_, dim=0)

    target_size = ori_size[::-1]
    ori_mask = F.interpolate(mask, size=target_size, mode='bilinear', align_corners=False)
    binary_mask = (torch.sigmoid(ori_mask) > 0.5).cpu().numpy()
    combined_mask = np.zeros(binary_mask.shape[1:], dtype=bool)
    for obj in range(binary_mask.shape[0]):
        combined_mask = np.logical_or(combined_mask, binary_mask[obj])

    return combined_mask


if __name__ == '__main__':
    model = build_model(args)

    last_ckpt = torch.load(args.pretrain_path, map_location=args.device, weights_only=False)

    key = model.load_state_dict(last_ckpt['model_state_dict'])
    print(key)
    model.eval()

    surgery_table = torch.load("datasets/surgery_table.pt")
    surgical_instruments = torch.load("datasets/surgical_instruments.pt")
    surgery_feat = [surgery_table, surgical_instruments]


    os.makedirs(args.result_root, exist_ok=True)

    for file in os.listdir(args.data_root):
        file = file + '/frame'
        os.makedirs(args.result_root + '/' + file, exist_ok=True)

        for img_file in os.listdir(args.data_root + '/' + file):
            print(args.data_root + '/' + file + '/' + img_file)
            output = process_frame_with_prompts([args.data_root + '/' + file + '/' + img_file])
            batch_input = collate_without_label(output)

            interval_prompts = batch_input["pre_interval_prompt"]
            interval_images = batch_input["pre_interval_image"]
            pre_interval_name = batch_input["pre_interval_name"]
            ori_size = batch_input["ori_size"]
            # print(interval_images.shape)

            result = run(interval_images, interval_prompts, pre_interval_name, ori_size)
            # print(len(output))
            original = cv2.imread(args.data_root + '/' + file + '/' + img_file)

            output = np.zeros_like(original)
            frame_mask = result[0]
            output[frame_mask] = original[frame_mask]

            save_name = args.result_root + '/' + file + '/' + img_file
            cv2.imwrite(save_name, output)
