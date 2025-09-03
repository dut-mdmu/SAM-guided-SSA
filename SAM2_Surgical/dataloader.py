import os
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset
from PIL import Image
import torch
import random
import numpy as np
from torchvision.transforms.functional import resize, to_pil_image
from monai import data, transforms
from torchvision.transforms import InterpolationMode
import cv2
import matplotlib.pyplot as plt
from transformers import CLIPProcessor, CLIPModel
from torch.utils.data.distributed import DistributedSampler

class Normalization(transforms.Transform):
    def __init__(self, keys):
        self.keys = keys
        pixel_mean=(0.485, 0.456, 0.406),
        pixel_std=(0.229, 0.224, 0.225),
        self.pixel_mean = torch.Tensor(pixel_mean).view(-1, 1, 1)
        self.pixel_std = torch.Tensor(pixel_std).view(-1, 1, 1)
    def __call__(self, data):
        d = dict(data)
        for key in self.keys:
            d[key] = d[key] / 255.0
            d[key] = (d[key] - self.pixel_mean) / self.pixel_std
        return d


class Resize(transforms.Transform):
    def __init__(self, keys, target_size, num_class):
        self.keys = keys
        self.target_size = target_size
        self.num_class = num_class
    def __call__(self, data):
        d = dict(data)
        for key in self.keys:
            if 'image' in key:
                image = resize(d[key], self.target_size, interpolation=InterpolationMode.NEAREST)
                d[key] = np.transpose(image, (2,0,1))
            elif key == 'label':
                label = d[key]
                resized_labels = np.zeros((self.num_class, self.target_size[0], self.target_size[1]))
                resized_label = resize(label, self.target_size, interpolation=InterpolationMode.NEAREST)
                resized_label = np.array(resized_label)
                resized_labels[0] = (resized_label == 128).astype(np.uint8)  # 值为 128 的像素设置为 1
                resized_labels[1] = (resized_label == 255).astype(np.uint8)  # 值为 255 的像素设置为 1
                d[key] = resized_labels
            else:
                raise ValueError(f"Unsupported image shape {image.shape} for key {key}. Expected (3, H, W) or (H, W, 3).") 
        return d


class DataLoader(Dataset):
    def __init__(self, args):
        # 初始化函数，读取所有data_path下的图片
        self.num_objs = args.num_objsx 
        self.image_size = args.image_size
        self.mode = args.mode
        self.slice_length = args.slice_length
        train_data, val_data = train_test_split(self.get_case_list(args.data_root), test_size=0.2, random_state=42)
        
        if self.mode == 'train':
            self.data_list = train_data
        else:
            self.data_list = val_data

        self.transform_2d = transforms.Compose(
                [
                    Resize(keys=["image", "label"], target_size=(self.image_size, self.image_size), num_class=2),  #
                    transforms.ToTensord(keys=["image", "label"]),
                    Normalization(keys=["image"]),
                ])

        surgery_table = torch.load("datasets/surgery_table.pt")
        surgical_instruments = torch.load("datasets/surgical_instruments.pt")
        self.surgery_feat = [surgery_table, surgical_instruments]

    def __getitem__(self, index):

        image_farme_path = self.data_list[index]
        output = self.process_frame_with_prompts(image_farme_path)
        if output is None:
            return self.__getitem__(np.random.randint(self.__len__()))
        return output

    def __len__(self):
        # 返回训练集大小
        return len(self.data_list)

    def get_strided_segments(self, case_list, stride=2):
        return case_list[::stride]

    def get_segments(self, case_list, segment_length=8, stride=1, window_size=None):
        if window_size is None:
            window_size = segment_length * stride
        return [
            self.get_strided_segments(case_list[i:i + window_size], stride)
            for i in range(0, len(case_list), window_size)
            if len(self.get_strided_segments(case_list[i:i + window_size], stride)) >= segment_length
        ]

    def get_case_list(self, data_path):
        case_paths = os.listdir(data_path)
        train_list = []
        for case_ in case_paths:
            case_path = os.path.join(data_path, case_, 'images')
            case_name_list = os.listdir(case_path)
            case_path_list = [os.path.join(case_path, name) for name in case_name_list]  # 拼接完整路径
            # 获取长度为8的片段
            segments_0 = self.get_segments(case_path_list, segment_length=self.slice_length, stride=1, window_size=self.slice_length*1)
            # 获取长度为16的片段，并每隔2个元素取一个，形成长度为8的片段
            segments_1 = self.get_segments(case_path_list, segment_length=self.slice_length, stride=2, window_size=self.slice_length*2)
            # 获取长度为24的片段，并每隔3个元素取一个，形成长度为8的片段
            segments_2 = self.get_segments(case_path_list, segment_length=self.slice_length, stride=3, window_size=self.slice_length*3)
            # 将结果添加到 train_list 中
            train_list.extend(segments_0 + segments_1 + segments_2)
        return train_list

    def process_frame_with_prompts(self, image_farme_path):
        start_slice = random.randint(0, self.slice_length - 1)
        label2d = Image.open(image_farme_path[start_slice].replace('images', 'masks').replace('.jpg', '.png'))

        start_objs = np.unique(label2d)
        start_objs = start_objs[start_objs != 0]
        start_objs[start_objs == 128] = 0
        start_objs[start_objs == 255] = 1
        num_obj = min(len(start_objs), self.num_objs) 
        select_obj = random.sample(list(start_objs), num_obj)
        select_obj = list(map(int, select_obj))  

        all_image = torch.zeros(len(image_farme_path), 3, self.image_size, self.image_size)
        all_label = {obj: [] for obj in select_obj}
        all_prompt = {obj: {'text_prompt': {}} for obj in select_obj}
        all_ori_label = []
        for slice_index, image_path in enumerate(image_farme_path):
            image2d = Image.open(image_path).convert('RGB')
            label2d = Image.open(image_path.replace('images', 'masks').replace('.jpg', '.png'))

            all_ori_label.append(np.array(label2d))

            if np.array(label2d).sum() == 0 or len(np.unique(label2d)) < 2:
                return None

            slice_item = self.transform_2d({"image":image2d, "label":label2d})
            all_image[slice_index] = slice_item["image"]

            for idx, obj in enumerate(select_obj):
                all_label[obj].append(slice_item["label"][obj])
                if slice_index == start_slice:
                    all_prompt[obj]['text_prompt'][slice_index] = self.surgery_feat[obj]

        for obj in select_obj:
            all_label[obj] = torch.stack(all_label[obj], dim=0)

        all_ori_label = np.stack(all_ori_label, axis=0)
 
        return {'image':all_image, 'label': all_label, 'prompt':all_prompt, 'ori_label': all_ori_label}



class DataLoader_without_label(Dataset):
    def __init__(self, args):
        # 初始化函数，读取所有data_path下的图片
        self.image_size = args.image_size
        self.mode = args.mode
        self.slice_length = args.slice_length
        self.data_list = self.get_case_list(args.data_root)
    
        self.transform_2d = transforms.Compose(
                [
                    Resize(keys=["image"], target_size=(self.image_size, self.image_size), num_class=2),  #
                    transforms.ToTensord(keys=["image"]),
                    Normalization(keys=["image"]),
                ])

        surgery_table = torch.load("datasets/surgery_table.pt")
        surgical_instruments = torch.load("datasets/surgical_instruments.pt")
        self.surgery_feat = [surgery_table, surgical_instruments]

    def __getitem__(self, index):

        image_farme_path = self.data_list[index]
        output = self.process_frame_with_prompts(image_farme_path)
        if output is None:
            return self.__getitem__(np.random.randint(self.__len__()))
        return output

    def __len__(self):
        # 返回训练集大小
        return len(self.data_list)

    def get_case_list(self, data_path):
        case_paths = sorted(os.listdir(data_path))
        test_list = []
        for case_ in case_paths:
            case_path = os.path.join(data_path, case_)
            test_list.append(case_path)
    
        sliced_list = [
            test_list[i:i + self.slice_length] 
            for i in range(0, len(test_list), self.slice_length)
            ]

        return sliced_list

    def process_frame_with_prompts(self, image_farme_path):
        start_slice = random.randint(0, self.slice_length - 1)
        select_obj = [0,1] 
        all_image = torch.zeros(len(image_farme_path), 3, self.image_size, self.image_size)
        all_image_name = []
        all_prompt = {obj: {'text_prompt': {}} for obj in select_obj}
        all_ori_label = []

        for slice_index, image_path in enumerate(image_farme_path):
            image2d = Image.open(image_path).convert('RGB')
            all_image_name.append(image_path.split('/')[-1].split('\\')[-1])
            slice_item = self.transform_2d({"image":image2d})
            all_image[slice_index] = slice_item["image"]

            for idx, obj in enumerate(select_obj):
                if slice_index == start_slice:
                    all_prompt[obj]['text_prompt'][slice_index] = self.surgery_feat[obj]

        return {'image':all_image, 'prompt':all_prompt, 'image_name': all_image_name, 'ori_size':image2d.size}




def sample_collate_fn(batch):
    assert len(batch) == 1, 'Please set batch size to 1 when testing mode'
    pre_interval_image = batch[0]["image"].unsqueeze(0)
    return {
        "pre_interval_image": pre_interval_image,
        "pre_interval_label": {0: batch[0]["label"]},
        "pre_interval_prompt":{0: batch[0]["prompt"]},
        'ori_label': batch[0].get('ori_label', None),
    }


def collate_without_label(batch):
    assert len(batch) == 1, 'Please set batch size to 1 when testing mode'
    pre_interval_image = batch[0]["image"].unsqueeze(0)
    return {
        "pre_interval_image": pre_interval_image,
        "pre_interval_prompt":{0: batch[0]["prompt"]},
        "pre_interval_name":{0: batch[0]["image_name"]},
        "ori_size":batch[0]["ori_size"],
    }

def get_dataset(args):
    dataset = DataLoader(args)
    print(f'{args.mode} dataset: {len(dataset)}')
    data_sampler = DistributedSampler(dataset, shuffle=False) if args.dist else None
    data_loader = data.DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=(data_sampler is None),
        num_workers=args.num_workers,
        sampler=data_sampler,
        pin_memory=True,
        persistent_workers=False,
        collate_fn=sample_collate_fn,
        )
    return data_loader


def get_dataset_without_label(args):
    dataset = DataLoader_without_label(args)
    fram_length = len(os.listdir(args.data_root))
    video_name = args.data_root.split('/')[-1]
    print(f'Video: {video_name}, frame length: {fram_length}')
    data_loader = data.DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        sampler=None,
        pin_memory=True,
        persistent_workers=False,
        collate_fn=collate_without_label,
        )
    return data_loader


if __name__ == "__main__":
    from torchvision import transforms as torch_trans 
    import argparse
    import time
    # dist.init_process_group(backend='nccl', init_method='tcp://localhost:23456', rank=0, world_size=1)
    def set_parse():
        parser = argparse.ArgumentParser()
        parser.add_argument("--data_root", type = str, default='datasets/test/Knot_Tying/image/Knot_Tying_B001_capture1')
        parser.add_argument('--image_size', type=int, default=512)
        parser.add_argument('--batch_size', type=int, default=1)
        parser.add_argument('--slice_length', type=int, default=20)
        parser.add_argument('--num_workers', type=int, default=2)
        parser.add_argument('--num_objs', type=int, default=2)
        parser.add_argument('--mode', type = str, default='test')
        parser.add_argument('--dist', dest='dist', type=bool, default=False, help='distributed training or not')
        args = parser.parse_args()
        return args
    
    args = set_parse()

    train_loader = get_dataset_without_label(args)
    for idx, batch_input in enumerate(train_loader):
        interval_prompts = batch_input["pre_interval_prompt"]
        interval_images = batch_input["pre_interval_image"]
        pre_interval_name = batch_input["pre_interval_name"]
        print(interval_images.shape, interval_prompts[0].keys())
        # print(interval_images.shape)




    # local_path = "openai/clip-vit-base-patch32"  # 可以选择其他 CLIP 模型
    # model = CLIPModel.from_pretrained(local_path)
    # processor = CLIPProcessor.from_pretrained(local_path)

    # text1 = "a photo of a surgery table"
    # text2 = "a photo of a surgical instruments"

    # inputs1 = processor(text=text1, return_tensors="pt", padding=True)  # 将文本转换为模型输入
    # inputs2 = processor(text=text2, return_tensors="pt", padding=True)  # 将文本转换为模型输入
    # with torch.no_grad():
    #     text_feature1, text_feature2 = model.get_text_features(**inputs1), model.get_text_features(**inputs2)
    # print(text_feature1.shape, text_feature2.shape)

    # torch.save(text_feature1, "datasets/surgery_table.pt")
    # torch.save(text_feature2, "datasets/surgical_instruments.pt")



