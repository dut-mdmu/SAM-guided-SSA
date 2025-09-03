##Suturing
 # python main_jigsaws_group.py --split_index 0 --num_samples 32 --batch_size 4 --val_split SuperTrialOut --task Suturing --num_epochs 40 --schedule_step 20 --learning_rate 3e-5 --num_parts 3 --shaping_weight 10 --scene_node --visualize

 # python main_jigsaws_group.py --split_index 0 --num_samples 32 --batch_size 4 --val_split SuperTrialOut --task Across --num_epochs 40 --schedule_step 20 --learning_rate 3e-5 --num_parts 3 --shaping_weight 10 --scene_node --visualize

import cv2
import os
import re

def sample_video(video_path, output_folder, fps, width, start_frame_number):
    # 创建输出目录（如果不存在）
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    # 打开视频文件
    cap = cv2.VideoCapture(video_path)
    
    # 获取视频的原始帧率
    original_fps = cap.get(cv2.CAP_PROP_FPS)
    
    # 计算每秒需要跳过的帧数
    skip_frames = int(original_fps / fps)
    
    # 初始化帧计数器
    frame_count = 0
    current_frame_number = start_frame_number
    
    while True:
        ret, frame = cap.read()
        
        if not ret:
            break
        
        # 每隔skip_frames帧保存一次
        if frame_count % skip_frames == 0:
            # 调整帧大小
            resized_frame = cv2.resize(frame, (width, int(width * frame.shape[0] / frame.shape[1])))
            
            # 保存帧
            frame_number = str(current_frame_number).zfill(5)  # 从1开始编号
            frame_path = os.path.join(output_folder, f'{frame_number}.jpg')
            cv2.imwrite(frame_path, resized_frame)
            
            current_frame_number += 1
        
        frame_count += 1
    
    # 释放视频文件
    cap.release()
    
    return current_frame_number

def batch_process_videos(input_dir, output_root_dir, fps, width):
    # 收集所有视频文件
    videos = []
    for filename in os.listdir(input_dir):
        if filename.endswith('.avi'):
            video_path = os.path.join(input_dir, filename)
            
            # 使用正则表达式提取视频编号和capture编号
            match = re.search(r'Suturing_([A-Z]\d+)_capture(\d+)', filename)
            if match:
                video_number = match.group(1)
                capture_number = int(match.group(2))
                videos.append((video_path, video_number, capture_number))
            else:
                print(f"无法解析视频编号: {filename}")
    
    # 按视频编号和capture编号排序
    videos.sort(key=lambda x: (x[1], x[2]))
    
    # 维护一个字典来跟踪每个视频编号的当前帧计数器
    frame_counters = {}
    
    # 处理每个视频
    for video_path, video_number, _ in videos:
        # 获取或初始化当前视频编号的帧计数器
        if video_number not in frame_counters:
            frame_counters[video_number] = 1
        
        output_folder = os.path.join(output_root_dir, video_number, 'frame')
        start_frame_number = frame_counters[video_number]
        
        # 处理视频并更新帧计数器
        frame_counters[video_number] = sample_video(video_path, output_folder, fps, width, start_frame_number)

# 定义输入文件夹和输出根目录
input_dir = '/root/autodl-tmp/Surgical-Skill-Assessment-via-Video-Semantic-Aggregation/data/jigsaws/jigsaws_annot/Suturing/video'
output_root_dir = '/root/autodl-tmp/Surgical-Skill-Assessment-via-Video-Semantic-Aggregation/data/jigsaws/Suturing/Suturing_160x120'

# 调用批量处理函数
batch_process_videos(input_dir, output_root_dir, fps=5, width=160)
