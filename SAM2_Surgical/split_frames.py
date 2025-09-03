import os
import cv2

def extract_frames_from_video(video_path, output_folder, frame_interval=5):
    """
    从视频中提取帧并保存到指定文件夹
    :param video_path: 视频文件路径
    :param output_folder: 保存帧的文件夹路径
    :param frame_interval: 帧间隔，默认为 5（每 5 帧保存一次）
    """
    # 创建输出文件夹
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    # 打开视频文件
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"无法打开视频文件: {video_path}")
        return

    frame_count = 0  # 视频帧计数器
    saved_frame_count = 0  # 保存的帧计数器

    while True:
        # 读取一帧
        ret, frame = cap.read()
        if not ret:
            break  # 视频读取完毕

        # 每 frame_interval 帧保存一次
        if frame_count % frame_interval == 0:
            # 保存帧为图片
            frame_filename = os.path.join(output_folder, f"frame_{saved_frame_count:04d}.jpg")
            cv2.imwrite(frame_filename, frame)
            saved_frame_count += 1

        frame_count += 1

    # 释放视频对象
    cap.release()
    print(f"视频 {video_path} 已提取 {saved_frame_count} 帧到 {output_folder}")


def extract_frames_from_folder(video_folder, frame_interval=5):
    """
    从文件夹中的所有 .avi 视频中提取帧
    :param video_folder: 包含 .avi 视频的文件夹路径
    :param frame_interval: 帧间隔，默认为 5（每 5 帧保存一次）
    """
    # 遍历文件夹中的所有文件
    for filename in os.listdir(video_folder):
        if filename.endswith(".avi"):
            # 获取视频文件的完整路径
            video_path = os.path.join(video_folder, filename)

            # 创建以视频名字命名的文件夹
            video_name = os.path.splitext(filename)[0]
            output_folder = os.path.join(video_folder.replace('\\video', '\\image'), video_name)

            # 提取帧并保存
            extract_frames_from_video(video_path, output_folder, frame_interval)


if __name__ == "__main__":
    # 设置视频文件夹路径
    video_folder = r"datasets/test/Knot_Tying/video"
    # 提取帧，每 5 帧保存一次
    extract_frames_from_folder(video_folder, frame_interval=5)