import numpy as np
import cv2
import torch

source_directory_img_path="F:/zhangqi/datasets/WIDERface/WIDERface_easy/images/val"
source_directory_label_path="F:/zhangqi/datasets/WIDERface/WIDERface_easy/labels/val"
target_directory_path= "F:/zhangqi/datasets/WIDERface/WIDERface_easy/check_2"

#坐标转换，原始存储的是YOLOv5格式
# Convert nx4 boxes from [x, y, w, h] normalized to [x1, y1, x2, y2] where xy1=top-left, xy2=bottom-right
def xywhn2xyxy(x, w, h, padw=0, padh=0):

    y = x.clone() if isinstance(x, torch.Tensor) else np.copy(x)
    y[:, 0] = w * (x[:, 0] - x[:, 2] / 2) + padw  # top left x
    y[:, 1] = h * (x[:, 1] - x[:, 3] / 2) + padh  # top left y
    y[:, 2] = w * (x[:, 0] + x[:, 2] / 2) + padw  # bottom right x
    y[:, 3] = h * (x[:, 1] + x[:, 3] / 2) + padh  # bottom right y
    return y

import os
def draw_label(image_path,label_path):
    with open(label_path, 'r') as f:
        lb = np.array([x.split() for x in f.read().strip().splitlines()], dtype=np.float32)  # labels
        #print(lb)
    # 读取图像文件
    img = cv2.imread(str(image_path))
    h, w = img.shape[:2]
    print(h,w)
    lb[:, 1:] = xywhn2xyxy(lb[:, 1:], w, h, 0, 0)  # 反归一化
    #print(lb)

    # 绘图
    for _, x in enumerate(lb):
        class_label = int(x[0])  # class
        x = x.astype(int)
        cv2.rectangle(img, (x[1], x[2]), (x[3], x[4]), (0, 255, 0))
        cv2.putText(img, str(class_label), (int(x[1]), int(x[2] - 2)), fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=1,
                    color=(0, 0, 255), thickness=2)
    return     img

if __name__ == '__main__':
    for root, dirs, files in os.walk(source_directory_img_path):
            for f in files:
                file_name = f.split('.')[0]+".txt"
                image_path = os.path.join(source_directory_img_path, f)
                label_path =os.path.join(source_directory_label_path, file_name)
                target =os.path.join(target_directory_path, f)
                img= draw_label(image_path, label_path)
                print(target)
                cv2.imwrite(target, img);