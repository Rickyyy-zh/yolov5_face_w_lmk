import json

import sys

import numpy

from scipy import misc

import cv2

import os

import shutil

def draw_bb(image_dir, label_dir, out_path):

    img = cv2.imread(image_dir)

    shape = img.shape[:2]
    w_act = 1
    h_act = 1
    print(w_act,h_act)
    with open(label_dir,'r') as f:

        list_label = f.readlines()

        print(list_label)

        for i in range(len(list_label))     :

            list_label_tem = list_label[i].strip('\n')

            print(list_label_tem)

            new_list = list_label_tem.split(" ")

            print(new_list)

            list_label_tem2 = list(map(float,new_list[:5]))

            print(list_label_tem2)

            x, y, w, h = int(w_act*float(list_label_tem2[1])),int(h_act*float(list_label_tem2[2])),int(w_act*float(list_label_tem2[3])),int(h_act*float(list_label_tem2[4]))
            print(x,y,w,h)
            cv2.rectangle(img, (x,y), (x+w,y+h),(0,255,0), 1) #(, (x1,y1), (x2,y2),...) corner points


    cv2.imwrite( out_path+'test.jpg', img)


if __name__ == '__main__':
    draw_bb('data/images/trainwider_1.jpg', 'data/trainwider_1.txt', 'runs/detect/')

