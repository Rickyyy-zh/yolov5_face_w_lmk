import json
import sys
import numpy
from scipy import misc
import cv2
import os
import shutil


# file_dir = '/mnt/d/codes/datasets/widerface/wider_face_split/wider_face_test_filelist.txt'
file_dir = 'D:/codes/datasets/widerface/wider_face_split/'
train_image_dir = 'D:\\codes\\datasets\\WIDERface\\WIDER_train\\WIDER_train\\images\\'
val_image_dir = 'D:\\codes\\datasets\\WIDERface\\WIDER_val\\WIDER_val\\images\\'
test_image_dir = 'D:\\codes\\datasets\\WIDERface\\WIDER_test\\WIDER_test\\images\\'
train_output_dir = 'D:/codes/datasets/WIDERface/labels/train/'
val_output_dir = 'D:/codes/datasets/WIDERface/labels/val/'
img_train_out_dir = 'D:/codes/datasets/WIDERface/images/train/'
img_val_out_dir = 'D:/codes/datasets/WIDERface/images/val/'
img_test_out_dir = 'D:/codes/datasets/WIDERface/images/test/'
test_filename_dir = file_dir + 'wider_face_test_filelist.txt'
train_filename_dir = file_dir + 'wider_face_train_bbx_gt.txt'
val_filename_dir = file_dir + 'wider_face_val_bbx_gt.txt'



def read_cocojson():
    print('please type which json file you want to operate')
    #json_file_dir = '/mnt/d/codes/datasets/coco2017/annotations/'
    json_file_dir = 'D:/codes/datasets/coco2017/annotations/'
    json_file = json_file_dir + 'test.json'  #input('')

    data = json.load(open(json_file))

    data_info = data['info']
    data_image = data['images']
    data_cat = data['categories']
    print('The dataset info: %s ' %str(data_info['description']))
    print('The dataset has %d images, %d classes' %(len(data_image) ,len(data_cat)))

def modify_name(path):

    actname = str(path).strip('/.jpg\n')
    hash_position = actname.find('/')
    name = str(actname[hash_position+1:])
    dir_ = str(actname[:hash_position])
    return name, dir_

def modify_label(label_ori,org_shape):
    tem_lab = label_ori.split()
    tem_label = list(map(int, tem_lab[:4]))
    #print(tem_label)
    tem_label_new = {}
    x1 = tem_label[0] / org_shape[1] #x1
    y1 = tem_label[1] / org_shape[0] #y1
    w = tem_label[2] / org_shape[1] #w
    h = tem_label[3] / org_shape[0] #h
    x = x1 + w/2
    y = y1 + h/2
    tem_label_new[0] = x
    tem_label_new[1] = y
    tem_label_new[2] = w
    tem_label_new[3] = h
   #print(tem_label_new)
    str_label = list(map(str, list(tem_label_new.values())))
    #print(str_label)
    label = ' '.join(str_label)

    return label

def read_imageshape(dir):
    img = cv2.imread(dir)
    shape = img.shape[:2]  # (H, W)
    return shape

def convert(list_content, image_dir, out_dir):
    i = 0
    while i < len(list_content):
        if '.jpg' in str(list_content[i]):
            image_name, image_ab_dir = modify_name(list_content[i])
            img_shape = read_imageshape(image_dir + image_ab_dir + '\\' + image_name + '.jpg')
            num_face = int(list_content[i + 1])
            labels = ''
            lab = {}
            for j in range(num_face):  # find labels
                lab[j] = str('0 ' + modify_label(list_content[i + 1 + j + 1], img_shape) + '\n')
                labels = labels + lab[j]

            i = i + num_face + 2
            with open(out_dir + image_name + '.txt', 'w') as file_write:
                file_write.write(labels)

        else:
            i = i + 1
        print(i)

def widerface2yolo():


    #with open(test_filename_dir,'r') as test_file:
     #   list_test_filename = test_file.readlines()
    #modify_label([31,])

    with open(train_filename_dir, 'r') as train_file:
        list_train_content = train_file.readlines()
        convert(list_train_content, train_image_dir,train_output_dir)
        """
        name = modify_name(list_train_content[3])
        print(name)
        label_1 = '0 ' + modify_label(list_train_content[2])
        label_2 = '0 ' + modify_label(list_train_content[2])
        label = label_2+'\n'+label_1
        print(label_1)
        print(label_2)

    write_file = open('D:/codes/datasets/WIDERface/test/'+name+'.txt', 'w')
    write_file.write( label + '\n')


        i = 0
        while i <len(list_train_content):  #find image name
            if '.jpg' in str(list_train_content[i]):
                image_name, image_ab_dir = modify_name(list_train_content[i])
                img_shape = read_imageshape(image_dir+image_ab_dir+'\\'+image_name+'.jpg')
                num_face = int(list_train_content[i+1])
                labels = ''
                lab = {}
                for j in range(num_face):  #find labels
                    lab[j] = str('0 ' + modify_label(list_train_content[i+1+j+1],img_shape) + '\n')
                    labels = labels + lab[j]

                i = i+num_face+2
                with open(output_dir + image_name + '.txt' , 'w') as file_write:
                    file_write.write(labels)

            else:
                i = i+1
            print(i)
"""
    with open(val_filename_dir,'r') as val_file:
        list_val_content = val_file.readlines()
        convert(list_val_content, val_image_dir,val_output_dir)


def image_copy(image_dir, img_out_dir):
    list_doc_name = os.listdir(image_dir);
    for i in range(len(list_doc_name)):
        list_file_name = os.listdir(image_dir+list_doc_name[i])
        for j in range(len(list_file_name)):
            shutil.copyfile(image_dir+list_doc_name[i]+'\\'+list_file_name[j], img_out_dir+list_file_name[j])


def draw_bb(image_dir, label_dir, out_path):
    img = cv2.imread(image_dir)
    shape = img.shape[:2]
    with open(label_dir,'r') as f:
        list_label = f.readlines()
        print(list_label)
        for i in range(len(list_label))     :
            list_label_tem = list_label[i].strip('\n')
            print(list_label_tem)
            new_list = list_label_tem.split(" ")
            print(new_list)
            list_label_tem2 = list(map(int,new_list[:5]))
            print(list_label_tem2)
            x, y, w, h = int(list_label_tem2[0]),int(list_label_tem2[1]), int(list_label_tem2[2]),int(list_label_tem2[3])
            cv2.rectangle(img, (x,y), (x+w,y+h),(0,255,0), 1) #(, (x1,y1), (x2,y2),...) corner points

    cv2.imwrite( out_path+'test.jpg', img)



def main():
    #widerface2yolo(int(sys.argv[1]))
    widerface2yolo()
    #image_copy(train_image_dir,img_train_out_dir)
    #image_copy(val_image_dir, img_val_out_dir)
    #image_copy(test_image_dir, img_test_out_dir)


if __name__ == '__main__':
    #draw_bb('D:/codes/datasets/WIDERface/images/train/0_Parade_Parade_0_782.jpg', 'D:/codes/datasets/WIDERface/test/0_Parade_Parade_0_782.txt', 'D:/codes/datasets/WIDERface/test/')
    main()