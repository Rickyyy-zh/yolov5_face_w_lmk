import random
import os
from shutil import copyfile
source_path = 'F:\\zhangqi\\datasets\\WIDERface\\WIDERface_yolo_w_landmarks\\'
tar_path = 'F:\\zhangqi\\datasets\\WIDERface\\widerfacetest100\\'
l = []
v = []
for i in range(100):
    l.append('trainwider_'+str(random.randint(1,12880)))

for k in l:
    s_lables = source_path+'labels\\train\\'+k+'.txt' 
    t_lables = tar_path+'labels\\train\\'+k+'.txt' 
    s_image = source_path+'images\\train\\'+k+'.jpg'
    t_image = tar_path+'images\\train\\'+k+'.jpg' 

    os.system('copy %s %s' % (s_lables,t_lables))
    os.system('copy %s %s' % (s_image,t_image))

for j in range(50):
    v.append('trainwider_'+str(random.randint(1,12880)))

for x in v:
    s_lables = source_path+'labels\\train\\'+x+'.txt' 
    t_lables = tar_path+'labels\\val\\'+x+'.txt' 
    s_image = source_path+'images\\train\\'+x+'.jpg'
    t_image = tar_path+'images\\val\\'+x+'.jpg' 

    os.system('copy %s %s' % (s_lables,t_lables))
    os.system('copy %s %s' % (s_image,t_image))