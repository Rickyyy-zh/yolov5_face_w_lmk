"""
Use COCO128 as the train dataset, with pre-trained weight and model yolov5s

"""
import detect
import train
import val
import export
import wandb

#detect.run(weights = 'yolov5s.pt',source = 'data/images/bus.jpg',device='cpu')
#wandb.init(project="yolov5_test_coco128")

def main():
    train.run(imgsz = 640, weight = 'yolov5s.pt', batch =16, epochs=5, data = 'data/coco128.yaml',device = '0',project="yolov5_test_coco128",num_workers=0)
    i = 1
    print(i)

if __name__ == '__main__':
    main()
"""
print('run normally')
def val_widerface():
    val.run(weights= 'yolov5s.pt',batch_size=32, imgsz=640, device='0', data= 'data/coco.yaml')
    print('run normally')

if __name__ == '__main__':
    val_widerface()
"""