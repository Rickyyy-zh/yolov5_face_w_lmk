#import detect
import train
#import val

#device = "cuda" if torch.cuda.is_available() else "cpu"
#print("Using {} device".format(device)+torch.cuda.get_device_name(1))


def main():
    train.run(weights='yolov5_weights_210723/yolov5s.pt', batch_size=32, epochs=10, data='data/widerface_w_landmarks.yaml', device='cpu',
              project="test_WIDERface_landmarks_PC", single_cls= True, cfg='models/yolov5s_widerface.yaml', name='test_',upload_dataset= False,hyp = 'data/hyps/hyp.widerface_test.yaml')


def detect_test():
    detect.run(weights='yolov5_weights_210723/best100.pt', source='data/images/big face/', device='cpu',visualize=True)


def val_():
    val.run(weights='yolov5_weights_210723/yolov5s-face.pt', batch_size=32, device='cpu', name='test_lmk',project="test",
            data='data/widerface_w_landmarks.yaml', single_cls = True,task='val', imgsz=640 )


if __name__ == '__main__':
   main()
   #val_()
   #detect_test()
