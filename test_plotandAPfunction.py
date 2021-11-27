import torch
from pathlib import Path
import numpy as np
from utils.torch_utils import select_device,time_sync

from models.experimental import attempt_load
from utils.face_datasets import create_dataloader
from utils.metrics import ConfusionMatrix ,ap_per_class

from tqdm import tqdm
from utils.general import coco80_to_coco91_class, check_dataset, check_file, check_img_size, check_requirements, \
    box_iou, non_max_suppression,non_max_suppression_face, scale_coords, xyxy2xywh, xywh2xyxy, set_logging, increment_path, colorstr
from threading import Thread
from utils.plots import plot_images, output_to_target, plot_one_box
class Params:
    def setKpParams(self):
        self.imgIds = []
        self.catIds = []
        # np.arange causes trouble.  the data point on arange is slightly larger than the true value
        self.iouThrs = np.linspace(.5, 0.95, int(np.round((0.95 - .5) / .05)) + 1, endpoint=True)
        self.recThrs = np.linspace(.0, 1.00, int(np.round((1.00 - .0) / .01)) + 1, endpoint=True)
        self.maxDets = [20]
        self.areaRng = [[0 ** 2, 1e5 ** 2], [32 ** 2, 96 ** 2], [96 ** 2, 1e5 ** 2]]
        self.areaRngLbl = ['all', 'medium', 'large']
        self.useCats = 1
        self.kpt_oks_sigmas = np.array([.26, .25, .25, .35, .35, .79, .79, .72, .72, .62,.62, 1.07, 1.07, .87, .87, .89, .89])/10.0

    def __init__(self, iouType='segm'):

        self.setKpParams()
        
def computeOks(self, imgId, catId):
    p = self.params
    # dimention here should be Nxm
    gts = self._gts[imgId, catId]
    dts = self._dts[imgId, catId]
    inds = np.argsort([-d['score'] for d in dts], kind='mergesort')
    dts = [dts[i] for i in inds]
    if len(dts) > p.maxDets[-1]:
        dts = dts[0:p.maxDets[-1]]
    # if len(gts) == 0 and len(dts) == 0:
    if len(gts) == 0 or len(dts) == 0:
        return []
    ious = np.zeros((len(dts), len(gts)))
    sigmas = p.kpt_oks_sigmas
    vars = (sigmas * 2)**2
    k = len(sigmas)
    # compute oks between each detection and ground truth object
    for j, gt in enumerate(gts):
        # create bounds for ignore regions(double the gt bbox)
        g = np.array(gt['keypoints'])
        xg = g[0::3]; yg = g[1::3]; vg = g[2::3]
        k1 = np.count_nonzero(vg > 0)
        bb = gt['bbox']
        x0 = bb[0] - bb[2]; x1 = bb[0] + bb[2] * 2
        y0 = bb[1] - bb[3]; y1 = bb[1] + bb[3] * 2
        for i, dt in enumerate(dts):
            d = np.array(dt['keypoints'])
            xd = d[0::3]; yd = d[1::3]
            if k1>0:
                # measure the per-keypoint distance if keypoints visible
                dx = xd - xg
                dy = yd - yg
            else:
                # measure minimum distance to keypoints in (x0,y0) & (x1,y1)
                z = np.zeros((k))
                dx = np.max((z, x0-xd),axis=0)+np.max((z, xd-x1),axis=0)
                dy = np.max((z, y0-yd),axis=0)+np.max((z, yd-y1),axis=0)
            e = (dx**2 + dy**2) / vars / (gt['area']+np.spacing(1)) / 2
            if k1 > 0:
                e=e[vg > 0]
            ious[i, j] = np.sum(np.exp(-e)) / e.shape[0]
    return ious

def process_batch(predictions, labels, iouv):
    # Evaluate 1 batch of predictions
    correct = torch.zeros(predictions.shape[0], len(iouv), dtype=torch.bool, device=iouv.device)
    detected = []  # label indices
    tcls, pcls = labels[:, 0], predictions[:, 5]
    nl = labels.shape[0]  # number of labels
    for cls in torch.unique(tcls):
        ti = (cls == tcls).nonzero().view(-1)  # label indices
        pi = (cls == pcls).nonzero().view(-1)  # prediction indices
        if pi.shape[0]:  # find detections
            ious, i = box_iou(predictions[pi, 0:4], labels[ti, 1:5]).max(1)  # best ious, indices
            detected_set = set()
            for j in (ious > iouv[0]).nonzero():
                d = ti[i[j]]  # detected label
                if d.item() not in detected_set:
                    detected_set.add(d.item())
                    detected.append(d)  # append detections
                    correct[pi[j]] = ious[j] > iouv  # iou_thres is 1xn
                    if len(detected) == nl:  # all labels already located in image
                        break
    return correct    

def main(weights, batch_size = 16, device = 'cpu'):
    label_path = './data/widerfacetest100/lables/val'
    img_path = './data/widerfacetest100/images/val'
    imgsz = 640
    nc = 1
    device = select_device(device, batch_size=batch_size) 
    model = attempt_load(weights, map_location=device)  # load FP32 model
    gs = max(int(model.stride.max()), 32)  # grid size (max stride)
    iouv = torch.linspace(0.5, 0.95, 10).to(device)  # iou vector for mAP@0.5:0.95
    niou = iouv.numel()
    model.half().float()
    model(torch.zeros(1, 3, imgsz, imgsz).to(device).type_as(next(model.parameters())))  # run once
    dataloader = create_dataloader(img_path, imgsz, batch_size, gs,False, pad=0.5, rect=True)[0]
    
    
    seen = 0
    #confusion_matrix = ConfusionMatrix(nc=nc)
    names = {k: v for k, v in enumerate(model.names if hasattr(model, 'names') else model.module.names)}
    #class_map = list(range(1000))
    s = ('%20s' + '%11s' * 6) % ('Class', 'Images', 'Labels', 'P', 'R', 'mAP@.5', 'mAP@.5:.95')
    p, r, f1, mp, mr, map50, map, t0, t1, t2 = 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.
    #loss = torch.zeros(4, device=device)
    jdict, stats, ap, ap_class = [], [], [], []
    for batch_i, (img, targets, paths, shapes) in enumerate(tqdm(dataloader, desc=s)):
        t_ = time_sync()
        img = img.to(device, non_blocking=True)
        img = img.half().float()   # uint8 to fp16/32
        img /= 255.0  # 0 - 255 to 0.0 - 1.0
        targets = targets.to(device)
        nb, _, height, width = img.shape  # batch size, channels, height, width
        t = time_sync()
        t0 += t - t_

        # Run model
        out, train_out = model(img, augment=False)  # inference and training outputs
        t1 += time_sync() - t

        # Compute loss
        #if compute_loss:
            #loss += compute_loss([x.float() for x in train_out], targets)[1][:4]  # box, obj, cls

        # Run NMS
        targets[:, 2:6] *= torch.Tensor([width, height, width, height]).to(device)  # to pixels
        #lb = [targets[targets[:, 0] == i, 1:] for i in range(nb)] if save_hybrid else []  # for autolabelling
        t = time_sync()
        out = non_max_suppression_face(out, 0.001, 0.6, labels=[])
        t2 += time_sync() - t

        # Statistics per image
        for si, pred in enumerate(out):
            labels = targets[targets[:, 0] == si, 1:]
            nl = len(labels)
            tcls = labels[:, 0].tolist() if nl else []  # target class
            path, shape = Path(paths[si]), shapes[si][0]
            seen += 1
            if len(pred) == 0:
                if nl:
                    stats.append((torch.zeros(0, niou, dtype=torch.bool), torch.Tensor(), torch.Tensor(), tcls))
                continue

            # Predictions

            predn = pred.clone()
            scale_coords(img[si].shape[1:], predn[:, :4], shape, shapes[si][1])  # native-space pred

            # Evaluate
            if nl:
                tbox = xywh2xyxy(labels[:, 1:5])  # target boxes
                scale_coords(img[si].shape[1:], tbox, shape, shapes[si][1])  # native-space labels
                labelsn = torch.cat((labels[:, 0:1], tbox), 1)  # native-space labels
                correct = process_batch(predn, labelsn, iouv)

            else:
                correct = torch.zeros(pred.shape[0], niou, dtype=torch.bool)
            stats.append((correct.cpu(), pred[:, 4].cpu(), pred[:, 5].cpu(), tcls))  # (correct, conf, pcls, tcls)
            print(pred[:,4])
        # Plot images
        f = './runs/testPic'
        f1 = f+'/oriPic'
        f2 = f+'/predPic'
        plot_images(img, targets, paths =None, fname = f1+str(batch_i)+'.jpg')
        plot_images(img, output_to_target(out), paths =None, fname = f2+str(batch_i)+'.jpg')
        #Thread(target=plot_images, args=(img, targets, paths, f1, names), daemon=True).start()
        #Thread(target=plot_images, args=(img, output_to_target(out), paths, f2, names), daemon=True).start()

    # Compute statistics
    stats = [np.concatenate(x, 0) for x in zip(*stats)]  # to numpy
    if len(stats) and stats[0].any():
        p, r, ap, f1, ap_class = ap_per_class(*stats, plot=False, save_dir=f, names=names)
        ap50, ap = ap[:, 0], ap.mean(1)  # AP@0.5, AP@0.5:0.95
        mp, mr, map50, map = p.mean(), r.mean(), ap50.mean(), ap.mean()
        nt = np.bincount(stats[3].astype(np.int64), minlength=nc)  # number of targets per class
    else:
        nt = torch.zeros(1)

    # Print results
    pf = '%20s' + '%11i' * 2 + '%11.3g' * 4  # print format
    print(pf % ('all', seen, nt.sum(), mp, mr, map50, map))
    
"""    # Print results per class
    if (verbose or (nc < 50 and not training)) and nc > 1 and len(stats):
        for i, c in enumerate(ap_class):
            print(pf % (names[c], seen, nt[c], p[i], r[i], ap50[i], ap[i]))

    # Print speeds
    t = tuple(x / seen * 1E3 for x in (t0, t1, t2))  # speeds per image
    if not training:
        shape = (batch_size, 3, imgsz, imgsz)
        print(f'Speed: %.1fms pre-process, %.1fms inference, %.1fms NMS per image at shape {shape}' % t)
"""

if __name__ == "__main__":
    main(weights= 'weights\stem_lr0.005_100.pt',batch_size = 16, device = 'cpu')