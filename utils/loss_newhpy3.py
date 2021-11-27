# Loss functions

import torch
import torch.nn as nn
import numpy as np
from utils.metrics import bbox_iou
from utils.torch_utils import is_parallel


def smooth_BCE(eps=0.1):  # https://github.com/ultralytics/yolov3/issues/238#issuecomment-598028441
    # return positive, negative label smoothing BCE targets
    return 1.0 - 0.5 * eps, 0.5 * eps


class BCEBlurWithLogitsLoss(nn.Module):
    # BCEwithLogitLoss() with reduced missing label effects.
    def __init__(self, alpha=0.05):
        super(BCEBlurWithLogitsLoss, self).__init__()
        self.loss_fcn = nn.BCEWithLogitsLoss(reduction='none')  # must be nn.BCEWithLogitsLoss()
        self.alpha = alpha

    def forward(self, pred, true):
        loss = self.loss_fcn(pred, true)
        pred = torch.sigmoid(pred)  # prob from logits
        dx = pred - true  # reduce only missing label effects
        # dx = (pred - true).abs()  # reduce missing label and false label effects
        alpha_factor = 1 - torch.exp((dx - 1) / (self.alpha + 1e-4))
        loss *= alpha_factor
        return loss.mean()


class FocalLoss(nn.Module):
    # Wraps focal loss around existing loss_fcn(), i.e. criteria = FocalLoss(nn.BCEWithLogitsLoss(), gamma=1.5)
    def __init__(self, loss_fcn, gamma=1.5, alpha=0.25):
        super(FocalLoss, self).__init__()
        self.loss_fcn = loss_fcn  # must be nn.BCEWithLogitsLoss()
        self.gamma = gamma
        self.alpha = alpha
        self.reduction = loss_fcn.reduction
        self.loss_fcn.reduction = 'none'  # required to apply FL to each element

    def forward(self, pred, true):
        loss = self.loss_fcn(pred, true)
        # p_t = torch.exp(-loss)
        # loss *= self.alpha * (1.000001 - p_t) ** self.gamma  # non-zero power for gradient stability

        # TF implementation https://github.com/tensorflow/addons/blob/v0.7.1/tensorflow_addons/losses/focal_loss.py
        pred_prob = torch.sigmoid(pred)  # prob from logits
        p_t = true * pred_prob + (1 - true) * (1 - pred_prob)
        alpha_factor = true * self.alpha + (1 - true) * (1 - self.alpha)
        modulating_factor = (1.0 - p_t) ** self.gamma
        loss *= alpha_factor * modulating_factor

        if self.reduction == 'mean':
            return loss.mean()
        elif self.reduction == 'sum':
            return loss.sum()
        else:  # 'none'
            return loss


class QFocalLoss(nn.Module):
    # Wraps Quality focal loss around existing loss_fcn(), i.e. criteria = FocalLoss(nn.BCEWithLogitsLoss(), gamma=1.5)
    def __init__(self, loss_fcn, gamma=1.5, alpha=0.25):
        super(QFocalLoss, self).__init__()
        self.loss_fcn = loss_fcn  # must be nn.BCEWithLogitsLoss()
        self.gamma = gamma
        self.alpha = alpha
        self.reduction = loss_fcn.reduction
        self.loss_fcn.reduction = 'none'  # required to apply FL to each element

    def forward(self, pred, true):
        loss = self.loss_fcn(pred, true)

        pred_prob = torch.sigmoid(pred)  # prob from logits
        alpha_factor = true * self.alpha + (1 - true) * (1 - self.alpha)
        modulating_factor = torch.abs(true - pred_prob) ** self.gamma
        loss *= alpha_factor * modulating_factor

        if self.reduction == 'mean':
            return loss.mean()
        elif self.reduction == 'sum':
            return loss.sum()
        else:  # 'none'
            return loss

"""
added by zq in 1st Sep
"""
class wingloss(nn.Module):
    def __init__(self,w=10,e=2):   # according to wingloss paper, w=10 and e=2 lead to lower loss
        super(wingloss, self).__init__()
        self.w = w
        self.e = e
        self.C = self.w - self.w * np.log(1 + self.w/self.e)

    def forward(self, x, t ):
        weight = torch.ones_like(t)
        weight[torch.where(t==-1)] = 0
        diff = weight * (x - t)

        abs_diff = diff.abs()

        flag = (abs_diff.data < self.w).float()
        y = flag * self.w * torch.log(1 + abs_diff / self.e) + (1 - flag) * (abs_diff - self.C)
        return y.sum()

class landmarksloss(nn.Module):
    def __init__(self, alpha = 1.0):
        super(landmarksloss, self).__init__()
        self.loss_fcn = wingloss()
        self.alpha = alpha

    def forward(self, pred, truel, mask):
        loss = self.loss_fcn(pred*mask, truel*mask)
        return loss / (torch.sum(mask)+10e-14)

class ComputeLoss:
    # Compute losses
    def __init__(self, model, autobalance=False):
        super(ComputeLoss, self).__init__()
        self.sort_obj_iou = False
        device = next(model.parameters()).device  # get model device
        h = model.hyp  # hyperparameters

        # Define criteria
        BCEcls = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([h['cls_pw']], device=device))
        BCEobj = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([h['obj_pw']], device=device))

        #add landmarks loss
        landmarks_loss = landmarksloss(1.0)

        # Class label smoothing https://arxiv.org/pdf/1902.04103.pdf eqn 3
        self.cp, self.cn = smooth_BCE(eps=h.get('label_smoothing', 0.0))  # positive, negative BCE targets

        # Focal loss
        g = h['fl_gamma']  # focal loss gamma
        if g > 0:
            BCEcls, BCEobj = FocalLoss(BCEcls, g), FocalLoss(BCEobj, g)

        det = model.module.model[-1] if is_parallel(model) else model.model[-1]  # Detect() module
        self.balance = {3: [4.0, 1.0, 0.4]}.get(det.nl, [4.0, 1.0, 0.25, 0.06, .02])  # P3-P7
        self.ssi = list(det.stride).index(16) if autobalance else 0  # stride 16 index
        self.BCEcls, self.BCEobj, self.gr, self.hyp, self.autobalance = BCEcls, BCEobj, model.gr, h, autobalance
        self.landmarks_loss = landmarks_loss
        for k in 'na', 'nc', 'nl', 'anchors':
            setattr(self, k, getattr(det, k))

    def __call__(self, p, targets):  # predictions, targets, model
        device = targets.device
        no = len(p)
        lcls, lbox, lobj, lmark = torch.zeros(1, device=device), torch.zeros(1, device=device), torch.zeros(1, device=device),torch.zeros(1, device=device)
        tcls, tbox, indices, anchors, tlandmarks, lmks_mask = self.build_targets(p, targets)  # targets

        # Losses
        for i, pi in enumerate(p):  # layer index, layer predictions
            b, a, gj, gi = indices[i]  # image, anchor, gridy, gridx
            tobj = torch.zeros_like(pi[..., 0], device=device)  # target obj

            n = b.shape[0]  # number of targets
            if n:
                ps = pi[b, a, gj, gi]  # prediction subset corresponding to targets

                # Regression
                pxy = ps[:, :2].sigmoid() * 2. - 0.5
                pwh = (ps[:, 2:4].sigmoid() * 2) ** 2 * anchors[i]
                pbox = torch.cat((pxy, pwh), 1)  # predicted box
                iou = bbox_iou(pbox.T, tbox[i], x1y1x2y2=False, CIoU=True)  # iou(prediction, target)
                lbox += (1.0 - iou).mean()  # iou loss

                # Objectness
                score_iou = iou.detach().clamp(0).type(tobj.dtype)
                if self.sort_obj_iou:
                    sort_id = torch.argsort(score_iou)
                    b, a, gj, gi, score_iou = b[sort_id], a[sort_id], gj[sort_id], gi[sort_id], score_iou[sort_id]
                tobj[b, a, gj, gi] = (1.0 - self.gr) + self.gr * score_iou  # iou ratio

                # Classification
                if self.nc > 1:  # cls loss (only if multiple classes)
                    t = torch.full_like(ps[:, 15:], self.cn, device=device)  # targets
                    t[range(n), tcls[i]] = self.cp
                    lcls += self.BCEcls(ps[:, 15:], t)  # BCE

                # Append targets to text file
                # with open('targets.txt', 'a') as file:
                #     [file.write('%11.5g ' * 4 % tuple(x) + '\n') for x in torch.cat((txy[i], twh[i]), 1)]

                #landmarks
                plandmarks = ps[:,5:15]
               	# print(plandmarks.shape,anchors[i].shape)
                plandmarks[:, 0:2] = plandmarks[:, 0:2] * anchors[i]
                plandmarks[:, 2:4] = plandmarks[:, 2:4] * anchors[i]
                plandmarks[:, 4:6] = plandmarks[:, 4:6] * anchors[i]
                plandmarks[:, 6:8] = plandmarks[:, 6:8] * anchors[i]
                plandmarks[:, 8:10] = plandmarks[:, 8:10] * anchors[i]

                lmark += self.landmarks_loss(plandmarks, tlandmarks[i], lmks_mask[i])



            obji = self.BCEobj(pi[..., 4], tobj)
            lobj += obji * self.balance[i]  # obj loss
            if self.autobalance:
                self.balance[i] = self.balance[i] * 0.9999 + 0.0001 / obji.detach().item()

        if self.autobalance:
            self.balance = [x / self.balance[self.ssi] for x in self.balance]

        s = 3/no

        lbox *= self.hyp['box']*s
        lobj *= self.hyp['obj']* s * (1.4 if no == 4 else 1.)
        lcls *= self.hyp['cls']*s
        lmark *= self.hyp['landmark'] * s

        bs = tobj.shape[0]  # batch size

        loss = lbox + lobj + lcls + lmark
        return loss * bs, torch.cat((lbox, lobj, lcls, lmark, loss)).detach()

    def build_targets(self, p, targets):
        # Build targets for compute_loss(), input targets(image,class,x,y,w,h)
        na, nt = self.na, targets.shape[0]  # number of anchors, targets
        #tcls, tbox, indices, anch = [], [], [], []
        tcls, tbox, indices, anch, landmarks, lmks_mask = [], [],[],[],[],[]
        #gain = torch.ones(7, device=targets.device)  # normalized to gridspace gain
        gain = torch.ones(17, device=targets.device)  # normalized to gridspace gain
        ai = torch.arange(na, device=targets.device).float().view(na, 1).repeat(1, nt)  # same as .repeat_interleave(nt)
        targets = torch.cat((targets.repeat(na, 1, 1), ai[:, :, None]), 2)  # append anchor indices

        g = 0.5  # bias
        off = torch.tensor([[0, 0],
                            [1, 0], [0, 1], [-1, 0], [0, -1],  # j,k,l,m
                            # [1, 1], [1, -1], [-1, 1], [-1, -1],  # jk,jm,lk,lm
                            ], device=targets.device).float() * g  # offsets

        for i in range(self.nl):
            anchors = self.anchors[i]
            gain[2:6] = torch.tensor(p[i].shape)[[3, 2, 3, 2]]  # xyxy gain
            gain[6:16] = torch.tensor(p[i].shape)[[3, 2, 3, 2, 3, 2, 3, 2, 3, 2]]
            # Match targets to anchors
            t = targets * gain
            if nt:
                # Matches
                r = t[:, :, 4:6] / anchors[:, None]  # wh ratio
                j = torch.max(r, 1. / r).max(2)[0] < self.hyp['anchor_t']  # compare
                # j = wh_iou(anchors, t[:, 4:6]) > model.hyp['iou_t']  # iou(3,n)=wh_iou(anchors(3,2), gwh(n,2))
                t = t[j]  # filter

                # Offsets
                gxy = t[:, 2:4]  # grid xy
                gxi = gain[[2, 3]] - gxy  # inverse
                j, k = ((gxy % 1. < g) & (gxy > 1.)).T
                l, m = ((gxi % 1. < g) & (gxi > 1.)).T
                j = torch.stack((torch.ones_like(j), j, k, l, m))
                t = t.repeat((5, 1, 1))[j]
                offsets = (torch.zeros_like(gxy)[None] + off[:, None])[j]
            else:
                t = targets[0]
                offsets = 0

            # Define
            b, c = t[:, :2].long().T  # image, class
            gxy = t[:, 2:4]  # grid xy
            gwh = t[:, 4:6]  # grid wh
            gij = (gxy - offsets).long()
            gi, gj = gij.T  # grid xy indices

            # Append
            a = t[:, 16].long()  # anchor indices
            indices.append((b, a, gj.clamp_(0, gain[3] - 1), gi.clamp_(0, gain[2] - 1)))  # image, anchor, grid indices
            tbox.append(torch.cat((gxy - gij, gwh), 1))  # box
            anch.append(anchors[a])  # anchors
            tcls.append(c)  # class

            #landmarks
            lks = t[:, 6:16]
            lks_mks = torch.where(lks <0, torch.full_like(lks,0), torch.full_like(lks, 1.0))
 
            lks[:, [0, 1]] = (lks[:, [0, 1]] - gij)
            lks[:, [2, 3]] = (lks[:, [2, 3]] - gij)
            lks[:, [4, 5]] = (lks[:, [4, 5]] - gij)
            lks[:, [6, 7]] = (lks[:, [6, 7]] - gij)
            lks[:, [8, 9]] = (lks[:, [8, 9]] - gij)
            """
	    #anch_w = torch.ones(5, device=targets.device).fill_(anchors[0][0])
            #anch_wh = torch.ones(5, device=targets.device)
            anch_f_0 = (a == 0).unsqueeze(1).repeat(1, 5)
            anch_f_1 = (a == 1).unsqueeze(1).repeat(1, 5)
            anch_f_2 = (a == 2).unsqueeze(1).repeat(1, 5)
            lks[:, [0, 2, 4, 6, 8]] = torch.where(anch_f_0, lks[:, [0, 2, 4, 6, 8]] / anchors[0][0], lks[:, [0, 2, 4, 6, 8]])
            lks[:, [0, 2, 4, 6, 8]] = torch.where(anch_f_1, lks[:, [0, 2, 4, 6, 8]] / anchors[1][0], lks[:, [0, 2, 4, 6, 8]])
            lks[:, [0, 2, 4, 6, 8]] = torch.where(anch_f_2, lks[:, [0, 2, 4, 6, 8]] / anchors[2][0], lks[:, [0, 2, 4, 6, 8]])

            lks[:, [1, 3, 5, 7, 9]] = torch.where(anch_f_0, lks[:, [1, 3, 5, 7, 9]] / anchors[0][1], lks[:, [1, 3, 5, 7, 9]])
            lks[:, [1, 3, 5, 7, 9]] = torch.where(anch_f_1, lks[:, [1, 3, 5, 7, 9]] / anchors[1][1], lks[:, [1, 3, 5, 7, 9]])
            lks[:, [1, 3, 5, 7, 9]] = torch.where(anch_f_2, lks[:, [1, 3, 5, 7, 9]] / anchors[2][1], lks[:, [1, 3, 5, 7, 9]])
	    #new_lks = lks[lks_mask>0]
            #print('new_lks:   min --- ', torch.min(new_lks), '  max --- ', torch.max(new_lks))

            lks_mask_1 = torch.where(lks < -3, torch.full_like(lks, 0.), torch.full_like(lks, 1.0))
            lks_mask_2 = torch.where(lks > 3, torch.full_like(lks, 0.), torch.full_like(lks, 1.0))

            lks_mask_new = lks_mks * lks_mask_1 * lks_mask_2
            lks_mask_new[:, 0] = lks_mask_new[:, 0] * lks_mask_new[:, 1]
            lks_mask_new[:, 1] = lks_mask_new[:, 0] * lks_mask_new[:, 1]
            lks_mask_new[:, 2] = lks_mask_new[:, 2] * lks_mask_new[:, 3]
            lks_mask_new[:, 3] = lks_mask_new[:, 2] * lks_mask_new[:, 3]
            lks_mask_new[:, 4] = lks_mask_new[:, 4] * lks_mask_new[:, 5]
            lks_mask_new[:, 5] = lks_mask_new[:, 4] * lks_mask_new[:, 5]
            lks_mask_new[:, 6] = lks_mask_new[:, 6] * lks_mask_new[:, 7]
            lks_mask_new[:, 7] = lks_mask_new[:, 6] * lks_mask_new[:, 7]
            lks_mask_new[:, 8] = lks_mask_new[:, 8] * lks_mask_new[:, 9]
            lks_mask_new[:, 9] = lks_mask_new[:, 8] * lks_mask_new[:, 9]
            """

            lks_mask_new = lks_mks
            lmks_mask.append(lks_mask_new)
            landmarks.append(lks)

        return tcls, tbox, indices, anch, landmarks, lmks_mask


