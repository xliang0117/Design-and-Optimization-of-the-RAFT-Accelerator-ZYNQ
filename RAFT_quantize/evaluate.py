import sys
sys.path.append('core')

from PIL import Image
import argparse
import os
import time
import numpy as np
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
import cv2

import datasets
from utils import flow_viz
from utils import frame_utils

from raft import RAFT
from utils.utils import InputPadder, forward_interpolate

import brevitas


@torch.no_grad()
def create_sintel_submission(model, iters=32, warm_start=False, output_path='sintel_submission'):
    """ Create submission for the Sintel leaderboard """
    model.eval()
    for dstype in ['clean', 'final']:
        test_dataset = datasets.MpiSintel(split='test', aug_params=None, dstype=dstype)
        
        flow_prev, sequence_prev = None, None
        for test_id in range(len(test_dataset)):
            image1, image2, (sequence, frame) = test_dataset[test_id]
            if sequence != sequence_prev:
                flow_prev = None
            
            padder = InputPadder(image1.shape)
            image1, image2 = padder.pad(image1[None].cuda(), image2[None].cuda())
            images = torch.cat((image1, image2), dim=0)

            flow_low, flow_pr = model(images, iters=iters, flow_init=flow_prev, test_mode=True)
            flow = padder.unpad(flow_pr[0]).permute(1, 2, 0).cpu().numpy()

            if warm_start:
                flow_prev = forward_interpolate(flow_low[0])[None].cuda()
            
            output_dir = os.path.join(output_path, dstype, sequence)
            output_file = os.path.join(output_dir, 'frame%04d.flo' % (frame+1))

            if not os.path.exists(output_dir):
                os.makedirs(output_dir)

            frame_utils.writeFlow(output_file, flow)
            sequence_prev = sequence


@torch.no_grad()
def create_kitti_submission(model, iters=24, output_path='kitti_submission'):
    """ Create submission for the Sintel leaderboard """
    model.eval()
    test_dataset = datasets.KITTI(split='testing', aug_params=None)

    if not os.path.exists(output_path):
        os.makedirs(output_path)

    for test_id in range(len(test_dataset)):
        image1, image2, (frame_id, ) = test_dataset[test_id]
        padder = InputPadder(image1.shape, mode='kitti')
        image1, image2 = padder.pad(image1[None].cuda(), image2[None].cuda())
        images = torch.cat((image1, image2), dim=0)

        _, flow_pr = model(images, iters=iters, test_mode=True)
        flow = padder.unpad(flow_pr[0]).permute(1, 2, 0).cpu().numpy()

        output_filename = os.path.join(output_path, frame_id)
        frame_utils.writeFlowKITTI(output_filename, flow)


@torch.no_grad()
def validate_chairs(model, iters=24):
    """ Perform evaluation on the FlyingChairs (test) split """
    model.eval()
    epe_list = []

    val_dataset = datasets.FlyingChairs(split='validation')
    for val_id in range(len(val_dataset)):
        image1, image2, flow_gt, _ = val_dataset[val_id]
        image1 = image1[None].cuda()
        image2 = image2[None].cuda()
        images = torch.cat((image1, image2), dim=0)

        _, flow_pr, _ = model(images, iters=iters, test_mode=True)
        epe = torch.sum((flow_pr[0].cpu() - flow_gt)**2, dim=0).sqrt()
        epe_list.append(epe.view(-1).numpy())

    epe = np.mean(np.concatenate(epe_list))
    print("Validation Chairs EPE: %f" % epe)
    return {'chairs': epe}


@torch.no_grad()
def validate_sintel(model, iters=32, hist=False, farneback=False, lk=False):
    """ Peform validation using the Sintel (train) split """
    model.eval()
    results = {}
    histRes = np.zeros(11, dtype=np.int32)
    for dstype in ['clean', 'final']:
        val_dataset = datasets.MpiSintel(split='training', dstype=dstype)
        epe_list = []
        epe_fb_list = []
        epe_lk_list = []

        for val_id in range(len(val_dataset)):
            image1, image2, flow_gt, _ = val_dataset[val_id]
            image1 = image1[None].cuda()
            image2 = image2[None].cuda()

            padder = InputPadder(image1.shape)
            image1, image2 = padder.pad(image1, image2)
            images = torch.cat((image1, image2), dim=0)

            flow_low, flow_pr, histFlow = model(images, iters=iters, test_mode=True)
            flow = padder.unpad(flow_pr[0]).cpu()
            if farneback:
                img1 = cv2.cvtColor(image1.squeeze().permute(1,2,0).cpu().numpy(), cv2.COLOR_RGB2GRAY)
                img2 = cv2.cvtColor(image2.squeeze().permute(1,2,0).cpu().numpy(), cv2.COLOR_RGB2GRAY)
                flow_fb = cv2.calcOpticalFlowFarneback(img1, img2, None, pyr_scale=0.5, levels=3, winsize=15, iterations=3, poly_n=5, poly_sigma=1.1, flags=0)
                flow_fb = padder.unpad(torch.tensor(np.transpose(flow_fb, (2,0,1)))).cpu()
                epe_fb = torch.sum((flow_fb - flow_gt)**2, dim=0).sqrt()
                epe_fb_list.append(epe_fb.view(-1).numpy())
            if lk:
                img1 = cv2.cvtColor(padder.unpad(image1).squeeze().permute(1,2,0).cpu().numpy(), cv2.COLOR_RGB2GRAY).astype(np.uint8)
                img2 = cv2.cvtColor(padder.unpad(image2).squeeze().permute(1,2,0).cpu().numpy(), cv2.COLOR_RGB2GRAY).astype(np.uint8)
                feature_params = dict( maxCorners = 100, qualityLevel = 0.3, minDistance = 7, blockSize = 7 )
                lk_params = dict( winSize  = (15, 15), maxLevel = 2, criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03))
                p0 = cv2.goodFeaturesToTrack(img1, mask = None, **feature_params)
                p1, st, err = cv2.calcOpticalFlowPyrLK(img1, img2, p0, None, **lk_params)
                flow_gt_lk = flow_gt.numpy()
                if np.sum(st) != 0:
                    p0_valid = p0[st == 1].astype(np.uint32)
                    p1_valid = p1[st == 1]
                    flow_gt_valid = np.transpose(flow_gt_lk[:, p0_valid[:,1], p0_valid[:,0]])
                    # print(flow_gt_valid.shape)
                    # print(p0_valid.shape)
                    epe_lk_sum = np.sum(np.sqrt(np.sum((flow_gt_valid + p0_valid - p1_valid) ** 2, axis=1))) / np.sum(st)
                    epe_lk_list.append(epe_lk_sum)

            if hist:
                histRes = histFlow + histRes
            
            print("\rWaiting: {:.2f}%".format(val_id * 100 / len(val_dataset)), end="")
            sys.stdout.flush()

            epe = torch.sum((flow - flow_gt)**2, dim=0).sqrt()
            epe_list.append(epe.view(-1).numpy())

        epe_all = np.concatenate(epe_list)
        epe = np.mean(epe_all)
        px1 = np.mean(epe_all<1)
        px3 = np.mean(epe_all<3)
        px5 = np.mean(epe_all<5)

        print("Validation (%s) EPE: %f, 1px: %f, 3px: %f, 5px: %f" % (dstype, epe, px1, px3, px5))
        if hist:
            print("hist result: {}, 0~1:{}%".format(histRes, histRes[0]*100/np.sum(histRes)))
        if farneback:
            epe_fb_all = np.concatenate(epe_fb_list)
            epe_fb = np.mean(epe_fb_all)
            print("Farneback Validation (%s) EPE: %f" % (dstype, epe_fb))
        if lk:
            epe_lk = np.mean(epe_lk_list)
            print("lk Validation (%s) EPE: %f" % (dstype, epe_lk))
        results[dstype] = np.mean(epe_list)

    return results


@torch.no_grad()
def validate_middlebury(model, iters=32, hist=False):
    """ Peform validation using the Sintel (train) split """
    model.eval()
    results={}
    temp_results = []
    for scene in ['Dimetrodon', 'Grove2','Grove3', 'Hydrangea', 'RubberWhale', 'Venus']:
        val_dataset = datasets.Middlebury(scene=scene)
        epe_list = []

        for val_id in range(len(val_dataset)):
            image1, image2, flow_gt, _ = val_dataset[val_id]
            flow_gt = torch.where(torch.gt(flow_gt, 1e9), torch.tensor(0), flow_gt)
            image1 = image1[None].cuda()
            image2 = image2[None].cuda()

            padder = InputPadder(image1.shape)
            image1, image2 = padder.pad(image1, image2)
            images = torch.cat((image1, image2), dim=0)

            flow_low, flow_pr, histFlow = model(images, iters=iters, test_mode=True)
            flow = padder.unpad(flow_pr[0]).cpu()

            epe = torch.sum((flow - flow_gt)**2, dim=0).sqrt()
            epe_list.append(epe.view(-1).numpy())

        epe_all = np.concatenate(epe_list)
        epe = np.mean(epe_all)
        px1 = np.mean(epe_all<1)
        px3 = np.mean(epe_all<3)
        px5 = np.mean(epe_all<5)

        print("Validation (%s) EPE: %f, 1px: %f, 3px: %f, 5px: %f" % (scene, epe, px1, px3, px5))
        temp_results.append(np.mean(epe_list))

    results["middlebury"] = (sum(temp_results)/len(temp_results))
    return results

@torch.no_grad()
def validate_kitti(model, iters=24):
    """ Peform validation using the KITTI-2015 (train) split """
    model.eval()
    val_dataset = datasets.KITTI(split='training')

    out_list, epe_list = [], []
    for val_id in range(len(val_dataset)):
        image1, image2, flow_gt, valid_gt = val_dataset[val_id]
        image1 = image1[None].cuda()
        image2 = image2[None].cuda()

        padder = InputPadder(image1.shape, mode='kitti')
        image1, image2 = padder.pad(image1, image2)
        images = torch.cat((image1, image2), dim=0)

        flow_low, flow_pr, _ = model(images, iters=iters, test_mode=True)
        flow = padder.unpad(flow_pr[0]).cpu()

        epe = torch.sum((flow - flow_gt)**2, dim=0).sqrt()
        mag = torch.sum(flow_gt**2, dim=0).sqrt()

        epe = epe.view(-1)
        mag = mag.view(-1)
        val = valid_gt.view(-1) >= 0.5

        out = ((epe > 3.0) & ((epe/mag) > 0.05)).float()
        epe_list.append(epe[val].mean().item())
        out_list.append(out[val].cpu().numpy())

    epe_list = np.array(epe_list)
    out_list = np.concatenate(out_list)

    epe = np.mean(epe_list)
    f1 = 100 * np.mean(out_list)

    print("Validation KITTI: %f, %f" % (epe, f1))
    return {'kitti-epe': epe, 'kitti-f1': f1}


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', help="restore checkpoint")
    parser.add_argument('--dataset', help="dataset for evaluation")
    parser.add_argument('--small', action='store_true', help='use small model')
    parser.add_argument('--mixed_precision', action='store_true', help='use mixed precision')
    parser.add_argument('--alternate_corr', action='store_true', help='use efficent correlation implementation')
    parser.add_argument('--hist', action='store_true', help='hist for flow change')
    parser.add_argument('--del_norm', action='store_true', help='delete norm layer')
    parser.add_argument('--farneback', action='store_true', help='using farneback to evaluate optical flow')
    parser.add_argument('--lk', action='store_true', help='using lk to evaluate optical flow')
    args = parser.parse_args()

    brevitas.config.IGNORE_MISSING_KEYS = True

    model = RAFT(args)
    model.load_state_dict(torch.load(args.model))

    model.cuda()
    model.eval()

    # create_sintel_submission(model.module, warm_start=True)
    # create_kitti_submission(model.module)

    with torch.no_grad():
        if args.dataset == 'chairs':
            validate_chairs(model)

        elif args.dataset == 'sintel':
            validate_sintel(model, hist=args.hist, farneback=args.farneback, lk=args.lk)

        elif args.dataset == 'kitti':
            validate_kitti(model)

        elif args.dataset == 'middlebury':
            validate_middlebury(model)


