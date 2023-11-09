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
def validate_chairs(model, iters=20, hist=False):
    """ Perform evaluation on the FlyingChairs (test) split """
    model.eval()
    epe_list = []
    flowPRRes = [torch.zeros((3072, 5, 5), dtype=torch.int32) for x in range(iters-1)]
    flowPRSquareRes = [torch.zeros((3072, 5, 5), dtype=torch.int32) for x in range(iters-1)]
    centerPRSquareRes = [torch.zeros((3072), dtype=torch.int32) for x in range(iters-1)]
    comeAndStayRes = [torch.zeros((3072), dtype=torch.int32) for x in range(iters-1)]

    val_dataset = datasets.FlyingChairs(split='validation')
    for val_id in range(len(val_dataset)):
        image1, image2, flow_gt, _ = val_dataset[val_id]
        image1 = image1[None].cuda()
        image2 = image2[None].cuda()
        images = torch.cat((image1, image2), dim=0)

        _, flow_pr, flowPR = model(images, iters=iters, test_mode=True)
        if hist:
            for i in range(iters-1):
                flowPRRes[i] = flowPR[0][i] + flowPRRes[i]
                flowPRSquareRes[i] = flowPR[1][i] + flowPRSquareRes[i]
                centerPRSquareRes[i] = flowPR[2][i] + centerPRSquareRes[i]
                comeAndStayRes[i] = flowPR[3][i] + comeAndStayRes[i]
            print("\rWating: {:.2f}%".format(val_id * 100 / len(val_dataset)), end="")
            sys.stdout.flush()
        epe = torch.sum((flow_pr[0].cpu() - flow_gt)**2, dim=0).sqrt()
        epe_list.append(epe.view(-1).numpy())

    epe = np.mean(np.concatenate(epe_list))
    print("Validation Chairs EPE: %f" % epe)
    if hist:
        for i in range(iters-1):
            np.save("hist/flow_dir_pr_{}.npy".format(i), flowPRRes[i].cpu().numpy())
            np.save("hist/flow_dir_pr_square_{}.npy".format(i), flowPRSquareRes[i].cpu().numpy())
            np.save("hist/center_pr_square_{}.npy".format(i), (centerPRSquareRes[i]/len(val_dataset)).cpu().numpy())
            np.save("hist/come_and_stay_{}.npy".format(i), (comeAndStayRes[i]/len(val_dataset)).cpu().numpy())
    return {'chairs': epe}


@torch.no_grad()
def validate_sintel(model, iters=10, hist=False):
    """ Peform validation using the Sintel (train) split """
    model.eval()
    results = {}

    for dstype in ['clean', 'final']:
        flowPRRes = [torch.zeros((55*128, 5, 5), dtype=torch.int32) for x in range(iters-1)]
        flowPRSquareRes = [torch.zeros((55*128, 5, 5), dtype=torch.int32) for x in range(iters-1)]
        centerPRSquareRes = [torch.zeros((1), dtype=torch.int32) for x in range(iters-1)]
        comeAndStayRes = [torch.zeros((1), dtype=torch.float32) for x in range(iters-1)]
        val_dataset = datasets.MpiSintel(split='training', dstype=dstype)
        epe_list = []

        for val_id in range(len(val_dataset)):
            image1, image2, flow_gt, _ = val_dataset[val_id]
            image1 = image1[None].cuda()
            image2 = image2[None].cuda()

            padder = InputPadder(image1.shape)
            image1, image2 = padder.pad(image1, image2)
            images = torch.cat((image1, image2), dim=0)

            flow_low, flow_pr, flowPR = model(images, iters=iters, test_mode=True)
            flow = padder.unpad(flow_pr[0]).cpu()
            if hist:
                for i in range(iters-1):
                    flowPRRes[i] = flowPR[0][i] + flowPRRes[i]
                    flowPRSquareRes[i] = flowPR[1][i] + flowPRSquareRes[i]
                    centerPRSquareRes[i] = flowPR[2][i] + centerPRSquareRes[i]
                    comeAndStayRes[i] = flowPR[3][i] + comeAndStayRes[i]
                print("\rWating: {:.2f}%".format(val_id * 100 / len(val_dataset)), end="")
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
            for i in range(iters-1):
                np.save("hist/flow_dir_pr_{}_{}.npy".format(dstype, i), flowPRRes[i].cpu().numpy())
                np.save("hist/flow_dir_pr_square_{}_{}.npy".format(dstype, i), flowPRSquareRes[i].cpu().numpy())
                np.save("hist/center_pr_square_{}_{}.npy".format(dstype, i), (centerPRSquareRes[i]/len(val_dataset)).cpu().numpy())
                np.save("hist/come_and_stay_{}_{}.npy".format(dstype, i), (comeAndStayRes[i]/len(val_dataset)).cpu().numpy())

        results[dstype] = np.mean(epe_list)

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
            validate_chairs(model, hist=args.hist)

        elif args.dataset == 'sintel':
            validate_sintel(model, hist=args.hist)

        elif args.dataset == 'kitti':
            validate_kitti(model)


