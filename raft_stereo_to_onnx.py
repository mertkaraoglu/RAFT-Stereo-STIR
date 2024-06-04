import sys
sys.path.append('core')

import onnxruntime as ort
import argparse
import os
import cv2
import glob
import numpy as np
import torch
from PIL import Image
from torch import nn
import torch.nn.functional as F

from raft_stereo import RAFTStereo
from utils.utils import InputPadder

NUMITERS = 7

DEVICE = 'cuda'

def load_image(imfile):
    img = cv2.imread(imfile)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, (640, 512))
    img = torch.from_numpy(img).permute(2, 0, 1).float()
    return img[None].to(DEVICE)

def to_numpy(tensor):
        return tensor.detach().cpu().numpy() if tensor.requires_grad else tensor.cpu().numpy()

def viz(img, flo1, flo2):
    # TODO fix this
    img = img[0].permute(1,2,0).cpu().numpy()
    img = (img[:, :, 0] + img[:, :, 1] + img[:, :, 2]) / 3

    flo1 = flo1[0, 0].cpu().numpy()
    flo2 = flo2[0, 0].cpu().numpy()
    flo_desc = flo1 - flo2
    flo1 = (flo1 - flo1.min()) / (flo1.max() - flo1.min()) * 255.0
    flo2 = (flo2 - flo2.min()) / (flo2.max() - flo2.min()) * 255.0

    # map flow to rgb image
    img_flo1 = np.concatenate([img, flo1], axis=0)
    img_flo2 = np.concatenate([img, flo2], axis=0)

    cv2.imshow('flo1', img_flo1 / 255.0)
    cv2.imshow('flo2', img_flo2 / 255.0)
    cv2.waitKey()


def testconvertmodel(args):
    model = torch.nn.DataParallel(RAFTStereo(args))
    model.load_state_dict(torch.load(args.restore_ckpt))

    model = model.module
    model.to(DEVICE)
    model.eval()

    with torch.no_grad():
        images = glob.glob(os.path.join(args.path, '*.png')) + \
                 glob.glob(os.path.join(args.path, '*.jpg'))

        images = sorted(images)
        imfile1, imfile2 = images[0], images[1]
        image1 = load_image(imfile1)
        image2 = load_image(imfile2)

        padder = InputPadder(image1.shape)
        image1, image2 = padder.pad(image1, image2)

        flow_low, flow_up = model(image1, image2, iters=args.valid_iters, test_mode=True)
        kwargs = {"iters":args.valid_iters, "test_mode": True}
        # args = (image1, image2)
        onnx_model = torch.onnx.export(model,
                (image1, image2,kwargs),
                "iraftstereo_rvc.onnx",
                export_params=True,
                opset_version=17,
                do_constant_folding=True,
                input_names = ['image1', 'image2'],
                output_names = ['flow_low', 'flow_up'],
                verbose=False)
        ort_session = ort.InferenceSession("iraftstereo_rvc.onnx")

        outputs = ort_session.run(
                    None,
                    {"image1": to_numpy(image1), "image2": to_numpy(image2)},
                        )
        estimated_flow = torch.from_numpy(outputs[1]).to(DEVICE)
        if not torch.allclose(flow_up, estimated_flow, atol=1e-2, rtol=1e-2):
            print("ONNX model not close to true model")
            print("Two outputs has absoule average disc: ", (estimated_flow-flow_up).abs().mean())
        # viz(image1, flow_up, estimated_flow)

def convertmodeldirect(args):
    model = torch.nn.DataParallel(RAFTStereo(args))
    model.load_state_dict(torch.load(args.restore_ckpt))

    model = model.module
    model.to(DEVICE)
    model.eval()

    with torch.no_grad():
        image1 = torch.randn(1, 3, 512, 640, device="cuda")
        image2 = torch.randn(1, 3, 512, 640, device="cuda") ## THESE MUST be divisible by 8, otherwise need to pad using inputpadder

        flow_low, flow_up = model(image1, image2, iters=args.valid_iters, test_mode=True)
        kwargs = {"iters":args.valid_iters, "test_mode": True}
        # args = (image1, image2)
        onnx_model = torch.onnx.export(model,
                (image1, image2,kwargs),
                "iraftstereo_rvc.onnx",
                export_params=True,
                opset_version=17,
                do_constant_folding=True,
                input_names = ['image1', 'image2'],
                output_names = ['flow_low', 'flow_up'],
                verbose=False)


def bilinear_sampler(img, coords, mode='bilinear', mask=False):
    """ Wrapper for grid_sample, uses pixel coordinates """
    H, W = img.shape[-2:]
    xgrid, ygrid = coords.split([1,1], dim=-1)
    xgrid = 2*xgrid/(W-1) - 1
    ygrid = 2*ygrid/(H-1) - 1

    grid = torch.cat([xgrid, ygrid], dim=-1)
    img = F.grid_sample(img, grid, align_corners=True)

    if mask:
        mask = (xgrid > -1) & (ygrid > -1) & (xgrid < 1) & (ygrid < 1)
        return img, mask.float()

    return img

class RAFTStereoOnTrackingPoints(nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, pointlist, image1, image2):
        """ takes in pointlist of size 1 npts 2 (xy)
        im1 1 C H W
        im2 1 C H W
        returns:
        point_disparity: 1 npts 1"""
        _, flow_up = self.model(image1, image2, iters=args.valid_iters, test_mode=True)

        point_disparity = bilinear_sampler(flow_up, pointlist.unsqueeze(2))
        point_disparity = point_disparity.squeeze(3).permute(0, 2, 1) # N npts 1
        return point_disparity

def convertmodelpointtrack(args):
    model = torch.nn.DataParallel(RAFTStereo(args))
    model.load_state_dict(torch.load(args.restore_ckpt))

    model = model.module
    model.to(DEVICE)
    model.eval()
    raft_stereo_on_tracking_points = RAFTStereoOnTrackingPoints(model).to(DEVICE).eval()

    with torch.no_grad():
        image1 = torch.randn(1, 3, 512, 640, device="cuda")
        image2 = torch.randn(1, 3, 512, 640, device="cuda") ## THESE MUST be divisible by 8, otherwise need to pad using inputpadder
        points = torch.rand(1, 64, 2, device="cuda") * 512. 
        images = glob.glob(os.path.join(args.path, '*.png')) + \
                 glob.glob(os.path.join(args.path, '*.jpg'))

        TEST = False
        if TEST:
            images = sorted(images)
            imfile1, imfile2 = images[0], images[1]
            image1 = load_image(imfile1)
            image2 = load_image(imfile2)
            padder = InputPadder(image1.shape)
            image1, image2 = padder.pad(image1, image2)

        point_disparity = raft_stereo_on_tracking_points(points, image1, image2)
        args = (points, image1, image2)
        onnx_model = torch.onnx.export(raft_stereo_on_tracking_points,
                args,
                "iraftstereo_rvc_ontrackingpointsSTIR.onnx",
                export_params=True,
                opset_version=17,
                do_constant_folding=True,
                input_names = ['pointlist', 'image1', 'image2'],
                output_names = ['end_points'],
                dynamic_axes={ "pointlist": {1: "numpts"},
                    "end_points": [1],
                    },
                verbose=False)
        ort_session = ort.InferenceSession("iraftstereo_rvc_ontrackingpointsSTIR.onnx")
        print("Saved ONNX model")

        outputs = ort_session.run(
                    None,
                    {"pointlist": to_numpy(points), "image1": to_numpy(image1), "image2": to_numpy(image2)},
                        )
        estimated_point_disparity = torch.from_numpy(outputs[0]).to(DEVICE)
        if TEST:
            if not torch.allclose(point_disparity, estimated_point_disparity, atol=1e-2, rtol=1e-2):
                print("ONNX model not close to true model")
                # breakpoint()
        print(estimated_point_disparity-point_disparity)


        traced_track = torch.jit.trace(raft_stereo_on_tracking_points, (points, image1, image2))
        traced_track.save('iraftstereo_rvc_ontrackingpointsSTIR.pt')
        print("Saved torchscript model")

        loaded = torch.jit.load('iraftstereo_rvc_ontrackingpointsSTIR.pt')
        point_disparity_test = loaded(points, image1, image2)
        if TEST:
            if not torch.allclose(point_disparity, point_disparity_test):
                print("ONNX model not close to true model")
                # breakpoint()
                print(point_disparity_test-point_disparity)


# https://pytorch.org/docs/stable/onnx_torchscript.html
# https://pytorch.org/tutorials/advanced/super_resolution_with_onnxruntime.html

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--restore_ckpt', help="restore checkpoint")
    parser.add_argument('--save_numpy', action='store_true', help='save output as numpy arrays')
    parser.add_argument('--path', help="dataset for evaluation")
    parser.add_argument('--output_directory', help="directory to save output", default="demo_output")
    parser.add_argument('--mixed_precision', action='store_true', help='use mixed precision')
    parser.add_argument('--valid_iters', type=int, default=32, help='number of flow-field updates during forward pass')

    # Architecture choices
    parser.add_argument('--hidden_dims', nargs='+', type=int, default=[128]*3, help="hidden state and context dimensions")
    parser.add_argument('--corr_implementation', choices=["reg", "alt", "reg_cuda", "alt_cuda"], default="reg", help="correlation volume implementation")
    parser.add_argument('--shared_backbone', action='store_true', help="use a single backbone for the context and feature encoders")
    parser.add_argument('--corr_levels', type=int, default=4, help="number of levels in the correlation pyramid")
    parser.add_argument('--corr_radius', type=int, default=4, help="width of the correlation pyramid")
    parser.add_argument('--n_downsample', type=int, default=2, help="resolution of the disparity field (1/2^K)")
    parser.add_argument('--context_norm', type=str, default="batch", choices=['group', 'batch', 'instance', 'none'], help="normalization of context encoder")
    parser.add_argument('--slow_fast_gru', action='store_true', help="iterate the low-res GRUs more frequently")
    parser.add_argument('--n_gru_layers', type=int, default=3, help="number of hidden GRU levels")

    args = parser.parse_args()
    args.path = 'demo-frames'
    args.restore_ckpt =  'models/iraftstereo_rvc.pth'
    args.context_norm = "instance"

    testconvertmodel(args)
    convertmodeldirect(args)
    convertmodelpointtrack(args)