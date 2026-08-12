import sys
sys.path.append('droid_slam')

from tqdm import tqdm
import numpy as np
import torch
import lietorch
import cv2
import os
import glob 
import time
import argparse

from torch.multiprocessing import Process
from droid import Droid
from droid_async import DroidAsync

import torch.nn.functional as F


def show_image(image):
    image = image.permute(1, 2, 0).cpu().numpy()
    cv2.imshow('image', image / 255.0)
    cv2.waitKey(1)

def image_stream(imagedir, calib, depthdir=None, stride=1):
    """image generator yielding (t, image, depth, intrinsics)"""

    calib = np.loadtxt(calib, delimiter=" ")
    fx, fy, cx, cy = calib[:4]

    K = np.eye(3)
    K[0,0] = fx
    K[0,2] = cx
    K[1,1] = fy
    K[1,2] = cy

    image_list = sorted(os.listdir(imagedir))[::stride]

    for t, imfile in enumerate(image_list):
        # 1. Load RGB Frame
        image = cv2.imread(os.path.join(imagedir, imfile))
        if len(calib) > 4:
            image = cv2.undistort(image, K, calib[4:])

        h0, w0, _ = image.shape
        h1 = int(h0 * np.sqrt((384 * 512) / (h0 * w0)))
        w1 = int(w0 * np.sqrt((384 * 512) / (h0 * w0)))

        image = cv2.resize(image, (w1, h1))
        image = image[:h1-h1%8, :w1-w1%8]
        image = torch.as_tensor(image).permute(2, 0, 1)

        # 2. Rescale Camera Intrinsics
        intrinsics = torch.as_tensor([fx, fy, cx, cy])
        intrinsics[0::2] *= (w1 / w0)
        intrinsics[1::2] *= (h1 / h0)

        # 3. Load & Process Depth Map to (43, 70)
        depth_tensor = None
        if depthdir:
            depth_file = os.path.join(depthdir, os.path.basename(imfile))

            if not os.path.exists(depth_file):
                depth_list = sorted(glob.glob(os.path.join(depthdir, "*.png")))[
                    ::stride
                ]
                depth_file = depth_list[t] if t < len(depth_list) else None

            if depth_file and os.path.exists(depth_file):
                depth_mm = cv2.imread(depth_file, cv2.IMREAD_UNCHANGED)

                if depth_mm is not None:
                    if depth_mm.ndim == 3:
                        depth_mm = depth_mm[:, :, 0]

                    # Convert mm to meters
                    depth_m = depth_mm.astype(np.float32) / 1000.0

                    if len(calib) > 4:
                        depth_m = cv2.undistort(depth_m, K, calib[4:])

                    # Resize to (560, 350) using OpenCV's (width, height) order
                    depth_resized = cv2.resize(
                        depth_m, (w1, h1), interpolation=cv2.INTER_NEAREST
                    )

                    # Crop to multiples of 8
                    depth_cropped = depth_resized[: h1 - h1 % 8, : w1 - w1 % 8]

                    # Output 2D Float Tensor of shape
                    depth_tensor = (
                        torch.from_numpy(depth_cropped).float().to("cuda")
                    )

            yield t, image[None], depth_tensor, intrinsics
        else:
            yield t, image[None], intrinsics


def save_reconstruction(droid, save_path):

    if hasattr(droid, "video2"):
        video = droid.video2
    else:
        video = droid.video

    t = video.counter.value
    save_data = {
        "tstamps": video.tstamp[:t].cpu(),
        "images": video.images[:t].cpu(),
        "disps": video.disps_up[:t].cpu(),
        "poses": video.poses[:t].cpu(),
        "intrinsics": video.intrinsics[:t].cpu()
    }

    torch.save(save_data, save_path)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--imagedir", type=str, help="path to image directory")
    parser.add_argument("--depthdir", type=str, default=None, help="path to depth directory (optional)")
    parser.add_argument("--calib", type=str, help="path to calibration file")
    parser.add_argument("--t0", default=0, type=int, help="starting frame")
    parser.add_argument("--stride", default=3, type=int, help="frame stride")

    parser.add_argument("--weights", default="droid.pth")
    parser.add_argument("--buffer", type=int, default=512)
    parser.add_argument("--image_size", default=[240, 320])
    parser.add_argument("--disable_vis", action="store_true")

    parser.add_argument("--beta", type=float, default=0.3, help="weight for translation / rotation components of flow")
    parser.add_argument("--filter_thresh", type=float, default=2.4, help="how much motion before considering new keyframe")
    parser.add_argument("--warmup", type=int, default=8, help="number of warmup frames")
    parser.add_argument("--keyframe_thresh", type=float, default=4.0, help="threshold to create a new keyframe")
    parser.add_argument("--frontend_thresh", type=float, default=16.0, help="add edges between frames whithin this distance")
    parser.add_argument("--frontend_window", type=int, default=25, help="frontend optimization window")
    parser.add_argument("--frontend_radius", type=int, default=2, help="force edges between frames within radius")
    parser.add_argument("--frontend_nms", type=int, default=1, help="non-maximal supression of edges")

    parser.add_argument("--backend_thresh", type=float, default=22.0)
    parser.add_argument("--backend_radius", type=int, default=2)
    parser.add_argument("--backend_nms", type=int, default=3)
    parser.add_argument("--upsample", action="store_true")
    parser.add_argument("--asynchronous", action="store_true")
    parser.add_argument("--frontend_device", type=str, default="cuda")
    parser.add_argument("--backend_device", type=str, default="cuda")
    
    parser.add_argument("--reconstruction_path", help="path to saved reconstruction")
    args = parser.parse_args()

    args.stereo = False
    torch.multiprocessing.set_start_method('spawn')

    droid = None

    # need high resolution depths
    if args.reconstruction_path is not None:
        args.upsample = True

    tstamps = []
    for (t, image, disparity, intrinsics) in tqdm(image_stream(args.imagedir, args.calib, args.depthdir, args.stride)):
        if t < args.t0:
            continue

        if not args.disable_vis:
            show_image(image[0])

        if droid is None:
            args.image_size = [image.shape[2], image.shape[3]]
            droid = DroidAsync(args) if args.asynchronous else Droid(args)

        droid.track(t, image, depth=disparity, intrinsics=intrinsics)
    
    traj_est = droid.terminate(image_stream(args.imagedir, args.calib, depthdir=None, stride=args.stride)) # trajectory filler doesn't accept depth tensor
    
    if args.reconstruction_path is not None:
        save_reconstruction(droid, args.reconstruction_path)
