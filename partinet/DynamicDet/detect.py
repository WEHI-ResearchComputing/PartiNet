import argparse
import time
import logging
import os
from pathlib import Path
import queue
from threading import Thread
import sys

import cv2
import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn
from numpy import random

import partinet.DynamicDet
from partinet.DynamicDet.models.yolo import Model
from partinet.DynamicDet.utils.datasets import LoadStreams, LoadImages
from partinet.DynamicDet.utils.general import (
    check_img_size, check_imshow, non_max_suppression,
    scale_coords, xyxy2xywh, strip_optimizer, set_logging, increment_path
)
from partinet.DynamicDet.utils.plots import plot_one_box
from partinet.DynamicDet.utils.torch_utils import select_device, time_synchronized, intersect_dicts

# --- Logging setup ---
logger = logging.getLogger('partinet_detect')
logger.setLevel(logging.INFO)
formatter = logging.Formatter('%(asctime)s | %(levelname)s | %(message)s', datefmt='%Y-%m-%d %H:%M:%S')

sh = logging.StreamHandler(sys.stdout)
sh.setFormatter(formatter)
sh.setLevel(logging.INFO)
logger.addHandler(sh)

fh = logging.FileHandler('partinet_detect.log')
fh.setFormatter(formatter)
fh.setLevel(logging.INFO)
logger.addHandler(fh)

sys.stdout.reconfigure(line_buffering=True)
sys.stderr.reconfigure(line_buffering=True)

# --- Detection function ---
def detect(opt, save_img=False):
    source, cfg, weight, view_img, save_txt, nc, imgsz = (
        opt.source,
        os.path.join(partinet.DynamicDet.__path__[0], "cfg", f"dy-{opt.backbone_detector}-step2.yaml"),
        opt.weight,
        opt.view_img,
        opt.save_txt,
        opt.num_classes,
        opt.img_size,
    )
    save_img = not opt.nosave and not source.endswith('.txt')
    webcam = source.isnumeric() or source.endswith('.txt') or source.lower().startswith(('rtsp://', 'rtmp://', 'http://', 'https://'))

    save_dir = Path(increment_path(Path(opt.project) / opt.name, exist_ok=opt.exist_ok))
    (save_dir / 'labels' if save_txt else save_dir).mkdir(parents=True, exist_ok=True)

    set_logging()
    devices = [f'cuda:{i}' for i in range(torch.cuda.device_count())] if torch.cuda.is_available() else ['cpu']
    logger.info(f"Using devices: {devices}")

    # Load model onto each device
    models = {}
    for dev in devices:
        model_instance = Model(cfg, ch=3, nc=nc)
        state_dict = torch.load(weight, map_location='cpu')['model']
        state_dict = intersect_dicts(state_dict, model_instance.state_dict())
        model_instance.load_state_dict(state_dict, strict=False)
        model_instance.to(dev)
        model_instance.eval()
        for m in model_instance.modules():
            if type(m) in [nn.Hardswish, nn.LeakyReLU, nn.ReLU, nn.ReLU6, nn.SiLU]:
                m.inplace = True
            elif type(m) is nn.Upsample:
                m.recompute_scale_factor = None
        if dev != 'cpu':
            model_instance.half()
        models[dev] = model_instance
        logger.info(f"Model loaded on {dev}")

    stride = int(max(models[devices[0]].stride))
    if webcam:
        view_img = check_imshow()
        cudnn.benchmark = True
        dataset = LoadStreams(source, img_size=imgsz, stride=stride)
    else:
        dataset = LoadImages(source, img_size=imgsz, stride=stride)

    names = models[devices[0]].module.names if hasattr(models[devices[0]], 'module') else models[devices[0]].names
    colors = [[random.randint(0, 255) for _ in range(3)] for _ in names]

    vid_path, vid_writer = None, None

    img_queue = queue.Queue(maxsize=32)  # Only buffer 32 images

    def producer():
        for i, (path, img, im0s, vid_cap) in enumerate(dataset):
            frame = getattr(dataset, 'frame', 0)
            img_queue.put((path, img, im0s, frame, vid_cap))
        for _ in range(len(devices)):  # signal end of queue
            img_queue.put(None)

    Thread(target=producer, daemon=True).start()

    # Dataset stats
    total_particles = 0
    num_images = 0
    dataset_start_time = time.time()

    def worker(device):
        nonlocal total_particles, num_images, vid_path, vid_writer
        model = models[device]
        while True:
            item = img_queue.get()
            if item is None:
                break
            path, img, im0s, frame, vid_cap = item

            img_start = time.time()
            logger.info(f"[{device}] Processing {Path(path).name}")

            # Prepare image
            img_tensor = torch.from_numpy(img).to(device)
            img_tensor = img_tensor.half() if device != 'cpu' else img_tensor.float()
            img_tensor /= 255.0
            if img_tensor.ndimension() == 3:
                img_tensor = img_tensor.unsqueeze(0)

            # Forward pass
            with torch.no_grad():
                pred = model(img_tensor, augment=opt.augment)[0]

            # NMS
            pred = non_max_suppression(pred, opt.conf_thres, opt.iou_thres, classes=opt.classes, agnostic=opt.agnostic_nms)

            # Post-processing per image
            num_particles_img = 0
            for i_det, det in enumerate(pred):
                if webcam:
                    p, s, im0, f_idx = path[i_det], f'{i_det}: ', im0s[i_det].copy(), frame
                else:
                    p, s, im0, f_idx = path, '', im0s, frame

                p = Path(p)
                save_path = str(save_dir / p.name)
                txt_path = str(save_dir / 'labels' / p.stem) + ('' if dataset.mode == 'image' else f'_{f_idx}')
                gn = torch.tensor(im0.shape)[[1, 0, 1, 0]]

                if len(det):
                    det[:, :4] = scale_coords(img_tensor.shape[2:], det[:, :4], im0.shape).round()
                    num_particles_img = len(det)
                    total_particles += num_particles_img
                    for c in det[:, -1].unique():
                        n = (det[:, -1] == c).sum()
                        s += f"{n} {names[int(c)]}{'s' * (n > 1)}, "

                    for *xyxy, conf, cls in reversed(det):
                        if save_txt:
                            xywh = (xyxy2xywh(torch.tensor(xyxy).view(1, 4)) / gn).view(-1).tolist()
                            line = (cls, *xywh, conf) if opt.save_conf else (cls, *xywh)
                            with open(txt_path + '.txt', 'a') as f_txt:
                                f_txt.write(('%g ' * len(line)).rstrip() % line + '\n')

                        if save_img or view_img:
                            label = f'{names[int(cls)]} {conf:.2f}'
                            plot_one_box(xyxy, im0, label=label, color=colors[int(cls)], line_thickness=1)

                if view_img:
                    cv2.imshow(str(p), im0)
                    cv2.waitKey(1)

                if save_img:
                    if dataset.mode == 'image':
                        cv2.imwrite(save_path, im0)
                    else:
                        if vid_path != save_path:
                            vid_path = save_path
                            if isinstance(vid_writer, cv2.VideoWriter):
                                vid_writer.release()
                            if webcam and dataset.mode != 'image':
                                fps = vid_cap.get(cv2.CAP_PROP_FPS)
                                w = int(vid_cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                                h = int(vid_cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                            else:
                                fps, w, h = 30, im0.shape[1], im0.shape[0]
                                save_path += '.mp4'
                            vid_writer = cv2.VideoWriter(save_path, cv2.VideoWriter_fourcc(*'mp4v'), fps, (w, h))
                        vid_writer.write(im0)

            img_time_ms = (time.time() - img_start) * 1000
            num_images += 1
            logger.info(f"[{device}] {p.name}: {num_particles_img} particles, {img_time_ms:.1f} ms")

    # Start one thread per GPU
    threads = [Thread(target=worker, args=(dev,)) for dev in devices]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    total_time = time.time() - dataset_start_time
    avg_micrographs_per_sec = num_images / total_time if total_time > 0 else 0

    logger.info(f"Dataset complete: {num_images} micrographs processed in {total_time:.2f} s")
    logger.info(f"Average processing rate: {avg_micrographs_per_sec:.2f} micrographs/s")
    logger.info(f"Total particles detected: {total_particles}")