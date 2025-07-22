import argparse
import time
import logging
import os
import glob
from pathlib import Path
import threading
from queue import Queue
from concurrent.futures import ThreadPoolExecutor

import cv2
import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn
from torch.nn.parallel import DataParallel
import numpy as np
from numpy import random

import partinet.DynamicDet
from partinet.DynamicDet.models.yolo import Model
from partinet.DynamicDet.utils.datasets import LoadStreams, LoadImages
from partinet.DynamicDet.utils.general import check_img_size, check_imshow, non_max_suppression, \
    scale_coords, xyxy2xywh, strip_optimizer, set_logging, increment_path
from partinet.DynamicDet.utils.plots import plot_one_box
from partinet.DynamicDet.utils.torch_utils import select_device, time_synchronized, intersect_dicts


logger = logging.getLogger(__name__)

class BatchedLoadImages:
    """Batched version of LoadImages for multi-GPU inference"""
    
    def __init__(self, path, img_size=640, stride=32, batch_size=64):
        p = str(Path(path).absolute())
        if '*' in p:
            files = sorted(glob.glob(p, recursive=True))
        elif os.path.isdir(p):
            files = sorted(glob.glob(os.path.join(p, '*.*')))
        elif os.path.isfile(p):
            files = [p]
        else:
            raise Exception(f'ERROR: {p} does not exist')
            
        # Filter for supported image formats
        img_formats = ['bmp', 'jpg', 'jpeg', 'png', 'tif', 'tiff', 'dng', 'webp', 'mpo']
        images = [x for x in files if x.split('.')[-1].lower() in img_formats]
        
        self.img_size = img_size
        self.stride = stride
        self.batch_size = batch_size
        self.files = images
        self.nf = len(images)
        self.count = 0
        
        assert self.nf > 0, f'No images found in {p}'
        
    def __iter__(self):
        self.count = 0
        return self
    
    def __next__(self):
        if self.count >= self.nf:
            raise StopIteration
            
        batch_paths = []
        batch_imgs = []
        batch_img0s = []
        
        # Collect batch_size images
        for _ in range(min(self.batch_size, self.nf - self.count)):
            if self.count >= self.nf:
                break
                
            path = self.files[self.count]
            img0 = cv2.imread(path)
            assert img0 is not None, f'Image Not Found {path}'
            
            # Padded resize
            from partinet.DynamicDet.utils.datasets import letterbox
            img = letterbox(img0, self.img_size, stride=self.stride)[0]
            img = img[:, :, ::-1].transpose(2, 0, 1)  # BGR to RGB, to 3xHxW
            img = np.ascontiguousarray(img)
            
            batch_paths.append(path)
            batch_imgs.append(img)
            batch_img0s.append(img0)
            self.count += 1
        
        if not batch_imgs:
            raise StopIteration
            
        # Stack images into batch tensor
        batch_tensor = np.stack(batch_imgs, axis=0)
        
        return batch_paths, batch_tensor, batch_img0s
    
    def __len__(self):
        return (self.nf + self.batch_size - 1) // self.batch_size  # ceiling division

def select_multi_device(device='', batch_size=None):
    """Enhanced device selection for multi-GPU setup"""
    s = f'DynamicDet 🚀 torch {torch.__version__} '
    cpu = device.lower() == 'cpu'
    
    if cpu:
        os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
        return torch.device('cpu'), [torch.device('cpu')]
    elif device:
        os.environ['CUDA_VISIBLE_DEVICES'] = device
        assert torch.cuda.is_available(), f'CUDA unavailable, invalid device {device} requested'
    
    cuda = not cpu and torch.cuda.is_available()
    devices = []
    
    if cuda:
        if device:
            # Specific GPUs requested
            gpu_ids = [int(x.strip()) for x in device.split(',')]
        else:
            # Use all available GPUs
            gpu_ids = list(range(torch.cuda.device_count()))
        
        n = len(gpu_ids)
        if n > 1 and batch_size:
            assert batch_size % n == 0, f'batch-size {batch_size} not multiple of GPU count {n}'
        
        space = ' ' * len(s)
        for i, gpu_id in enumerate(gpu_ids):
            p = torch.cuda.get_device_properties(gpu_id)
            s += f"{'' if i == 0 else space}CUDA:{gpu_id} ({p.name}, {p.total_memory / 1024 ** 2:.0f}MB)\n"
            devices.append(torch.device(f'cuda:{gpu_id}'))
        
        primary_device = devices[0]
    else:
        s += 'CPU\n'
        primary_device = torch.device('cpu')
        devices = [primary_device]
    
    logger.info(s)
    return primary_device, devices

class MultiGPUModel:
    """Wrapper for multi-GPU model inference"""
    
    def __init__(self, model, devices, cfg, nc):
        self.devices = devices
        self.models = []
        
        if len(devices) == 1:
            # Single GPU or CPU
            self.models.append(model.to(devices[0]))
            self.use_parallel = False
        else:
            # Multi-GPU setup - create separate model instances for each GPU
            original_state_dict = model.state_dict()
            for device in devices:
                # Create new model instance with same config
                model_copy = Model(cfg, ch=3, nc=nc)
                model_copy.load_state_dict(original_state_dict, strict=False)
                model_copy = model_copy.to(device).eval()
                
                # Apply same modifications as original model
                for p in model_copy.parameters():
                    p.requires_grad = False
                model_copy.float().fuse()
                
                # Compatibility updates
                for m in model_copy.modules():
                    if type(m) in [nn.Hardswish, nn.LeakyReLU, nn.ReLU, nn.ReLU6, nn.SiLU]:
                        m.inplace = True
                    elif type(m) is nn.Upsample:
                        m.recompute_scale_factor = None
                
                self.models.append(model_copy)
            self.use_parallel = True
    
    def __call__(self, batch_tensor, augment=False):
        if not self.use_parallel:
            # Single GPU inference - process each image individually to avoid tensor boolean issue
            results = []
            for i in range(batch_tensor.shape[0]):
                single_img = batch_tensor[i:i+1]  # Keep batch dimension
                with torch.no_grad():
                    result = self.models[0](single_img, augment=augment)[0]
                results.append(result)
            return torch.cat(results, dim=0)
        
        # Multi-GPU inference - distribute individual images across GPUs
        batch_size = batch_tensor.shape[0]
        num_gpus = len(self.devices)
        
        # Distribute images across GPUs
        tasks = []
        for i in range(batch_size):
            gpu_idx = i % num_gpus
            device = self.devices[gpu_idx]
            model = self.models[gpu_idx]
            single_img = batch_tensor[i:i+1].to(device)  # Keep batch dimension, move to specific GPU
            tasks.append((model, single_img, device, i, augment))
        
        # Process in parallel
        results = [None] * batch_size
        with ThreadPoolExecutor(max_workers=num_gpus) as executor:
            futures = []
            for task in tasks:
                future = executor.submit(self._inference_worker, *task)
                futures.append(future)
            
            for idx, future in enumerate(futures):
                result, original_idx = future.result()
                results[original_idx] = result.cpu()
        
        # Concatenate results in original order
        return torch.cat(results, dim=0)
    
    def _inference_worker(self, model, single_img, device, original_idx, augment):
        with torch.no_grad():
            result = model(single_img, augment=augment)[0]
            return result, original_idx

def detect(opt, save_img=False):
    source, cfg = opt.source, os.path.join(partinet.DynamicDet.__path__[0], "cfg", f"dy-{opt.backbone_detector}-step2.yaml")
    weight, view_img, save_txt, nc, imgsz = opt.weight, opt.view_img, opt.save_txt, opt.num_classes, opt.img_size
    batch_size = getattr(opt, 'batch_size', 8)
    
    save_img = not opt.nosave and not source.endswith('.txt')
    
    # Directories
    save_dir = Path(increment_path(Path(opt.project) / opt.name, exist_ok=opt.exist_ok))
    (save_dir / 'labels' if save_txt else save_dir).mkdir(parents=True, exist_ok=True)
    
    # Initialize
    set_logging()
    device, devices = select_multi_device(opt.device, batch_size)
    half = device.type != 'cpu'
    
    # Load model
    model = Model(cfg, ch=3, nc=nc)
    state_dict = torch.load(weight, map_location='cpu')['model']
    state_dict = intersect_dicts(state_dict, model.state_dict())
    model.load_state_dict(state_dict, strict=False)
    
    logger.info('Transferred %g/%g items from %s' % (len(state_dict), len(model.state_dict()), weight))
    
    for p in model.parameters():
        p.requires_grad = False
    model.float().fuse().eval()
    
    # Compatibility updates
    for m in model.modules():
        if type(m) in [nn.Hardswish, nn.LeakyReLU, nn.ReLU, nn.ReLU6, nn.SiLU]:
            m.inplace = True
        elif type(m) is nn.Upsample:
            m.recompute_scale_factor = None
    
    stride = int(model.stride.max())
    imgsz = check_img_size(imgsz, s=stride)
    
    if hasattr(model, 'dy_thres'):
        model.dy_thres = opt.dy_thres
        logger.info('Set dynamic threshold to %f' % opt.dy_thres)
    
    # Create multi-GPU model wrapper
    multi_gpu_model = MultiGPUModel(model, devices, cfg, nc)
    
    if half:
        for model_instance in multi_gpu_model.models:
            model_instance.half()
    
    # Set Dataloader with batching
    dataset = BatchedLoadImages(source, img_size=imgsz, stride=stride, batch_size=batch_size)
    
    # Get names and colors
    names = model.module.names if hasattr(model, 'module') else model.names
    colors = [[random.randint(0, 255) for _ in range(3)] for _ in names]
    
    # Warmup
    dummy_batch = torch.zeros(batch_size, 3, imgsz, imgsz).to(device)
    if half:
        dummy_batch = dummy_batch.half()
    
    logger.info('Running warmup...')
    for _ in range(3):
        _ = multi_gpu_model(dummy_batch, augment=opt.augment)
    
    t0 = time.time()
    total_processed = 0
    
    for batch_paths, batch_tensor, batch_img0s in dataset:
        batch_tensor = torch.from_numpy(batch_tensor).to(device)
        batch_tensor = batch_tensor.half() if half else batch_tensor.float()
        batch_tensor /= 255.0
        
        current_batch_size = batch_tensor.shape[0]
        
        # Inference
        t1 = time_synchronized()
        pred = multi_gpu_model(batch_tensor, augment=opt.augment)
        t2 = time_synchronized()
        
        # Apply NMS
        pred = non_max_suppression(pred, opt.conf_thres, opt.iou_thres, 
                                 classes=opt.classes, agnostic=opt.agnostic_nms)
        t3 = time_synchronized()
        
        # Process detections for each image in batch
        for i, (path, det, im0) in enumerate(zip(batch_paths, pred, batch_img0s)):
            p = Path(path)
            save_path = str(save_dir / p.name)
            txt_path = str(save_dir / 'labels' / p.stem)
            gn = torch.tensor(im0.shape)[[1, 0, 1, 0]]  # normalization gain whwh
            
            s = f'image {total_processed + i + 1}: '
            
            if len(det):
                # Rescale boxes from img_size to im0 size
                det[:, :4] = scale_coords(batch_tensor.shape[2:], det[:, :4], im0.shape).round()
                
                # Print results
                for c in det[:, -1].unique():
                    n = (det[:, -1] == c).sum()
                    s += f"{n} {names[int(c)]}{'s' * (n > 1)}, "
                
                # Write results
                for *xyxy, conf, cls in reversed(det):
                    if save_txt:
                        xywh = (xyxy2xywh(torch.tensor(xyxy).view(1, 4)) / gn).view(-1).tolist()
                        line = (cls, *xywh, conf) if opt.save_conf else (cls, *xywh)
                        with open(txt_path + '.txt', 'a') as f:
                            f.write(('%g ' * len(line)).rstrip() % line + '\n')
                    
                    if save_img or view_img:
                        label = f'{names[int(cls)]} {conf:.2f}'
                        plot_one_box(xyxy, im0, label=label, color=colors[int(cls)], line_thickness=1)
            
            print(f'{s}Done.')
            
            # Save results
            if save_img:
                cv2.imwrite(save_path, im0)
                print(f"Image saved: {save_path}")
            
            if view_img:
                cv2.imshow(str(p), im0)
                cv2.waitKey(1)
        
        total_processed += current_batch_size
        
        # Print timing for batch
        inf_time = 1E3 * (t2 - t1) / current_batch_size  # ms per image
        nms_time = 1E3 * (t3 - t2) / current_batch_size  # ms per image
        print(f'Batch of {current_batch_size}: ({inf_time:.1f}ms/img) Inference, ({nms_time:.1f}ms/img) NMS')
    
    total_time = time.time() - t0
    avg_time_per_image = total_time / total_processed if total_processed > 0 else 0
    
    if save_txt or save_img:
        s = f"\n{len(list(save_dir.glob('labels/*.txt')))} labels saved to {save_dir / 'labels'}" if save_txt else ''
        print(f"Results saved to {save_dir}{s}")
    
    print(f'Done. ({total_time:.3f}s total, {avg_time_per_image:.3f}s/image average, {total_processed} images)')
    print(f'Speed: {total_processed/total_time:.1f} images/second')

# Usage example:
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--source', type=str, default='data/images', help='source')
    parser.add_argument('--weight', type=str, default='weights/best.pt', help='model.pt path')
    parser.add_argument('--img-size', type=int, default=640, help='inference size (pixels)')
    parser.add_argument('--conf-thres', type=float, default=0.25, help='object confidence threshold')
    parser.add_argument('--iou-thres', type=float, default=0.45, help='IOU threshold for NMS')
    parser.add_argument('--device', default='', help='cuda device, i.e. 0 or 0,1,2,3 or cpu')
    parser.add_argument('--batch-size', type=int, default=8, help='batch size for inference')
    parser.add_argument('--view-img', action='store_true', help='display results')
    parser.add_argument('--save-txt', action='store_true', help='save results to *.txt')
    parser.add_argument('--save-conf', action='store_true', help='save confidences in --save-txt labels')
    parser.add_argument('--nosave', action='store_true', help='do not save images/videos')
    parser.add_argument('--classes', nargs='+', type=int, help='filter by class: --class 0, or --class 0 2 3')
    parser.add_argument('--agnostic-nms', action='store_true', help='class-agnostic NMS')
    parser.add_argument('--augment', action='store_true', help='augmented inference')
    parser.add_argument('--project', default='runs/detect', help='save results to project/name')
    parser.add_argument('--name', default='exp', help='save results to project/name')
    parser.add_argument('--exist-ok', action='store_true', help='existing project/name ok, do not increment')
    
    # Add your custom arguments
    parser.add_argument('--backbone-detector', default='yolov5s', help='backbone detector')
    parser.add_argument('--num-classes', type=int, default=80, help='number of classes')
    parser.add_argument('--dy-thres', type=float, default=0.5, help='dynamic threshold')
    
    opt = parser.parse_args()
    
    detect_multi_gpu(opt)