import click
import sys

__version__ = "0.0.1"

def print_params(params: dict) -> None:
    """
    Prints all the parameter-value pairs after cli processing with Click
    """
    all_params_str = ", ".join(
        [f"{k}: {v}" for k, v in params.items()]
    )
    click.echo(all_params_str)

@click.group()
@click.version_option(__version__)
def main():
    pass

@main.command()
def preprocess():
    click.echo("This will preprocess the micrographs.")
    raise NotImplementedError("Not implemented yet!")

@main.group()
@click.option('--cfg', type=str, help='model.yaml path', required=True)
@click.option('--weight', type=str, help='initial weights path', required=True)
@click.option('--data', type=str, default='data/coco.yaml', help='data.yaml path', show_default=True)
@click.option('--hyp', type=str, default='hyp/hyp.scratch.p5.yaml', help='hyperparameters path', show_default=True)
@click.option('--epochs', type=int, default=300, show_default=True)
@click.option('--batch-size', type=int, default=16, help='total batch size for all GPUs', show_default=True)
@click.option('--img-size', nargs=2, type=int, default=[640, 640], help='[train, test] image sizes', show_default=True)
@click.option('--rect', is_flag=True, help='rectangular training')
@click.option('--resume', is_flag=True, help='resume most recent training')
@click.option('--resume-ckpt', type=str, default='', help='checkpoint to resume from')
@click.option('--nosave', is_flag=True, help='only save final checkpoint')
@click.option('--notest', is_flag=True, help='only test final epoch')
@click.option('--noautoanchor', is_flag=True, help='disable autoanchor check')
@click.option('--bucket', type=str, default='', help='gsutil bucket')
@click.option('--cache-images', is_flag=True, help='cache images for faster training')
@click.option('--image-weights', is_flag=True, help='use weighted image selection for training')
@click.option('--device', default='', help='cuda device, i.e. 0 or 0,1,2,3 or cpu', show_default=True)
@click.option('--multi-scale', is_flag=True, help='vary img-size +/- 50%%')
@click.option('--single-cls', is_flag=True, help='train multi-class data as single-class')
@click.option('--adam', is_flag=True, help='use torch.optim.Adam() optimizer')
@click.option('--sync-bn', is_flag=True, help='use SyncBatchNorm, only available in DDP mode')
@click.option('--local_rank', type=int, default=-1, help='DDP parameter, do not modify', show_default=True)
@click.option('--workers', type=int, default=8, help='maximum number of dataloader workers', show_default=True)
@click.option('--project', default='runs/train', help='save to project/name', show_default=True)
@click.option('--entity', default=None, help='W&B entity', show_default=True)
@click.option('--name', default='exp', help='save to project/name', show_default=True)
@click.option('--exist-ok', is_flag=True, help='existing project/name ok, do not increment')
@click.option('--quad', is_flag=True, help='quad dataloader')
@click.option('--label-smoothing', type=float, default=0.0, help='Label smoothing epsilon', show_default=True)
@click.option('--upload_dataset', is_flag=True, help='Upload dataset as W&B artifact table')
@click.option('--bbox_interval', type=int, default=-1, help='Set bounding-box image logging interval for W&B', show_default=True)
@click.option('--save_period', type=int, default=-1, help='Log model after every "save_period" epoch', show_default=True)
@click.option('--artifact_alias', type=str, default="latest", help='version of dataset artifact to be used', show_default=True)
@click.option('--freeze', multiple=True, type=int, default=[0], help='Freeze layers: backbone of yolov7=50, first3=0 1 2', show_default=True)
@click.option('--v5-metric', is_flag=True, help='assume maximum recall as 1.0 in AP calculation', show_default=True)
@click.pass_context
def train(ctx, cfg, weight, data, hyp, epochs, batch_size, img_size, rect, resume, resume_ckpt, nosave, notest, noautoanchor, bucket, cache_images, image_weights, device, multi_scale, single_cls, adam, sync_bn, local_rank, workers, project, entity, name, exist_ok, quad, label_smoothing, upload_dataset, bbox_interval, save_period, artifact_alias, freeze, v5_metric):
    # ensure that ctx.obj exists as dict
    ctx.ensure_object(dict)
    # add train args to ctx.obj, which will be passed to step1 and step2
    ctx.obj.update(click.get_current_context().params)

@train.command()
@click.option('--single-backbone', is_flag=True, help='train single backbone model')
@click.option('--linear-lr', is_flag=True, help='linear LR')
@click.pass_context
def step1(ctx, single_backbone, linear_lr):
    # grab params from train group
    params = ctx.obj
    # update params with step1 params
    params.update(ctx.params)

    # dump params to terminal
    click.echo("Performing DynamicDet training step 1 with config:\n    ", nl=False)
    print_params(params)

    # convert params to argparse namespace since that's how DynamicDet is programmed
    import argparse
    opt = argparse.Namespace(**params)

    # run train_step1.py
    import partinet.DynamicDet.train_step1
    partinet.DynamicDet.train_step1.main(opt)

@train.command()
@click.pass_context
def step2(ctx):
    # grab params from train group (train step2 doesn't have special params)
    params = ctx.obj

    click.echo("Performing DynamicDet training step 2 with config:\n    ", nl=False)
    print_params(params)

    # convert params to argparse namespace since that's how DynamicDet is programmed
    import argparse
    opt = argparse.Namespace(**params)

    # run train_step1.py
    import partinet.DynamicDet.train_step2
    partinet.DynamicDet.train_step2.main(opt)

@main.command()
@click.option('--cfg', type=str, help='model.yaml path', required=True)
@click.option('--weight', type=str, help='model.pt path(s)', required=True)
@click.option('--source', type=str, default='inference/images', help='source', show_default=True)  # file/folder, 0 for webcam
@click.option('--num-classes', type=int, default=80, help='number of classes', show_default=True)
@click.option('--img-size', type=int, default=640, help='inference size (pixels)', show_default=True)
@click.option('--conf-thres', type=float, default=0.25, help='object confidence threshold', show_default=True)
@click.option('--iou-thres', type=float, default=0.45, help='IOU threshold for NMS', show_default=True)
@click.option('--device', default='', help='cuda device, i.e. 0 or 0,1,2,3 or cpu', show_default=True)
@click.option('--view-img', is_flag=True, help='display results')
@click.option('--save-txt', is_flag=True, help='save results to *.txt')
@click.option('--save-conf', is_flag=True, help='save confidences in --save-txt labels')
@click.option('--nosave', is_flag=True, help='do not save images/videos')
@click.option('--classes', multiple=True, type=int, default=[], help='filter by class: --classes 0, or --classes 0 --classes 2 --classes 3', show_default=True)
@click.option('--agnostic-nms', is_flag=True, help='class-agnostic NMS')
@click.option('--augment', is_flag=True, help='augmented inference')
@click.option('--project', default='runs/detect', help='save results to project/name', show_default=True)
@click.option('--name', default='exp', help='save results to project/name', show_default=True)
@click.option('--exist-ok', is_flag=True, help='existing project/name ok, do not increment')
@click.option('--dy-thres', type=float, default=0.5, help='dynamic thres', show_default=True)
def detect(cfg, weight, num_classes, source, img_size, conf_thres, iou_thres, device, view_img, save_txt, save_conf, nosave, classes, agnostic_nms, augment, project, name, exist_ok, dy_thres):

    click.echo("Performing DynamicDet detection with config:\n    ", nl=False)
    print_params(click.get_context().params)

    import partinet.DynamicDet.detect
    with partinet.DynamicDet.detect.torch.no_grad():
        partinet.DynamicDet.detect.detect(cfg, weight, num_classes, source, img_size, conf_thres, iou_thres, device, view_img, save_txt, save_conf, nosave, classes, agnostic_nms, augment, project, name, exist_ok, dy_thres)

if __name__ == "__main__":
    main()