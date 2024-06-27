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

def train_common_args(f):

    f = click.option('--v5-metric', is_flag=True, help='assume maximum recall as 1.0 in AP calculation', show_default=True)(f)
    f = click.option('--freeze', multiple=True, type=int, default=[0], help='Freeze layers: backbone of yolov7=50, first3=0 1 2', show_default=True)(f)
    f = click.option('--artifact_alias', type=str, default="latest", help='version of dataset artifact to be used', show_default=True)(f)
    f = click.option('--save_period', type=int, default=-1, help='Log model after every "save_period" epoch', show_default=True)(f)
    f = click.option('--bbox_interval', type=int, default=-1, help='Set bounding-box image logging interval for W&B', show_default=True)(f)
    f = click.option('--upload_dataset', is_flag=True, help='Upload dataset as W&B artifact table')(f)
    f = click.option('--label-smoothing', type=float, default=0.0, help='Label smoothing epsilon', show_default=True)(f)
    f = click.option('--quad', is_flag=True, help='quad dataloader')(f)
    f = click.option('--exist-ok', is_flag=True, help='existing project/name ok, do not increment')(f)
    f = click.option('--name', default='exp', help='save to project/name', show_default=True)(f)
    f = click.option('--entity', default=None, help='W&B entity', show_default=True)(f)
    f = click.option('--project', default='runs/train', help='save to project/name', show_default=True)(f)
    f = click.option('--workers', type=int, default=8, help='maximum number of dataloader workers', show_default=True)(f)
    f = click.option('--local_rank', type=int, default=-1, help='DDP parameter, do not modify', show_default=True)(f)
    f = click.option('--sync-bn', is_flag=True, help='use SyncBatchNorm, only available in DDP mode')(f)
    f = click.option('--adam', is_flag=True, help='use torch.optim.Adam() optimizer')(f)
    f = click.option('--single-cls', is_flag=True, help='train multi-class data as single-class')(f)
    f = click.option('--multi-scale', is_flag=True, help='vary img-size +/- 50%%')(f)
    f = click.option('--device', default='', help='cuda device, i.e. 0 or 0,1,2,3 or cpu', show_default=True)(f)
    f = click.option('--image-weights', is_flag=True, help='use weighted image selection for training')(f)
    f = click.option('--cache-images', is_flag=True, help='cache images for faster training')(f)
    f = click.option('--bucket', type=str, default='', help='gsutil bucket')(f)
    f = click.option('--noautoanchor', is_flag=True, help='disable autoanchor check')(f)
    f = click.option('--notest', is_flag=True, help='only test final epoch')(f)
    f = click.option('--nosave', is_flag=True, help='only save final checkpoint')(f)
    f = click.option('--resume-ckpt', type=str, default='', help='checkpoint to resume from')(f)
    f = click.option('--resume', is_flag=True, help='resume most recent training')(f)
    f = click.option('--rect', is_flag=True, help='rectangular training')(f)
    f = click.option('--img-size', nargs=2, type=int, default=[640, 640], help='[train, test] image sizes', show_default=True)(f)
    f = click.option('--batch-size', type=int, default=16, help='total batch size for all GPUs', show_default=True)(f)
    f = click.option('--epochs', type=int, default=300, show_default=True)(f)
    f = click.option('--hyp', type=str, default='hyp/hyp.scratch.p5.yaml', help='hyperparameters path', show_default=True)(f)
    f = click.option('--data', type=str, default='data/coco.yaml', help='data.yaml path', show_default=True)(f)
    f = click.option('--weight', type=str, help='initial weights path', required=True)(f)
    f = click.option('--cfg', type=str, help='model.yaml path', required=True)(f)

    return f

@click.group()
@click.version_option(__version__)
def main():
    pass

@main.command()
def preprocess():
    click.echo("This will preprocess the micrographs.")
    raise NotImplementedError("Not implemented yet!")

@main.group()
def train():
    pass

@train.command()
@train_common_args
@click.option('--single-backbone', is_flag=True, help='train single backbone model')
@click.option('--linear-lr', is_flag=True, help='linear LR')
def step1(**params):

    # dump params to terminal
    click.echo("Performing DynamicDet training step 1 with config:\n    ", nl=False)
    params["img_size"] = list(params["img_size"])
    print_params(params)

    # convert params to argparse namespace since that's how DynamicDet is programmed
    import argparse
    opt = argparse.Namespace(**params)

    # run train_step1.py
    import partinet.DynamicDet.train_step1
    partinet.DynamicDet.train_step1.main(opt)

@train.command()
@train_common_args
def step2(**params):

    click.echo("Performing DynamicDet training step 2 with config:\n    ", nl=False)
    params["img_size"] = list(params["img_size"])
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
def detect(**params):

    click.echo("Performing DynamicDet detection with config:\n    ", nl=False)
    print_params(params)

    import argparse
    opt = argparse.Namespace(**params)

    import partinet.DynamicDet.detect
    with partinet.DynamicDet.detect.torch.no_grad():
        partinet.DynamicDet.detect.detect(opt)

if __name__ == "__main__":
    main()