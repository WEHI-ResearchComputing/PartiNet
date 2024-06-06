import click
import sys

__version__ = "0.0.1"

def print_params():
    """
    Prints all the parameter-value pairs after cli processing with Click
    """
    all_params_str = ", ".join(
        [f"{k}: {v}" for k, v in click.get_current_context().params.items()]
    )
    click.echo(all_params_str)

@click.group()
@click.version_option(__version__)
def main():
    pass

@main.command()
def preprocess():
    click.echo("This will preprocess the micrographs.")

@main.command()
def train1():
    click.echo("This will perform DynamicDet training step 1.")

@main.command()
def train2():
    click.echo("This will perform DynamicDet training step 2.")

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
    print_params()

    import partinet.DynamicDet.detect
    with partinet.DynamicDet.detect.torch.no_grad():
        partinet.DynamicDet.detect.detect(cfg, weight, num_classes, source, img_size, conf_thres, iou_thres, device, view_img, save_txt, save_conf, nosave, classes, agnostic_nms, augment, project, name, exist_ok, dy_thres)

if __name__ == "__main__":
    main()