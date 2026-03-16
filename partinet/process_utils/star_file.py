import math
import os
import pandas as pd
import csv
import cv2
import argparse
from typing import List, Dict, Tuple, Optional
from multiprocessing import Pool, cpu_count

def yolo_to_starfile(yolo_coords: Dict[str, float], image_width: int, image_height: int, diameters: List[int]) -> Tuple[int, int, int]:
    x_center = math.ceil(yolo_coords["x_centre"] * image_width)
    y_center = math.ceil(yolo_coords["y_centre"] * image_height)
    width = yolo_coords["width"] * image_width
    height = yolo_coords["height"] * image_height
    diameter = math.ceil(max(width, height))
    diameters.append(diameter)
    return x_center, y_center, diameter

def generate_output(labels: pd.DataFrame, filename: str, img_width: int, img_height: int) -> List[Tuple[str, int, int, int]]:
    diameters: List[int] = []
    output_rows = []
    for _, row in labels.iterrows():
        x_centre, y_centre, diameter = yolo_to_starfile(row, img_width, img_height, diameters)
        star_filename = f"{filename}.mrc"
        output_rows.append((star_filename, x_centre, y_centre, diameter))
    return output_rows

def process_image(args_tuple) -> List[Tuple[str, int, int, int]]:
    image_file, labels_path, images_path, conf_thresh = args_tuple
    filename = os.path.splitext(image_file)[0]
    label_file_path = os.path.join(labels_path, f"{filename}.txt")

    if not os.path.exists(label_file_path):
        # skip missing labels
        return []

    image = cv2.imread(os.path.join(images_path, image_file))
    if image is None:
        return []
    img_width, img_height = image.shape[1], image.shape[0]

    custom_headers = ["class", "x_centre", "y_centre", "width", "height", "conf"]
    labels = pd.read_csv(label_file_path, header=None, names=custom_headers, sep=r"\s+")
    labels = labels[labels["conf"] > float(conf_thresh)]
    if labels.empty:
        return []

    return generate_output(labels, filename, img_width, img_height)

def write_cryosparc_star(all_rows: List[Tuple[str, int, int, int]], star_out_path: str) -> None:
    with open(star_out_path, "w", newline="") as star_file:
        star_writer = csv.writer(star_file, delimiter=" ", lineterminator="\n")
        star_writer.writerow([])
        star_writer.writerow(["data_"])
        star_writer.writerow([])
        star_writer.writerow(["loop_"])
        star_writer.writerow(["_rlnMicrographName", "#1"])
        star_writer.writerow(["_rlnCoordinateX", "#2"])
        star_writer.writerow(["_rlnCoordinateY", "#3"])
        star_writer.writerow(["_rlnDiameter", "#4"])
        star_writer.writerows(all_rows)

def write_relion_coordinate_star(path: str, coords: List[Tuple[float, float]]) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", newline="\n") as f:
        f.write("data_\n\n")
        f.write("loop_\n")
        f.write("_rlnCoordinateX #1\n")
        f.write("_rlnCoordinateY #2\n")
        for x, y in coords:
            f.write(f"{x:.6f} {y:.6f}\n")

def write_relion_pick_star(rows: List[Tuple[str, str]], out_path: str) -> None:
    with open(out_path, "w", newline="\n") as f:
        f.write("data_coordinate_files\n\n")
        f.write("loop_\n")
        f.write("_rlnMicrographName #1\n")
        f.write("_rlnMicrographCoordinates #2\n")
        for mic, coord in rows:
            f.write(f"{mic} {coord}\n")

def relion_write(all_rows: List[Tuple[str, int, int, int]], pick_out: str, coords_dir: str, mrc_prefix: str, extension: str = ".star") -> None:
    grouped: Dict[str, List[Tuple[float, float]]] = {}
    for mic, x, y, d in all_rows:
        grouped.setdefault(mic, []).append((float(x), float(y)))

    partinet_root = os.path.basename(os.path.dirname(pick_out))  # expects pick_out .../partinet/pick.star
    mapping: List[Tuple[str, str]] = []
    for mic, coords in grouped.items():
        root = os.path.splitext(os.path.basename(mic))[0]
        coord_star_name = f"{root}{extension}"
        coord_star_path = os.path.join(coords_dir, coord_star_name)
        write_relion_coordinate_star(coord_star_path, coords)

        mic_name = os.path.join(mrc_prefix, os.path.basename(mic)) if mrc_prefix else mic
        coord_entry = os.path.join(partinet_root, "movies", coord_star_name)
        mapping.append((mic_name, coord_entry))

    write_relion_pick_star(mapping, pick_out)

def main(labels_path: str, images_path: str, star_out_path: str, conf_thresh: float, relion: bool = False, relion_project_dir: Optional[str] = None, relion_pick: Optional[str] = None, relion_coord_dir: Optional[str] = None, mrc_prefix: str = "") -> None:
    image_files = [f for f in os.listdir(images_path) if os.path.splitext(f)[1].lower() in [".mrc", ".tif", ".tiff", ".png", ".jpg", ".jpeg"]]
    args_list = [(img_file, labels_path, images_path, conf_thresh) for img_file in image_files]

    with Pool(cpu_count()) as pool:
        results = pool.map(process_image, args_list)

    all_rows = [row for result in results for row in result]
    if not all_rows:
        print("No particle rows produced.")
        return

    write_cryosparc_star(all_rows, star_out_path)
    print(f"Wrote cryosparc-compatible star to: {star_out_path}")

    if relion:
        if relion_project_dir is None:
            raise ValueError("For --relion, --relion-project-dir must be provided.")
        relion_partinet = os.path.join(relion_project_dir, "partinet")
        relion_pickstar = os.path.join(relion_partinet, "pick.star")
        relion_coorddir = os.path.join(relion_partinet, "movies")
        os.makedirs(relion_coorddir, exist_ok=True)
        relion_write(all_rows, relion_pickstar, relion_coorddir, mrc_prefix)
        print(f"Wrote relion pick.star: {relion_pickstar}")
        print(f"Wrote relion per-micrograph stars under: {relion_coorddir}")

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate STAR files from YOLO labels (CryoSPARC and optional RELION)")
    parser.add_argument("--labels", required=True, help="Path to the labels directory")
    parser.add_argument("--images", required=True, help="Path to the images directory")
    parser.add_argument("--output", required=True, help="Path to the output STAR file (CryoSPARC style)")
    parser.add_argument("--conf", required=True, type=float, help="Minimum confidence threshold for predictions")
    parser.add_argument("--relion", action="store_true", help="Also generate RELION pick.star + per-micrograph coordinate star files")
    parser.add_argument("--relion-project-dir", default=None, help="RELION project root; outputs go to <project>/partinet/pick.star and <project>/partinet/movies/*.star")
    parser.add_argument("--mrc-prefix", default="", help="Prefix for micrograph paths in RELION pick.star (e.g. MotionCorr/job003/movies)")
    return parser.parse_args()

if __name__ == "__main__":
    args = parse_args()
    main(
        args.labels,
        args.images,
        args.output,
        args.conf,
        relion=args.relion,
        relion_project_dir=args.relion_project_dir,
        mrc_prefix=args.mrc_prefix,
    )