import os
import shutil
import argparse
import cv2
import numpy as np
from typing import Dict, List
from sklearn.model_selection import train_test_split as tts


def parse_star_file(star_path: str) -> List[Dict[str, str]]:
    """
    Parse a STAR file and extract particle coordinates
    Returns a list of dictionaries with micrograph names and coordinates
    """
    particles = []
    in_data_section = False
    headers = []
    header_indices = {}
    
    with open(star_path, 'r') as f:
        for line in f:
            line = line.strip()
            
            # Skip empty lines and comments
            if not line or line.startswith('#'):
                continue
            
            # Check for data section
            if line.startswith('data_'):
                in_data_section = True
                continue
            
            # Check for loop section
            if line.startswith('loop_'):
                continue
            
            # Parse headers
            if line.startswith('_rln'):
                parts = line.split()
                header_name = parts[0]
                if len(parts) > 1:
                    header_idx = int(parts[1].replace('#', '')) - 1  # Convert to 0-indexed
                    headers.append(header_name)
                    header_indices[header_name] = header_idx
                continue
            
            # Parse data rows
            if in_data_section and headers:
                parts = line.split()
                if len(parts) >= len(headers):
                    particle = {}
                    for header in headers:
                        idx = header_indices[header]
                        particle[header] = parts[idx]
                    particles.append(particle)
    
    return particles


def starfile_to_yolo(x_coord: int, y_coord: int, diameter: int, 
                     image_width: int, image_height: int, class_id: int = 0) -> Dict[str, float]:
    """
    Convert STAR file coordinates to YOLO format
    """
    # Calculate YOLO normalized coordinates
    x_center = x_coord / image_width
    y_center = y_coord / image_height
    width = diameter / image_width
    height = diameter / image_height
    
    return {
        'class': class_id,
        'x_center': x_center,
        'y_center': y_center,
        'width': width,
        'height': height
    }


def group_particles_by_micrograph(particles: List[Dict[str, str]]) -> Dict[str, List[Dict[str, str]]]:
    """
    Group particles by their micrograph name
    """
    grouped = {}
    for particle in particles:
        micrograph = particle.get('_rlnMicrographName', '')
        if micrograph not in grouped:
            grouped[micrograph] = []
        grouped[micrograph].append(particle)
    return grouped


def convert_star_to_yolo(star_path: str, images_path: str, output_labels_path: str, class_id: int = 0) -> List[str]:
    """
    Convert STAR file to YOLO format labels
    
    Args:
        star_path: Path to input STAR file
        images_path: Path to directory containing micrograph images
        output_labels_path: Path to output directory for YOLO label files
        class_id: Class ID to assign to all particles (default: 0)
    
    Returns:
        List of processed filenames (without extensions)
    """
    # Create output directory if it doesn't exist
    os.makedirs(output_labels_path, exist_ok=True)
    
    # Parse STAR file
    print(f"Parsing STAR file: {star_path}")
    particles = parse_star_file(star_path)
    print(f"Found {len(particles)} particles")
    
    # Group particles by micrograph
    grouped_particles = group_particles_by_micrograph(particles)
    print(f"Found {len(grouped_particles)} unique micrographs")
    
    processed_files = []
    
    # Process each micrograph
    for micrograph_name, micrograph_particles in grouped_particles.items():
        # Remove .mrc extension and get base filename
        base_filename = os.path.splitext(os.path.basename(micrograph_name))[0]
        
        # Find corresponding image file (try common extensions)
        image_file = None
        image_ext = None
        for ext in ['.png', '.jpg', '.jpeg', '.tif', '.tiff', '.mrc']:
            potential_path = os.path.join(images_path, base_filename + ext)
            if os.path.exists(potential_path):
                image_file = potential_path
                image_ext = ext
                break
        
        if image_file is None:
            print(f"Warning: Image file not found for {base_filename}. Skipping.")
            continue
        
        # Read image to get dimensions
        image = cv2.imread(image_file)
        if image is None:
            print(f"Warning: Could not read image {image_file}. Skipping.")
            continue
        
        img_height, img_width = image.shape[:2]
        
        # Convert particles to YOLO format
        yolo_labels = []
        for particle in micrograph_particles:
            x_coord = int(particle.get('_rlnCoordinateX', 0))
            y_coord = int(particle.get('_rlnCoordinateY', 0))
            diameter = int(particle.get('_rlnDiameter', 0))
            
            yolo_coords = starfile_to_yolo(x_coord, y_coord, diameter, 
                                          img_width, img_height, class_id)
            yolo_labels.append(yolo_coords)
        
        # Write YOLO label file
        output_file = os.path.join(output_labels_path, f"{base_filename}.txt")
        with open(output_file, 'w') as f:
            for label in yolo_labels:
                f.write(f"{label['class']} {label['x_center']:.6f} {label['y_center']:.6f} "
                       f"{label['width']:.6f} {label['height']:.6f}\n")
        
        processed_files.append((base_filename, image_ext))
        print(f"Processed {base_filename}: {len(yolo_labels)} particles")
    
    print(f"\nConversion complete! Labels saved to {output_labels_path}")
    return processed_files


def split_train_val(labels_path: str, images_path: str, output_dir: str, test_size: float = 0.25):
    """
    Splits a dataset of images and labels into training and validation sets and organizes them into 
    a specified output directory. Generates corresponding .txt files for train/val data and a 
    cryo_training.yaml file for use in model training.

    Args:
        labels_path: Path to the directory containing label files
        images_path: Path to the directory containing image files
        output_dir: Path to the output directory where split data will be saved
        test_size: Proportion of dataset to use for validation (default: 0.25)
    """
    # Create output directories if they do not exist
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    os.makedirs(os.path.join(output_dir, "images", "train"), exist_ok=True)
    os.makedirs(os.path.join(output_dir, "images", "val"), exist_ok=True)
    os.makedirs(os.path.join(output_dir, "labels", "train"), exist_ok=True)
    os.makedirs(os.path.join(output_dir, "labels", "val"), exist_ok=True)

    # List all label files
    files = [f for f in os.listdir(labels_path) if f.endswith('.txt')]

    # Split data into training and validation indices
    train_idx, val_idx = tts(np.arange(0, len(files), 1), test_size=test_size, shuffle=True, random_state=42)

    # Iterate through files and copy them into train/val directories
    for idx, file in enumerate(files):
        file_name = os.path.splitext(file)[0]
        
        # Find the image file (try common extensions)
        image_file = None
        for ext in ['.png', '.jpg', '.jpeg', '.tif', '.tiff', '.mrc']:
            potential_image = os.path.join(images_path, file_name + ext)
            if os.path.exists(potential_image):
                image_file = file_name + ext
                break
        
        if image_file is None:
            print(f"Warning: Image file not found for {file_name}. Skipping.")
            continue
        
        if idx in train_idx:
            # Copy training images and labels
            shutil.copy(
                os.path.join(images_path, image_file),
                os.path.join(output_dir, "images", "train", image_file)
            )
            shutil.copy(
                os.path.join(labels_path, file),
                os.path.join(output_dir, "labels", "train", file)
            )
        elif idx in val_idx:
            # Copy validation images and labels
            shutil.copy(
                os.path.join(images_path, image_file),
                os.path.join(output_dir, "images", "val", image_file)
            )
            shutil.copy(
                os.path.join(labels_path, file),
                os.path.join(output_dir, "labels", "val", file)
            )

    # Create val.txt file listing validation image paths
    with open(os.path.join(output_dir, "val.txt"), "w") as f:
        for file in sorted(os.listdir(os.path.join(output_dir, "images", "val"))):
            f.write(str(os.path.join(output_dir, "images", "val", file)) + "\n")

    # Create train.txt file listing training image paths
    with open(os.path.join(output_dir, "train.txt"), "w") as f:
        for file in sorted(os.listdir(os.path.join(output_dir, "images", "train"))):
            f.write(str(os.path.join(output_dir, "images", "train", file)) + "\n")

    # Create cryo_training.yaml file with dataset configuration
    to_write = f"""train: {os.path.join(output_dir, "train.txt")}
val: {os.path.join(output_dir, "val.txt")}

# number of classes
nc: 1

# class names
names: [ 'particle' ]"""

    with open(os.path.join(output_dir, "cryo_training.yaml"), "w") as f:
        f.write(to_write)
    
    train_count = len(os.listdir(os.path.join(output_dir, "images", "train")))
    val_count = len(os.listdir(os.path.join(output_dir, "images", "val")))
    print(f"\nDataset split complete!")
    print(f"Training samples: {train_count}")
    print(f"Validation samples: {val_count}")
    print(f"Configuration saved to: {os.path.join(output_dir, 'cryo_training.yaml')}")


def main(star_path: str, images_path: str, output_dir: str, class_id: int = 0, 
         test_size: float = 0.25, split_only: bool = False):
    """
    Main function to convert STAR file to YOLO format and split into train/val sets
    
    Args:
        star_path: Path to input STAR file (or labels directory if split_only=True)
        images_path: Path to directory containing micrograph images
        output_dir: Path to output directory
        class_id: Class ID to assign to all particles (default: 0)
        test_size: Proportion of dataset to use for validation (default: 0.25)
        split_only: If True, skip conversion and only split existing labels
    """
    if split_only:
        # Skip conversion, just split existing labels
        print("Splitting existing labels into train/val sets...")
        split_train_val(star_path, images_path, output_dir, test_size)
    else:
        # Create temporary directory for converted labels
        temp_labels_dir = os.path.join(output_dir, "temp_labels")
        
        # Convert STAR to YOLO
        print("=" * 60)
        print("Step 1: Converting STAR file to YOLO format")
        print("=" * 60)
        convert_star_to_yolo(star_path, images_path, temp_labels_dir, class_id)
        
        # Split into train/val
        print("\n" + "=" * 60)
        print("Step 2: Splitting data into train/val sets")
        print("=" * 60)
        split_train_val(temp_labels_dir, images_path, output_dir, test_size)
        
        # Clean up temporary labels directory
        shutil.rmtree(temp_labels_dir)
        print(f"\nAll done! Training data ready in {output_dir}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Convert STAR file to YOLO format and split into train/val sets"
    )
    parser.add_argument(
        "--star", 
        required=True, 
        help="Path to input STAR file (or labels directory if using --split-only)"
    )
    parser.add_argument(
        "--images", 
        required=True, 
        help="Path to directory containing micrograph images"
    )
    parser.add_argument(
        "--output", 
        required=True, 
        help="Path to output directory for organized train/val data"
    )
    parser.add_argument(
        "--class-id", 
        type=int, 
        default=0, 
        help="Class ID to assign to all particles (default: 0)"
    )
    parser.add_argument(
        "--test-size",
        type=float,
        default=0.25,
        help="Proportion of dataset to use for validation (default: 0.25)"
    )
    parser.add_argument(
        "--split-only",
        action="store_true",
        help="Skip STAR conversion and only split existing labels (use --star to specify labels directory)"
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    main(args.star, args.images, args.output, args.class_id, args.test_size, args.split_only)