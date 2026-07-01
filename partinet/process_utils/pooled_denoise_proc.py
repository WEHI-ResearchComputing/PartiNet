import os
import cv2
from pathlib import Path
from partinet.process_utils.guided_denoiser import denoise
import logging
import multiprocessing
import argparse
import gc
from concurrent.futures import ProcessPoolExecutor
from typing import List, Tuple, Optional
import mrcfile
import sys

logger = logging.getLogger("partinet_denoise")


def _configure_denoise_logging(log_path: str) -> None:
    """Attach file and stream handlers once per run; remove stale handlers first."""
    fmt = logging.Formatter("%(asctime)s - %(message)s")
    logger.setLevel(logging.INFO)
    logger.propagate = False
    for h in logger.handlers[:]:
        logger.removeHandler(h)
        h.close()
    fh = logging.FileHandler(log_path)
    fh.setFormatter(fmt)
    logger.addHandler(fh)
    sh = logging.StreamHandler(sys.stdout)
    sh.setFormatter(fmt)
    logger.addHandler(sh)


# Function to perform CLAHE-based denoising

def clahe_denoise(args: Tuple[str, str, str]) -> None:
    """
    Applies the guided denoising algorithm to an input image and saves the denoised result.

    Args:
        args (Tuple[str, str]):
            - src_path: Path to the source micrograph file.
            - dest_path: Path to save the denoised image.
            - img_format: output format of denoised images

    Raises:
        Exception: Logs any exceptions that occur during processing.
    """
    try:
        src_path, dest_path, img_format = args
        # Perform denoising
        denoised = denoise(src_path)
        if img_format == "mrc":
            mrcfile.write(dest_path,data=denoised)
        else:
            cv2.imwrite(dest_path, denoised)
        logger.info(f"Processed image {src_path} to dest. {dest_path}")
        del denoised
        gc.collect()
    except Exception as e:
        logger.error(f"Failed to process {src_path}: {str(e)}")

# Function to process all files in a directory

def process_directory(
    micrographs_dir: str,
    clahe_denoised_dir: str,
    max_workers: int,
    img_format: str,
    log_path: Optional[str] = None,
) -> None:
    """
    Processes all `.mrc` files in the given directory using parallel workers for denoising.

    Args:
        micrographs_dir (str): Path to the directory containing raw micrograph files.
        clahe_denoised_dir (str): Path to the directory where denoised images will be saved.
        max_workers (int): Number of worker processes to use for parallel processing.
        img_format (str): Output format of denoised images

    Notes:
        - Only `.mrc` files are processed.
        - Existing denoised images are skipped.
    """
    os.makedirs(clahe_denoised_dir, exist_ok=True)
    logger.info(f"Directory ready: {clahe_denoised_dir}")

    tasks: List[Tuple[str, str, str]] = []
    # Iterate through files in the directory
    for file_name in os.listdir(micrographs_dir):
        if file_name.endswith(".mrc"):
            src_path = os.path.join(micrographs_dir, file_name)
            dest_path = os.path.join(clahe_denoised_dir, file_name.replace(".mrc", "."+img_format))
            if os.path.exists(dest_path):
                logger.info(f"{dest_path} already exists!")
            else:
                tasks.append((src_path, dest_path, img_format))

    if not tasks:
        logger.info("No new micrographs to denoise.")
        return

    pool_kwargs = {}
    if log_path:
        pool_kwargs = {"initializer": _configure_denoise_logging, "initargs": (log_path,)}
    with ProcessPoolExecutor(max_workers=max_workers, **pool_kwargs) as executor:
        futures = [executor.submit(clahe_denoise, task) for task in tasks]
        for future in futures:
            future.result()
    gc.collect()

# Main function

def main(source_dir: str, project_dir: str, ncpu: int, img_format: str) -> None:
    """
    Main function to set up logging, determine available CPUs, and start the denoising process.

    Args:
        source_dir (str): Path to the directory containing raw micrographs.
        project_dir (str): Path to the project directory where denoised images will be saved.
        ncpu (int): Number of CPUs to use for parallel processing. Defaults to half the available CPUs.
        img_format (str): Output format of denoised images

    Notes:
        - Logging is configured to write messages to a log file and the console.
        - Ensures at least one CPU is used for processing.
    """
    # Prepare output directory and log file
    denoise_dir = os.path.join(project_dir, "denoised")
    log_path = os.path.join(project_dir, "partinet_denoise.log")
    _configure_denoise_logging(log_path)

    # Determine number of available CPUs
    max_available_cpus = multiprocessing.cpu_count()
    max_workers = max(1, max_available_cpus // 2)

    # Adjust number of CPUs to use
    if ncpu is not None:
        ncpu = min(ncpu, max_workers)
    else:
        ncpu = max_workers

    logger.info(f"Using {ncpu} workers out of {max_available_cpus} available CPUs.")
    logger.info(f"Processing raw micrographs in {source_dir}")
    logger.info(f"Saving denoised micrographs in {denoise_dir}")

    process_directory(source_dir, denoise_dir, ncpu, img_format, log_path=log_path)

# Command-line argument parsing

def parse_args() -> argparse.Namespace:
    """
    Parses command-line arguments for the script.

    Returns:
        argparse.Namespace: Parsed arguments with the following attributes:
            - raw: Path to raw micrographs.
            - project: Path to the project directory.
            - ncpu: Number of CPUs to use.
            - img_format: output format of denoised images
    """
    parser = argparse.ArgumentParser(description="Denoise micrographs with guided CryoSegNet-style filter")
    parser.add_argument("--raw", required=True, help="Path to raw micrographs")
    parser.add_argument("--project", required=True, help="Denoised micrographs saved in project/denoised")
    parser.add_argument('--ncpu', type=int, default=None, help='Number of CPUs to use')
    parser.add_argument('--img_format', type=str, choices=["png","jpg","mrc"], default="png", help='Output format of denoised images')
    return parser.parse_args()

# Entry point for the script

if __name__ == "__main__":
    args = parse_args()
    main(args.raw, args.project, args.ncpu, args.img_format)
