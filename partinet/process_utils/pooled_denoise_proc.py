import os
import cv2
from pathlib import Path
from partinet.process_utils.guided_denoiser import denoise
import logging
import multiprocessing
import argparse
import gc
from concurrent.futures import ProcessPoolExecutor

# MAX_WORKERS = max(1, multiprocessing.cpu_count() // 2)

def clahe_denoise(args):
    try:
        src_path, dest_path = args
        denoised = denoise(src_path)
        cv2.imwrite(dest_path, denoised)
        logging.info(f"Processed image {src_path} to dest. {dest_path}")
        del denoised
        gc.collect()
    except Exception as e:
        logging.error(f"Failed to process {src_path}: {str(e)}")

def process_directory(micrographs_dir, clahe_denoised_dir,MAX_WORKERS):
    os.makedirs(clahe_denoised_dir, exist_ok=True)
    logging.info(f"Directory ready: {clahe_denoised_dir}")

    tasks = []
    for file_name in os.listdir(micrographs_dir):
        if file_name.endswith(".mrc"):
            src_path = os.path.join(micrographs_dir, file_name)
            dest_path = os.path.join(clahe_denoised_dir, file_name.replace(".mrc", ".png"))
            if os.path.exists(dest_path):
                logging.info(f"{dest_path} already exists!")
            else:
                tasks.append((src_path, dest_path))

    with ProcessPoolExecutor(max_workers=MAX_WORKERS) as executor:
        futures = [executor.submit(clahe_denoise, task) for task in tasks]
    for future in futures:
            future.result()
    gc.collect()

def main(source_dir, project_dir, ncpu):
    logging.basicConfig(filename="partinet_denoise.log", level=logging.INFO, format='%(asctime)s - %(message)s')
    logging.getLogger().addHandler(logging.StreamHandler())

    denoise_dir = os.path.join(project_dir,"denoised")
    logger_name = project_dir + "/partinet_denoise.log"
    logging.basicConfig(filename=logger_name, level=logging.INFO, format='%(asctime)s - %(message)s')

    max_available_cpus = multiprocessing.cpu_count()
    max_workers = max(1, max_available_cpus // 2)

    if ncpu is not None:
        ncpu = min(ncpu, max_workers)
    else:
        ncpu = max_workers

    logging.info(f"Using {ncpu} workers out of {max_available_cpus} available CPUs.")
    
    # num_cpus = max(multiprocessing.cpu_count(),ncpu)
    # logging.info(f"Number of available CPUs: {num_cpus}")

    logging.info(f"Processing raw micrographs in {source_dir}")
    logging.info(f"Saving denoised micrographs in {denoise_dir}")
    
    # process_directory(source_dir, denoise_dir,ncpu)

def parse_args():
    parser = argparse.ArgumentParser(description="Denoise micrographs with guided CryoSegNet-style filter")
    parser.add_argument("--raw", required=True, help="Path to raw micrographs")
    parser.add_argument("--project", required=True, help="Denoised micrographs saved in project/denoised")
    parser.add_argument('--ncpu', type=int, default=None, help='Number of CPUs to use')

    
    return parser.parse_args()

if __name__ == "__main__":
    args = parse_args()
    main(args.raw, args.project, args.ncpu)