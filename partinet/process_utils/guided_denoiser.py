# Adapted from CryoSegNet
# https://github.com/jianlin-cheng/CryoSegNet

import numpy as np
import mrcfile
import cv2
from numpy.fft import fft2, ifft2
from scipy.signal import gaussian


def transform(image: np.ndarray) -> np.ndarray:
    """
    Normalize and scale an image to 8-bit range (0-255).

    Args:
        image (np.ndarray): Input image array.

    Returns:
        np.ndarray: Normalized and scaled 8-bit image.
    """
    i_min = image.min()
    i_max = image.max()
    if i_max == i_min:
        # avoid division by zero; return a zero array when input is constant
        return np.zeros_like(image, dtype=np.uint8)
    image = ((image - i_min) / (i_max - i_min)) * 255
    return image.astype(np.uint8)


def standard_scaler(image: np.ndarray) -> np.ndarray:
    """
    Apply Gaussian blur and standardize the image to have zero mean and unit variance.

    The input is cast to ``float32`` before any OpenCV operations to avoid the
    ``CV_16F`` kernel-type error that occurs when processing 16‑bit micrographs
    (see issue #41). After blurring and normalization we transform the result to
    eight‑bit for downstream filters.

    Args:
        image (np.ndarray): Input image array.

    Returns:
        np.ndarray: Scaled and transformed image.
    """
    # convert to a supported floating point type for OpenCV kernels
    image = image.astype(np.float32)

    kernel_size = 9
    image = cv2.GaussianBlur(image, (kernel_size, kernel_size), 0)
    mu = np.mean(image)
    sigma = np.std(image)
    image = (image - mu) / sigma
    return transform(image).astype(np.uint8)


def contrast_enhancement(image: np.ndarray) -> np.ndarray:
    """
    Enhance the contrast of the image using Non-Local Means denoising.

    Args:
        image (np.ndarray): Input image array.

    Returns:
        np.ndarray: Contrast-enhanced image.
    """
    return cv2.fastNlMeansDenoising(image, None, h=10, templateWindowSize=7, searchWindowSize=21)


def gaussian_kernel(kernel_size: int = 3) -> np.ndarray:
    """
    Generate a 2D Gaussian kernel.

    Args:
        kernel_size (int, optional): Size of the kernel. Defaults to 3.

    Returns:
        np.ndarray: Gaussian kernel.
    """
    h = gaussian(kernel_size, kernel_size / 3).reshape(kernel_size, 1)
    h = np.dot(h, h.transpose())
    h /= np.sum(h)
    return h


def wiener_filter(img: np.ndarray, kernel: np.ndarray, K: float) -> np.ndarray:
    """
    Apply Wiener filtering to the image.

    Args:
        img (np.ndarray): Input image array.
        kernel (np.ndarray): Convolution kernel.
        K (float): Noise-to-signal power ratio.

    Returns:
        np.ndarray: Filtered image.
    """
    kernel /= np.sum(kernel)
    dummy = fft2(img)
    kernel = fft2(kernel, s=img.shape)
    kernel = np.conj(kernel) / (np.abs(kernel) ** 2 + K)
    dummy *= kernel
    return np.abs(ifft2(dummy))


def clahe(image: np.ndarray) -> np.ndarray:
    """
    Apply Contrast Limited Adaptive Histogram Equalization (CLAHE) to the image.

    Args:
        image (np.ndarray): Input image array.

    Returns:
        np.ndarray: CLAHE-processed image.
    """
    clahe_instance = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(16, 16))
    return clahe_instance.apply(transform(image))


def guided_filter(input_image: np.ndarray, guidance_image: np.ndarray, radius: int = 20, epsilon: float = 0.1) -> np.ndarray:
    """
    Apply guided filtering to refine the image using a guidance image.

    Args:
        input_image (np.ndarray): Input image array.
        guidance_image (np.ndarray): Guidance image array.
        radius (int, optional): Radius of the filter. Defaults to 20.
        epsilon (float, optional): Regularization parameter. Defaults to 0.1.

    Returns:
        np.ndarray: Filtered image.
    """
    input_image = input_image.astype(np.float32) / 255.0
    guidance_image = guidance_image.astype(np.float32) / 255.0

    mean_guidance = cv2.boxFilter(guidance_image, -1, (radius, radius))
    mean_input = cv2.boxFilter(input_image, -1, (radius, radius))
    mean_guidance_input = cv2.boxFilter(guidance_image * input_image, -1, (radius, radius))
    covariance_guidance_input = mean_guidance_input - mean_guidance * mean_input

    mean_guidance_sq = cv2.boxFilter(guidance_image * guidance_image, -1, (radius, radius))
    variance_guidance = mean_guidance_sq - mean_guidance * mean_guidance

    a = covariance_guidance_input / (variance_guidance + epsilon)
    b = mean_input - a * mean_guidance
    mean_a = cv2.boxFilter(a, -1, (radius, radius))
    mean_b = cv2.boxFilter(b, -1, (radius, radius))

    output_image = mean_a * guidance_image + mean_b
    return transform(output_image)


def denoise(image_path: str) -> np.ndarray:
    """
    Apply a sequence of denoising operations to an input image.

    Args:
        image_path (str): Path to the input .mrc image file.

    Returns:
        np.ndarray: Fully denoised image.
    """
    kernel = gaussian_kernel(kernel_size=9)
    image = mrcfile.read(image_path)

    # some MRCs are stored as 16‑bit integers; ensure we work in float32 so that
    # subsequent OpenCV calls (GaussianBlur, etc.) don't raise the ktype error
    # described in https://github.com/WEHI-ResearchComputing/PartiNet/issues/41
    image = image.astype(np.float32)

    image = image.T
    image = np.rot90(image)
    normalized_image = standard_scaler(np.array(image))
    contrast_enhanced_image = contrast_enhancement(normalized_image)
    weiner_filtered_image = wiener_filter(contrast_enhanced_image, kernel, K=30)
    clahe_image = clahe(weiner_filtered_image)
    guided_filter_image = guided_filter(clahe_image, weiner_filtered_image)
    return guided_filter_image
