from PIL import ImageTk, Image, ImageEnhance, ImageFilter, ImageOps
import cv2
import numpy as np


def Saturation(image):
    """
    Adjust the saturation of an image.

    Parameters:
    - image: PIL Image object
    - factor: float, saturation factor (0.0 to 1.0)

    Returns:
    - PIL Image object with adjusted saturation
    """
    enhancer = ImageEnhance.Color(image)
    return enhancer.enhance(1.5)

def HistogramEqualization(image):
    """
    Apply histogram equalization to an image.

    Parameters:
    - image: PIL Image object

    Returns:
    - PIL Image object with histogram equalization applied
    """
    image = image.convert("L")  # Convert to grayscale
    equalized_image = ImageOps.equalize(image)
    return equalized_image


def GammaCorrection(image, gamma):
    """
    Apply gamma correction to an image.

    Parameters:
    - image: PIL Image object
    - gamma: float, gamma value for correction

    Returns:
    - PIL Image object with gamma correction applied
    """
    inv_gamma = 1.0 / gamma
    table = [((i / 255.0) ** inv_gamma) * 255 for i in range(256)]
    table = [int(value) for value in table]

    if image.mode in ['L', 'P']:  # Grayscale or palette-based image
        return image.point(table)
    elif image.mode == 'RGB':  # RGB image
        r, g, b = image.split()
        r = r.point(table)
        g = g.point(table)
        b = b.point(table)
        return Image.merge('RGB', (r, g, b))
    else:
        raise ValueError(f"Unsupported image mode: {image.mode}")
    

def BilateralFiltering(image, diameter, sigma_color, sigma_space):
    """
    Apply bilateral filtering to an image.

    Parameters:
    - image: PIL Image object
    - diameter: int, diameter of each pixel neighborhood
    - sigma_color: float, filter sigma in the color space
    - sigma_space: float, filter sigma in the coordinate space

    Returns:
    - PIL Image object with bilateral filtering applied
    """

    # Convert PIL Image to OpenCV format
    image_cv = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)

    # Apply bilateral filter
    filtered_image_cv = cv2.bilateralFilter(image_cv, diameter, sigma_color, sigma_space)

    # Convert back to PIL Image
    filtered_image = Image.fromarray(cv2.cvtColor(filtered_image_cv, cv2.COLOR_BGR2RGB))
    return filtered_image


def GaussianBlur(image, kernel_size, sigma):

    """
    Apply Gaussian blur to an image.

    Parameters:
    - image: PIL Image object
    - kernel_size: int, size of the Gaussian kernel (must be odd)
    - sigma: float, standard deviation of the Gaussian distribution

    Returns:
    - PIL Image object with Gaussian blur applied
    """

    # Convert PIL Image to OpenCV format
    image_cv = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)

    # Apply Gaussian blur
    blurred_image_cv = cv2.GaussianBlur(image_cv, (kernel_size, kernel_size), sigma)

    # Convert back to PIL Image
    blurred_image = Image.fromarray(cv2.cvtColor(blurred_image_cv, cv2.COLOR_BGR2RGB))
    return blurred_image


def MedianBlur(image, kernel_size):
    """
    Apply median blur to an image.

    Parameters:
    - image: PIL Image object
    - kernel_size: int, size of the kernel (must be odd)

    Returns:
    - PIL Image object with median blur applied
    """

    # Convert PIL Image to OpenCV format
    image_cv = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)

    # Apply median blur
    blurred_image_cv = cv2.medianBlur(image_cv, kernel_size)

    # Convert back to PIL Image
    blurred_image = Image.fromarray(cv2.cvtColor(blurred_image_cv, cv2.COLOR_BGR2RGB))
    return blurred_image


def AverageBlur(image, kernel_size):
    """
    Apply average blur to an image.

    Parameters:
    - image: PIL Image object
    - kernel_size: int, size of the kernel (must be odd)

    Returns:
    - PIL Image object with average blur applied
    """

    # Convert PIL Image to OpenCV format
    image_cv = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)

    # Apply average blur
    blurred_image_cv = cv2.blur(image_cv, (kernel_size, kernel_size))

    # Convert back to PIL Image
    blurred_image = Image.fromarray(cv2.cvtColor(blurred_image_cv, cv2.COLOR_BGR2RGB))
    return blurred_image


def SobelOperator(image, dx, dy, ksize):
    """
    Apply the Sobel operator to an image.

    Parameters:
    - image: PIL Image object
    - dx: int, order of the derivative x
    - dy: int, order of the derivative y
    - ksize: int, size of the extended Sobel kernel (must be 1, 3, 5, or 7)

    Returns:
    - PIL Image object with Sobel operator applied
    """

    # Convert PIL Image to OpenCV format
    image_cv = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)

    # Apply Sobel operator
    sobel_image_cv = cv2.Sobel(image_cv, cv2.CV_64F, dx, dy, ksize=ksize)

    # Convert back to uint8
    sobel_image_cv = cv2.convertScaleAbs(sobel_image_cv)

    # Convert back to PIL Image
    sobel_image = Image.fromarray(cv2.cvtColor(sobel_image_cv, cv2.COLOR_BGR2RGB))
    return sobel_image


def LaplacianOfGaussian(image, kernel_size, sigma):
    """
    Apply Laplacian of Gaussian (LoG) to an image.

    Parameters:
    - image: PIL Image object
    - kernel_size: int, size of the Gaussian kernel (must be odd)
    - sigma: float, standard deviation of the Gaussian distribution

    Returns:
    - PIL Image object with Laplacian of Gaussian applied
    """

    # Convert PIL Image to OpenCV format
    image_cv = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)

    # Apply Gaussian blur
    blurred_image_cv = cv2.GaussianBlur(image_cv, (kernel_size, kernel_size), sigma)

    # Apply Laplacian operator
    log_image_cv = cv2.Laplacian(blurred_image_cv, cv2.CV_64F)

    # Convert back to uint8
    log_image_cv = cv2.convertScaleAbs(log_image_cv)

    # Convert back to PIL Image
    log_image = Image.fromarray(cv2.cvtColor(log_image_cv, cv2.COLOR_BGR2RGB))
    return log_image


def PrewittOperator(image, dx, dy):
    """
    Apply the Prewitt operator to an image.

    Parameters:
    - image: PIL Image object
    - dx: int, order of the derivative x (1 for horizontal edges, 0 otherwise)
    - dy: int, order of the derivative y (1 for vertical edges, 0 otherwise)

    Returns:
    - PIL Image object with Prewitt operator applied
    """

    # Convert PIL Image to OpenCV format
    image_cv = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2GRAY)

    # Define Prewitt kernels
    kernel_x = np.array([[1, 0, -1], [1, 0, -1], [1, 0, -1]], dtype=np.float32)
    kernel_y = np.array([[1, 1, 1], [0, 0, 0], [-1, -1, -1]], dtype=np.float32)

    # Apply the appropriate kernel
    if dx == 1 and dy == 0:
        prewitt_image_cv = cv2.filter2D(image_cv, -1, kernel_x)
    elif dx == 0 and dy == 1:
        prewitt_image_cv = cv2.filter2D(image_cv, -1, kernel_y)
    else:
        raise ValueError("Invalid values for dx and dy. Use (1, 0) or (0, 1).")

    # Convert back to PIL Image
    prewitt_image = Image.fromarray(prewitt_image_cv)
    return prewitt_image


def RobertsCrossOperator(image):
    """
    Apply the Roberts Cross operator to an image.

    Parameters:
    - image: PIL Image object

    Returns:
    - PIL Image object with Roberts Cross operator applied
    """

    # Convert PIL Image to OpenCV format
    image_cv = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2GRAY)

    # Define Roberts Cross kernels
    kernel_x = np.array([[1, 0], [0, -1]], dtype=np.float32)
    kernel_y = np.array([[0, 1], [-1, 0]], dtype=np.float32)

    # Apply the kernels
    roberts_x = cv2.filter2D(image_cv, -1, kernel_x)
    roberts_y = cv2.filter2D(image_cv, -1, kernel_y)

    # Combine the results
    roberts_image_cv = cv2.addWeighted(roberts_x, 0.5, roberts_y, 0.5, 0)

    # Convert back to PIL Image
    roberts_image = Image.fromarray(roberts_image_cv)
    return roberts_image


def ScharrOperator(image, dx, dy):
    """
    Apply the Scharr operator to an image.

    Parameters:
    - image: PIL Image object
    - dx: int, order of the derivative x (1 for horizontal edges, 0 otherwise)
    - dy: int, order of the derivative y (1 for vertical edges, 0 otherwise)

    Returns:
    - PIL Image object with Scharr operator applied
    """

    # Convert PIL Image to OpenCV format
    image_cv = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2GRAY)

    # Apply Scharr operator
    scharr_x = cv2.Scharr(image_cv, cv2.CV_64F, 1, 0) if dx == 1 else None
    scharr_y = cv2.Scharr(image_cv, cv2.CV_64F, 0, 1) if dy == 1 else None

    if dx == 1 and dy == 0:
        scharr_image_cv = cv2.convertScaleAbs(scharr_x)
    elif dx == 0 and dy == 1:
        scharr_image_cv = cv2.convertScaleAbs(scharr_y)
    elif dx == 1 and dy == 1:
        scharr_x = cv2.convertScaleAbs(scharr_x)
        scharr_y = cv2.convertScaleAbs(scharr_y)
        scharr_image_cv = cv2.addWeighted(scharr_x, 0.5, scharr_y, 0.5, 0)
    else:
        raise ValueError("Invalid values for dx and dy. Use (1, 0), (0, 1), or (1, 1).")

    # Convert back to PIL Image
    scharr_image = Image.fromarray(scharr_image_cv)
    return scharr_image


def CannyEdgeDetection(image, threshold1, threshold2):
    """
    Apply Canny edge detection to an image.

    Parameters:
    - image: PIL Image object
    - threshold1: int, first threshold for the hysteresis procedure
    - threshold2: int, second threshold for the hysteresis procedure

    Returns:
    - PIL Image object with Canny edge detection applied
    """

    # Convert PIL Image to OpenCV format
    image_cv = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2GRAY)

    # Apply Canny edge detection
    edges_cv = cv2.Canny(image_cv, threshold1, threshold2)

    # Convert back to PIL Image
    edges_image = Image.fromarray(edges_cv)
    return edges_image


def BinaryThresholding(image, threshold):
    """
    Apply binary thresholding to an image.

    Parameters:
    - image: PIL Image object
    - threshold: int, threshold value (0-255)

    Returns:
    - PIL Image object with binary thresholding applied
    """

    # Convert PIL Image to OpenCV format
    image_cv = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2GRAY)

    # Apply binary thresholding
    _, binary_image_cv = cv2.threshold(image_cv, threshold, 255, cv2.THRESH_BINARY)

    # Convert back to PIL Image
    binary_image = Image.fromarray(binary_image_cv)
    return binary_image


def AdaptiveThresholding(image, max_value, adaptive_method, threshold_type, block_size, C):
    """
    Apply adaptive thresholding to an image.

    Parameters:
    - image: PIL Image object
    - max_value: int, maximum value to use with the THRESH_BINARY and THRESH_BINARY_INV thresholding types
    - adaptive_method: int, adaptive method (cv2.ADAPTIVE_THRESH_MEAN_C or cv2.ADAPTIVE_THRESH_GAUSSIAN_C)
    - threshold_type: int, thresholding type (cv2.THRESH_BINARY or cv2.THRESH_BINARY_INV)
    - block_size: int, size of a pixel neighborhood used to calculate the threshold value (must be odd)
    - C: int, constant subtracted from the mean or weighted mean

    Returns:
    - PIL Image object with adaptive thresholding applied
    """

    # Convert PIL Image to OpenCV format
    image_cv = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2GRAY)

    if adaptive_method == "cv2.ADAPTIVE_THRESH_MEAN_C":
        adaptive_method = cv2.ADAPTIVE_THRESH_MEAN_C
        
    elif adaptive_method == "cv2.ADAPTIVE_THRESH_GAUSSIAN_C":
        adaptive_method = cv2.ADAPTIVE_THRESH_GAUSSIAN_C

    if threshold_type == "cv2.THRESH_BINARY":
        threshold_type = cv2.THRESH_BINARY

    elif threshold_type == "cv2.THRESH_BINARY_INV":
        threshold_type = cv2.THRESH_BINARY_INV

    # Apply adaptive thresholding
    adaptive_thresh_cv = cv2.adaptiveThreshold(image_cv, max_value, adaptive_method, threshold_type, block_size, C)

    # Convert back to PIL Image
    adaptive_thresh_image = Image.fromarray(adaptive_thresh_cv)
    return adaptive_thresh_image


def GrayLevelSlicing(image, min_gray, max_gray, highlight_value):
    """
    Apply gray level slicing to an image.

    Parameters:
    - image: PIL Image object
    - min_gray: int, minimum gray level to highlight
    - max_gray: int, maximum gray level to highlight
    - highlight_value: int, value to assign to the highlighted range (default is 255)

    Returns:
    - PIL Image object with gray level slicing applied
    """

    # Convert PIL Image to grayscale
    image_gray = image.convert("L")

    # Convert to numpy array for processing
    image_array = np.array(image_gray)

    # Apply gray level slicing
    sliced_array = np.where((image_array >= min_gray) & (image_array <= max_gray), highlight_value, image_array)

    # Convert back to PIL Image
    sliced_image = Image.fromarray(sliced_array.astype(np.uint8))
    return sliced_image


def BitPlaneSlicing(image, bit_plane):
    """
    Perform bit plane slicing on an image.

    Parameters:
    - image: PIL Image object
    - bit_plane: int, bit plane to extract (0 for LSB, 7 for MSB)

    Returns:
    - PIL Image object with the specified bit plane extracted
    """

    # Convert PIL Image to grayscale
    image_gray = image.convert("L")

    # Convert to numpy array for processing
    image_array = np.array(image_gray)

    # Extract the specified bit plane
    bit_plane_image = (image_array >> bit_plane) & 1
    bit_plane_image = (bit_plane_image * 255).astype(np.uint8)

    # Convert back to PIL Image
    result_image = Image.fromarray(bit_plane_image)
    return result_image