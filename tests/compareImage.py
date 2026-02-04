def _image_pixel_differences(imageRef, imageToComp):
    """
    Calculates the bounding box of the non-zero regions in the image.
    :param base_image: target image to find
    :param compare_image:  set of images containing the target image
    :return: The bounding box is returned as a 4-tuple defining the
             left, upper, right, and lower pixel coordinate. If the image
             is completely empty, this method returns None.
    """
    # This module is used to load images
    from PIL import Image

    # This module contains a number of arithmetical image operations
    from PIL import ImageChops

    base_image = Image.open(imageRef)
    compare_image = Image.open(imageToComp)
    # Returns the absolute value of the pixel-by-pixel
    # difference between two images.
    diff = ImageChops.difference(base_image, compare_image)

    return diff.getbbox() is not None


def _HistCompare(imageRef, imageToComp):
    """
    Histogram Comparison: This method compares the histograms of the
    two images, looking at the distribution of pixel values.
    """
    import cv2

    image1 = cv2.imread(imageRef)
    image2 = cv2.imread(imageToComp)
    histogram1 = cv2.calcHist(
        [image1], [0, 1, 2], None, [256, 256, 256], [0, 256, 0, 256, 0, 256]
    )
    histogram2 = cv2.calcHist(
        [image2], [0, 1, 2], None, [256, 256, 256], [0, 256, 0, 256, 0, 256]
    )
    diff = cv2.compareHist(histogram1, histogram2, cv2.HISTCMP_CORREL)
    return diff


def _SSIM(imageRef, imageToComp):
    """
    Structural Similarity Index
    """
    import cv2

    from skimage.metrics import structural_similarity as ssim

    image1 = cv2.imread(imageRef)
    image2 = cv2.imread(imageToComp)

    # Resize the images to have the same dimensions
    height, width = image1.shape[:2]
    image2 = cv2.resize(image2, (width, height))

    # Convert the images to grayscale
    gray_image1 = cv2.cvtColor(image1, cv2.COLOR_BGR2GRAY)
    gray_image2 = cv2.cvtColor(image2, cv2.COLOR_BGR2GRAY)

    # Compute the SSIM between the two images
    ssim_score = ssim(
        gray_image1, gray_image2, data_range=gray_image2.max() - gray_image2.min()
    )

    return ssim_score


def _relativeCompareImage(imageRef, imageToComp):
    """
    Load the two images into your Python program. You can use the cv2.imread()
    function from the OpenCV library to load images.

    Check the size of the two images. If they are not of the same size, you may
    need to resize one or both images using the cv2.resize() function.

    Convert the images to grayscale. You can use the cv2.cvtColor() function to
    convert an image to grayscale.

    Compute the absolute difference between the two grayscale images using the
    cv2.absdiff() function.

    Compute a threshold value using the cv2.threshold() function. This value will
    be used to determine which pixels in the absolute difference image should be
    considered as "different" between the two images.

    Apply the threshold value to the absolute difference image using the cv2.threshold()
    function again.

    Compute the percentage of pixels that are different between the two images.
    This can be done by counting the number of pixels that are above the threshold
    value and dividing by the total number of pixels in the image.
    """
    import cv2

    # Load the two images
    img1 = cv2.imread(imageRef)
    img2 = cv2.imread(imageToComp)

    # Check the size of the images and resize if necessary
    if img1.shape != img2.shape:
        img2 = cv2.resize(img2, img1.shape[:2][::-1])

    # Convert images to grayscale
    gray1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
    gray2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)

    # Compute absolute difference between grayscale images
    diff = cv2.absdiff(gray1, gray2)

    # Set threshold value
    thresh = 10

    # Apply threshold to difference image
    _, thresh_img = cv2.threshold(diff, thresh, 255, cv2.THRESH_BINARY)

    # Compute percentage of different pixels
    diff_percent = (
        cv2.countNonZero(thresh_img) / (gray1.shape[0] * gray1.shape[1])
    ) * 100

    return diff_percent


def compareImage(imageRef, imageToComp):
    result = {}

    result["1to1Compare"] = _image_pixel_differences(imageRef, imageToComp)
    result["histCompare"] = _HistCompare(imageRef, imageToComp)
    result["SsimCompare"] = _SSIM(imageRef, imageToComp)
    result["relativeCompare"] = _relativeCompareImage(imageRef, imageToComp)
    return result
