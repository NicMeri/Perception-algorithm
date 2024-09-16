"""
Threshold module

This module provides functions for thresholding images.
"""

import logging
import os
import cv2

# Configuring logging
logging.basicConfig(level=logging.DEBUG)

# pylint: disable=no-member


def apply_threshold_and_save_image(
    img_path="photo/road.jpg", output_path=None, is_preview_enable=False
):
    """
    Apply_threshold_and_save_image function.

    Args:
        img_path (str): the path to the input image.
        output_path (str, optional): the path to save the output image.
        If not provided, the same directory as the input is used.

    Returns:
        None
    """

    # Input parameter validation
    if isinstance(img_path, str):
        dirname = os.path.dirname(__file__)
        img_path = os.path.join(dirname, img_path)
    if not os.path.isfile(img_path):
        logging.error("Invalid input image path.")
        return None

    # Output parameter validation
    if output_path is None:
        input_dir = os.path.dirname(img_path)
        filename, extension = os.path.splitext(os.path.basename(img_path))
        output_filename = filename + "_thresholded" + extension
        output_path = os.path.join(input_dir, output_filename)
    elif not isinstance(output_path, str):
        logging.warning("Invalid output image path.")
        return None

    original_frame = cv2.imread(img_path)

    frame_grayscale_applied = cv2.cvtColor(original_frame, cv2.COLOR_BGR2GRAY)

    # cv2.imshow("Gray", frame_grayscale_applied)

    # Simple thresholding
    _, frame_threshold_applied = cv2.threshold(
        src=frame_grayscale_applied, thresh=205, maxval=255, type=cv2.THRESH_BINARY
    )

    if is_preview_enable:
        cv2.imshow("Road", original_frame)
        cv2.imshow("Gray", frame_grayscale_applied)
        cv2.imshow("Simple thresholding", frame_threshold_applied)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    # Output image saving
    cv2.imwrite(output_path, frame_threshold_applied)
    logging.info("Output image saved successfully at %s.", output_path)
    return output_path


def main():
    """
    Main function.

    This function is responsible for executing the apply_threshold_and_save_image function.
    """

    apply_threshold_and_save_image()


if __name__ == "__main__":
    main()
