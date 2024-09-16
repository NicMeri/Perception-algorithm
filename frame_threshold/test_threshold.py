"""
Test module for thresholding function.
"""

import os
import pytest
import cv2
import numpy as np
from frame_threshold import apply_threshold_and_save_image

# Path control for the input image (path_checking) (control if is .jpg)
# Output path control if the script works correctly
# No output path if the script doesn't work correctly
# Test for the grayscale and test thresholding (work on the image)
# Output control

# pylint: disable=no-member


# Path control for the input image (path_checking)
@pytest.mark.parametrize(
    ("image_path", "expected_output_path"),
    [
        ("Dummy_path", None),
        (54, None),
        ("photo/road.jpg", "photo/road_thresholded.jpg"),
    ],
)
def test_input_and_output_path(image_path, expected_output_path):
    """
    Test input and output path.
    """
    # Given

    # When
    output_path = apply_threshold_and_save_image(image_path)
    if output_path is not None:
        dirname = os.path.dirname(__file__)
        expected_output_path = os.path.join(dirname, expected_output_path)
        assert os.path.exists(output_path) is True
    # Then
    assert output_path == expected_output_path


# Test for the grayscale and test thresholding (work on the image)
@pytest.mark.parametrize(
    ("image_path"),
    [
        ("photo/road.jpg"),
    ],
)
def test_grayscale_conversion(image_path):
    """
    Test grayscale conversion.
    """
    # Given
    original_frame = cv2.imread(image_path)

    # When
    output_path = apply_threshold_and_save_image(image_path)
    output_frame = cv2.imread(output_path)

    # Then
    assert output_frame is not None
    assert output_frame.shape == original_frame.shape
    assert np.logical_or(
        np.isclose(output_frame, 0, atol=7), np.isclose(output_frame, 255, atol=7)
    ).all()
