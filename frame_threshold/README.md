# Threshold Module

This is a module that provides functions for thresholding images, in order to convert first the image

in a grayscale and then apply the threshold.

It take as input an image path and returning the path to save the output image.

## `frame_threshold.py`

### Usage

Example usage:

```python

from frame_threshold import apply_threshold_and_save_image

apply_threshold_and_save_image("input_image.jpg", "output_image.jpg", is_preview_enable=True)

```

# Note

In order to avoid Pylint errors is strongly recommended to add the following comment:

```
# pylint: disable=no-member
```

# Test Module for Thresholding Function

This module contains tests for the thresholding function.

## Test Cases

### Test Input Path (`test_input_path`)

This test checks if the input path is correctly handled.

### Test Output Path (`test_output_path`)

This test verifies if the output path is correctly handled based on the function's behavior.

### Test Grayscale Conversion (`test_grayscale_conversion`)

This test ensures that the grayscale conversion is working properly.

## Usage

To run the tests, execute the following command:

```bash

pytest  test_threshold.py

```
