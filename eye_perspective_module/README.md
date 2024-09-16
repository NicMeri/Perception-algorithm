# Eye Perspective Module

The `eye_perspective_module` provides functions for processing video frames, specifically transforming an image to simulate an eye perspective view using geometrical transformations.

## Features

- **Frame Processing**: Transforms input frames by applying a perspective transformation based on predefined coordinates.

## Requirements

- Python 3.x
- OpenCV
- NumPy

## Installation

1. Ensure you have Python 3 installed.
2. Install the required packages:

```sh
pip install numpy opencv-python
```

## Usage

### Functions

#### `apply_eye_perspective_transform(frame)`

Processes an input frame to apply a perspective transformation.

- **Args**:
  - `frame` (numpy.ndarray): The input frame to be processed.
- **Returns**:
  - Transformed frame (numpy.ndarray)

### Example

```python
import cv2
from eye_perspective_module import apply_eye_perspective_transform

# Load an example frame (replace 'input_image.jpg' with your image path)
frame = cv2.imread('input_image.jpg')

# Process the frame
frame_with_eye_perspective_view = apply_eye_perspective_transform(frame)

# Display the original and transformed frames
cv2.imshow('Original Frame', frame)
cv2.imshow('Transformed Frame', frame_with_eye_perspective_view)
cv2.waitKey(0)
cv2.destroyAllWindows()
```