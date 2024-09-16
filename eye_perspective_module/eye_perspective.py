"""
eye_perspective.py

This module provides functionality to apply a perspective transform to an image,
simulating an "eye perspective" effect. The main components include:

1. `PerspectiveTransformParameters`:
   A data class that holds parameters for the perspective transformation, such as
   source and destination points, circle drawing settings, and constraints for
   adjusting points based on white pixels in the image.

2. `is_white_pixel(frame, point)`:
   Checks if a given point in the frame is a white pixel.

3. `find_nearest_white_pixel(frame, point, max_distance)`:
   Finds the nearest white pixel to the given point along the x-axis within a specified
   maximum distance.

4. `validate_point(frame, point, max_distance, max_move_distance, internal_trapezium_point=None, check_right=None)`:
   Validates and adjusts a single point based on specified conditions, ensuring it is a
   white pixel within an allowable move distance and optionally within trapezium constraints.

5. `validate_and_adjust_detection_points(frame, params)`:
   Validates and adjusts the top left, top right, bottom left, and bottom right points
   before applying the perspective transform to ensure they are white pixels and within
   allowable distances.

6. `apply_eye_perspective_transform(frame, params)`:
   Applies the perspective transformation to the input frame using the provided
   parameters. It first validates and adjusts the points, then draws circles at the
   source points and performs the transformation.

7. `main()`:
   The main function that reads an input image, applies the perspective transform, and
   displays and saves the original and transformed frames.
"""

import logging
from dataclasses import dataclass
import cv2
import numpy as np

# pylint: disable=no-member

# This check is disabled because we are using an external library OpenCv
# that Pylint cannot correctly analyze. We are confident that the members we are accessing do exist.

# pylint: disable=I1101

# Configuring logging
logging.basicConfig(level=logging.DEBUG)


@dataclass
class PerspectiveTransformParameters:
    """
    Data class for perspective transform parameters.
    """

    output_width: int = 1920
    output_height: int = 1080
    top_left: tuple = (851, 590)
    bottom_left: tuple = (300, 944)
    top_right: tuple = (1105, 590)
    bottom_right: tuple = (1690, 944)
    destination_top_left: tuple = (50, -100)
    destination_bottom_left: tuple = (50, 1080)
    destination_top_right: tuple = (1870, -100)
    destination_bottom_right: tuple = (1870, 1080)
    circle_radius: int = 5
    circle_color: tuple = (0, 0, 255)
    circle_thickness: int = -1
    max_distance: int = 150  # Maximum limit for searching for white pixels
    max_move_distance: int = (
        150  # Maximum distance to move points from the initial position
    )
    top_left_internal_trapezium: tuple = (938, 570)
    top_right_internal_trapezium: tuple = (988, 570)


def is_white_pixel(frame, point):
    """
    Check if the given point in the frame is a white pixel.

    Args:
        frame (numpy.ndarray): Input frame to be processed.
        point (tuple): The point to check.

    Returns:
        bool: True if the point is a white pixel, False otherwise.
    """
    x, y = point
    # Check if the point is within the frame boundaries
    if x < 0 or x >= frame.shape[1] or y < 0 or y >= frame.shape[0]:
        return False
    # Check if the pixel is white
    return np.array_equal(frame[y, x], [255, 255, 255])


def find_nearest_white_pixel(frame, point, max_distance):
    """
    Find the nearest white pixel to the given point along the x-axis within the max distance.

    Args:
        frame (numpy.ndarray): Input frame to be processed.
        point (tuple): The point to check.
        max_distance (int): Maximum distance to search.

    Returns:
        tuple: The new point with the nearest white pixel.
    """
    x, y = point
    for offset in range(1, max_distance + 1):
        # Check pixel to the right
        if is_white_pixel(frame, (x + offset, y)):
            return (x + offset, y)
        # Check pixel to the left
        if is_white_pixel(frame, (x - offset, y)):
            return (x - offset, y)
    return point


def validate_point(
    frame,
    point,
    max_distance,
    max_move_distance,
    internal_trapezium_point=None,
    check_right=None,
):
    """
    Validate and adjust a single point based on the conditions.

    Args:
        frame (numpy.ndarray): Input frame to be processed.
        point (tuple): The point to validate and adjust.
        max_distance (int): Maximum distance to search for a white pixel.
        max_move_distance (int): Maximum allowed movement distance.
        internal_trapezium_point (tuple): Internal trapezium boundary point for validation.
        check_right (bool): If True, check if the new point is not to the right of the internal trapezium point.
                            If False, check if the new point is not to the left of the internal trapezium point.

    Returns:
        tuple: The adjusted point or the original point if no valid adjustment found.
    """

    def valid_move(new_point, point, max_move_distance):
        return (
            np.linalg.norm(np.array(new_point) - np.array(point)) <= max_move_distance
        )

    def within_trapezium_constraints(new_point, internal_trapezium_point, check_right):
        if internal_trapezium_point is None or check_right is None:
            return True
        return (check_right and new_point[0] <= internal_trapezium_point[0]) or (
            not check_right and new_point[0] >= internal_trapezium_point[0]
        )

    if is_white_pixel(frame, point):
        return point

    logging.info(f"Adjusting point from {point}")
    new_point = find_nearest_white_pixel(frame, point, max_distance)

    if new_point == point:
        logging.info(
            f"No white pixel found within {max_distance} pixels for the point. Keeping original position."
        )
        return point

    if not valid_move(new_point, point, max_move_distance):
        logging.info(
            f"New point {new_point} exceeds max move distance. Keeping original position."
        )
        return point

    if not within_trapezium_constraints(
        new_point, internal_trapezium_point, check_right
    ):
        logging.info(
            f"New point {new_point} is invalid based on internal trapezium constraints. Keeping original position."
        )
        return point

    logging.info(f"Adjusted point to {new_point}")
    return new_point


def validate_and_adjust_detection_points(frame, params):
    """
    Validate and adjust the top left, top right, bottom left, and bottom right points before applying the perspective transform.

    Args:
        frame (numpy.ndarray): Input frame to be processed.
        params (PerspectiveTransformParameters): Parameters for perspective transformation.

    Returns:
        PerspectiveTransformParameters: Updated parameters with validated and adjusted points.
    """
    max_distance = params.max_distance
    max_move_distance = params.max_move_distance

    # Validate and adjust top points with internal trapezium constraints
    params.top_left = validate_point(
        frame,
        params.top_left,
        max_distance,
        max_move_distance,
        params.top_left_internal_trapezium,
        check_right=True,
    )
    params.top_right = validate_point(
        frame,
        params.top_right,
        max_distance,
        max_move_distance,
        params.top_right_internal_trapezium,
        check_right=False,
    )

    # Validate and adjust bottom points without internal trapezium constraints
    params.bottom_left = validate_point(
        frame, params.bottom_left, max_distance, max_move_distance
    )
    params.bottom_right = validate_point(
        frame, params.bottom_right, max_distance, max_move_distance
    )

    return params


def apply_eye_perspective_transform(frame, params=PerspectiveTransformParameters()):
    """
    Apply eye perspective transformation function.

    Args:
        frame (numpy.ndarray): Input frame to be processed.
        params (PerspectiveTransformParameters): Parameters for perspective transformation.

    Returns:
        tuple: A tuple containing the transformed frame and the transformation matrix.
    """

    # Validate and adjust points before applying the transformation
    params = validate_and_adjust_detection_points(frame, params)

    # List of source points
    source_points = [
        params.top_left,
        params.bottom_left,
        params.top_right,
        params.bottom_right,
    ]

    # Draw circles on the frame at the source points
    for point in source_points:
        cv2.circle(
            frame,
            point,
            params.circle_radius,
            params.circle_color,
            params.circle_thickness,
        )

    # Apply geometrical transformation
    source_points_array = np.float32(source_points)

    # Define new destination points for a higher view
    destination_points = np.float32(
        [
            params.destination_top_left,
            params.destination_bottom_left,
            params.destination_top_right,
            params.destination_bottom_right,
        ]
    )

    matrix = cv2.getPerspectiveTransform(source_points_array, destination_points)
    transformed_frame = cv2.warpPerspective(
        frame, matrix, (params.output_width, params.output_height)
    )

    return frame, source_points_array


def main():
    """
    Main function.
    """

    # Read the input image
    input_image_path = "input.jpg"
    frame = cv2.imread(input_image_path)

    if frame is None:
        logging.error("Error loading image")
        return

    # Apply the perspective transform
    params = PerspectiveTransformParameters()
    original_frame, transformed_frame = apply_eye_perspective_transform(frame, params)

    if transformed_frame is not None:
        # Display the original and transformed frames
        cv2.imshow("Original Frame", original_frame)
        cv2.imshow("Transformed Frame", transformed_frame)

        # Save the transformed frame
        output_image_path = "transformed.jpg"  # Change this to your desired output path
        cv2.imwrite(output_image_path, transformed_frame)
        logging.info(f"Transformed image saved to {output_image_path}")

        # Wait for a key press and close the windows
        cv2.waitKey(0)
        cv2.destroyAllWindows()
    else:
        logging.error("Transformed frame is None. Skipping display and save.")


if __name__ == "__main__":
    main()
