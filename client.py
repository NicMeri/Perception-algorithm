"""
Carla Simulation with Thresholded Camera:

This script connects to a Carla simulator instance, spawns a vehicle,
and attaches an RGB camera to it.
Images captured by the camera are showed in a separate window after applying a thresholding filter.
The script also adds additional vehicles to the simulation.
The thresholded camera feed is displayed in a window.

Requirements:
- Carla simulator installed and running
- Frame thresholding module ('frame_threshold.frame_threshold') for image processing
"""

import sys
import random
import time
import tempfile
from dataclasses import dataclass
import logging

import carla

import cv2
import numpy as np

from frame_threshold.frame_threshold import apply_threshold_and_save_image
from eye_perspective_module.eye_perspective import apply_eye_perspective_transform

from lux_ad_carla.PythonAPI.examples.controller import VehiclePIDController

# pylint: disable=no-member

# This check is disabled because we are using an external library OpenCv
# that Pylint cannot correctly analyze. We are confident that the members we are accessing do exist.

# pylint: disable=I1101

# Configure logging
logging.basicConfig(
    level=logging.DEBUG,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[logging.FileHandler("carla_simulation.log"), logging.StreamHandler()],
)


@dataclass
class CameraSettings:
    """
    A class to manage settings for an RGB camera in a Carla simulation.
    """

    image_size_x: int = 1920
    image_size_y: int = 1080
    field_of_view: float = 120
    sensor_tick: float = 0.0333  # 30 frames per second
    location: carla.Location = carla.Location(1.8, 0, 1.3)
    rotation: carla.Rotation = carla.Rotation(-10, 0, 0)


@dataclass
class ThresholdingSettings:
    """
    A class to manage settings for thresholding in the camera callback.
    """

    extern_mask_points: list = ((180, 944), (823, 475), (1103, 475), (1750, 944))
    intern_mask_points: list = (
        (368, 944),
        (931, 570),
        (988, 570),
        (1585, 944),
    )
    alpha: float = 0.5  # Transparency factor for blending
    line_width: int = 3


def initialize_client():
    """
    Initializes the client and returns the client and the world.
    If the connection fails, return None.
    """
    try:
        # First of all, we need to create the client that will send the requests
        # to the simulator. Here we assume the simulator is accepting
        # requests at localhost port 2000
        client = carla.Client("localhost", 2000)
        client.set_timeout(15.0)

        # Once we have a client we can retrieve the world that is currently running
        world = client.get_world()

        # The world contains the list of blueprints that we can use for adding new
        # actors into the simulation
        blueprint_library = world.get_blueprint_library()

        logging.info("Client and world initialized successfully")
        return client, world, blueprint_library
    except carla.ServerError as e:
        logging.error("Server error occurred during client initialization: %s", e)
        sys.exit(1)
    except carla.ClientConnectionError as e:
        logging.error("Error connecting to the Carla server: %s", e)
        sys.exit(2)


def spawn_vehicle(world, actor_list, default_color="255, 0, 0"):
    """
    Spawns a vehicle in the Carla simulation world.

    Args:
        world (carla.World): The Carla simulation world.
        actor_list (list): A list to store the spawned actor.
        default_color (str): The default color for the spawned vehicle in RGB format.

    Returns:
        carla.Actor or None: The spawned vehicle actor if successful, None otherwise.
    """
    # Get the blueprint library from the world
    blueprint_library = world.get_blueprint_library()

    # Find the blueprint for the vehicle
    bp = blueprint_library.find("vehicle.mercedes.coupe_2020")

    # Set the default color attribute for the vehicle blueprint if available
    if bp.has_attribute("color"):
        bp.set_attribute("color", default_color)

    # Initialize vehicle as None
    vehicle = None

    # Attempt to spawn the vehicle multiple times
    for _ in range(10):
        # Select a spawn point from the map
        transform = random.choice(world.get_map().get_spawn_points())

        # Try to spawn the vehicle at the selected spawn point
        vehicle = world.try_spawn_actor(bp, transform)

        # If successful, add the vehicle to the actor list and print a success message
        if vehicle is not None:
            actor_list.append(vehicle)
            logging.info("Created %s", vehicle.type_id)
            break
        # If unsuccessful, print a retry message
        logging.debug("Retrying to spawn vehicle...")

    # If the vehicle is still None after multiple attempts, print a failure message
    if vehicle is None:
        logging.error("Failed to spawn vehicle after multiple attempts.")
        sys.exit(3)

    return vehicle


def spawn_camera(world, vehicle, camera_settings=CameraSettings()):
    """
    Spawns an RGB camera attached to the given vehicle.
    Returns the spawned camera actor.
    """
    cam_bp = world.get_blueprint_library().find("sensor.camera.rgb")
    cam_bp.set_attribute("image_size_x", str(camera_settings.image_size_x))
    cam_bp.set_attribute("image_size_y", str(camera_settings.image_size_y))
    cam_bp.set_attribute("fov", str(camera_settings.field_of_view))
    cam_bp.set_attribute("sensor_tick", str(0.0333))  # 30 frames per second
    cam_transform = carla.Transform(camera_settings.location, camera_settings.rotation)
    ego_cam = world.spawn_actor(
        cam_bp,
        cam_transform,
        attach_to=vehicle,
        attachment_type=carla.AttachmentType.Rigid,
    )

    return ego_cam, cam_bp


def create_mask(image_shape, points):
    """
    Create a binary mask with a filled polygon.

    Args:
        image_shape (tuple): Shape of the image (height, width).
        points (list of tuples): Polygon vertices as (x, y) coordinates.

    Returns:
        numpy.ndarray: Binary mask with the polygon filled.
    """

    mask = np.zeros(image_shape, dtype=np.uint8)
    points_array = np.array(points, dtype=np.int32)
    cv2.fillPoly(mask, [points_array], 255)
    return mask


def draw_polygon(image, points, color, line_width):
    """
    Draw a polygon on an image by connecting the given points.

    Args:
        image (numpy.ndarray): The image to draw on.
        points (list of tuples): Polygon vertices as (x, y) coordinates.
        color (tuple): Color of the polygon in BGR format.
        line_width (int): Thickness of the polygon edges.
    """
    num_points = len(points)
    for i in range(num_points):
        start_point = points[i]
        end_point = points[
            (i + 1) % num_points
        ]  # Ensures that the last point connects to the first
        cv2.line(image, start_point, end_point, color, line_width)


def camera_callback(image, data_dict, thresholding_settings=ThresholdingSettings()):
    """
    Custom callback for processing camera images.
    """

    # Convert the raw image to a numpy array
    image_array = np.array(image.raw_data)
    image_np = image_array.reshape((image.height, image.width, 4))

    # Save the image to a temporary file
    with tempfile.NamedTemporaryFile(suffix=".jpg", delete=False) as temp_file:
        temp_image_path = temp_file.name
        image.save_to_disk(temp_image_path)

        # Apply thresholding to the image
        thresholded_image_path = apply_threshold_and_save_image(temp_image_path)

        # Load the thresholded image back
        thresholded_image = cv2.imread(thresholded_image_path)

        # Create external mask
        external_mask = create_mask(
            thresholded_image.shape[:2], thresholding_settings.extern_mask_points
        )
        masked_image = cv2.bitwise_and(
            thresholded_image, thresholded_image, mask=external_mask
        )

        # Create internal mask
        internal_mask = create_mask(
            masked_image.shape[:2], thresholding_settings.intern_mask_points
        )
        inverted_internal_mask = cv2.bitwise_not(internal_mask)

        masked_image = cv2.bitwise_and(
            masked_image, masked_image, mask=inverted_internal_mask
        )

        # Apply the eye perspective
        eye_perspective_image, source_points = apply_eye_perspective_transform(
            masked_image
        )

        draw_polygon(
            eye_perspective_image,
            thresholding_settings.extern_mask_points,
            (0, 0, 255),
            thresholding_settings.line_width,
        )
        draw_polygon(
            eye_perspective_image,
            thresholding_settings.intern_mask_points,
            (255, 0, 0),
            thresholding_settings.line_width,
        )

        # Draw the polygon using the source_points
        source_points = np.array(
            source_points, dtype=np.int32
        )  # Ensure source_points are in correct format

        # Ensure the points are in the correct order: top_left, bottom_left, bottom_right, top_right
        polygon_points = np.array(
            [source_points[0], source_points[1], source_points[3], source_points[2]],
            dtype=np.int32,
        )

        # Create an overlay image for the transparent polygon
        overlay = image_np.copy()

        # Draw the filled polygon on the overlay
        cv2.fillPoly(overlay, [polygon_points], (0, 255, 0))

        # Blend the overlay with the original image using addWeighted
        cv2.addWeighted(
            overlay,
            thresholding_settings.alpha,
            image_np,
            1 - thresholding_settings.alpha,
            0,
            image_np,
        )

        # Update the data dictionary with the result image
        data_dict["image"] = image_np
        data_dict["eye_perspective_image"] = eye_perspective_image


def setup_camera(ego_cam, cam_bp, camera_data):
    """
    Set up the RGB camera with the given parameters.
    """
    # Set the camera recording data:
    image_w = cam_bp.get_attribute("image_size_x").as_int()
    image_h = cam_bp.get_attribute("image_size_y").as_int()

    camera_data["image"] = np.zeros((image_h, image_w, 4), dtype=np.uint8)
    camera_data["eye_perspective_image"] = np.zeros(
        (image_h, image_w, 4), dtype=np.uint8
    )

    # Define the camera callback
    ego_cam.listen(lambda image: camera_callback(image, camera_data))

    # Create named windows and display the camera feed
    cv2.namedWindow("RGB Camera", cv2.WINDOW_NORMAL)
    cv2.namedWindow("Eye Perspective View", cv2.WINDOW_NORMAL)

    # Resize windows to occupy half the screen width
    screen_width = int(cv2.getWindowImageRect("RGB Camera")[2])
    screen_height = int(cv2.getWindowImageRect("RGB Camera")[3])
    cv2.resizeWindow("RGB Camera", screen_width // 2, screen_height)
    cv2.resizeWindow("Eye Perspective View", screen_width // 2, screen_height)

    # Move the window of the RGB Camera to the left side of the screen
    cv2.moveWindow("RGB Camera", 0, 0)

    # Move the window of the Eye Perspective View to the right side of the screen
    cv2.moveWindow("Eye Perspective View", screen_width // 2, 0)

    cv2.waitKey(1)


def get_target_waypoint(vehicle, world):
    """
    Function to get the target waypoint for the vehicle.
    """

    # Get the map from the world
    carla_map = world.get_map()

    # Get the current location of the vehicle
    location = vehicle.get_location()

    # Get the waypoint closest the current location
    waypoint = carla_map.get_waypoint(location)

    # Get the next waypoint at a distance of 2.0 meters ahead
    next_waypoint = waypoint.next(2.0)[0]

    # Set the target speed for the vehicle
    target_speed = 20.0

    # Return the next waypoint and the target speed
    return next_waypoint, target_speed


def main():
    """
    This main function connects to a Carla simulator instance,
    spawns a vehicle with an attached RGB camera, and sets the vehicle to drive in autopilot mode.
    Images captured by the camera undergo thresholding and are displayed in a window.
    Additional vehicles are also spawned in the simulation.
    """
    actor_list = []
    ego_cam = None

    random.seed(500)

    # Initialize the client, world, and blueprint library
    client, world, blueprint_library = initialize_client()

    # Spawn the vehicle
    vehicle = spawn_vehicle(world, actor_list)

    # Define transform
    transform = vehicle.get_transform()

    # Spawn attached RGB camera
    ego_cam, cam_bp = spawn_camera(world, vehicle)

    # Set up the camera
    camera_data = {}
    setup_camera(ego_cam, cam_bp, camera_data)

    # Define the proportional (K_P), integral (K_I),
    # and derivative (K_D) gains for lateral control
    args_lateral = {"K_P": 1.0, "K_I": 0.0, "K_D": 0.0}

    # Define the proportional (K_P), integral (K_I),
    # and derivative (K_D) gains for longitudinal control
    args_longitudinal = {"K_P": 1.0, "K_I": 0.0, "K_D": 0.0}

    # Create a PID controller instance for the vehicle
    # using the lateral and longitudinal control gains
    pid_controller = VehiclePIDController(vehicle, args_lateral, args_longitudinal)

    try:

        while True:

            # Get the next waypoint and target speed for the vehicle using the defined function
            waypoint, target_speed = get_target_waypoint(vehicle, world)

            # Compute the control command for the vehicle based on
            # the target speed and waypoint using the PID controller
            control = pid_controller.run_step(target_speed, waypoint)

            # Apply the computed control command to the vehicle
            vehicle.apply_control(control)

            cv2.imshow("RGB Camera", camera_data["image"])
            cv2.imshow("Eye Perspective View", camera_data["eye_perspective_image"])

            if cv2.waitKey(1) == ord("q"):  # Close windows when press `q`
                break

        cv2.destroyAllWindows()

        # Add a few more vehicles to the simulation
        transform.location += carla.Location(x=40, y=-3.2)
        transform.rotation.yaw = -180.0
        for _ in range(0, 10):
            transform.location.x += 8.0

            bp = random.choice(blueprint_library.filter("vehicle"))

            # Use try_spawn_actor. If the spot is occupied by another object,
            # the function will return None.
            npc = world.try_spawn_actor(bp, transform)
            if npc is not None:
                actor_list.append(npc)
                npc.set_autopilot(True)
                logging.info("created %s", npc.type_id)

        time.sleep(5)

    except (carla.ServerError, carla.ClientError, carla.RPCError) as e:
        logging.error("Carla error occurred: %s", e)
    except UnexpectedError as e:
        logging.error("An unexpected error occurred: %s", e)

    finally:
        logging.info("destroying actors")
        if ego_cam is not None:
            ego_cam.destroy()
        client.apply_batch([carla.command.DestroyActor(x) for x in actor_list])
        logging.info("done.")


if __name__ == "__main__":
    main()
