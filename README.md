# Perception-algorithm
Perception algorithm for Lane Detection for master's thesis project

## Description of the project
This project was developed as a master's thesis in the Mechatronics Engineering program at the Polytechnic of Turin, during my internship at the Luxoft company.
The aim of the project is the recognition of road lanes in an urban environment in a simulated world thanks to CARLA.
The project was implemented in Python and uses the OpenCV library to manipulate images.

## Requirements
Before running the project, make sure you have the following requirements installed:
- Python 3.x (preferably 3.7)
- OpenCV
- CARLA
- NumPy

Other standard Python libraries:
  - `sys`
  - `random`
  - `time`
  - `tempfile`
  - `dataclasses`
  - `logging`

## Installation

### 1. Preparation of the environment
Before running the project, make sure you have downloaded and installed CARLA correctly. You will need to create a folder called `lux_ad_carla`, where you will put all the files downloaded from CARLA.

### 2. Configuring the virtual environment
Make sure you have: [Conda] [https://docs.conda.io/projects/conda/en/latest/user-guide/install/index.html] installed to create the virtual environment:
1. Create a new Conda environment and install Python 3.7:
   ```bash
   conda create -n carlaEnv python=3.7
   ```
2. Activate the environment:
   ```
   conda activate carlaEnv
   ```
3. Installing dependencies:
   ```
   pip install opencv-python numpy
   ```
4. Configuration of CARLA and Luxoft toolkit: Make sure you download the files required by CARLA and follow the instructions in the luxad_toolkit.sh file found in the repository. To activate the toolkit and correctly configure the CARLA environment, run the following commands:
   ```
   source luxad_toolkit.sh
   luxad_run_server
   luxad_run_client
   ```

The client performs road lane recognition in a simulated urban environment within CARLA.

## Project Structure
- `client.py`: Main script for lane recognition.
- `eye_perspective.py`: Submodule for managing the perspective view.
- `frame_threshold.py`: Submodule for image processing via thresholding.
- `luxad_toolkit.sh`: Script for configuring and running the server and client in CARLA.

## Additional notes
- This project requires CARLA version 0.9.15. Make sure you follow all installation instructions correctly.
- It is recommended to use a computer with an NVIDIA graphics card to optimize the performance of the CARLA simulator.

## Authors
Niccolò Mariani
