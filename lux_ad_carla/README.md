# Carla Project 0.9.15 - Setup

## Description
This project uses **CARLA 0.9.15** for simulating urban environments. Before you can run the project, you need to download and install the correct version of CARLA.

## Installation

### 1. Download CARLA

First, download the **CARLA 0.9.15** version for Linux from the following link:

[Download CARLA 0.9.15 for Linux](https://tiny.carla.org/carla-0-9-15-linux)

### 2. Extract the files

After downloading the file, extract it to the desired folder on your system.

### 3. Copy to the project folder

Once the files are extracted, copy the entire folder to this project directory.

For example, run the following command in the terminal (replacing `<path_to_download>` with the path where you downloaded CARLA):

```bash
cp -r <path_to_download>/CARLA_0.9.15 <path_to_your_project_folder>
```

### 4. Verify

Make sure that the CARLA files have been copied correctly by running the following command to display the version:

```bash
./CARLA_0.9.15/CarlaUE4.sh --version
```

### 5. Run CARLA

Once the CARLA version is installed, you can start the simulation using the command:

```bash
./CARLA_0.9.15/CarlaUE4.sh
```

## System requirements

- **Operating system:** Linux (Ubuntu recommended)
- **Version CARLA:** 0.9.15
- **Other requirements:** OpenGL, Python packages for client API

## Additional notes

Make sure you have all system requirements and dependencies installed before running CARLA.

For more information, see the official [CARLA Simulator](https://carla.org/) documentation.
