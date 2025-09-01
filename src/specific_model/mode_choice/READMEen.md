# ModeChoice Model Docker Execution Guide

## Overview
This project runs estimation and simulation of mode choice models (such as MNL) in a Docker environment.

## Directory Structure
```
code/           # Python code (e.g., main_mc.py)
data/input/     # Input data
data/output/    # Output results
requirements.txt
DockerFile
docker-compose.yml
.env            # Settings for execution mode and paths
```

## Preparation
1. Place the necessary files (data, code, .env).
2. Python dependencies are managed in `requirements.txt`.

## Example .env File
```
ESTIMATE_MODE=true
INPUT=input/test
OUTPUT=output/test
MODEL_NAME=MNL
```

## Build and Run
```sh
# Build the image
docker-compose build

# Start the container (main_mc.py will run according to .env settings)
docker-compose up
```

## Command & Mode Switching
- Change `ESTIMATE_MODE` in `.env` to true or false to switch execution mode.
- Change the `INPUT` and `OUTPUT` paths as needed.
- Change `MODEL_NAME` to switch the model used (currently only those in code/model folder are supported).

## Notes
- For data and output persistence, local directories are bound to container directories via `volumes`.
- Modify Python code in the `code/` directory.

---

# Jupyter Notebook Execution Guide

Follow the instructions in code/main.ipynb and execute the cells sequentially.
