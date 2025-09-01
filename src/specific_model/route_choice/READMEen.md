# RouteChoice Model Docker Execution Guide

## Overview
This project runs estimation and simulation of route choice models (such as RL/Discounted RL) in a Docker environment.

## Directory Structure
```
code/           # Python code (e.g., main_rl.py)
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
ESTIMATE_MODE=walk
INPUT=input/test
OUTPUT=output/test
```

## Build and Run
```sh
# Build the image
docker-compose build

# Start the container (main_rl.py will run according to .env settings)
docker-compose up
```

## Command & Mode Switching
- Change `ESTIMATE_MODE` in `.env` to 0 or 1 to switch execution mode.
- Change the `INPUT` and `OUTPUT` paths as needed.

## Notes
- For data and output persistence, local directories are bound to container directories via `volumes`.
- Modify Python code in the `code/` directory.

---

# Jupyter Notebook Execution Guide

Follow the instructions in code/main.ipynb and execute the cells sequentially.
