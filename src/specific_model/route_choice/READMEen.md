# RouteChoice Model — Docker Execution Guide

## Overview
This project runs estimation and simulation for route choice models (e.g., RL, Discounted RL) inside a Docker environment.

## Directory structure
```
code/           # Python code (e.g., main_rl.py)
data/input/     # Input data
data/output/    # Output results
requirements.txt
DockerFile
docker-compose.yml
.env            # Execution mode and path settings
```

## Preparation
1. Place the required files (data, code, and a `.env` file) in the project directory.
2. Python dependencies are managed via `requirements.txt`.

## Example `.env` file
```ini
NETWORK_FROM_OSM=true
MAPMATCHING=true
ESTIMATE=true
SIMULATE=true
ASSIGNMENT=true

INPUT=./input
OUTPUT=./output
MODEL_NAME=RL
TRANSPORTATION_MODE=300

POLYGON_COORD='[[139.698544, 35.660225], [139.698544, 35.656913], [139.705410, 35.656913], [139.705410, 35.660225]]'
```

## Build and run
```sh
# Build the Docker image
docker-compose build

# Start the container (main_rl.py will run according to the settings in .env)
docker-compose up
```

## Command & mode switching
- Toggle execution behavior by editing the `.env` file. The following keys are commonly used:

	- `NETWORK_FROM_OSM` (boolean): when true, build the network from OSM.
	- `MAPMATCHING` (boolean): enable or disable map-matching steps.
	- `ESTIMATE` (boolean): run the estimation pipeline when enabled.
	- `SIMULATE` (boolean): run the simulation pipeline when enabled.
	- `ASSIGNMENT` (boolean): run the assignment pipeline when enabled.
	- `MODEL_NAME` (string): model identifier, e.g. `RL`
	- `TRANSPORTATION_MODE` (integer or string): mode code used by your PP/feeder data — set this to match your inputs.
	- `INPUT` / `OUTPUT` (paths): directories mounted into the container for inputs and outputs.

- In many scripts, boolean values accept `true`/any other word.

## Notes
- Local directories are bound to container directories using Docker `volumes` so that data and output persist on the host.
- Modify Python code inside the `code/` directory.

---

# Jupyter Notebook execution
Follow the instructions in `code/main.ipynb` and run cells sequentially.
