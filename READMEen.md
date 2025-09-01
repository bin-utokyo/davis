# Davis Project

## Overview
This repository provides tools for distributing data usable at the Summer School on Behavioral Models and sample codes for behavioral models:
- Dataset management tool: Display information and download data
- Individual models: Estimation and simulation code for behavioral models
- Base model: Micro traffic simulator

## Structure
```
packages/
  dataset_cli/          # Dataset management tool

src/
  specific_model/       # Individual behavior models
    mode_choice/        # Mode choice models (e.g., MNL)
    route_choice/       # Route choice models (e.g., RL)

  base_model/           # Base model (MFD-RL+Hongo)
    Hongo/              # Hongo simulator source
    MFDRL-Hongo/        # MFD-RL+Hongo simulator source
```

## Download
Please clone the latest code from GitHub:
```sh
git clone https://github.com/bin-utokyo/davis.git
```

## Notes
- For details and .env examples, please refer to each model's README.

## Documentation
- For details on the dataset management tool, see `packages/dataset_cli/README.md`.
- For details on the base model, see `src/base_model/README.md`.
- For details on each behavioral model, see `src/specific_model/{model_name}/README.md`.

---

For questions or contributions, please contact the organizers of the Summer School on Behavioral Models.
