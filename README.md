# Ex-MDPO

# Getting Started

## Prerequisites

All dependencies are provided in a python virtual-env `requirements.txt` file. Majorly, you would need to install `stable-baselines3`, `pytorch`, and `mujoco_py`.

## Installation

1. Activate `virtual-env` using the `requirement.txt` file provided.

    ```bash
    python3 -m venv my_env  #python>=3.8
    source <virtual env path>/my_env/bin/activate
    pip3 install -r requirement.txt
    ```

2. Install stable-baselines3, rllte, gymnasium, Grid2Op and QFB

    ```bash
    pip3 install stable-baselines3==2.4.1
    pip3 install gymnasium[atari]==0.29.1
    pip3 install grid2op
    git clone https://github.com/leandergrech/qfb_env.git
    pip3 install -e .
    git clone https://github.com/RLE-Foundation/rllte.git
    pip3 install -e .
    ```

3. Download and copy MuJoCo library and license files into a `.mujoco/` directory. We use `mujoco200` for this project.

    ```bash
     git clone https://github.com/openai/mujoco-py
     cd mujoco-py
     pip install -r requirements.txt
     pip install -r requirements.dev.txt
     pip install -e . --no-cache
    ```

4. Clone MDPO and copy the `mdpo` directories inside (https://github.com/DLR-RM/stable-baselines3/tree/master/stable_baselines3).

# Example

Use the `bash runs_Mujoco.sh` to run model.
