# Incremental Learning

This folder lists all the necessary elements that have been implemented to develop incremental learning methods for autonomous driving.

# Requirements

All the packages needed to run the files:
- gym package with box2d environment (for the simulation): *!pip3 install gym[box2d]==0.21.0*
- pyglet precise version (for the simulation): *!pip3 install pyglet==1.5.27*
- pygame (for the simulation): *!pip3 install pygame*
- StableBaselines 3 packages (for the PPO algorithm): *!pip3 install stable-baselines3*
- Pillow (for images): *!python3 -m pip install --upgrade Pillow*
- OpenCV (for images): *!pip3 install opencv-python*
- torch, torchvision: *for the installation, refer to https://pytorch.org/*
- numpy
- matplotlib

# The environment

The environment considered in this section is the *CarRacing-v0* environment from Gym: https://www.gymlibrary.dev/environments/box2d/car_racing/

# Description of the folder

This folder contains four files:
- car_racing_human.py: a script that allows the user to test the environment by playing the game using the keyboar;
- train_ppo.py: a script that launch a training process of a PPO algorithm considering a specific sub-environment of *CarRacing-v0*;
- plot_reward_func.py: a script that allows the user to visualize the leanring process during the training;
- play_with_ppo.py: a script that allows the user to launch the game that will be played by a trained model.

This folder contains four sub-folders:
- envs: the different environments considered for the experiments, all derived from the original *CarRacing-v0*.
**BE CAREFULL, BECAUSE OF SOME BUGS WHEN USING DERIVED CLASS, THE USER HAVE TO MANUALLY CHANGE THE ORIGINAL CARRACING-V0 ENVIRONMENT DIRECTLY IN THE SOURCE OF GYM.**
**THE CODE THAT HAVE TO BE REPLACED CAN BE FOUND */path_to_site_packages/gym/envs/box2d/car_racing.py.***
**COPY PAST THE CODE FOR THE envs FOLDER INTO THIS SPECIFIC FILE**
- logs: the logs of the training process for all the environment tested in this project
- models: the models that have been saved from the experiments
- Modified_PPO: the modified files of *StableBaselines3* PPO implementation that implement *Policy Relaxation* and *Importance Weighting*

# How to use it

To train a new model from scratch, launch *python3 train_ppo.py* considering the variable *from_pre_trained* set to **False**
To train a new model from a pre-trained model, launch *python3 train_ppo.py* considering the variable *from_pre_trained* set to **True**

# Contact

May you encounter any problem, please contact: thomas.raynaud@ensta-paris.fr