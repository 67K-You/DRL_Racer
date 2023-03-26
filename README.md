# DRL_Racer

Git that inspires this project: https://github.com/trackmania-rl/tmrl

# DARLA

This section is inspired by the git: https://github.com/BCHoagland/DARLA?fbclid=IwAR1o6RzmMkMUY4dwjs1KallFpv_9riNHDvJscgbaXCsegjuKHOaE6Q_L6fk


## Getting started :

The following packages should be installed
<code>pip install tmrl==0.3.1</code>
<code>pip install rtgym==0.7</code>

For the training of DARLA you should follow this procedure : 

1. Step 1 (vision module training) :

* Launch TMRL server for worker and trainer communication :
<code>python -m tmrl --server</code>

* Launch TMRL worker to make the a pre-trained AI play the game :
<code>python -m tmrl --worker</code>

* Launch the vision module trainer :
<code>python tuto_competition_step_1.py</code>

2. Step 2 (policy learning with pretrained vision module) :
Make sure you have a pretrained vision module (Beta-VAE) state dict savec in the same folder as tuto_competition_darla_step_2_log.py under the name bvae-test-model.pkl

* Launch TMRL server for worker and trainer communication :
<code>python -m tmrl --server</code>

* Launch custom TMRL that accepts the new network architecture :
<code>python worker.py</code>

* Launch the SAC trainer :
<code>python tuto_competition_darla_step_2_log.py</code>

The three files *tuto_competition_step_1.py*, *tuto_competition_darla_step_2.py* and *tuto_competition_darla_step_2_log.py* are modified version of python files given by the 
*tmrl* git. The original files that helped building these three scripts can be found in the folder *tmrl_original_files/*.

## config.json shenanigans

If you wish to use a remote training architecture you should modify config.json in TmrlData/config accordingdly : put the public ip of the server under PUBLIC_IP_SERVER and disable LOCALHOST_WORKER/TRAINER depending on which machine those program run relative to the server. Finally don't forget to open the communication ports specified in the config.json to allow for communication

# Incremental Learning

For this project, incremental learning methods have been implemented. All the work that hav been done and all the documentation can be found in the folder *INCREMENTAL_LEARNING/*
