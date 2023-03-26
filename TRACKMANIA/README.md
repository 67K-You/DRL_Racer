# DRL_Racer

Git that inspires this project: https://github.com/trackmania-rl/tmrl

# DARLA

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

## config.json shenanigans

If you wish to use a remote training architecture you should modify config.json in TmrlData/config accordingdly : put the public ip of the server under PUBLIC_IP_SERVER and disable LOCALHOST_WORKER/TRAINER depending on which machine those program run relative to the server. Finally don't forget to open the communication ports specified in the config.json to allow for communication
