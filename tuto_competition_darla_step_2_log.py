"""
=======================================================================
COMPETITION TUTORIAL #1: Custom model and RL algorithm
=======================================================================

In this tutorial, we customize the default TrackMania pipeline.

To submit an entry to the TMRL competition, we essentially need a
trained policy. In TMRL, this is encapsulated in an ActorModule.

Note: this tutorial describes implementing a TrainingAgent in TMRL.
The TMRL framework is relevant if you want to implement RL approaches.
If you plan to try non-RL approaches instead, this is also accepted:
just use the Gym environment and do whatever you need,
then, wrap your trained policy in an ActorModule, and submit :)
"""

# Okay, first, let us import some useful stuff.
# The constants that are defined in config.json:
import tmrl.config.config_constants as cfg
# Higher-level partially instantiated classes that are fixed for the competition:
# (in particular this includes the Gym environment)
import tmrl.config.config_objects as cfg_obj  # higher-level constants that are fixed for the competition
# The utility that is used in TMRL to partially instantiate classes:
from tmrl.util import partial
# The main TMRL components of a training pipeline:
from tmrl.networking import Server, RolloutWorker, Trainer

# The training object that we will customize with our own algorithm to replace the default SAC trainer:
from tmrl.training_offline import TrainingOffline
from tmrl.training import TrainingAgent
from tmrl.nn import no_grad, copy_shared

from gym.spaces.box import Box

# External libraries:
import numpy as np
import torch


import torch.nn as nn
from torch.autograd import Variable
from tmrl.util import *
from torch.nn import functional as F
import itertools
import math

cuda = torch.cuda.is_available()
# =====================================================================
# USEFUL PARAMETERS
# =====================================================================
# You can change these parameters here directly,
# or you can change them in the config.json file.

# maximum number of training 'epochs':
# training is checkpointed at the end of each 'epoch'
# this is also when training metrics can be logged to wandb
epochs = cfg.TMRL_CONFIG["MAX_EPOCHS"]

# number of rounds per 'epoch':
# training metrics are displayed in terminal at the end of each round
rounds = cfg.TMRL_CONFIG["ROUNDS_PER_EPOCH"]

# number of training steps per round:
# (a training step is a call to the train() function that we will define later)
steps = cfg.TMRL_CONFIG["TRAINING_STEPS_PER_ROUND"]

# minimum number of environment steps collected before training starts
# (this is useful when you want to fill your replay buffer with samples from a baseline policy)
start_training = cfg.TMRL_CONFIG["ENVIRONMENT_STEPS_BEFORE_TRAINING"]

# maximum training steps / env steps ratio:
# (if training becomes faster than this ratio, it will be paused waiting for new samples from the environment)
max_training_steps_per_env_step = cfg.TMRL_CONFIG["MAX_TRAINING_STEPS_PER_ENVIRONMENT_STEP"]

# number of training steps between when the Trainer broadcasts policy updates:
update_model_interval = cfg.TMRL_CONFIG["UPDATE_MODEL_INTERVAL"]

# number of training steps between when the Trainer updates its replay buffer with the buffer of received samples:
update_buffer_interval = cfg.TMRL_CONFIG["UPDATE_BUFFER_INTERVAL"]

# training device (e.g., "cuda:0"):
# if None, the device will be selected automatically
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(f"Training device : {device}")
# maximum size of the replay buffer:
memory_size = cfg.TMRL_CONFIG["MEMORY_SIZE"]

# batch size for training:
batch_size = cfg.TMRL_CONFIG["BATCH_SIZE"]


# =====================================================================
# ADVANCED PARAMETERS
# =====================================================================
# You may want to change the following in advanced applications;
# however, most competitors will not need to change this.
# If interested, read the full TMRL tutorial.

# base class of the replay memory:
memory_base_cls = cfg_obj.MEM

# sample preprocessor for data augmentation:
sample_preprocessor = None

# path from where an offline dataset can be loaded:
dataset_path = cfg.DATASET_PATH


# =====================================================================
# COMPETITION FIXED PARAMETERS
# =====================================================================
# Competitors CANNOT change the following parameters.
# (Note: For models such as RNNs, you don't need to use imgs_buf_len
# and act_buf_len, but your ActorModule implementation needs to work
# with the observations corresponding to their default values. The rule
# about these history lengths is only here for simplicity. You are
# allowed to hack this within your ActorModule implementation by, e.g.,
# storing histories if you like.)

# rtgym environment class (full TrackMania Gym environment):
env_cls = cfg_obj.ENV_CLS

# number of consecutive screenshots (this is part of observations):
imgs_buf_len = cfg.IMG_HIST_LEN

# number of actions in the action buffer (this is part of observations):
act_buf_len = cfg.ACT_BUF_LEN


# =====================================================================
# MEMORY CLASS
# =====================================================================
# Nothing to do here.
# This is the memory class passed to the Trainer.
# If you need a custom memory, change the relevant advanced parameters.

memory_cls = partial(memory_base_cls,
                     memory_size=memory_size,
                     batch_size=batch_size,
                     sample_preprocessor=sample_preprocessor,
                     dataset_path=cfg.DATASET_PATH,
                     imgs_obs=imgs_buf_len,
                     act_buf_len=act_buf_len,
                     crc_debug=False,
                     use_dataloader=False,
                     pin_memory=False)


# =====================================================================
# CUSTOM MODEL
# =====================================================================
# Alright, now for the fun part.
# Our goal in this competition is to come up with the best trained
# ActorModule for TrackMania 2020, where an 'ActorModule' is a policy.
# In this tutorial, we present a deep RL-way of tackling this problem:
# we implement our own deep neural network architecture (ActorModule),
# and then we implement our own RL algorithm to train this module..


# We implement SAC and a hybrid CNN/MLP model.
# The following constants are from the Spinnup implementation of SAC
# that we simply adapt in this tutorial.
LOG_STD_MAX = 2
LOG_STD_MIN = -20


from tmrl.actor import ActorModule
import torch
from numpy import floor
from copy import deepcopy


# In the full version of the TrackMania 2020 environment, the
# observation-space comprises a history of screenshots. Thus, we need
# Computer Vision layers such as a CNN in our model to process these.
# The observation space also comprises single floats representing speed,
# rpm and gear. We will merge these with the information contained in
# screenshots thanks to an MLP following our CNN layers.


# Let us first define a simple MLP:
def mlp(sizes, activation, output_activation=nn.Identity):
    layers = []
    for j in range(len(sizes) - 1):
        act = activation if j < len(sizes) - 2 else output_activation
        layers += [nn.Linear(sizes[j], sizes[j + 1]), act()]
    return nn.Sequential(*layers)



class BetaVAE(nn.Module):
    def __init__(self, image_shape,latent_dim):
        super(BetaVAE,self).__init__()

        self.image_shape = image_shape # a 28x28 image corresponds to 4 on the FC layer, a 64x64 image corresponds to 13
                            # can calculate this using output_after_conv() in utils.py
        self.latent_dim = latent_dim

        self.encoder = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=4, stride=1),
            nn.ReLU(),
            nn.Conv2d(32, 32, kernel_size=4, stride=1),
            nn.ReLU(),
            nn.Conv2d(32, 32, kernel_size=4, stride=2),
            nn.ReLU(),
            nn.Conv2d(32, 32, kernel_size=4, stride=2),
            nn.ReLU())
        self.x_size = int(math.ceil((((((image_shape[0] - 10) / 2 ) + 1 ) - 4 ) / 2 ) + 1))
        self.y_size = int(math.ceil((((((image_shape[1] - 10) / 2 ) + 1 ) - 4 ) / 2 ) + 1))
        self.fc_mu = nn.Linear(32 * self.x_size * self.y_size, self.latent_dim)
        self.fc_sigma = nn.Linear(32 * self.x_size * self.y_size, self.latent_dim)
        self.fc_up = nn.Linear(self.latent_dim, 32 * self.x_size * self.y_size)
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(32, 32, kernel_size=4, stride=2),
            nn.ReLU(),
            nn.ConvTranspose2d(32, 32, kernel_size=4, stride=2),
            nn.ReLU(),
            nn.ConvTranspose2d(32, 32, kernel_size=4, stride=1),
            nn.ReLU(),
            nn.ConvTranspose2d(32, 1, kernel_size=4, stride=1),
            nn.Sigmoid())

    def forward(self, x):
        n = x.size()[0]
        z = self.encoder(x)
        z = z.view(-1, 32*self.x_size*self.y_size)
        mu_z = self.fc_mu(z)
        log_sigma_z = self.fc_sigma(z)
        sample_z = mu_z + log_sigma_z.exp()*Variable(torch.randn(n, self.latent_dim)).to(device)
        x_hat = self.fc_up(sample_z)
        x_hat = x_hat.view(-1, 32, self.x_size, self.y_size)
        x_hat = self.decoder(x_hat)

        return mu_z, log_sigma_z, x_hat
    
    def infer_latent_representation(self, x):
        n = x.size()[0]
        z = self.encoder(x)
        z = z.view(-1, 32*self.x_size*self.y_size)
        mu_z = self.fc_mu(z)
        log_sigma_z = self.fc_sigma(z)
        sample_z = mu_z + log_sigma_z.exp()*Variable(torch.randn(n, self.latent_dim)).to(device)
        return sample_z

# Training agent:


class MyCriticModule(torch.nn.Module):
    def __init__(self, observation_space, action_space, hidden_sizes=(256, 256), activation=torch.nn.ReLU, latent_dim=16,vision_module = None):
        super().__init__()
        mlp_input = 0
        speed, gear, rpm, images, act1, act = observation_space
        mlp_input += prod(s for s in speed.shape)
        mlp_input += prod(s for s in gear.shape)
        mlp_input += prod(s for s in rpm.shape)
        mlp_input += latent_dim
        act_dim = action_space.shape[0]
        self.vision_module = deepcopy(vision_module).to(device)
        for param in vision_module.parameters():
            param.requires_grad = False
        self.q_full = mlp([mlp_input + act_dim, hidden_sizes[0]], activation)
        self.q_rec = torch.nn.LSTM(hidden_sizes[-1],hidden_sizes[-1])
        self.linear = nn.Linear(hidden_sizes[-1], 1)

    def forward(self, obs, act):
        speed, gear, rpm, images, act1, act = obs
        x = torch.cat((speed, gear, rpm, self.vision_module.infer_latent_representation(Variable(images[:,3:,:,:]).to(device)), act), -1)
        mlp_out = self.q_full(x)
        self.q_rec.flatten_parameters()
        rnn_out, _ = self.q_rec(mlp_out)
        q = self.linear(rnn_out)
        return torch.squeeze(q, -1)


class MyActorCriticModule(torch.nn.Module):
    def __init__(self, observation_space, action_space, hidden_sizes=(256, 256), activation=torch.nn.ReLU,latent_dim=16,image_shape = [64,64]):
        super().__init__()
        self.bvae = BetaVAE(image_shape,latent_dim).to(device)
        bvae_state_dict = torch.load('bvae-test-model.pkl')
        self.bvae.load_state_dict(bvae_state_dict)
        self.actor = MyActorModule(observation_space, action_space, hidden_sizes, activation, latent_dim=16, vision_module = self.bvae)  # our ActorModule :)
        self.q1 = MyCriticModule(observation_space, action_space, hidden_sizes, activation, latent_dim=16, vision_module = self.bvae)  # Q network 1
        self.q2 = MyCriticModule(observation_space, action_space, hidden_sizes, activation, latent_dim=16, vision_module = self.bvae)  # Q network 2


class MyActorModule(ActorModule):
    """
    Directly adapted from the Spinup implementation of SAC
    """
    def __init__(self, observation_space, action_space, hidden_sizes=(256, 256), activation=torch.nn.ReLU, latent_dim=16, vision_module = None):
        super().__init__(observation_space, action_space)
        mlp_input = 0
        self.observation_space = observation_space
        speed, gear, rpm, images, act1, act = observation_space
        mlp_input += prod(s for s in speed.shape)
        mlp_input += prod(s for s in gear.shape)
        mlp_input += prod(s for s in rpm.shape)
        mlp_input += latent_dim
        dim_act = action_space.shape[0]
        act_limit = action_space.high[0]
        self.vision_module = deepcopy(vision_module)
        self.vision_module = self.vision_module.to(device)
        for param in vision_module.parameters():
            param.requires_grad = False
        self.net = mlp([mlp_input,hidden_sizes[0]], activation, activation)
        self.rnn_module = nn.LSTM(hidden_sizes[-1],hidden_sizes[-1])
        self.mu_layer = torch.nn.Linear(hidden_sizes[-1], dim_act)
        self.log_std_layer = torch.nn.Linear(hidden_sizes[-1], dim_act)
        self.act_limit = act_limit

    def forward(self, obs, test=False, with_logprob=True):
        speed, gear, rpm, images, act1, act = obs
        custom_obs = torch.cat((speed, gear, rpm, self.vision_module.infer_latent_representation(Variable(images[:,3:,:,:]).to(device))), -1)
        net_out = self.net(custom_obs)
        self.rnn_module.flatten_parameters()
        rnn_out, _ = self.rnn_module(net_out)
        mu = self.mu_layer(rnn_out)
        log_std = self.log_std_layer(rnn_out)
        log_std = torch.clamp(log_std, LOG_STD_MIN, LOG_STD_MAX)
        std = torch.exp(log_std)
        pi_distribution = torch.distributions.normal.Normal(mu, std)
        if test:
            pi_action = mu
        else:
            pi_action = pi_distribution.rsample()
        if with_logprob:
            logp_pi = pi_distribution.log_prob(pi_action).sum(axis=-1)
            logp_pi -= (2 * (np.log(2) - pi_action - F.softplus(-2 * pi_action))).sum(axis=1)
        else:
            logp_pi = None
        pi_action = torch.tanh(pi_action)
        pi_action = self.act_limit * pi_action
        pi_action = pi_action.squeeze()
        return pi_action, logp_pi

    def act(self, obs, test=False):
        with torch.no_grad():
            a, _ = self.forward(obs, test, False)
            return a.numpy()

# =====================================================================
# CUSTOM TRAINING ALGORITHM
# =====================================================================
# So far, we have implemented our custom model.
# We have also wrapped it in an ActorModule, which we will train and
# submit as an entry to the TMRL competition.
# Our ActorModule will be used in Workers to collect training data.
# Our VanillaCNNActorCritic will be used in the Trainer for training
# this ActorModule. Let us now tackle the training algorithm per-se.
# In TMRL, this is done by implementing a custom TrainingAgent.

# A TrainingAgent must implement two methods:
# - train(batch): optimizes the model from a batch of RL samples
# - get_actor(): outputs a copy of the current ActorModule
# In this tutorial, we will implement the Soft Actor-Critic algorithm
# by adapting the OpenAI Spinnup implementation to the TMRL library.
class SACTrainingAgent(TrainingAgent):
    """
    Our custom training algorithm (SAC).

    Args:
        observation_space (Gym.spaces.Space): observation space (here for your convenience)
        action_space (Gym.spaces.Space): action space (here for your convenience)
        device (str): torch device that should be used for training (e.g., `"cpu"` or `"cuda:0"`)
    """

    # no-grad copy of the model used to send the Actor weights in get_actor():
    model_nograd = cached_property(lambda self: no_grad(copy_shared(self.model)))

    def __init__(self,
                 observation_space=None,  # Gym observation space (required argument here for your convenience)
                 action_space=None,  # Gym action space (required argument here for your convenience)
                 device=None,  # Device our TrainingAgent should use for training (required argument)
                 model_cls=MyActorCriticModule,  # an actor-critic module, encapsulating our ActorModule
                 gamma=0.99,  # discount factor
                 polyak=0.995,  # exponential averaging factor for the target critic
                 alpha=0.2,  # fixed (SAC v1) or initial (SAC v2) value of the entropy coefficient
                 beta=16,   # dimension of the latent space for the ß-VAE
                 lr_actor=1e-3,  # learning rate for the actor
                 lr_critic=1e-3,  # learning rate for the critic
                 lr_entropy=1e-3,  # entropy autotuning coefficient (SAC v2)
                 learn_entropy_coef=True,  # if True, SAC v2 is used, else, SAC v1 is used
                 target_entropy=None,  # if None, the target entropy for SAC v2 is set automatically
                 image_shape=[64,64],
                 latent_dim = 16):
        super().__init__(observation_space=observation_space,
                         action_space=action_space,
                         device=device)
        model = model_cls(observation_space, action_space)
        self.model = model.to(device)
        self.model_target = no_grad(deepcopy(self.model))
        self.gamma = gamma
        self.polyak = polyak
        self.alpha = alpha
        self.lr_actor = lr_actor
        self.lr_critic = lr_critic
        self.lr_entropy = lr_entropy
        self.learn_entropy_coef=learn_entropy_coef
        self.target_entropy = target_entropy
        self.q_params = itertools.chain(self.model.q1.parameters(), self.model.q2.parameters())
        self.pi_optimizer = torch.optim.Adam(self.model.actor.parameters(), lr=self.lr_actor)
        self.q_optimizer = torch.optim.Adam(self.q_params, lr=self.lr_critic)
        if self.target_entropy is None:
            self.target_entropy = -np.prod(action_space.shape).astype(np.float32)
        else:
            self.target_entropy = float(self.target_entropy)
        if self.learn_entropy_coef:
            self.log_alpha = torch.log(torch.ones(1, device=self.device) * self.alpha).requires_grad_(True)
            self.alpha_optimizer = torch.optim.Adam([self.log_alpha], lr=self.lr_entropy)
        else:
            self.alpha_t = torch.tensor(float(self.alpha)).to(self.device)
        self.bvae = BetaVAE(image_shape,latent_dim).to(device)
        bvae_state_dict = torch.load('bvae-test-model.pkl')
        self.bvae.load_state_dict(bvae_state_dict)
        self.iter = 0

    def get_actor(self):
        return self.model_nograd.actor

    def train(self, batch):
        o, a, r, o2, d, _ = batch
        pi, logp_pi = self.model.actor(o)
        loss_alpha = None
        if self.learn_entropy_coef:
            alpha_t = torch.exp(self.log_alpha.detach())
            loss_alpha = -(self.log_alpha * (logp_pi + self.target_entropy).detach()).mean()
        else:
            alpha_t = self.alpha_t
        if loss_alpha is not None:
            self.alpha_optimizer.zero_grad()
            loss_alpha.backward()
            self.alpha_optimizer.step()
        q1 = self.model.q1(o, a)
        q2 = self.model.q2(o, a)
        with torch.no_grad():
            a2, logp_a2 = self.model.actor(o2)
            q1_pi_targ = self.model_target.q1(o2, a2)
            q2_pi_targ = self.model_target.q2(o2, a2)
            q_pi_targ = torch.min(q1_pi_targ, q2_pi_targ)
            backup = r + self.gamma * (1 - d) * (q_pi_targ - alpha_t * logp_a2)
        loss_q1 = ((q1 - backup)**2).mean()
        loss_q2 = ((q2 - backup)**2).mean()
        loss_q = loss_q1 + loss_q2
        self.q_optimizer.zero_grad()
        loss_q.backward()
        self.q_optimizer.step()
        for p in self.q_params:
            p.requires_grad = False
        q1_pi = self.model.q1(o, pi)
        q2_pi = self.model.q2(o, pi)
        q_pi = torch.min(q1_pi, q2_pi)
        loss_pi = (alpha_t * logp_pi - q_pi).mean()
        self.pi_optimizer.zero_grad()
        loss_pi.backward()
        self.pi_optimizer.step()
        for p in self.q_params:
            p.requires_grad = True
        with torch.no_grad():
            for p, p_targ in zip(self.model.parameters(), self.model_target.parameters()):
                p_targ.data.mul_(self.polyak)
                p_targ.data.add_((1 - self.polyak) * p.data)
        if self.iter % 20 == 0:
            with open("logs.txt", "a") as f:
                f.write(f"{loss_q} {loss_pi}\n")
            print(f"[Iter {self.iter}] Loss Q value: {loss_q} Loss action Pi: {loss_pi}")
        self.iter += 1
        ret_dict = dict(
            loss_actor=loss_pi.detach(),
            loss_critic=loss_q.detach(),
        )
        if self.learn_entropy_coef:
            ret_dict["loss_entropy_coef"] = loss_alpha.detach()
            ret_dict["entropy_coef"] = alpha_t.item()
        return ret_dict


training_agent_cls = partial(SACTrainingAgent,
                             model_cls=MyActorCriticModule,
                             gamma=0.99,
                             polyak=0.995,
                             alpha=0.2,
                             lr_actor=1e-3,
                             lr_critic=1e-3,
                             lr_entropy=1e-3,
                             learn_entropy_coef=True,
                             target_entropy=None)


# Trainer instance:

training_agent_cls = partial(SACTrainingAgent,
                             model_cls=MyActorCriticModule,
                             gamma=0.99,
                             polyak=0.995,
                             alpha=0.2,
                             lr_actor=1e-3,
                             lr_critic=1e-3,
                             lr_entropy=1e-3,
                             learn_entropy_coef=True,
                             target_entropy=None)


training_cls = partial(TrainingOffline,
    env_cls=env_cls,
    memory_cls=memory_cls,
    training_agent_cls=training_agent_cls,
    epochs=epochs,
    rounds=rounds,
    steps=steps,
    update_buffer_interval=update_buffer_interval,
    update_model_interval=update_model_interval,
    max_training_steps_per_env_step=max_training_steps_per_env_step,
    start_training=start_training,
    device=device)

if __name__ == "__main__":
    my_trainer = Trainer(training_cls=training_cls)
    my_trainer.run()
