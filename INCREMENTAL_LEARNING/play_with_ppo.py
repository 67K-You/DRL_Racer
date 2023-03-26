from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env
from pyglet.window import key
import gym


def key_press(k, mod):
    global restart
    if k == 0xFF0D:
        restart = True

# Pre-trained model
model_name = "models/ppo_normal_road"

# Creation of the environment
env = make_vec_env("CarRacing-v0", n_envs=1)

# Creation of the model
model = PPO.load(model_name)
    
log = "logs/test/ppo_normalt_road.txt"
file = open(log, "w")
    
max_episode = 100
num_episode = 0

while num_episode <= max_episode:
    obs = env.reset()
    total_reward = 0.0
    restart = False
    while True:
        action, _ = model.predict(obs)
        obs, r, done, info = env.step(action)
        total_reward += r[0]
        isopen = env.render()
        if done or restart:
            print(total_reward)
            file.write(str(total_reward) + "\n")
            num_episode += 1
            break
        
file.close()