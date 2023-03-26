import os
import matplotlib.pyplot as plt
import numpy as np

# Plot the evolution of the reward over time for each simulation of the ppo training
def plot_ppo_reward_simulation(*folders):
    for folder in folders:
        reward_dict = {}

        for file in os.listdir(folder):
            var = file.split(".")[1] + "_" + file.split(".")[0]
            path = folder + file
            
            reward_dict[var] = []
            
            # Read the reward from log file
            with open(path, 'r') as monitor:
                for index, line in enumerate(monitor):
                    if index >= 2:
                        reward = float(line.split(",")[0])
                        reward = np.asarray(reward)
                        reward_dict[var].append(reward)
                
        reward_dict = dict(sorted(reward_dict.items()))

        # Creation of the subplot according to monitors
        fig, axis = plt.subplots(4, 2, figsize=(15, 10))
        axis[0, 0].plot(reward_dict["monitor_0"], label="monitor_0", color='red')
        axis[0, 1].plot(reward_dict["monitor_1"], label="monitor_1", color='green')
        axis[1, 0].plot(reward_dict["monitor_2"], label="monitor_2", color='blue')
        axis[1, 1].plot(reward_dict["monitor_3"], label="monitor_3", color='orange')
        axis[2, 0].plot(reward_dict["monitor_4"], label="monitor_4", color='purple')
        axis[2, 1].plot(reward_dict["monitor_5"], label="monitor_5", color='grey')
        axis[3, 0].plot(reward_dict["monitor_6"], label="monitor_6", color='black')
        axis[3, 1].plot(reward_dict["monitor_7"], label="monitor_7", color='cyan')

        for ax in axis.flat:
            ax.set(xlabel='Episodes', ylabel='Total Reward')
            ax.legend()
            
        fig.tight_layout()
        plt.show()
    
# Plot the mean and the std evolution of the reward over time
def plot_ppo_reward_mean_std(*folders):
    colors = ["blue", "red", "green"]
    plt.figure()
    num_episode_max = 10_000
    for index_folder, folder in enumerate(folders):
        reward_dict = {}

        for file in os.listdir(folder):
            var = file.split(".")[1] + "_" + file.split(".")[0]
            path = folder + file
            
            reward_dict[var] = []
            
            # Read the reward from log file
            num_episodes = 0
            with open(path, 'r') as monitor:
                for index, line in enumerate(monitor):
                    if index >= 2:
                        reward = float(line.split(",")[0])
                        reward = np.asarray(reward)
                        reward_dict[var].append(reward)
                        num_episodes += 1
                        
            if num_episodes < num_episode_max:
                num_episode_max = num_episodes
                
        reward_dict = dict(sorted(reward_dict.items()))

        reward = []
        for monitor, rewards in reward_dict.items():
            reward.append(rewards[:num_episode_max])
            
        reward = np.asarray(reward)

        mean_reward = np.mean(reward, axis=0)
        std_reward = np.std(reward, axis=0)

        x = np.linspace(0, num_episode_max, len(mean_reward))

        # Creation of plot for each variation of gamma
        gamma = folder.split("_")[-1].split("/")[0]
        plt.plot(x, mean_reward, color=colors[index_folder], label=f"Mean")
        plt.fill_between(x, mean_reward - std_reward, mean_reward + std_reward, color=colors[index_folder], alpha=0.2, label=f"Mean +- Std")
        plt.legend()
    plt.show()
    
if __name__ == '__main__':

    # plot_ppo_reward_simulation("logs/ppo_car_racing/")
    plot_ppo_reward_mean_std("logs/training/ppo_hard_road/")
    plot_ppo_reward_mean_std("logs/training/ppo_dirt_road/")
    plot_ppo_reward_mean_std("logs/training/ppo_snow_road/")
    # plot_ppo_reward_mean_std("saves/logs/training/ppo_car_racing/", "logs/training/ppo_normal_road/")