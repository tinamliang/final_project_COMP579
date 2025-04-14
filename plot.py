import matplotlib.pyplot as plt
import numpy as np
import pickle

# def load_rewards(filename):
#     with open(filename, "rb") as f:
#         return pickle.load(f)

# reward = load_rewards("../discrete_DQN/final_project_COMP579/cont_eps_dqn_car_racing.pkl")
# print(reward)

# plt.figure(figsize=(8, 5))
# plt.plot(range(len(reward)), reward, 'r-')
# plt.xlabel('Step', fontsize=16)
# plt.ylabel('AvgReturn', fontsize=16)
# plt.xticks(fontsize=14)
# plt.yticks(fontsize=14)
# plt.grid(axis='y')
# plt.show()

import re

def plot_running_avg(totalrewards):
  N = len(totalrewards)
  running_avg = np.empty(N)
  for t in range(N):
    running_avg[t] = np.mean(totalrewards[max(0, t-10):(t+1)])
  return running_avg

total_rewards = []
avg_rewards = []

with open("results/rewards.txt", "r") as f:
    for line in f:
        total_match = re.search(r"total reward:\s*(-?\d+\.?\d*)", line)
        avg_match = re.search(r"avg reward \(last 100\):\s*(-?\d+\.?\d*)", line)
        if total_match and avg_match:
            total_rewards.append(float(total_match.group(1)))
            avg_rewards.append(float(avg_match.group(1)))

print("Total Rewards:", total_rewards)
print("Average Rewards:", avg_rewards)

avg_reward = plot_running_avg(avg_rewards)

plt.figure(figsize=(8, 5))
plt.plot(range(len(total_rewards)), total_rewards, color='orchid', alpha=0.4)
plt.plot(range(len(avg_reward)), avg_reward, color='darkslategray', label='Avg Reward')
plt.xlabel('Episodes', fontsize=16)
plt.ylabel('Reward', fontsize=16)
plt.xticks(fontsize=14)
plt.yticks(fontsize=14)
plt.grid(axis='y')
plt.savefig('results/total_rewards_continuous_starting_pt.png')