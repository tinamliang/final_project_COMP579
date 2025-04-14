import matplotlib.pyplot as plt
import numpy as np
import pickle

def load_rewards(filename):
    with open(filename, "rb") as f:
        return pickle.load(f)

reward = load_rewards("car_racing_dqn_640431.pkl")
print(reward)

def plot_running_avg(totalrewards):
  N = len(totalrewards)
  running_avg = np.empty(N)
  for t in range(N):
    running_avg[t] = np.mean(totalrewards[max(0, t-10):(t+1)])
  return running_avg

avg_reward = plot_running_avg(reward)

plt.figure(figsize=(8, 5))
plt.plot(range(len(reward)), reward, color='orchid', alpha=0.5)
plt.plot(range(len(avg_reward)), avg_reward, color='darkslategrey', alpha=0.8)
plt.xlabel('Step', fontsize=16)
plt.ylabel('AvgReturn', fontsize=16)
plt.xticks(fontsize=14)
plt.yticks(fontsize=14)
plt.grid(axis='y')
plt.savefig(f"plot_car_racing_dqn.png")