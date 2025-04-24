## COMP 579 final project : Group 3 (Tina Liang, Alexander Wang, Minos Papadopoulos)

### Investigating Deep Q-Learning with Different Exploration Strategies for Autonomous Driving Environments

In this project, we refer from the paper published by (Rodrigues and Vieira, "Optimizing agent training with deep q-learning on a self-
driving reinforcement learning environment", 2020) that trains a DQN agent to optimally learn and navigate OpenAI's CarRacing-v2 environment autonomously.

We train the agent with some environment-specific adaptations and explore the agent's performance with more advanced RL strategies such as NoisyDQN and Random Network Distillation.

Our baseline implementations were based off of [Luc Prieur's](https://gist.github.com/lmclupr/b35c89b2f8f81b443166e88b787b03ab#file-race-car-cv2-nn-network-td0-15-possible-actions-ipynb) work. All results like plots, test videos and rewards are found in the results folder. 

To run our code with environment-specific adaptations:
- Run the script: race_car_run.py

To run advanced RL algorithms you can: 
- Run batch job with script: launch.py
  OR
- Run individual scripts : rnd.py, noisynet_dqn.py and lmcdqn.py
