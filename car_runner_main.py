# pekaalto implementation

"""
Python 3.5, tensorflow 1.0.0

Trains first for train_episodes amount of episodes
and then starts playing with the best known policy with no exploration

Optionally save checkpoints to checkpoint_path (set checkpoint_path=None to not save anything)
Experience history is never saved

Training and playing can be early stopped by giving input (pressing enter in console)
"""

from dqn.agent import CarRacingDQN
import os
import tensorflow as tf
import numpy as np
import gym
import matplotlib.pyplot as plt
import _thread
import pickle
from tensorflow.python.keras.models import load_model
import re
import sys

# SETTINGS
load_checkpoint = False
checkpoint_path = "/Users/tinaliang/Documents/COMP579/continuous_DQN/data/checkpoint02"
train_episodes = 8000 #or just give higher value to train the existing checkpoint more

reward_list = []

model_config = dict(
    min_epsilon=0.1,
    max_negative_rewards=12,
    min_experience_size=int(1e4),
    num_frame_stack=3,
    frame_skip=3,
    train_freq=4,
    batchsize=64,
    epsilon_decay_steps=int(1e5),
    network_update_freq=int(1e3),
    experience_capacity=int(4e4),
    gamma=0.95
)

print(model_config)
########

plot_history = None
env_name = "CarRacing-v2"
env = gym.make(env_name, render_mode="rgb_array")

# tf.reset_default_graph()
dqn_agent = CarRacingDQN(env=env, **model_config)
dqn_agent.build_graph()

sess = tf.compat.v1.InteractiveSession()
dqn_agent.session = sess

saver = tf.compat.v1.train.Saver(max_to_keep=100)
tf.compat.v1.global_variables_initializer().run()

def save_results():

    filename = f"car_racing_dqn_{dqn_agent.global_counter}.pkl"
    with open(filename, "wb") as f:
        pickle.dump(history, f)
    print(f"Training complete. car_racing_dqn_{dqn_agent.global_counter}.pkl")

    if checkpoint_path:
        saver.save(sess, '/Users/tinaliang/Documents/COMP579/continuous_DQN/data/checkpoint02/model.ckpt', global_step=dqn_agent.global_counter)
        print(f"Checkpoint saved at step {dqn_agent.global_counter}")

def one_episode():
    reward, frames = dqn_agent.play_episode()
    print("episode: %d, reward: %f, length: %d, total steps: %d" %
          (dqn_agent.episode_counter, reward, frames, dqn_agent.global_counter))
    return reward
        
def input_thread(list):
    input("...enter to stop after current episode\n")
    list.append("OK")

history = []
def main_loop():
    """
    This just calls training function
    as long as we get input to stop
    """
    list = []
    _thread.start_new_thread(input_thread, (list,))
    while True:
        if list:
            break
        
        if dqn_agent.do_training and dqn_agent.episode_counter > train_episodes:
            print(f'{dqn_agent.episode_counter} eps reached, done training!')
            save_results()
            ret = evaluate_loop()
            history.append(ret)
            print(f'evaluation reward at the end: {ret}')
            break

        if dqn_agent.episode_counter % 40 == 0:
            ret = evaluate_loop()
            history.append(ret)
            print(f'evaluation reward: {ret}')

        if dqn_agent.episode_counter % 1000 == 0:

            save_results()

            plt.figure(figsize=(8, 5))
            plt.plot(range(len(history)), history, 'r-')
            plt.xlabel('Step', fontsize=16)
            plt.ylabel('AvgReturn', fontsize=16)
            plt.xticks(fontsize=14)
            plt.yticks(fontsize=14)
            plt.grid(axis='y')
            plt.savefig(f"plot_{dqn_agent.episode_counter}_car_racing_dqn_car_racing.png")

        one_episode()

    print("done")

def evaluate_loop():
    dqn_agent.do_training = False
    score = 0
    for _ in range(10):
        reward, _ = dqn_agent.play_episode()
        score += reward
    dqn_agent.do_training = True
    return np.round(score / 10, 4)

if train_episodes > 0:
    print("now training... you can early stop with enter...")
    print("##########")
    main_loop()
  