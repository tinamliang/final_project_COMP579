import gym
import os
import numpy as np
import matplotlib.pyplot as plt
from gym import wrappers
from datetime import datetime
import random
import pickle
from tensorflow import keras
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Activation, Conv2D, Flatten
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.saving import load_model
import tensorflow as tf
import cv2
from collections import deque

np.float_ = np.float64
np.bool8 = np.bool_

action_space = [
    [-1, 0, 0],
    [1, 0, 0],
    [0, 1, 0],
    [0, 0, 0.5],
    [0, 0, 0]
]
gas_actions = np.array([a[1] == 1 and a[2] == 0 for a in action_space])

def plot_running_avg(totalrewards):
    N = len(totalrewards)
    running_avg = np.empty(N)
    for t in range(N):
        running_avg[t] = totalrewards[max(0, t-100):(t+1)].mean()
    plt.plot(running_avg)
    plt.title("Running Average")
    plt.savefig('results/rnd_data/run_avg_rnd_dqn.png')
    plt.close()

env = gym.make('CarRacing-v2')

def transform(s):
    bottom_black_bar = s[84:, 12:]
    img = cv2.cvtColor(bottom_black_bar, cv2.COLOR_RGB2GRAY)
    bottom_black_bar_bw = cv2.threshold(img, 1, 255, cv2.THRESH_BINARY)[1]
    bottom_black_bar_bw = cv2.resize(bottom_black_bar_bw, (84, 12), interpolation=cv2.INTER_NEAREST)

    upper_field = s[:84, 6:90]
    img = cv2.cvtColor(upper_field, cv2.COLOR_RGB2GRAY)
    upper_field_bw = cv2.threshold(img, 120, 255, cv2.THRESH_BINARY)[1]
    upper_field_bw = cv2.resize(upper_field_bw, (10, 10), interpolation=cv2.INTER_NEAREST)
    upper_field_bw = upper_field_bw.astype('float') / 255

    car_field = s[66:78, 43:53]
    img = cv2.cvtColor(car_field, cv2.COLOR_RGB2GRAY)
    car_field_bw = cv2.threshold(img, 80, 255, cv2.THRESH_BINARY)[1]
    car_field_t = [car_field_bw[:, 3].mean() / 255, car_field_bw[:, 4].mean() / 255,
                   car_field_bw[:, 5].mean() / 255, car_field_bw[:, 6].mean() / 255]

    return bottom_black_bar_bw, upper_field_bw, car_field_t

def compute_steering_speed_gyro_abs(a):
    right_steering = a[6, 36:46].mean() / 255
    left_steering = a[6, 26:36].mean() / 255
    steering = (right_steering - left_steering + 1.0) / 2

    left_gyro = a[6, 46:60].mean() / 255
    right_gyro = a[6, 60:76].mean() / 255
    gyro = (right_gyro - left_gyro + 1.0) / 2

    speed = a[:, 0][:-2].mean() / 255
    abs1 = a[:, 6][:-2].mean() / 255
    abs2 = a[:, 8][:-2].mean() / 255
    abs3 = a[:, 10][:-2].mean() / 255
    abs4 = a[:, 12][:-2].mean() / 255

    return [steering, speed, gyro, abs1, abs2, abs3, abs4]

vector_size = 10 * 10 + 7 + 4

class ReplayBuffer:
    def __init__(self, capacity):
        self.buffer = deque(maxlen=capacity)

    def push(self, state, action, reward, next_state, done):
        self.buffer.append((state, action, reward, next_state, done))

    def sample(self, batch_size):
        batch = random.sample(self.buffer, batch_size)
        states = np.array([x[0] for x in batch])
        actions = np.array([x[1] for x in batch])
        rewards = np.array([x[2] for x in batch])
        next_states = np.array([x[3] for x in batch])
        dones = np.array([x[4] for x in batch])
        return states, actions, rewards, next_states, dones

    def __len__(self):
        return len(self.buffer)

def create_rnd_network():
    model = Sequential([
        Dense(256, input_shape=(vector_size,), activation='relu'),
        Dense(128, activation='relu'),
        Dense(64, activation='linear')  # out dim for RND
    ])
    return model

def create_q_network():
    model_path = '/network/scratch/g/guangyuan.wang/comp579_final_proj/results/rnd_data/rnd_dqn_model.keras'
    if os.path.exists(model_path):
        print("loading Q-network...")
        return load_model(model_path)

    model = Sequential([
        Dense(256, input_shape=(vector_size,), activation='relu'),
        Dense(128, activation='relu'),
        Dense(5, activation='linear')  # 5 actions
    ])
    adam = Adam(learning_rate=0.0005)  # default alpha = 0.0005
    model.compile(loss='mse', optimizer=adam)
    model.summary()
    return model

# RND model class
class Model:
    def __init__(self, env, buffer_size=200000, batch_size=64, gamma=0.99, tau=0.001, beta=0.01):
        self.env = env
        self.gamma = gamma
        self.batch_size = batch_size
        self.tau = tau
        self.beta = beta  # intrinsic reward scale
        self.memory = ReplayBuffer(buffer_size)
        self.model = create_q_network()
        self.target_model = create_q_network()
        self.target_model.set_weights(self.model.get_weights())
        self.rnd_target = create_rnd_network()  # fixed random network
        self.rnd_predictor = create_rnd_network()  # trainable predictor
        self.rnd_optimizer = Adam(learning_rate=0.0001)  # default alpha = 0.0001 for predictor TODO
        self.rnd_std = deque(maxlen=1000)  # for intrinsic reward normalization
        self.epsilon = 0.05  # default epsilon = 0.05  # TODO : tune this

    def predict(self, s, target=False):
        model = self.target_model if target else self.model
        return model.predict(s.reshape(-1, vector_size), verbose=0)[0]

    def compute_intrinsic_reward(self, state):
        state_tensor = tf.convert_to_tensor(state.reshape(-1, vector_size), dtype=tf.float32)
        with tf.GradientTape() as tape:
            pred = self.rnd_predictor(state_tensor)
            target = self.rnd_target(state_tensor)
            loss = tf.reduce_mean(tf.square(pred - target))
        grads = tape.gradient(loss, self.rnd_predictor.trainable_variables)
        self.rnd_optimizer.apply_gradients(zip(grads, self.rnd_predictor.trainable_variables))
        intrinsic_reward = loss.numpy()
        self.rnd_std.append(intrinsic_reward)
        std = np.std(self.rnd_std) if len(self.rnd_std) > 10 else 1.0
        return intrinsic_reward / max(std, 1e-8)  # normalize

    def update(self, states, targets):
        self.model.fit(states, targets, epochs=1, verbose=0)

    def sample_action(self, s):
        if np.random.random() < self.epsilon:
            action_weights = 14.0 * gas_actions + 1.0  # action weighting from earlier code
            action_weights /= np.sum(action_weights)
            return np.random.choice(len(action_space), p=action_weights), self.predict(s)
        qval = self.predict(s)
        return np.argmax(qval), qval

    def learn(self):
        if len(self.memory) < self.batch_size:
            return
        states, actions, rewards, next_states, dones = self.memory.sample(self.batch_size)
        targets = self.model.predict(states, verbose=0)
        next_qvals = self.target_model.predict(next_states, verbose=0)
        for i in range(self.batch_size):
            intrinsic_reward = self.compute_intrinsic_reward(next_states[i])
            total_reward = np.clip(rewards[i] + self.beta * intrinsic_reward, -1, 1)  # reward clipping
            target = total_reward + (1 - dones[i]) * self.gamma * np.max(next_qvals[i])
            targets[i, actions[i]] = target
        self.update(states, targets)
        self.soft_update()

    def soft_update(self):
        target_weights = self.target_model.get_weights()
        model_weights = self.model.get_weights()
        for i in range(len(target_weights)):
            target_weights[i] = self.tau * model_weights[i] + (1 - self.tau) * target_weights[i]
        self.target_model.set_weights(target_weights)

def play_one(env, model, gamma):
    (observation, _) = env.reset()
    done = False
    totalreward = 0
    iters = 0
    while not done:
        a, b, c = transform(observation)
        state = np.concatenate((np.array([compute_steering_speed_gyro_abs(a)]).reshape(1, -1).flatten(),
                                b.reshape(1, -1).flatten(), c), axis=0)
        argmax_qval, qval = model.sample_action(state)
        action = action_space[argmax_qval]
        observation, reward, terminated, truncated, _ = env.step(action)
        done = terminated or truncated

        a, b, c = transform(observation)
        next_state = np.concatenate((np.array([compute_steering_speed_gyro_abs(a)]).reshape(1, -1).flatten(),
                                     b.reshape(1, -1).flatten(), c), axis=0)

        model.memory.push(state, argmax_qval, reward, next_state, done)
        model.learn()

        totalreward += reward
        iters += 1
        if iters > 1500:
            print("This episode is stuck")
            break
    return totalreward, iters

# train loop
model = Model(env)
gamma = 0.99
N = 1000
totalrewards = np.empty(N)
intrinsic_rewards = np.empty(N)  # track intrinsic rewards for early stopping
for n in range(N):
    totalreward, iters = play_one(env, model, gamma)
    totalrewards[n] = totalreward
    intrinsic_rewards[n] = np.mean(model.rnd_std) if model.rnd_std else 0
    if n % 1 == 0:
        print(f"episode: {n}, iters: {iters}, total reward: {totalreward}, "
              f"avg reward (last 100): {totalrewards[max(0, n-100):(n+1)].mean()}, "
              f"avg intrinsic reward: {intrinsic_rewards[max(0, n-100):(n+1)].mean()}")
    if n % 50 == 0:
        print("saving model...")
        model.model.save('/network/scratch/g/guangyuan.wang/comp579_final_proj/results/rnd_data/rnd_dqn_model.keras')
        filename = "results/rnd_data/rnd_dqn.pkl"
        with open(filename, "wb") as f:
            pickle.dump(totalrewards, f)
        print(f"wrote to pkl file: {filename}")
    # early stopping if intrinsic rewards plateau
    if n > 100 and np.std(intrinsic_rewards[max(0, n-100):n]) < 0.01:
        print("Intrinsic rewards plateaued, stopping training")
        break

plt.plot(totalrewards)
plt.title("Rewards")
plt.savefig('results/rnd_data/rnd_dqn.png')
plt.close()
plot_running_avg(totalrewards)

model.model.save('/network/scratch/g/guangyuan.wang/comp579_final_proj/results/rnd_data/rnd_dqn_model.keras')
env.close()

# eval (10 episodes)
eval_env = gym.make('CarRacing-v2', render_mode='rgb_array')
model = Model(eval_env)
print('starting evaluation...')
eval_rewards = []
for ep in range(10):
    frames = []
    (s, _) = eval_env.reset()
    done = False
    totalreward = 0
    while not done:
        frames.append(eval_env.render())
        a, b, c = transform(s)
        state = np.concatenate((np.array([compute_steering_speed_gyro_abs(a)]).reshape(1, -1).flatten(),
                                b.reshape(1, -1).flatten(), c), axis=0)
        argmax_qval, qval = model.sample_action(state)
        action = action_space[argmax_qval]
        s, reward, terminated, truncated, _ = eval_env.step(action)
        totalreward += reward
        done = terminated or truncated
    eval_rewards.append(totalreward)
    print(f"eval episode {ep+1}, score: {totalreward}")
    if ep == 0:
        def animate(imgs, video_name):
            import cv2
            import string
            import random
            video_name = video_name or ''.join(random.choice(string.ascii_letters) for i in range(18)) + '.webm'
            height, width, _ = imgs[0].shape
            fourcc = cv2.VideoWriter_fourcc(*'VP90')
            video = cv2.VideoWriter(video_name, fourcc, 10, (width, height))
            for img in imgs:
                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                video.write(img)
            video.release()
        animate(frames, "results/rnd_data/rnd_dqn_eval.webm")
print(f"average eval score: {np.mean(eval_rewards)}")
eval_env.close()