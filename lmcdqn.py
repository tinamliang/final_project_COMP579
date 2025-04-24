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
from tensorflow.keras.layers import Dense, Activation
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.saving import load_model
import tensorflow as tf
import cv2
from collections import deque

np.float_ = np.float64
np.bool8 = np.bool_

# action space
action_space = [
    [-1, 0, 0],
    [1, 0, 0],
    [0, 1, 0],
    [0, 0, 0.5],
    [0, 0, 0]
]
gas_actions = np.array([a[1] == 1 and a[2] == 0 for a in action_space])

# plotting function
def plot_running_avg(totalrewards):
    n = len(totalrewards)
    running_avg = np.empty(n)
    for t in range(n):
        running_avg[t] = totalrewards[max(0, t-100):(t+1)].mean()
    plt.plot(running_avg)
    plt.title("running average")
    plt.savefig('results/lmcdqn_data/run_avg_lmcdqn.png')
    plt.close()

# environment
env = gym.make('CarRacing-v2')

# preprocessing functions
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

# replay buffer
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

# q-network (standard dqn, no noise)
def create_nn():
    model_path = '/network/scratch/g/guangyuan.wang/comp579_final_proj/results/lmcdqn_data/lmcdqn_model.keras'
    if os.path.exists(model_path):
        print("loading model...")
        return load_model(model_path)

    model = Sequential([
        Dense(256, input_shape=(vector_size,), activation='relu'),
        Dense(128, activation='relu'),
        Dense(5, activation='linear')  # 5 actions
    ])
    adam = Adam(learning_rate=0.00025)  # alpha = 0.00025
    model.compile(loss='mse', optimizer=adam)
    model.summary()
    return model

# model class with lmcdqn
class Model:
    def __init__(self, env, buffer_size=100000, batch_size=64, gamma=0.99, lambda_reg=0.1):
        self.env = env
        self.gamma = gamma
        self.batch_size = batch_size
        self.lambda_reg = lambda_reg  # regularization strength
        self.memory = ReplayBuffer(buffer_size)
        self.model = create_nn()  # q-network
        self.target_model = create_nn()  # target network
        self.target_model.set_weights(self.model.get_weights())
        self.epsilon = 0.2  # methodology: epsilon = 0.2
        self.epsilon_min = 0.01
        self.epsilon_decay = (self.epsilon - self.epsilon_min) / 300  # decay over 300 episodes
        self.update_counter = 0  # for target network updates
        self.loss_history = deque(maxlen=100)  # for dynamic lambda

    def predict(self, s, target=False):
        model = self.target_model if target else self.model
        return model.predict(s.reshape(-1, vector_size), verbose=0)[0]

    # new loss with \lambda \cdot R(\theta)
    def compute_regularization_loss(self, states, targets):
        # gradient norm penalty to escape local minima
        with tf.GradientTape() as tape:
            predictions = self.model(states, training=True)
            mse_loss = tf.reduce_mean(tf.square(predictions - targets))
        grads = tape.gradient(mse_loss, self.model.trainable_variables)
        grad_norm = tf.sqrt(sum(tf.reduce_sum(tf.square(g)) for g in grads if g is not None))
        return grad_norm

    def update(self, states, targets):
        # custom training with regularization
        states_tensor = tf.convert_to_tensor(states, dtype=tf.float32)
        targets_tensor = tf.convert_to_tensor(targets, dtype=tf.float32)
        with tf.GradientTape() as tape:
            predictions = self.model(states_tensor, training=True)
            mse_loss = tf.reduce_mean(tf.square(predictions - targets_tensor))
            reg_loss = self.compute_regularization_loss(states_tensor, targets_tensor)
            total_loss = mse_loss + self.lambda_reg * reg_loss
        grads = tape.gradient(total_loss, self.model.trainable_variables)
        self.model.optimizer.apply_gradients(zip(grads, self.model.trainable_variables))
        self.loss_history.append(mse_loss.numpy())
        # dynamic lambda adjustment
        if len(self.loss_history) >= 100:
            loss_std = np.std(self.loss_history)
            self.lambda_reg = max(0.01, min(0.1, self.lambda_reg * (1 - 0.1 * loss_std)))

    def sample_action(self, s, episode):
        # epsilon-greedy with action weighting for first 100 episodes
        if episode < 100 and np.random.random() < self.epsilon:
            action_weights = 14.0 * gas_actions + 1.0  # 85% acceleration preference
            action_weights /= np.sum(action_weights)
            return np.random.choice(len(action_space), p=action_weights), self.predict(s)
        elif np.random.random() < self.epsilon:
            return np.random.choice(len(action_space)), self.predict(s)
        qval = self.predict(s)
        return np.argmax(qval), qval

    def learn(self):
        if len(self.memory) < self.batch_size:
            return
        states, actions, rewards, next_states, dones = self.memory.sample(self.batch_size)
        targets = self.model.predict(states, verbose=0)
        next_qvals = self.target_model.predict(next_states, verbose=0)
        for i in range(self.batch_size):
            total_reward = np.clip(rewards[i], -1, 1)  # reward clipping
            target = total_reward + (1 - dones[i]) * self.gamma * np.max(next_qvals[i])
            targets[i, actions[i]] = target
        self.update(states, targets)
        self.update_counter += 1
        # hard target update every 1000 steps
        if self.update_counter % 1000 == 0:
            self.target_model.set_weights(self.model.get_weights())

# play one episode
def play_one(env, model, gamma, episode):
    (observation, _) = env.reset()
    done = False
    totalreward = 0
    iters = 0
    while not done:
        a, b, c = transform(observation)
        state = np.concatenate((np.array([compute_steering_speed_gyro_abs(a)]).reshape(1, -1).flatten(),
                                b.reshape(1, -1).flatten(), c), axis=0)
        argmax_qval, qval = model.sample_action(state, episode)
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
            print("this episode is stuck")
            break
    model.epsilon = max(model.epsilon_min, model.epsilon - model.epsilon_decay)
    return totalreward, iters

# training loop
model = Model(env)
gamma = 0.99
n = 1000
totalrewards = np.empty(n)
for episode in range(n):
    totalreward, iters = play_one(env, model, gamma, episode)
    totalrewards[episode] = totalreward
    if episode % 1 == 0:
        print(f"episode: {episode}, iters: {iters}, total reward: {totalreward}, "
              f"avg reward (last 100): {totalrewards[max(0, episode-100):(episode+1)].mean()}, "
              f"epsilon: {model.epsilon:.3f}, lambda: {model.lambda_reg:.3f}")
    if episode % 50 == 0:
        print("saving model...")
        model.model.save('/network/scratch/g/guangyuan.wang/comp579_final_proj/results/lmcdqn_data/lmcdqn_model.keras')
        filename = "results/lmcdqn_data/lmcdqn.pkl"
        with open(filename, "wb") as f:
            pickle.dump(totalrewards, f)
        print(f"wrote to pkl file: {filename}")

plt.plot(totalrewards)
plt.title("rewards")
plt.savefig('results/lmcdqn_data/lmcdqn.png')
plt.close()
plot_running_avg(totalrewards)

model.model.save('/network/scratch/g/guangyuan.wang/comp579_final_proj/results/lmcdqn_data/lmcdqn_model.keras')
env.close()

# evaluation (10 episodes)
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
        argmax_qval, qval = model.sample_action(state, episode=1000)  # use greedy policy
        action = action_space[argmax_qval]
        s, reward, terminated, truncated, _ = eval_env.step(action)
        totalreward += reward
        done = terminated or truncated
    eval_rewards.append(totalreward)
    print(f"eval episode {ep+1}, score: {totalreward}")
    # save video for first episode
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
        animate(frames, "results/lmcdqn_data/lmcdqn_eval.webm")
print(f"average eval score: {np.mean(eval_rewards)}")
eval_env.close()