import gym
import os
import sys
import numpy as np
import matplotlib.pyplot as plt
from gym import wrappers
from datetime import datetime
import random
import pickle
from tensorflow import keras
from tf_keras.models import Sequential
from tf_keras.layers import Dense, Activation
from tf_keras.optimizers.legacy import Adamax
from tf_keras.saving import load_model
import cv2

np.float_ = np.float64
np.bool8 = np.bool_

action_space = [
    [-1, 0, 0], 
    [1,0,0], 
    [0,1,0],
    [0,0,0.5],
    [0,0,0]]

gas_actions = np.array([a[1] == 1 and a[2] == 0 for a in action_space])

def plot_running_avg(totalrewards):
  N = len(totalrewards)
  running_avg = np.empty(N)
  for t in range(N):
    running_avg[t] = totalrewards[max(0, t-100):(t+1)].mean()
  plt.plot(running_avg)
  plt.title("Running Average")
  plt.savefig('results/run_avg_prioritized_eps_exp.png')
  #plt.show()

env = gym.make('CarRacing-v2')

def transform(s):

    # cv2.imshow('original', s)
    # cv2.waitKey(1)

    # bottom_black_bar is the section of the screen with steering, speed, abs and gyro information.
    # we crop off the digits on the right as they are illigible, even for ml.
    # since color is irrelavent, we grayscale it.

    bottom_black_bar = s[84:, 12:]
    img = cv2.cvtColor(bottom_black_bar, cv2.COLOR_RGB2GRAY)
    bottom_black_bar_bw = cv2.threshold(img, 1, 255, cv2.THRESH_BINARY)[1]
    bottom_black_bar_bw = cv2.resize(bottom_black_bar_bw, (84, 12), interpolation = cv2.INTER_NEAREST)

    # this is the section of the screen that contains the track.
    upper_field = s[:84, 6:90] # we crop side of screen as they carry little information
    img = cv2.cvtColor(upper_field, cv2.COLOR_RGB2GRAY)
    upper_field_bw = cv2.threshold(img, 120, 255, cv2.THRESH_BINARY)[1]
    upper_field_bw = cv2.resize(upper_field_bw, (10, 10), interpolation = cv2.INTER_NEAREST) # re scaled to 7x7 pixels
    # cv2.imshow('video', upper_field_bw)
    # cv2.waitKey(1)
    upper_field_bw = upper_field_bw.astype('float')/255

    car_field = s[66:78, 43:53]
    img = cv2.cvtColor(car_field, cv2.COLOR_RGB2GRAY)
    # cv2.imshow('video', img)
    # cv2.waitKey(1)
    car_field_bw = cv2.threshold(img, 80, 255, cv2.THRESH_BINARY)[1]

    car_field_t = [car_field_bw[:, 3].mean()/255, car_field_bw[:, 4].mean()/255, car_field_bw[:, 5].mean()/255, car_field_bw[:, 6].mean()/255]

#     rotated_image = rotateImage(car_field_bw, 45)
#     cv2.imshow('video rotated', rotated_image)
#     cv2.waitKey(1)

    return bottom_black_bar_bw, upper_field_bw, car_field_t

# this function uses the bottom black bar of the screen and extracts steering setting, speed and gyro data
def compute_steering_speed_gyro_abs(a):
    right_steering = a[6, 36:46].mean()/255
    left_steering = a[6, 26:36].mean()/255
    steering = (right_steering - left_steering + 1.0)/2

    left_gyro = a[6, 46:60].mean()/255
    right_gyro = a[6, 60:76].mean()/255
    gyro = (right_gyro - left_gyro + 1.0)/2

    speed = a[:, 0][:-2].mean()/255
    abs1 = a[:, 6][:-2].mean()/255
    abs2 = a[:, 8][:-2].mean()/255
    abs3 = a[:, 10][:-2].mean()/255
    abs4 = a[:, 12][:-2].mean()/255

#     cv2.imshow('sensors', speed_display)
#     cv2.waitKey(1)


    return [steering, speed, gyro, abs1, abs2, abs3, abs4]

vector_size = 10*10 + 7 + 4

def create_nn():
    #print(os.path.exists('/Users/tinaliang/Documents/COMP579/final_project/results/model.keras'))
    if os.path.exists('/Users/tinaliang/Documents/COMP579/continuous_DQN_v2/code/results/prioritized_eps_exp_model.keras'):
        print("loading model...")
        return load_model('/Users/tinaliang/Documents/COMP579/continuous_DQN_v2/code/results/prioritized_eps_exp_model.keras')

    model = Sequential()
    model.add(Dense(512, kernel_initializer='lecun_uniform', input_shape=(vector_size,))) # 7x7 + 3.  or 14x14 + 3
    model.add(Activation('relu'))

    model.add(Dense(5, kernel_initializer='lecun_uniform'))
    model.add(Activation('linear')) #linear output so we can have range of real-valued outputs

    adamax = Adamax() 
    model.compile(loss='mse', optimizer=adamax)
    model.summary()

    return model

class Model:
    def __init__(self, env):
        self.env = env
        self.model = create_nn()  # one feedforward nn for all actions.

    def predict(self, s):
        return self.model.predict(s.reshape(-1, vector_size), verbose=0)[0]

    def update(self, s, G):
        self.model.fit(s.reshape(-1, vector_size), np.array(G).reshape(-1, 5), epochs=1, verbose=0)

    def sample_action(self, s, eps):
        qval = self.predict(s)
        if np.random.random() < eps:
            action_weights = 14.0 * gas_actions + 1.0
            action_weights /= np.sum(action_weights)

            return np.random.choice(len(action_space), p=action_weights), qval
           # return random.randint(0, len(action_space)-1), qval
        else:
            return np.argmax(qval), qval

def convert_argmax_qval_to_env_action(output_value):
    # we reduce the action space to 15 values.  9 for steering, 6 for gaz/brake.
    # to reduce the action space, gaz and brake cannot be applied at the same time.
    # as well, steering input and gaz/brake cannot be applied at the same time.
    # similarly to real life drive, you brake/accelerate in straight line, you coast while sterring.

    gaz = 0.0
    brake = 0.0
    steering = 0.0

    # output value ranges from 0 to 10

    if output_value <= 8:
        # steering. brake and gaz are zero.
        output_value -= 4
        steering = float(output_value)/4
    elif output_value >= 9 and output_value <= 9:
        output_value -= 8
        gaz = float(output_value)/3 # 33%
    elif output_value >= 10 and output_value <= 10:
        output_value -= 9
        brake = float(output_value)/2 # 50% brakes
    else:
        print("error")

    #white = np.ones((round(brake * 100), 10))
    #black = np.zeros((round(100 - brake * 100), 10))
    #brake_display = np.concatenate((black, white))*255

    #white = np.ones((round(gaz * 100), 10))
    #black = np.zeros((round(100 - gaz * 100), 10))
    #gaz_display = np.concatenate((black, white))*255

    # control_display = np.concatenate((brake_display, gaz_display), axis=1)

    # cv2.imshow('controls', control_display)
    # cv2.waitKey(1)

    return [steering, gaz, brake]

def play_one(env, model, eps, gamma):
    (observation, _) = env.reset()
    done = False
    totalreward = 0
    iters = 0
    while not done:
        a, b, c = transform(observation)
        state = np.concatenate((np.array([compute_steering_speed_gyro_abs(a)]).reshape(1,-1).flatten(), b.reshape(1,-1).flatten(), c), axis=0) # this is 3 + 7*7 size vector.  all scaled in range 0..1
        argmax_qval, qval = model.sample_action(state, eps)
        prev_state = state
        action = action_space[argmax_qval]
        observation, reward, terminated, truncated, _ = env.step(action)

        done = terminated or truncated

        a, b, c = transform(observation)
        state = np.concatenate((np.array([compute_steering_speed_gyro_abs(a)]).reshape(1,-1).flatten(), b.reshape(1,-1).flatten(), c), axis=0) # this is 3 + 7*7 size vector.  all scaled in range 0..1

        # update the model
        # standard Q learning TD(0)
        next_qval = model.predict(state)
        G = reward + gamma*np.max(next_qval)
        y = qval[:]
        y[argmax_qval] = G
        model.update(prev_state, y)
        totalreward += reward
        iters += 1

        if iters > 1500:
            print("This episode is stuck")
            break

    return totalreward, iters

model = Model(env)
gamma = 0.99

N = 550
totalrewards = np.empty(N)
costs = np.empty(N)
for n in range(N):
    eps = 0.5/np.sqrt(n+1 + 900)
    totalreward, iters = play_one(env, model, eps, gamma)
    totalrewards[n] = totalreward
    if n % 1 == 0:
      print("episode:", n, "iters", iters, "total reward:", totalreward, "eps:", eps, "avg reward (last 100):", totalrewards[max(0, n-100):(n+1)].mean())
    if n % 50 == 0:
        print("saving model...")
        model.model.save('/Users/tinaliang/Documents/COMP579/continuous_DQN_v2/code/results/prioritized_eps_exp_model.keras')

        filename = f"results/prioritized_eps_exp.pkl"
        with open(filename, "wb") as f:
            pickle.dump(totalrewards, f)
        print(f"wrote to pkl file. prioritized_eps_exp.pkl")

plt.plot(totalrewards)
plt.title("Rewards")
#plt.show()
plt.savefig('results/prioritized_eps_exp.png')

plot_running_avg(totalrewards)

model.model.save('/Users/tinaliang/Documents/COMP579/continuous_DQN_v2/code/results/prioritized_eps_exp_model.keras')

env.close()

frames = []
eval_env = gym.make('CarRacing-v2', render_mode='rgb_array')
model = Model(eval_env)
print('starting eval episode ....')
(s, _) = eval_env.reset()
done = False
totalreward = 0
scores = 0
while not done:
    frames.append(eval_env.render())
    a, b, c = transform(s)
    state = np.concatenate((np.array([compute_steering_speed_gyro_abs(a)]).reshape(1,-1).flatten(), b.reshape(1,-1).flatten(), c), axis=0) # this is 3 + 7*7 size vector.  all scaled in range 0..1
    argmax_qval, qval = model.sample_action(state, 0)
    prev_state = state
    action = action_space[argmax_qval]
    s, reward, terminated, truncated, _ = eval_env.step(action)
    totalreward += reward
    done = terminated or truncated
scores += totalreward
print('final score: ', scores)
print('finished eval episode')

def animate(imgs, video_name, _return=True):
    import cv2
    import string
    import random

    if video_name is None:
        video_name = ''.join(random.choice(string.ascii_letters) for i in range(18)) + '.webm'
    height, width, _ = imgs[0].shape
    fourcc = cv2.VideoWriter_fourcc(*'VP90')
    video = cv2.VideoWriter(video_name, fourcc, 10, (width, height))

    for img in imgs:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        video.write(img)
    video.release()
    if _return:
        from IPython.display import Video
        return Video(video_name)

print('recording video')
animate(frames, "results/prioritized_eps_exp.webm")
    
