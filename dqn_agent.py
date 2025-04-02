import tensorflow as tf
from tensorflow.python.keras import Adam
from replay_buffer import ReplayBuffer
import numpy as np
import torch
import random

action_space = [
    (-1, 0, 0), # turn left
    (1, 0, 0),  # turn right
    (0, 1, 0), # accelerate
    (0, 0, 0.5),  # brake
    (0, 0, 0) # do nothing
]

class DQN():
    def __init__(self, state_dim, action_dim):
        self.action_space = action_dim
        self.gamma = 0.95
        self.initial_epsilon = 1.0
        self.min_epsilon = 0.1
        self.model = self.build_model()
        self.target_model = self.build_model()
        self.buffer = ReplayBuffer(state_dim, (1,), max_size=100000)
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        self.model.to(self.device)
        self.target_model.to(self.device)

        self.epsilon_decay = (self.initial_epsilon - self.min_epsilon) / 1e6

    def build_model(self):
        model = tf.keras.models.Sequential([
            tf.keras.layers.Conv2D(filters=8, kernel_size=(7, 7), strides=3, activation='relu', input_shape=(96, 96, 3)),
            tf.keras.layers.MaxPooling2D(pool_size=(2, 2), strides=2),
            tf.keras.layers.Conv2D(filters=16, kernel_size=(3, 3), strides=1, activation='relu'),
            tf.keras.layers.MaxPooling2D(pool_size=(2, 2), strides=2),
            tf.keras.layers.Flatten(),
            tf.keras.layers.Dense(400, activation='relu'),
            tf.keras.Dense(len(self.action_space), activation=None)
        ])

        model.compile(optimizer=Adam(learning_rate=0.001, epsilon=1e-7), loss='mse')
        return model
    
    def update_target_model(self):
        self.target_model.set_weights(self.model.get_weights())

    def save_model(self, path):
        self.target_model.save_weights(path)

