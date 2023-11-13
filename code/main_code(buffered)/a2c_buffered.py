import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import tensorflow_probability as tfp
from tensorflow import keras
from keras import Model
from env import JammingEnv

hidden_size = 24
buffer_size = 5
class Actor(Model):
    def __init__(
        self,
        state_size: int,
        action_size: int, 
    ):
        """Initialization."""
        super(Actor, self).__init__()
        
        self.state_size = state_size
        self.action_size = action_size
        self.layer1 = tf.keras.layers.Dense(hidden_size, activation='relu')
        self.layer2 = tf.keras.layers.Dense(hidden_size, activation='relu')
        self.policy = tf.keras.layers.Dense(self.action_size,activation='softmax')

    def call(self, state):
        x = tf.reshape(state, shape=(-1, self.state_size * buffer_size))
        layer1 = self.layer1(x)
        layer2 = self.layer2(layer1)
        policy = self.policy(layer2)
        return policy
    
class CriticV(Model):
    def __init__(
        self, 
        state_size: int
    ):
        """Initialize."""
        super(CriticV, self).__init__()
        
        self.state_size = state_size
        self.layer1 = tf.keras.layers.Dense(hidden_size, activation='relu')
        self.layer2 = tf.keras.layers.Dense(hidden_size, activation='relu')
        self.value = tf.keras.layers.Dense(1, activation = None)

    def call(self, state):
        layer1 = self.layer1(state)
        layer2 = self.layer2(layer1)
        value = self.value(layer2)
        return value
    
class StateBuffer:
    def __init__(self, buffer_size, state_size=5):
        self.buffer_size = buffer_size
        self.state_size = state_size
        self.buffer = np.zeros((buffer_size, state_size))

    def add_state(self, state):
        # 새 상태를 버퍼에 추가하고 가장 오래된 상태를 삭제
        self.buffer = np.roll(self.buffer, shift=1, axis=0)
        self.buffer[0,:] = state[:self.state_size]

    def get_buffer(self):
        return self.buffer

    def is_full(self):
        return len(self.buffer) == self.buffer_size
    
class ActorCritic():
    def __init__(self, state_size = 5, action_size = 5):
        self.state_size = state_size
        self.action_size = action_size
        self.actor_lr = 5e-3
        self.critic_lr = 5e-3
        self.gamma = 0.99    # discount rate
        self.actor = Actor(self.state_size, self.action_size)
        self.critic = CriticV(self.state_size)
       
        self.a_opt = tf.keras.optimizers.Adam(learning_rate=self.actor_lr)
        self.c_opt = tf.keras.optimizers.Adam(learning_rate=self.critic_lr)
        self.log_prob = None
        
    def get_action(self, buffer):
        prob = self.actor(np.array(buffer))
        # print('action prob',prob.numpy()[0]) 
        dist = tfp.distributions.Categorical(probs=prob, dtype=tf.float32)
        action = dist.sample()
        return int(action.numpy()[0]) 
    
    def actor_loss(self, prob, action, TD):
        dist = tfp.distributions.Categorical(probs=prob, dtype=tf.float32)
        log_prob = dist.log_prob(action)
        loss = -log_prob*TD
        return loss
    
    def train_step(self, buffer, action, reward, next_state, done):
        next_state = np.array([next_state])
       
        with tf.GradientTape() as tape1, tf.GradientTape() as tape2:
            curr_P = self.actor(buffer, training=True)
            curr_Q = self.critic(buffer,training=True)
            next_Q = self.critic(next_state, training=True)
            expected_Q = reward + self.gamma*next_Q*(1-int(done))
            TD = expected_Q - curr_Q
            critic_loss = tf.keras.losses.MSE(expected_Q, curr_Q)
            actor_loss = self.actor_loss(curr_P, action, TD)
            
        actorGrads = tape1.gradient(actor_loss,  self.actor.trainable_variables)
        criticGrads = tape2.gradient(critic_loss, self.critic.trainable_variables)
        self.a_opt.apply_gradients(zip(actorGrads, self.actor.trainable_variables))
        self.c_opt.apply_gradients(zip(criticGrads, self.critic.trainable_variables))
        
        return actor_loss, critic_loss
    
    def save_model(self, actor_path, critic_path):
        self.actor.save(actor_path)
        self.critic.save(critic_path)
        
    def load_model(self, actor_path, critic_path):
        self.actor = tf.keras.models.load_model(actor_path)
        self.critic = tf.keras.models.load_model(critic_path)