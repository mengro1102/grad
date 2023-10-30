import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.layers import Dense
import numpy as np

class ActorCritic:
    def __init__(self, n_channels=10, learning_rate=0.01):
        self.n_channels = n_channels
        
        self.actor_model = self.build_actor_model()
        self.critic_model = self.build_critic_model()
        
        self.actor_optimizer = tf.keras.optimizers.Adam(learning_rate)
        self.critic_optimizer = tf.keras.optimizers.Adam(learning_rate)
    
    def build_actor_model(self):
        model = keras.Sequential()
        model.add(Dense(24, input_dim=self.n_channels, activation='relu'))
        model.add(Dense(self.n_channels, activation='softmax'))
        return model
    
    def build_critic_model(self):
        model = keras.Sequential()
        model.add(Dense(24, input_dim=self.n_channels, activation='relu'))
        model.add(Dense(1, activation='linear'))
        return model
    
    def choose_action(self, state):
        probabilities = self.actor_model.predict(np.expand_dims(state, axis=0))
        action = np.random.choice(self.n_channels, p=probabilities[0])
        return action

    def train(self, state, action, reward, next_state):
        with tf.GradientTape() as actor_tape, tf.GradientTape() as critic_tape:
            action_probs = self.actor_model(np.expand_dims(state, axis=0))
            action_prob = action_probs[0, action]

            value = self.critic_model(np.expand_dims(state, axis=0))
            next_value = self.critic_model(np.expand_dims(next_state, axis=0))
            
            td_error = reward + next_value - value
            actor_loss = -tf.math.log(action_prob) * td_error
            critic_loss = td_error**2
            
        actor_grads = actor_tape.gradient(actor_loss, self.actor_model.trainable_variables)
        critic_grads = critic_tape.gradient(critic_loss, self.critic_model.trainable_variables)
        
        self.actor_optimizer.apply_gradients(zip(actor_grads, self.actor_model.trainable_variables))
        self.critic_optimizer.apply_gradients(zip(critic_grads, self.critic_model.trainable_variables))
