import numpy as np

class JammingEnv:
    def __init__(self, n_channels=5, max_steps=200):
        self.n_channels = n_channels
        self.state = np.zeros(n_channels)  # Initial state with all channels free
        # self.last_action = last_action
        self.jammer_channel = np.random.randint(0, self.n_channels)  # Initialize with a random channel
        self.sweep_direction = np.random.choice([-1, 1])  # Choose an initial sweep direction (-1: down, 1: up)
        self.prev_action = None
        self.action_count = 0
        self.current_step = 0
        self.max_steps = max_steps
        self.observation_space = []
        
    def reset(self):
        """Reset the environment to its initial state."""
        self.state = np.zeros(self.n_channels)
        # self.current_step = 0
        
        return self.state
   
    def step(self, action, time_step):
        reward = 0
        self.reset()
        
        if time_step >= self.max_steps: # timestep이 Episode 마지막에 도달할 때
            done = True
            ''' Sweeping '''
            self.jammer_channel += self.sweep_direction
            if self.jammer_channel >= self.n_channels:
                self.jammer_channel = 0
            elif self.jammer_channel < 0:
                self.jammer_channel = self.n_channels - 1

            self.state[self.jammer_channel] = 1  # Jam the selected channel
        else: # timestep이 에피소드 종료 직전 상황일 때 
            done = False
            if time_step >= 100: # timestep의 절반에 도달했을 땐 Sweeping 재밍 전략 
                ''' Sweeping '''
                self.jammer_channel += self.sweep_direction
                if self.jammer_channel >= self.n_channels:
                    self.jammer_channel = 0
                elif self.jammer_channel < 0:
                    self.jammer_channel = self.n_channels - 1

                self.state[self.jammer_channel] = 1  # Jam the selected channel 
            else: # timestep의 절반에 도달하기 전에는 Random 재밍 전략
                ''' Random '''
                self.jammer_channel = np.random.randint(0, self.n_channels)  # Randomly select a channel to jam
                self.state[self.jammer_channel] = 1  # Jam the selected channel

        if self.state[action] == 1:  # Jammed
            reward = -self.n_channels # -self.n_channels
        else:
            reward = 1
        
        next_state = self.state.copy()
        self.observation_space = next_state
        
        return next_state, reward, done
        
    def render(self):
        """Visualize the current state of the environment."""
        print(self.state)

""" Jammer type = Random
self.jammer_channel = np.random.randint(0, self.n_channels)  # Randomly select a channel to jam
self.state[self.jammer_channel] = 1  # Jam the selected channel
"""
""" Jammer type = Sweeping 
self.jammer_channel += self.sweep_direction
if self.jammer_channel >= self.n_channels:
    self.jammer_channel = 0
elif self.jammer_channel < 0:
    self.jammer_channel = self.n_channels - 1

self.state[self.jammer_channel] = 1  # Jam the selected channel 
"""

''' Reward control 
if self.state[action] == 1:  # Jammed
    # reward = -self.n_channels * 10
    reward = -self.n_channels * 10
else:
    if self.prev_action == action:
        reward = 1
    else:
        self.action_count += 1
        reward = 1-(0.04*self.action_count)
'''