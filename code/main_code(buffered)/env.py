import numpy as np

class JammingEnv:
    def __init__(self, n_channels=5, max_steps=200):
        self.n_channels = n_channels
        self.state = np.zeros(n_channels)
        # self.last_action = last_action
        self.jammer_channel = np.random.randint(0, self.n_channels)
        self.sweep_direction = np.random.choice([-1, 1])
        self.prev_action = None
        self.action_count = 0
        self.current_step = 0
        self.max_steps = max_steps
        self.observation_space = []
        
    def reset(self):
        self.state = np.zeros(self.n_channels)
        # self.current_step = 0
        
        return self.state
   
    def step(self, action, time_step, num_jammer):
        reward = 0
        self.reset()
        
        # ===== Jammer =====
        if num_jammer == 1: # Single Jammer
            '''
            self.jammer_channel = np.random.randint(0, self.n_channels)  # Randomly select a channel to jam
            self.state[self.jammer_channel] = 1  # Jam the selected channel
            '''
            """ Jammer type = Sweeping  """
            self.jammer_channel += self.sweep_direction
            if self.jammer_channel >= self.n_channels:
                self.jammer_channel = 0
            elif self.jammer_channel < 0:
                self.jammer_channel = self.n_channels - 1

            self.state[self.jammer_channel] = 1  # Jam the selected channel
        else: # Multiple Jammer
            self.jammer_channel += self.sweep_direction
            if self.jammer_channel >= self.n_channels:
                self.jammer_channel = 0
            elif self.jammer_channel < 0:
                self.jammer_channel = self.n_channels - 1
            pass
   
          #test
        
        
        ''' Technical 
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
        ''' Vanilla '''
        if self.state[action] == 1:  # Jammed
            reward = -1 # -self.n_channels
        else:
            reward = 1
        
        
        if time_step >= self.max_steps:
            done = True
        else:
            done = False
        
        next_state = self.state.copy()
        self.observation_space = next_state
        
        return next_state, reward, done
        
    def render(self):
        """Visualize the current state of the environment."""
        print(self.state)
