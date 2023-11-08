import numpy as np

class Jammer:
    def __init__(self, initial_position, sweep_direction):
        self.position = initial_position
        self.sweep_direction = sweep_direction

    def move(self, n_channels):
        self.position += self.sweep_direction
        if self.position >= n_channels:
            self.position = 0
        elif self.position < 0:
            self.position = n_channels - 1

class JammingEnv:
    def __init__(self, n_channels=5, max_steps=200, num_jammers=1):
        self.n_channels = n_channels
        self.state = np.zeros(n_channels)
        self.num_jammers = num_jammers
        self.jammers = [Jammer(initial_position=np.random.randint(0, self.n_channels), sweep_direction=np.random.choice([-1, 1])) for _ in range(num_jammers)]
        self.done = False
        self.action_count = 0
        self.max_steps = max_steps
        
    def reset(self):
        if self.done == True:
            for jammer in self.jammers:
                jammer.__init__(initial_position=np.random.randint(0, self.n_channels), sweep_direction=np.random.choice([-1, 1]))
        self.state = np.zeros(self.n_channels)
        self.buffer_index = 0
        self.buffer_full = False
        return self.state
    
    def get_jammer_positions(self):
        positions = [jammer.position for jammer in self.jammers]
        return positions
   
    def step(self, action, time_step):
        reward = 0
        self.reset()

        for jammer in self.jammers:
            jammer.move(self.n_channels)
            self.state[jammer.position] = 1

        if self.state[action] == 1:  # Jammed
            reward = -1
        else:
            reward = 1

        if time_step >= self.max_steps:
            done = True
            self.done = done
        else:
            done = False
            self.done = done
        
        next_state = self.state

        return next_state, reward, done
        
    def render(self):
        """Visualize the current state of the environment."""
        print(self.state)
