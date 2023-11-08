import numpy as np

class Jammer:
    def __init__(self, n_channels):
        self.n_channels = n_channels
        self.position = np.random.randint(0, n_channels)
        self.sweep_direction = np.random.choice([-1, 1])

    def update(self):
        self.position += self.sweep_direction
        if self.position >= self.n_channels:
            self.position = 0
        elif self.position < 0:
            self.position = self.n_channels - 1

class JammingEnv:
    def __init__(self, n_channels=5, max_steps=200, num_jammers=1):
        self.n_channels = n_channels
        self.state = np.zeros(n_channels)
        # self.jammer_channel = np.random.randint(0, self.n_channels)
        # self.sweep_direction = np.random.choice([-1, 1])
        self.num_jammers = num_jammers
        self.jammers = [Jammer(n_channels) for _ in range(num_jammers)]
        self.action_count = 0
        self.max_steps = max_steps
        self.observation_space = []
        
    def reset(self):
        self.state = np.zeros(self.n_channels)
        for jammer in self.jammers:
            jammer.__init__(self.n_channels)
        
        return self.state
   
        def step(self, action, time_step):
            reward = 0
        self.reset()

        for jammer in self.jammers:
            self.state[jammer.position] = 1

        if self.state[action] == 1:  # Jammed
            reward = -1
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
