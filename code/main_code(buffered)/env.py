import numpy as np
    
class Jammer:
    def __init__(self, policy, n_channels):
        self.policy = policy
        self.n_channels = n_channels
        self.sweep_direction = np.random.choice([-1, 1])
        self.position = np.random.randint(0, self.n_channels)

    def jam_channel(self):
        if self.policy == "sweeping":
            return self.sweeping_policy()
        elif self.policy == "random":
            return self.random_policy()
        else:
            raise ValueError("Unsupported jamming policy")

    def sweeping_policy(self):
        self.position += self.sweep_direction
        if self.position >= self.n_channels:
            self.position = 0
        elif self.position < 0:
            self.position = self.n_channels - 1

        return self.position

    def random_policy(self):
        self.position = np.random.randint(0, self.n_channels) 
        return self.position
        
class JammingEnv:
    def __init__(self, n_channels=5, max_steps=200, num_jammers=1):
        self.n_channels = n_channels
        self.state = np.zeros(n_channels)
        self.num_jammers = num_jammers
        self.policy = 'sweeping'
        self.jammers = [Jammer(self.policy, self.n_channels) for _ in range(num_jammers)]
        self.done = False
        self.switching = False
        self.max_steps = max_steps
        self.collisions_cnt = 0
        
    def reset(self):
        if self.switching == True:
            for jammer in self.jammers:
                jammer.__init__(self.policy, self.n_channels)
        self.state = np.zeros(self.n_channels)
        
        return self.state
    
    def get_jammer_positions(self):
        positions = [jammer.position for jammer in self.jammers]
        return positions
   
    def step(self, action, time_step):
        reward = 0
        
        self.reset()

        for jammer in self.jammers:
            jammer.jam_channel()
            self.state[jammer.position] = 1

        if self.state[action] == 1:  # Jammed
            reward = -1
        else:
            reward = 1
<<<<<<< HEAD
        '''
        if time_step > int(self.max_steps/2):            
            self.policy = 'sweeping'
            self.switching = True
        else:
            # self.policy = 'sweeping'
            self.switching = False
        '''
=======
            
>>>>>>> 4763182922896133469ba02780c416f94f5f8a16
        ''' 
        if reward == -1:
            self.collisions_cnt += 1
            print(self.collisions_cnt)
        '''
        if time_step >= self.max_steps:
            done = True
<<<<<<< HEAD
            self.switching = True
=======
            #self.policy = 'sweeping'
>>>>>>> 4763182922896133469ba02780c416f94f5f8a16
            self.done = done
        else:
            done = False
            self.switching = False
            self.done = done
        
        next_state = self.state

        return next_state, reward, done
        
    def render(self):
        """Visualize the current state of the environment."""
        print(self.state)
