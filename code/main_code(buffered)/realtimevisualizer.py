import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from matplotlib.colors import BoundaryNorm
from matplotlib.patches import Patch
# from matplotlib.animation import FuncAnimation
import os
class RealTimeVisualizer:
    def __init__(self, n_channels):
        self.n_channels = n_channels
        plt.ion()  # Interactive mode on
        self.fig, self.ax = plt.subplots(figsize=(10, 6))
        self.time = []
        self.states = []
        self.max_timesteps = 50
        self.cmap = ListedColormap(['white', 'blue', 'red', 'gray'])
        self.norm = BoundaryNorm([0, 1, 2, 3, 4], self.cmap.N)
        self.save_folder = 'C:/code/image/original'
        
    def update(self, time_step, state, agent_action):
        if not os.path.exists(self.save_folder):
            os.makedirs(self.save_folder)
        self.time.append(time_step)
        if time_step == 0:
            colors = []
            for i in range(len(state)):
                s = state[i]
                # print(state[0][i])
                if s == 1:
                    if i == agent_action:
                        colors.append(3)  # gray (Agent action matches jammed state)
                    else:
                        colors.append(2)    # red (Jammed state)
                elif s == 0:
                    if i == agent_action:
                        colors.append(1)    # blue (Agent action)
                    else:
                        colors.append(0)    # white (Idle state)
            self.states.append(colors)

            if len(self.states) > self.max_timesteps:
                self.states.pop(0)
                self.time.pop(0)
            
            self.ax.clear()
            X, Y = np.meshgrid(np.arange(len(self.time)), np.arange(len(state)))
            self.ax.pcolormesh(X, Y, np.array(self.states).T, cmap=self.cmap, norm=self.norm, shading='auto', edgecolors='k')

            self.ax.set_yticks(list(range(self.n_channels)))
            self.ax.set_yticklabels(list(range(1, self.n_channels+1)))
        
            if time_step <= self.max_timesteps:
                locs = list(range(0, self.max_timesteps+1, int(self.max_timesteps*.1)))
                self.ax.set_xticks(locs)  # 레이블의 위치 설정
                self.ax.set_xticklabels(locs)  # 레이블의 텍스트 설정
            else:
                start = time_step - self.max_timesteps    
                self.ax.set_xticklabels(list(range(start, time_step + 1, int(self.max_timesteps*.1))))
            
            self.ax.set_title('Environments State Transition')
            self.ax.set_ylabel('Channels')
            self.ax.set_xlabel('Time step')
            
            plt.draw()
            plt.pause(0.001)
            filepath = f"{self.save_folder}/image_{time_step}.png"
            plt.savefig(filepath)
        else:
            colors = []
            for i in range(len(state)):  # Assuming state is a 2D numpy array with shape (1, num_states)
                s = state[i]
                if s == 1:
                    if i == agent_action:
                        colors.append(3)  # gray (Agent action matches jammed state)
                    else:
                        colors.append(2)    # red (Jammed state)
                elif s == 0:
                    if i == agent_action:
                        colors.append(1)    # blue (Agent action)
                    else:
                        colors.append(0)    # white (Idle state)
            self.states.append(colors)

            if len(self.states) > self.max_timesteps:
                self.states.pop(0)  # Remove the oldest state
                self.time.pop(0)  # Remove the corresponding time
            
            # self.ax.clear()
            # Create a colormap
            X, Y = np.meshgrid(np.arange(len(self.time)), np.arange(len(state)))
            self.ax.pcolormesh(X, Y, np.array(self.states).T, cmap=self.cmap, norm=self.norm, shading='auto', edgecolors='k')

            # y축 설정
            self.ax.set_yticks(list(range(self.n_channels)))  # 0~5까지의 위치
            self.ax.set_yticklabels(list(range(1, self.n_channels+1)))  # 1~5까지의 레이블
            
            # x축 label 설정
            if time_step <= self.max_timesteps:
                self.ax.set_xticklabels(list(range(0, self.max_timesteps+1, int(self.max_timesteps*.1))))
            else:
                start = time_step - self.max_timesteps    
                self.ax.set_xticklabels(list(range(start, time_step + 1, int(self.max_timesteps*.1))))
            
            self.ax.set_xticks(list(range(0, self.max_timesteps+1, int(self.max_timesteps*.1))))
            self.ax.set_title('Real-time State & Agent Actions')
            self.ax.set_ylabel('Channels')
            self.ax.set_xlabel('Time step')
            
            plt.draw()
            plt.pause(0.001)
            filepath = f"{self.save_folder}/image_{time_step}.png"
            plt.savefig(filepath)
            # plt.savefig(filepath, bbox_inches="tight")
            
    def close(self):
        plt.ioff()
        plt.show()
