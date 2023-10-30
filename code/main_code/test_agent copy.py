# test.py 파일
import numpy as np
from a2c import ActorCriticAgent
from env import JammingEnv
from realtimevisualizer import RealTimeVisualizer
import matplotlib.pyplot as plt

if __name__ == "__main__":
    n_channels = 10
    agent = ActorCriticAgent(n_channels=n_channels)
    agent.load_model("actor_model_test.h5", "critic_model_test.h5")
    
    env = JammingEnv(n_channels=n_channels)
    
    total_reward = 0
    n_episodes = 30
    episode_rewards_list = []

    visualizer = RealTimeVisualizer(n_channels=n_channels)
    
    ani_data = []
    time_step = 0
    for _ in range(n_episodes):
        if time_step == 0:
            print("for loop(Main)")
            state = env.reset()
            state = np.reshape(state, [1, n_channels])
            episode_reward = 0
        else:
            print("for loop(Main)")
            state = env.reset()
            state = np.reshape(state, [1, n_channels])
            episode_reward = 0
        
        done = False
        while not done:
            print("while loop(Main)")
            action = agent.choose_action(state)
            print("Action(Main):", action+1)
            next_state, reward, done = env.step(action)
            print("Reward(Main):",reward)
            next_state = np.reshape(next_state, [1, n_channels])
            visualizer.update(time_step, state, action)
            print("Visualizer Finished(Main)")
            
            total_reward += reward
            episode_reward += reward
            state = next_state
            time_step = time_step + 1
            if done:  # If episode ends, break out of the loop
                break
        #episode_rewards_list.append(episode_reward)
    
    avg_reward = total_reward / n_episodes
    print("Average Reward over {} episodes: {}".format(n_episodes, avg_reward))
    '''
    plt.plot(range(1,len(episode_rewards_list)+1), episode_rewards_list)
    plt.title('Episode-wise Total Rewards')
    plt.xlabel('Episode')
    plt.ylabel('Total Reward')
    plt.show()
    '''