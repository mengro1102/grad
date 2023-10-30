# test.py 파일
import numpy as np
from a2c import ActorCriticAgent
from env import JammingEnv
from realtimevisualizer import RealTimeVisualizer
import matplotlib.pyplot as plt

if __name__ == "__main__":
    # Initialize environment and agent
    channel = 10
    time = 50
    time_step = 0
    n_episodes = 30

    episodic_reward = []
    mean_episodic_reward = []
    
    env = JammingEnv(n_channels=channel,max_steps=time)
    agent = ActorCriticAgent(n_channels=channel)
    agent.load_model("actor_model_test.h5", "critic_model_test.h5")
    visualizer = RealTimeVisualizer(n_channels=channel)
    
    for episode in range(n_episodes):
        print('======================{}=========================='.format(episode+1))
        state = env.reset()
        state = np.reshape(state, [1, channel])
        episode_reward = 0
        
        done = False
        time_step = 0
        reward_return = []
        # print("return:",sum(reward_return))
        while not done:
            time_step += 1
            action = agent.choose_action(state, False)
            next_state, reward, done = env.step(action, time_step)
            next_state = np.reshape(next_state, [1, channel])
            visualizer.update(time_step, state, action)
            
            episode_reward += reward
            state = next_state
            reward_return.append(episode_reward)

            if done:
                break
                # ============
        '''
        while not done:
            if time_step == 0:
                time_step += 1
                if time_step >= time:
                    done = True
                else:
                    done = False
                    
                action = np.random.randint(0, channel) # agent.choose_action
        
                # env.step(action)
                
                jammer_channel = np.random.randint(0, channel)
                state[0][jammer_channel] = 1
                
                if state[0][action] == 1:  # Jammed
                    reward = -channel
                else:
                    reward = 1
                
                # training
                first_state = state.copy()
                first_state = np.reshape(first_state, [1, channel])
                visualizer.update(time_step, state, action)   
                
                episode_reward += reward
                state = first_state
                reward_return.append(episode_reward)
                # print("return:",sum(reward_return))
                if done:
                    break          
            else:
                time_step += 1
                action = agent.choose_action(state, False)
                next_state, reward, done = env.step(action, time_step)
                next_state = np.reshape(next_state, [1, channel])
                visualizer.update(time_step, state, action)
                
                episode_reward += reward
                state = next_state
                reward_return.append(episode_reward)

                if done:
                    break
        '''
    print("return:",sum(reward_return))
    episodic_reward.append(sum(reward_return))
    # mean_episodic_reward.append((sum(reward_return)/time))
    
    '''
    plt.plot(range(1,len(episode_rewards_list)+1), episode_rewards_list)
    plt.title('Episode-wise Total Rewards')
    plt.xlabel('Episode')
    plt.ylabel('Total Reward')
    plt.show()
    '''