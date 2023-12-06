from env import JammingEnv
from a2c import ActorCritic
from realtimevisualizer import RealTimeVisualizer
import matplotlib.pyplot as plt
import numpy as np

def exponential_moving_average(data, alpha):
    ema = [data[0]]  # 초기값은 첫 번째 데이터 포인트로 설정
    for i in range(1, len(data)):
        ema.append(alpha * data[i] + (1 - alpha) * ema[-1])
    return np.array(ema)

'''
def moving_average(a, n=3) :
    ret = np.cumsum(a, dtype=float)
    ret[n:] = ret[n:] - ret[:-n]
    return ret[n - 1:] / n

window_size = 4  # Adjust the window size as needed
rolling_avg_rewards = moving_average(scores, window_size)
'''

''' Initialize '''
channel = 10
time = (200)-1
time_step = 0
n_episodes = 100
number_of_jammer = 5

visualizer = RealTimeVisualizer(n_channels=channel)
env = JammingEnv(n_channels=channel,max_steps=time,num_jammers=number_of_jammer)
agent = ActorCritic(state_size=channel,action_size=channel)
scores = []
actor_loss = []
critic_loss = []

''' Learning '''
for episode in range(n_episodes):
    print('======================{}=========================='.format(episode+1))
    state = env.reset()
    state = np.reshape(state, [1, channel])
    episode_reward = 0
    
    done = False
    time_step = 0
    reward_return = []

    while not done:
        # print(time_step)
        action = agent.get_action(state)
        next_state, reward, done = env.step(action, time_step)
        #if episode == n_episodes-1:
        #    visualizer.update(time_step, next_state, action, save=False)
        time_step += 1
        aloss, closs = agent.train_step(state, action, reward, next_state, done)
        state = next_state
        episode_reward += reward
        
        actor_loss.append(aloss[0][0].numpy())
        critic_loss.append(closs[0].numpy())
        
        if done:
            scores.append(episode_reward)
            print("Episode " + str(episode+1) + ": " + str(episode_reward))
            break

alpha = 0.2
ema_result = exponential_moving_average(scores, alpha)

visualizer.close()
''' Graph Image Save''' 
img_path = ''

# 1.5e-3
plt.rcParams.update({'font.size':10})
plt.plot(scores, label=f'Instant Reward')
plt.plot(ema_result, label=f'Moving Average Reward')
plt.ylabel('Reward')
plt.xlabel('Episode')
plt.ylim(0, 200)
plt.legend(bbox_to_anchor =(1.04, 1.09), ncol = 2)
plt.savefig(img_path+'Conventional('+str(channel)+'channel,'+str(n_episodes)+'Episode,'+str(number_of_jammer)+'Jammers-).png')
plt.show()

print("Training finished.")