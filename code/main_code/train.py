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
number_of_jammer = 2

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
        print(time_step)
        action = agent.get_action(state)
        next_state, reward, done = env.step(action, time_step)
        if episode == n_episodes-1:
            visualizer.update(time_step, next_state, action)
        time_step += 1
        aloss, closs = agent.train_step(state, action, reward, next_state, done)
        # Our new state is state
        state = next_state
        #print('reward(Timestep):',reward)
        
        episode_reward += reward
        
        # if episode ends
        actor_loss.append(aloss[0][0].numpy())
        critic_loss.append(closs[0].numpy())
        # print('actor_loss len:',len(actor_loss),', critic_loss len',len(critic_loss))
        
        if done:
            scores.append(episode_reward)
            print("Episode " + str(episode+1) + ": " + str(episode_reward))
            # print('\nactor_loss:',actor_loss)
            # print('\ncritic_loss:',critic_loss)
            break
    # print('reward(episode):',episode_reward)

alpha = 0.2
ema_result = exponential_moving_average(scores, alpha)

visualizer.close()
''' Graph Image Save''' 
img_path = ''

plt.plot(scores)
plt.plot(ema_result)
plt.title('Score Graph (Reward for each Episode)')
plt.ylabel('Reward')
plt.xlabel('Episode')
plt.savefig(img_path+'Figure_1('+str(channel)+'channel,'+str(n_episodes)+'Episode,'+str(number_of_jammer)+'Jammers).png')
plt.show()
'''
print(actor_loss)
plt.plot(actor_loss)
plt.title('Actor Loss Graph (Policy Gradient Loss)')
plt.ylabel('Loss')
plt.xlabel('Timestep')
plt.savefig(img_path+'Figure_2('+str(channel)+'channel,'+str(n_episodes)+'Episode).png')
plt.show()

plt.plot(critic_loss)
plt.title('Critic Loss Graph (MSE Loss)')
plt.ylabel('Loss')
plt.xlabel('Timestep')
plt.savefig(img_path+'Figure_3('+str(channel)+'channel,'+str(n_episodes)+'Episode).png')
plt.show()
'''
# agent.save_model("actor_model_test.h5", "critic_model_test.h5")
print("Training finished.")