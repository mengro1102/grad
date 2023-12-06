from env import JammingEnv
from a2c_buffered import ActorCritic
from a2c_buffered import StateBuffer
from realtimevisualizer import RealTimeVisualizer
import matplotlib.pyplot as plt
import numpy as np

def exponential_moving_average(data, alpha):
    ema = [data[0]]  # 초기값은 첫 번째 데이터 포인트로 설정
    for i in range(1, len(data)):
        ema.append(alpha * data[i] + (1 - alpha) * ema[-1])
    return np.array(ema)

''' Initialize '''
channel = 10
time = (200)-1
time_step = 0
n_episodes = 50
number_of_jammer = 5
buffer_size = 5

visualizer = RealTimeVisualizer(n_channels=channel)
buffer = StateBuffer(buffer_size, channel)
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
    # env.jamming_cnt = 0
    while not done:
        buffer.add_state(state)
        action = agent.get_action(buffer.get_buffer())
        next_state, reward, done = env.step(action, time_step)
        #if episode == n_episodes-1:
        #    visualizer.update(time_step, next_state, action)
        time_step += 1
        aloss, closs = agent.train_step(buffer.get_buffer(), action, reward, next_state, done)
        
        state = next_state
        episode_reward += reward

        actor_loss.append(aloss[0][0].numpy())
        critic_loss.append(closs[0].numpy())
        
        if done:
            scores.append(episode_reward)
            print("Episode " + str(episode+1) + ": " + str(episode_reward) + " ============================")
            break


alpha = 0.2
ema_result = exponential_moving_average(scores, alpha)

visualizer.close()
''' Graph Image Save'''
img_path = 'buffered/'

# 1.5e-3
plt.rcParams.update({'font.size':10})
plt.plot(scores, label=f'Instant Reward')
plt.plot(ema_result, label=f'Moving Average Reward')
plt.ylabel('Reward')
plt.xlabel('Episode')
plt.ylim(0, 200)
plt.legend(bbox_to_anchor =(1.04, 1.09), ncol = 2)
plt.savefig(img_path+'Buffered('+str(channel)+'channel,'+str(n_episodes)+'Episode,'+str(number_of_jammer)+'Jammers).png')
plt.show()
plt.close()
'''
plt.plot(actor_loss)
plt.title('Actor Loss Graph (Policy Gradient Loss)')
plt.ylabel('Loss')
plt.xlabel('Timestep')
plt.savefig(img_path+'Figure_2('+str(channel)+'channel,'+str(n_episodes)+'Episode,'+str(number_of_jammer)+'Jammers).png')
plt.show()
plt.close()

plt.plot(critic_loss)
plt.title('Critic Loss Graph (MSE Loss)')
plt.ylabel('Loss')
plt.xlabel('Timestep')
plt.savefig(img_path+'Figure_3('+str(channel)+'channel,'+str(n_episodes)+'Episode,'+str(number_of_jammer)+'Jammers).png')
plt.show()
plt.close()
'''
print("Training finished.")