from env import JammingEnv
from a2c import ActorCritic
from realtimevisualizer import RealTimeVisualizer
import matplotlib.pyplot as plt
import numpy as np

channel = 10
time = (100)-1
time_step = 0
n_episodes = 100
number_of_jammer = 1

visualizer = RealTimeVisualizer(n_channels=channel)
env = JammingEnv(n_channels=channel,max_steps=time,num_jammers=number_of_jammer)
agent = ActorCritic(state_size=channel,action_size=channel)
scores = []
actor_loss = []
critic_loss = []

random_ag_scores = []
random_ag_reward = 0

for episode in range(n_episodes):
    state = env.reset()
    state = np.reshape(state, [1, channel])
    episode_reward = 0    
    done = False
    time_step = 0
    reward_return = []
    env.collisions_cnt = 0
     
    while not done:
        action = agent.get_action(state)
        next_state, reward, done = env.step(action, time_step)
        # if episode == 99:
            # visualizer.update(time_step, next_state, action)
        time_step += 1
        aloss, closs = agent.train_step(state, action, reward, next_state, done)
        state = next_state
        episode_reward += reward
        actor_loss.append(aloss[0].numpy())
        critic_loss.append(closs[0].numpy())
        
        if done:
            scores.append(episode_reward)
            print("Episode " + str(episode+1) + ": " + str(episode_reward)+"===========================")
            # print("Jammed of Episode " + str(episode+1) + ": " + str(env.collisions_cnt))
            break

def moving_average(a, n=3) :
    ret = np.cumsum(a, dtype=float)
    ret[n:] = ret[n:] - ret[:-n]
    return ret[n - 1:] / n

window_size = 2  # Adjust the window size as needed
rolling_avg_rewards = moving_average(scores, window_size)

visualizer.close()

img_path = 'C:/code/image/'
# Plotting
plt.plot(rolling_avg_rewards, label=f'Rolling Average Reward')
plt.plot(scores, label='Episode Rewards')
plt.legend()

# Save or show the plot
plt.title('100 Episode Simulation (Sweeping Jammer)')
plt.ylabel('Reward')
plt.xlabel('Episode')
plt.savefig(img_path + 'Rolling_Average_Figure('+str(channel)+'channel,'+str(n_episodes)+'Episode).png')
plt.show()

plt.plot([item[0] for item in actor_loss], label='Actor Loss')
plt.title('Actor Loss Graph (Policy Gradient Loss)')
plt.ylabel('Loss')
plt.xlabel('Timestep')
plt.savefig(img_path+'Figure_2('+str(channel)+'channel,'+str(n_episodes)+'Episode).png')
plt.show()

plt.plot([item[0] for item in critic_loss], label='Critic Loss')
plt.title('Critic Loss Graph (MSE Loss)')
plt.ylabel('Loss')
plt.xlabel('Timestep')
plt.savefig(img_path+'Figure_3('+str(channel)+'channel,'+str(n_episodes)+'Episode).png')
plt.show()

print("Training finished.")