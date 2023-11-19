from env_random import JammingEnv
from a2c import ActorCritic
from realtimevisualizer import RealTimeVisualizer
import matplotlib.pyplot as plt
import numpy as np

channel = 10
time = (200)-1
time_step = 0
n_episodes = 100
number_of_jammer = 2

# visualizer = RealTimeVisualizer(n_channels=channel)
env = JammingEnv(n_channels=channel,max_steps=time,num_jammers=number_of_jammer)
agent = ActorCritic(state_size=channel,action_size=channel)
scores = []
actor_loss = []
critic_loss = []

random_ag_scores = []
random_ag_action = 0

follow_ag_scores = []
follow_ag_action = 0

for episode in range(n_episodes):
    state = env.reset()
    state = np.reshape(state, [1, channel])
    episode_reward = 0
    follow_ag_epi_reward = 0
    random_ag_epi_reward = 0
    done = False
    time_step = 0
    reward_return = []
    env.collisions_cnt = 0
    follow_ag_action = np.random.randint(0, channel)
    
    while not done:
        action = agent.get_action(state)
        random_ag_action = np.random.randint(0, channel)
        
        next_state, reward, done, follow_ag_reward, random_ag_reward = env.step(action, time_step, follow_ag_action, random_ag_action)
        # print(reward)
        episode_reward += reward
        random_ag_epi_reward += random_ag_reward
        follow_ag_epi_reward += follow_ag_reward
        #print('RL: ',episode_reward)
        #print('Random: ',random_ag_epi_reward)
        #print('Follow: ',follow_ag_epi_reward)
        
        # visualizer.update(time_step, next_state, action)
        time_step += 1
        aloss, closs = agent.train_step(state, action, reward, next_state, done)
        state = next_state
        
        follow_ag_action = np.where(state == 1)[0][0]
        actor_loss.append(aloss[0].numpy())
        critic_loss.append(closs[0].numpy())
        
        if done:
            scores.append(episode_reward)
            follow_ag_scores.append(follow_ag_epi_reward)
            random_ag_scores.append(random_ag_epi_reward)
            print("Episode " + str(episode+1) + ": " + str(episode_reward)+" ================================================================")
            # print("Jammed of Episode " + str(episode+1) + ": " + str(env.collisions_cnt))
            break

# visualizer.close()

def moving_average(a, n=3):
    ret = np.cumsum(a, dtype=float)
    ret[n:] = ret[n:] - ret[:-n]
    return ret[n - 1:] / n

window_size = 4  # Adjust the window size as needed
rolling_avg_rewards = moving_average(scores, window_size)
<<<<<<< HEAD
rolling_avg_random = moving_average(follow_ag_scores, window_size)
rolling_avg_follow = moving_average(random_ag_scores, window_size)
img_path = 'C:/code/image/'
=======
rolling_avg_follow = moving_average(follow_ag_scores, window_size)
rolling_avg_random = moving_average(random_ag_scores, window_size)
img_path = 'C:/code/'
>>>>>>> 4763182922896133469ba02780c416f94f5f8a16
# Plotting
plt.plot(scores, label=f'Score (Actor-critic Agent)')
plt.plot(random_ag_scores, label=f'Score (Random Agent)')
plt.plot(follow_ag_scores, label=f'Score (Following Agent)')
plt.legend()
plt.title('Scroe Graph')
plt.ylabel('Reward')
plt.xlabel('Episode')
plt.savefig(img_path + 'Score_Figure('+str(channel)+'channel,'+str(n_episodes)+'Episode).png')
plt.show()

plt.plot(rolling_avg_rewards, label=f'Rolling Average Reward (Actor-critic Agent)')
plt.plot(rolling_avg_random, label=f'Rolling Average Reward (Random Agent)')
plt.plot(rolling_avg_follow, label=f'Rolling Average Reward (Following Agent)')
plt.legend()
plt.title('Scroe Graph')
plt.ylabel('Rolling Average Reward')
plt.xlabel('Episode')
plt.savefig(img_path + 'Rolling_Average_Figure('+str(channel)+'channel,'+str(n_episodes)+'Episode).png')
plt.show()

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

print("Training finished.")