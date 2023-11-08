from env import JammingEnv
from a2c import ActorCritic
import matplotlib.pyplot as plt
import numpy as np

''' Initialize '''
channel = 10
time = (200)-1
time_step = 0
n_episodes = 10
number_of_jammer = 1

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
        
        # TAKING ACTION
        next_state, reward, done = env.step(action, time_step)
        time_step += 1
        aloss, closs = agent.train_step(state, action, reward, next_state, done)
        # jammer_positions = env.get_jammer_positions()
        # print(jammer_positions)
        state = next_state
        episode_reward += reward
        actor_loss.append(aloss[0].numpy())
        critic_loss.append(closs[0].numpy())
        
        if done:
            scores.append(episode_reward)
            print("Episode " + str(episode+1) + ": " + str(episode_reward))
            break

''' Graph Image Save'''
img_path = 'C:/code/result/'    

plt.plot(scores)
plt.title('Score Graph (Reward for each Episode)')
plt.ylabel('Reward')
plt.xlabel('Episode')
plt.savefig(img_path+'Figure_1('+str(channel)+'channel,'+str(n_episodes)+'Episode).png')
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

# agent.save_model("actor_model_test.h5", "critic_model_test.h5")
print("Training finished.")