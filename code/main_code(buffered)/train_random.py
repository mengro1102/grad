from env_random import JammingEnv
from a2c import ActorCritic
from realtimevisualizer import RealTimeVisualizer
import matplotlib.pyplot as plt
import numpy as np

def exponential_moving_average(data, alpha):
    ema = [data[0]]  # 초기값은 첫 번째 데이터 포인트로 설정
    for i in range(1, len(data)):
        ema.append(alpha * data[i] + (1 - alpha) * ema[-1])
    return np.array(ema)


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
        # visualizer.update(time_step, next_state, follow_ag_action) # follow 정책 검증
        episode_reward += reward
        random_ag_epi_reward += random_ag_reward
        follow_ag_epi_reward += follow_ag_reward
        
        time_step += 1
        aloss, closs = agent.train_step(state, action, reward, next_state, done)
        state = next_state
        follow_ag_action = np.where(state == 1)[0][0]

        actor_loss.append(aloss[0][0].numpy())
        critic_loss.append(closs[0].numpy())
        
        if done:
            scores.append(episode_reward)
            follow_ag_scores.append(follow_ag_epi_reward)
            random_ag_scores.append(random_ag_epi_reward)
            print("Episode " + str(episode+1) + ": " + str(episode_reward)+" ================================================================")
            break



alpha = 0.2
ema_result = exponential_moving_average(scores, alpha)
ema_result_random = exponential_moving_average(random_ag_scores, alpha)
ema_result_follow = exponential_moving_average(follow_ag_scores, alpha)

img_path = ''
# Plotting
plt.plot(scores, label=f'Score (Actor-critic Agent)')
plt.plot(random_ag_scores, label=f'Score (Random Agent)')
plt.plot(follow_ag_scores, label=f'Score (Following Agent)')
plt.legend()
plt.title('Scroe Graph')
plt.ylabel('Reward')
plt.xlabel('Episode')
plt.savefig(img_path + 'Score_Figure('+str(channel)+'channel,'+str(n_episodes)+'Episode,'+str(number_of_jammer)+'Jammers).png')
plt.show()

plt.plot(ema_result, label=f'Moving Average Reward (Actor-critic Agent)')
plt.plot(ema_result_random, label=f'Moving Average Reward (Random Agent)')
plt.plot(ema_result_follow, label=f'Moving Average Reward (Following Agent)')
plt.legend()
plt.title('Scroe Graph')
plt.ylabel('Moving Average Reward')
plt.xlabel('Episode')
plt.savefig(img_path + 'Moving_Average_Figure('+str(channel)+'channel,'+str(n_episodes)+'Episode,'+str(number_of_jammer)+'Jammers).png')
plt.show()

print("Training finished.")