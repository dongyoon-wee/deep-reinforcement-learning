import torch
import numpy as np

from collections import deque
from unityagents import UnityEnvironment
from time import time

from ddpg_agent import Agent
from opt import opt
from utils import get_settings


def train_agent(env, agent, brain_name, model_path='{0}_{1}.pth', n_episodes=2000, max_t=1000, success_score=30):

    scores = []  # list containing scores from each episode
    scores_window = deque(maxlen=100)  # last 100 scores
    for i_episode in range(1, n_episodes + 1):
        start_time = time()
        env_info = env.reset(train_mode=True)[brain_name]  # reset the environment
        state = env_info.vector_observations[0]  # get the current state
        score = 0
        num_iters = 0
        while True:
        #for _ in range(max_t):
            action = agent.act(state)
            env_info = env.step(action)[brain_name]
            next_state = env_info.vector_observations[0]  # get the next state
            reward = env_info.rewards[0]  # get the reward
            done = env_info.local_done[0]  # see if episode has finished

            #print('\rAction: {0}\tReward: {1}'.format(action, reward), end="\r")

            agent.step(state, action, reward, next_state, done)
            state = next_state
            score += reward
            num_iters += 1
            if done:
                break
        scores_window.append(score)  # save most recent score
        scores.append(score)  # save most recent score
        print('\rEpisode {0}\tAverage Score: {1:.2f}\tAverage Time: {2:.2f}\tNum Iters: {3}=='.format(i_episode, np.mean(scores_window), time()-start_time, num_iters), end="")
        if i_episode % 100 == 0:
            print('\rEpisode {}\tAverage Score: {:.2f}\tAverage Time: {}'.format(i_episode, np.mean(scores_window), time()-start_time), end="\r")
            torch.save(agent.actor_local.state_dict(), model_path.format('actor', i_episode))
            torch.save(agent.critic_local.state_dict(), model_path.format('critic', i_episode))
        if np.mean(scores_window) >= success_score:
            print('\nEnvironment solved in {:d} episodes!\tAverage Score: {:.2f}'.format(i_episode - 100,
                                                                                         np.mean(scores_window)))
            torch.save(agent.actor_local.state_dict(), model_path.format('actor', 'success'))
            torch.save(agent.critic_local.state_dict(), model_path.format('critic', 'success'))
            break
    return scores


if __name__ == '__main__':

    print(opt)
    env = UnityEnvironment(file_name="Reacher.app")
    brain_name = env.brain_names[0]
    brain = env.brains[brain_name]
    env_info = env.reset(train_mode=True)[brain_name]
    state_size, action_size = get_settings(env_info, brain)

    list_fc_dims_actor = map(int, opt.list_fc_dims_actor.split(','))
    list_fc_dims_critic = map(int, opt.list_fc_dims_critic.split(','))

    agent = Agent(state_size, action_size, opt.seed, opt.buffer_size, opt.batch_size, opt.gamma, opt.tau, opt.lr_actor,
                  opt.lr_critic, opt.weight_decay, list_fc_dims_actor, list_fc_dims_critic)
    scores = train_agent(env, agent, brain_name, opt.n_episodes, opt.eps_start, opt.eps_end, opt.eps_decay)
    env.close()
