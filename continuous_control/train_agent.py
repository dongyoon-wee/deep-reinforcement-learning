import torch
import numpy as np

from collections import deque
from unityagents import UnityEnvironment
from time import time

from ddpg_agent import Agent
from opt import opt
from utils import get_settings


def train_agent(env, agent, brain_name, model_path='models/{0}_{1}.pth', n_episodes=2000, success_score=30, scores=None):

    scores_window = deque(maxlen=100)  # last 100 scores
    if scores is None:
        scores = []  # list containing scores from each episode
    else:
        scores_window.extend(scores)

    init_episode = len(scores) + 1

    for i_episode in range(init_episode, n_episodes + 1):

        start_time = time()
        env_info = env.reset(train_mode=True)[brain_name]  # reset the environment
        state = env_info.vector_observations[0]  # get the current state
        score = 0
        num_iters = 0

        while True:
            action = agent.act(state)
            env_info = env.step(action)[brain_name]
            next_state = env_info.vector_observations[0]  # get the next state
            reward = env_info.rewards[0]  # get the reward
            done = env_info.local_done[0]  # see if episode has finished
            agent.step(state, action, reward, next_state, done)
            state = next_state
            score += reward
            num_iters += 1
            #if num_iters % 10 == 0:
            #    print('\rNumber of iterations: {0}'.format(num_iters), end='\r')
            if done:
                break
        scores_window.append(score)  # save most recent score
        scores.append(score)  # save most recent score
        print('\rEpisode {0}\tAverage Score: {1:.2f}\tAverage Time: {2:.2f}\tNum Iters: {3}=='.format(i_episode, np.mean(scores_window), time()-start_time, num_iters), end="\r")

        if i_episode % 100 == 0:
            print('\rEpisode {}\tAverage Score: {:.2f}\tAverage Time: {}'.format(i_episode, np.mean(scores_window), time()-start_time), end="\r")
            torch.save(agent.actor_local.state_dict(), model_path.format('actor', i_episode))
            torch.save(agent.critic_local.state_dict(), model_path.format('critic', i_episode))
            np.save('scores_{0}.npy'.format(i_episode), scores)

        if np.mean(scores_window) >= success_score:
            tag = 'success'
            print('\nEnvironment solved in {:d} episodes!\tAverage Score: {:.2f}'.format(i_episode - 100,
                                                                                         np.mean(scores_window)))
            torch.save(agent.actor_local.state_dict(), model_path.format('actor', tag))
            torch.save(agent.critic_local.state_dict(), model_path.format('critic', tag))
            np.save('scores_{0}.npy'.format(tag), score)
            break
    return scores


if __name__ == '__main__':

    print(opt)
    env = UnityEnvironment(file_name="Reacher.app")
    brain_name = env.brain_names[0]
    brain = env.brains[brain_name]
    env_info = env.reset(train_mode=True)[brain_name]
    state_size, action_size = get_settings(env_info, brain)

    agent = Agent(state_size, action_size, opt.seed, opt.buffer_size, opt.batch_size, opt.gamma, opt.tau, opt.lr_actor,
                  opt.lr_critic, opt.weight_decay)
    scores = train_agent(env, agent, brain_name, opt.model_path, opt.n_episodes, opt.success_score)
    env.close()
