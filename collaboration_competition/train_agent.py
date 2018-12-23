import torch
import numpy as np

from collections import deque
from unityagents import UnityEnvironment
from time import time

from ddpg_agent import Agent, ReplayBuffer
from opt import opt
from utils import get_settings


def train_agent(env, agent1, agent2, brain_name, model_path='models/{0}_{1}.pth', n_episodes=2000, success_score=30, list_scores=None):

    scores_window = deque(maxlen=100)  # last 100 scores
    if list_scores is None:
        list_scores = []  # list containing scores from each episode
    else:
        scores_window.extend(list_scores)

    init_episode = len(list_scores) + 1

    for i_episode in range(init_episode, n_episodes + 1):

        start_time = time()
        env_info = env.reset(train_mode=True)[brain_name]  # reset the environment
        states = env_info.vector_observations  # get the current state
        scores = [0, 0]
        num_iters = 0

        while True:
            action1 = agent1.act(states[0])
            action2 = agent2.act(states[1])
            action = np.concatenate([action1, action2])
            env_info = env.step(action)[brain_name]
            next_states = env_info.vector_observations  # get the next state
            rewards = env_info.rewards  # get the reward
            done = env_info.local_done[0]  # see if episode has finished
            agent1.step(states[0], action1, rewards[0], next_states[0], done)
            agent2.step(states[1], action2, rewards[1], next_states[1], done)
            states = next_states
            scores[0] += rewards[0]
            scores[1] += rewards[1]
            num_iters += 1
            #if num_iters % 10 == 0:
            #    print('\rNumber of iterations: {0}'.format(num_iters), end='\r')
            if done:
                break
        score = np.max(scores)
        scores_window.append(score)  # save most recent score
        list_scores.append(score)
        print('\rEpisode {0}\tAverage Score: {1:.2f}\tAverage Time: {2:.2f}\tNum Iters: {3}=='.format(i_episode, np.mean(scores_window), time()-start_time, num_iters), end="\r")

        if i_episode % 500 == 0:
            print('\rEpisode {}\tAverage Score: {:.2f}\tAverage Time: {}'.format(i_episode, np.mean(scores_window), time()-start_time), end="\r")
            torch.save(agent1.actor_local.state_dict(), model_path.format('actor1', i_episode))
            torch.save(agent1.critic_local.state_dict(), model_path.format('critic1', i_episode))
            torch.save(agent2.actor_local.state_dict(), model_path.format('actor2', i_episode))
            torch.save(agent2.critic_local.state_dict(), model_path.format('critic2', i_episode))
            np.save('scores_{0}.npy'.format(i_episode), scores)

        if np.mean(scores_window) >= success_score:
            tag = 'success'
            print('\nEnvironment solved in {:d} episodes!\tAverage Score: {:.2f}'.format(i_episode - 100,
                                                                                         np.mean(scores_window)))
            torch.save(agent1.actor_local.state_dict(), model_path.format('actor1', tag))
            torch.save(agent1.critic_local.state_dict(), model_path.format('critic1', tag))
            torch.save(agent2.actor_local.state_dict(), model_path.format('actor2', tag))
            torch.save(agent2.critic_local.state_dict(), model_path.format('critic2', tag))
            np.save('scores_{0}.npy'.format(tag), score)
            break
    return list_scores


if __name__ == '__main__':

    print(opt)
    env = UnityEnvironment(file_name="Tennis.app")
    brain_name = env.brain_names[0]
    brain = env.brains[brain_name]
    env_info = env.reset(train_mode=True)[brain_name]
    state_size, action_size = get_settings(env_info, brain)

    memory = ReplayBuffer(action_size, opt.buffer_size, opt.batch_size, opt.seed)

    agent1 = Agent(state_size, action_size, opt.seed, opt.buffer_size, opt.batch_size, opt.gamma, opt.tau, opt.lr_actor,
                   opt.lr_critic, opt.weight_decay)
    agent2 = Agent(state_size, action_size, opt.seed, opt.buffer_size, opt.batch_size, opt.gamma, opt.tau, opt.lr_actor,
                   opt.lr_critic, opt.weight_decay)
    scores = train_agent(env, agent1, agent2, brain_name, opt.model_path, opt.n_episodes, opt.success_score)
    env.close()
