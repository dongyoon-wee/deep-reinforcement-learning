import torch
import numpy as np

from collections import deque
from unityagents import UnityEnvironment

from ddpg_agent import Agent
from utils import get_settings
from opt import opt


def infer_agent(env, agent, brain_name, n_episodes=2000):
    """DDPG.

    Params
    ======
        n_episodes (int): maximum number of training episodes
    """

    scores = []  # list containing scores from each episode
    scores_window = deque(maxlen=100)  # last 100 scores
    for i_episode in range(1, n_episodes + 1):
        env_info = env.reset(train_mode=True)[brain_name]  # reset the environment
        state = env_info.vector_observations[0]  # get the current state
        score = 0
        while True:
            action = agent.act(state)
            env_info = env.step(action)[brain_name]
            next_state = env_info.vector_observations[0]  # get the next state
            reward = env_info.rewards[0]  # get the reward
            done = env_info.local_done[0]  # see if episode has finished
            #agent.step(state, action, reward, next_state, done)
            state = next_state
            score += reward
            if done:
                break
        scores_window.append(score)  # save most recent score
        scores.append(score)  # save most recent score
        print('\rEpisode {}\tAverage Score: {:.2f}'.format(i_episode, np.mean(scores_window)), end="")
        if i_episode % 100 == 0:
            print('\rEpisode {}\tAverage Score: {:.2f}'.format(i_episode, np.mean(scores_window)))
    return scores


if __name__ == '__main__':

    print(opt)
    env = UnityEnvironment(file_name="Reacher.app")
    brain_name = env.brain_names[0]
    brain = env.brains[brain_name]
    env_info = env.reset(train_mode=True)[brain_name]
    state_size, action_size = get_settings(env_info, brain)

    agent = Agent(state_size, action_size, opt.seed)
    agent = agent.load_actor(opt.actor_model_path)
    agent = agent.load_critic(opt.critic_model_path)
    scores = infer_agent(env, agent, brain_name)
    env.close()
