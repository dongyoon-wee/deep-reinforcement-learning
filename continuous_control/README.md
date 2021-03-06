[//]: # (Image References)

[image1]: https://user-images.githubusercontent.com/10624937/43851024-320ba930-9aff-11e8-8493-ee547c6af349.gif "Trained Agent"
[image2]: outputs/scores_318.png "Crawler"


# Project 2: Continuous Control

### Project details

For this project, a single agent (version 1) is trained on [Reacher](https://github.com/Unity-Technologies/ml-agents/blob/master/docs/Learning-Environment-Examples.md#reacher) environment.

![Trained Agent][image1]

In this environment, a double-jointed arm can move to target locations. 

#### Reward

The environment provides a reward +0.1 for each step that the agent's hand is in the goal location. 

Thus, the goal of your agent is to maintain its position at the target location for as many time steps as possible.

#### States

The observation space consists of 33 variables corresponding to position, rotation, velocity, and angular velocities of the arm. 

#### Actions

The action space consists of 4 variables corresponding to torque applicable to two joints. Every entry in the action vector should be a number between -1 and 1.

#### Success condition
The task is episodic, and in order to solve the environment,  an agent must get an average score of +30 over 100 consecutive episodes.

### Getting Started

1. Download the environment from one of the links below.  You need only select the environment that matches your operating system:

    - **_Version 1: One (1) Agent_**
        - Linux: [click here](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P2/Reacher/one_agent/Reacher_Linux.zip)
        - Mac OSX: [click here](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P2/Reacher/one_agent/Reacher.app.zip)
        - Windows (32-bit): [click here](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P2/Reacher/one_agent/Reacher_Windows_x86.zip)
        - Windows (64-bit): [click here](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P2/Reacher/one_agent/Reacher_Windows_x86_64.zip)

    - **_Version 2: Twenty (20) Agents_**
        - Linux: [click here](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P2/Reacher/Reacher_Linux.zip)
        - Mac OSX: [click here](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P2/Reacher/Reacher.app.zip)
        - Windows (32-bit): [click here](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P2/Reacher/Reacher_Windows_x86.zip)
        - Windows (64-bit): [click here](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P2/Reacher/Reacher_Windows_x86_64.zip)
    
    - **_For this project version 1 is used_**
    
    (_For Windows users_) Check out [this link](https://support.microsoft.com/en-us/help/827218/how-to-determine-whether-a-computer-is-running-a-32-bit-version-or-64) if you need help with determining if your computer is running a 32-bit version or 64-bit version of the Windows operating system.

    (_For AWS_) If you'd like to train the agent on AWS (and have not [enabled a virtual screen](https://github.com/Unity-Technologies/ml-agents/blob/master/docs/Training-on-Amazon-Web-Service.md)), then please use [this link](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P2/Reacher/one_agent/Reacher_Linux_NoVis.zip) (version 1) or [this link](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P2/Reacher/Reacher_Linux_NoVis.zip) (version 2) to obtain the "headless" version of the environment.  You will **not** be able to watch the agent without enabling a virtual screen, but you will be able to train the agent.  (_To watch the agent, you should follow the instructions to [enable a virtual screen](https://github.com/Unity-Technologies/ml-agents/blob/master/docs/Training-on-Amazon-Web-Service.md), and then download the environment for the **Linux** operating system above._)
    
2. Place the file in the DRLND GitHub repository, in the `continuous-control/` folder, and unzip (or decompress) the file. 

### Instructions

#### Train an agent

`python train_agent.py --[OPTION]`

[OPTION] : Reference on opt.py

#### Infer an agent

`python infer_agent.py --[OPTION]`

[OPTION] : Reference on opt.py

#### Read report

Follow the instructions in `Report.ipynb`.  

### Result summary

The agent success to solve the task in 318 episodes. The below plot shows the scores which the agent earns as the number of epiosdes increase.

![Score_plot][image2]

### Future works

For future works, A3C, A2C will be implemented and compare performances with current DDPG agent. Besides, different environment with multi-agent will be next step.

### References

[1] Timothy P. Lillicrap. CONTINUOUS CONTROL WITH DEEP REINFORCEMENT
LEARNING. ICLR 2016

[2] David Silver. Deterministic Policy Gradient Algorithms. 2014.

[3] Volodymyr Mnih. Playing Atari with Deep Reinforcement Learning. 2015.

[4] [Udacity deep-reinforcement-learning repository](https://github.com/udacity/deep-reinforcement-learning)