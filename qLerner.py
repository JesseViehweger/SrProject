# plot this info in a csv
# episode num, mandatory, optional, step count

import gymnasium as gym
import math
import random
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from collections import namedtuple, deque
from itertools import count
from PIL import Image

from pathlib import Path
import pickle
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torchvision.transforms as T

env = gym.make('MiniGrid-Custom-TestEnv', render_mode='rgb_array').unwrapped
env = gym.make('MiniGrid-Custom-TestEnv', render_mode='rgb_array')
env.reset(seed=None)
# env.step_count = 0

env.step(0)

# set up matplotlib
is_ipython = 'inline' in matplotlib.get_backend()
if is_ipython:
    from IPython import display

# plt.ion()

# if gpu is to be used
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

#get desired dropout
import argparse
parser = argparse.ArgumentParser()
parser.add_argument(
    "--drop", help="amount of dropout to be used in the model", default=0
)
args = parser.parse_args()
drop = args.drop
drop = int(drop)


######################################################################
# Replay Memory
# -------------
#
# We'll be using experience replay memory for training our DQN. It stores
# the transitions that the agent observes, allowing us to reuse this data
# later. By sampling from it randomly, the transitions that build up a
# batch are decorrelated. It has been shown that this greatly stabilizes
# and improves the DQN training procedure.
#
# For this, we're going to need two classses:
#
# -  ``Transition`` - a named tuple representing a single transition in
#    our environment. It essentially maps (state, action) pairs
#    to their (next_state, reward) result, with the state being the
#    screen difference image as described later on.
# -  ``ReplayMemory`` - a cyclic buffer of bounded size that holds the
#    transitions observed recently. It also implements a ``.sample()``
#    method for selecting a random batch of transitions for training.
#

Transition = namedtuple('Transition',
                        ('state', 'action', 'next_state', 'reward'))

class ReplayMemory(object):

    def __init__(self, capacity):
        self.memory = deque([],maxlen=capacity)

    def push(self, *args):
        """Save a transition"""
        self.memory.append(Transition(*args))

    def sample(self, batch_size):
        return random.sample(self.memory, batch_size)

    def __len__(self):
        return len(self.memory)

class DQN(nn.Module):

    def __init__(self, h, w, outputs, drop):
        super(DQN, self).__init__()
        # from convolutional2d, change to linear layers followed by an activation(49 inputs 3 outputs, a few layers in the middle 49-20-3 etc) 
        #example
        self.layer1 = nn.Linear(147, 20)
        self.activation1 = nn.Tanh()
        self.layer2 = nn.Linear(20, 10)
        self.activation2 = nn.Tanh()
        self.layer3 = nn.Linear(10, 3)
        self.dropout = nn.Dropout(p=(drop/100))


        # self.activation3 = nn.Tanh()
        
        # self.head = nn.Linear(3,1)

        # Number of Linear input connections depends on output of conv2d layers
        # and therefore the input image size, so compute it.
        # def conv2d_size_out(size, kernel_size = 5, stride = 2):
        #     return (size - (kernel_size - 1) - 1) // stride  + 1
        # convw = conv2d_size_out(conv2d_size_out(conv2d_size_out(w)))
        # convh = conv2d_size_out(conv2d_size_out(conv2d_size_out(h)))
        # linear_input_size = convw * convh * 32
        # self.head = nn.Linear(linear_input_size, outputs)

    # Called with either one element to determine next action, or a batch
    # during optimization. Returns tensor([[left0exp,right0exp]...]).
    def forward(self, x):
        
        # x = self.activation1(self.layer1(x.type(torch.float32)))
        # x = self.activation2(self.layer2(x))
        # x = self.layer3(x)

        x = self.activation1(self.dropout(self.layer1(x.type(torch.float32))))
        x = self.activation2(self.dropout(self.layer2(x)))
        x = self.layer3(x)

        return x

env.reset()
# plt.figure()
# plt.show()

######################################################################
# Training
# --------
#
# Hyperparameters and utilities
# ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
# This cell instantiates our model and its optimizer, and defines some
# utilities:
#
# -  ``select_action`` - will select an action accordingly to an epsilon
#    greedy policy. Simply put, we'll sometimes use our model for choosing
#    the action, and sometimes we'll just sample one uniformly. The
#    probability of choosing a random action will start at ``EPS_START``
#    and will decay exponentially towards ``EPS_END``. ``EPS_DECAY``
#    controls the rate of the decay.
# -  ``plot_durations`` - a helper for plotting the durations of episodes,
#    along with an average over the last 100 episodes (the measure used in
#    the official evaluations). The plot will be underneath the cell
#    containing the main training loop, and will update after every
#    episode.
#

BATCH_SIZE = 128
GAMMA = 0.999
EPS_START = 0.9
EPS_END = 0.05
EPS_DECAY = 200
TARGET_UPDATE = 10

# Get screen size so that we can initialize layers correctly based on shape
# returned from AI gym. Typical dimensions at this point are close to 3x40x90
# which is the result of a clamped and down-scaled render buffer in get_screen()
# init_screen = get_screen()
# _, _, screen_height, screen_width = init_screen.shape

# Get number of actions from gym action space

# save networks for later running.
n_actions = env.action_space.n
#if pickle exists use it instead
file = "test" + str(drop) + ".pkl"
path = Path(file)

if(path.is_file()):
    infile = open(file,'rb')
    policy_net = pickle.load(infile)
    infile.close()
    print("Reusing Trained Netowrk")

policy_net = DQN(8, 14, n_actions, drop).to(device)
target_net = DQN(8, 14, n_actions, drop).to(device)
target_net.load_state_dict(policy_net.state_dict())
target_net.eval()

optimizer = optim.RMSprop(policy_net.parameters())
memory = ReplayMemory(10000)


steps_done = 0


def select_action(state):
    global steps_done
    sample = random.random()
    eps_threshold = EPS_END + (EPS_START - EPS_END) * \
        math.exp(-1. * steps_done / EPS_DECAY)
    steps_done += 1
    if sample > eps_threshold:
        with torch.no_grad():
            # t.max(1) will return largest column value of each row.
            # second column on max result is index of where max element was
            # found, so we pick action with the larger expected reward.
            
            return policy_net(state).argmax()
    else:
        #may need to change this to only sampe the actions we want to take in the environment
        return torch.tensor([[random.randrange(0,3)]], device=device, dtype=torch.long)


episode_durations = []


def plot_durations():
    plt.figure(2)
    plt.clf()
    durations_t = torch.tensor(episode_durations, dtype=torch.float)
    plt.title('Training...')
    plt.xlabel('Episode')
    plt.ylabel('Duration')
    plt.plot(durations_t.numpy())
    # Take 100 episode averages and plot them too
    if len(durations_t) >= 100:
        means = durations_t.unfold(0, 100, 1).mean(1).view(-1)
        means = torch.cat((torch.zeros(99), means))
        plt.plot(means.numpy())

    plt.pause(0.001)  # pause a bit so that plots are updated
    if is_ipython:
        display.clear_output(wait=True)
        display.display(plt.gcf())

        ######################################################################
# Training loop
# ^^^^^^^^^^^^^
#
# Finally, the code for training our model.
#
# Here, you can find an ``optimize_model`` function that performs a
# single step of the optimization. It first samples a batch, concatenates
# all the tensors into a single one, computes :math:`Q(s_t, a_t)` and
# :math:`V(s_{t+1}) = \max_a Q(s_{t+1}, a)`, and combines them into our
# loss. By definition we set :math:`V(s) = 0` if :math:`s` is a terminal
# state. We also use a target network to compute :math:`V(s_{t+1})` for
# added stability. The target network has its weights kept frozen most of
# the time, but is updated with the policy network's weights every so often.
# This is usually a set number of steps but we shall use episodes for
# simplicity.
#

def optimize_model():
    if len(memory) < BATCH_SIZE:
        return
    transitions = memory.sample(BATCH_SIZE)

    # print(transitions[0])

    # Transpose the batch (see https://stackoverflow.com/a/19343/3343043 for
    # detailed explanation). This converts batch-array of Transitions
    # to Transition of batch-arrays.
    batch = Transition(*zip(*transitions))

    # Compute a mask of non-final states and concatenate the batch elements
    # (a final state would've been the one after which simulation ended)
    non_final_mask = torch.tensor(tuple(map(lambda s: s is not None,
                                          batch.next_state)), device=device, dtype=torch.bool)
    non_final_next_states = torch.cat([s for s in batch.next_state
                                                if s is not None])
    
    
    state_batch = torch.cat(batch.state)

    

    action_batch = torch.cat(batch.action)
    reward_batch = torch.cat(batch.reward)

    # Compute Q(s_t, a) - the model computes Q(s_t), then we select the
    # columns of actions taken. These are the actions which would've been taken
    # for each batch state according to policy_net

    #am i feeding this the wrong dimensioned/formatted. since the size of state_batch is is 
    # 18816 and 18816/147 = 128 (147 is the first layers input size) is this batch size 128?
    # print(state_batch.size())
   
    
    state_batch = state_batch.reshape([BATCH_SIZE, 147])

    # print(state_batch.size())

    # print(action_batch.size())

    #need a better explenation of why gather is being used here.

    # state_action_values = policy_net(state_batch).gather(1, action_batch)
    state_action_values = policy_net(state_batch).gather(0, action_batch)

    # Compute V(s_{t+1}) for all next states.
    # Expected values of actions for non_final_next_states are computed based
    # on the "older" target_net; selecting their best reward with max(1)[0].
    # This is merged based on the mask, such that we'll have either the expected
    # state value or 0 in case the state was final.
    next_state_values = torch.zeros(BATCH_SIZE, device=device)

    # print(non_final_next_states.size())
    non_final_next_states = non_final_next_states.reshape([BATCH_SIZE, 147])


    next_state_values[non_final_mask] = target_net(non_final_next_states).max(1)[0].detach()
    # Compute the expected Q values
    expected_state_action_values = (next_state_values * GAMMA) + reward_batch

    # Compute Huber loss
    criterion = nn.SmoothL1Loss()
    loss = criterion(state_action_values, expected_state_action_values.unsqueeze(1))

    # Optimize the model
    optimizer.zero_grad()
    loss.backward()
    for param in policy_net.parameters():
        param.grad.data.clamp_(-1, 1)
    optimizer.step()


######################################################################
#
# Below, you can find the main training loop. At the beginning we reset
# the environment and initialize the ``state`` Tensor. Then, we sample
# an action, execute it, observe the next screen and the reward (always
# 1), and optimize our model once. When the episode ends (our model
# fails), we restart the loop.
#
# Below, `num_episodes` is set small. You should download
# the notebook and run lot more epsiodes, such as 300+ for meaningful
# duration improvements.
#
i = 0
num_episodes = 25
blueNum = 0
greenNum = 0
for i_episode in range(num_episodes):
    # Initialize the environment and state
    env.reset()

    state = env.gen_obs()
    image = state['image']
    image = image.flatten()
    # print(type(image))
    #maybe i dont need below conversion
    image = torch.from_numpy(image)

    state = image

    # state = 
    step = 0
    done = False

    #plot counts of obtaining each goal (green and blue) for a depressed vs healthy model.
    #for t in count():
    while done != True:
        # Select and perform an action
        action = select_action(state)
        obs, reward, done, _, _ = env.step(action.item())

        step = step + 1
        # print(step)
        if (reward != 0):
            # print(reward)
            if((reward*100)%2 == 1):
                print("green")
                greenNum = greenNum + 1
            else:
                print("blue")
                blueNum = blueNum + 1
           

        reward = torch.tensor([reward], device=device)

        test = torch.tensor([0], device=device)
        
        # if (reward != test):
        #     print(step)
        #     print(reward)

        # Observe new state
        obs = torch.tensor(obs['image'])
        obs = obs.flatten()

        # Store the transition in memory
    

        if(action.dim() == 0):
            action = torch.tensor([[action]], device=device, dtype=torch.long)
            action = action

        memory.push(state, action, obs, reward)

        # Move to the next state
        state = obs

        # Perform one step of the optimization (on the policy network)
        optimize_model()
        if done:
            episode_durations.append(step + 1)
            # plot_durations()
            break
    # Update the target network, copying all weights and biases in DQN
    if i_episode % TARGET_UPDATE == 0:
        target_net.load_state_dict(policy_net.state_dict())

    print("Episode complete!")
data = {'Optional':blueNum, 'Mandatory':greenNum}

diffRewards = list(data.keys())
values = list(data.values())
  
# creating the bar plot
plt.bar(diffRewards, values, color ='maroon',
        width = 0.4)
plt.show()


with open(file, 'wb') as f:
    pickle.dump(policy_net, f)

print('Complete')
env.render()
env.close()
# plt.ioff()
# plt.show()

######################################################################
# Here is the diagram that illustrates the overall resulting data flow.
#
# .. figure:: /_static/img/reinforcement_learning_diagram.jpg
#
# Actions are chosen either randomly or based on a policy, getting the next
# step sample from the gym environment. We record the results in the
# replay memory and also run optimization step on every iteration.
# Optimization picks a random batch from the replay memory to do training of the
# new policy. "Older" target_net is also used in optimization to compute the
# expected Q values; it is updated occasionally to keep it current.
#

