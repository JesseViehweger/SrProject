import random

import numpy as np
import gymnasium as gym
from gym.spaces import Discrete, MultiDiscrete
import torch
from torch import nn
import matplotlib.pyplot as plt
import pickle


class TinyMDP(gym.Env):
    """
    An implementation of the small markov decision process illustrated at
    https://en.wikipedia.org/wiki/Markov_decision_process#/media/File:Markov_Decision_Process.svg
    """

    def __init__(self, **kwargs):
        # nested dict of state, action -> new_state -> probability
        self.transition_probabilities = {
            0: {
                0: {0: 0.5, 2: 0.5},
                1: {2: 1.0}
            },
            1: {
                0: {1: 0.1, 0: 0.7, 2: 0.2},
                1: {1: 0.95, 2: 0.05}
            },
            2: {
                0: {2: 0.6, 0: 0.4},
                1: {2: 0.4, 0: 0.3, 1: 0.3}
            }
        }

        # dict of (state, action, new_state) -> reward
        self.reward_probabilities = {
            (1, 0, 0): 5,
            (2, 1, 0): -1
        }

        self.current_state = np.random.choice(3)

    def reset(self, **kwargs) -> np.ndarray:
        """
        randomize state and return observation
        :return: observation for the new state
        """
        self.current_state = np.random.randint(3)
        return self._get_observation()

    def _get_observation(self) -> np.ndarray:
        """
        convert numerical state number into a one-hot state vector
        :return: numpy array with one element set to one (representing current state)
        """
        obs = np.zeros(3)
        obs[self.current_state] = 1
        return obs

    def step(self, action: int) -> tuple[np.ndarray, float, bool, bool, dict]:
        """
        execute one step in the environment
        :param action: 0, 1, or 2
        :return: new state, reward, and default values for terminated, truncated, info, and done
        """
        # get dict of possible new states & probabilities
        possible_new_states = self.transition_probabilities[self.current_state][action]

        # randomly choose next state, per the probabilities
        new_state = np.random.choice(list(possible_new_states.keys()), p=list(possible_new_states.values()))

        # decide reward
        reward = self.reward_probabilities.get((self.current_state, action, new_state), 0)

        self.current_state = new_state
        return self._get_observation(), reward, False, False, {}

    def render(self):
        pass

    def close(self):
        pass


class MinimalDQN:
    """
    A *very* minimal Deep Q Network
    Thanks to https://towardsdatascience.com/deep-q-learning-for-the-cartpole-44d761085c2f for getting started
    todo: implement a memory to store transitions
    todo: batch gradient descent - sample a batch of transitions from the memory & run update using the whole batch
    todo: (optional?) use separate value and policy networks as described in the link above
    todo: allow more than one hidden layer
    todo: add a parameter to dial up dropout/regularization-based depression simulation
    """

    def __init__(self, n_inputs, hidden_layer_size, n_outputs, memory_size: int, batch_size: int, learning_rate=1e-2, gamma=0.95, epsilon=0.1, device='cpu'):

        self.device = torch.device("cuda" if device == 'cuda' and torch.cuda.is_available() else "cpu")
        print(f'setting device to {self.device}')

        # construct the actual neural network. It will predict values for each action given a state
        self.n_outputs = n_outputs
        self.net = nn.Sequential(
            nn.Linear(n_inputs, hidden_layer_size),
            nn.Tanh(),
            nn.Linear(hidden_layer_size, n_outputs)
        )
        self.net.to(self.device)

        # attributes related to network training
        self.learning_rate = learning_rate
        self.loss_fn = torch.nn.MSELoss()
        self.optimizer = torch.optim.Adam(self.net.parameters(), lr=learning_rate)

        # attributes related to RL behavior
        self.gamma = gamma
        self.epsilon = epsilon

        # currently unused
        self.memory_size = memory_size
        self.batch_size = batch_size

    def select_action(self, state: np.ndarray) -> int:
        """
        select an action using epsilon-greedy selection
        :param state: state vector
        :return: integer representing the action chosen
        """

        # with probability set by epsilon, choose a random action...
        if random.random() < self.epsilon:
            return np.random.choice(self.n_outputs)

        # ...otherwise compute q values for each action given this state, and pick the action with max value
        # use torch.no_grad b/c this use of the network is unrelated to learning, so doesn't need gradients calculated
        with torch.no_grad():
            action_q_values = self.net(torch.tensor(state.astype(np.float32), device=self.device))
            max_q, index = action_q_values.max(0)
        return index.item()

    def update(self, state: np.ndarray, action: int, reward: float, new_state: np.ndarray) -> None:
        """
        use the given state transition to tune the network weights, giving better value predictions for next time
        :param state: beginning state vector
        :param action: action taken
        :param reward: reward received
        :param new_state: new state after executing the action
        :return: None
        """

        # convert state vectors to tensor format
        state = torch.tensor(state.astype(np.float32), device=self.device).reshape(1, -1)
        new_state = torch.tensor(new_state.astype(np.float32), device=self.device).reshape(1, -1)

        # get network's current value estimates for the state
        action_q_values = self.net(state)

        # get predicted value of next state (max value available from that state)
        with torch.no_grad():
            new_state_value, _ = self.net(new_state).max(1)

        # get target value estimates, based on actual rewards and network's predictions of next-state value
        target_q_values = action_q_values.clone().detach()
        target_q_values[0, action] = reward + self.gamma * new_state_value

        # tune network to predict something closer to target q values next time
        loss = self.loss_fn(action_q_values, target_q_values)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()


if __name__ == '__main__':

    # instantiate environment
    env = TinyMDP()

    # instantiate DQN
    # agent = MinimalDQN(
    #     n_inputs=3,
    #     hidden_layer_size=4,
    #     n_outputs=2,
    #     memory_size=None,
    #     batch_size=None,
    # )
    with open('test.pkl', 'rb') as f:
        agent = pickle.load(f)

    # run a number of steps in the environment
    n_steps = 2000
    reward_history = np.zeros(n_steps)

    state = env.reset()
    for step in range(n_steps):
        action = agent.select_action(state)
        new_state, reward, _, _, _ = env.step(action)
        agent.update(state, action, reward, new_state)

        state = new_state
        reward_history[step] = reward

    # plot cumulative reward
    plt.plot(reward_history.cumsum())
    plt.title('cumulative reward')
    plt.xlabel('step #')
    plt.show()

    with open('test.pkl', 'wb') as f:
        pickle.dump(agent, f)
