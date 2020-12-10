import collections
import enum
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from agents.dqn_transformer import DQNTransformer
from agents.ReservoirBuffer import ReservoirBuffer
from utils.utils import remove_illegal

Transition = collections.namedtuple('Transition', 'info_state action_probs')
MODE = enum.Enum('mode', 'best_response average_policy')

class NFSPTranformerAgent(object):
    def __init__(self,
                 scope,
                 action_num=4,
                 state_shape=None,
                 hidden_layers_sizes = None,
                 reservoir_buffer_capacity=int(1e5),
                 anticipatory_param=0.1,
                 batch_size=256,
                 train_every=1,
                 rl_learning_rate=0.1,
                 sl_learning_rate=0.005,
                 min_buffer_size_to_learn=1000,
                 q_replay_memory_size=30000,
                 q_replay_memory_init_size=1000,
                 q_update_target_estimator_every=1000,
                 q_discount_factor=0.99,
                 q_epsilon_start=0.06,
                 q_epsilon_end=0,
                 q_epsilon_decay_steps=int(1e6),
                 q_batch_size=256,
                 q_train_every=1,
                 evaluate_with='average_policy',
                 device=None):
        self.use_raw = False
        self._scope = scope
        self._action_num = action_num
        self._state_shape = state_shape
        self._layer_sizes = hidden_layers_sizes + [action_num]
        self._batch_size = batch_size
        self._train_every = train_every
        self._sl_learning_rate = sl_learning_rate
        self._anticipatory_param = anticipatory_param
        self._min_buffer_size_to_learn = min_buffer_size_to_learn

        self._reservoir_buffer = ReservoirBuffer(reservoir_buffer_capacity)
        self._prev_timestep = None
        self._prev_action = None
        self.evaluate_with = evaluate_with

        if device is None:
            self.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        else:
            self.device = device

        self.total_t =0
        self._step_counter = 0

        self._rl_agent = DQNTransformer(
            scope = scope+"transformer",
            replay_memory_size=q_replay_memory_size,
            replay_memory_init_size=q_replay_memory_init_size,
            update_target_estimator_every=q_update_target_estimator_every,
            discount_factor=q_discount_factor,
            epsilon_start=q_epsilon_start,
            epsilon_end=q_epsilon_end,
            epsilon_decay_steps=q_epsilon_decay_steps,
            batch_size=q_batch_size,
            action_num=action_num,
            state_shape=state_shape,
            train_every=q_train_every,
            mlp_layers=None,
            learning_rate=rl_learning_rate,
            device=None
            )
        self._build_model()
        self.sample_episode_policy()

    def _build_model(self):
        policy_network = AveragePolicyNetwork(self._action_num,
                                              self._state_shape,
                                              self._layer_sizes)
        policy_network = policy_network.to(self.device)
        self.policy_network = policy_network
        self.policy_network.eval()

        # xavier init
        for p in self.policy_network.parameters():
            if len(p.data.shape) > 1:
                nn.init.xavier_uniform_(p.data)

        # configure optimizer
        self.policy_network_optimizer = torch.optim.Adam(self.policy_network.parameters(), lr=self._sl_learning_rate)
    def feed(self, ts):
        ''' Feed data to inner RL agent

        Args:
            ts (list): A list of 5 elements that represent the transition.
        '''
        self._rl_agent.feed(ts)
        self.total_t += 1
        if self.total_t > 0 and len(
                self._reservoir_buffer) >= self._min_buffer_size_to_learn and self.total_t % self._train_every == 0:
            sl_loss = self.train_sl()
            print('\rINFO - Agent {}, step {}, sl-loss: {}'.format(self._scope, self.total_t, sl_loss), end='')

    def step(self, state):
        ''' Returns the action to be taken.

        Args:
            state (dict): The current state

        Returns:
            action (int): An action id
        '''
        obs = state['obs']
        legal_actions = state['legal_actions']
        if self._mode == MODE.best_response:
            probs = self._rl_agent.predict(obs)
            self._add_transition(obs, probs)

        elif self._mode == MODE.average_policy:
            probs = self._act(obs)

        probs = remove_illegal(probs, legal_actions)
        action = np.random.choice(len(probs), p=probs)

        return action

    def eval_step(self, state):
        ''' Use the average policy for evaluation purpose

        Args:
            state (dict): The current state.

        Returns:
            action (int): An action id.
        '''
        if self.evaluate_with == 'best_response':
            action, probs = self._rl_agent.eval_step(state)
        elif self.evaluate_with == 'average_policy':
            obs = state['obs']
            legal_actions = state['legal_actions']
            probs = self._act(obs)
            probs = remove_illegal(probs, legal_actions)
            action = np.random.choice(len(probs), p=probs)
        else:
            raise ValueError("'evaluate_with' should be either 'average_policy' or 'best_response'.")
        return action, probs

    def sample_episode_policy(self):
        ''' Sample average/best_response policy
        '''
        if np.random.rand() < self._anticipatory_param:
            self._mode = MODE.best_response
        else:
            self._mode = MODE.average_policy

    def _act(self, info_state):
        ''' Predict action probability givin the observation and legal actions
            Not connected to computation graph
        Args:
            info_state (numpy.array): An obervation.

        Returns:
            action_probs (numpy.array): The predicted action probability.
        '''
        info_state = np.expand_dims(info_state, axis=0)
        info_state = torch.from_numpy(info_state).float().to(self.device)

        with torch.no_grad():
            log_action_probs = self.policy_network(info_state).cpu().numpy()

        action_probs = np.exp(log_action_probs)[0]

        return action_probs

    def _add_transition(self, state, probs):
        ''' Adds the new transition to the reservoir buffer.

        Transitions are in the form (state, probs).

        Args:
            state (numpy.array): The state.
            probs (numpy.array): The probabilities of each action.
        '''
        transition = Transition(
            info_state=state,
            action_probs=probs)
        self._reservoir_buffer.add(transition)

    def train_sl(self):
        ''' Compute the loss on sampled transitions and perform a avg-network update.

        If there are not enough elements in the buffer, no loss is computed and
        `None` is returned instead.

        Returns:
            loss (float): The average loss obtained on this batch of transitions or `None`.
        '''
        if (len(self._reservoir_buffer) < self._batch_size or
                len(self._reservoir_buffer) < self._min_buffer_size_to_learn):
            return None

        transitions = self._reservoir_buffer.sample(self._batch_size)
        info_states = [t.info_state for t in transitions]
        action_probs = [t.action_probs for t in transitions]

        self.policy_network_optimizer.zero_grad()
        self.policy_network.train()

        # (batch, state_size)
        info_states = torch.from_numpy(np.array(info_states)).float().to(self.device)

        # (batch, action_num)
        eval_action_probs = torch.from_numpy(np.array(action_probs)).float().to(self.device)

        # (batch, action_num)
        log_forecast_action_probs = self.policy_network(info_states)

        ce_loss = - (eval_action_probs * log_forecast_action_probs).sum(dim=-1).mean()
        ce_loss.backward()

        self.policy_network_optimizer.step()
        ce_loss = ce_loss.item()
        self.policy_network.eval()

        return ce_loss

    def get_state_dict(self):
        ''' Get the state dict to save models

        Returns:
            (dict): A dict of model states
        '''
        state_dict = self._rl_agent.get_state_dict()
        state_dict[self._scope] = self.policy_network.state_dict()
        return state_dict

    def load(self, checkpoint):
        ''' Load model

        Args:
            checkpoint (dict): the loaded state
        '''
        self.policy_network.load_state_dict(checkpoint[self._scope])

class AveragePolicyNetwork(nn.Module):
    '''
    Approximates the history of action probabilities
    given state (average policy). Forward pass returns
    log probabilities of actions.
    '''

    def __init__(self, action_num=2, state_shape=None, mlp_layers=None):
        ''' Initialize the policy network.  It's just a bunch of ReLU
        layers with no activation on the final one, initialized with
        Xavier (sonnet.nets.MLP and tensorflow defaults)

        Args:
            action_num (int): number of output actions
            state_shape (list): shape of state tensor for each sample
            mlp_laters (list): output size of each mlp layer including final
        '''
        super(AveragePolicyNetwork, self).__init__()

        self.action_num = action_num
        self.state_shape = state_shape
        self.mlp_layers = mlp_layers

        # set up mlp w/ relu activations
        layer_dims = [np.prod(self.state_shape)] + self.mlp_layers
        mlp = [nn.Flatten()]
        mlp.append(nn.BatchNorm1d(layer_dims[0]))
        for i in range(len(layer_dims) - 1):
            mlp.append(nn.Linear(layer_dims[i], layer_dims[i + 1]))
            if i != len(layer_dims) - 2:  # all but final have relu
                mlp.append(nn.ReLU())
        self.mlp = nn.Sequential(*mlp)

    def forward(self, s):
        ''' Log action probabilities of each action from state

        Args:
            s (Tensor): (batch, state_shape) state tensor

        Returns:
            log_action_probs (Tensor): (batch, action_num)
        '''
        logits = self.mlp(s)
        log_action_probs = F.log_softmax(logits, dim=-1)
        return log_action_probs



















