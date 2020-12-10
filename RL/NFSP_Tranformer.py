import os

import torch

from agents import NFSPTranformerAgent
from agents import RandomAgent
from envs import DoudizhuEnv
from games.doudizhu import Player,Dealer,Round,Judger,Game
from utils import Logger, tournament
from rootpath import Root_Path
env = DoudizhuEnv()
env_eval = DoudizhuEnv()


evaluate_every = 1000
evaluate_num = 1000
episode_num = 1000

memory_init_size = 1000
train_every = 64

log_dir = './experiments/nfsp_transfomer_result/'
agents = []
for i in range(env.player_num):
    agent = NFSPTranformerAgent(
        scope='DouDiZhuTransformerSelfPlay',
        action_num=env.action_num,
        state_shape=env.state_shape,
        hidden_layers_sizes=[512,1024,2048,1024,512],
        anticipatory_param=0.5,
        batch_size=256,
        train_every=train_every,
        rl_learning_rate=0.1,
        sl_learning_rate=0.05,
        min_buffer_size_to_learn=memory_init_size,
        q_replay_memory_size=int(1e5),
        q_replay_memory_init_size=memory_init_size,
        q_train_every=train_every,
        q_batch_size=256,
        evaluate_with='average_policy',
        device=None)
    agents.append(agent)

random_agent = RandomAgent(action_num=env_eval.action_num)
env.set_agents(agents)
env_eval.set_agents([agents[0], random_agent, random_agent])
logger = Logger(log_dir)
for episode in range(episode_num):
    for agent in agents:
        agent.sample_episode_policy()

    # Generate data from the environment
    trajectories, _ = env.run(is_training=True)

    # Feed transitions into agent memory, and train the agent
    for i in range(env.player_num):
        for ts in trajectories[i]:
            agents[i].feed(ts)

    # Evaluate the performance. Play with random agents.
    if episode % evaluate_every == 0:
        logger.log_performance(env.timestep, tournament(env_eval, evaluate_num)[0])

logger.close_files()

# Plot the learning curve
logger.plot('NFSP')

# Save model
save_dir = 'models/doudizhu_nfsp'
state_dict = {}
for agent in agents:
    state_dict.update(agent.get_state_dict())
torch.save(state_dict, os.path.join(save_dir, 'model.pth'))

