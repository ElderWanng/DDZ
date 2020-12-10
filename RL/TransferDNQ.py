# from agents import DQNAgentPytorch as DQNAgent
from agents import DQNTransformer
from agents import RandomAgent
from envs import DoudizhuEnv
from games.doudizhu import Player,Dealer,Round,Judger,Game
from utils import Logger, tournament
from rootpath import Root_Path
env = DoudizhuEnv()
env_eval = DoudizhuEnv()


evaluate_every = 2
evaluate_num = 100
episode_num = 100

memory_init_size = 100

train_every = 1
agent = DQNTransformer(
    scope='DouDiZhuTransformer',
    action_num=env.action_num,
    replay_memory_init_size=memory_init_size,
    train_every=train_every,
    state_shape=env.state_shape,
    mlp_layers=[512,512]
)
log_dir = Root_Path+'./experiment_log/dqn/'
logger = Logger(log_dir)
random_agent = RandomAgent(action_num=env_eval.action_num)
env.set_agents([agent, random_agent, random_agent])
env_eval.set_agents([agent, random_agent, random_agent])


for episode in range(episode_num):

    # Generate data from the environment
    trajectories, _ = env.run(is_training=True)

    # Feed transitions into agent memory, and train the agent
    for ts in trajectories[0]:
        agent.feed(ts)

    if episode % evaluate_every == 0:
        logger.log_performance(env.timestep, tournament(env_eval, evaluate_num)[0])

logger.close_files()
logger.plot('DQN')

