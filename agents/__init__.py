import subprocess
import sys
from packaging import version

reqs = subprocess.check_output([sys.executable, '-m', 'pip', 'freeze'])
installed_packages = [r.decode().split('==')[0] for r in reqs.split()]

if 'torch' in installed_packages:
    from agents.dqn_agent_pytorch import DQNAgent as DQNAgentPytorch
    from agents.nfsp_agent_pytorch import NFSPAgent as NFSPAgentPytorch
    from agents.dqn_transformer import DQNTransformer


from agents.random_agent import RandomAgent
