import warnings

from agents.random_agent import RandomAgent
import random
import numpy as np

def hash_obsevation(obs):
    try:
        val = hash(obs.tobytes())
        return val
    except AttributeError:
        try:
            return hash(obs)
        except TypeError:
            warnings.warn("Observation not an int or an Numpy array")
            return 0

def rand_iter(n):
    for x in range(n+1):
        random.randint(0, 1000)
        np.random.normal(size=100)

def gather_observations(env, actions, num_rand_steps):
    rand_iter(num_rand_steps)
    state, player_id = env.reset()
    rand_iter(num_rand_steps)

    action_idx = 0
    observations = []
    while not env.is_over() and action_idx < len(actions):
        # Agent plays
        rand_iter(num_rand_steps)
        legals = state['legal_actions']
        action = legals[actions[action_idx]%len(legals)]
        # Environment steps
        next_state, next_player_id = env.step(action)
        # Set the state and player
        state = next_state
        player_id = next_player_id

        action_idx += 1
        # Save state.
        if not env.game.is_over():
            observations.append(state)

    return observations

