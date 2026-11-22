# For dtype checks
import numpy as np
# SB3's validator
from stable_baselines3.common.env_checker import check_env
# Our class and feature table
from hecate_ppo.routing_env import HecateRoutingEnv, FEATURES

# Builds an env and hands to check_env, which calls reset, calls step a few times with random
# valid actions
def test_check_env_passes():
    check_env(HecateRoutingEnv(), warn=True)

# Check observation is a 13-vector of float32.
def test_shapes_and_dtype():
    env = HecateRoutingEnv()
    # One step with action 0; unpack the five values; assert reward is a float and the two
    # flags are bools.
    obs, info = env.reset(seed=0)
    assert obs.shape == (len(FEATURES),) and obs.dtype == np.float32
    obs, r, term, trunc, info = env.step(0)
    assert isinstance(r, float) and isinstance(term, bool) and isinstance(trunc, bool)

# Build an env with a 5-interval budget, run the standard loop counting steps, assert it stopped
# within 5.
# This is the six-line loop from CartPole pointed at the environment.
def test_episode_ends_at_budget():
    env = HecateRoutingEnv(max_intervals=5)
    env.reset(seed=0)
    done, n = False, 0
    while not done:
        _, _, term, trunc, _ = env.step(0)
        done = term or trunc
        n += 1
    assert n <= 5