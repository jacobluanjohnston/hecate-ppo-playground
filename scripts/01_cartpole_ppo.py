import gymnasium as gym
# defines observation_space, action_space, reset(), step(), close()
from stable_baselines3 import PPO
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.monitor import Monitor

# Build CartPole and monitor
env = Monitor(gym.make("CartPole-v1"))
print("observation space:", env.observation_space)
print("action space:     ", env.action_space)

# SB3's defaults (Schulman et al. 2017, arXiv 1707.06347)
model = PPO(
    "MlpPolicy", env, seed=0, verbose=0,
    # verbose=0 silences logs
    learning_rate=3e-4, # step size alpha
                            # (how far each gradient step moves the weights)
                            # (000 §4)

    n_steps=2048,       # rollout batch per update
                            # (how many env steps to collect before one update (the rollout))
                            # (003 §13, "collect a batch then update")

    batch_size=64,      # minibatch inside each epoch
                            # (2048 steps split into 64 chunks for the gradient passes)
                            # (minibatch SGD from 000)

    n_epochs=10,        # passes over the same rollout
                            # (over the same 2048 steps)
                            # (003 §7, s13)
                            # why PPO can reuse data and vanilla PG can't

    gamma=0.99,         # discount in the return G_t
                            # discount is a reward k steps ahead counts γ^k
                            #   The return from a step is a sum of all future
                            #   rewards in the episode. Undiscounted, a reward
                            #.  100 steps away counts the same as the next one.
                            #.  With γ = 0.99: next step counts 0.99, 10 steps away 0.90, 100 steps away 0.37, 500 steps away 0.007
                            # (001 §2, return G_t)

    gae_lambda=0.95,    # * smoothing of the advantage estimate
                            # GAE blends "observed return" and "critic's estimate" when
                            #   computing the advantage; 1.0 = pure observed return
                            # advantage A_t — 001 §12 (GAE itself is not in the notes;
                            #   it's Schulman et al. 2016, arXiv 1506.02438)

    clip_range=0.2,     # ε (epsilon) in the clipped ratio
                            # ε: the probability ratio new/old is clipped to [0.8, 1.2]
                            # (003 §8, the PPO clip)

    ent_coef=0.0,       # entropy bonus, off here
                            # bonus for keep action probabilities spread out
                            # not in the notes; PPO paper §5. In Step 6 we set 0.01.
                            # A choice to defend, not a default.
    # Hecate Alpha Plan 7.1
    # Create two hidden layers of 64 units (SB3's default for PPO)
    policy_kwargs=dict(net_arch=dict(pi=[64, 64], vf=[64, 64])),
    # MLP: Vector -> Hidden Layer -> Hidden Layer -> Output
    # Each layer a matrix multiply plus a nonlinearity
    # "MlpPolicy" tells SB3 to build two of them from the observation
    # - the actor `pi`, outputs action probabilities
    # - the critic `vf`, outputs one number
    #   (predicted return from this state)
)

# Before training
# Sample actions from the untrained actor
# (how PPO behaves while learning). Expected ~20
before, _ = evaluate_policy(model, env, n_eval_episodes=20, deterministic=False)
# Plays 20 full games of CartPole using the actor (random weights) and returns
# two numbers: the average score and the standard deviation.
# We keep the average in `before` and throw the deviation away
# deterministic=False: at each tick the actor outputs two probabilities, False
#   means roll the dice with those odds. An untrained actor's dice are nearly 50/50.

# Training: ~15 rounds of [collect 2048 steps -> returns & advantages
# -> 10 epochs of clipped updates]
model.learn(total_timesteps=30_000)
# Actual PPO
# Inside this one call, SB3 loops:
# 1. Play CartPole for 2,048 ticks (n_steps), rolling the actor's dice at each tick,
#   recording (observation, action, reward) for every tick. Episodes end and restart
#   inside this as the pole falls.
# 2. For each recorded tick, compute the return (the discounted sum of rewards that
#   followed) and the advantage (that return minus what the critic predicted for
#   that observation).
# 3. Ten times (n_epochs), go through those 2,048 ticks in chunks of 64 (batch_size)
#   and adjust the weights:
#       - the actor's, so actions with positive advantage become
#           more likely (capped by clip_range);
#       - the critic's, so its predictions move toward the observed returns
# 4. Back to 1 with the improved actor.

# After training: greedy (argmax) actions, how you would deploy it. Expect ~500.
after, _ = evaluate_policy(model, env, n_eval_episodes=20, deterministic=True)
print(f"before {before:.1f} after {after:.1f} (max 500)")
# Same measurement as before, on the trained actor.
# `deterministic=True` means don't roll dice, always take the probable action.
# That's how we'd use the policy for real.
# Expect = ~500, the cap.

# Deployment is the actor's weights only:
# Save, reload, act. No critic, no gradients.
# Writes the model to `cartpole_ppo.zip` and reads it back into a new variable.
# Pointless in a script, except as proof: a trained policy is a file.
# You can train on one machine and run it anywhere.
model.save("cartpole_ppo")
loaded = PPO.load("cartpole_ppo")
# Start a fresh game. reset() returns the first observation (4 numbers) and an `info`
# dict we ignore.
obs, _ = env.reset(seed=0)
# Score so far, and "is the game over?" Then loop until it is
total, done = 0.0, False
while not done:
    # Observation -> Actor -> Action
    # Hand the 4 numbers to the reloaded actor; get back 0 or 1.
    # This is the entire runtime of a deployed policy: one line.
    # (`predict` also returns a hidden state used by recurrent policies, we don't have one, _)
    action, _ = loaded.predict(obs, deterministic=True)
    # The Gynasium step contract
    # Apply the push. Env returns five things in the Gynasium step contract
    # `obs` - the new 4 numbers, `r` - the reward for this tick (+1 in CartPole)
    # `terminated` - True if the pole fell or cart left track (Task itself ended)
    # `truncated` - True if the 500-tick time limit hit (cut off, not failed)
    # `info` - extra info, ignored (human logs)
    obs, r, terminated, truncated, _ = env.step(action)
    # Add the reward to the score; stop when either ending flag is set
    total += r
    done = terminated or truncated
    # Expect 500
print("reloaded policy, one episode:", total)

# Observation and action vectors -> actor and critic
# Each tick (1/2048 ticks)
# --------------------------------------------------
# 1. Environment hands over the observation: [0.02, -0.15, 0.03, 0.31] (position, velocity, angle, angular velocity — the pole is tipping right and speeding up).
# 2. The actor MLP takes those 4 numbers and outputs 2 probabilities: [0.3, 0.7] → 70% "push right." Sample → action 1.
# 3. The critic MLP takes the same 4 numbers and outputs one number: 43.2 → "from a state like this, I expect about 43 more reward before the pole falls."
# 4. env.step(1) applies the push; back comes reward +1 and the next observation. Repeat.
#
# For each tick, compute what actually happened afterward, and compare with what the critic said
# "Pushing right here went better than expected" PPO then raises the probability of "right" in states like that
# and nudges the critic's prediction (critic update, plain regression). States where the
# advantage is negative get the opposite.
# Over ~15 rounds of this, the actor's probabilities become sharp and correct, and
# the critic becomes an accurate forecaster. It is never used after training. Existed
# only to make the actor's learning signal less noisy.

# For the router, swap in:
# - Observation = the 13 numbers from advisor + telemetry
# - Actor output = [P(M1), P(M2)];
# - Critic output = expected 1[resolved] - lambda * cost
# - Reward per step = −λ·Δcost/C_ref, plus 1 if resolved.

# Sources
# -------
# - Schulman, Wolski, Dhariwal, Radford, Klimov. Proximal Policy Optimization Algorithms. arXiv:1707.06347, 2017.
# - Schulman, Moritz, Levine, Jordan, Abbeel. High-Dimensional Continuous Control Using Generalized Advantage Estimation. arXiv:1506.02438, 2016.
# - Raffin et al. Stable-Baselines3: Reliable Reinforcement Learning Implementations. JMLR 22(268), 2021 — and the PPO docs page for defaults.
# - Towers et al. Gymnasium: A Standard Interface for Reinforcement Learning Environments. arXiv:2407.17032, 2024.
# - Nath, RL notes 000–004