# Study notes — PPO, Gymnasium, and the router

## 1. An environment

A Gymnasium environment is a class with two declared spaces and two main methods.
- `observation_space`: the shape and bounds of what the environment hands the agent each step.
  `Box(low, high, shape, dtype)` = an array of floats with per-entry bounds; `Discrete(n)` = one integer in 0..n-1.
- `action_space`: the set of choices the agent may make.
- `reset(seed=...)` -> `(observation, info)`. Starts an episode. The seed fixes the environment's random
  generator, so the same seed gives the same hidden task.
- `step(action)` -> `(observation, reward, terminated, truncated, info)`. `terminated` = the task itself
  ended; `truncated` = cut off from outside (a time limit). `info` is a dict for humans, never for learning.
- An episode is one run from reset until terminated or truncated.

The trainer (SB3) reads the spaces to size its networks, then only ever calls reset and step. That is why one
training script can drive CartPole today and the router environment later.

CartPole: observation = cart position, cart velocity, pole angle, pole angular velocity; action = push
left (0) / right (1); reward +1 per tick; 500 max.

## 2. The two networks

Both take the observation vector as input. Both are MLPs (multilayer perceptrons: input -> 64 -> 64 -> output,
each layer a matrix multiply plus a nonlinearity), built by SB3 in PyTorch.
- Actor (`pi`): output layer has one unit per action; softmax turns the scores into probabilities summing to 1.
  Sample from them while training (exploration); take the argmax to deploy. This is the policy, the thing you keep.
- Critic (`vf`): output layer is one unit; its number is a prediction of the return from this state. Trained by
  regression toward observed returns. Used only during training, to make the actor's signal less noisy.

## 3. Return, advantage, and the PPO update

- Return of a step = the discounted sum of the rewards that followed it. Followed, because only later rewards
  can be consequences; sum, because consequences arrive late; discounted (gamma = 0.99, reward k steps later
  counts gamma^k), so near consequences dominate and the sum stays finite. Horizon about 1/(1-gamma) = 100 ticks.
- Advantage of a step = its return minus the critic's prediction for that state: "better or worse than usual
  from here?" The subtraction isolates the action's own contribution. Without it every action in an easy
  episode looks good.
- One `learn()` round: collect `n_steps` (2048) ticks with the current actor -> compute returns and advantages
  -> `n_epochs` (10) passes over the batch in minibatches of `batch_size` (64), each an Adam step on the loss ->
  repeat. 30,000 timesteps is about 15 rounds.
- Loss = clipped actor loss + critic squared error + `ent_coef` * entropy. The clip: for each recorded action,
  ratio = P_new / P_old; once the ratio leaves [1-eps, 1+eps] (eps = 0.2) that sample stops contributing. So one
  batch cannot move the policy far from the one that generated the data, which keeps the data valid for all
  10 epochs. The entropy term rewards keeping probabilities spread out; 0 by default, 0.01 later as a guard
  against locking onto one action early.
- Evaluate with sampling (`deterministic=False`) to see how an untrained policy really behaves (~20); with
  argmax (`deterministic=True`) to see how the trained one deploys (~500).
- Every knob in `PPO(...)` is SB3's default: learning_rate 3e-4, n_steps 2048, batch_size 64, n_epochs 10,
  gamma 0.99, gae_lambda 0.95, clip_range 0.2, ent_coef 0.0, net_arch 64x64.

## 4. Where the LLMs are

Nowhere in the policy. PPO here trains two tiny MLPs. In the real Hecate router the LLMs live inside the
environment's `step()`: the coding agent (Codex) running on M1 or M2 does ~5 tool calls, and a frozen
"Execution Advisor" LLM scores the trajectory. Their output is turned into 13 numbers; the policy choosing
M1/M2 stays a 2x64 MLP. (The lab's GRPO thread is the one where the trained policy *is* an LLM.)

## 5. The router's contract

**Observation — 13 floats, in this order**

| # | group | name | meaning | range |
|---|---|---|---|---|
| 0 | advisor (frozen LLM, JSON) | difficulty | how hard the remaining work looks | 0..1 |
| 1 | advisor | progress | how far along the task seems | 0..1 |
| 2 | advisor | stuckness | trapped in an unproductive loop? | 0..1 |
| 3 | advisor | confidence | how sure the advisor is of its scores | 0..1 |
| 4 | telemetry (counted, no LLM) | steps_used | tool interactions used / budget | 0..1 |
| 5 | telemetry | failed_commands | share of recent commands that failed | 0..1 |
| 6 | telemetry | error_repetition | same error message recurring | 0..1 |
| 7 | telemetry | test_delta | change in test results since last checkpoint | −1..1 |
| 8 | telemetry | no_progress_steps | intervals since anything improved / budget | 0..1 |
| 9 | cost ledger | cost_fraction | spend so far / C_ref | may exceed 1 |
| 10 | cost ledger | context_utilization | how full the model's context window is | 0..1 |
| 11 | session | current_model | 0 = M1, 1 = M2 | {0, 1} |
| 12 | mask | test_observed | 1 if tests ran; 0 = test_delta is unknown, not zero | {0, 1} |

**Action** — `Discrete(2)`: 0 = run the next interval (~5 tool calls) on M1 (cheap); 1 = on M2 (strong).
V1 is one-way: once on M2, stay on M2.

**Reward** — per task: `1[resolved] − λ · C_total / C_ref`, paid as you go:
- each interval: `−λ · (that interval's cost) / C_ref`
- the interval the task resolves: `+1`
- `C_ref` = budget × M1 price = 12 × 1.0 = 12, fixed, so hard tasks are not graded on a curve
- λ is the exchange rate between cost and success: no cost term → always-M2; no success term → always-M1
- λ is a design decision, not a constant from the paper. On the real router it has to be chosen by the same sweep as A2, with real costs.

| with λ = 0.5 | score |
|---|---|
| one M1 interval | −0.042 |
| one M2 interval | −0.167 |
| easy task, M1, solved in 4 intervals | +0.83 |
| any task, M2, solved in 3 intervals | +0.50 |
| hard task, M1 for all 12 intervals, unsolved | −0.50 |

## Sources

- Schulman, Wolski, Dhariwal, Radford, Klimov. Proximal Policy Optimization Algorithms. arXiv:1707.06347, 2017.
- Schulman, Moritz, Levine, Jordan, Abbeel. High-Dimensional Continuous Control Using Generalized Advantage
  Estimation. arXiv:1506.02438, 2016.
- Raffin et al. Stable-Baselines3: Reliable Reinforcement Learning Implementations. JMLR 22(268), 2021; and the
  SB3 PPO documentation for parameter defaults.
- Towers et al. Gymnasium: A Standard Interface for Reinforcement Learning Environments. arXiv:2407.17032, 2024.
- Nath, RL notes 000-004 (lab-internal). Hecate Alpha: Architecture and Engineering Plan v0.2 (lab-internal).