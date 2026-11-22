# hecate-ppo-playground

A toy [Gymnasium](https://gymnasium.farama.org/) environment with the same
contract as the Hecate router — a 13-number observation, a two-way action (cheap model / strong model),
and a reward of `+1 if the task resolves − λ · cost` — trained with
[Stable-Baselines3](https://stable-baselines3.readthedocs.io/) PPO. Everything here is CPU-only, runs in
minutes, and calls no LLM: the "task" is a hidden difficulty number and a progress bar. The point is to
understand the shape of the routing problem before the real environment exists.

## Run

    uv venv --python 3.12 .venv && source .venv/bin/activate
    uv pip install -r requirements.txt
    python scripts/01_cartpole_ppo.py     # PPO black box on CartPole: ~20 -> 500
    python -m pytest -q                   # the environment contract tests

## Build log (one commit per step)

| step | what | status |
|---|---|---|
| 0 | repo, venv, dependencies | done |
| 1 | PPO as a black box on CartPole; every knob annotated | done — 18.7 → 500.0 |
| 2 | empty `HecateRoutingEnv` passing SB3's `check_env`: 13-float `Box` observation, `Discrete(2)` action | done — 3 tests |
| 3 | hidden task dynamics and the incremental cost-aware reward | next |
| 4 | observation features: telemetry, advisor estimates, validity mask | |
| 5 | baselines (M1-only, M2-only, fixed rule, advisor rule) and seeded evaluation | |
| 6 | PPO training, telemetry-only ablation, behaviour readout | |
| 7 | results, caveats, toy-to-real mapping | |

## What I learned (updated each step)

- Step 1: an environment is `reset()` + `step()`; PPO trains two small MLPs (actor → action probabilities,
  critic → expected return); the return is the discounted sum of rewards that followed; the advantage is
  return minus the critic's prediction; the clip bounds how far one batch can move the policy, which is why
  PPO can reuse a batch for 10 epochs. Every knob in `PPO(...)` was SB3's default.
- Step 2: `Box(low, high)` and `Discrete(n)` are the contract SB3 reads to size the networks; `check_env`
  enforces it. Three different errors with one cause (`__init` vs `__init__`) — read what failures share.

## Notes

`docs/` holds my study notes. Sources are listed there.