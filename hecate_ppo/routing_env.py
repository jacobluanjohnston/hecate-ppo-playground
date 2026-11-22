from __future__ import annotations
import gymnasium as gym
import numpy as np
from gymnasium import spaces

# Named constants so code never says bare `0` or `1` for model
M1, M2 = 0, 1

# Plan's §3 table as data.
FEATURES = [
    # Execution Advisor: a frozen LLM reads issue + trajectory so far,
    #   returns these four as JSON (plan 6.1)
    ("advisor",   "difficulty"),    # how hard the remaining work looks, 0 easy .. 1 hard
    ("advisor",   "progress"),      # how far along the task seems, 0 .. 1
    ("advisor",   "stuckness"),     # does the agent look trapped in an unproductive loop, 0 .. 1
    ("advisor",   "confidence"),    # how sure the advisor is of its own three scores, 0 .. 1
    # Telemetry Collector: plain code counting Codex tool events, no LLM (plan 4.2)
    ("telemetry", "steps_used"),        # tool interactions used / budget, 0 .. 1
    ("telemetry", "failed_commands"),   # share of recent commands that failed, 0 .. 1
    ("telemetry", "error_repetition"),  # how often the same error message keeps coming back, 0 .. 1
    ("telemetry", "test_delta"),        # change in test results since last checkpoint, -1 worse .. +1 better
    ("telemetry", "no_progress_steps"), # intervals since anything measurably improved / budget, 0 .. 1
    # Cost ledger: the proxy's record of spend (plan 4.4)
    ("cost",      "cost_fraction"),         # spend so far / C_ref; can exceed 1 when M2 overspends the M1 reference
    ("cost",      "context_utilization"),   # how full the current model's context window is, 0 .. 1
    # Session controller
    ("session",   "current_model"),     # 0 = M1 (cheap), 1 = M2 (strong)
    # Validity mask (plan 3.1): 'tests did not run' is not the same as 'tests unchanged'
    ("mask",      "test_observed"),     # 1 if tests ran this interval, 0 if test_delta is unknown
]
FEATURE_NAMES = [n for _, n in FEATURES]
IDX = {n: i for i, n in enumerate(FEATURE_NAMES)}

# Subclassing gym.Env is what makes it a Gynasium env
class HecateRoutingEnv(gym.Env):
    metadata = {"render_modes": []}

    # Constructor store task's economics:
    #   - budget of 12 intervals,
    #   - M1 costs 1,
    #   - M2 costs 4,
    #   - C_ref= 12,
    #   - λ = 0.5;
    #   and declares the two spaces.
    def __init__(self, max_intervals: int = 12,
              cost_m1: float = 1.0,
              cost_m2: float = 4.0,
              lam: float = 0.5
              ):
        super().__init__()
        self.max_intervals = max_intervals
        self.cost = {M1: cost_m1, M2: cost_m2}
        # C_ref is the plan's fixed reference cost:
        # "What the cheapest possible full run would cost"
        # = 12 intervals x M1's price 1.0 = 12
        # Every cost in the reward is divided by it, so the numbers are "fractions of a cheap full run,"
        # rather than raw dollars.
        # The plan says C_ref must be fixed, not per-task, so a hard task isn't graded on a curve.
        self.c_ref = max_intervals * cost_m1
        # `Box` needs, for each of the 13 entries, the smallest and largest values it may take.
        # So we build two arrays of 13 numbers: `low` starts as all zeroes, `high` as al ones
        low = np.zeros(len(FEATURES), dtype=np.float32)
        high = np.ones(len(FEATURES), dtype=np.float32)
        # Two exceptions: test_delta can be negative (tests got worse).
        low[IDX["test_delta"]] = -1.0
        # cost_fraction can exceed 1: the worst case is M2 for the whole budget, 12 x 4.0 = 48,
        # divided by C_ref 12 = 4.0; the + 1.0 is headroom so a rounding error never lands
        # exactly on the boundary; so position 9's ceiling is 5.0. Those arrays become
        # Box(low, high). check_env wll later verify for every observation you return stays
        # inside these bounds.
        high[IDX["cost_fraction"]] = max_intervals * cost_m2 / self.c_ref + 1.0
        self.observation_space = spaces.Box(low=low, high=high, dtype=np.float32)
        self.action_space = spaces.Discrete(2)

    # Signature for Gymnasium. * means seed and options must be passed by name
    def reset(self, *, seed: int | None = None, options: dict | None = None):
        # Hands seed to the parent, which creates self.np_random
        # Same seed -> identical random draws -> identical hidden task in Step 3
        super().reset(seed=seed)
        self._t = 0 # Interval counter
        # The required pair (observation, info). Observation is 13 zeroes as 32 bit floats
        # the dtype must match the Box exactly or check_env fails.
        return np.zeros(len(FEATURES), dtype=np.float32), {}

    # Count one interval; the episode is over when the budget is spent; return the required
    # five-tuple: observation (13 zeroes), reward (0.0, a Python float), terminated (Python bool),
    # truncated (False, never cut off externally in the toy), info ({}).
    # The action is accepted and ignored for now, Step 3 gives it meaning.
    def step(self, action: int):
        self._t += 1
        terminated = self._t >= self.max_intervals
        return np.zeros(len(FEATURES), dtype=np.float32), 0.0, terminated, False, {}
