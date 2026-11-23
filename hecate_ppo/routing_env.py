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
              lam: float = 0.5,
              one_way: bool = True
              ):
        super().__init__()
        self.max_intervals = max_intervals
        self.cost = {M1: cost_m1, M2: cost_m2}
        self.lam = lam          # lambda: how much cost hurts relative to success (plan eq. 3)
        self.one_way = one_way  # plan 1.1: V1 forbids M2 -> M1
        # C_ref is the plan's fixed reference cost:
        # "What the cheapest possible full run would cost"
        # = 12 intervals x M1's price 1.0 = 12
        # Every cost in the reward is divided by it, so the numbers are "fractions of a cheap full run,"
        # rather than raw dollars.
        # The plan says C_ref must be fixed, not per-task, so a hard task isn't graded on a curve.
        self.c_ref = max_intervals * cost_m1
        # `Box` needs, for each of the 13 entries, the smallest and largest values it may take.
        # So we build two arrays of 13 numbers: `low` starts as all zeroes, `high` as all ones
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

    # THE DICE ROLL. The pretend coding agent works for a little while on one model.
    def _run_interval(self, model: int) -> float:
        # Simulate one interval (~5 tool calls) of the coding agent running on `model`.
        # Called by reset() once (M1) and by step() every time. Returns the progress gained.

        d = self._difficulty    # hidden true difficulty, 0 easy .. 1 hard
        if model == M1:
            p_success = 0.90 - 0.80 * d # chance this interval moves forward: 90% easy -> 10% hard
            gain = 0.35 * (1.0 - 0.60 * d) # how much progress if it does: 0.35 easy -> 0.14 hard
        else:
            p_success = 0.95 - 0.25 * d # M2 is robust to difficulty: 95% -> 70%
            gain = 0.45 * (1.0 - 0.20 * d) # and makes bigger jumps: 0.45 -> 0.36

        # roll a die between 0 and 1. If it's under the chance, success; this turn made progress.
        made_progress = self.np_random.random() < p_success

        if made_progress:
            delta = gain * self.np_random.uniform(0.7, 1.3)
            # progress made this turn, wobbled a bit so turns aren't identical
        else:
            delta = 0.0
            # nothing got done this turn

        self._progress = min(1.0, self._progress + delta)
        # add it to the progress bar; the bar stops at 100%

        self._t += 1
        # one turn has been used (telemetry, steps used)

        self._cost_total += self.cost[model]
        # pay for the turn: 1 if it was the cheap model, 4 if the strong one (cost ledger)

        if delta > 0.02:
            self._no_progress = 0
            # we moved, so the "turns since we last moved" counter goes back to zero
        else:
            self._no_progress += 1
            # we didn't move; one more turn of being stuck

        return delta
        # hand back how much got done, for later

    # START A NEW GAME. A game is one task.
    def reset(self, *, seed: int | None = None, options: dict | None = None):
        super().reset(seed=seed)
        # roll how hard THIS task is, 0 to 1
        self._difficulty = float(self.np_random.random())
        self._progress = 0.0    # nothing fixed yet, progress bar at 0
        self._t = 0             # turns/intervals used
        self._cost_total = 0.0  # money spent
        self._no_progress = 0   # intervals since last progress, not stuck yet
        self._model = M1        # plan: every task starts on the cheap model
        self._switches = 0      # how many times the player/router changed model (for evaluation/scoreboard)
        self._invalid = 0       # attempted M2 -> M1 under the one-way rule (for monitoring)
                                # how many times the player tried to go back to cheap after going storng
        delta = self._run_interval(M1)
        # plan 7.1: one M1 interval first, so there is a trajectory to judge
        # take the first turn on the cheap model automatically, so there's something to look at

        return self._observe(delta), self._info(False)
        # first observation (zeros until step 4) + analysis info
        # show the player the board (still all zeroes for now) and the scoreboard info

    # ONE TURN. The player has said 0 (cheap model) or 1 (strong model) for the next turn.
    def step(self, action: int):
        # One routing decision: run the next interval on the chosen model. Called in a loop until the episode ends.

        action = int(action)    # SB3 may pass a numpy int; make it a plain int
        cost_before = self._cost_total  # remember spend from before turn so we can charge only this interval/turn

        if self.one_way and self._model == M2 and action == M1:
            self._invalid += 1  # plan 1.1: V1 is one-way; count the illegal request. once strong can't go cheap.
            action = M2         # ...and stay on M2, ignore it.

        if action != self._model:
            self._switches += 1 # record an escalation if player changed model, note for scoreboard

        # the model for this interval. this turn runs on the model player picked
        self._model = action

        # simulate the interval on it. roll dice for this turn on that model
        delta = self._run_interval(self._model)

        # did the progress bar just hit 100%? then the task is solved
        resolved = self._progress >= 1.0

        # score for this turn: a small penalty for what this turn cost
        # plan eq. 3, paid per interval: -lambda * dC / C_ref
        reward = -self.lam * (self._cost_total - cost_before) / self.c_ref

        # +1 for solving the task
        if resolved:
            reward += 1.0

        terminated = resolved or self._t >= self.max_intervals
        # game over if solved, or if all 12 turns are used up

        return self._observe(delta), float(reward), terminated, False, self._info(resolved)
        # show the board, the score for this turn, whether the game is over,
        # (never cut off from outside), scoreboard info

    def _info(self, resolved: bool) -> dict:
        # the scoreboard: numbers for us humans to analyze. The player never sees this.
        return {"resolved": resolved, "cost_total": self._cost_total, "intervals": self._t,
                "switches": self._switches, "invalid_transitions": self._invalid,
                "difficulty_true": self._difficulty}

    def _observe(self, delta: float) -> np.ndarray:
        # the board the player sees. Blank for now; next step we draw the 13 numbers on it.
        return np.zeros(len(FEATURES), dtype=np.float32)