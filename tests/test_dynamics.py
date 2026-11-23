from hecate_ppo.routing_env import HecateRoutingEnv, M1, M2

def run(policy_action, n=300):
    # Play n games always choosing the same model. Report: share of all games solved,
    # and share of HARD games (difficulty > 0.7) solved.
    env = HecateRoutingEnv()
    resolved, hard_resolved, hard_n = 0, 0, 0
    for i in range(n):
        _, info = env.reset(seed=i)     # game i, same every run thanks to the seed
        hard = info["difficulty_true"] > 0.7    # peek at scoreboard (player may not)
        done = False
        while not done:
            _, _, term, trunc, info = env.step(policy_action)
            done = term or trunc
        resolved += info["resolved"]
        if hard:
            hard_n += 1
            hard_resolved += info["resolved"]
    return resolved / n, hard_resolved / max(hard_n, 1)

def test_m1_fails_hard_tasks_m2_does_not():
    # The central claim of the toy: cheap-only fails hard tasks; strong-only solves nearly everything.
    _, m1_hard = run(M1)
    m2_all, m2_hard = run(M2)
    assert m1_hard < 0.3
    assert m2_hard > 0.9 and m2_all > 0.95

def test_same_seed_same_task():
    # Seeding works: two envs reset with seed 7 roll same difficulty.
    a, b = HecateRoutingEnv(), HecateRoutingEnv()
    _, ia = a.reset(seed=7)
    _, ib = b.reset(seed=7)
    assert ia["difficulty_true"] == ib["difficulty_true"]

def test_one_way_escalation():
    # Going strong then asking for cheap is overruled and counted.
    env = HecateRoutingEnv(one_way=True)
    env.reset(seed=0)
    env.step(M2)
    _, _, _, _, info = env.step(M1)
    assert info["invalid_transitions"] == 1

def test_reward_bounds():
    # A whole game's score stays within [-1, 1] (with lambda = 0.5 it cannot leave that range).
    env = HecateRoutingEnv()
    env.reset(seed=0)
    done, total = False, 0.0
    while not done:
        _, r, term, trunc, _ = env.step(M2)
        total += r
        done = term or trunc
    assert -1.0 <= total <= 1.0