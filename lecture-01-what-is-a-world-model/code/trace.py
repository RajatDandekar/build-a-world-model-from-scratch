"""Generate the concrete traces used on the slides — so every number on screen is real."""
import numpy as np
from world_model_lecture1 import BouncingBall, ReplayBuffer, collect

# ---- 1. one timestep, in full detail -------------------------------------
env = BouncingBall(seed=3)
env.x, env.y, env.vx, env.vy = 7.00, 5.00, 0.80, 0.60   # pin a clean starting state
print("=== ONE TIMESTEP ===")
print("state  s_t :", env.state.round(2))
a = 2
print("action a_t :", a, "(nudge right: vx += 0.15)")
o, r, d = env.step(a)
print("state  s_t+1:", env.state.round(2))
print("reward r_t :", round(r, 3))
print("obs    o_t+1: 16x16 image, nonzero pixels =", int((o > 0).sum()))

# ---- 2. five steps of the true state --------------------------------------
print("\n=== FIVE STEPS (true state) ===")
env2 = BouncingBall(seed=3)
env2.x, env2.y, env2.vx, env2.vy = 12.60, 5.00, 0.80, 0.60
print(f"{'t':>2} {'x':>6} {'y':>6} {'vx':>6} {'vy':>6}   action")
print(f"{0:>2} {env2.x:>6.2f} {env2.y:>6.2f} {env2.vx:>6.2f} {env2.vy:>6.2f}   -")
for t in range(1, 6):
    act = 1
    env2.step(act)
    print(f"{t:>2} {env2.x:>6.2f} {env2.y:>6.2f} {env2.vx:>6.2f} {env2.vy:>6.2f}   {act} (nothing)")

# ---- 3. two states, one observation ---------------------------------------
print("\n=== TWO STATES, ONE OBSERVATION ===")
A = BouncingBall(seed=1); A.x, A.y, A.vx, A.vy = 8.0, 8.0,  0.7,  0.5
B = BouncingBall(seed=1); B.x, B.y, B.vx, B.vy = 8.0, 8.0, -0.7, -0.5
oa, ob = A.observe(), B.observe()
print("state A :", A.state.round(2))
print("state B :", B.state.round(2))
print("observations identical?", np.array_equal(oa, ob), " pixel distance =", float(np.abs(oa-ob).sum()))
for _ in range(4):
    A.step(1); B.step(1)
print("after 4 steps  A:", A.state.round(2), " B:", B.state.round(2))
print("now observations differ by", round(float(np.abs(A.observe()-B.observe()).sum()), 2), "pixels of mass")

# ---- 4. the twin-frame experiment, reported cleanly ------------------------
print("\n=== TWIN-FRAME EXPERIMENT ===")
env3 = BouncingBall(seed=0)
buf = ReplayBuffer(capacity=10_000, obs_shape=(16, 16), seed=0)
collect(env3, buf, steps=10_000)
flat = buf.obs[:len(buf)].reshape(len(buf), -1)
d0 = np.linalg.norm(flat - flat[0], axis=1); d0[0] = np.inf
j = int(d0.argmin())
print(f"frame 0 and frame {j}: pixel distance = {d0[j]:.3f}")
for k in (1, 3, 5, 10):
    print(f"  after {k:>2} steps -> distance {np.linalg.norm(flat[0+k]-flat[j+k]):.3f}")
