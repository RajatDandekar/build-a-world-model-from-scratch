"""
Lecture 1 — What is a world model, really?
The data substrate every later lecture builds on:
  1. an instrumented environment that exposes BOTH the true state and the observation
  2. a replay buffer that stores transitions and samples SEQUENCES

Pure Python + numpy. Runs in a second on a laptop.
    python3 world_model_lecture1.py
"""
import numpy as np

# ----------------------------------------------------------------------------
# 1. The environment: a ball bouncing in a box.
#    True state  = (x, y, vx, vy)   — 4 numbers, fully describes the future
#    Observation = a 16x16 image    — 256 numbers, and NO velocity in a single frame
# ----------------------------------------------------------------------------
class BouncingBall:
    """A tiny, fully-instrumented environment.

    We deliberately expose the true state as well as the observation, so we can
    *prove* to ourselves what the observation is missing. Real environments
    never give you this — which is exactly the problem a world model solves.
    """

    SIZE = 16          # observation is SIZE x SIZE grayscale
    ACTIONS = 3        # 0 = nudge left, 1 = do nothing, 2 = nudge right

    def __init__(self, seed=0):
        self.rng = np.random.default_rng(seed)
        self.reset()

    def reset(self):
        # true state: position and velocity, in continuous coordinates
        self.x = self.rng.uniform(4, 12)
        self.y = self.rng.uniform(4, 12)
        self.vx = self.rng.choice([-1.0, 1.0]) * self.rng.uniform(0.4, 0.9)
        self.vy = self.rng.choice([-1.0, 1.0]) * self.rng.uniform(0.4, 0.9)
        self.t = 0
        return self.observe()

    @property
    def state(self):
        """The TRUE state — 4 numbers. Markovian: enough to predict the future."""
        return np.array([self.x, self.y, self.vx, self.vy], dtype=np.float32)

    def observe(self):
        """What the agent actually gets — pixels. 256 numbers, velocity invisible."""
        img = np.zeros((self.SIZE, self.SIZE), dtype=np.float32)
        cx, cy = int(round(self.x)), int(round(self.y))
        for dx in (-1, 0, 1):
            for dy in (-1, 0, 1):
                px, py = cx + dx, cy + dy
                if 0 <= px < self.SIZE and 0 <= py < self.SIZE:
                    img[py, px] = 1.0 if (dx == 0 or dy == 0) else 0.6
        return img

    def step(self, action):
        """One tick of the world: apply action, advance physics, return o, r, done."""
        # the action nudges horizontal velocity
        self.vx += (-0.15, 0.0, 0.15)[action]
        self.vx = float(np.clip(self.vx, -1.2, 1.2))

        self.x += self.vx
        self.y += self.vy

        # bounce off the walls (this is the "physics" the world model must learn)
        if self.x < 1:  self.x, self.vx = 1, -self.vx
        if self.x > 14: self.x, self.vx = 14, -self.vx
        if self.y < 1:  self.y, self.vy = 1, -self.vy
        if self.y > 14: self.y, self.vy = 14, -self.vy

        # reward: keep the ball near the centre column
        reward = 1.0 - abs(self.x - 7.5) / 7.5
        self.t += 1
        done = self.t >= 200
        return self.observe(), float(reward), done


# ----------------------------------------------------------------------------
# 2. The replay buffer: the data substrate for every later lecture.
#    A world model is trained on SEQUENCES, not single steps — it has to learn
#    how the world unrolls over time, so we sample contiguous chunks.
# ----------------------------------------------------------------------------
class ReplayBuffer:
    def __init__(self, capacity, obs_shape, seed=0):
        self.capacity = capacity
        self.obs = np.zeros((capacity, *obs_shape), dtype=np.float32)
        self.act = np.zeros((capacity,), dtype=np.int64)
        self.rew = np.zeros((capacity,), dtype=np.float32)
        self.done = np.zeros((capacity,), dtype=bool)
        self.idx = 0
        self.full = False
        self.rng = np.random.default_rng(seed)

    def add(self, obs, action, reward, done):
        """Store one transition: (o_t, a_t, r_t, done_t)."""
        self.obs[self.idx] = obs
        self.act[self.idx] = action
        self.rew[self.idx] = reward
        self.done[self.idx] = done
        self.idx = (self.idx + 1) % self.capacity
        self.full = self.full or self.idx == 0

    def __len__(self):
        return self.capacity if self.full else self.idx

    def sample_sequences(self, batch_size, seq_len):
        """Sample B contiguous windows of length L that don't straddle an episode end."""
        n = len(self)
        assert n > seq_len + 1, "not enough data yet"
        starts = []
        while len(starts) < batch_size:
            s = int(self.rng.integers(0, n - seq_len))
            if not self.done[s:s + seq_len - 1].any():   # no episode boundary inside
                starts.append(s)
        idx = np.array([np.arange(s, s + seq_len) for s in starts])
        return {
            "obs": self.obs[idx],       # (B, L, 16, 16)
            "action": self.act[idx],    # (B, L)
            "reward": self.rew[idx],    # (B, L)
            "done": self.done[idx],     # (B, L)
        }


# ----------------------------------------------------------------------------
# 3. Collect experience with a random policy, then look at what we have.
# ----------------------------------------------------------------------------
def collect(env, buffer, steps, seed=0):
    rng = np.random.default_rng(seed)
    obs = env.reset()
    for _ in range(steps):
        action = int(rng.integers(0, env.ACTIONS))     # random policy, for now
        next_obs, reward, done = env.step(action)
        buffer.add(obs, action, reward, done)
        obs = env.reset() if done else next_obs
    return buffer


if __name__ == "__main__":
    env = BouncingBall(seed=0)
    buf = ReplayBuffer(capacity=10_000, obs_shape=(env.SIZE, env.SIZE), seed=0)
    collect(env, buf, steps=10_000)

    print(f"buffer: {len(buf)} transitions")
    print(f"true state:  {env.state.shape[0]} numbers  -> {env.state.round(2)}")
    print(f"observation: {env.observe().size} numbers  (a {env.SIZE}x{env.SIZE} image)")
    print(f"compression the world model must find: {env.observe().size / env.state.shape[0]:.0f}x")

    batch = buf.sample_sequences(batch_size=8, seq_len=16)
    print("\nsampled batch — the shape every later lecture expects:")
    for k, v in batch.items():
        print(f"  {k:7s} {str(v.shape):20s} {v.dtype}")

    # --- the point of the whole lecture, in two numbers -----------------------
    # From ONE frame, can you tell where the ball is going? Find two moments in
    # the buffer whose images are nearly identical but whose futures diverge.
    flat = buf.obs[:len(buf)].reshape(len(buf), -1)
    i = 0
    dists = np.linalg.norm(flat - flat[i], axis=1)
    dists[i] = np.inf
    j = int(dists.argmin())
    print(f"\nframes {i} and {j} differ by only {dists[j]:.3f} in pixel space,")
    print(f"but 5 steps later they differ by "
          f"{np.linalg.norm(flat[i+5] - flat[j+5]):.3f}.")
    print("=> a single observation does NOT determine the future. That gap is")
    print("   partial observability — and closing it is what a world model does.")
