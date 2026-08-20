"""MiniPong — the world of the re-recorded Lecture 3.

A 32x32 RGB paddle game:
  * a bright paddle at the bottom, moved by 3 actions (left / stay / right)
  * a soft glowing ball bouncing off the left, right, top and bottom walls,
    and bending off the paddle
  * the ball's VELOCITY is invisible in a single frame — the memory story
  * dark background: the objects carry all the signal (easy, honest training)

Two properties matter for learnability, and we chose both deliberately:
  * DETERMINISTIC physics — no randomness after reset, so a predictor can in
    principle be exact (genuine randomness needs richer heads; a later lecture)
  * CONTINUOUS motion with anti-aliased rendering — the ball's sub-pixel
    position shows up smoothly in pixel intensities, so nearby states get
    nearby codes. A world snapped to integer pixels creates isolated islands
    of codes, and a predicted code that lands between islands paints nothing.

No reward is defined: this lecture builds ONLY the world model.
"""
import numpy as np


class MiniPong:
    SIZE = 32
    ACTIONS = 3            # 0 = paddle left, 1 = stay, 2 = paddle right
    PADDLE_W = 8
    PADDLE_Y = 29          # paddle row (2 px tall: rows 29-30)
    PADDLE_SPEED = 2
    BG = np.array([0.05, 0.05, 0.08], np.float32)
    WALL = np.array([0.35, 0.35, 0.40], np.float32)
    PADDLE = np.array([0.20, 0.85, 0.80], np.float32)   # bright teal
    BALL = np.array([1.00, 0.85, 0.30], np.float32)     # bright warm yellow

    _YY, _XX = np.mgrid[0:32, 0:32].astype(np.float32)

    def __init__(self, seed=0):
        self.rng = np.random.default_rng(seed)
        self.reset()

    def reset(self):
        self.px = float(self.rng.integers(6, self.SIZE - 6 - self.PADDLE_W))  # paddle left edge
        self.bx = float(self.rng.uniform(6, self.SIZE - 6))
        self.by = float(self.rng.uniform(4, 12))
        # continuous, deterministic velocity: fixed after reset, ~1 px per step
        self.vx = float(self.rng.choice([-1, 1]) * self.rng.uniform(0.55, 0.95))
        self.vy = float(self.rng.uniform(0.55, 0.95))
        self.t = 0
        return self.observe()

    def observe(self):
        img = np.tile(self.BG, (self.SIZE, self.SIZE, 1)).astype(np.float32)
        img[0, :] = self.WALL
        img[:, 0] = self.WALL
        img[:, -1] = self.WALL
        p = int(round(self.px))
        img[self.PADDLE_Y:self.PADDLE_Y + 2, p:p + self.PADDLE_W] = self.PADDLE
        # the ball is a soft glow: its sub-pixel position shows in the intensities
        g = 1.5 * np.exp(-(((self._XX - self.bx) ** 2 + (self._YY - self.by) ** 2)
                           / (2 * 1.15 ** 2)))[..., None]
        img = np.clip(img + g * self.BALL, 0.0, 1.0).astype(np.float32)
        return img

    def step(self, action):
        # paddle
        self.px += (action - 1) * self.PADDLE_SPEED
        self.px = float(np.clip(self.px, 1, self.SIZE - 1 - self.PADDLE_W))
        # ball — continuous, deterministic motion with mirror reflections
        self.bx += self.vx
        self.by += self.vy
        if self.bx < 2.0:                        # left wall
            self.bx = 4.0 - self.bx; self.vx = abs(self.vx)
        if self.bx > self.SIZE - 3.0:            # right wall
            self.bx = 2 * (self.SIZE - 3.0) - self.bx; self.vx = -abs(self.vx)
        if self.by < 2.0:                        # top wall
            self.by = 4.0 - self.by; self.vy = abs(self.vy)
        # paddle bounce — and the key pressed at contact deterministically bends the ball
        if self.vy > 0 and self.by >= self.PADDLE_Y - 1.5:
            if self.px - 1 <= self.bx <= self.px + self.PADDLE_W:
                self.by = 2 * (self.PADDLE_Y - 1.5) - self.by
                self.vy = -abs(self.vy)
                if action == 0:
                    self.vx = -0.9
                elif action == 2:
                    self.vx = 0.9
        # bottom wall: the ball bounces (no random respawns — the world stays
        # perfectly deterministic, which is exactly what a first world model needs)
        if self.by > self.SIZE - 2.5:
            self.by = 2 * (self.SIZE - 2.5) - self.by; self.vy = -abs(self.vy)
        self.t += 1
        return self.observe(), self.t >= 120


if __name__ == "__main__":
    env = MiniPong(seed=0)
    obs = env.reset()
    rng = np.random.default_rng(0)
    n_bounce = 0
    for t in range(240):
        vy_before = env.vy
        obs, done = env.step(int(rng.integers(0, 3)))
        if vy_before > 0 and env.vy < 0:
            n_bounce += 1
    print(f"smoke test: 240 steps, {n_bounce} upward bounces, ball at ({env.bx:.1f},{env.by:.1f}), frame {obs.shape}")
