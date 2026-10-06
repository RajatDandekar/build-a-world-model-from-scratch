#!/usr/bin/env python3
"""Convert a LeRobot SO-101 dataset into the HDF5 layout LeWorldModel trains on.

LeWorldModel (github.com/lucas-maes/le-wm) loads data through stable-worldmodel, whose HDF5 reader
expects ONE file with flat, episode-concatenated columns plus two index arrays:

    pixels       (N, 224, 224, 3)  uint8    one camera, resized for ViT-Tiny
    action       (N, A)            float32  SO-101: 6 = 5 joints + gripper
    proprio      (N, P)            float32  observation.state
    state        (N, P)            float32  observation.state (LeWM normalises every loaded column)
    episode_idx  (N,)              int64
    step_idx     (N,)              int64
    ep_len       (E,)              int32
    ep_offset    (E,)              int64

This is exactly the schema of the official Push-T file (quentinll/lewm-pusht).

Usage:
    pip install lerobot h5py
    python convert_lerobot_to_h5.py --repo-id lerobot/svla_so101_pickplace \
        --out $STABLEWM_HOME/datasets/so101_pickplace_train.h5
    # optional: --camera observation.images.side  --max-episodes 10  --size 224

Several datasets can be pooled into one file (Option C, "One model, four tasks"):
    python convert_lerobot_to_h5.py --repo-id lerobot/svla_so101_pickplace your_org/so101_pour_beads ... --out so101_multitask_train.h5
"""
import argparse

import h5py
import numpy as np
import torch
import torch.nn.functional as F


def to_uint8_hwc(img: torch.Tensor, size: int) -> np.ndarray:
    """LeRobot returns float CHW in [0, 1]; LeWM wants uint8 HWC at size x size."""
    img = F.interpolate(img.unsqueeze(0).float(), size=(size, size), mode="bilinear", antialias=True)[0]
    return (img.clamp(0, 1) * 255).round().byte().permute(1, 2, 0).numpy()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--repo-id", nargs="+", required=True, help="one or more LeRobot dataset ids")
    ap.add_argument("--out", required=True, help="output .h5 path (put it under $STABLEWM_HOME/datasets/)")
    ap.add_argument("--camera", default=None, help="camera key; defaults to the dataset's first camera")
    ap.add_argument("--size", type=int, default=224)
    ap.add_argument("--max-episodes", type=int, default=None)
    ap.add_argument("--workers", type=int, default=8)
    args = ap.parse_args()

    from lerobot.datasets.lerobot_dataset import LeRobotDataset

    with h5py.File(args.out, "w") as f:
        cols, ep_lens, n_written, ep_global = {}, [], 0, 0

        def append(name, arr):
            if name not in cols:
                cols[name] = f.create_dataset(name, shape=(0, *arr.shape[1:]), maxshape=(None, *arr.shape[1:]),
                                              dtype=arr.dtype, chunks=(min(256, len(arr)), *arr.shape[1:]))
            d = cols[name]
            d.resize(d.shape[0] + len(arr), axis=0)
            d[-len(arr):] = arr

        for repo_id in args.repo_id:
            ds = LeRobotDataset(repo_id)
            cam = args.camera or ds.meta.camera_keys[0]
            n_eps = ds.num_episodes if args.max_episodes is None else min(args.max_episodes, ds.num_episodes)
            print(f"{repo_id}: {ds.num_episodes} episodes, {ds.num_frames} frames, camera={cam}, "
                  f"action dim={ds.meta.features['action']['shape']}")
            ep_index = np.asarray(ds.hf_dataset["episode_index"])
            keep = np.where(ep_index < n_eps)[0]
            loader = torch.utils.data.DataLoader(torch.utils.data.Subset(ds, keep.tolist()), batch_size=64,
                                                 num_workers=args.workers, shuffle=False)
            cur_ep, cur = None, {"pixels": [], "action": [], "proprio": []}

            def flush():
                nonlocal ep_global, n_written
                if not cur["pixels"]:
                    return
                T = len(cur["pixels"])
                state = np.stack(cur["proprio"]).astype(np.float32)
                append("pixels", np.stack(cur["pixels"]))
                append("action", np.stack(cur["action"]).astype(np.float32))
                append("proprio", state)
                append("state", state)
                append("episode_idx", np.full(T, ep_global, dtype=np.int64))
                append("step_idx", np.arange(T, dtype=np.int64))
                ep_lens.append(T)
                ep_global += 1
                n_written += T
                for v in cur.values():
                    v.clear()

            for batch in loader:
                for i in range(len(batch["episode_index"])):
                    e = int(batch["episode_index"][i])
                    if e != cur_ep:
                        flush()
                        cur_ep = e
                    cur["pixels"].append(to_uint8_hwc(batch[cam][i], args.size))
                    cur["action"].append(batch["action"][i].numpy())
                    cur["proprio"].append(batch["observation.state"][i].numpy())
            flush()
            print(f"  -> {ep_global} episodes, {n_written} frames so far")

        lens = np.asarray(ep_lens, dtype=np.int32)
        f.create_dataset("ep_len", data=lens)
        f.create_dataset("ep_offset", data=np.concatenate([[0], np.cumsum(lens)[:-1]]).astype(np.int64))
    print(f"wrote {args.out}: {len(ep_lens)} episodes, {n_written} frames")


if __name__ == "__main__":
    main()
