"""Modal app for the Lecture 6 I-JEPA study.

Usage (profile rajatworkspace):
  modal run modal_app/app.py::smoke                      # tiny end-to-end check
  modal run modal_app/app.py::prep --dataset stl10       # stage data on volume
  modal run modal_app/app.py::launch_e1                  # 5 STL-10 runs, parallel
  modal run modal_app/app.py::probe_runs                 # probe all checkpoints
  modal run modal_app/app.py::launch_e2                  # official-ckpt probes
  modal run modal_app/app.py::train_one --arch ijepa --dataset in100 --epochs 150
"""
import modal

image = (
    modal.Image.debian_slim(python_version="3.12")
    .pip_install(
        "torch==2.8.0",
        "torchvision==0.23.0",
        "transformers==4.56.1",
        "datasets==3.6.0",
        "numpy",
        "pillow",
        "scipy",
        "scikit-learn",
    )
    .add_local_dir("ijepa", remote_path="/root/ijepa")
    .add_local_dir("toyworld", remote_path="/root/toyworld")
)

data_vol = modal.Volume.from_name("ijepa-l6-data", create_if_missing=True)
results_vol = modal.Volume.from_name("ijepa-l6-results", create_if_missing=True)
VOLS = {"/data": data_vol, "/results": results_vol}

app = modal.App("ijepa-lecture6", image=image)


@app.function(volumes=VOLS, timeout=4 * 3600, cpu=8, memory=32768)
def prep(dataset: str = "stl10"):
    import torchvision
    if dataset == "stl10":
        torchvision.datasets.STL10("/data", split="unlabeled", download=True)
        torchvision.datasets.STL10("/data", split="train", download=True)
        torchvision.datasets.STL10("/data", split="test", download=True)
    elif dataset == "in100":
        from datasets import load_dataset
        load_dataset("clane9/imagenet-100", cache_dir="/data/hf")
    data_vol.commit()
    return "ok"


@app.function(gpu="H100", volumes=VOLS, timeout=20 * 3600, cpu=16,
              memory=65536)
def train_remote(arch: str, dataset: str = "stl10", epochs: int = 100,
                 batch_size: int = 256, base_lr: float = 5e-4, seed: int = 0,
                 max_steps: int = 0, run_name: str = ""):
    from ijepa.train import train_run
    out = train_run(arch, dataset=dataset, epochs=epochs,
                    batch_size=batch_size, base_lr=base_lr, seed=seed,
                    max_steps=max_steps or None, run_name=run_name or None)
    results_vol.commit()
    return out


@app.function(gpu="H100", volumes=VOLS, timeout=6 * 3600, cpu=16,
              memory=65536)
def probe_remote(run_name: str, dataset: str = "stl10",
                 random_init: bool = False, ckpt: str = "ckpt_final.pt"):
    from ijepa.probe import evaluate_checkpoint
    res = evaluate_checkpoint(f"/results/{run_name}/{ckpt}", dataset=dataset,
                              random_init=random_init,
                              tag=(run_name + ("_randominit" if random_init
                                               else "")))
    results_vol.commit()
    return {run_name: res}


@app.function(gpu="H100", volumes=VOLS, timeout=8 * 3600, cpu=16,
              memory=65536)
def probe_pretrained_remote(model_key: str):
    from ijepa.pretrained import probe_pretrained
    res = probe_pretrained(model_key)
    results_vol.commit()
    return {model_key: res}


@app.function(gpu="A10G", volumes=VOLS, timeout=2 * 3600, cpu=8,
              memory=32768)
def energy_remote(n_images: int = 256):
    from ijepa.energy import run_energy_study
    res = run_energy_study(n_images=n_images)
    results_vol.commit()
    return res


@app.function(gpu="A10G", volumes=VOLS, timeout=2 * 3600, cpu=8,
              memory=32768)
def toy_remote():
    from toyworld.run import main
    res = main()
    results_vol.commit()
    return res


@app.function(volumes=VOLS, timeout=3600, cpu=8, memory=32768)
def viz_remote(mae_run: str = ""):
    from ijepa.viz import dump_mask_examples, dump_mae_recon
    done = dump_mask_examples()
    if mae_run:
        dump_mae_recon(mae_run)
    results_vol.commit()
    return done


@app.function(gpu="A10G", volumes=VOLS, timeout=3600, cpu=8, memory=16384)
def smoke_remote():
    """Tiny end-to-end validation of all five objectives + probe path."""
    import torch
    from ijepa.train import train_run
    from ijepa.probe import evaluate_checkpoint
    out = {}
    for arch in ["jea", "mae", "genmask", "ijepa", "ijepa_nostop"]:
        d = train_run(arch, dataset="stl10", epochs=1, batch_size=64,
                      max_steps=8, log_every=2, num_workers=4,
                      run_name=f"smoke_{arch}")
        out[arch] = "train_ok"
    res = evaluate_checkpoint("/results/smoke_ijepa/ckpt_final.pt",
                              dataset="stl10", fracs=(0.1,),
                              tag="smoke_probe")
    out["probe"] = res
    out["cuda"] = torch.cuda.get_device_name(0)
    results_vol.commit()
    return out


@app.local_entrypoint()
def smoke():
    print(smoke_remote.remote())


@app.local_entrypoint()
def launch_e1(epochs: int = 100):
    """Five STL-10 runs in parallel."""
    archs = ["jea", "mae", "genmask", "ijepa", "ijepa_nostop"]
    calls = [train_remote.spawn(a, "stl10", epochs) for a in archs]
    for a, c in zip(archs, calls):
        print(a, c.get())


@app.local_entrypoint()
def probe_runs(dataset: str = "stl10"):
    archs = ["jea", "mae", "genmask", "ijepa", "ijepa_nostop"]
    runs = [f"{dataset}_{a}_s0" for a in archs]
    calls = [probe_remote.spawn(r, dataset) for r in runs]
    calls.append(probe_remote.spawn(f"{dataset}_ijepa_s0", dataset, True))
    for c in calls:
        print(c.get())


@app.local_entrypoint()
def launch_e2():
    keys = ["ijepa_vith14", "mae_vith", "random_vith", "supervised_vitb"]
    calls = [probe_pretrained_remote.spawn(k) for k in keys]
    for k, c in zip(keys, calls):
        print(k, c.get())


@app.local_entrypoint()
def train_one(arch: str = "ijepa", dataset: str = "in100", epochs: int = 150,
              batch_size: int = 256):
    print(train_remote.remote(arch, dataset, epochs, batch_size))


@app.local_entrypoint()
def launch_e3(epochs: int = 150):
    """Spawn-based (survives local client death, unlike .remote())."""
    c = train_remote.spawn("ijepa", "in100", epochs)
    print("spawned", c.object_id)


@app.function(gpu="A10G", volumes=VOLS, timeout=3600, cpu=8, memory=32768)
def camera_remote(seed: int = 0):
    from toyworld.camera_run import main
    res = main(seed)
    results_vol.commit()
    return res


@app.local_entrypoint()
def launch_camera():
    for s in (0, 1, 2):
        c = camera_remote.spawn(s)
        print("spawned seed", s, c.object_id)


@app.local_entrypoint()
def launch_toy():
    """Spawn-based: survives local client death."""
    c = toy_remote.spawn()
    print("spawned", c.object_id)


@app.function(gpu="A10G", volumes=VOLS, timeout=3600, cpu=8, memory=32768)
def review_dump_remote(val_indices: list):
    from ijepa.review_dump import run
    res = run(val_indices)
    results_vol.commit()
    return res


@app.local_entrypoint()
def launch_review_dump():
    import json
    spec = json.load(open("../results/neighbors_spec.json"))
    c = review_dump_remote.spawn(spec["val_indices_needed"])
    print("spawned", c.object_id)
