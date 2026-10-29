"""Per-frame HEAL detections, cooperative and ego-only, for one zoo checkpoint and one dataset split.

Same model loading, dataset building and post-processing as qcar/compare.py (run_method): the ego-only pass
drops the peer by setting comm_range to a tiny value. Writes one JSON line per frame:

  {"split", "scenario", "index" (frame index inside the scenario), "lidar_pose" (agent 1 yaml, x10 model units),
   "coop": {"boxes": [[8 x [x, y, z]]...], "scores": [...]}, "ego_only": {...}, "gt": [[8 x [x, y, z]]...]}

Boxes are the post-processed corner boxes in the ego frame, model units (x10 horizontal), as returned by
ds.post_process. GT is the cooperative-pass ground truth of the same frame (as compare.py scores it).

Run from the HEAL repo root with its interpreter:
    python <this file> --method_dir checkpoints/qcar/zoo/pyramid --split train --out <file.jsonl>
         [--scenario_filter <prefix>]
"""
import argparse
import glob
import json
import os
import sys

import torch
import yaml
from torch.utils.data import DataLoader

sys.path.insert(0, os.getcwd())
from opencood.data_utils.datasets import build_dataset  # noqa: E402
from opencood.tools import train_utils  # noqa: E402
from qcar import config, registry  # noqa: E402
from opencood.data_utils.datasets.basedataset.opv2v_basedataset import OPV2VBaseDataset  # noqa: E402

# Windows path handling, as in the user's Cooperative Perception shim (code/src/heal_windows_paths.py):
# timestamps from '/'-normalised file names, and scenario roots ending in '/' for HEAL's scenario-name parser.
OPV2VBaseDataset.extract_timestamps = staticmethod(
    lambda yaml_files: [str(p).replace("\\", "/").rsplit("/", 1)[-1].replace(".yaml", "") for p in yaml_files])


def build_split(hypes, split, comm_range=None):
    h = json.loads(json.dumps(hypes))
    key = {"train": "root_dir", "validate": "validate_dir"}[split]
    h["validate_dir"] = h[key].replace("\\", "/").rstrip("/") + "/"
    if comm_range is not None:
        h["comm_range"] = comm_range
    return build_dataset(h, visualize=False, train=False)


def tolist(t):
    return [] if t is None else t.detach().float().cpu().numpy().round(4).tolist()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--method_dir", required=True)
    ap.add_argument("--split", choices=["train", "validate"], required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--ego_only_comm_range", type=float, default=0.001)
    ap.add_argument("--checkpoint", default=None, help="default: the single net_epoch_bestval_at*.pth in method_dir")
    ap.add_argument("--scenario_filter", default="", help="only write scenarios whose name contains this text")
    a = ap.parse_args()

    hypes = json.load(open(os.path.join(a.method_dir, "resolved_hypes.json")))
    config.resolve_modules(argparse.Namespace(plugins=None, model_packages=None, loss_packages=None), hypes,
                           config.load_conf(config.DEFAULT_CONF))
    model = registry.create_model(hypes)
    # Same loading as train_utils.load_saved_model (torch.load + strict=False), without its '/'-based
    # parsing of the file name, which fails on Windows paths.
    ckpt = a.checkpoint or glob.glob(os.path.join(a.method_dir, "net_epoch_bestval_at*.pth"))
    if isinstance(ckpt, list):
        if len(ckpt) != 1:
            sys.exit("expected one net_epoch_bestval_at*.pth in %s, found %d" % (a.method_dir, len(ckpt)))
        ckpt = ckpt[0]
    state = torch.load(ckpt, map_location="cpu")
    own = model.state_dict()
    missing = [k for k in own if k not in state]
    unexpected = [k for k in state if k not in own]
    model.load_state_dict(state, strict=False)
    print("checkpoint", ckpt, "tensors", len(state), "missing", len(missing), "unexpected", len(unexpected))
    if missing:
        print("first missing keys:", missing[:5])
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device).eval()

    ds = {"coop": build_split(hypes, a.split), "ego_only": build_split(hypes, a.split, a.ego_only_comm_range)}
    assert len(ds["coop"]) == len(ds["ego_only"])
    ends = list(ds["coop"].len_record)
    folders = getattr(ds["coop"], "scenario_folders", None)
    if folders is None:  # HEAL lists scenario folders sorted by name
        root = hypes["validate_dir"] if a.split == "validate" else hypes["root_dir"]
        root = root if os.path.isabs(root) else os.path.join(os.getcwd(), root)
        folders = sorted(os.path.join(root, x) for x in os.listdir(root) if os.path.isdir(os.path.join(root, x)))
    folders = [str(p) for p in folders]
    yamls = {f: sorted(glob.glob(os.path.join(f, "1", "[0-9]" * 6 + ".yaml"))) for f in folders}
    counts = [e - s for s, e in zip([0] + ends[:-1], ends)]
    if len(folders) != len(ends) or any(len(yamls[f]) != c for f, c in zip(folders, counts)):
        sys.exit("scenario order or frame counts differ from the HEAL dataset; refusing to label frames")

    loaders = {m: iter(DataLoader(d, batch_size=1, shuffle=False, collate_fn=d.collate_batch_test)) for m, d in ds.items()}
    n, scen, start = 0, 0, 0
    with open(a.out, "w") as out, torch.inference_mode():
        for i in range(len(ds["coop"])):
            while i >= ends[scen]:
                start = ends[scen]
                scen += 1
            folder = folders[scen]
            rec = {"split": a.split, "scenario": os.path.basename(folder), "index": i - start}
            res = {}
            for m in ("coop", "ego_only"):
                batch = next(loaders[m])
                if batch is None:
                    res[m] = None
                    continue
                batch = train_utils.to_device(batch, device)
                with torch.cuda.amp.autocast(enabled=False):
                    o = model(batch["ego"])
                res[m] = ds[m].post_process(batch, {"ego": o})
            if a.scenario_filter and a.scenario_filter not in rec["scenario"]:
                continue
            y = yamls[folder][rec["index"]] if rec["index"] < len(yamls[folder]) else None
            rec["yaml"] = os.path.basename(y) if y else None
            rec["lidar_pose"] = yaml.safe_load(open(y))["lidar_pose"] if y else None
            for m in ("coop", "ego_only"):
                rec[m] = None if res[m] is None else {"boxes": tolist(res[m][0]), "scores": tolist(res[m][1])}
            rec["gt"] = None if res["coop"] is None else tolist(res["coop"][2])
            out.write(json.dumps(rec) + "\n")
            n += 1
    print("wrote", n, "frames to", a.out)


if __name__ == "__main__":
    main()
