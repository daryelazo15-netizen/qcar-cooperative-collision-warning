# Cooperative collision warning on QCars

Code for a cooperative collision warning on two QCar vehicles. The ego car warns of a hidden, non-connected vehicle from its own camera detections, from a cooperating car's detections, or from both, and keeps warning through message loss. Three parts:

| folder | what it is |
|---|---|
| `perception/` | HEAL (pyramid fusion on front-camera images) running on the two QCars: the pinned fork commit, our changes to it, plugins, model configs, the detection export to world coordinates |
| `tracking/` | the trackers that turn per-frame detections into tracks with a velocity: a reference tracker (gated association, two-point velocity) and a constant-velocity Kalman tracker |
| `collision/` | the collision module: swept-footprint geometry, the risk model with four stale-track policies, and a frame-by-frame replay over detection streams with message loss |
| `live/` | the real-time loop (both cars' streams -> HEAL -> detections), the online collision wrapper that runs the collision module on them, and a viewer |

Used in our IEEE IV 2027 submission on cooperative collision warning under occlusion (citation to be added).

## Pipeline

    QCar 1 (ego)  --camera + motion-capture pose--+
                                                  +--> live/run_live.py: pair frames, HEAL (perception/)
    QCar 2 (peer) --camera + motion-capture pose--+          |
                                                             v  one JSON line per pair: boxes in the ego frame (model units) and world metres
                                       live/online_collision.py   (or perception/heal_export + collision/frames.py offline)
                                          world boxes -> remove the two cars' own detections -> tracker -> collision model -> warning row
                                                             |
                                                             v  live/viewer.py: banner and tracks on the camera views and a bird's-eye view

The detector produces boxes; the tracker (`tracking/`) associates them over time and estimates velocity; the collision model (`collision/risk.py`) extrapolates each track at constant velocity, optionally with a growing position-uncertainty envelope, and warns when the earliest possible contact is within the warning horizon. When the cooperative stream is lost for a frame, the ego-only stream is used for that frame and the stale cooperative track is handled by the chosen policy:

| method | behaviour on a frame in which a track got no update |
|---|---|
| `ego_only`, `cooperative_ideal` | not applicable: always the own stream / always the cooperative stream |
| `no_fallback` | the track is not evaluated |
| `hold` | the track is evaluated at its last state |
| `cv` | the track is propagated at constant velocity |
| `uncertainty` | propagated, with the footprint inflated by the growing position uncertainty (the conservative envelope) |
| `uncertainty_nofb` | the `uncertainty` envelope on updated tracks only |

Tracks expire when their last update is older than `risk.expire_source_age_s`. Settings are in `collision/configs/default.json`.

## Installation

There are two levels, depending on what you want to run.

**1. Tracking and collision module only** (replay over detection streams; no GPU, no HEAL, no cars):

    git clone https://github.com/daryelazo15-netizen/qcar-cooperative-collision-warning
    cd qcar-cooperative-collision-warning
    python3 -m venv .venv && . .venv/bin/activate
    pip install -r requirements.txt

Python 3.11. `tracking/` and `collision/` need only the standard library and `shapely`; the viewer adds `numpy`, `opencv-python` and `Pillow`. Run everything from the repository root.

**2. Perception and the live loop** (cooperative detection on the cars). In addition to level 1, set up the HEAL fork that provides cooperative perception:

    git clone https://github.com/Saavkd7/QCAR-HEAL_Integration
    cd QCAR-HEAL_Integration && git checkout 71606bc005eafde25c5c673a77251d2e1296c3e3
    git apply /path/to/qcar-cooperative-collision-warning/perception/diffs/extrinsic.diff
    git apply /path/to/qcar-cooperative-collision-warning/perception/diffs/vehicles.diff
    python opencood/utils/setup.py build_ext --inplace
    cp qcar_testbed_integration/machine_paths.example.json qcar_testbed_integration/machine_paths.json   # then fill in local paths

Install the fork's environment (Python 3.8, torch 1.12, spconv; see the fork's README), download the two checkpoints from the release [`checkpoints-v1`](https://github.com/daryelazo15-netizen/qcar-cooperative-collision-warning/releases/tag/checkpoints-v1) into `perception/model_dirs/coop/` and `perception/model_dirs/single/`, and set `QCAR_HEAL_ROOT` to the fork clone. The live loop also needs a ROS 1 network to the two cars. Details: `perception/README.md` and `live/README.md`.

## Replay (no cars, no detector)

An episode file holds one episode per line (format in the docstring of `collision/replay.py`: world-frame boxes of the cooperative and the ego-only stream per frame, the ego pose and velocity). Replay it with a tracker, a loss pattern and a set of methods:

    python3 -m collision.replay --episodes episodes.jsonl --out rows.jsonl \
        --tracker kalman --sigma-a 0.1 --methods ego_only,cooperative_ideal,uncertainty \
        --loss bernoulli --drop-probability 0.25 --seed 3
    python3 -m collision.replay --episodes episodes.jsonl --out rows.jsonl --loss burst --burst 11.0 12.0

`rows.jsonl` has one row per episode, method and frame: `warning`, `evidence_status`, and for every track its mode (`live`, `fallback`, `expired`), contact times, urgency and state. `--tracker reference` (default) is the two-point tracker; `--tracker kalman` is the Kalman tracker with process noise `--sigma-a` (default 0.1 m/s^2). From Python: `collision.replay.replay(episode, settings, mask, method, tracker_factory)`.

`collision/frames.py` builds episodes from detection rows in world coordinates (known-vehicle removal by bird's-eye-view IoU, ego velocity from the pose samples); `perception/heal_export/` exports detections from a HEAL checkpoint and converts them to world boxes.

## Live

Two terminals on the workstation, started in this order (details, environment variables and clock offsets in `live/README.md`):

    # 1. the loop: both cars -> HEAL -> live.jsonl
    QCAR_LIVE_YAW_OFFSET_DEG=3.0 QCAR_HEAL_ROOT=<fork clone> python live/run_live.py --realtime_conf live/conf_live.json --model_dir <model dir> --output_jsonl live.jsonl
    # 2. the collision module, tailing live.jsonl
    python live/online_collision.py --live live.jsonl --offsets offset_before.json --yaw 3.0 --out online/

`live/run_live.py` never commands a vehicle; it only subscribes to the cars' topics. Moving the cars is not part of this repository.

## Relation to the HEAL fork

Cooperative perception is not implemented in this repository. It comes from [`Saavkd7/QCAR-HEAL_Integration`](https://github.com/Saavkd7/QCAR-HEAL_Integration), a fork of [HEAL](https://github.com/yifanlu0227/HEAL) (Lu et al., ICLR 2024) that runs HEAL on the QCar testbed. This repository holds only our changes to it (`perception/`) and the parts built on its detections: tracking, the collision module and the live loop.

The detector is the fork `github.com/Saavkd7/QCAR-HEAL_Integration` at the pinned commit **`71606bc005eafde25c5c673a77251d2e1296c3e3`**. Its files are not copied here. Our changes are patches against that commit (`perception/diffs/`: camera extrinsics, the +3.0 degree yaw offset, the live-loop and viewer patches) plus plugins and model configs (`perception/plugins/`, `perception/model_dirs/`). `live/` holds the patched live-loop files as complete files; the patches in `perception/diffs/` are their differences to the fork. Setup steps are in `perception/README.md`.

## Checkpoints

Two checkpoints (cooperative, single-agent) are release assets, not in git: download them from the release [`checkpoints-v1`](https://github.com/daryelazo15-netizen/qcar-cooperative-collision-warning/releases/tag/checkpoints-v1). Names, sizes and SHA-256 are in `checkpoints/MANIFEST.md`.

## Tests

    scripts/run_tests.sh [log folder]

prints the `unittest` summary of the `tracking`, `collision`, `perception` and `live` suites. No test needs a vehicle, the detector or data outside this repository.

## Authors and credits

- **Daryel León** (PhD student, Concordia University): tracking, collision module, online collision wrapper, and the changes to the perception and live-loop code for this system.
- **Kevin Saavedra** (MSc, University of Calabria): integration of HEAL on the QCar testbed ([`QCAR-HEAL_Integration`](https://github.com/Saavkd7/QCAR-HEAL_Integration)), developed as part of the **MITACS** program, including the original live loop. The files in `live/` (`run_live.py`, `heal_input.py`, `viewer.py`) derive from his `qcar/realtime/` code; `perception/diffs/` shows the changes.
- Supervisors: **Chadi Assi** (Concordia University), **Wissam Fawaz** (Lebanese American University) and **Floriano De Rango** (University of Calabria).

Perception is built on **HEAL** by Yifan Lu et al. (*An Extensible Framework for Open Heterogeneous Collaborative Perception*, ICLR 2024). If you use this work, please also cite HEAL:

```bibtex
@inproceedings{lu2024an,
  title={An Extensible Framework for Open Heterogeneous Collaborative Perception},
  author={Lu, Yifan and Hu, Yue and Zhong, Yiqi and Wang, Dequan and Chen, Siheng and Wang, Yanfeng},
  booktitle={The Twelfth International Conference on Learning Representations},
  year={2024}
}
```

## License

The code in this repository is released under the MIT License (`LICENSE`). The HEAL fork it builds on is not part of this repository and keeps its own license; the perception and live-loop code here load HEAL and are subject to that license for academic use.
