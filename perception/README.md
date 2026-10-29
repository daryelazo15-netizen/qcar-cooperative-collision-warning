# Perception: HEAL on the QCars

The detector is HEAL (pyramid fusion) on front-camera images of two QCars. The fork is **not copied here**; it is used at one pinned commit and our changes are patches and new files.

**Pinned commit: `71606bc005eafde25c5c673a77251d2e1296c3e3`** of `github.com/Saavkd7/QCAR-HEAL_Integration`.

## Set up the fork

    git clone https://github.com/Saavkd7/QCAR-HEAL_Integration && cd QCAR-HEAL_Integration && git checkout 71606bc005eafde25c5c673a77251d2e1296c3e3
    git apply /path/to/this/repo/perception/diffs/extrinsic.diff
    git apply /path/to/this/repo/perception/diffs/vehicles.diff

These two patches are what the fork needs for the extrinsics. The other patches (`heal_input`, `run`, `viewer`, `conf`) document how the complete files in `../live/` differ from the fork's `qcar/realtime/` files; they assume the `live/` folder layout and are not meant to be applied inside the fork. All six pass `git apply --check` against the pinned commit, separately and together.

Install the fork's environment (torch, opencood; see its README) and set `QCAR_HEAL_ROOT` to the clone when you run `../live/run_live.py`. A fresh clone also needs two files that the fork does not track:

- the compiled box-overlap extension: run `python opencood/utils/setup.py build_ext --inplace` in the clone;
- `qcar_testbed_integration/machine_paths.json`: copy `machine_paths.example.json` and fill in the local paths.

| patch | against | what it does |
|---|---|---|
| `diffs/extrinsic.diff` | `qcar/patches/patch_real_extrinsic.py` | real camera-to-vehicle rotation instead of HEAL's hard-coded fix-up matrix; car 1 fitted, car 2 nominal (same content as `plugins/camera_extrinsic.py`) |
| `diffs/vehicles.diff` | `qcar_testbed_integration/calibration/vehicles.json` | car 1 uses its fitted extrinsic entry, car 2 a nominal extrinsic tilted 3.0 degrees down (new entry) |
| `diffs/heal_input.diff` | `qcar/realtime/heal_input.py` | +3.0 degree (per-session) Vicon yaw offset on every agent yaw, the extrinsic plugin registered under the fork's module name, checkpoint hash and tensor-by-tensor check, retry of a Windows file replace |
| `diffs/run.diff`, `diffs/viewer.diff`, `diffs/conf.diff` | `qcar/realtime/{run.py,viewer.py,conf.json}` | the live loop with loop modes and link impairment, the viewer with the collision alarm, the live configuration (see `../live/`) |

## Plugins and model configs

| path | what |
|---|---|
| `plugins/camera_extrinsic.py` | registered as `qcar.patches.patch_real_extrinsic` (the live loop does this itself) |
| `plugins/camera_tilt_car2.py` | tilts car 2's camera extrinsic by 3.0 degrees down; loaded after `camera_extrinsic.py` |
| `plugins/exclude_reference_ego.py` | removes the reference ego car from fused targets and predictions |
| `model_dirs/coop/resolved_hypes.json`, `model_dirs/single/resolved_hypes.json` | resolved HEAL configuration of the cooperative and the single-agent detector; the checkpoint goes next to it (`../checkpoints/MANIFEST.md`) |
| `model_dirs/modality_assign_real.json` | modality assignment for both agents in the live loop (copy it next to each `resolved_hypes.json`). It has one generic `"scenario"` key. For offline export on a built dataset, point `heter.assignment_path` at the `modality_assign.json` that the dataset builder writes next to the data instead; that file is keyed by scenario name |

Placeholders to replace in `resolved_hypes.json`: `<REPO_ROOT>` (this repository), `<MODEL_DIR>` (folder holding the checkpoint and `modality_assign_real.json`), `<DATASET_ROOT>` (only used for training or offline export).

## Detection export to world coordinates

`heal_export/export_detections.py` (run from the fork's root with its interpreter) writes cooperative and ego-only detections per frame for one checkpoint and dataset split. `heal_export/to_world.py` converts them to world boxes (`heal_export/world_boxes.box_world`, the function the online collision wrapper uses) and matches every exported frame to its source frame by the ego car's motion-capture pose (poses file format and options in its docstring; `--ego-car` names the ego car in that file). The rows it writes carry `frame`, `time`, `session`, `ego` pose and the boxes; add `image_time_vicon_s` (image capture time on the pose clock) and they go to `../collision/frames.py`.

## Open

- Training script: the checkpoints were trained with a `qcar/train.py` that differs from the pinned commit's only in line endings and in one usage example in its docstring (`-y qcar/configs/camera_attfuse_pretrain.yaml` instead of `-y qcar/configs/zoo/pyramid.yaml`); the pinned commit's script is used as is.
- Training (data builder, configs, job scripts) is not part of this repository.
