# Live loop, online collision module and viewer

| file | role |
|---|---|
| `run_live.py` | the real-time loop: both cars' ROS topics -> frame pairing -> HEAL -> one JSON line per pair (boxes in the ego frame and in world metres, poses, timing). Modes `coop`, `ego_only`, `fallback` (with the link-impairment injector). Only subscribes to topics; it never publishes or moves a car |
| `heal_input.py` | patched copy of the fork's `qcar/realtime/heal_input.py` (yaw offset, extrinsic plugin, checkpoint check) |
| `online_collision.py` | the collision module on the live records, one frame at a time (tails the jsonl or `--simulate` on a finished file); writes `online_rows*.jsonl` and `online_log.jsonl` |
| `viewer.py` | live viewer: camera views with detections, bird's-eye view with tracks, collision banner from the online log |
| `impairment.py` | link-impairment injector (`window`, `window_trigger`, `bernoulli`, `delay`) for `--mode fallback` |
| `clock_offset.py` | measures the clock offset car - workstation over ssh (read-only `date` queries) for `online_collision.py --offsets` |
| `conf_live.json`, `walls.json` | live configuration (template: replace `<QCAR_LIVE_ROOT>`), wall segments for the bird's-eye view |
| `tests/`, `fixtures/` | unit tests (no vehicle, no HEAL): online wrapper online == batch in every mode and for both trackers, impairment, viewer alarm |

## Environment

| variable | meaning |
|---|---|
| `QCAR_HEAL_ROOT` | clone of the HEAL fork at the pinned commit, patches applied (`../perception/README.md`) |
| `QCAR_LIVE_YAW_OFFSET_DEG` | degrees added to every Vicon yaw (3.0 for the checkpoints in `../checkpoints/MANIFEST.md`); no default, the loop refuses to start without it |
| `QCAR_LIVE_EXTRINSIC_OVERRIDE` | set to `0` for control runs that must use the fork's own extrinsic module |
| `QCAR_EGO_HOST` | `user@host` of the ego car, the key of its entry in `offset_before.json` (default `nvidia@EGO_HOST`; or `--ego_host`) |
| `QCAR_SSH_KEY` | optional ssh key file for `clock_offset.py` |

`conf_live.json` is a template: replace `<QCAR_LIVE_ROOT>` (workspace folder with `model_dir/`), and the car addresses come from the fork's `calibration/vehicles.json` (`--master 1=http://<ego-ip>:11311` overrides one).
`run_live.py --warmup_dir <folder>` needs `front_1.png`, `front_2.png`, `pose_1.json`, `pose_2.json`: one stored camera pair (ego, peer) with its motion-capture poses; every model is loaded and warmed up on it before the first live pair. TODO: no warm-up pair is shipped.

## Run

    python live/clock_offset.py --out offset_before.json --n 20 nvidia@<EGO_HOST> nvidia@<PEER_HOST>     # Windows Python of the workstation
    QCAR_LIVE_YAW_OFFSET_DEG=3.0 QCAR_HEAL_ROOT=<fork clone> python live/run_live.py \
        --realtime_conf live/conf_live.json --model_dir <coop model dir> --output_jsonl live.jsonl [--view --composite --online_log online/online_log.jsonl]
    python live/online_collision.py --live live.jsonl --offsets offset_before.json --yaw 3.0 --out online/ [--mode coop|ego_only|fallback] [--tracker kalman --sigma-a 0.1]

`online_collision.py` prints one `WARNING ...` line when a warning starts. In `fallback` mode the primary method is `uncertainty`; `no_fallback`, `hold` and `cv` are logged too (`online_rows_<method>.jsonl`).

Online == batch: `collision.replay.replay` on the frames the wrapper built gives the same rows (`tests/test_online_collision.py`).
