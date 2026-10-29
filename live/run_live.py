"""Real-time loop: the QCars' ROS topics -> HEAL -> world-frame detections (one JSON line per pair), with an optional live viewer.
Patched copy of qcar/realtime/run.py of the HEAL fork (changes: ../perception/diffs/run.diff). Pair the output with
online_collision.py, which reads the jsonl while it is written.

    python live/run_live.py --realtime_conf live/conf_live.json --model_dir <model dir> [--view [--composite] --online_log <online_log.jsonl>]
    python live/run_live.py --mode ego_only --single_model_dir <single-agent model dir> ...
    python live/run_live.py --mode fallback --single_model_dir <...> --impair '{"type": "bernoulli", "p": 0.25, "seed": 1}'
    python live/run_live.py --no_model                # receive + format only (check the link)

QCAR_LIVE_YAW_OFFSET_DEG (degrees added to every Vicon yaw) and QCAR_HEAL_ROOT (clone of the HEAL fork at the pinned commit) must be set.

Modes
  --mode coop       cooperative model (model_dir), both cars fused
  --mode ego_only   single-agent model (--single_model_dir) on the ego frame only; the peer is paired for its pose but never fused
  --mode fallback   both models loaded; a pair whose peer frame the injector withholds runs the single-agent model, the others the cooperative one
  --impair <json>   injector spec of impairment.py (fallback mode only), decided at the pairing step, logged in every record
Every record gets: mode, stream (coop | ego_only: which model produced the boxes), peer_status (used | delayed | withheld), peer_delay_s,
impairment (reason), model_tag. READY is printed once all models of the mode are loaded: `[take] READY t0=<unix>`; the injector's t0 is that time.

Pipeline, per car (both run concurrently):
  1. ros1.TopicSubscriber streams /qcar/csi_front (+ other cameras if asked)
     and /qcar/vicon straight from that car's roscore (no ROS install here).
  2. sync.CarSync assembles frames with the dataset converter's own
     causal bounded-wait logic -> the same content as a converted frame
     folder (front.png + ego_vicon_pose.json + timestamp.json).
Then, across cars:
  3. the newest ego frame (agent 1) is paired with the peer frame (agent 2)
     nearest in source time, within build_inference.py's pair_tolerance_sec.
     Older unprocessed frames are skipped -- real time means latest wins.
  4. heal_input.LiveHealInput writes the pair in build_inference.py's exact
     InferenceFront format (on tmpfs), and heal_input.LiveModel pushes it
     through HEAL's own dataset + the chosen model + decode/NMS.
  5. one JSON line per fused pair goes to output_jsonl: detections (ego frame,
     model space, and world metres), scores, timing, pairing offsets.

Cross-car pairing compares the two cars' own ROS stamps, exactly like the
offline dataset; the cars' clocks must be synchronized (NTP/chrony) for it
to mean anything. The periodic stats print each car's (arrival - stamp) so a
clock offset is visible instead of silently breaking the pairing.
"""
import argparse
import collections
import datetime
import json
import os
import queue
import statistics
import sys
import tempfile
import threading
import time

sys.dont_write_bytecode = True  # never write .pyc into the HEAL tree
HERE = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.dirname(HERE)
HEAL_ROOT = os.environ.get("QCAR_HEAL_ROOT", "<QCAR_HEAL_ROOT>")
sys.path[:0] = [HERE, REPO_ROOT, HEAL_ROOT]

import cv2
import numpy as np

from qcar import config
from qcar.realtime import ros1
from qcar.realtime.sync import CarSync
import heal_input  # patched copy of the fork's qcar/realtime/heal_input.py, same folder
from impairment import Injector  # noqa: E402

DEFAULT_CONF = os.path.join(HERE, "conf_live.json")


def realtime_conf_path():
    argv = sys.argv[1:]
    for i, arg in enumerate(argv):
        if arg == "--realtime_conf" and i + 1 < len(argv):
            return os.path.abspath(argv[i + 1])
        if arg.startswith("--realtime_conf="):
            return os.path.abspath(arg.split("=", 1)[1])
    return DEFAULT_CONF


def parse_args():
    rt = config.load_conf(realtime_conf_path())
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--realtime_conf", default=DEFAULT_CONF,
                    help="this script's conf.json")
    ap.add_argument("--model_dir", default=None,
                    help="trained run dir with resolved_hypes.json (default: conf model_dir)")
    ap.add_argument("--master", action="append", default=[], metavar="AGENT=URI",
                    help="override one car's ROS master, e.g. 1=http://<ego-ip>:11311")
    ap.add_argument("--cameras", default=None,
                    help="comma list of cameras to subscribe to (default: conf cameras)")
    ap.add_argument("--live_dir", default=None, help="tmpfs dir HEAL reads (default: conf)")
    ap.add_argument("--record_dir", default=None,
                    help="also save every fused pair in converter layout (default: conf)")
    ap.add_argument("--output_jsonl", default=None, help="default: conf output_jsonl")
    ap.add_argument("--amp", dest="amp", action="store_true")
    ap.add_argument("--no_amp", dest="amp", action="store_false")
    ap.set_defaults(amp=rt["amp"])
    ap.add_argument("--no_view", dest="view", action="store_false",
                    help="do not open the live viewer (default: conf view)")
    ap.set_defaults(view=rt["view"])
    ap.add_argument("--view", dest="view", action="store_true", help="open the live viewer (conf_live.json has view=false)")
    ap.add_argument("--composite", action="store_true", help="one window: ego POV + cooperator POV on top, bird's-eye view below, alarm banner across")
    ap.add_argument("--fullscreen", action="store_true", help="start the composite window fullscreen (f / F11 toggles)")
    ap.add_argument("--online_log", default=None, help="the online module's online_log.jsonl of this session (read-only tail; drives the alarm and the tracks)")
    ap.add_argument("--snapshot_dir", default=None, help="save the composited image as PNG here, plus alarm_trace.jsonl")
    ap.add_argument("--snapshot_every", type=int, default=10, help="one PNG every N draws")
    ap.add_argument("--walls_frame", default="old", choices=["old", "new"], help="which walls.json frame to draw ({old,new} = the two motion-capture frames of the lab)")
    ap.add_argument("--walls_json", default=os.path.join(HERE, "walls.json"), help="wall segments of the scene in metres, per motion-capture frame")
    ap.add_argument("--warmup_dir", default=os.path.join(HERE, "warmup_pair"), help="folder with front_1.png, front_2.png, pose_1.json, pose_2.json: one stored camera pair (ego, peer) and its motion-capture poses; every model is loaded and warmed up on it before the first live pair")
    ap.add_argument("--source_label", default="live", help="shown in the status line and the title: 'live' or e.g. 'REPLAY of dev episode (no cars)'")
    ap.add_argument("--view_offscreen", action="store_true", help="TEST: draw with the same code but open no window (snapshots only)")
    ap.add_argument("--no_model", action="store_true",
                    help="receive, pair and format only; do not load/run HEAL")
    ap.add_argument("--mode", choices=("coop", "ego_only", "fallback"), default="coop")
    ap.add_argument("--single_model_dir", default=None, help="single-agent model dir (ego_only and fallback modes)")
    ap.add_argument("--impair", default=None, help="injector spec, JSON (impairment.py); fallback mode only")
    ap.add_argument("--max_pairs", type=int, default=None,
                    help="stop after this many fused pairs (testing)")
    config.add_module_args(ap)  # --conf (qcar/conf.json), --plugins, --set, ...
    opt = ap.parse_args()

    opt.model_dir = config.cli_path(opt.model_dir) if opt.model_dir else rt.path_of("model_dir")
    opt.live_dir = config.cli_path(opt.live_dir) if opt.live_dir else rt.path_of("live_dir")
    if sys.platform.startswith("win") and opt.live_dir.replace("\\", "/").startswith("/dev/shm"):
        # no tmpfs on Windows: use the user's temp dir instead of C:\dev\shm
        opt.live_dir = os.path.join(tempfile.gettempdir(), "qcar_realtime")
    if opt.mode != "coop" and not opt.single_model_dir:
        raise SystemExit("--mode %s needs --single_model_dir" % opt.mode)
    if opt.impair and opt.mode != "fallback":
        raise SystemExit("--impair needs --mode fallback (the cooperative-only and single-agent-only loops have no peer frame to withhold)")
    opt.single_model_dir = config.cli_path(opt.single_model_dir) if opt.single_model_dir else None
    opt.record_dir = (config.cli_path(opt.record_dir) if opt.record_dir
                      else rt.path_of("record_dir"))
    opt.output_jsonl = (config.cli_path(opt.output_jsonl) if opt.output_jsonl
                        else rt.path_of("output_jsonl"))
    opt.cameras = config.split_list(opt.cameras) if opt.cameras else list(rt["cameras"])
    opt.master_overrides = {}
    for item in opt.master:
        aid, _, uri = item.partition("=")
        opt.master_overrides[aid.strip()] = uri.strip()
    return opt, rt


def box_record(corners, score, ego_pose):
    """One detection: ego-frame corners in model space (x10 world, what HEAL
    outputs) plus the centre in world metres (Vicon frame)."""
    center = corners.mean(axis=0)
    yaw = ego_pose["yaw"] + heal_input.YAW_OFFSET_RAD  # the yaw the model was given (the training data's convention)
    c, s = np.cos(yaw), np.sin(yaw)
    wx = (c * center[0] - s * center[1]) / 10.0 + ego_pose["x"]
    wy = (s * center[0] + c * center[1]) / 10.0 + ego_pose["y"]
    return {"score": float(score), "corners_ego_model": np.round(corners, 4).tolist(),
            "center_world_m": [round(float(wx), 4), round(float(wy), 4)]}


class Recorder(threading.Thread):
    """Saves fused pairs in bag_to_dataset_rosbags.py's per-frame layout,
    off the inference thread (PNG encoding is slow)."""

    def __init__(self, root, session, bi, conv, syncs):
        threading.Thread.__init__(self, daemon=True)
        self.q = queue.Queue(maxsize=64)
        self.dirs = dict((aid, os.path.join(root, bi.CAR_REGISTRY[aid]["name"], "dataset",
                                            "converted_live_%s" % session))
                         for aid in bi.CAR_REGISTRY)
        self.conv, self.syncs = conv, syncs
        self.dropped = 0

    def put(self, pair_id, frames):
        try:
            self.q.put_nowait((pair_id, frames))
        except queue.Full:
            self.dropped += 1

    def run(self):
        while True:
            pair_id, frames = self.q.get()
            for aid, fr in frames.items():
                d = os.path.join(self.dirs[aid], "%06d" % pair_id)
                os.makedirs(d, exist_ok=True)
                for name, msg in fr["images"].items():
                    img = fr["front_bgr"] if name == "front" else self.syncs[aid].decode_image(msg)
                    cv2.imwrite(os.path.join(d, name + ".png"), img,
                                [cv2.IMWRITE_PNG_COMPRESSION, 0])
                self.conv.atomic_json(os.path.join(d, "ego_vicon_pose.json"), fr["ego_pose"])
                ts = dict(fr["timestamp"], frame_id="%06d" % pair_id)
                self.conv.atomic_json(os.path.join(d, "timestamp.json"), ts)
            self.q.task_done()


def main():
    opt, rt = parse_args()
    qconf = config.load_conf(opt.conf)
    os.chdir(config.REPO_ROOT)
    session = datetime.datetime.now().strftime("%Y%m%d%H%M%S")

    bi = heal_input.import_script(rt.path_of("build_inference_py"), "qcar_rt_build_inference")
    conv = heal_input.import_script(rt.path_of("converter_py"), "qcar_rt_bag_to_dataset")
    sync_conf = json.load(open(rt.path_of("convert_conf")))
    agents = sorted(bi.CAR_REGISTRY)
    ego_id, peer_id = bi.EGO_ID, bi.PEER_ID
    # each car's ROS master: its current IP (calibration/vehicles.json) in conf.json's ros_master_uri
    opt.masters = dict((a, rt["ros_master_uri"].format(ip=bi.CAR_REGISTRY[a]["ip"])) for a in agents)
    opt.masters.update(opt.master_overrides)
    missing = [a for a in agents if a not in opt.masters]
    if missing:
        raise SystemExit("no ROS master configured for agent(s) %s" % missing)
    pair_tol = float(bi.PAIR_TOLERANCE_SEC)
    pair_wait = float(rt["pair_wait_sec"])

    print("[realtime] session %s | ego=%s peer=%s | cameras=%s | fps=%s wait=%.3fs "
          "pair_tol=%.3fs" % (session, ego_id, peer_id, opt.cameras, sync_conf["fps"],
                               sync_conf["wait_sec"], pair_tol))

    live = heal_input.LiveHealInput(bi, opt.live_dir)
    models = {}                      # "coop" / "ego_only" -> LiveModel
    injector = Injector(json.loads(opt.impair) if opt.impair else None)
    if not opt.no_model:
        # Load and warm up EVERY model of the mode now, on a stored pair, before any live pair: in a real take the cars' topics only
        # exist once the user starts the launch (which also starts the motion), so waiting for the first live pair would never print READY.
        # The warm-up pair goes through the same formatter and the same infer(); it is not a record: nothing is written to the output jsonl
        # or the recorder, and the online module reads only that jsonl. The first live pair overwrites the files in place (the normal path).
        wp = opt.warmup_dir
        wframes = {aid: {"front_bgr": cv2.imread(os.path.join(wp, "front_%s.png" % aid)),
                         "ego_pose": json.load(open(os.path.join(wp, "pose_%s.json" % aid)))} for aid in (ego_id, peer_id)}
        live.write_pair(wframes)
        print("[realtime] loading model(s) for mode %s on the stored warm-up pair" % opt.mode, flush=True)
        if opt.mode in ("coop", "fallback"):
            models["coop"] = heal_input.LiveModel(opt.model_dir, opt.live_dir, qconf, opt, opt.amp)
        if opt.mode in ("ego_only", "fallback"):
            models["ego_only"] = heal_input.LiveModel(opt.single_model_dir, opt.live_dir, qconf, opt, opt.amp)
        for tag, m in models.items():
            print("[realtime] model %s on %s, amp=%s" % (tag, m.device, m.amp))
        warm = {}
        for tag, m in models.items():       # one discarded inference per model (first-use cost: 0.52 s measured for a cold single-agent model)
            tw = time.time()
            m.infer()
            warm[tag] = time.time() - tw
        import torch
        mem = {"cuda_allocated_mb": torch.cuda.memory_allocated() / 2 ** 20, "cuda_reserved_mb": torch.cuda.memory_reserved() / 2 ** 20}
        t_ready = time.time()
        injector.set_ready(t_ready)
        meta = {"t_ready_ws": t_ready, "mode": opt.mode, "impair": json.loads(opt.impair) if opt.impair else None,
                "model_dir": opt.model_dir if "coop" in models else None, "single_model_dir": opt.single_model_dir if "ego_only" in models else None,
                "yaw_offset_deg": heal_input.YAW_OFFSET_DEG, "amp": opt.amp, "warmup_s": warm, "ready_before_first_pair": True, **mem}
        json.dump(meta, open(opt.output_jsonl.format(session=session) + ".meta.json", "w"), indent=1)
        print("[take] READY t0=%.3f mode=%s impair=%s" % (t_ready, opt.mode, opt.impair), flush=True)
    syncs = dict((a, CarSync(conv, opt.cameras, sync_conf, rt["max_network_lag_sec"]))
                 for a in agents)
    subs = []
    for aid in agents:
        for name in syncs[aid].cameras + ["vicon"]:
            cb = (lambda s, n: lambda raw, mt, arr: s.push(n, raw, mt, arr))(syncs[aid], name)
            sub = ros1.TopicSubscriber(opt.masters[aid], rt["topics"][name], cb,
                                       float(rt["reconnect_sec"]))
            sub.agent, sub.name = aid, name
            sub.start()
            subs.append(sub)
        print("[realtime] agent %s (%s) <- %s" % (aid, bi.CAR_REGISTRY[aid]["name"],
                                                  opt.masters[aid]))

    recorder = None
    if opt.record_dir:
        recorder = Recorder(opt.record_dir, session, bi, conv, syncs)
        recorder.start()
        print("[realtime] recording to %s" % opt.record_dir)
    out = None
    if not opt.no_model:
        os.makedirs(os.path.dirname(opt.output_jsonl.format(session=session)), exist_ok=True)
        out = open(opt.output_jsonl.format(session=session), "w")

    viewer = None
    if opt.view:
        if sys.platform.startswith("linux") and not os.environ.get("DISPLAY") and not opt.view_offscreen:
            print("[realtime] no DISPLAY -- viewer disabled (use --no_view to silence this)")
        else:
            from viewer import LiveViewer
            walls = json.load(open(opt.walls_json))["walls"][opt.walls_frame]     # static_targets / parked_vehicles are never drawn
            import hashlib
            print("[realtime] viewer walls: %s walls.%s (%d segments) sha256 %s" % (opt.walls_json, opt.walls_frame, len(walls), hashlib.sha256(open(opt.walls_json, "rb").read()).hexdigest()), flush=True)
            viewer = LiveViewer(agents, ego_id,
                                    dict((a, bi.CAR_REGISTRY[a]["name"]) for a in agents),
                                    dict((a, bi.CAM_PARAMS[a]["K"]) for a in agents),
                                    walls, rt["view_bev_range_m"], rt["view_max_hz"], online_log=opt.online_log, mode=opt.mode,
                                    composite=opt.composite, fullscreen=opt.fullscreen, snapshot_dir=opt.snapshot_dir,
                                    snapshot_every=opt.snapshot_every, offscreen=opt.view_offscreen, yaw_offset_rad=heal_input.YAW_OFFSET_RAD, source=opt.source_label)
            print("[realtime] viewer open (%s%s; alarm log: %s)" % ("composite" if opt.composite else "3 windows", ", offscreen" if opt.view_offscreen else "", opt.online_log))

    peer_hist = collections.deque()  # recent peer frames (only kept for the delay injector)
    queues = dict((a, collections.deque(maxlen=64)) for a in agents)
    stats = collections.Counter()
    latencies = collections.deque(maxlen=200)
    skew = dict((a, collections.deque(maxlen=200)) for a in agents)
    last_stats = time.time()
    last_bytes = dict((id(s), 0) for s in subs)
    n_pairs = 0

    try:
        while opt.max_pairs is None or n_pairs < opt.max_pairs:
            now = time.time()
            for aid in agents:
                for fr in syncs[aid].poll(now):
                    queues[aid].append(fr)
                    if aid == peer_id and injector.needs_history():
                        peer_hist.append(fr)
                        while peer_hist and fr["stamp"] - peer_hist[0]["stamp"] > 3.0:
                            peer_hist.popleft()
                    skew[aid].append(fr["arrival"] - fr["stamp"])

            pair = None
            eq, pq = queues[ego_id], queues[peer_id]
            for i in range(len(eq) - 1, -1, -1):
                e = eq[i]
                cands = [p for p in pq if abs(p["stamp"] - e["stamp"]) <= pair_tol]
                if cands:
                    pair = (e, min(cands, key=lambda p: abs(p["stamp"] - e["stamp"])))
                    stats["skipped_stale_ego"] += i
                    for _ in range(i + 1):
                        eq.popleft()
                    while pq and pq[0]["stamp"] <= pair[1]["stamp"]:
                        pq.popleft()
                    break
            while eq and pair is None and now - eq[0]["arrival"] > pair_wait:
                eq.popleft()
                stats["dropped_no_peer"] += 1
            while pq and now - pq[0]["arrival"] > pair_wait + 1.0:
                pq.popleft()

            if viewer is not None:
                if viewer.quit:
                    break
                if n_pairs == 0 and now - viewer.last_draw >= 1.0:
                    viewer.waiting("Waiting for data... frames: %s | pairs: 0" % ", ".join(
                        "CAV %s=%d" % (a, syncs[a].frames) for a in agents))
                    viewer.last_draw = now
                viewer.pump()
            if pair is None:
                time.sleep(0.002)
            else:
                e, p = pair
                decision = injector.decide(now, e, p, list(peer_hist)) if models else {"status": "used", "peer": p, "delay_s": e["stamp"] - p["stamp"], "reason": None, "peer_none": True}
                use_peer = decision["peer"]
                # fused model input: the ego frame and the peer frame the injector lets through (the paired one when withheld: a withheld pair
                # is processed by the single-agent model, which excludes the peer by comm_range)
                frames = {}
                for aid, fr in zip((ego_id, peer_id), (e, use_peer if use_peer is not None else p)):
                    fr["front_bgr"] = syncs[aid].decode_image(fr["images"]["front"])
                    frames[aid] = fr
                t_write = time.perf_counter()
                live.write_pair(frames)
                format_s = time.perf_counter() - t_write
                if recorder is not None:
                    recorder.put(n_pairs, frames)
                if not opt.no_model:
                    stream = ("ego_only" if opt.mode == "ego_only" or decision["status"] == "withheld" else "coop")
                    model = models[stream]
                    boxes, scores, timing = model.infer()
                    done = time.time()
                    rec = {
                        "pair": n_pairs,
                        "ego_stamp": e["stamp"], "peer_stamp": p["stamp"], "peer_stamp_used": None if decision["peer"] is None else decision["peer"]["stamp"],
                        "pair_offset_sec": p["stamp"] - e["stamp"],
                        "latency_from_last_arrival_sec": done - max(e["arrival"], p["arrival"]),
                        "latency_from_ego_capture_sec": done - e["stamp"],
                        "format_s": format_s, **timing, "yaw_offset_deg": heal_input.YAW_OFFSET_DEG,
                        "mode": opt.mode, "stream": stream, "model_tag": "coop_epoch5" if stream == "coop" else "single_agent",
                        "peer_status": decision["status"] if opt.mode != "ego_only" else "not_used",
                        "peer_delay_s": decision["delay_s"], "impairment_reason": decision["reason"],
                        "ego_pose": e["ego_pose"], "peer_pose": p["ego_pose"],
                        "detections": [box_record(b, s, e["ego_pose"])
                                       for b, s in zip(boxes, scores)],
                    }
                    for k in ("draw", "history_frames", "nearest_gap_s"):
                        if k in decision:
                            rec["impairment_" + k] = decision[k]
                    out.write(json.dumps(rec) + "\n")
                    out.flush()
                    latencies.append(rec["latency_from_last_arrival_sec"])
                    stats["detections"] += len(scores)
                    stats["stream_" + stream] += 1
                if viewer is not None and viewer.due():
                    if opt.no_model:
                        viewer.update(frames, np.zeros((0, 8, 3)), np.zeros(0),
                                      {"pair": n_pairs, "mode": opt.mode, "status": "no model (--no_model)"})
                    else:
                        viewer.update(frames, boxes, scores, {
                            "pair": n_pairs, "n_agents_fused": timing.get("n_agents_fused", "-"),
                            "mode": opt.mode, "latency_s": rec["latency_from_ego_capture_sec"], "peer_status": rec["peer_status"],
                            "status": "latency %.0f ms | model %.0f ms" % (
                                rec["latency_from_last_arrival_sec"] * 1000,
                                timing.get("model_s", 0) * 1000)})
                n_pairs += 1
                stats["pairs"] += 1

            if now - last_stats >= float(rt["stats_every_sec"]):
                dt = now - last_stats
                parts = []
                for s in subs:
                    rate = (s.bytes - last_bytes[id(s)]) / dt / 1e6
                    last_bytes[id(s)] = s.bytes
                    parts.append("%s/%s %d msgs %.1fMB/s" % (s.agent, s.name, s.messages, rate))
                print("[stats] " + " | ".join(parts))
                for aid in agents:
                    med = statistics.median(skew[aid]) if skew[aid] else float("nan")
                    print("[stats] agent %s: %d frames, arrival-stamp median %.3fs, rejected %s"
                          % (aid, syncs[aid].frames, med, dict(syncs[aid].rejections)))
                lat = statistics.median(latencies) if latencies else float("nan")
                print("[stats] pairs %d, dropped_no_peer %d, skipped_stale %d, "
                      "median latency %.3fs%s" % (stats["pairs"], stats["dropped_no_peer"],
                                                  stats["skipped_stale_ego"], lat,
                                                  (", recorder dropped %d" % recorder.dropped)
                                                  if recorder else ""))
                last_stats = now
    except KeyboardInterrupt:
        pass
    finally:
        for s in subs:
            s.stop()
        if recorder is not None:
            recorder.q.join()  # finish writing queued pairs before exiting
        if out is not None:
            out.close()
            print("[realtime] detections: %s" % out.name)
        print("[realtime] %d pairs fused, %s" % (n_pairs, dict(stats)))


if __name__ == "__main__":
    main()
