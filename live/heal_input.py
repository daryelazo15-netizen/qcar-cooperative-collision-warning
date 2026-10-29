"""Live HEAL input and model for the QCar testbed. Patched copy of qcar/realtime/heal_input.py of the HEAL fork (diff: ../perception/diffs/heal_input.diff).
Changes against the fork: (1) a per-session Vicon yaw offset (QCAR_LIVE_YAW_OFFSET_DEG) is added to every agent yaw, as the training data
were built; (2) the camera extrinsic of ../perception/plugins/camera_extrinsic.py is installed under its module name before the plugins load;
(3) the loaded checkpoint is hashed and checked tensor by tensor (HEAL's loader uses strict=False); (4) os.replace is retried on Windows'
transient PermissionError (no change on the success path); (5) the extrinsic registration is idempotent, so a second LiveModel
(single-agent model next to the cooperative one) can be built in the same process. Everything else is the original.

Turn one live (ego, peer) frame pair into HEAL's input, and run the model.

FORMAT: identical to InferenceFront/ (build_inference.py), because the
formatting code is IMPORTED from build_inference.py, not re-derived:
CAR_REGISTRY (agent "1" = ego car, "2" = cooperating car -- the ids
the extrinsic patch keys its calibration on), CAM_PARAMS (real per-car
fisheye K/D + BEV visibility mask) and build_agent_yaml() (x10 world scale,
lidar_pose, camera0 cords/intrinsic, empty vehicles{}). The only change is
undistortion: the same cv2.fisheye model (Knew=K) through a remap table
computed once per car, instead of undistortImage() recomputing it per frame.

HOW HEAL CONSUMES IT: one fixed scenario on tmpfs,

    <live_dir>/heal/qcar_live/1/000000.yaml, 000000_camera0.png, 000000_bev_visibility.png
    <live_dir>/heal/qcar_live/2/...
    <live_dir>/staging/   (files are written here, then os.replace'd into heal/;
                           kept outside heal/ because HEAL treats every
                           folder under its root as a scenario)

HEAL's dataset is built over it ONCE (it only records paths at init) and
re-reads the files on every ds[0], so each new pair is written in place
(atomic os.replace) and ds[0] goes through HEAL's full, unmodified
preprocessing -- plugins (1-camera loader, real extrinsic), LSS geometry,
comm_range, collate. Writing and reading happen in the same thread, so a
read never sees a half-updated pair.
"""
import hashlib
import importlib
import importlib.util
import json
import math
import os
import shutil
import sys
import time

import cv2
import numpy as np
import torch

SCENARIO = "qcar_live"
FRAME = "000000"
HERE = os.path.dirname(os.path.abspath(__file__))
EXTRINSIC_PLUGIN = os.path.join(os.path.dirname(HERE), "perception", "plugins", "camera_extrinsic.py")
# The training data were built with a per-session offset added to every Vicon yaw. No default:
# the live session's value must be stated by the caller.
if "QCAR_LIVE_YAW_OFFSET_DEG" not in os.environ:
    raise SystemExit("set QCAR_LIVE_YAW_OFFSET_DEG (deg added to every Vicon yaw, as the training data were built)")
YAW_OFFSET_DEG = float(os.environ["QCAR_LIVE_YAW_OFFSET_DEG"])
YAW_OFFSET_RAD = math.radians(YAW_OFFSET_DEG)
print("[live] Vicon yaw offset +%.3f deg added to every agent yaw" % YAW_OFFSET_DEG)


def replace_retry(src, dst, tries=20, wait=0.05):
    """os.replace, retried on Windows' transient PermissionError (a scanner or indexer holding the target for a moment;
    seen once in 3263 pairs of the equivalence test: [WinError 5] on staging/000000.yaml)."""
    for i in range(tries):
        try:
            return os.replace(src, dst)
        except PermissionError:
            if i == tries - 1:
                raise
            time.sleep(wait)


def install_extrinsic():
    """Register perception/plugins/camera_extrinsic.py (car 1 fitted / car 2 nominal)
    under the module name qcar.patches.patch_real_extrinsic, before config.resolve_modules imports it.
    camera_tilt_car2.py then edits this same module object (car 2 tilt)."""
    name = "qcar.patches.patch_real_extrinsic"
    if os.environ.get("QCAR_LIVE_EXTRINSIC_OVERRIDE", "1") == "0":  # control runs only: the fork's own version
        print("[live] CONTROL: the fork's patch_real_extrinsic left in place")
        return
    if name in sys.modules:
        if getattr(sys.modules[name], "__file__", None) and os.path.samefile(sys.modules[name].__file__, EXTRINSIC_PLUGIN):
            return      # a second LiveModel in the same process (loop modes with two models): already registered
        raise SystemExit("%s already imported; camera_extrinsic.py must be registered first" % name)
    importlib.import_module("qcar.patches")
    path = EXTRINSIC_PLUGIN
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    print("[live] patch_real_extrinsic registered from %s" % path)


def import_script(path, name):
    """Import a pipeline script by path with a clean argv: those scripts
    parse flags at import time (parse_known_args), and this process's own
    flags must not leak into them."""
    saved = sys.argv
    sys.argv = [path]
    try:
        spec = importlib.util.spec_from_file_location(name, path)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
    finally:
        sys.argv = saved
    return module


class LiveHealInput:
    def __init__(self, build_inference, live_dir):
        self.bi = build_inference
        self.live_dir = live_dir
        self.agent_ids = sorted(self.bi.CAR_REGISTRY)
        self.maps = {}
        if os.path.isdir(live_dir):
            shutil.rmtree(live_dir)
        for aid in self.agent_ids:
            os.makedirs(self._cav_dir(aid))
            vis_dst = os.path.join(self._cav_dir(aid), "%s_bev_visibility.png" % FRAME)
            shutil.copy2(self.bi.CAM_PARAMS[aid]["vis_mask"], vis_dst)
        os.makedirs(os.path.join(live_dir, "staging"))

    @staticmethod
    def heal_root(live_dir):
        return os.path.join(live_dir, "heal")

    def _cav_dir(self, aid):
        return os.path.join(self.heal_root(self.live_dir), SCENARIO, aid)

    def undistort(self, aid, bgr):
        """== build_inference.undistort (cv2.fisheye.undistortImage, Knew=K)."""
        h, w = bgr.shape[:2]
        key = (aid, w, h)
        if key not in self.maps:
            p = self.bi.CAM_PARAMS[aid]
            self.maps[key] = cv2.fisheye.initUndistortRectifyMap(
                p["K"], p["D"], np.eye(3), p["K"], (w, h), cv2.CV_16SC2)
        m1, m2 = self.maps[key]
        return cv2.remap(bgr, m1, m2, interpolation=cv2.INTER_LINEAR,
                         borderMode=cv2.BORDER_CONSTANT)

    def write_pair(self, frames):
        """frames: {agent_id: {'front_bgr': ndarray, 'ego_pose': dict}}."""
        staging = os.path.join(self.live_dir, "staging")
        for aid in self.agent_ids:
            f = frames[aid]
            pose = f["ego_pose"]
            me = {"x": pose["x"], "y": pose["y"], "z": pose["z"], "yaw_rad": pose["yaw"] + YAW_OFFSET_RAD}
            self.bi.build_agent_yaml(me, aid, staging, FRAME)
            replace_retry(os.path.join(staging, FRAME + ".yaml"),
                          os.path.join(self._cav_dir(aid), FRAME + ".yaml"))
            tmp_png = os.path.join(staging, FRAME + "_camera0.png")
            # PNG level 1: lossless like the offline pipeline, fast on tmpfs.
            f["front_undist"] = self.undistort(aid, f["front_bgr"])  # what HEAL sees; the viewer reuses it
            if not cv2.imwrite(tmp_png, f["front_undist"], [cv2.IMWRITE_PNG_COMPRESSION, 1]):
                raise IOError("could not write %s" % tmp_png)
            replace_retry(tmp_png, os.path.join(self._cav_dir(aid), FRAME + "_camera0.png"))


class LiveModel:
    """Model + dataset built exactly like qcar/eval.py (resolved_hypes.json
    replayed, plugins imported, best checkpoint loaded), pointed at the live
    scenario."""

    def __init__(self, model_dir, live_dir, qcar_conf, opt, amp):
        from opencood.data_utils.datasets import build_dataset
        from opencood.tools import train_utils
        from qcar import config, registry
        import qcar.patches.patch_windows_paths  # noqa: F401 -- no-op off Windows

        self.train_utils = train_utils
        hypes = json.load(open(os.path.join(model_dir, "resolved_hypes.json")))
        install_extrinsic()
        config.resolve_modules(opt, hypes, qcar_conf)
        config.apply_overrides(hypes, qcar_conf["overrides"], opt.set)

        # The live scenario is not in the model's assignment json; give it
        # the modality the model was trained with for each agent (taken from
        # the model's own assignment file, so nothing is invented here).
        heter = hypes.get("heter") or {}
        if heter.get("assignment_path"):
            original = json.load(open(heter["assignment_path"]))
            first = original[sorted(original)[0]]
            live_assign = os.path.join(live_dir, "live_modality_assign.json")
            json.dump({SCENARIO: first}, open(live_assign, "w"), indent=2)
            heter["assignment_path"] = live_assign
        hypes["validate_dir"] = LiveHealInput.heal_root(live_dir)
        self.hypes = hypes

        self.ds = build_dataset(hypes, visualize=False, train=False)
        if len(self.ds) != 1:
            raise RuntimeError("live dataset should hold exactly 1 frame, has %d" % len(self.ds))
        self.model = registry.create_model(hypes)
        # HEAL's load_saved_model silently keeps RANDOM weights when the dir
        # has no .pth (epoch 0) -- and checkpoints are gitignored, so a fresh
        # clone has resolved_hypes.json but no weights. Seen live:
        # hundreds of 0.51-score boxes on a grid. Fail loudly instead.
        epoch, self.model = train_utils.load_saved_model(model_dir, self.model)
        if not epoch:
            raise SystemExit(
                "No checkpoint (.pth) in %s -- the model would run with random weights.\n"
                "Download the zoo:  hf download Saavkd7/qcar-heal-zoo --local-dir checkpoints/qcar/zoo"
                % model_dir)
        print("[realtime] weights loaded: epoch %s from %s" % (epoch, model_dir))
        self.verify_checkpoint(model_dir, epoch, hypes)
        # CUDA only: HEAL's camera encoder (heter_encoders.py) calls .cuda()
        # in its constructor, so a camera model cannot be built on CPU.
        self.device = torch.device("cuda")
        self.model.to(self.device).eval()
        self.amp = bool(amp)

    def verify_checkpoint(self, model_dir, epoch, hypes):
        """HEAL loads with strict=False: prove here that the file is the expected one and that every
        tensor of the model equals the file's tensor."""
        path = os.path.join(model_dir, "net_epoch%d.pth" % epoch)
        digest = hashlib.sha256()
        with open(path, "rb") as f:
            for block in iter(lambda: f.read(1 << 22), b""):
                digest.update(block)
        expected = hypes.get("_live_expected_checkpoint_sha256")
        if expected and digest.hexdigest() != expected:
            raise SystemExit("checkpoint %s sha256 %s != expected %s" % (path, digest.hexdigest(), expected))
        state = torch.load(path, map_location="cpu")
        own = self.model.state_dict()
        missing = [k for k in own if k not in state]
        unexpected = [k for k in state if k not in own]
        unequal = [k for k in own if k in state and not torch.equal(own[k].cpu(), state[k])]
        print("[live] checkpoint %s sha256 %s | tensors in file %d, in model %d, missing %d, "
              "unexpected %d, unequal %d" % (path, digest.hexdigest(), len(state), len(own), len(missing),
                                             len(unexpected), len(unequal)))
        if missing or unexpected or unequal:
            raise SystemExit("checkpoint not loaded completely: missing %s unexpected %s unequal %s"
                             % (missing[:3], unexpected[:3], unequal[:3]))

    @torch.inference_mode()
    def infer(self):
        """Read the live pair through HEAL's dataset, run the model, decode.
        Returns (pred_corners [N,8,3] np, scores [N] np, timings dict)."""
        t0 = time.perf_counter()
        batch = self.ds.collate_batch_test([self.ds[0]])
        if batch is None:
            return np.zeros((0, 8, 3)), np.zeros(0), {"preprocess_s": time.perf_counter() - t0}
        batch = self.train_utils.to_device(batch, self.device)
        t1 = time.perf_counter()
        with torch.cuda.amp.autocast(enabled=self.amp):
            out = self.model(batch["ego"])
        out = {k: (v.float() if torch.is_tensor(v) and v.is_floating_point() else v)
               for k, v in out.items()}
        if self.device.type == "cuda":
            torch.cuda.synchronize()
        t2 = time.perf_counter()
        boxes, scores, _ = self.ds.post_process(batch, {"ego": out})
        t3 = time.perf_counter()
        boxes = np.zeros((0, 8, 3)) if boxes is None else boxes.cpu().numpy()
        scores = np.zeros(0) if scores is None else scores.cpu().numpy()
        return boxes, scores, {"preprocess_s": t1 - t0, "model_s": t2 - t1,
                               "postprocess_s": t3 - t2,
                               "n_agents_fused": int(batch["ego"]["record_len"].sum())}
