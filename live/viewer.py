"""Live viewer: a copy of qcar/realtime/viewer.py of the HEAL fork plus

  * the collision ALARM: the tail of the session's online/online_log.jsonl is read without blocking; the latest line whose `pair` is not newer
    than the pair being drawn decides. `warning` true -> red banner "COLLISION WARNING - contact in X.XX s" (X = smallest nominal_contact_s of the
    warning tracks) and a red border on every panel. The banner is held on screen for at least HOLD_S = 1.0 s after the last warning frame was
    drawn. THE HOLD IS A DISPLAY EFFECT ONLY: it is wall-clock time of this viewer and changes no record, row, threshold or log line.
  * the TRACKS of that log line in the bird's-eye view (position + velocity arrow x 1 s), red when the track warns;
  * --composite: one window (ego POV and cooperator POV on top, bird's-eye view below, banner across); f / F11 toggles fullscreen, q closes;
    without it the three original windows (q / Esc closes);
  * a status line: mode, fps, capture-to-output latency of the last pair, peer used/withheld, alarm lag;
  * --snapshot_dir: the composited image every N draws as PNG, plus alarm_trace.jsonl (one line per draw: pair, log pair used, banner state);
  * offscreen=True: no Tk at all, the same drawing code writes the snapshots (tests over ssh, where no desktop exists).

The viewer only draws. It reads the online log (read-only) and never writes anything but its snapshots; it changes no record of the loop.
The online module processes a pair AFTER the loop wrote it, so when pair k is drawn the log normally reaches pair k-1 (or earlier): the banner
follows the module by that lag (shown in the status line). Between drawn pairs nothing is redrawn; if no new pair arrives for STALE_S the panels are
marked "NO NEW DATA" and the banner is dropped instead of freezing a warning on an old image.

Box projection, geometry and the detections are as in the fork's viewer, with one change: the Vicon yaw offset the model was given
(heal_input.YAW_OFFSET_RAD, added to every agent yaw) is also used here when moving boxes ego-model -> world -> peer-model, so the detections
land where box_record's center_world_m and the online module's tracks put them. The cars' own drawn heading is unchanged (raw yaw).
"""
import collections
import json
import math
import os
import time

import cv2
import numpy as np

EGO_COLOR = (80, 200, 60)      # BGR green
PEER_COLOR = (230, 140, 40)    # BGR blue-ish
DET_COLOR = (255, 0, 255)      # magenta, same as qcar/visualize.py predictions
TRACK_COLOR = (0, 140, 255)    # BGR orange: a track that does not warn
ALARM_COLOR = (0, 0, 220)      # BGR red
QCAR_LENGTH, QCAR_WIDTH = 0.42, 0.19   # metres, for drawing only
HOLD_S = 1.0                   # minimum time the banner stays after the last warning frame (display only)
STALE_S = 1.5                  # no new pair for this long -> panels marked, banner dropped
ARROW_S = 1.0                  # a track's velocity arrow is v * ARROW_S metres long


def _rot(yaw):
    c, s = math.cos(yaw), math.sin(yaw)
    return np.array([[c, -s], [s, c]])


def ego_model_to_world_m(xy_model, ego_pose, yaw_offset=0.0):
    """(N,2) ego-frame x10 model units -> (N,2) Vicon world metres."""
    return (xy_model / 10.0) @ _rot(ego_pose["yaw"] + yaw_offset).T + [ego_pose["x"], ego_pose["y"]]


def world_m_to_cav_model(xy_world, cav_pose, yaw_offset=0.0):
    return ((xy_world - [cav_pose["x"], cav_pose["y"]]) @ _rot(cav_pose["yaw"] + yaw_offset)) * 10.0


# ------------------------------------------------------------------ the alarm
class OnlineLogTail:
    """Non-blocking incremental reader of online_log.jsonl: new complete lines only, a partial last line waits for its newline. Keeps the last
    `keep` lines that carry a `warning` key (skipped / error lines carry none and are ignored). Never opens the file for writing."""

    def __init__(self, path, keep=600, min_t_now=None):
        self.path, self.keep, self.min_t_now = path, keep, min_t_now
        self.pos, self.partial = 0, b""
        self.entries = collections.deque(maxlen=keep)
        self.lines_read = 0

    def poll(self):
        try:
            size = os.path.getsize(self.path)
        except OSError:                      # the module creates the file when it starts
            return 0
        if size < self.pos:                  # a new file under the same name: start over
            self.pos, self.partial = 0, b""
            self.entries.clear()
        if size == self.pos:
            return 0
        try:
            with open(self.path, "rb") as f:
                f.seek(self.pos)
                chunk = f.read(size - self.pos)
        except OSError:
            return 0
        self.pos += len(chunk)
        data = self.partial + chunk
        *lines, self.partial = data.split(b"\n")
        n = 0
        for raw in lines:
            if not raw.strip():
                continue
            try:
                e = json.loads(raw)
            except ValueError:
                continue
            self.lines_read += 1
            if "warning" not in e or not isinstance(e.get("pair"), int):
                continue
            if self.min_t_now is not None and e.get("t_now") is not None and e["t_now"] < self.min_t_now:
                continue                     # a line of an older session
            self.entries.append(e)
            n += 1
        return n

    def latest_upto(self, pair):
        """The newest line whose pair is not newer than `pair` (lines arrive in pair order)."""
        for e in reversed(self.entries):
            if e["pair"] <= pair:
                return e
        return None


def warning_contact_s(entry):
    """X of the banner: the smallest nominal_contact_s among the warning tracks of the line."""
    c = [t["nominal_contact_s"] for t in entry.get("tracks", []) if t.get("warning") and t.get("nominal_contact_s") is not None]
    if not c:
        c = [entry["min_nominal_contact_s"]] if entry.get("min_nominal_contact_s") is not None else []
    return min(c) if c else None


class Alarm:
    """Banner state for the pair being drawn: `at(pair, now)` -> dict. `now` is the wall time of the draw (a parameter, so the logic is testable)."""

    def __init__(self, tail, hold_s=HOLD_S):
        self.tail, self.hold_s = tail, hold_s
        self.last_warn_wall, self.last_contact = None, None

    def at(self, pair, now, update_hold=True):
        self.tail.poll()
        e = self.tail.latest_upto(pair)
        warn = bool(e and e["warning"])
        contact = warning_contact_s(e) if warn else None
        if warn and update_hold:
            self.last_warn_wall, self.last_contact = now, contact
        held = (not warn) and self.last_warn_wall is not None and (now - self.last_warn_wall) < self.hold_s
        return {"on": warn or held, "warning": warn, "held": held, "contact_s": contact if warn else (self.last_contact if held else None),
                "log_pair": e["pair"] if e else None, "lag_pairs": (pair - e["pair"]) if e else None,
                "tracks": e["tracks"] if e else [], "pair": pair}


# ------------------------------------------------------------------- viewer
class LiveViewer:
    def __init__(self, agents, ego_id, car_labels, cam_K, walls, bev_range, max_hz, online_log=None, mode="coop", composite=False,
                 fullscreen=False, snapshot_dir=None, snapshot_every=10, offscreen=False, size=(1920, 1080), yaw_offset_rad=0.0, hold_s=HOLD_S, source="live"):
        from qcar.patches.patch_real_extrinsic import CAMERA_PARAMS
        from qcar.visualize import draw_boxes

        self.draw_boxes, self.cam_extr = draw_boxes, CAMERA_PARAMS
        self.agents, self.ego_id, self.labels, self.K = agents, ego_id, car_labels, cam_K
        self.walls, self.range = walls, bev_range
        self.period = 1.0 / float(max_hz)
        self.last_draw = 0.0
        self.last_pump = 0.0
        self.quit = False
        self.quit_reason = None
        self.scale = 100.0  # px per metre
        self.mode, self.composite, self.offscreen, self.source = mode, composite, offscreen, source
        self.yaw_offset = yaw_offset_rad
        self.size = size
        self.fs = 1.0
        self.fullscreen = fullscreen
        self.tail = OnlineLogTail(online_log, min_t_now=time.time() - 5.0) if online_log else None
        self.alarm = Alarm(self.tail, hold_s) if self.tail else None
        self.snapshot_dir, self.snapshot_every = snapshot_dir, max(1, int(snapshot_every))
        self.n_draws = 0
        self.draw_times = collections.deque(maxlen=20)
        self.last_args, self.stale, self.last_stale_draw = None, False, 0.0
        self.trace = None
        if snapshot_dir:
            os.makedirs(snapshot_dir, exist_ok=True)
            self.trace = open(os.path.join(snapshot_dir, "alarm_trace.jsonl"), "w", buffering=1)

        self.labels_tk, self.photos = {}, {}
        self.root = None
        if offscreen:
            self.waiting("waiting for the first pair")
            return
        import tkinter as tk
        from PIL import Image, ImageTk
        self.Image, self.ImageTk = Image, ImageTk
        self.tk = tk
        self.root = tk.Tk()
        self.root.withdraw()  # only the Toplevels are shown
        if composite:
            titles = {"main": "Cooperative perception + collision warning (%s, %s)  -  f: fullscreen, q: close" % (mode, source)}
        else:
            titles = dict((a, "POV CAV %s - %s%s" % (a, car_labels[a], " (ego)" if a == ego_id else "")) for a in agents)
            titles["bev"] = "Bird's-eye view (Vicon, metres)"
        self.wins = {}
        for key, title in titles.items():
            win = tk.Toplevel(self.root)
            win.title(title)
            win.protocol("WM_DELETE_WINDOW", self._on_close)
            win.bind("<Key-q>", lambda _e: self._on_close())
            if composite:
                win.bind("<Key-f>", lambda _e: self._toggle_fullscreen())
                win.bind("<F11>", lambda _e: self._toggle_fullscreen())
                win.bind("<Escape>", lambda _e: self._set_fullscreen(False))   # Esc only leaves fullscreen: a stray key must not stop the viewer
            else:
                win.bind("<Escape>", lambda _e: self._on_close())
            label = tk.Label(win, bg="black")
            label.pack(fill="both", expand=True)
            self.labels_tk[key] = label
            self.wins[key] = win
        if composite:
            self._set_fullscreen(fullscreen)
        self.waiting("start the car launches now (waiting for the first pair)")

    # ----------------------------------------------------------- windowing
    def _set_fullscreen(self, on):
        self.fullscreen = bool(on)
        if self.root is not None and "main" in self.wins:
            self.wins["main"].attributes("-fullscreen", self.fullscreen)

    def _toggle_fullscreen(self):
        self._set_fullscreen(not self.fullscreen)

    def waiting(self, text):
        """Screen shown from READY (models loaded) until the first fused pair: a large READY, the signal to start the car launches, and `text` below."""
        keys = list(self.labels_tk) if not self.offscreen else (["main"] if self.composite else [])
        for key in keys:
            comp = key == "main"
            size = (self.size[1], self.size[0]) if comp else ((480, 640) if key != "bev" else (600, 700))
            img = np.full(size + (3,), (20, 70, 20), dtype=np.uint8)
            h, w = size
            self._fit_text(img, "READY", (0, int(h * 0.12), w, int(h * 0.62)), (255, 255, 255), 18 if comp else 5, 9.0 if comp else 3.0)
            self._fit_text(img, "start the car launches now", (0, int(h * 0.62), w, int(h * 0.78)), (200, 255, 200), 5 if comp else 2, 2.6 if comp else 1.0)
            self._fit_text(img, text, (0, int(h * 0.80), w, int(h * 0.92)), (200, 200, 200), 2 if comp else 1, 1.2 if comp else 0.5)
            if self.offscreen:
                if self.snapshot_dir and not os.path.exists(os.path.join(self.snapshot_dir, "ready.png")):
                    cv2.imwrite(os.path.join(self.snapshot_dir, "ready.png"), img)
            else:
                self._show(key, img)
        if not self.offscreen:
            self._tk_update()

    def _tk_update(self):
        try:
            self.root.update()
        except Exception as exc:  # noqa: BLE001 -- window torn down
            self._set_quit("Tk update failed: %r" % (exc,))

    def _set_quit(self, reason):
        if not self.quit:
            self.quit, self.quit_reason = True, reason
            print("[viewer] stop requested: %s" % reason, flush=True)

    def _on_close(self):
        self._set_quit("window closed or q pressed")

    def pump(self):
        """Keep the windows responsive while waiting for pairs; mark the panels stale when no pair has been drawn for STALE_S."""
        now = time.time()
        if now - self.last_pump >= 0.03:
            if self.root is not None:
                self._tk_update()
            self.last_pump = now
        if self.last_args is not None and now - self.last_draw >= STALE_S and now - self.last_stale_draw >= 0.5:
            self.stale = True
            self.last_stale_draw = now
            self._draw(*self.last_args, stale=True)

    def due(self):
        return time.time() - self.last_draw >= self.period

    # --------------------------------------------------------------- drawing
    def _show(self, key, bgr):
        if self.offscreen:
            return
        if key == "main" and self.fullscreen:
            sw, sh = self.wins["main"].winfo_screenwidth(), self.wins["main"].winfo_screenheight()
            h, w = bgr.shape[:2]
            k = min(sw / w, sh / h)
            if abs(k - 1.0) > 1e-3:
                bgr = cv2.resize(bgr, (int(w * k), int(h * k)), interpolation=cv2.INTER_AREA if k < 1 else cv2.INTER_LINEAR)
        photo = self.ImageTk.PhotoImage(self.Image.fromarray(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)))
        self.labels_tk[key].configure(image=photo)
        self.photos[key] = photo  # Tk keeps no reference of its own

    def _pov(self, aid, frame, boxes_ego, frames):
        img = frame["front_undist"].copy()
        n_drawn = 0
        if len(boxes_ego):
            boxes = boxes_ego.copy()
            if aid != self.ego_id:
                world = ego_model_to_world_m(boxes[:, :, :2].reshape(-1, 2), frames[self.ego_id]["ego_pose"], self.yaw_offset)
                boxes[:, :, :2] = world_m_to_cav_model(world, frame["ego_pose"], self.yaw_offset).reshape(-1, 8, 2)
            R_cb, t_cb = self.cam_extr[aid]
            for b in boxes:
                ctx = {"im": img, "K": self.K[aid], "R_cb": R_cb, "t_cb": t_cb}
                n_drawn += bool(self.draw_boxes(ctx, b[None], DET_COLOR))
        color = EGO_COLOR if aid == self.ego_id else PEER_COLOR
        fs = self.fs
        hh = int(26 * fs)
        cv2.rectangle(img, (0, 0), (img.shape[1], hh), (0, 0, 0), -1)
        cv2.putText(img, "CAV %s %s | detections visible here: %d"
                    % (aid, "(ego)" if aid == self.ego_id else "(peer)", n_drawn),
                    (8, int(18 * fs)), cv2.FONT_HERSHEY_SIMPLEX, 0.5 * fs, color, 1 + (fs > 1.2), cv2.LINE_AA)
        return img

    def _px(self, xy):
        xmin, xmax, ymin, ymax = self.range
        return (int(round((xy[0] - xmin) * self.scale)), int(round((ymax - xy[1]) * self.scale)))

    def _bev(self, frames, boxes_ego, scores, info, al):
        xmin, xmax, ymin, ymax = self.range
        fs = self.fs
        w, h = int((xmax - xmin) * self.scale), int((ymax - ymin) * self.scale)
        img = np.full((h, w, 3), 255, dtype=np.uint8)
        for gx in np.arange(math.ceil(xmin * 2) / 2, xmax, 0.5):
            cv2.line(img, self._px((gx, ymin)), self._px((gx, ymax)), (235, 235, 235), 1)
        for gy in np.arange(math.ceil(ymin * 2) / 2, ymax, 0.5):
            cv2.line(img, self._px((xmin, gy)), self._px((xmax, gy)), (235, 235, 235), 1)
        cv2.line(img, self._px((0, ymin)), self._px((0, ymax)), (200, 200, 200), 1)
        cv2.line(img, self._px((xmin, 0)), self._px((xmax, 0)), (200, 200, 200), 1)
        for wall in self.walls:
            cv2.line(img, self._px(wall["a"]), self._px(wall["b"]), (60, 60, 60), 4)

        for aid in self.agents:
            pose = frames[aid]["ego_pose"]
            color = EGO_COLOR if aid == self.ego_id else PEER_COLOR
            half = np.array([[QCAR_LENGTH, QCAR_WIDTH], [QCAR_LENGTH, -QCAR_WIDTH],
                             [-QCAR_LENGTH, -QCAR_WIDTH], [-QCAR_LENGTH, QCAR_WIDTH]]) / 2
            poly = half @ _rot(pose["yaw"]).T + [pose["x"], pose["y"]]
            cv2.fillPoly(img, [np.array([self._px(p) for p in poly], dtype=np.int32)], color)
            nose = np.array([pose["x"], pose["y"]]) + 0.35 * np.array(
                [math.cos(pose["yaw"]), math.sin(pose["yaw"])])
            cv2.arrowedLine(img, self._px((pose["x"], pose["y"])), self._px(nose), color, 2,
                            tipLength=0.3)
            cv2.putText(img, "CAV %s%s" % (aid, " ego" if aid == self.ego_id else ""),
                        self._px((pose["x"] + 0.12, pose["y"] + 0.18)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5 * fs, color, 1 + (fs > 1.2), cv2.LINE_AA)

        ego_pose = frames[self.ego_id]["ego_pose"]
        for b, s in zip(boxes_ego, scores):
            world = ego_model_to_world_m(b[:4, :2], ego_pose, self.yaw_offset)
            pts = np.array([self._px(p) for p in world], dtype=np.int32)
            cv2.polylines(img, [pts], True, DET_COLOR, 2)
            cv2.putText(img, "%.2f" % s, tuple(pts.max(axis=0) + [3, 0]),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.45 * fs, DET_COLOR, 1, cv2.LINE_AA)

        # tracks of the online module (world metres, from the log line the alarm uses)
        for t in (al["tracks"] if al else []):
            if t.get("x") is None:
                continue
            col = ALARM_COLOR if t.get("warning") else TRACK_COLOR
            c = self._px((t["x"], t["y"]))
            cv2.circle(img, c, int(0.17 * self.scale), col, 3 if t.get("warning") else 2, cv2.LINE_AA)
            tip = self._px((t["x"] + t["vx"] * ARROW_S, t["y"] + t["vy"] * ARROW_S))
            cv2.arrowedLine(img, c, tip, col, 3, cv2.LINE_AA, tipLength=0.25)
            label = "%s" % t["id"]
            if t.get("warning") and t.get("nominal_contact_s") is not None:
                label += "  contact %.2f s" % t["nominal_contact_s"]
            cv2.putText(img, label, (c[0] + 14, c[1] - 14), cv2.FONT_HERSHEY_SIMPLEX, 0.5 * fs, col, 1 + (fs > 1.2), cv2.LINE_AA)
        cv2.putText(img, "tracks: arrow = velocity x %.0f s (online module, pair %s)" % (ARROW_S, al["log_pair"] if al and al["log_pair"] is not None else "-"),
                    (10, h - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.45 * fs, (90, 90, 90), 1, cv2.LINE_AA)

        if not self.composite:
            lines = ["pair %d | agents fused: %s | detections: %d" % (info["pair"], info.get("n_agents_fused", "-"), len(scores)),
                     self._status(info)]
            for i, text in enumerate(l for l in lines if l):
                cv2.putText(img, text, (10, 22 + 20 * i), cv2.FONT_HERSHEY_SIMPLEX, 0.55,
                            (0, 0, 0), 1, cv2.LINE_AA)
        return img

    def _status(self, info):
        fps = ""
        if len(self.draw_times) >= 2 and self.draw_times[-1] > self.draw_times[0]:
            fps = "%.1f" % ((len(self.draw_times) - 1) / (self.draw_times[-1] - self.draw_times[0]))
        lat = info.get("latency_s")
        peer = info.get("peer_status", "-")
        return "%s | mode %s | fps %s | capture->output %s | peer %s" % (
            self.source, info.get("mode", self.mode), fps or "-", ("%.0f ms" % (lat * 1000)) if lat is not None else "-",
            {"used": "used", "withheld": "WITHHELD", "not_used": "not used (ego only)", "delayed": "delayed"}.get(peer, peer))

    # --------------------------------------------------------- alarm overlays
    @staticmethod
    def _fit_text(img, text, rect, color, thick, max_scale):
        """Centered text, as large as fits in rect (x0, y0, x1, y1)."""
        x0, y0, x1, y1 = rect
        scale = max_scale
        while scale > 0.3:
            (tw, th), base = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, scale, thick)
            if tw <= (x1 - x0) - 20 and th + base <= (y1 - y0) - 6:
                break
            scale -= 0.1
        (tw, th), base = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, scale, thick)
        org = (x0 + ((x1 - x0) - tw) // 2, y0 + ((y1 - y0) + th) // 2)
        cv2.putText(img, text, org, cv2.FONT_HERSHEY_SIMPLEX, scale, color, thick, cv2.LINE_AA)

    @staticmethod
    def banner_text(al):
        c = al["contact_s"]
        return "COLLISION WARNING - contact in %s s" % ("%.2f" % c if c is not None else "--.--")

    def _panel_alarm(self, img, al, stale):
        """Three-window mode: banner across the top of the panel and a red border."""
        h, w = img.shape[:2]
        if stale:
            cv2.rectangle(img, (0, 0), (w, 44), (60, 60, 60), -1)
            self._fit_text(img, "NO NEW DATA", (0, 0, w, 44), (200, 200, 200), 2, 1.0)
            return img
        if not (al and al["on"]):
            return img
        cv2.rectangle(img, (0, 0), (w, 44), ALARM_COLOR, -1)
        self._fit_text(img, self.banner_text(al), (0, 0, w, 44), (255, 255, 255), 2, 1.0)
        cv2.rectangle(img, (0, 0), (w - 1, h - 1), ALARM_COLOR, 8)
        return img

    # -------------------------------------------------------------- composite
    @staticmethod
    def _paste(canvas, img, cell):
        x0, y0, x1, y1 = cell
        h, w = img.shape[:2]
        k = min((x1 - x0) / w, (y1 - y0) / h)
        nw, nh = int(w * k), int(h * k)
        im = cv2.resize(img, (nw, nh), interpolation=cv2.INTER_AREA if k < 1 else cv2.INTER_LINEAR)
        ox, oy = x0 + ((x1 - x0) - nw) // 2, y0 + ((y1 - y0) - nh) // 2
        canvas[oy:oy + nh, ox:ox + nw] = im
        return (ox, oy, ox + nw, oy + nh)

    def _compose(self, povs, bev, al, info, stale):
        W, H = self.size
        canvas = np.full((H, W, 3), 28, np.uint8)
        banner_h, status_h = 100, 44
        body = H - banner_h - status_h
        top_h = int(body * 0.47)
        on = bool(al and al["on"]) and not stale
        # banner (always reserved, so the layout never jumps)
        if stale:
            canvas[:banner_h] = (60, 60, 60)
            self._fit_text(canvas, "NO NEW DATA (no pair for %.1f s) - session still running" % (time.time() - self.last_draw), (0, 0, W, banner_h), (200, 200, 200), 3, 2.0)
        elif on:
            canvas[:banner_h] = ALARM_COLOR
            self._fit_text(canvas, self.banner_text(al), (0, 0, W, banner_h), (255, 255, 255), 5, 2.4)
        else:
            canvas[:banner_h] = (50, 50, 50)
            self._fit_text(canvas, "no collision warning", (0, 0, W, banner_h), (120, 200, 120), 3, 1.6)
        # panels
        rects = []
        ids = list(self.agents)
        pw = int((top_h - 12) * povs[ids[0]].shape[1] / povs[ids[0]].shape[0])     # the two POVs side by side, centred
        for i, aid in enumerate(ids):
            x0 = W // 2 - pw - 8 if i == 0 else W // 2 + 8
            rects.append(self._paste(canvas, povs[aid], (x0, banner_h + 8, x0 + pw, banner_h + top_h - 4)))
        bev_cell = (10, banner_h + top_h + 4, int(W * 0.40), banner_h + body - 6)
        rects.append(self._paste(canvas, bev, bev_cell))
        # info panel (bottom right): the tracks of the log line in use
        x0, y0 = int(W * 0.40) + 30, banner_h + top_h + 50
        lines = [("Online collision module", (220, 220, 220), 0.9)]
        if al is None:
            lines.append(("alarm log not connected", (120, 120, 255), 0.9))
        else:
            lines.append(("log pair %s  (drawing pair %d, lag %s)" % (al["log_pair"], al["pair"], al["lag_pairs"]), (180, 180, 180), 0.8))
            for t in al["tracks"]:
                if t.get("x") is None:
                    continue
                col = ALARM_COLOR if t.get("warning") else TRACK_COLOR
                col = (80, 80, 255) if t.get("warning") else (60, 170, 255)
                cs = t.get("nominal_contact_s")
                lines.append(("%s  %s  contact in %s s  v=(%.2f, %.2f) m/s" % (
                    t["id"], "WARNING" if t.get("warning") else "tracked", ("%.2f" % cs) if cs is not None else "-", t["vx"], t["vy"]), col, 0.9))
            if not al["tracks"]:
                lines.append(("no active track", (150, 150, 150), 0.9))
        lines.append(("detections: %d   fused agents: %s" % (info.get("n_det", 0), info.get("n_agents_fused", "-")), (180, 180, 180), 0.8))
        for i, (text, col, sc) in enumerate(lines):
            cv2.putText(canvas, text, (x0, y0 + i * 52), cv2.FONT_HERSHEY_SIMPLEX, sc * 1.15, col, 2, cv2.LINE_AA)
        # status line
        cv2.rectangle(canvas, (0, H - status_h), (W, H), (0, 0, 0), -1)
        cv2.putText(canvas, self._status(info) + (" | alarm lag %s pair(s)" % al["lag_pairs"] if al and al["lag_pairs"] is not None else ""),
                    (14, H - 13), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2, cv2.LINE_AA)
        if on:
            for r in rects:
                cv2.rectangle(canvas, (r[0] - 3, r[1] - 3), (r[2] + 2, r[3] + 2), ALARM_COLOR, 6)
            cv2.rectangle(canvas, (0, 0), (W - 1, H - 1), ALARM_COLOR, 12)
        return canvas

    # ----------------------------------------------------------------- update
    def update(self, frames, boxes_ego, scores, info):
        """frames: {agent: frame with 'front_undist' and 'ego_pose'}; boxes_ego: (N,8,3) HEAL output (ego frame, x10); scores: (N,);
        info: pair, n_agents_fused, mode, latency_s (capture->output of this pair), peer_status."""
        self.last_args = (frames, boxes_ego, scores, info)
        self.stale = False
        self._draw(frames, boxes_ego, scores, info, stale=False)

    def _draw(self, frames, boxes_ego, scores, info, stale):
        t0 = time.time()
        al = self.alarm.at(info["pair"], t0, update_hold=not stale) if self.alarm else None
        info = dict(info, n_det=len(scores))
        if not stale:
            self.draw_times.append(t0)
        want_snapshot = (self.snapshot_dir is not None and not stale and self.n_draws % self.snapshot_every == 0)
        self.fs = 1.3 if (self.composite or want_snapshot) else 1.0
        povs = dict((aid, self._pov(aid, frames[aid], boxes_ego, frames)) for aid in self.agents)
        bev = self._bev(frames, boxes_ego, scores, info, al)
        snap_name = None
        if self.composite or want_snapshot:
            canvas = self._compose(povs, bev, al, info, stale)
            if self.composite:
                self._show("main", canvas)
            if want_snapshot:
                snap_name = "snap_%06d_pair%06d.png" % (self.n_draws, info["pair"])
                cv2.imwrite(os.path.join(self.snapshot_dir, snap_name), canvas, [cv2.IMWRITE_PNG_COMPRESSION, 1])
        if not self.composite:
            for aid in self.agents:
                self._show(aid, self._panel_alarm(povs[aid], al, stale))
            self._show("bev", self._panel_alarm(bev, al, stale))
        if self.root is not None:
            self._tk_update()
        if not stale:
            self.last_draw = time.time()
            if self.trace is not None:
                self.trace.write(json.dumps({
                    "draw": self.n_draws, "pair": info["pair"], "t_wall": t0, "log_pair": al["log_pair"] if al else None,
                    "lag_pairs": al["lag_pairs"] if al else None, "warning": al["warning"] if al else None, "held": al["held"] if al else None,
                    "banner_on": al["on"] if al else False, "contact_s": al["contact_s"] if al else None, "snapshot": snap_name,
                    "draw_ms": (self.last_draw - t0) * 1000.0}) + "\n")
            self.n_draws += 1
