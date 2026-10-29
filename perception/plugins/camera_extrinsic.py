"""Monkeypatch OPV2VBaseDataset.get_ext_int so HEAL's LSS geometry uses the real camera-to-vehicle rotation instead of HEAL's hard-coded
CARLA/UE4-to-OpenCV fix-up matrix, for any ego pose (the two-agent dataset has each agent at its own moving motion-capture pose).

Why: HEAL's stock get_ext_int() computes camera_to_lidar = x1_to_x2(camera_coords, lidar_pose_clean) @ FIX, with
FIX = [[0,0,1,0],[1,0,0,0],[0,-1,0,0],[0,0,0,1]]. On this testbed the camera cords' yaw always equals its ego's yaw and roll/pitch are 0, so
the rotation of x1_to_x2 is the identity and the result is FIX itself, a reflection (det = -1), not the camera-to-vehicle rotation (det = +1).
The translation of x1_to_x2 is R_ego_yaw^T @ (camera_world_pos - ego_world_pos), the camera mounting offset in the ego body frame; this patch
computes it directly and applies the real rotation R_cb instead of FIX.

CAMERA_PARAMS (agent id -> (R_cb, t_cb)): car 1 uses a rotation fitted to image / motion-capture correspondences; car 2 uses the
manufacturer's nominal rotation (camera_tilt_car2.py then tilts it by 3 degrees). Replace them if the cameras are re-calibrated.
"""
import numpy as np

from opencood.data_utils.datasets.basedataset.opv2v_basedataset import (
    OPV2VBaseDataset,
)
from opencood.utils.transformation_utils import x_to_world

R_CB_NOMINAL = np.array([[0, 0, 1], [-1, 0, 0], [0, -1, 0]], dtype=np.float64)

# car 1: rotation and translation fitted to image / motion-capture correspondences.
R_CB_CAR1 = np.array([
    [-0.00168635, -0.07682121, 0.99704346],
    [-0.99920993, -0.03946071, -0.00473042],
    [0.03970744, -0.9962637, -0.07669397],
], dtype=np.float64)
T_CB_CAR1 = np.array([0.21306336, -0.00169975, 0.11930449], dtype=np.float64)

CAMERA_PARAMS = {
    "1": (R_CB_CAR1, T_CB_CAR1),   # car 1, fitted
    "2": (R_CB_NOMINAL, np.array([0.1930, 0.0, 0.0953])),  # car 2, manufacturer nominal
}


def get_ext_int_qcar_real(self, params, camera_id):
    lidar_pose = params["lidar_pose_clean"]
    x, y, z, roll, yaw, pitch = lidar_pose
    if abs(roll) > 1e-6 or abs(pitch) > 1e-6:
        raise NotImplementedError(
            "patch_real_extrinsic assumes zero roll/pitch on lidar_pose "
            "(true for every qcar_real dataset built so far)."
        )
    camera_coords = np.array(
        params["camera%d" % camera_id]["cords"]).astype(np.float64)
    cam_yaw = camera_coords[4]
    if abs(cam_yaw - yaw) > 1e-3:
        raise NotImplementedError(
            "patch_real_extrinsic assumes camera cords' yaw always equals "
            "its own ego's yaw (true for every qcar_real build script so "
            "far) -- got camera yaw=%.3f vs ego yaw=%.3f" % (cam_yaw, yaw)
        )

    cav_id = params.get("qcar_provenance", {}).get("cav_id") or params.get("_qcar_cav_id")
    # a yaml without the cav_id field is from car 1: default to its geometry.
    R_cb, t_cb = CAMERA_PARAMS.get(str(cav_id), CAMERA_PARAMS["1"])

    ego_yaw_rad = np.radians(yaw)
    c, s = np.cos(ego_yaw_rad), np.sin(ego_yaw_rad)
    R_ego = np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]], dtype=np.float64)
    t_ego = np.array([x, y, z], dtype=np.float64)
    t_cam_world = camera_coords[:3]

    translation_in_ego_frame = R_ego.T @ (t_cam_world - t_ego)

    camera_to_lidar = np.eye(4, dtype=np.float32)
    camera_to_lidar[:3, :3] = R_cb
    camera_to_lidar[:3, 3] = translation_in_ego_frame
    camera_intrinsic = np.array(
        params["camera%d" % camera_id]["intrinsic"]).astype(np.float32)
    return camera_to_lidar, camera_intrinsic


OPV2VBaseDataset.get_ext_int = get_ext_int_qcar_real
print("[patch_real_extrinsic] OPV2VBaseDataset.get_ext_int -> real, "
      "data-refined camera-to-vehicle geometry (was HEAL's CARLA/UE4 mirror)")
