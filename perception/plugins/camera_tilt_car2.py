"""Camera extrinsic of car 2: the manufacturer's nominal rotation tilted 3.0 deg down.

Load after camera_extrinsic.py (registered as qcar.patches.patch_real_extrinsic), which already gives car 1 its fitted rotation.
"""
import numpy as np

import qcar.patches.patch_real_extrinsic as p

R_CAR2 = np.array([[0.0, -0.05233596, 0.99862953],
                   [-1.0, 0.0, 0.0],
                   [0.0, -0.99862953, -0.05233596]], dtype=np.float64)
assert np.allclose(R_CAR2 @ R_CAR2.T, np.eye(3), atol=1e-7) and abs(np.linalg.det(R_CAR2) - 1) < 1e-7
p.CAMERA_PARAMS["2"] = (R_CAR2, p.CAMERA_PARAMS["2"][1])
print("[camera_tilt_car2] car 1 R:", p.CAMERA_PARAMS["1"][0].round(4).tolist(),
      "| car 2 R:", p.CAMERA_PARAMS["2"][0].round(4).tolist())
