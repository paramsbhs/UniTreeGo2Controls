import numpy as np
from scipy.spatial.transform import Rotation

from go2_wbc.utils.transforms import (
    quat_to_rotmat,
    quat_to_rpy,
    rpy_to_quat,
    tilt_angle,
    wxyz_to_xyzw,
    xyzw_to_wxyz,
)

rng = np.random.default_rng(0)


def random_quats(n=50):
    q = rng.normal(size=(n, 4))
    return q / np.linalg.norm(q, axis=1, keepdims=True)


def test_quat_order_roundtrip():
    for q in random_quats():
        np.testing.assert_allclose(xyzw_to_wxyz(wxyz_to_xyzw(q)), q)


def test_rotmat_matches_scipy():
    for q in random_quats():
        ref = Rotation.from_quat(wxyz_to_xyzw(q)).as_matrix()  # scipy is scalar-last
        np.testing.assert_allclose(quat_to_rotmat(q), ref, atol=1e-12)


def test_rpy_roundtrip_away_from_gimbal_lock():
    for _ in range(50):
        rpy = rng.uniform([-3, -1.4, -3], [3, 1.4, 3])
        np.testing.assert_allclose(quat_to_rpy(rpy_to_quat(rpy)), rpy, atol=1e-10)


def test_tilt_ignores_yaw():
    assert tilt_angle(rpy_to_quat([0, 0, 2.0])) < 1e-12
    np.testing.assert_allclose(tilt_angle(rpy_to_quat([0.3, 0, 1.0])), 0.3, atol=1e-12)
