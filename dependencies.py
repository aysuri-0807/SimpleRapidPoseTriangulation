"""Geometry, synthetic inputs, matching, and plotting for a tiny RPT-style demo.

This module implements an educational subset of the paper's 2D-to-3D stage.
The 2D detections are simulated so the reconstruction logic is easy to inspect.
"""

from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from numpy.typing import NDArray


FloatArray = NDArray[np.float64]
JOINT_NAMES = ("left shoulder", "right shoulder", "hip", "left wrist", "right wrist")
CORE_JOINTS = (0, 1, 2)
SKELETON_EDGES = ((0, 1), (0, 2), (1, 2), (0, 3), (1, 4))


@dataclass(frozen=True)
class Camera:
    """Calibrated camera: X_camera = rotation @ (X_world - center)."""

    intrinsic: FloatArray  # 3x3 camera matrix K, in pixel units
    rotation: FloatArray  # 3x3 world-to-camera rotation R
    center: FloatArray  # camera position C in world coordinates, in metres


@dataclass(frozen=True)
class DemoScene:
    """2D inputs and 3D ground truth; truth is used only for demo evaluation."""

    cameras: dict[str, Camera]
    detections: dict[str, list[FloatArray]]  # [person][joint, (u, v)]
    truth: FloatArray  # [person, joint, (x, y, z)]


def project_points(camera: Camera, points_3d: FloatArray) -> FloatArray:
    """Project world-space 3D points into pixel (u, v) coordinates."""
    points = np.atleast_2d(np.asarray(points_3d, dtype=float))
    camera_xyz = (camera.rotation @ (points - camera.center).T).T
    if np.any(camera_xyz[:, 2] <= 0):
        raise ValueError("Points must be in front of the camera")
    normalized_xy = camera_xyz[:, :2] / camera_xyz[:, 2:3]
    return (camera.intrinsic[:2, :2] @ normalized_xy.T).T + camera.intrinsic[:2, 2]


def make_demo_scene(seed: int = 7, noise_px: float = 0.6) -> DemoScene:
    """Simulate 2D pose detections from two calibrated cameras and two people."""
    if noise_px < 0:
        raise ValueError("noise_px must be nonnegative")

    k = np.array([[900.0, 0.0, 640.0], [0.0, 900.0, 360.0], [0.0, 0.0, 1.0]])
    cameras = {
        "A": Camera(k, np.eye(3), np.array([-1.0, 0.0, 0.0])),
        "B": Camera(k, np.eye(3), np.array([1.0, 0.0, 0.0])),
    }
    # Five joints relative to the hip; both people have the same skeleton shape.
    offsets = np.array(
        [[-0.25, 0.55, 0], [0.25, 0.55, 0], [0, 0, 0],
         [-0.45, 0.15, 0], [0.45, 0.15, 0]]
    )
    truth = np.array([offsets + [-0.85, 0.35, 5.0],
                      offsets + [0.85, -0.35, 6.2]])
    rng = np.random.default_rng(seed)
    detections = {
        name: [project_points(camera, pose) + rng.normal(0, noise_px, (5, 2))
               for pose in truth]
        for name, camera in cameras.items()
    }
    # A camera's local detection number is not a global person identity.
    detections["B"].reverse()
    return DemoScene(cameras=cameras, detections=detections, truth=truth)


def pixel_ray(camera: Camera, pixel: FloatArray) -> tuple[FloatArray, FloatArray]:
    """Convert a pixel to a world-space ray (origin, unit direction)."""
    camera_direction = np.linalg.solve(camera.intrinsic, [pixel[0], pixel[1], 1.0])
    direction = camera.rotation.T @ camera_direction
    return camera.center, direction / np.linalg.norm(direction)


def triangulate_midpoint(
    camera_a: Camera, pixel_a: FloatArray, camera_b: Camera, pixel_b: FloatArray
) -> FloatArray:
    """Find the midpoint of the shortest segment between two viewing rays."""
    c1, d1 = pixel_ray(camera_a, pixel_a)
    c2, d2 = pixel_ray(camera_b, pixel_b)
    if np.linalg.norm(np.cross(d1, d2)) < 1e-8:
        raise ValueError("Nearly parallel rays cannot give a reliable 3D point")

    # Minimize ||(c1 + t*d1) - (c2 + s*d2)|| to find the closest ray points.
    t, s = np.linalg.lstsq(np.column_stack((d1, -d2)), c2 - c1, rcond=None)[0]
    return ((c1 + t * d1) + (c2 + s * d2)) / 2


def pair_score(
    camera_a: Camera, pose_a: FloatArray, camera_b: Camera, pose_b: FloatArray
) -> float:
    """Mean pixel reprojection error for three core joints in a proposed pair."""
    proposed_3d = np.array([
        triangulate_midpoint(camera_a, pose_a[j], camera_b, pose_b[j])
        for j in CORE_JOINTS
    ])
    error_a = np.linalg.norm(
        project_points(camera_a, proposed_3d) - pose_a[list(CORE_JOINTS)], axis=1
    )
    error_b = np.linalg.norm(
        project_points(camera_b, proposed_3d) - pose_b[list(CORE_JOINTS)], axis=1
    )
    return float(np.mean(np.r_[error_a, error_b]))


def score_candidates(scene: DemoScene) -> FloatArray:
    """Score every cross-camera person pairing; rows=A, columns=B."""
    return np.array([
        [pair_score(scene.cameras["A"], a, scene.cameras["B"], b)
         for b in scene.detections["B"]]
        for a in scene.detections["A"]
    ])


def select_matches(scores: FloatArray, threshold_px: float) -> list[tuple[int, int]]:
    """Keep the lowest-error pairs under a threshold, one per person in each view."""
    if threshold_px <= 0:
        raise ValueError("threshold_px must be positive")
    matches: list[tuple[int, int]] = []
    used_a: set[int] = set()
    used_b: set[int] = set()
    for i, j in sorted(np.ndindex(scores.shape), key=lambda ij: scores[ij]):
        if scores[i, j] < threshold_px and i not in used_a and j not in used_b:
            matches.append((i, j))
            used_a.add(i)
            used_b.add(j)
    return matches


def reconstruct_pose(
    camera_a: Camera, pose_a: FloatArray, camera_b: Camera, pose_b: FloatArray
) -> FloatArray:
    """Triangulate every joint, after selecting a plausible person pairing."""
    if pose_a.shape != pose_b.shape or pose_a.shape != (len(JOINT_NAMES), 2):
        raise ValueError("Both 2D poses must have the same known joint layout")
    return np.array([
        triangulate_midpoint(camera_a, pose_a[j], camera_b, pose_b[j])
        for j in range(len(JOINT_NAMES))
    ])


def save_visualization(
    scene: DemoScene, reconstructions: dict[int, FloatArray], output_path: Path
) -> None:
    """Save 2D detections and the resulting world-space 3D skeletons."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig = plt.figure(figsize=(14, 4.8), constrained_layout=True)
    colors = ("tab:blue", "tab:orange")

    for panel, name in enumerate(("A", "B"), 1):
        ax = fig.add_subplot(1, 3, panel)
        for detection_index, points in enumerate(scene.detections[name]):
            # Synthetic identity is used only to colour the display, never to match.
            person_id = detection_index if name == "A" else 1 - detection_index
            for left, right in SKELETON_EDGES:
                ax.plot(points[[left, right], 0], points[[left, right], 1],
                        color=colors[person_id])
            ax.scatter(points[:, 0], points[:, 1], color=colors[person_id], s=20)
            ax.annotate(f"detection {detection_index}", points[2] + [5, -5],
                        color=colors[person_id])
        ax.set(xlabel="pixel u", ylabel="pixel v", title=f"Camera {name}: 2D detections")
        ax.invert_yaxis()
        ax.set_aspect("equal", adjustable="datalim")

    ax3d = fig.add_subplot(1, 3, 3, projection="3d")
    ax3d.view_init(elev=20, azim=-65)
    for person_id, pose in reconstructions.items():
        for left, right in SKELETON_EDGES:
            ax3d.plot(pose[[left, right], 0], pose[[left, right], 2],
                      pose[[left, right], 1], color=colors[person_id])
        ax3d.scatter(pose[:, 0], pose[:, 2], pose[:, 1],
                     color=colors[person_id], s=22)
        ax3d.scatter(scene.truth[person_id, :, 0], scene.truth[person_id, :, 2],
                     scene.truth[person_id, :, 1], color="black", marker="x", s=22)
    ax3d.set(xlabel="world x (m)", ylabel="depth z (m)", zlabel="height y (m)",
             title="Recovered 3D (black x = truth)")
    fig.savefig(output_path, dpi=130, bbox_inches="tight", pad_inches=0.25)
    plt.close(fig)
