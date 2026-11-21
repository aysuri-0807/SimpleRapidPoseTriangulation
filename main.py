"""Run the small, explainable RapidPoseTriangulation geometry demo."""

import argparse
from pathlib import Path

import numpy as np

from dependencies import (
    make_demo_scene,
    pair_score,
    reconstruct_pose,
    save_visualization,
    score_candidates,
    select_matches,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed", type=int, default=7, help="Seed for simulated detections")
    parser.add_argument("--noise-px", type=float, default=0.6,
                        help="Standard deviation of 2D keypoint noise in pixels")
    parser.add_argument("--threshold-px", type=float, default=4.0,
                        help="Maximum mean reprojection error for a pair")
    parser.add_argument("--corruption-px", type=float, default=40.0,
                        help="Pixel shift for the final one-joint failure experiment")
    parser.add_argument("--output", type=Path, default=Path("outputs/reconstruction.png"),
                        help="Path of the resulting visualization")
    args = parser.parse_args()
    if args.corruption_px < 0:
        parser.error("--corruption-px must be nonnegative")
    if args.noise_px < 0:
        parser.error("--noise-px must be nonnegative")
    if args.threshold_px <= 0:
        parser.error("--threshold-px must be positive")
    return args


def main() -> None:
    args = parse_args()
    scene = make_demo_scene(seed=args.seed, noise_px=args.noise_px)

    # The 3D algorithm sees only these unordered 2D detections and calibration.
    scores = score_candidates(scene)
    print("Candidate score matrix (pixel error; rows=A, columns=B):")
    print(np.array2string(scores, precision=2, suppress_small=True))
    matches = select_matches(scores, threshold_px=args.threshold_px)
    print("Cross-camera matches (A index, B index):", matches)

    reconstructions = {}
    for i, j in matches:
        pose_3d = reconstruct_pose(scene.cameras["A"], scene.detections["A"][i],
                                   scene.cameras["B"], scene.detections["B"][j])
        reconstructions[i] = pose_3d
        # Synthetic ground truth is for reporting only. It is not used above.
        mean_error_cm = np.mean(np.linalg.norm(pose_3d - scene.truth[i], axis=1)) * 100
        print(f"A{i} -> B{j}: mean 3D joint error = {mean_error_cm:.2f} cm")

    save_visualization(scene, reconstructions, args.output)
    print(f"Saved visualization: {args.output}")

if __name__ == "__main__":
    main()
