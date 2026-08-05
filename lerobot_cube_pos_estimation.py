#!/usr/bin/env python

# Estimate the real-world cube position from a recorded LeRobot episode by
# running forward kinematics at the moment the gripper actually closed
# around it. This avoids manually measuring/taping the cube's position and
# guessing the URDF's world-frame origin and axis conventions -- since the
# recorded joint trajectory and the Drake URDF already share the same
# coordinate frame, the fingers' midpoint at the grasp frame is the cube
# position in that frame.
#
# Usage:
#   python lerobot_cube_pos_estimation.py --data-root ./data/pick_place_3 --episode 0 --arm right

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from pydrake.all import AddMultibodyPlantSceneGraph, DiagramBuilder, Parser, RigidTransform

LEROBOT_JOINT_ORDER = [
    "left_joint_0", "left_joint_1", "left_joint_2", "left_joint_3",
    "left_joint_4", "left_joint_5", "left_joint_6",
    "right_joint_0", "right_joint_1", "right_joint_2", "right_joint_3",
    "right_joint_4", "right_joint_5", "right_joint_6",
]


def load_episode(data_root, episode_index, source_col="observation.state",
                  chunks_size=1000):
    data_root = Path(data_root)
    chunk = episode_index // chunks_size
    parquet_path = (
        data_root / "data" / f"chunk-{chunk:03d}"
        / f"episode_{episode_index:06d}.parquet"
    )
    df = pd.read_parquet(parquet_path)
    times = df["timestamp"].to_numpy().astype(float)
    times = times - times[0]
    q = np.stack(df[source_col].to_numpy()).astype(float)
    return times, q


def build_actuator_reorder(plant, lerobot_names=LEROBOT_JOINT_ORDER):
    actuator_indices = plant.GetJointActuatorIndices()
    drake_joint_names = [
        plant.get_joint_actuator(idx).joint().name() for idx in actuator_indices
    ]
    reorder = []
    for drake_name in drake_joint_names:
        # The gripper/carriage joint doesn't follow the "joint_N" naming
        # pattern (e.g. 'follower_left_left_carriage_joint'), so match it
        # explicitly to LeRobot's joint_6 (the gripper column) instead.
        if "carriage" in drake_name.lower():
            if drake_name.startswith("follower_left_"):
                match = lerobot_names.index("left_joint_6")
            elif drake_name.startswith("follower_right_"):
                match = lerobot_names.index("right_joint_6")
            else:
                match = None
        else:
            match = next(
                (i for i, lr in enumerate(lerobot_names) if lr in drake_name), None
            )
        if match is None:
            raise ValueError(f"No LeRobot match for Drake joint '{drake_name}'")
        reorder.append(match)
    return np.array(reorder)


# def find_grasp_frame_near(gripper_vals, approx_idx, window=30):
#     lo = max(0, approx_idx - window)
#     hi = min(len(gripper_vals), approx_idx + window)
#     local_min_offset = np.argmin(gripper_vals[lo:hi])
#     return lo + local_min_offset

def find_grasp_frame_from_closure(gripper_vals, times, vel_thresh=0.002,
                                   plateau_len=8, deadband=0.005):
    """Detect the grasp frame from the gripper trajectory's shape, without
    assuming which direction (increasing or decreasing) corresponds to
    closing -- different arms/joints can use opposite sign conventions.
    """
    vel = np.gradient(gripper_vals, times)
    open_val = np.median(gripper_vals[:10])

    # Infer closing direction from the largest deviation away from the
    # initial (open) value, rather than assuming decrease = closing.
    max_dev_idx = np.argmax(np.abs(gripper_vals - open_val))
    direction = np.sign(gripper_vals[max_dev_idx] - open_val)
    if direction == 0:
        raise RuntimeError("Gripper value never moved -- check --arm is correct "
                            "and this episode actually grasped the cube.")

    closing_start = None
    for i in range(len(gripper_vals)):
        moved_enough = direction * (gripper_vals[i] - open_val) > deadband
        moving_now = direction * vel[i] > vel_thresh
        if moved_enough and moving_now:
            closing_start = i
            break
    if closing_start is None:
        raise RuntimeError("No closing motion detected -- check --arm is correct "
                            "and that this episode actually grasped the cube.")

    for i in range(closing_start, len(gripper_vals) - plateau_len):
        if np.all(np.abs(vel[i:i + plateau_len]) < vel_thresh):
            return i

    raise RuntimeError("Gripper closed but never stabilized -- may have "
                        "released before settling. Inspect gripper_trace.png.")


def set_actuated_positions(plant, context, actuator_indices, q_actuated):
    """SetPositions requires a vector of length plant.num_positions(), which
    can be larger than the number of actuators if the URDF has passive or
    mimic-linked joints (e.g. a second gripper finger driven off the first).
    Start from the default/home full position vector and only overwrite the
    entries that correspond to actuated joints.
    """
    q_full = plant.GetPositions(context).copy()
    for act_idx, val in zip(actuator_indices, q_actuated):
        joint = plant.get_joint_actuator(act_idx).joint()
        q_full[joint.position_start()] = val
    plant.SetPositions(context, q_full)
    return q_full


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", required=True)
    parser.add_argument("--episode", type=int, default=0)
    parser.add_argument("--urdf", default="urdf/stationary_ai.urdf")
    parser.add_argument("--source", default="observation.state",
                         choices=["action", "observation.state"])
    parser.add_argument("--arm", default="left", choices=["left", "right"],
                         help="Which arm actually grasped the cube in your recording.")
    parser.add_argument("--grasp-frame", type=int, default=None,
                         help="Manually specify the frame index of the grasp "
                              "instead of auto-detecting it from gripper closure.")
    parser.add_argument("--video-time", type=float, default=12.0,
                     help="Approximate time (s) in the recorded video where the gripper visibly holds the cube.")
    args = parser.parse_args()

    builder = DiagramBuilder()
    plant, _ = AddMultibodyPlantSceneGraph(builder, time_step=0.0)
    Parser(plant).AddModels(args.urdf)
    plant.Finalize()

    times, q_lerobot = load_episode(args.data_root, args.episode, args.source)
    dt = np.median(np.diff(times))
    fps = 1.0 / dt
    print(f"Estimated fps from timestamps: {fps:.3f}")
    print(f"Episode duration from parquet: {times[-1]:.2f}s, num frames: {len(times)}")

    target_time = 12.0
    approx_idx = int(round(target_time / dt))
    approx_idx = np.clip(approx_idx, 0, len(times) - 1)
    print(f"Approx frame index for t={target_time}s: {approx_idx} (actual t={times[approx_idx]:.3f}s)")

    reorder = build_actuator_reorder(plant)
    q_drake_order = q_lerobot[:, reorder]

    actuator_indices = plant.GetJointActuatorIndices()
    drake_joint_names = [
        plant.get_joint_actuator(idx).joint().name() for idx in actuator_indices
    ]

    # Diagnostic: print the gripper value range for both arms so you can
    # confirm which one actually closed on the cube in this episode.
    for side in ("left", "right"):
        lr_idx = 6 if side == "left" else 13
        drake_idx = np.where(reorder == lr_idx)[0][0]
        vals = q_drake_order[:, drake_idx]
        print(f"{side} gripper column range: min={vals.min():.4f}, max={vals.max():.4f}")

    # Index of this arm's gripper joint within the 14 LeRobot columns:
    # left_joint_6 -> index 6, right_joint_6 -> index 13.
    gripper_lerobot_idx = 6 if args.arm == "left" else 13
    # Find where that column landed after reordering into Drake actuator order.
    gripper_drake_idx = np.where(reorder == gripper_lerobot_idx)[0][0]
    gripper_vals = q_drake_order[:, gripper_drake_idx]

    # grasp_idx = args.grasp_frame if args.grasp_frame is not None \
    #     else find_grasp_frame(gripper_vals)

    # print(f"Using grasp frame index {grasp_idx} (t={times[grasp_idx]:.2f}s), "
    #       f"gripper value at that frame: {gripper_vals[grasp_idx]:.4f} "
    #       f"(min={gripper_vals.min():.4f}, max={gripper_vals.max():.4f})")

    if args.grasp_frame is not None:
        grasp_idx = args.grasp_frame
    else:
        grasp_idx = find_grasp_frame_from_closure(gripper_vals, times)

    print(f"Refined grasp frame: {grasp_idx} (t={times[grasp_idx]:.2f}s), "
      f"gripper value: {gripper_vals[grasp_idx]:.4f}")

    lo, hi = max(0, approx_idx - 60), min(len(gripper_vals), approx_idx + 60)
    for i in range(lo, hi, 2):
        print(f"  frame {i:4d}  t={times[i]:6.2f}s  gripper={gripper_vals[i]:.4f}")

    plt.plot(times[lo:hi], gripper_vals[lo:hi])
    plt.axvline(times[approx_idx], color='r', linestyle='--', label='video t=12s')
    plt.xlabel("time (s)"); plt.ylabel("gripper value"); plt.legend()
    plt.savefig(f"{args.data_root}/gripper_trace.png")

    q_at_grasp = q_drake_order[grasp_idx]
    tmp_ctx = plant.CreateDefaultContext()
    set_actuated_positions(plant, tmp_ctx, actuator_indices, q_at_grasp)

    pad_offset_left = np.array([0.0286, -0.0135, 0.0037])
    pad_offset_right = np.array([0.0286, 0.0135, 0.0037])

    left_finger = plant.GetFrameByName(f"follower_{args.arm}_gripper_left")
    right_finger = plant.GetFrameByName(f"follower_{args.arm}_gripper_right")

    X_W_left  = left_finger.CalcPoseInWorld(tmp_ctx)
    X_W_right = right_finger.CalcPoseInWorld(tmp_ctx)

    lf_pos = X_W_left.multiply(pad_offset_left)     # transforms local offset point into world frame
    rf_pos = X_W_right.multiply(pad_offset_right)

    gap = np.linalg.norm(lf_pos - rf_pos)
    print(f"Fingertip gap at grasp frame: {gap*100:.2f} cm")

    cube_pos_estimate = (lf_pos + rf_pos) / 2

    print(f"\nLeft finger position:  {lf_pos}")
    print(f"Right finger position: {rf_pos}")
    print(f"\nEstimated cube position (finger midpoint): {cube_pos_estimate}")
    print(f"\nPlug this into REPLAY_CONFIG['cube_position'] in the sim script:")
    print(f'  "cube_position": [{cube_pos_estimate[0]:.4f}, '
          f'{cube_pos_estimate[1]:.4f}, {cube_pos_estimate[2]:.4f}],')


if __name__ == "__main__":
    main()
