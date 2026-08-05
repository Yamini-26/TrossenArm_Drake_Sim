#!/usr/bin/env python

# Replay a recorded LeRobot teleoperation episode in sim

# Save every 10th frame of LeRobot teleop video to PNG frames for comparison with sim:
# Usage: ffmpeg -i data/pick_place_3/videos/chunk-000/observation.images.cam_low/episode_000000.mp4 -vf "select='not(mod(n,10))'" -vsync vfr data/pick_place_3/frames/cam_low/frame_%06d.png

from pathlib import Path

import json
import os
import time
import numpy as np
import pandas as pd
import xml.etree.ElementTree as ET

from pydrake.all import (
    StartMeshcat,
    DiagramBuilder,
    AddMultibodyPlantSceneGraph,
    Parser,
    Simulator,
    SimulatorConfig,
    ApplySimulatorConfig,
    SceneGraphConfig,
    ApplyVisualizationConfig,
    VisualizationConfig,
    LeafSystem,
    RigidTransform,
    EventStatus,
    StateInterpolatorWithDiscreteDerivative,
    PiecewisePolynomial,
    RgbdSensor,
    CameraInfo,
    ClippingRange,
    DepthRange,
    DepthRenderCamera,
    ColorRenderCamera,
    RenderCameraCore,
    MakeRenderEngineVtk,
    RenderEngineVtkParams,
    AbstractValue,
    ImageRgba8U,
    LightParameter,
    Rgba,
)
from pydrake.common.yaml import yaml_load_file
from PIL import Image
from pydrake.systems.primitives import LogVectorOutput


# Camera specs matching real Intel RealSense D405 hardware
CAM_WIDTH = 640
CAM_HEIGHT = 480
# CAM_FOCAL = 605.0
CAM_FOCAL = 370.2

CAMERA_FRAME_MAP = {
    "cam_high": "cam_high_color_optical_frame",
    "cam_low": "cam_low_color_optical_frame",
    "cam_left_wrist": "follower_left_camera_color_optical_frame",
    "cam_right_wrist": "follower_right_camera_color_optical_frame",
}

# LeRobot's 14-dim action/observation.state column order
LEROBOT_JOINT_ORDER = [
    "left_joint_0", "left_joint_1", "left_joint_2", "left_joint_3",
    "left_joint_4", "left_joint_5", "left_joint_6",
    "right_joint_0", "right_joint_1", "right_joint_2", "right_joint_3",
    "right_joint_4", "right_joint_5", "right_joint_6",
]


# LeRobot episode loading + joint-order mapping

def load_episode(data_root, episode_index, source_col="observation.state",
                  chunks_size=1000):
    """Read one episode's parquet file. Returns (times, q) with q shape (T, 14)
    in LEROBOT_JOINT_ORDER order.

    source_col: "observation.state" (what the follower arm actually measured
    -- use this for a faithful replay) or "action" (what the leader commanded).
    """
    data_root = Path(data_root)
    chunk = episode_index // chunks_size
    parquet_path = (
        data_root / "data" / f"chunk-{chunk:03d}"
        / f"episode_{episode_index:06d}.parquet"
    )
    if not parquet_path.exists():
        raise FileNotFoundError(f"No such episode file: {parquet_path}")

    df = pd.read_parquet(parquet_path)
    times = df["timestamp"].to_numpy().astype(float)
    times = times - times[0]

    # grip = np.stack(df["observation.state"].to_numpy())[:, LEROBOT_JOINT_ORDER.index("right_joint_6")]
    # for t, g in zip(times, grip):
    #     print(f"t={t:.2f}s  gripper={g:.4f}")

    # with open(Path(data_root) / "meta" / "info.json") as f:
    #     info = json.load(f)
    # print(info["features"]["observation.state"]["names"])

    # Result: pick_place_3 t=0 gripper=0.0000; cube pick -> t=12.47s  gripper=0.0155; cube drop -> t=20.43s  gripper=0.0155

    q = np.stack(df[source_col].to_numpy()).astype(float)
    # names = info["features"]["observation.state"]["names"]
    # for i, name in enumerate(names):
    #     col = q[:, i]
    #     print(f"{i:2d} {name:20s} min={col.min():.4f} max={col.max():.4f} range={col.max()-col.min():.4f}")
    print(f"Loaded {parquet_path}  (source={source_col})")
    print(f"  frames: {q.shape[0]}, duration: {times[-1]:.2f}s")
    return times, q


def build_actuator_reorder(plant, lerobot_names=LEROBOT_JOINT_ORDER):
    """Return index array so q_drake_order = q_lerobot_order[:, reorder].
    Matches by substring: assumes each Drake actuator's joint name CONTAINS
    the LeRobot column name, e.g. 'follower_left_joint_3' <-> 'left_joint_3'.
    """
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
            raise ValueError(
                f"Could not match Drake actuator/joint '{drake_name}' to any "
                f"LeRobot column name. Update LEROBOT_JOINT_ORDER or this "
                f"matching rule."
            )
        reorder.append(match)

    # print("Drake actuator order  <-  LeRobot column:")
    # for name, idx in zip(drake_joint_names, reorder):
    #     print(f"  {name:35s} <- {lerobot_names[idx]}")

    return np.array(reorder)


class ReplayController(LeafSystem):
    """Outputs joint positions from a PiecewisePolynomial built from the
    recorded episode (already reordered into Drake actuator order)."""

    def __init__(self, times, q_drake_order, num_joints):
        LeafSystem.__init__(self)
        self._num_joints = num_joints
        self._traj = PiecewisePolynomial.FirstOrderHold(times, q_drake_order.T)
        self._end_time = times[-1]
        self.DeclareVectorOutputPort(
            "desired_positions", num_joints, self.CalcOutput
        )

    def CalcOutput(self, context, output):
        t = min(context.get_time(), self._end_time)
        output.SetFromVector(self._traj.value(t).flatten())

    @property
    def end_time(self):
        return self._end_time


# Cameras

class ImageSaver(LeafSystem):
    """Save frames from one RgbdSensor color port to disk."""

    def __init__(self, camera_name, save_dir,
                 save_interval=0.5, width=CAM_WIDTH, height=CAM_HEIGHT):
        LeafSystem.__init__(self)
        self._camera_name = camera_name
        self._save_dir = os.path.join(save_dir, camera_name)
        self._frame_count = 0
        self._width = width
        self._height = height
        os.makedirs(self._save_dir, exist_ok=True)

        self._image_port = self.DeclareAbstractInputPort(
            "color_image",
            AbstractValue.Make(ImageRgba8U(width, height))
        )
        self.DeclarePeriodicPublishEvent(save_interval, 0.0, self._save_image)
        print(f"  [{camera_name}] -> '{self._save_dir}/' every {save_interval}s")

    def _save_image(self, context):
        try:
            image = self._image_port.Eval(context)
        except Exception as e:
            print(f"  [{self._camera_name}] WARNING: {e}")
            return
        img_array = np.frombuffer(image.data, dtype=np.uint8)
        img_array = img_array.reshape((self._height, self._width, 4))
        filename = os.path.join(self._save_dir, f"frame_{self._frame_count:05d}.png")
        Image.fromarray(img_array[:, :, :3], mode="RGB").save(filename)
        self._frame_count += 1
        t = context.get_time()
        if self._frame_count % 10 == 0:
            print(f"  [{self._camera_name}] frame {self._frame_count} t={t:.2f}s")


def add_cameras_from_urdf(builder, plant, scene_graph, renderer_name,
                           save_dir="simulation_frames", save_interval=0.5):
    """Attach one RgbdSensor per camera using the exact body frame from the URDF."""
    cam_info = CameraInfo(
        width=CAM_WIDTH, height=CAM_HEIGHT,
        focal_x=CAM_FOCAL, focal_y=CAM_FOCAL,
        center_x=CAM_WIDTH / 2.0, center_y=CAM_HEIGHT / 2.0,
    )
    color_cam = ColorRenderCamera(
        RenderCameraCore(renderer_name, cam_info,
                          ClippingRange(0.01, 10.0), RigidTransform()),
        show_window=False,
    )
    depth_cam = DepthRenderCamera(
        RenderCameraCore(renderer_name, cam_info,
                          ClippingRange(0.01, 10.0), RigidTransform()),
        DepthRange(0.01, 10.0),
    )

    sensors = {}
    print("\nAdding cameras from URDF frames:")

    for cam_name, frame_name in CAMERA_FRAME_MAP.items():
        urdf_frame = plant.GetFrameByName(frame_name)
        parent_body = urdf_frame.body()
        parent_frame_id = plant.GetBodyFrameIdOrThrow(parent_body.index())
        X_BF = urdf_frame.GetFixedPoseInBodyFrame()

        sensor = builder.AddSystem(RgbdSensor(
            parent_id=parent_frame_id,
            X_PB=X_BF,
            color_camera=color_cam,
            depth_camera=depth_cam,
        ))
        sensor.set_name(f"rgbd_{cam_name}")

        builder.Connect(
            scene_graph.get_query_output_port(),
            sensor.query_object_input_port(),
        )

        saver = builder.AddSystem(ImageSaver(cam_name, save_dir, save_interval))
        builder.Connect(sensor.color_image_output_port(), saver.get_input_port(0))

        sensors[cam_name] = sensor

        default_ctx = plant.CreateDefaultContext()
        X_WF = urdf_frame.CalcPoseInWorld(default_ctx)
        pos = X_WF.translation()
        fwd = X_WF.rotation().matrix() @ np.array([0, 0, 1])
        print(f"  {cam_name:20s} frame='{frame_name}'")
        print(f"    world pos (home): [{pos[0]:.4f}, {pos[1]:.4f}, {pos[2]:.4f}]")
        print(f"    optical fwd (Z):  [{fwd[0]:.3f}, {fwd[1]:.3f}, {fwd[2]:.3f}]")
        print(f"    parent body:      '{parent_body.name()}'")

    return sensors


def load_cube_urdf(mass: float, friction: float, side: float) -> str:
    tree = ET.parse("urdf/cube.urdf")
    root = tree.getroot()

    root.find(".//mass").set("value", str(mass))

    I = mass * side**2 / 6.0
    inertia = root.find(".//inertia")
    for attr in ["ixx", "iyy", "izz"]:
        inertia.set(attr, f"{I:.6e}")
    for attr in ["ixy", "ixz", "iyz"]:
        inertia.set(attr, "0")

    for box in root.findall(".//box"):
        box.set("size", f"{side} {side} {side}")

    collision = root.find(".//collision")
    contact = collision.find("contact")
    if contact is None:
        contact = ET.SubElement(collision, "contact")

    def set_or_create(parent, tag, value):
        el = parent.find(tag)
        if el is None:
            el = ET.SubElement(parent, tag)
        el.set("value", str(value))

    set_or_create(contact, "lateral_friction", friction)
    set_or_create(contact, "rolling_friction", 0.001)
    set_or_create(contact, "spinning_friction", 0.001)

    return ET.tostring(root, encoding="unicode")


# Config

REPLAY_CONFIG = {
    "data_root": "./data/pick_place_3",
    "episode_index": 0,
    "source_col": "observation.state",
    "save_dir": f"simulation_frames/replay_{int(time.time())}",
    "cube_mass": 0.05,
    "cube_friction": 0.8,
    "cube_side": 0.02,
    # Starting position of the cube in the sim
    # "cube_position": [-0.010, -0.197, 0.015],
    # "cube_position": [-0.0105, -0.1844, 0.0983],
    # "cube_position": [-0.0114, -0.1735, 0.0716],
    "cube_position": [-0.0100, -0.1950, 0.0120],  # pick_place_3
    # "cube_position": [-0.3338, 0.0076, 0.0166],    # pick_place_center_2
    # "lights": [
    #     LightParameter(
    #         type="directional",
    #         direction=[0.5, -0.5, -1.0],
    #         color=Rgba(1.0, 0.95, 0.85, 1.0),
    #         intensity=2.5,
    #         frame="world",
    #     )
    # ],
    "lights": [
        LightParameter(
            type="directional",
            direction=[0.0, 0.0, -1.0],
            color=Rgba(0.8, 0.8, 0.8, 1.0),
            intensity=1.2,
            frame="world",
        )
    ],
}


def run_simulation(config: dict):
    builder = DiagramBuilder()
    plant, scene_graph = AddMultibodyPlantSceneGraph(builder, time_step=0.0)

    renderer_name = "renderer"
    light_cfg = config.get("lights", [])
    renderer_params = RenderEngineVtkParams()
    renderer_params.lights = light_cfg
    renderer = MakeRenderEngineVtk(renderer_params)
    scene_graph.AddRenderer(renderer_name, renderer)

    model_indices = Parser(plant).AddModels("urdf/stationary_ai.urdf")

    cube_urdf_str = load_cube_urdf(config["cube_mass"], config["cube_friction"], config["cube_side"])
    Parser(plant).AddModelsFromString(cube_urdf_str, "urdf")

    cube_body = plant.GetBodyByName("cube_link")
    X = RigidTransform()
    X.set_translation(config["cube_position"])
    plant.SetDefaultFloatingBaseBodyPose(cube_body, X)
    plant.Finalize()

    default_context = plant.CreateDefaultContext()
    cube_pose_default = cube_body.EvalPoseInWorld(default_context)
    print(f"[INIT] Default cube pose after Finalize: {cube_pose_default.translation()}")

    # Inspect the cube's friction
    # inspector = scene_graph.model_inspector()
    # for geom_id in inspector.GetAllGeometryIds():
    #     if inspector.GetName(geom_id).startswith("cube") or "cube" in inspector.GetName(geom_id).lower():
    #         props = inspector.GetProximityProperties(geom_id)
    #         if props and props.HasProperty("material", "coulomb_friction"):
    #             fric = props.GetProperty("material", "coulomb_friction")
    #             print(f"{inspector.GetName(geom_id)}: static={fric.static_friction()}, dynamic={fric.dynamic_friction()}")
    #         else:
    #             print(f"{inspector.GetName(geom_id)}: NO friction property found — using Drake default")

    # position_names = plant.GetPositionNames()
    # print("Position names and their order in q:")
    # for i, name in enumerate(position_names):
    #     print(f"  Index {i:2d}: {name}")

    scene_graph_config = SceneGraphConfig()
    scene_graph_config.default_proximity_properties.compliance_type = "compliant"
    scene_graph.set_config(scene_graph_config)

    add_cameras_from_urdf(
        builder, plant, scene_graph,
        renderer_name=renderer_name,
        save_dir=config["save_dir"],
        save_interval=0.5,
    )

    meshcat = StartMeshcat()
    visualization_config = VisualizationConfig()
    visualization_config.publish_proximity = True
    visualization_config.publish_period = np.inf
    ApplyVisualizationConfig(visualization_config, builder=builder, meshcat=meshcat)

    meshcat_config = yaml_load_file("meshcat_config.yaml")
    for p in meshcat_config["initial_properties"]:
        meshcat.SetProperty(p["path"], p["property"], p["value"])
    meshcat.SetCameraPose([0.9, 0.0, 0.9], [0.0, 0.0, 0.4])

    nu = len(plant.GetJointActuatorIndices())

    # Load the recorded episode and build the replay controller
    times, q_lerobot = load_episode(
        config["data_root"], config["episode_index"], config["source_col"]
    )
    reorder = build_actuator_reorder(plant)
    q_drake_order = q_lerobot[:, reorder]
    if q_drake_order.shape[1] != nu:
        raise ValueError(
            f"Recorded data has {q_drake_order.shape[1]} joints but plant has "
            f"{nu} actuators -- check LEROBOT_JOINT_ORDER / URDF."
        )

    actuator_indices = plant.GetJointActuatorIndices()
    drake_joint_names = [
        plant.get_joint_actuator(idx).joint().name() for idx in actuator_indices
    ]
    gripper_cols = [i for i, name in enumerate(drake_joint_names) if "carriage" in name.lower()]

    # Harcoded grasp/release to pinch hold the cube, since the recorded gripper values are not reliable.
    # grasp window from gripper data
    t_grasp, t_release = 12.47, 20.43
    grasp_mask = (times >= t_grasp) & (times <= t_release)

    fully_closed_value = 0.0
    grasp_row_idx = np.where(grasp_mask)[0]
    q_drake_order[np.ix_(grasp_row_idx, gripper_cols)] = fully_closed_value

    for col in gripper_cols:
        q_drake_order[grasp_mask, col] = fully_closed_value

    print(q_drake_order[grasp_mask][:, gripper_cols][:5])  # sanity check after the fix

    controller = builder.AddSystem(ReplayController(times, q_drake_order, nu))

    state_interpolator = builder.AddSystem(
        StateInterpolatorWithDiscreteDerivative(nu, 0.01, True)
    )
    builder.Connect(controller.get_output_port(0), state_interpolator.get_input_port())
    builder.Connect(
        state_interpolator.get_output_port(),
        plant.get_desired_state_input_port(model_indices[0]),
    )

    state_logger = LogVectorOutput(plant.get_state_output_port(), builder)
    state_logger.set_name("state_logger")

    diagram = builder.Build()
    context = diagram.CreateDefaultContext()
    plant_context = plant.GetMyContextFromRoot(context)

    simulator = Simulator(diagram, context)
    sim_config = SimulatorConfig()
    sim_config.integration_scheme = "cenic"
    sim_config.accuracy = 1e-3
    sim_config.max_step_size = 0.1
    sim_config.use_error_control = True
    sim_config.publish_every_time_step = True
    ApplySimulatorConfig(sim_config, simulator)
    simulator.set_target_realtime_rate(1.0)
    simulator.Initialize()

    plant_context_after_init = plant.GetMyContextFromRoot(simulator.get_context())
    cube_pose_after_init = cube_body.EvalPoseInWorld(plant_context_after_init)
    print(f"[INIT] Cube pose after simulator.Initialize(): {cube_pose_after_init.translation()}")

    input("Waiting for meshcat... press [ENTER] to start replaying.")
    print("")
    print("Watch for movement in MeshCat!")
    print("Press Ctrl+C to stop the simulation.")

    try:
        meshcat.StartRecording()
        simulator.AdvanceTo(controller.end_time)
        diagram_context = simulator.get_context()
        state_log = state_logger.FindLog(diagram_context)

        log_times = state_log.sample_times()
        state_data = state_log.data()

        nq = plant.num_positions()
        nv = plant.num_velocities()
        q_log = state_data[:nq, :].T

        ee_frame = plant.GetFrameByName("follower_right_ee_gripper_link")
        tmp_ctx = plant.CreateDefaultContext()

        ee_pos_log = np.zeros((len(log_times), 3))
        obj_pos_log = np.zeros((len(log_times), 3))
        obj_quat_log = q_log[:, nq - 7: nq - 3]

        for i, q in enumerate(q_log):
            plant.SetPositions(tmp_ctx, q)
            ee_pos_log[i] = ee_frame.CalcPoseInWorld(tmp_ctx).translation()
            obj_pos_log[i] = cube_body.EvalPoseInWorld(tmp_ctx).translation()

        save_dir = config["save_dir"]
        os.makedirs(save_dir, exist_ok=True)

        left_finger_frame = plant.GetFrameByName("follower_right_gripper_left")   # match the grasping arm
        right_finger_frame = plant.GetFrameByName("follower_right_gripper_right")

        # Use the pad offsets derived from the URDF collision origins
        # using grep -A 20 '<link name="follower_right_gripper_left">' urdf/stationary_ai.urdf  and right_gripper_right
        # replace with actual left/right values
        pad_offset_left = np.array([0.0286, -0.0135, 0.0037])
        pad_offset_right = np.array([0.0286, 0.0135, 0.0037])

        # Extend existing per-frame loop
        left_finger_pos_log = np.zeros((len(log_times), 3))
        right_finger_pos_log = np.zeros((len(log_times), 3))
        gripper_gap_log = np.zeros(len(log_times))

        for i, q in enumerate(q_log):
            plant.SetPositions(tmp_ctx, q)
            ee_pos_log[i] = ee_frame.CalcPoseInWorld(tmp_ctx).translation()
            obj_pos_log[i] = cube_body.EvalPoseInWorld(tmp_ctx).translation()

            X_W_left = left_finger_frame.CalcPoseInWorld(tmp_ctx)
            X_W_right = right_finger_frame.CalcPoseInWorld(tmp_ctx)
            lf_pos = X_W_left.multiply(pad_offset_left)
            rf_pos = X_W_right.multiply(pad_offset_right)

            left_finger_pos_log[i] = lf_pos
            right_finger_pos_log[i] = rf_pos
            gripper_gap_log[i] = np.linalg.norm(lf_pos - rf_pos)

        # at t=12.47s in sim log, find the matching index and print:
        idx = np.argmin(np.abs(log_times - 12.47))
        print(f"gripper_gap at grasp = {gripper_gap_log[idx]*100:.3f} cm")
        midpoint = (left_finger_pos_log[idx] + right_finger_pos_log[idx]) / 2
        print(f"finger midpoint at t={log_times[idx]:.2f}s: {midpoint}")

        np.savez_compressed(
            os.path.join(config["save_dir"], "gripper_trace.npz"),
            times=log_times,
            left_finger_pos=left_finger_pos_log,
            right_finger_pos=right_finger_pos_log,
            gripper_gap=gripper_gap_log,
            cube_pos=obj_pos_log,
        )

        np.savez_compressed(
            os.path.join(save_dir, "trajectory_states.npz"),
            times=log_times,
            q=q_log,
            ee_pos=ee_pos_log,
            obj_pos=obj_pos_log,
            obj_quat=obj_quat_log,
        )

        data = np.load(os.path.join(config["save_dir"], "trajectory_states.npz"))
        times = data["times"]

        with open(os.path.join(save_dir, "trajectory_meta.json"), "w") as f:
            json.dump({
                "nq": nq, "nv": nv, "n_frames": len(log_times),
                "episode": config["episode_index"],
                "source_col": config["source_col"],
            }, f, indent=2)

        print(f"  [StateLog] Saved {len(log_times)} timesteps -> {save_dir}/trajectory_states.npz")
        
        meshcat.StopRecording()
        meshcat.PublishRecording()
    except KeyboardInterrupt:
        EventStatus.Killed(diagram, "Simulation stopped by user.")


if __name__ == "__main__":
    run_simulation(REPLAY_CONFIG)
