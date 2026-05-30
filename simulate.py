#!/usr/bin/env python

##
#
# Run a simple automatic controller simulation of the Trossen Stationary bimanual robot.
#
##

from typing import List
from functools import partial

import os
import numpy as np
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
    Meshcat,
    LeafSystem,
    Multiplexer,
    ConstantVectorSource,
    RigidTransform,
    EventStatus,
    StateInterpolatorWithDiscreteDerivative,
    FrameIndex,
    InverseKinematics,
    RotationMatrix,
    Solve,
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
)
from pydrake.common.yaml import yaml_load_file
from PIL import Image 


# Camera specs matching real Intel RealSense D405 hardware
CAM_WIDTH  = 640
CAM_HEIGHT = 480
CAM_FOCAL  = 605.0   # D405 focal length in pixels at 640×480
 
# Map logical camera names → the URDF color optical frame names
# Use the color_optical_frame because:
#   1. It is the true optical centre of the RGB sensor
#   2. Drake's RgbdSensor uses the same convention (Z forward, X right, Y down)
#   3. The URDF already applies the ROS→optical rotation (-pi/2, 0, -pi/2) so no extra rotation transform is needed
CAMERA_FRAME_MAP = {
    "cam_high":         "cam_high_color_optical_frame",
    "cam_low":          "cam_low_color_optical_frame",
    "cam_left_wrist":   "follower_left_camera_color_optical_frame",
    "cam_right_wrist":  "follower_right_camera_color_optical_frame",
}

# def inspect_camera_frames(plant):
#     """
#     Print all frames for camera
#     Extract frame names from URDF so that RgbdSensors can be attached to them
#     """
#     print("\n" + "=" * 60)
#     print("CAMERA / SENSOR FRAMES IN URDF")
#     print("=" * 60)
#     camera_keywords = ["cam", "camera", "realsense", "sensor", "optical"]
#     found = []
#     for i in range(plant.num_frames()):
#         frame = plant.get_frame(FrameIndex(i))
#         name = frame.name().lower()
#         if any(kw in name for kw in camera_keywords):
#             found.append(frame.name())
#     if found:
#         for f in found:
#             print(f"  {f}")
#     else:
#         print("  (none found)")
#     print("=" * 60 + "\n")
#     return found


def solve_ik(plant, plant_context, ee_frame, target_position, target_rotation=None, q0=None):
    """
    Solve IK for the right arm to place ee_frame at target_position.
    
    Args:
        plant: the MultibodyPlant
        plant_context: current plant context (used for initial guess)
        ee_frame: the end-effector frame to position (follower_left_ee_gripper_link)
        target_position: np.array [x, y, z] in world frame
        target_rotation: RotationMatrix for orientation (None = unconstrained)
        q0: initial joint guess (None = use current context)
    
    Returns:
        q_sol: joint positions solution, or None if failed
    """
    ik = InverseKinematics(plant, plant_context)
    
    # Position constraint: place ee within 1cm of target
    ik.AddPositionConstraint(
        frameB=ee_frame,           # gripper frame
        p_BQ=np.zeros(3),          # point at ee_frame origin
        frameA=plant.world_frame(), # expressed in world frame
        p_AQ_lower=target_position - 0.002,  # 0.2cm tolerance box
        p_AQ_upper=target_position + 0.002,
    )
    
    # Orientation constraint: point gripper downward (optional but important for grasping)
    if target_rotation is not None:
        ik.AddOrientationConstraint(
            frameAbar=plant.world_frame(),
            R_AbarA=target_rotation,
            frameBbar=ee_frame,
            R_BbarB=RotationMatrix(),  # identity = no extra rotation on ee
            theta_bound=0.1,           # 0.1 rad (~6 deg) tolerance
        )
    
    # Set initial guess
    prog = ik.prog()
    q_vars = ik.q()
    
    if q0 is not None:
        prog.SetInitialGuess(q_vars, q0)
    else:
        q0 = plant.GetPositions(plant_context)
        prog.SetInitialGuess(q_vars, q0)
    
    result = Solve(prog)
    
    if result.is_success():
        return result.GetSolution(q_vars)
    else:
        print(f"  IK failed for target {target_position}")
        return None
    

class PickAndPlaceController(LeafSystem):
    """
    State machine controller for pick-and-place.
    Computes IK waypoints and interpolates between them.
    """
    
    # State machine phases
    APPROACH  = 0  # Move above cube
    DESCEND   = 1  # Lower to cube
    GRASP     = 2  # Close gripper
    LIFT      = 3  # Lift cube
    TRANSPORT = 4  # Move above drop zone
    DROP      = 5  # Lower to drop zone
    RELEASE   = 6  # Open gripper
    RESET     = 7  # Return to home
    
    def __init__(self, plant, num_joints, cube_position, drop_position):
        LeafSystem.__init__(self)
        self._plant = plant
        self._plant_context = plant.CreateDefaultContext()
        self._num_joints = num_joints
        
        # Key positions
        self._cube_pos = np.array(cube_position)
        self._drop_pos = np.array(drop_position)

        # Positions that we know from inspecting the model:
        #   EE frame position: (-0.020, 0.204, 0.183)
        #   right finger:      (-0.043, 0.273, 0.183)
        #   left finger:       (0.003, 0.273, 0.183)
        # EE (End Effector) starts at: [-0.02, 0.20, 0.18] and the cube is at: [0.10, 0.15, 0.02]
        # The difference is:
        #   X: needs to move +0.12m (12cm to the right)
        #   Y: needs to move -0.05m (5cm backward)
        #   Z: needs to move -0.16m (16cm down)

        # End-effector frame
        self._ee_frame = plant.GetFrameByName("follower_left_ee_gripper_link")
        self._left_finger  = plant.GetFrameByName("follower_left_gripper_left")
        self._right_finger = plant.GetFrameByName("follower_left_gripper_right")

        ee_pose = self._ee_frame.CalcPoseInWorld(self._plant_context).translation()
        lf_pose = self._left_finger.CalcPoseInWorld(self._plant_context).translation()
        rf_pose = self._right_finger.CalcPoseInWorld(self._plant_context).translation()

        finger_center = (lf_pose + rf_pose) / 2

        # Offset from EE frame to the actual grasping point where the cube should be relative to EE
        self._grasp_offset = finger_center - ee_pose  # Vector from EE origin to center between fingers (grasp point)

        # Adjust the target positions by the offset
        self._target_approach = self._cube_pos - self._grasp_offset + np.array([0, 0, 0.15])  # Hover above cube
        # cube is 2cm tall, half of that is 1cm (0.01m), so we add that to the Z to ensure we are above the cube when grasping
        self._target_grasp = self._cube_pos - self._grasp_offset + np.array([0, 0.05, 0.01])   # Position EE so cube is between fingers
        self._target_lift = self._cube_pos - self._grasp_offset + np.array([0, 0, 0.15])  # Lift cube
        self._target_transport = self._drop_pos - self._grasp_offset + np.array([0, 0, 0.15])  # Transport cube
        self._target_drop = self._drop_pos - self._grasp_offset + np.array([0, 0.05, 0.01])   # Drop cube

        # Print debug info about initial positions
        print(f"cube position:   {self._cube_pos}")
        print(f"EE origin:      {ee_pose}")
        print(f"Left finger:    {lf_pose}")
        print(f"Right finger:   {rf_pose}")
        print(f"Finger center:  {finger_center}")
        # print(f"  Finger vs target Y: {finger_center[1] - self._cube_pos[1]:.4f}  ← real Y error")
        # print(f"  Finger vs target Z: {finger_center[2] - self._cube_pos[2]:.4f}  ← real Z error")
        # print(f"Finger gap (Y):     {abs(lf_pose[1] - rf_pose[1]):.4f}m")
        print(f"Offset EE→fingers: {finger_center - ee_pose}")        
        print(f"\n{'='*60}")
        print(f"GRIPPER CALIBRATION")
        print(f"{'='*60}")
        print(f"EE frame position:  {ee_pose}")
        print(f"Grasp offset:       {self._grasp_offset}")
        print(f"\nTarget positions (EE frame targets, not cube targets):")
        print(f"  Approach: {self._target_approach}")
        print(f"  Grasp:    {self._target_grasp}")
        print(f"  Lift:     {self._target_lift}")
        print(f"  Transport:{self._target_transport}")
        print(f"  Drop:     {self._target_drop}")
        
        # Gripper control
        self._gripper_open = 0.044   # Fully open
        self._gripper_closed = 0.005 # Closed (grasping)
        
        # Downward-facing rotation: gripper X-axis points down (-Z world)
        # Adjust based on gripper's resting direction
        self._grasp_rotation = None #RotationMatrix.MakeYRotation(np.pi / 2)
        
        # State tracking
        self._state = self.APPROACH
        self._state_start_time = 0.0
        self._phase_duration = 2.0  # seconds per phase (tune as needed)
        
        # IK solutions: computed once, interpolated during execution
        self._q_start = None   # joint positions at start of current phase
        self._q_target = None  # joint positions at end of current phase
        self._phase_computed = False
        
        # Output port
        self.DeclareVectorOutputPort("desired_positions", num_joints, self.CalcOutput)
        
        # State names for logging
        self._state_names = ["APPROACH", "DESCEND", "GRASP", "LIFT",
                             "TRANSPORT", "DROP", "RELEASE", "RESET"]
    
    def _get_left_arm_gripper_index(self):
        """Left arm gripper is actuator index 6 (follower_left_left_carriage_joint)."""
        return 6
    
    def _solve_ik_for_position(self, target_pos, q_current):
        """Solve IK for a Cartesian target, starting from q_current."""
        return solve_ik(
            self._plant,
            self._plant_context,
            self._ee_frame,
            target_pos,
            target_rotation=self._grasp_rotation,
            q0=q_current,
        )
    
    def _get_target_for_state(self, state):
        """Return the Cartesian target position for each state."""
        targets = {
            self.APPROACH:  self._target_approach,
            self.DESCEND:   self._target_grasp,
            self.GRASP:     self._target_grasp,    # stay in place, just close gripper
            self.LIFT:      self._target_lift,
            self.TRANSPORT: self._target_transport,
            self.DROP:      self._target_drop,
            self.RELEASE:   self._target_drop,    # stay in place, just open gripper
            self.RESET:     self._target_approach, # return to neutral hover
        }

        return targets.get(state)
    
    def CalcOutput(self, context, output):
        t = context.get_time()
        
        # Initialize on first call
        if self._q_start is None:
            self._q_start = self._plant.GetPositions(self._plant_context)
            self._q_target = self._q_start.copy()
            self._state_start_time = t
        
        # Compute IK for the current phase if not done yet
        if not self._phase_computed:
            self._phase_computed = True
            target_pos = self._get_target_for_state(self._state)
            print(f"[t={t:.2f}] Phase: {self._state_names[self._state]} -> target EE Position: {target_pos}")
            
            q_sol = self._solve_ik_for_position(target_pos, self._q_start)
            
            if q_sol is not None:
                self._q_target = q_sol
                # Update plant context so next IK starts from a good guess
                self._plant.SetPositions(self._plant_context, q_sol)
                print(f"  IK SUCCESS: Solution found")
            else:
                # IK failed: hold current position
                self._q_target = self._q_start.copy()
                print(f"  IK FAILED: Holding current position")
        
        # Interpolate between q_start and q_target
        phase_t = (t - self._state_start_time) / self._phase_duration
        phase_t = np.clip(phase_t, 0.0, 1.0)
        # Smooth step: ease in/out
        alpha = phase_t * phase_t * (3 - 2 * phase_t)
        q_interp = (1 - alpha) * self._q_start + alpha * self._q_target
        
        # Override gripper position based on state
        gripper_idx = self._get_left_arm_gripper_index()
        if self._state in (self.GRASP, self.LIFT, self.TRANSPORT, self.DROP):
            q_interp[gripper_idx] = self._gripper_closed
        else:
            q_interp[gripper_idx] = self._gripper_open
        
        # Write to output
        for i in range(self._num_joints):
            output[i] = q_interp[i]
        
        # Advance state machine when phase duration elapsed
        if t - self._state_start_time >= self._phase_duration:
            next_state = (self._state + 1) % (self.RESET + 1)
            print(f"[t={t:.2f}] -> Transitioning to {self._state_names[next_state]}")
            self._q_start = self._q_target.copy()
            self._state = next_state
            self._state_start_time = t
            self._phase_computed = False


# System to track and print gripper positions
class GripperPositionTracker(LeafSystem):
    """
    Tracks and prints the position of the gripper frames.
    This helps us understand where the grippers are in space.
    """
    
    def __init__(self, plant, gripper_frames):
        LeafSystem.__init__(self)
        self._plant = plant
        self._gripper_frames = gripper_frames
        self._last_print_time = 0
        self._print_interval = 1.0  # Print every second
        
        # Declare an input port to get plant state
        self.DeclareAbstractInputPort("plant_state", plant.get_state_output_port().get_data_type())
        
    def DoCalcNextUpdateTime(self, context, events):
        # Not needed, just for periodic printing
        pass
    
    def CalcOutput(self, context, output):
        # Get the plant context from the state
        state = self.get_input_port(0).Eval(context)
        
        # Create a plant context
        plant_context = self._plant.CreateDefaultContext()
        
    def EvalAndPrint(self, context):
        t = context.get_time()
        
        if t - self._last_print_time >= self._print_interval:
            self._last_print_time = t
            
            # Get plant context
            plant_context = self._plant.CreateDefaultContext()
            
            print(f"Time: {t:.2f} seconds")
            
            for frame_name in self._gripper_frames:
                try:
                    frame = self._plant.GetFrameByName(frame_name)
                    pose = frame.CalcPoseInWorld(plant_context)
                    pos = pose.translation()
                    print(f"  {frame_name:30s} -> ({pos[0]:.3f}, {pos[1]:.3f}, {pos[2]:.3f})")
                except Exception as e:
                    print(f"  {frame_name:30s} -> ERROR: {e}")


# Inspect all frames in the model to find gripper-related ones
def inspect_frames(plant):
    """
    Print all frames to find gripper-related ones.
    """
    print("ALL FRAMES IN THE MODEL")
    
    gripper_frames = []
    carriage_frames = []
    
    num_frames = plant.num_frames()
    for i in range(num_frames):
        frame = plant.get_frame(FrameIndex(i))
        name = frame.name()
        print(f"  {name}")
        
        # Look for gripper-related frames
        if 'gripper' in name.lower():
            gripper_frames.append(name)
        if 'carriage' in name.lower():
            carriage_frames.append(name)
    
    print("GRIPPER-RELATED FRAMES:")
    for f in gripper_frames:
        print(f"  - {f}")
    
    print("CARRIAGE FRAMES (gripper bases):")
    for f in carriage_frames:
        print(f"  - {f}")
    
    return gripper_frames, carriage_frames


# Custom controller by inheriting from LeafSystem
class DebugTrajectory(LeafSystem):
    """Trajectory that prints its output to verify it's working."""
    
    def __init__(self, num_joints):
        LeafSystem.__init__(self)
        self._num_joints = num_joints
        self.DeclareVectorOutputPort(
            "desired_positions", # Name of the output port
            num_joints,
            self.CalcOutput      # Callback function to calculate the output values
        )
        # print(f"Trajectory created for {num_joints} joints")
    
    # Move only the left arm joints (the first 6 actuators are left arm(0-5))
    def CalcOutput(self, context, output):  # context: contains simulation state, output: array to fill with joint commands
        t = context.get_time()
       
        # Simple trajectory: cosine wave that changes over time
        for i in range(self._num_joints):
            if i < 6:  # First 6 joints (left arm)
                # Create a reaching motion - move to a position, hold, return
                cycle_time = 8.0  # 8 second cycle
                phase = (t % cycle_time) / cycle_time
                
                # Different motion for different joints
                if i == 0:  # Base rotation
                    output[i] = 0.5 * np.sin(2 * np.pi * phase)  # Rotate back and forth
                elif i == 1:  # Shoulder
                    output[i] = 0.3 + 0.3 * np.sin(2 * np.pi * phase)
                elif i == 2:  # Elbow
                    output[i] = 0.5 * np.sin(2 * np.pi * phase)
                else:
                    output[i] = 0.2 * np.sin(2 * np.pi * phase * 2)  # Faster for wrist
            else:
                output[i] = 0.0


class ImageSaver(LeafSystem):
    """Save frames from one RgbdSensor color port to disk."""
 
    def __init__(self, camera_name, save_dir,
                 save_interval=0.5, width=CAM_WIDTH, height=CAM_HEIGHT):
        LeafSystem.__init__(self)
        self._camera_name = camera_name
        self._save_dir    = os.path.join(save_dir, camera_name)
        self._frame_count = 0
        self._width       = width
        self._height      = height
        os.makedirs(self._save_dir, exist_ok=True)
 
        self._image_port = self.DeclareAbstractInputPort(
            "color_image",
            AbstractValue.Make(ImageRgba8U(width, height))
        )
        self.DeclarePeriodicPublishEvent(save_interval, 0.0, self._save_image)
        print(f"  [{camera_name}] → '{self._save_dir}/' every {save_interval}s")
 
    def _save_image(self, context):
        try:
            image = self._image_port.Eval(context)
        except Exception as e:
            print(f"  [{self._camera_name}] WARNING: {e}")
            return
        img_array = np.frombuffer(image.data, dtype=np.uint8)
        img_array = img_array.reshape((self._height, self._width, 4))
        filename  = os.path.join(self._save_dir, f"frame_{self._frame_count:05d}.png")
        Image.fromarray(img_array[:, :, :3], mode="RGB").save(filename)
        self._frame_count += 1
        t = context.get_time()
        if self._frame_count % 10 == 0:
            print(f"  [{self._camera_name}] frame {self._frame_count} t={t:.2f}s")
 
 
# Cameras from URDF-frame-based attachment
 
def add_cameras_from_urdf(builder, plant, scene_graph, renderer_name,
                           save_dir="simulation_frames", save_interval=0.5):
    """
    Attach one RgbdSensor per camera using the exact body frame from the URDF.
 
    For each camera:
      1. Get the named frame from the plant (e.g. cam_high_color_optical_frame)
      2. Get the body that frame is welded to (the parent link)
      3. Compute X_BF: the fixed transform from that body's origin to the frame
      4. Pass parent body's frame_id + X_BF to RgbdSensor
 
    This means:
      - cam_high / cam_low: fixed to the robot frame → world-fixed
      - cam_left/right_wrist: fixed to follower link_6 → move with the arm
    """
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
        # Step 1: get the Drake frame object
        urdf_frame = plant.GetFrameByName(frame_name)
 
        # Step 2: get the body this frame lives on and its scene-graph frame id
        parent_body    = urdf_frame.body()
        parent_frame_id = plant.GetBodyFrameIdOrThrow(parent_body.index())
 
        # Step 3: fixed transform from body origin → optical frame
        #   GetFixedPoseInBodyFrame() walks the fixed joint chain and gives
        #   us X_BF without needing any context (it's purely kinematic/fixed)
        X_BF = urdf_frame.GetFixedPoseInBodyFrame()
 
        # Step 4: create sensor parented to the body frame
        sensor = builder.AddSystem(RgbdSensor(
            parent_id=parent_frame_id,
            X_PB=X_BF,               # pose of sensor IN the parent body frame
            color_camera=color_cam,
            depth_camera=depth_cam,
        ))
        sensor.set_name(f"rgbd_{cam_name}")
 
        builder.Connect(
            scene_graph.get_query_output_port(),
            sensor.query_object_input_port(),
        )
 
        saver = builder.AddSystem(
            ImageSaver(cam_name, save_dir, save_interval)
        )
        builder.Connect(
            sensor.color_image_output_port(),
            saver.get_input_port(0),
        )
 
        sensors[cam_name] = sensor
 
        # Print where this camera actually lives in world at home pose
        default_ctx = plant.CreateDefaultContext()
        X_WF = urdf_frame.CalcPoseInWorld(default_ctx)
        pos  = X_WF.translation()
        fwd  = X_WF.rotation().matrix() @ np.array([0, 0, 1])  # optical Z = forward
        print(f"  {cam_name:20s} frame='{frame_name}'")
        print(f"    world pos (home): [{pos[0]:.4f}, {pos[1]:.4f}, {pos[2]:.4f}]")
        print(f"    optical fwd (Z):  [{fwd[0]:.3f}, {fwd[1]:.3f}, {fwd[2]:.3f}]")
        print(f"    parent body:      '{parent_body.name()}'")
 
    return sensors


def main():
    # Load the robot model.
    builder = DiagramBuilder()
    plant, scene_graph = AddMultibodyPlantSceneGraph(builder, time_step=0.0)

    # Add a render engine to the scene graph (needed for camera rendering)
    renderer_name = "renderer"
    scene_graph.AddRenderer(renderer_name, MakeRenderEngineVtk(RenderEngineVtkParams()))
    
    # Parse the URDF model of the robot and add it to the plant
    model_indices = Parser(plant).AddModels("urdf/stationary_ai.urdf")
    # Add a small cube to interact with
    Parser(plant).AddModels("urdf/cube.urdf")

    cube_position = [0.1, 0.15, 0.02]
    drop_position = [0.1, -0.15, 0.02]  # Drop zone in front of right arm

    # Set cube position to be just above the table, in front of the left arm
    cube_body = plant.GetBodyByName("cube_link")
    X = RigidTransform()
    X.set_translation(cube_position)
    plant.SetDefaultFloatingBaseBodyPose(cube_body, X)
    plant.Finalize()

    # inspect_camera_frames(plant)  # Print camera-related frames to find names for Phase 2

    # Enable hydroelastic contact
    scene_graph_config = SceneGraphConfig()
    scene_graph_config.default_proximity_properties.compliance_type = "compliant"
    scene_graph.set_config(scene_graph_config)

# ── Add all 4 Trossen cameras ──
    print("\nAdding cameras:")
    cameras = add_cameras_from_urdf(
        builder, plant, scene_graph,
        renderer_name=renderer_name,
        save_dir="simulation_frames",
        save_interval=0.5,
    )

    # Set up meshcat visualization
    meshcat = StartMeshcat()
    visualization_config = VisualizationConfig()
    visualization_config.publish_proximity = True
    visualization_config.publish_period = np.inf
    ApplyVisualizationConfig(visualization_config, builder=builder, meshcat=meshcat)

    meshcat_config = yaml_load_file("meshcat_config.yaml")
    for p in meshcat_config["initial_properties"]:
        meshcat.SetProperty(p["path"], p["property"], p["value"])
    meshcat.SetCameraPose([0.9, 0.0, 0.9], [0.0, 0.0, 0.4])

    # Inspect all frames to find gripper-related ones
    # gripper_frames, carriage_frames = inspect_frames(plant)

    # Get the number of actuators
    nu = len(plant.GetJointActuatorIndices())
    print(f"Number of actuators: {nu}")

    # Actuator names in order
    # print(f"Joint actuators found:")
    # for actuator_index in plant.GetJointActuatorIndices():
    #     actuator = plant.get_joint_actuator(actuator_index)
    #     print(f"- {actuator.joint().name()}")

    # Creates instance of the controller and adds it to the diagram builder - outputs only 14 joints
    # trajectory_source = builder.AddSystem(DebugTrajectory(nu))

    controller = builder.AddSystem(PickAndPlaceController(plant, nu, cube_position, drop_position))

    # Converts position commands to position + velocity commands
    # Robot's input port expects [position, velocities] (28 values total for 14 joints)
    state_interpolator = builder.AddSystem(StateInterpolatorWithDiscreteDerivative(nu, 0.01, True)) # 0.01 - time constant (how fast to compute velocities), True - suppresses initial velocity spike
        # Add image saver — saves a frame every 0.5 seconds

    builder.Connect(controller.get_output_port(0), state_interpolator.get_input_port())

    # Connect the controller to the plant's desired state input port.
    # builder.Connect(trajectory_source.get_output_port(0),state_interpolator.get_input_port())
    builder.Connect(
        state_interpolator.get_output_port(),
        plant.get_desired_state_input_port(model_indices[0]),
    )

    # Build daigram
    diagram = builder.Build()
    context = diagram.CreateDefaultContext()

    # Get plant context from diagram
    plant_context = plant.GetMyContextFromRoot(context)

    # Print initial gripper positions
    # print("INITIAL GRIPPER POSITIONS")
    
    # for frame_name in gripper_frames:
    #     try:
    #         frame = plant.GetFrameByName(frame_name)
    #         pose = frame.CalcPoseInWorld(plant_context)
    #         pos = pose.translation()
    #         print(f"  {frame_name:30s} -> ({pos[0]:.3f}, {pos[1]:.3f}, {pos[2]:.3f})")
    #     except Exception as e:
    #         print(f"  {frame_name:30s} -> ERROR: {e}")
    
    # print("CARRIAGE POSITIONS (these move with gripper)")
    
    # for frame_name in carriage_frames:
    #     try:
    #         frame = plant.GetFrameByName(frame_name)
    #         pose = frame.CalcPoseInWorld(plant_context)
    #         pos = pose.translation()
    #         print(f"  {frame_name:30s} -> ({pos[0]:.3f}, {pos[1]:.3f}, {pos[2]:.3f})")
    #     except Exception as e:
    #         print(f"  {frame_name:30s} -> ERROR: {e}")

    # Set up the simulator to use CENIC
    simulator = Simulator(diagram, context)
    config = SimulatorConfig()
    config.integration_scheme = "cenic"
    config.accuracy = 1e-3
    config.max_step_size = 0.1
    config.use_error_control = True
    config.publish_every_time_step = True
    ApplySimulatorConfig(config, simulator)
    simulator.set_target_realtime_rate(1.0)
    simulator.Initialize()

    # Run the simulation.
    input("Waiting for meshcat... press [ENTER] to start simulating.")
    print("")
    print("Watch for movement in MeshCat!")
    print("Press Ctrl+C to stop the simulation.")

    try:
        simulator.AdvanceTo(np.inf)
    except KeyboardInterrupt:
        EventStatus.Killed(diagram, "Simulation stopped by user.")

    # Print final positions
    # print("FINAL GRIPPER POSITIONS")
    
    # Update plant context
    # plant_context = plant.GetMyContextFromRoot(simulator.get_context())
    
    # for frame_name in gripper_frames:
    #     try:
    #         frame = plant.GetFrameByName(frame_name)
    #         pose = frame.CalcPoseInWorld(plant_context)
    #         pos = pose.translation()
    #         print(f"  {frame_name:30s} -> ({pos[0]:.3f}, {pos[1]:.3f}, {pos[2]:.3f})")
    #     except Exception as e:
    #         print(f"  {frame_name:30s} -> ERROR: {e}")

if __name__ == "__main__":
    main()
