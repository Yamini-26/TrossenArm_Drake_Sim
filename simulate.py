#!/usr/bin/env python

##
#
# Run a simple automatic controller simulation of the Trossen Stationary bimanual robot.
#
##

from typing import List
from functools import partial

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
)
from pydrake.common.yaml import yaml_load_file


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


def main():
    # Load the robot model.
    builder = DiagramBuilder()
    plant, scene_graph = AddMultibodyPlantSceneGraph(builder, time_step=0.0)
    
    # Parse the URDF model of the robot and add it to the plant
    model_indices = Parser(plant).AddModels("urdf/stationary_ai.urdf")
    # Add a small cube to interact with
    Parser(plant).AddModels("urdf/cube.urdf")

    # Set cube position to be just above the table, in front of the right arm
    cube_body = plant.GetBodyByName("cube_link")
    X = RigidTransform()
    X.set_translation([0.1, 0.15, 0.02])
    plant.SetDefaultFloatingBaseBodyPose(cube_body, X)

    plant.Finalize()

    # Enable hydroelastic contact
    scene_graph_config = SceneGraphConfig()
    scene_graph_config.default_proximity_properties.compliance_type = "compliant"
    scene_graph.set_config(scene_graph_config)

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
    gripper_frames, carriage_frames = inspect_frames(plant)

    # Get the number of actuators
    nu = len(plant.GetJointActuatorIndices())
    print(f"Number of actuators: {nu}")

    # Actuator names in order
    print(f"Joint actuators found:")
    for actuator_index in plant.GetJointActuatorIndices():
        actuator = plant.get_joint_actuator(actuator_index)
        print(f"- {actuator.joint().name()}")

    # Creates instance of the controller and adds it to the diagram builder - outputs only 14 joints
    trajectory_source = builder.AddSystem(DebugTrajectory(nu))

    # Converts position commands to position + velocity commands
    # Robot's input port expects [position, velocities] (28 values total for 14 joints)
    state_interpolator = builder.AddSystem(StateInterpolatorWithDiscreteDerivative(nu, 0.01, True)) # 0.01 - time constant (how fast to compute velocities), True - suppresses initial velocity spike

    # Connect the controller to the plant's desired state input port.
    builder.Connect(trajectory_source.get_output_port(0),state_interpolator.get_input_port())
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
    print("INITIAL GRIPPER POSITIONS")
    
    for frame_name in gripper_frames:
        try:
            frame = plant.GetFrameByName(frame_name)
            pose = frame.CalcPoseInWorld(plant_context)
            pos = pose.translation()
            print(f"  {frame_name:30s} -> ({pos[0]:.3f}, {pos[1]:.3f}, {pos[2]:.3f})")
        except Exception as e:
            print(f"  {frame_name:30s} -> ERROR: {e}")
    
    print("CARRIAGE POSITIONS (these move with gripper)")
    
    for frame_name in carriage_frames:
        try:
            frame = plant.GetFrameByName(frame_name)
            pose = frame.CalcPoseInWorld(plant_context)
            pos = pose.translation()
            print(f"  {frame_name:30s} -> ({pos[0]:.3f}, {pos[1]:.3f}, {pos[2]:.3f})")
        except Exception as e:
            print(f"  {frame_name:30s} -> ERROR: {e}")

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
    print("FINAL GRIPPER POSITIONS")
    
    # Update plant context
    plant_context = plant.GetMyContextFromRoot(simulator.get_context())
    
    for frame_name in gripper_frames:
        try:
            frame = plant.GetFrameByName(frame_name)
            pose = frame.CalcPoseInWorld(plant_context)
            pos = pose.translation()
            print(f"  {frame_name:30s} -> ({pos[0]:.3f}, {pos[1]:.3f}, {pos[2]:.3f})")
        except Exception as e:
            print(f"  {frame_name:30s} -> ERROR: {e}")

if __name__ == "__main__":
    main()
