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
)
from pydrake.common.yaml import yaml_load_file


# Load the robot model.
builder = DiagramBuilder()
plant, scene_graph = AddMultibodyPlantSceneGraph(builder, time_step=0.0)
model_indices = Parser(plant).AddModels("urdf/stationary_ai.urdf")
# print(f"model_indices: {model_indices[0]}")

# Add a small cube to interact with, and set it's default pose to be just above the table.
Parser(plant).AddModels("urdf/cube.urdf")
cube_body = plant.GetBodyByName("cube_link")
X = RigidTransform()
X.set_translation([0.0, 0.0, 0.02])
plant.SetDefaultFloatingBaseBodyPose(cube_body, X)
plant.Finalize()

# Enable hydroelastic contact.
scene_graph_config = SceneGraphConfig()
scene_graph_config.default_proximity_properties.compliance_type = "compliant"
scene_graph.set_config(scene_graph_config)

# Set up meshcat visualization.
meshcat = StartMeshcat()
visualization_config = VisualizationConfig()
visualization_config.publish_proximity = True
visualization_config.publish_period = np.inf
ApplyVisualizationConfig(visualization_config, builder=builder, meshcat=meshcat)

meshcat_config = yaml_load_file("meshcat_config.yaml")
for p in meshcat_config["initial_properties"]:
    meshcat.SetProperty(p["path"], p["property"], p["value"])
meshcat.SetCameraPose([0.9, 0.0, 0.9], [0.0, 0.0, 0.4])

# Custom controller by inheriting from LeafSystem
class DebugTrajectory(LeafSystem):
    """Trajectory that prints its output to verify it's working."""
    
    def __init__(self, num_joints):
        LeafSystem.__init__(self)
        self._num_joints = num_joints
        # self._counter = 0        
        self.DeclareVectorOutputPort(
            "desired_positions", # Name of the output port
            num_joints,
            self.CalcOutput      # Callback function to calculate the output values
        )
        # print(f"Trajectory created for {num_joints} joints")
    
    def CalcOutput(self, context, output):  # context: contains simulation state, output: array to fill with joint commands
        t = context.get_time()
       
        # Simple trajectory: cosine wave that changes over time
        for i in range(self._num_joints):
            # Simple oscillation
            output[i] = 0.5 * np.cos(t)  # All joints oscillate together at an amplitude 0.5 radians and speed 1 since cos(tx1)

        # Different joints, different motions
        # for i in range(self._num_joints):
        #     output[i] = 0.5 * np.sin(t * (i+1)/5)  # Each joint moves at different speed

        # Only move first 3 joints
        # for i in range(self._num_joints):
        #     if i < 3:
        #         output[i] = 0.3 * np.sin(t)
        #     else:
        #         output[i] = 0.0
        
# Get the number of actuators
nu = len(plant.GetJointActuatorIndices())
print(f"Number of actuators: {nu}")
print(f"Joint actuators found:")
for actuator_index in plant.GetJointActuatorIndices():
    actuator = plant.get_joint_actuator(actuator_index)
    print(f"- {actuator.joint().name()}")

# Creates instance of the controller and adds it to the diagram builder - outputs only 14 joints
trajectory_source = builder.AddSystem(DebugTrajectory(nu))
# Converts position commands to position + velocity commands
# Robot's input port expects [position, velocities] (28 values total for 14 joints)
state_interpolator = builder.AddSystem(StateInterpolatorWithDiscreteDerivative(nu, 0.01, True)) # 0.01 - time constant (how fast to compute velocities), True - suppresses initial velocity spike

# Connect the sliders to the plant's desired state input port.
builder.Connect(trajectory_source.get_output_port(0),state_interpolator.get_input_port())
builder.Connect(
    state_interpolator.get_output_port(),
    plant.get_desired_state_input_port(model_indices[0]),
)

diagram = builder.Build()
context = diagram.CreateDefaultContext()

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
