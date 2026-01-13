#!/usr/bin/env python

##
#
# Run a simple simulation of the Trossen Stationary bimanual robot arms.
#
##

import numpy as np
from pydrake.all import (
    StartMeshcat,
    DiagramBuilder,
    AddMultibodyPlantSceneGraph,
    Parser,
    Simulator,
    SceneGraphConfig,
    ApplyVisualizationConfig,
    VisualizationConfig,
)

# Load the robot model.
builder = DiagramBuilder()
plant, scene_graph = AddMultibodyPlantSceneGraph(builder, time_step=0.005)
model_indices = Parser(plant).AddModels("urdf/stationary_ai.urdf")
plant.Finalize()

# Enable hydroelastic contact.
scene_graph_config = SceneGraphConfig()
scene_graph_config.default_proximity_properties.compliance_type = "compliant"
scene_graph.set_config(scene_graph_config)

# Set up meshcat visualization.
meshcat = StartMeshcat()
visualization_config = VisualizationConfig()
visualization_config.publish_proximity = True
ApplyVisualizationConfig(visualization_config, builder=builder, meshcat=meshcat)

# Build the system diagram
diagram = builder.Build()
context = diagram.CreateDefaultContext()
plant_context = plant.GetMyMutableContextFromRoot(context)

# Fix joint targets for the implicit PD controllers in each joint
model_instance_index = model_indices[0]
nu = plant.num_actuators(model_instance_index)
q_desired = np.zeros(nu)
v_desired = np.zeros(nu)
x_desired = np.concatenate([q_desired, v_desired])
plant.get_desired_state_input_port(model_instance_index).FixValue(
    plant_context, x_desired
)

# Set up the simulator.
simulator = Simulator(diagram, context)
simulator.set_target_realtime_rate(1.0)
simulator.Initialize()

# Run the simulation.
input("Waiting for meshcat... press [ENTER] to continue.")
meshcat.StartRecording()
simulator.AdvanceTo(10.0)
meshcat.StopRecording()
meshcat.PublishRecording()
input("Simulation complete. Press [ENTER] to exit.")