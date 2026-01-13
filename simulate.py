#!/usr/bin/env python

##
#
# Run a simple simulation of the Trossen Stationary bimanual robot arms.
#
##

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
Parser(plant).AddModels("urdf/stationary_ai.urdf")
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

# Set up the simulator.
diagram = builder.Build()
simulator = Simulator(diagram)
simulator.set_target_realtime_rate(1.0)
simulator.Initialize()

# Run the simulation.
input("Waiting for meshcat... press [ENTER] to continue.")
meshcat.StartRecording()
simulator.AdvanceTo(10.0)
meshcat.StopRecording()
meshcat.PublishRecording()
input("Simulation complete. Press [ENTER] to exit.")