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
    AddDefaultVisualization,
)

# Initialize the meshcat visualizer.
meshcat = StartMeshcat()

# Set up the system diagram.
builder = DiagramBuilder()
plant, _ = AddMultibodyPlantSceneGraph(builder, time_step=0.005)
Parser(plant).AddModels("urdf/stationary_ai.urdf")
plant.Finalize()

AddDefaultVisualization(builder, meshcat)

diagram = builder.Build()
context = diagram.CreateDefaultContext()

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