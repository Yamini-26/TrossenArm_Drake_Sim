#!/usr/bin/env python

##
#
# Run a simple interactive simulation of the Trossen Stationary bimanual robot.
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
    SceneGraphConfig,
    ApplyVisualizationConfig,
    VisualizationConfig,
    Meshcat,
    LeafSystem,
    Multiplexer,
    ConstantVectorSource,
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

# Add joint sliders to meshcat for setting desired joint angles.
slider_names = []
for actuator_index in plant.GetJointActuatorIndices():
    actuator = plant.get_joint_actuator(actuator_index)
    if actuator.has_controller():
        name = actuator.joint().name()
        lower_limit = actuator.joint().position_lower_limits()[0]
        upper_limit = actuator.joint().position_upper_limits()[0]
        default = actuator.joint().default_positions()[0]
        meshcat.AddSlider(
            name=name, min=lower_limit, max=upper_limit, step=0.1, value=default
        )
        slider_names.append([name])
meshcat.AddButton("Stop Simulation")

# Add a little controller to send the slider values as joint position targets.
class MeshcatSliders(LeafSystem):
    """A system that outputs the values from meshcat sliders.

    An output port is created for each element in the list `slider_names`.
    Corresponding sliders with these names must have *already* been added to
    Meshcat via Meshcat.AddSlider().

    Adopted from https://github.com/RussTedrake/underactuated.
    """

    def __init__(self, meshcat: Meshcat, slider_names: List[str]):
        LeafSystem.__init__(self)

        self._meshcat = meshcat
        self._sliders = slider_names
        for i, slider_iterable in enumerate(self._sliders):
            port = self.DeclareVectorOutputPort(
                f"slider_group_{i}",
                len(slider_iterable),
                partial(self._DoCalcOutput, port_index=i),
            )
            port.disable_caching_by_default()

    def _DoCalcOutput(self, context, output, port_index):
        for i, slider in enumerate(self._sliders[port_index]):
            output[i] = self._meshcat.GetSliderValue(slider)

nu = len(slider_names)
assert nu == plant.num_actuators(model_indices[0]), (
    "Number of sliders must match number of actuated joints."
)
sliders = builder.AddSystem(MeshcatSliders(meshcat, slider_names))
q_desired = builder.AddSystem(Multiplexer(nu))
v_desired = builder.AddSystem(ConstantVectorSource(np.zeros(nu)))
x_desired = builder.AddSystem(Multiplexer([nu, nu]))

# Connect the sliders to the plant's desired state input port.
for i in range(nu):
    builder.Connect(sliders.get_output_port(i), q_desired.get_input_port(i))
builder.Connect(q_desired.get_output_port(), x_desired.get_input_port(0))
builder.Connect(v_desired.get_output_port(), x_desired.get_input_port(1))
builder.Connect(
    x_desired.get_output_port(),
    plant.get_desired_state_input_port(model_indices[0]),
)

# Set up the simulator.
diagram = builder.Build()
simulator = Simulator(diagram)
simulator.set_target_realtime_rate(1.0)
simulator.Initialize()

# Run the simulation.
input("Waiting for meshcat... press [ENTER] to continue.")
print("")
print("Use the meshcat sliders to control the robot.")
print("Press the 'Stop Simulation' button to quit.")
while meshcat.GetButtonClicks("Stop Simulation") < 1:
    simulator.AdvanceTo(simulator.get_context().get_time() + 0.1)
