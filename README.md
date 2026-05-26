# Trossen Arm Drake

A Drake-compatible model of the bimanual [Trossen Stationary
AI](https://www.trossenrobotics.com/stationary-ai) (formerly Aloha 2) robot with
[hydroelastic contact](https://arxiv.org/abs/2110.04157) and 
[CENIC](https://arxiv.org/abs/2511.08771) error-controlled integration.

![](img/trossen_drake_screenshot.png)

Adopted from the [Trossen Arm
Description](https://github.com/TrossenRobotics/trossen_arm_description).

## Installation

Clone this repository:

```bash
git clone https://github.com/vincekurtz/trossen_arm_drake.git
cd trossen_arm_drake
```

Create a virtual environment and install dependencies:

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Usage

Activate the virtual environment:

```bash
source .venv/bin/activate
```

View the model in MeshCat (with hydroelastics enabled):

```bash
python -m pydrake.visualization.model_visualizer urdf/stationary_ai.urdf --compliance_type compliant
```

Run an interactive simulation (use MeshCat sliders to control the robot):

```bash
./simulate.py
```

<!-- INITIAL GRIPPER POSITIONS
  follower_left_gripper_left     -> (0.003, 0.273, 0.183)
  follower_left_gripper_right    -> (-0.043, 0.273, 0.183)
  follower_left_ee_gripper_link  -> (-0.020, 0.204, 0.183)
  follower_right_gripper_left    -> (-0.043, -0.273, 0.183)
  follower_right_gripper_right   -> (0.003, -0.273, 0.183)
  follower_right_ee_gripper_link -> (-0.020, -0.204, 0.183)
  follower_left_left_gripper_joint_parent -> (0.003, 0.273, 0.183)
  follower_left_right_gripper_joint_parent -> (-0.043, 0.273, 0.183)
  follower_left_ee_gripper_parent -> (-0.020, 0.204, 0.183)
  follower_right_left_gripper_joint_parent -> (-0.043, -0.273, 0.183)
  follower_right_right_gripper_joint_parent -> (0.003, -0.273, 0.183)
  follower_right_ee_gripper_parent -> (-0.020, -0.204, 0.183)
CARRIAGE POSITIONS (these move with gripper)
  follower_left_carriage_right   -> (-0.043, 0.273, 0.183)
  follower_left_carriage_left    -> (0.003, 0.273, 0.183)
  follower_right_carriage_right  -> (0.003, -0.273, 0.183)
  follower_right_carriage_left   -> (-0.043, -0.273, 0.183)
  follower_left_right_carriage_joint_parent -> (-0.043, 0.273, 0.183)
  follower_left_left_carriage_joint_parent -> (0.003, 0.273, 0.183)
  follower_right_right_carriage_joint_parent -> (0.003, -0.273, 0.183)
  follower_right_left_carriage_joint_parent -> (-0.043, -0.273, 0.183)
  follower_left_right_carriage_joint_follower_left_right_carriage_joint_parent_F -> (-0.043, 0.273, 0.183)
  follower_left_right_carriage_joint_follower_left_carriage_right_M -> (-0.043, 0.273, 0.183)
  follower_right_right_carriage_joint_follower_right_right_carriage_joint_parent_F -> (0.003, -0.273, 0.183)
  follower_right_right_carriage_joint_follower_right_carriage_right_M -> (0.003, -0.273, 0.183) -->
