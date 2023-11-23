the environment of Kinova Gen3 arm in Gazebo and Robot
# how to use
## in Gazebo
### introduction

1. 环境：`./Gen3env/gen3env.py`and`./envrobot.py`
### code to use

1. `source devel/setup.bash`
2. `roslaunch kortex_gazebo spawn_kortex_robot.launch gripper:=robotiq_2f_85`
## in Robot
### introduction

1. 环境：`./gen3env.py`and`./robot.py`
### code to use

1. `source devel/setup.bash`
2. `roslaunch kortex_driver kortex_driver.launch gripper:=robotiq_2f_85`
# how to run

1. `source devel/setup.bash`
2. `roslaunch kortex_examples moveit_example.launch`
