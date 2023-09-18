# KinovaArm PegInsertTask with DRL
Train Kinova Gen3 arm in Gazebo with deep reinforcement learning

source devel/setup.bash
roslaunch kortex_driver kortex_driver.launch gripper:=robotiq_2f_85
roslaunch kortex_gazebo spawn_kortex_robot.launch gripper:=robotiq_2f_85

source devel/setup.bash
roslaunch kortex_examples moveit_example.launch

Gen3env文件夹里的gen3env.py是gazebo，使用的是envrobot.py
主文件夹的gen3env.py是实体机械臂，使用的是robot.py