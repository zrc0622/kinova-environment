# Kinova Gen3 Arm Peg Environment
the environment of Kinova Gen3 Arm Peg in Gazebo and Robot

## 运行流程
1. 先打开对应的[环境](#打开环境)
2. 在执行相应的功能，包括[采集模仿学习专家数据](#采集专家数据)(从环境中或从实体中)和[测试模型训练好的模型](#测试模型)

## 打开环境
### Gazebo环境
> 在Gazebo仿真环境上测试
1. 环境代码：`./Gen3env/gen3env.py`and`./envrobot.py`
2. 执行命令
   ```bash
   source devel/setup.bash
   roslaunch kortex_gazebo spawn_kortex_robot.launch gripper:=robotiq_2f_85
   ```

### Robot环境
> 在实体机械臂上测试
1. 环境代码：`./gen3env.py`and`./robot.py`
2. 执行命令
   ```bash
   source devel/setup.bash
   roslaunch kortex_gazebo spawn_kortex_robot.launch gripper:=robotiq_2f_85
   ```
## 采集专家数据
1. 代码
   1. 从Gazebo中采集：`./example_move_it_trajectories_env_get_data2.py`
   2. 从实体中采集：`./example_move_it_trajectories_robot_get_data.py`
2. 执行命令
   ```bash
   source devel/setup.bash
   roslaunch kortex_examples moveit_example.launch
   ```

## 测试模型
> 测试[模仿学习](https://github.com/zrc0622/kinova-imitation-learning)中训练的模型
1. 测试代码：`./example_move_it_trajectories_test_model.py`
2. 执行命令
   ```bash
   source devel/setup.bash
   roslaunch kortex_examples moveit_example.launch
   ```

