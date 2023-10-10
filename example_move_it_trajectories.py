#!/usr/bin/python3

import sys
sys.path.append('/catkin_workspace/src/ros_kortex/kortex_examples/src/move_it')
import time
import rospy
import moveit_commander
import moveit_msgs.msg
from gazebo_msgs.srv import *
from geometry_msgs.msg import *
from math import pi
from control_msgs.msg import *
from trajectory_msgs.msg import *
import actionlib
from std_srvs.srv import Empty
from tf import TransformListener
from robot import Robot
# from task import peg_in
from gen3env import gen3env as RobotEnv
import gym
import torch.nn as nn
from network import MLPModel, LSTMModel
import torch
from Gen3Env.gen3env import gen3env
from stable_baselines3 import TD3
import os
import numpy as np

def normalize_data(data):
    min_vals = np.array([0.299900302, -0.17102845, 0.05590736, -0.000087572115])
    max_vals = np.array([0.58015204, 0.00020189775, 0.299989649, 0.36635616])
    normalized_data = (data - min_vals) / (max_vals - min_vals)
    return normalized_data

def origin_data(data):
    min_vals = np.array([0.299900302, -0.17102845, 0.05590736, -0.000087572115])
    max_vals = np.array([0.58015204, 0.00020189775, 0.299989649, 0.36635616])
    origin_data_data = (max_vals - min_vals) * data + min_vals
    return origin_data_data

def rl_train():
   env=gym.make(id='peg_in_hole-v0')
   env.reset()
   log_path='./log'
   if not os.path.exists(log_path):
        os.makedirs(log_path)
  #  print(torch.cuda.is_available())
   if torch.cuda.is_available():
        print('cuda is available, train on GPU!')
   model=TD3('MlpPolicy', env, verbose=1,tensorboard_log=log_path,device='cuda')
   model.learn(total_timesteps=1000)

def bc_run():
    # env=gym.make(id='Pendulum-v1')
    model_name = 'lstm'
    run_env = 'env'
    
    frame = 5
    episodes = 10
    steps = 300
    model_path = '/catkin_workspace/src/ros_kortex/kortex_examples/src/move_it/model/52.pth'

    if run_env == 'env':
        env=gym.make(id='peg_in_hole-v0')
    
    if run_env == 'robot':
        env=RobotEnv()

    if model_name == 'mlp':
        model = MLPModel(frame*4, 4)
        model.load_state_dict(torch.load(model_path))

        # # absolute
        # for episode in range(10):
        #   print("episode: {}".format(episode))
        #   env.robot.move(pose=[0.3, 0, 0.3])
        #   obs = env.reset()
        #   obs = np.array([[0.3, 0, 0.3, 0],[0.3, 0, 0.3, 0],[0.3, 0, 0.3, 0],[0.3, 0, 0.3, 0]])
        #   # print(type(obs))
        #   # print(obs)
        #   model_obs = obs[:4]
        #   done = False
        #   # while not done:
        #   for step in range(2):
        #     with torch.no_grad():
        #       action = model(torch.Tensor(model_obs)).tolist()
        #       # print(type(action))
        #     next_obs,reward,done,_=env.step(action=action)
        #     # print('reward={}'.format(reward))
        #     model_obs = next_obs[:4]
        # env.robot.move(pose=[0.3, 0, 0.3])

        # new absolute
        for episode in range(episodes):
            print("episode: {}".format(episode))
            obs = env.reset()
            env.robot.move(pose=[0.3, 0, 0.3], tolerance=0.0001) # 前后 左右 上下
            env.robot.reach_gripper_position(0)
            obs = env.get_obs()
            obs = normalize_data(np.array(obs[:4]))
            obs = np.tile(obs, (frame, 1))
            obs = obs.flatten()
            with torch.no_grad():
                for step in range(steps):
                    print(f'step: {step}')
                    input_tensor = torch.Tensor(obs)
                    output_tensor = model(input_tensor)
                    action = output_tensor.tolist()
                    action = origin_data(action)
                    
                    env.robot.move(pose=action[:3])
                    env.robot.reach_gripper_position(action[-1])
                    
                    obs = obs[4:]
                    next_obs = env.get_obs()
                    next_obs = normalize_data(np.array(next_obs[:4]))

                    if frame !=1:
                        obs = np.concatenate((obs, next_obs))

        # # delt
        # for episode in range(10):
        #   print("episode: {}".format(episode))
        #   env.robot.move(pose=[0.5, 0, 0.5])
        #   print("start")
        #   obs = env.reset()
        #   model_obs = obs[:4]
        #   done = False
        #   # while not done:
        #   for step in range(2):
        #     with torch.no_grad():
        #       action = model(torch.Tensor(model_obs)).tolist() + model_obs
        #     next_obs,reward,done,_=env.step(action=action)
        #     # print('reward={}'.format(reward))
        #     model_obs = next_obs[:4]
        # env.robot.move(pose=[0.5, 0, 0.5])
  
    if model_name == 'lstm':
        model = LSTMModel(4, 128, 4)
        model.load_state_dict(torch.load(model_path))

        for episode in range(episodes):
            print(f"episode: {episode}")
            obs = env.reset()
            env.robot.move(pose=[0.3, 0, 0.3], tolerance=0.0001) # 前后 左右 上下
            env.robot.reach_gripper_position(0)
            obs = env.get_obs()
            obs = normalize_data(np.array(obs[:4]))

            with torch.no_grad():
                hidden = None
                for step in range(steps):
                    print(f'step: {step}')
                    input_tensor = torch.tensor(obs, dtype=torch.float32).unsqueeze(0).unsqueeze(0)
                    output_tensor, hidden = model(input_tensor, hidden) # (1,1,4)
                    action = output_tensor.squeeze().squeeze() # (4,)
                    action = action.tolist()
                    action = origin_data(action)
                    
                    print('1',action)
                    env.robot.move(pose=action[:3], tolerance=0.0001)

                    if step<55:
                        env.robot.reach_gripper_position(action[-1]/0.78)
                    # env.robot.reach_gripper_position(0.3)
                
                    # if step < 100:
                    #     env.robot.reach_gripper_position(action[-1]+0.1)

                    next_obs = env.get_obs()
                    next_obs = np.array(next_obs)

                    aaa=next_obs[:4]
                    print('2',aaa)

                    next_obs = normalize_data(next_obs[:4])

                    obs = next_obs
def test_gripper():
    env=gym.make('peg_in_hole-v0')
    env.reset()
    # env.robot.reach_gripper_position(0.6)
    # gripper_pos=env.robot.get_gripper_position()
    # print('='*50)
    # print(gripper_pos)

if __name__ == '__main__':
    bc_run()
    # test_gripper()
