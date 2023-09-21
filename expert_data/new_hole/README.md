# stl name
hole_new
# start pose
```
self.peg_pose=[-0.2,0,0.3] 
self.hole_pose=[0.7,-0.2,0.3] 
```
# task pose
```
peg_in(robot=env.robot,peg_pose=[0.32,-0.005,0S056],hole_pose=[0.54,-0.2,0.185])
```

# task function
```
def peg_in(robot,peg_pose,hole_pose,tolerance=0.001,success=True):
   rospy.loginfo('Execute peg in hole tast...')
   rospy.loginfo('Go back to initial pose')
   success&=robot.reach_gripper_position(0)
   success&=robot.move(pose=[peg_pose[0],peg_pose[1],peg_pose[2]], tolerance=0.0001)
   rospy.loginfo('Arrive peg pose, perpare for grabing peg...')
   success&=robot.reach_gripper_position(0.46) # 465
   time.sleep(2)
   success&=robot.move(pose=[peg_pose[0],peg_pose[1],peg_pose[2]+0.1], tolerance=0.001)
   rospy.loginfo('Start to peg in...')
   success&=robot.move(pose=[hole_pose[0],hole_pose[1],hole_pose[2]], tolerance=0.001)
   x = hole_pose[0]+0.08
   time.sleep(1)
   success&=robot.move(pose=[x,hole_pose[1],hole_pose[2]], tolerance=0.001)
   success&=robot.reach_gripper_position(0)
   time.sleep(0.5)
   print(success)
   return success
```


