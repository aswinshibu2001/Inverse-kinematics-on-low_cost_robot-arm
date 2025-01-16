import time
import mujoco
import mujoco.viewer
import numpy as np



m = mujoco.MjModel.from_xml_path('simulation/low_cost_robot/scene.xml')
d = mujoco.MjData(m)
# m.opt.noslip_iterations=3
# m.opt.gravity[:]=[0, 0 ,0]


def inverse_kinematics( ee_target_pos, joint_name='end_effector'):
  #print(ee_target_pos)
  joint_id = m.geom(joint_name).id
  ee_pos =d.geom_xpos[joint_id]
  jac = np.zeros((3, m.nv))
  mujoco.mj_jacGeom(m, d, jac, None, joint_id)
  qpos = d.qpos[:5]# getting the current joint angles of all the joints
  q_delta = np.dot(np.linalg.pinv(jac[:, :5]), ee_target_pos - ee_pos)
  q_target_pos = qpos + q_delta * 0.2
  return q_target_pos



with mujoco.viewer.launch_passive(m, d) as viewer:
  # start = time.time()
  i=0
  while viewer.is_running():
  
    # step_start = time.time()
    endeff=m.geom("end_effector").id
    end_effector_pos = d.geom_xpos[endeff]
    red_box=m.body("box").id
    red_box_pos = d.xpos[red_box].copy()
    
    red_box_pos[0]+=0.0075
    
    # print(red_box_pos)
    # print(end_effector_pos)
    # exit()
    
 
    err=np.linalg.norm(red_box_pos-end_effector_pos)
    # print(f"initial error: {err}")
    # exit()
    tolerance=0.01
    while err>tolerance:
      i+=1
     
      current_pos=inverse_kinematics(red_box_pos)
      # print(current_pos.shape)
      # exit()
      d.ctrl[:]  = current_pos
      
      
      if err<0.07 :
        d.ctrl[4]=-1.710
        
      
      # print(d.ctrl)
      mujoco.mj_forward(m,d)
      mujoco.mj_step(m, d)
      end_effector_pos = d.geom_xpos[endeff]
      # print(f"new end_effector_pos : {end_effector_pos}")
      err=np.linalg.norm(red_box_pos-end_effector_pos)
      # print(f"new error {i} : {err}") c
      print(f"err: {err} end effector: {end_effector_pos}, red box: {red_box_pos}")
      viewer.sync()
      time.sleep(0.01)
      # if err<0.015+0.010:
      #   box_orien=d.xquat[red_box]
      #   d.xquat[endeff]=box_orien
      
      #   # mujoco.mj_step(m, d)
      #   print(f"box orien {box_orien}")
      #   print(f"end eff {d.xquat[endeff]}")
        
        
    mujoco.mj_step(m, d)
    
    
    print("moving to new target with a pause")
    time.sleep(1)
    
    new_target=[-0.06,0.18,0.069]
    current_pos = d.geom_xpos[endeff]  # Start position of the end-effector
    err=np.linalg.norm(new_target-current_pos)
  
    t=0
    
    while err>0.005:
       
        t+=1
        current_pos=inverse_kinematics(new_target)
        print(f"t= {t}, err{err} redbox : {d.xpos[red_box]}")
        d.ctrl[:]=current_pos
        d.ctrl[4]=-0.250  
        # print(d.xquat[endeff])
    
        mujoco.mj_step(m, d)
        end_effector_pos = d.geom_xpos[endeff]
        err=np.linalg.norm(new_target-end_effector_pos)
        viewer.sync()
        time.sleep(0.05) 
    time.sleep(5)
    
    
    
    mujoco.mj_step(m, d)
    # time_until_next_step = m.opt.timestep - (time.time() - step_start)
    # if time_until_next_step > 0:
    #   time.sleep(time_until_next_step)