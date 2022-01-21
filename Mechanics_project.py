"""
PROJECT DESCRIPTION

Assess the interanl forces and torques applying in the joints of a climber 
during a fall with a security rope. 
The project included the study of dynamics with Python, Robotran and Tracker.

"""
#==============================
# Group MECA 4                #
#                             #  
# Barbolani Fabrizio          # 
#                             #
# LSM 2021-2022               #
#==============================

# =============================================================================
# Table of Content
# =============================================================================

# 0. User Parameter
# 1. Packages Loading
# 2. Project Loading
# 3. Definitions
# 4. Tracker Experiment
# 5. Robotran Model
# 6. Inverse Kinematics
# 7. Inverse Dynamics
# 8. Results 1
# 9. Second Optimisation
# 10. Results 2
# 11. Analysis of different falls
# 12. Energy analysis


# =============================================================================
# 0. Running Parameters
# =============================================================================

#Choose one
Fall_1 = True #1m fall
Fall_2 = False #1.5m fall
Fall_3 = False #2m fall
Fall_All = False #Comparison of falls ( est. running time = 8 min. )

#Choose one
Optimisation_ExtForceParameters = True
Optimisation_Masses = False

# =============================================================================
# 1. Packages Loading
# =============================================================================
try:
    import MBsysPy as Robotran
except:
    raise ImportError("MBsysPy not found/installed."
                      "See: https://www.robotran.eu/download/how-to-install/"
                      )
    
    
import scipy.optimize
import math
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from splineqdqdd import compute_q_qd_qdd
from MBsysPy import MbsSensor
from numpy import savetxt

#==============================================================================
# 2. Project Loading
#==============================================================================

mbs_data = Robotran.MbsData("../dataR/Project_freefall_climbing.mbs")
# The loading should point to your mbs file 
# Thus, the name here above should change depending on your MBsysPad project name
print(mbs_data)

#==============================================================================
# 3. Definition
#==============================================================================

names_sensors = [ "Head", "Shoulder", "Elbow", "Wrist", "Finger", "Pelvis", "Knee", "Ankle", "Foot"]
names_joints = ["Free T1", "Free T3", "Free R2",
                "Pelvis T1", "Pelvis T3", "Pelvis R2",
                "Thigh T1", "Thigh T3", "Thigh R2",
                "Knee T1", "Knee T3", "Knee R2", 
                "Ankle T1", "Ankle T3", "Ankle R2",
                "Shoulder T1", "Shoulder T3", "Shoulder R2",
                "Elbow T1", "Elbow T3", "Elbow R2",
                "Wrist T1", "Wrist T3", "Wrist R2",
                "Neck T1", "Neck T3", "Neck R2"]

#==============================================================================
# 4. Tracker Experiment
#==============================================================================

# 4.1 Load experiments data
#==========================

if Fall_1 == True:
    range_data = range(0,1)

elif Fall_2 == True:
    range_data = range(1,2)

elif Fall_3 == True:
    range_data = range(2,3)

elif Fall_All == True:
    range_data = range(0,3)

    
for i in range_data:

    if i == 0:
        fall = "1m"
        data_head = np.loadtxt('../dataR/Data_1m/Head.txt')
        data_head = data_head[0:137,:]
        data_shoulder = np.loadtxt('../dataR/Data_1m/Shoulder.txt')
        data_elbow = np.loadtxt('../dataR/Data_1m/Elbow.txt')
        data_wrist = np.loadtxt('../dataR/Data_1m/Wrist.txt')
        data_finger = np.loadtxt('../dataR/Data_1m/Finger.txt')
        data_pelvis = np.loadtxt('../dataR/Data_1m/Pelvis.txt')
        data_knee = np.loadtxt('../dataR/Data_1m/Knee.txt')
        data_ankle = np.loadtxt('../dataR/Data_1m/Ankle.txt')
        data_foot = np.loadtxt('../dataR/Data_1m/Foot.txt')

    
    if i == 1:
        fall = "1.5m"
        data_head = np.loadtxt('../dataR/Data_1.5m/Head.txt')
        data_shoulder = np.loadtxt('../dataR/Data_1.5m/Shoulder.txt')
        data_elbow = np.loadtxt('../dataR/Data_1.5m/Elbow.txt')
        data_wrist = np.loadtxt('../dataR/Data_1.5m/Wrist.txt')
        data_finger = np.loadtxt('../dataR/Data_1.5m/Finger.txt')
        data_pelvis = np.loadtxt('../dataR/Data_1.5m/Pelvis.txt')
        data_knee = np.loadtxt('../dataR/Data_1.5m/Knee.txt')
        data_ankle = np.loadtxt('../dataR/Data_1.5m/Ankle.txt')
        data_foot = np.loadtxt('../dataR/Data_1.5m/Foot.txt')
        
    
    if i == 2:
        fall = "2m"
        data_head = np.loadtxt('../dataR/Data_2m/Head.txt')
        data_shoulder = np.loadtxt('../dataR/Data_2m/Shoulder.txt')
        data_elbow = np.loadtxt('../dataR/Data_2m/Elbow.txt')
        data_wrist = np.loadtxt('../dataR/Data_2m/Wrist.txt')
        data_finger = np.loadtxt('../dataR/Data_2m/Finger.txt')
        data_pelvis = np.loadtxt('../dataR/Data_2m/Pelvis.txt')
        data_knee = np.loadtxt('../dataR/Data_2m/Knee.txt')
        data_ankle = np.loadtxt('../dataR/Data_2m/Ankle.txt')
        data_foot = np.loadtxt('../dataR/Data_2m/Foot.txt')
        

    datas = [data_head, data_shoulder, data_elbow, data_wrist, data_finger, data_pelvis, data_knee,
             data_ankle, data_foot]
    
    
    # 4.2 Smooth Experiments data
    #===========================
    
    xExp_smooth = []
    zExp_smooth = []
    
    for i in range(9):
        data = datas[i] 
        n_movemean = 10 
        size_movmean = len(data)-n_movemean+1
        moved_time = np.linspace(0, data[size_movmean-1][0], size_movmean) 
        xExp_smooth.append(pd.Series(data[:,1]).rolling(window=n_movemean).mean().iloc[n_movemean-1:].values) 
        zExp_smooth.append(pd.Series(-data[:,2]).rolling(window=n_movemean).mean().iloc[n_movemean-1:].values)
    
    timeExp_smooth = moved_time
    
    xExp = xExp_smooth
    zExp = zExp_smooth
    timeExp = timeExp_smooth
    length_Exp = len(timeExp_smooth)
    
    #==============================================================================
    # 5. Robotran Model
    #==============================================================================
    
    # 5.1 Initialise Robotran parameters
    #===================================
    
    if fall == "1m":
        mbs_data.user_model['ExtForce_rope']['K'] = 800
        mbs_data.user_model['ExtForce_rope']['gap'] = 1.9
        mbs_data.user_model['ExtForce_rope']['D'] = 80
        mbs_data.user_model['ExtForce_rope']['frot'] = 110
        
    if fall == "1.5m":
        mbs_data.user_model['ExtForce_rope']['K'] = 3149
        mbs_data.user_model['ExtForce_rope']['gap'] = 2.06
        mbs_data.user_model['ExtForce_rope']['D'] = 169
        mbs_data.user_model['ExtForce_rope']['frot'] = 110
        
    if fall == "2m":
        mbs_data.user_model['ExtForce_rope']['K'] = 3149
        mbs_data.user_model['ExtForce_rope']['gap'] = 2.55
        mbs_data.user_model['ExtForce_rope']['D'] = 169
        mbs_data.user_model['ExtForce_rope']['frot'] = 110
        
    
    # 5.1 Define ID joints
    #=====================
    
    id_T1_base_body = mbs_data.joint_id["T1_base_body"]
    id_T3_base_body = mbs_data.joint_id["T3_base_body"]
    id_R2_base_body = mbs_data.joint_id["R2_base_body"]
    id_R2_body_thigh = mbs_data.joint_id["R2_body_thigh"]
    id_R2_thigh_shank = mbs_data.joint_id["R2_thigh_shank"]
    id_R2_shank_foot = mbs_data.joint_id["R2_shank_foot"]
    id_R2_body_upperarm = mbs_data.joint_id["R2_body_upperarm"]
    id_R2_upperarm_forearm = mbs_data.joint_id["R2_upperarm_forearm"]
    id_R2_forearm_hand = mbs_data.joint_id["R2_forearm_hand"]
    id_R2_body_head = mbs_data.joint_id["R2_body_head"]
    
    # 5.2 Define ID sensors
    #======================
    
    id_sensor_head = mbs_data.sensor_id["sensor_head"]
    id_sensor_shoulder = mbs_data.sensor_id["sensor_shoulder"] 
    id_sensor_elbow = mbs_data.sensor_id["sensor_elbow"]
    id_sensor_wrist = mbs_data.sensor_id["sensor_wrist"]
    id_sensor_finger = mbs_data.sensor_id["sensor_finger"]
    id_sensor_pelvis = mbs_data.sensor_id["sensor_pelvis"]
    id_sensor_knee = mbs_data.sensor_id["sensor_knee"]
    id_sensor_ankle = mbs_data.sensor_id["sensor_ankle"]
    id_sensor_foot = mbs_data.sensor_id["sensor_foot"]
    
    # 5.3 Initialisatise
    #===================
    
    # Initial joint angles 
    q_init = [mbs_data.q[id_T1_base_body], mbs_data.q[id_T3_base_body],
              mbs_data.q[id_R2_base_body], mbs_data.q[id_R2_body_thigh],
              mbs_data.q[id_R2_thigh_shank], mbs_data.q[id_R2_shank_foot],
              mbs_data.q[id_R2_body_upperarm], mbs_data.q[id_R2_upperarm_forearm],
              mbs_data.q[id_R2_forearm_hand], mbs_data.q[id_R2_body_head]] 
    
    # Vector that will contain the resulting coordinates, after the kinematic optimization
    q_res = np.zeros((length_Exp, 10)) 
    xSimuTot = np.zeros((length_Exp,9))
    zSimuTot = np.zeros((length_Exp,9))
    
    # Accessing robotran's function "comp_s_sensor" !
    mbs_data.__load_symbolic_fct__(Robotran.mbs_data.__MODULE_DIR__, ["sensor"], mbs_data.symbolic_path)
    mbs_data.__assign_symb_fct__(["sensor"])
    
    # Adding sensors to the robotran sensor list
    mbs_data.sensors.append(Robotran.MbsSensor(mbs_data, id_sensor_head)) #10
    mbs_data.sensors.append(Robotran.MbsSensor(mbs_data, id_sensor_shoulder)) #2
    mbs_data.sensors.append(Robotran.MbsSensor(mbs_data, id_sensor_elbow)) #7
    mbs_data.sensors.append(Robotran.MbsSensor(mbs_data, id_sensor_wrist)) #8
    mbs_data.sensors.append(Robotran.MbsSensor(mbs_data, id_sensor_finger)) #9
    mbs_data.sensors.append(Robotran.MbsSensor(mbs_data, id_sensor_pelvis)) #1
    mbs_data.sensors.append(Robotran.MbsSensor(mbs_data, id_sensor_knee)) #4
    mbs_data.sensors.append(Robotran.MbsSensor(mbs_data, id_sensor_ankle)) #5
    mbs_data.sensors.append(Robotran.MbsSensor(mbs_data, id_sensor_foot)) #6
    
    #==============================================================================
    # 6. Inverse Kinematics
    #==============================================================================
    
    # 6.1 Kinematic optimisation
    #===========================
    
    error_opt1= np.zeros((length_Exp, 9))
    
    for t in range(0, length_Exp):
        # cost function
        def costFun(q):
            #updating Robotran coordinate (copy the q vectors)
            mbs_data.q[id_T1_base_body] = q[0]
            mbs_data.q[id_T3_base_body] = q[1]
            mbs_data.q[id_R2_base_body] = q[2]
            mbs_data.q[id_R2_body_thigh] = q[3]
            mbs_data.q[id_R2_thigh_shank] = q[4]
            mbs_data.q[id_R2_shank_foot] = q[5]
            mbs_data.q[id_R2_body_upperarm] = q[6]
            mbs_data.q[id_R2_upperarm_forearm] = q[7]
            mbs_data.q[id_R2_forearm_hand] = q[8]
            mbs_data.q[id_R2_body_head] = q[9]
            
            # calls Robotran sensor functions
            mbs_data.sensors[0].comp_s_sensor(id_sensor_head) 
            mbs_data.sensors[1].comp_s_sensor(id_sensor_shoulder)
            mbs_data.sensors[2].comp_s_sensor(id_sensor_elbow)
            mbs_data.sensors[3].comp_s_sensor(id_sensor_wrist)
            mbs_data.sensors[4].comp_s_sensor(id_sensor_finger)
            mbs_data.sensors[5].comp_s_sensor(id_sensor_pelvis)
            mbs_data.sensors[6].comp_s_sensor(id_sensor_knee)
            mbs_data.sensors[7].comp_s_sensor(id_sensor_ankle)
            mbs_data.sensors[8].comp_s_sensor(id_sensor_foot)
            
            #Table with sensor coordinates
            xSimu = [mbs_data.sensors[0].P[1], mbs_data.sensors[1].P[1], 
                     mbs_data.sensors[2].P[1], mbs_data.sensors[3].P[1], 
                     mbs_data.sensors[4].P[1], mbs_data.sensors[5].P[1], 
                     mbs_data.sensors[6].P[1], mbs_data.sensors[7].P[1], 
                     mbs_data.sensors[8].P[1]]
                     
            zSimu = [mbs_data.sensors[0].P[3], mbs_data.sensors[1].P[3], 
                     mbs_data.sensors[2].P[3], mbs_data.sensors[3].P[3], 
                     mbs_data.sensors[4].P[3], mbs_data.sensors[5].P[3], 
                     mbs_data.sensors[6].P[3], mbs_data.sensors[7].P[3], 
                     mbs_data.sensors[8].P[3]]
                     
            xSimuTot[t,:] = xSimu
            zSimuTot[t,:] = zSimu
            vect_xExp = [xExp[0][t], xExp[1][t], xExp[2][t], xExp[3][t], xExp[4][t], xExp[5][t], xExp[6][t], xExp[7][t], xExp[8][t]]
            vect_zExp = [zExp[0][t], zExp[1][t], zExp[2][t], zExp[3][t], zExp[4][t], zExp[5][t], zExp[6][t], zExp[7][t], zExp[8][t]]
            
            # compute the norm of the difference of the vectors
            for i in range(0, 9):
                error_opt1[t, i] = np.power(np.linalg.norm([np.subtract(zSimu[i],vect_zExp[i]), np.subtract(xSimu[i],vect_xExp[i])]),2)
            
            return np.power(np.linalg.norm([np.subtract(zSimu,vect_zExp), np.subtract(xSimu,vect_xExp)]),2)
    
        # optimization function that call 'costFun', with a tolerance of 'xtol'
        q_tmp = scipy.optimize.fmin(func=costFun, x0=q_init, xtol=0.001)
        q_res[t][:] = q_tmp
        q_init = q_tmp
    
    #print("Error from the inverse kynematics opitmisation = %.5f" % error_opt1[t])
    #Compute overall error
    
    # Remove first extreme values
    q_res = q_res[1:,:]
    timeExp = timeExp[1:]
    len(timeExp)
    
    savetxt('../resultsR/error kinematics opt.csv', error_opt1, delimiter=',')
    
    error_opt1 = error_opt1*100
    
    np.average(error_opt1)
    
    plt.rc('font', size=12)
    plt.rc('axes', labelsize=12)  

    fig, ax = plt.subplots(figsize=(6, 4))
    ax.boxplot(error_opt1[:,4:6], labels=['Finger', 'Pelvis'], widths = 0.5, patch_artist = True, boxprops=dict(facecolor= 'darkcyan', color= 'darkcyan'), medianprops=dict(color='navy'), flierprops=dict(color='slategrey', markeredgecolor='slategrey'))
    ax.set_ylabel('Error between the expermiental \n sensors and optimised position [cm]')
    ax.set_title("Relative poisiton error of the Finger and Pelvis") 
    
    # 6.2 Smooth q
    #=============
    
    n_movemean = 10
    size_movmean = len(q_res[:,i])-n_movemean+1
    q_res_smooth = np.zeros((size_movmean,10))
    
    for i in range(10):
        n_movemean = 10
        size_movmean = len(q_res[:,i])-n_movemean+1
        moved_time_q = np.linspace(0, timeExp[size_movmean-1], size_movmean) 
        q_res_smooth[:,i] = pd.Series(q_res[:,i]).rolling(window=n_movemean).mean().iloc[n_movemean-1:].values #change: .append
    
    
    plt.rc('font', size=12)  
    plt.rc('figure', titlesize=30)
    plt.rc('axes', labelsize=15)  
    plt.rc('xtick', labelsize=12)   
    plt.rc('ytick', labelsize=12)    
    plt.rc('legend', fontsize=12)  
    
    plt.figure(figsize=(7, 4.5))
    plt.plot(timeExp, q_res[:,3])
    plt.plot(moved_time_q + timeExp[int(n_movemean/2)], q_res_smooth[:,3])
    plt.xlabel('Time [s]')
    plt.ylabel('Angle [deg]')
    plt.title(" Pelvis R2 angles") 
    plt.legend([ 'normal', 'filtered'], loc=3)
    plt.show()
    
    q_res = q_res_smooth 
    timeExp = moved_time_q 
    
    # 6.3 Velocities and accelerations : spline interpolation
    #========================================================
    
    q, qd, qdd = compute_q_qd_qdd(timeExp ,q_res)
    
    ############Smooth Qdd pou l'écrire dans le point 6.4!!!!!!!!!!!!!!!!!!!!!!!!
    
    # 6.4 Write inverse kinematics results files
    #===========================================
    
    dataFile_q = open("../resultsR/inverse_kinematics_q.res","w+")
    dataFile_qd = open("../resultsR/inverse_kinematics_qd.res","w+")
    dataFile_qdd = open("../resultsR/inverse_kinematics_qdd.res","w+")
    
    for i in range(0,len(timeExp)):
         dataFile_q.write("%f %f %f %f %f %f %f %f %f %f %f\n" % (timeExp[i], q_res[i][0], q_res[i][1], q_res[i][2], q_res[i][3], q_res[i][4], q_res[i][5], q_res[i][6], q_res[i][7], q_res[i][8], q_res[i][9] ))
         dataFile_qd.write("%f %f %f %f %f %f %f %f %f %f %f\n" % (timeExp[i], qd[i][0], qd[i][1], qd[i][2], qd[i][3], qd[i][4], qd[i][5], qd[i][6], qd[i][7], qd[i][8], qd[i][9] ))
         dataFile_qdd.write("%f %f %f %f %f %f %f %f %f %f %f\n" % (timeExp[i], qdd[i][0], qdd[i][1], qdd[i][2], qdd[i][3], qdd[i][4], qdd[i][5], qdd[i][6], qdd[i][7], qdd[i][8], qdd[i][9] ))
    
    dataFile_q.close()
    dataFile_qd.close()
    dataFile_qdd.close()
    
    
    #==============================================================================
    # 6.bis STATIQUE
    #==============================================================================
    
    # dataFile_q = open("../resultsR/inverse_kinematics_q.res","w+")
    # dataFile_qd = open("../resultsR/inverse_kinematics_qd.res","w+")
    # dataFile_qdd = open("../resultsR/inverse_kinematics_qdd.res","w+")
    
    # for i in range(0,len(timeExp)):
    #     dataFile_q.write("%f %f %f %f %f %f %f %f %f %f %f\n" % (timeExp[i], q_res[i][0], q_res[i][1], q_res[i][2], q_res[i][3], q_res[i][4], q_res[i][5], q_res[i][6], q_res[i][7], q_res[i][8], q_res[i][9] ))
    #     dataFile_qd.write("%f %f %f %f %f %f %f %f %f %f %f\n" % (timeExp[i], 0, 0, 0, 0, 0, 0, 0, 0, 0, 0 ))
    #     dataFile_qdd.write("%f %f %f %f %f %f %f %f %f %f %f\n" % (timeExp[i], 0, 0, 0, 0, 0, 0, 0, 0, 0, 0 ))
    
    # dataFile_q.close()
    # dataFile_qd.close()
    # dataFile_qdd.close()
    
    
    #==============================================================================
    # 7. Inverse Dynamics
    #==============================================================================
    
    mbs_data.process = 6
    
    # set the two joints to be actuated
    mbs_data.set_qa(id_T1_base_body)
    mbs_data.set_qa(id_T3_base_body)
    mbs_data.set_qa(id_R2_base_body)
    mbs_data.set_qa(id_R2_body_thigh)
    mbs_data.set_qa(id_R2_thigh_shank)
    mbs_data.set_qa(id_R2_shank_foot)
    mbs_data.set_qa(id_R2_body_upperarm)
    mbs_data.set_qa(id_R2_upperarm_forearm)
    mbs_data.set_qa(id_R2_forearm_hand)
    mbs_data.set_qa(id_R2_body_head)
    
    t_0 = timeExp[2]
    t_f = timeExp[-2]
    
    mbs_invdyn = Robotran.MbsInvdyn(mbs_data)
    mbs_invdyn.set_options(trajectoryqname = "../resultsR/inverse_kinematics_q.res")
    mbs_invdyn.set_options(trajectoryqdname = "../resultsR/inverse_kinematics_qd.res")
    mbs_invdyn.set_options(trajectoryqddname = "../resultsR/inverse_kinematics_qdd.res")
    mbs_invdyn.set_options(t0 = t_0, tf = t_f, dt = 1e-3) # the final time "tf' must be less than the data file used in the optimization
    mbs_invdyn.set_options(motion = "trajectory")
    mbs_invdyn.run()
    
    #==============================================================================
    # 8. Results 1st Opti
    #==============================================================================
    
    # 8.1 Load results file
    #======================
    
    sol = np.loadtxt('../resultsR/inverse_kinematics_q.res')
    
    sol_qd = np.loadtxt('../resultsR/inverse_kinematics_qd.res')
    
    sol_qdd = np.loadtxt('../resultsR/inverse_kinematics_qdd.res')
    
    sol_inv_Qa = np.loadtxt('../resultsR/invdyn_Qa.res')
    
    sol_inv_Qc = np.loadtxt('../resultsR/invdyn_Qc.res')
    
    
    # 8.2 Transform in degrees
    #=========================
    pi = math.pi
    
    for i in range(0, len(q_res)) :
        for j in range(2,len(q_res[0])) :
            q_res[i][j] =(q_res[i][j])*180/pi
    
    for i in range(0, len(sol_qd)) :
        for j in range(2,len(sol_qd[0])) :
            sol_qd[i][j] =(sol_qd[i][j])*180/pi
            
    for i in range(0, len(sol_qdd)) :
        for j in range(2,len(sol_qdd[0])) :
            sol_qdd[i][j] =(sol_qdd[i][j])*180/pi
    
            
    # 8.3 Smooth torques
    #===================
    
    n_movemean = 100
    size_movmean = len(sol_inv_Qa[:,0])-n_movemean+1
    moved_time = np.linspace(0, sol_inv_Qa[size_movmean-1][0], size_movmean)
    qact_3_moved = pd.Series(sol_inv_Qa[:,3]).rolling(window=n_movemean).mean().iloc[n_movemean-1:].values
    qact_6_moved = pd.Series(sol_inv_Qa[:,6]).rolling(window=n_movemean).mean().iloc[n_movemean-1:].values
    qact_12_moved = pd.Series(sol_inv_Qa[:,12]).rolling(window=n_movemean).mean().iloc[n_movemean-1:].values
    qact_15_moved = pd.Series(sol_inv_Qa[:,15]).rolling(window=n_movemean).mean().iloc[n_movemean-1:].values
    qact_18_moved = pd.Series(sol_inv_Qa[:,18]).rolling(window=n_movemean).mean().iloc[n_movemean-1:].values
    qact_21_moved = pd.Series(sol_inv_Qa[:,21]).rolling(window=n_movemean).mean().iloc[n_movemean-1:].values
    qact_24_moved = pd.Series(sol_inv_Qa[:,24]).rolling(window=n_movemean).mean().iloc[n_movemean-1:].values
    qact_27_moved = pd.Series(sol_inv_Qa[:,27]).rolling(window=n_movemean).mean().iloc[n_movemean-1:].values
    
    # 8.4 Smooth internal forces
    #===========================
    
    n_movemean = 100
    size_movmean = len(sol_inv_Qa[:,0])-n_movemean+1
    moved_time = np.linspace(0, sol_inv_Qa[size_movmean-1][0], size_movmean)
    qact_1_moved = pd.Series(sol_inv_Qa[:,1]).rolling(window=n_movemean).mean().iloc[n_movemean-1:].values
    qact_2_moved = pd.Series(sol_inv_Qa[:,2]).rolling(window=n_movemean).mean().iloc[n_movemean-1:].values
    
    n_movemean = 100
    size_movmean = len(sol_inv_Qc[:,0])-n_movemean+1
    moved_time = np.linspace(0, sol_inv_Qc[size_movmean-1][0], size_movmean)
    qact_4_moved_c = pd.Series(sol_inv_Qc[:,4]).rolling(window=n_movemean).mean().iloc[n_movemean-1:].values
    qact_5_moved_c = pd.Series(sol_inv_Qc[:,5]).rolling(window=n_movemean).mean().iloc[n_movemean-1:].values
    qact_7_moved_c = pd.Series(sol_inv_Qc[:,7]).rolling(window=n_movemean).mean().iloc[n_movemean-1:].values
    qact_8_moved_c = pd.Series(sol_inv_Qc[:,8]).rolling(window=n_movemean).mean().iloc[n_movemean-1:].values
    qact_9_moved_c = pd.Series(sol_inv_Qc[:,9]).rolling(window=n_movemean).mean().iloc[n_movemean-1:].values
    qact_10_moved_c = pd.Series(sol_inv_Qc[:,10]).rolling(window=n_movemean).mean().iloc[n_movemean-1:].values
    qact_11_moved_c = pd.Series(sol_inv_Qc[:,11]).rolling(window=n_movemean).mean().iloc[n_movemean-1:].values
    qact_13_moved_c = pd.Series(sol_inv_Qc[:,13]).rolling(window=n_movemean).mean().iloc[n_movemean-1:].values
    qact_14_moved_c = pd.Series(sol_inv_Qc[:,14]).rolling(window=n_movemean).mean().iloc[n_movemean-1:].values
    qact_16_moved_c = pd.Series(sol_inv_Qc[:,16]).rolling(window=n_movemean).mean().iloc[n_movemean-1:].values
    qact_17_moved_c = pd.Series(sol_inv_Qc[:,17]).rolling(window=n_movemean).mean().iloc[n_movemean-1:].values
    qact_19_moved_c = pd.Series(sol_inv_Qc[:,19]).rolling(window=n_movemean).mean().iloc[n_movemean-1:].values
    qact_20_moved_c = pd.Series(sol_inv_Qc[:,20]).rolling(window=n_movemean).mean().iloc[n_movemean-1:].values
    qact_22_moved_c = pd.Series(sol_inv_Qc[:,22]).rolling(window=n_movemean).mean().iloc[n_movemean-1:].values
    qact_23_moved_c = pd.Series(sol_inv_Qc[:,23]).rolling(window=n_movemean).mean().iloc[n_movemean-1:].values
    qact_25_moved_c = pd.Series(sol_inv_Qc[:,25]).rolling(window=n_movemean).mean().iloc[n_movemean-1:].values
    qact_26_moved_c = pd.Series(sol_inv_Qc[:,26]).rolling(window=n_movemean).mean().iloc[n_movemean-1:].values
    
    # 8.5 Compute the norm
    #=====================
    
    qact_moved_T1_T3 = [qact_4_moved_c, qact_5_moved_c, qact_7_moved_c, qact_8_moved_c, 
                        qact_10_moved_c, qact_11_moved_c, qact_13_moved_c, qact_14_moved_c, 
                        qact_16_moved_c, qact_17_moved_c, qact_19_moved_c, qact_20_moved_c, 
                        qact_22_moved_c, qact_23_moved_c ]
    
    norm_T1_T3 = np.zeros((len(qact_moved_T1_T3[1]) ,int(len(qact_moved_T1_T3)/2)))
    
    for i in range(0, int(len(qact_moved_T1_T3)/2)):
        for j in range(0, len(qact_moved_T1_T3[1])):
            norm_T1_T3[j, i] = math.sqrt((qact_moved_T1_T3[2*i][j])**2 + (qact_moved_T1_T3[2*i+1][j])**2)
    

    
    # =============================================================================
    # 9. Second Optimization
    # =============================================================================
    
    # 9.1 Second Optimization T3 free
    # ================================
    
    if Optimisation_ExtForceParameters == False and Optimisation_Masses == False:
        
        print('Error: no Optimisation selected in "0. Running Parameters"')
    
    if Optimisation_ExtForceParameters == True:
    
        maxIter = 100
        
        if fall == "1m":
            names = ['K', 'D', 'frot']
        if fall == "1.5m" or fall == "2m":
            names = ['gap', 'frot']
        
        def costFunRope(opti): # Function that will be evaluate at each iteration, just like the first optimization
            
            for i in range(len(opti)):
                mbs_data.user_model['ExtForce_rope'][names[i]] = opti[i]
           
            # Inverse dynamics to compute joint forces (to get the new first three joints forces).
            mbs_data.process = 6 # Inverse dynamics
        
            mbs_data.set_qa(id_T1_base_body)
            mbs_data.set_qa(id_T3_base_body)
            mbs_data.set_qa(id_R2_base_body)
            mbs_data.set_qa(id_R2_body_thigh)
            mbs_data.set_qa(id_R2_thigh_shank)
            mbs_data.set_qa(id_R2_shank_foot)
            mbs_data.set_qa(id_R2_body_upperarm)
            mbs_data.set_qa(id_R2_upperarm_forearm)
            mbs_data.set_qa(id_R2_forearm_hand)
            mbs_data.set_qa(id_R2_body_head)
            
            t_0 = timeExp[2]
            t_f = timeExp[-2]
        
            mbs_invdyn = Robotran.MbsInvdyn(mbs_data)
            mbs_invdyn.set_options(trajectoryqname = "../resultsR/inverse_kinematics_q.res")
            mbs_invdyn.set_options(trajectoryqdname = "../resultsR/inverse_kinematics_qd.res")
            mbs_invdyn.set_options(trajectoryqddname = "../resultsR/inverse_kinematics_qdd.res")
            mbs_invdyn.set_options(t0 = t_0, tf = t_f, dt = 1e-3) # the final time "tf' must be less than the data file used in the optimization
            mbs_invdyn.set_options(motion="trajectory", verbose=0, resfilename="Optimised")
            mbs_invdyn.run()
            
            Qq = np.loadtxt('../resultsR/Optimised_Qa.res') # Loading joint forces that have been computed in the previous inverse dynamics.
            
            # errors = ["mean joint 1 absolute forces", "mean joint 2 absolute forces", "mean joint 3 absolute forces"]
            
            error = sum(abs(Qq[:,2]))/len(Qq[:, 2]) # error = mean of errors
        
            #error = sum(abs(Qq[:,2])/len(Qq[:,2]))
            
            print("----------------------------")
           
            for i in range(len(opti)):
                print (names[i] + " = " + f'{opti[i]:.2f}')
                
            print()    
            
            print("error = " + f'{error:.2f}')
            
            print("----------------------------")
        
            return error  # return the mean of the means.
        
        K_init = mbs_data.user_model['ExtForce_rope']['K']
        gap_init = gap = mbs_data.user_model['ExtForce_rope']['gap']
        D_init = mbs_data.user_model['ExtForce_rope']['D']
        frot_init = mbs_data.user_model['ExtForce_rope']['frot']
        
        if fall == "1m":
            K_init = mbs_data.user_model['ExtForce_rope']['K']
            gap_init = gap = mbs_data.user_model['ExtForce_rope']['gap']
            D_init = mbs_data.user_model['ExtForce_rope']['D']
            frot_init = mbs_data.user_model['ExtForce_rope']['frot']
            
            init = [K_init, D_init, frot_init]
            
        if fall == "1.5m":
            K_init = mbs_data.user_model['ExtForce_rope']['K']
            gap_init = gap = mbs_data.user_model['ExtForce_rope']['gap']
            D_init = mbs_data.user_model['ExtForce_rope']['D']
            frot_init = mbs_data.user_model['ExtForce_rope']['frot']
            
            init = [gap_init, frot_init]
            
        if fall == "2m":
            
            K_init = mbs_data.user_model['ExtForce_rope']['K']
            gap_init = gap = mbs_data.user_model['ExtForce_rope']['gap']
            D_init = mbs_data.user_model['ExtForce_rope']['D']
            frot_init = mbs_data.user_model['ExtForce_rope']['frot']
            
            init = [gap_init, frot_init]
            
        
        q_tmp = scipy.optimize.fmin(func=costFunRope, x0=init, xtol=0.0001, maxiter=maxIter)
        
        Optimised_Qa = np.loadtxt('../resultsR/Optimised_Qa.res')
        Optimised_Qc = np.loadtxt('../resultsR/Optimised_Qc.res')
    
    # 9.2 Second Optimization all joints free
    # ========================================
    
    if Optimisation_Masses == True:
    
        ids = [mbs_data.body_id["body"], mbs_data.body_id["thigh_1"], mbs_data.body_id["thigh_2"],
                mbs_data.body_id["shank"], mbs_data.body_id["foot"], mbs_data.body_id["upper_arm"],
                mbs_data.body_id["fore_arm"], mbs_data.body_id["hand"], mbs_data.body_id["head"]]
        
        names = ["body", "thigh 1", "thigh 2", "shank", "foot", "upper_arm", "fore_arm", "hand", "head"]
        
        R_inertia = [0.384, 0.329, 0.329, 0.251, 0.257, 0.285, 0.276, 0.628, 0.362]
        Length_inertia = [0.44, 0.106, 0.342, 0.43, 0.28, 0.24, 0.29, 0.15, 0.19]
        
        maxIter = 100 # maximum iterations of the optimization function
        
        bnds_mass = np.zeros((9, 2))
        for m in range (9):
            bnds_mass[m] = ((mbs_data.m[ids[m]]*1-0.2, mbs_data.m[ids[m]]*1+0.2))
        
        def costFunM(masses): # Function that will be evaluate at each iteration, just like the first optimization
            # In the following lines you should update your model parameters
            # Here masses are optimization parameters
            # This for loop is specific to this example
            # You should not have it in you own code
            # mbs_data.m[mbs_data.body_id["FirstBody"]] = masses[0]
            for i in range(9): 
                mbs_data.m[ids[i]] = masses[i]
                mbs_data.In[5, ids[i]] = masses[i]*(R_inertia[i]*Length_inertia[i])**2
            
            print("----------------------------")
           
            for i in range(len(names)):
                print ("Mass " + names[i] + " = " + f'{masses[i]:.2f}')
            
            print("----------------------------")
            
            # Inverse dynamics to compute joint forces (to get the new first three joints forces).
            mbs_data.process = 6 # Inverse dynamics
        
            mbs_data.set_qa(id_T1_base_body)
            mbs_data.set_qa(id_T3_base_body)
            mbs_data.set_qa(id_R2_base_body)
            mbs_data.set_qa(id_R2_body_thigh)
            mbs_data.set_qa(id_R2_thigh_shank)
            mbs_data.set_qa(id_R2_shank_foot)
            mbs_data.set_qa(id_R2_body_upperarm)
            mbs_data.set_qa(id_R2_upperarm_forearm)
            mbs_data.set_qa(id_R2_forearm_hand)
            mbs_data.set_qa(id_R2_body_head)
            
            t_0 = timeExp[2]
            t_f = timeExp[-2]
        
            mbs_invdyn = Robotran.MbsInvdyn(mbs_data)
            mbs_invdyn.set_options(trajectoryqname = "../resultsR/inverse_kinematics_q.res")
            mbs_invdyn.set_options(trajectoryqdname = "../resultsR/inverse_kinematics_qd.res")
            mbs_invdyn.set_options(trajectoryqddname = "../resultsR/inverse_kinematics_qdd.res")
            mbs_invdyn.set_options(t0 = t_0, tf = t_f, dt = 1e-3) # the final time "tf' must be less than the data file used in the optimization
            mbs_invdyn.set_options(motion="trajectory", verbose=0, resfilename="Optimised")
            mbs_invdyn.run()
           
            Qq = np.loadtxt('../resultsR/Optimised_Qa.res') # Loading joint forces that have been computed in the previous inverse dynamics.
            errors = np.zeros(3) # Evaluation of the error. Feel free to try to evaluate it in a different way.
            
            # errors = ["mean joint 1 absolute forces", "mean joint 2 absolute forces", "mean joint 3 absolute forces"]
            for i in range(3):
                errors[i]= sum(abs(Qq[:, i+1]))/len(Qq[:, i+1]) 
            error = sum(abs(errors))/len(errors) # error = mean of errors
        
            print("------ error -------")
            print(error)
        
            return error # return the mean of the means.
    
        masses = np.zeros(9)
        for i in range(9):
                masses[i] = mbs_data.m[ids[i]]
        
        q_tmp = scipy.optimize.minimize(fun=costFunM, x0=masses, method='trust-constr', bounds=bnds_mass, tol=0.05)
        
        Optimised_Qa = np.loadtxt('../resultsR/Optimised_Qa.res')
        Optimised_Qc = np.loadtxt('../resultsR/Optimised_Qc.res')

    

    # 9.3 Compute improvement form optimisation
    # =========================================
    
    if Optimisation_ExtForceParameters == True:
    
        error_no_opt = sum(abs(sol_inv_Qa[:, 2]))/len(sol_inv_Qa[:, 2]) 
        
        error_opt = sum(abs(Optimised_Qa[:, 2]))/len(Optimised_Qa[:, 2])
        
    
    if Optimisation_Masses == True:
    
        errors_no_opt = np.zeros(3)
        for i in range(3):
                errors_no_opt[i]= sum(abs(sol_inv_Qa[:, i + 1 ]))/len(sol_inv_Qa[:, i + 1]) 
        error_no_opt = sum(abs(errors_no_opt))/len(errors_no_opt)
        
        errors_opt = np.zeros(3)
        for i in range(3):
                errors_opt[i]= sum(abs(Optimised_Qa[:, i + 1]))/len(Optimised_Qa[:, i + 1]) 
        error_opt = sum(abs(errors_opt))/len(errors_opt)
        
    
    Improvement = -(error_opt-error_no_opt)/error_no_opt
    print(Improvement)
        


    # 9.4 Smooth internal forces and torques from second optimisation
    #=================================================================
    
    
    Optimised_Qa = np.loadtxt('../resultsR/Optimised_Qa.res')
    Optimised_Qc = np.loadtxt('../resultsR/Optimised_Qc.res')
    
    timeExp_opt = Optimised_Qa[:,0]
    
    
    n_movemean = 100
    size_movmean = len(Optimised_Qa[:,0])-n_movemean+1
    moved_time = np.linspace(0, timeExp_opt[size_movmean-1], size_movmean)
    timeExp_opt = moved_time
    
    Optimised_Qa_smooth = np.zeros((size_movmean,28))
    Optimised_Qc_smooth = np.zeros((size_movmean,28))
    
    
    for i in range(1,28):
        Optimised_Qa_smooth[:,i] = pd.Series(Optimised_Qa[:,i]).rolling(window=n_movemean).mean().iloc[n_movemean-1:].values
        Optimised_Qc_smooth[:,i] = pd.Series(Optimised_Qc[:,i]).rolling(window=n_movemean).mean().iloc[n_movemean-1:].values
        
    #==============================================================================
    # 10. Results 2
    #==============================================================================
    
    # 10.1 Compute the norm
    #======================
    
    norm_T1_T3_opt = np.zeros((len(Optimised_Qc_smooth[:,1]), 8))
    
    for i in range(1, 9):
        for j in range(0, len(Optimised_Qc_smooth[:,1])):
            norm_T1_T3_opt[j,i-1] = math.sqrt((Optimised_Qc_smooth[j][i*3+1])**2 + (Optimised_Qc_smooth[j][i*3+2])**2)
    
    
    # 10.2 Plot the not-optimised internal forces and net torque on free joint 
    #=========================================================================
    
    # Optimised_Qa = np.loadtxt('../resultsR/Optimised_Qa.res')
    # NotOptimised_Qa = np.loadtxt('../resultsR/invdyn_Qa.res')
    
    # plt.figure(figsize=(7, 5))
    # plt.plot(NotOptimised_Qa[:,0], NotOptimised_Qa[:,1])
    # plt.plot(NotOptimised_Qa[:,0], NotOptimised_Qa[:,2])
    # plt.plot(NotOptimised_Qa[:,0], NotOptimised_Qa[:,3])
    # plt.ylabel('Force [N] - Net Torques [Nm]')
    # plt.legend(['Force on I1 [N]', 'Force on I3 [N]', 'Torque on R2 [Nm]'], loc=4)
    # plt.title('Optimised forces and torque between the body and the ceiling')
        
    
    # 10.3 Plot the optimized forces and torque in the free joint 
    #============================================================
    
    plt.figure(figsize=(7, 5))
    plt.plot(moved_time+Optimised_Qa[int(n_movemean/2),0], Optimised_Qa_smooth[:,1])
    plt.plot(moved_time+Optimised_Qa[int(n_movemean/2),0], Optimised_Qa_smooth[:,2])
    plt.plot(moved_time+Optimised_Qa[int(n_movemean/2),0], Optimised_Qa_smooth[:,3])
    plt.ylabel('Force [N] - Net Torques [Nm]')
    plt.legend(['Force on I1 [N]', 'Force on I3 [N]', 'Torque on R2 [Nm]'], loc=4)
    plt.title('Optimised forces and torque between the body and the ceiling')
    
        
    # 10.4 Plot the resulting internal forces and the torques in Pelvis, Knee, Neck
    #==============================================================================

    plt.rc('font', size=12) 
    plt.rc('figure', titlesize=30)
    plt.rc('axes', labelsize=15)  
    plt.rc('xtick', labelsize=12)   
    plt.rc('ytick', labelsize=12)    
    plt.rc('legend', fontsize=12)  
    plt.figure(figsize=(8, 5))
    plt.plot(moved_time+Optimised_Qc[int(n_movemean/2),0], norm_T1_T3_opt[:,0])
    plt.plot(moved_time+Optimised_Qc[int(n_movemean/2),0], norm_T1_T3_opt[:,2])
    plt.plot(moved_time+Optimised_Qc[int(n_movemean/2),0], norm_T1_T3_opt[:,7])
    plt.xlabel('Time [s]')
    plt.ylabel('Force [N]')
    plt.title('Norm of the internal forces in the Pelvis, Knee and Neck')
    plt.legend(['Pelvis', 'Knee', 'Neck'], loc=1)
        
    
    # 10.5 Plot the internal forces and net torque in the thigh
    #===========================================================
    
    plt.figure(figsize=(8, 5))
    plt.plot(moved_time+Optimised_Qc[int(n_movemean/2),0], Optimised_Qc_smooth[:,7])
    plt.plot(moved_time+Optimised_Qc[int(n_movemean/2),0], Optimised_Qc_smooth[:,8])
    plt.plot(moved_time+Optimised_Qc[int(n_movemean/2),0], Optimised_Qc_smooth[:,9])
    plt.xlabel('Time [s]')
    plt.ylabel('Force [N] - Torque [Nm]')
    plt.title('Internal forces and torque in the thigh')
    plt.legend(['Shear force [N]', 'Compression force [N]', 'Bending moment [Nm]'], loc=1, ncol=1)
    
    
    # 10.6 Plot the external force
    #=============================
    
    Fz = np.loadtxt('../resultsR/invdyn_res_ext_force.res')
    Fz_optimized = np.loadtxt('../resultsR/optimised_res_ext_force.res')
    
    nn_movemean = 100
    sizen_movmean = len(Optimised_Qa[:,0])-nn_movemean+1
    Optimised_T3_smooth = pd.Series(Optimised_Qa[:,2]).rolling(window=nn_movemean).mean().iloc[nn_movemean-1:].values
    movedn_time = np.linspace(0, sol_inv_Qa[size_movmean-1][0], sizen_movmean)
     
    plt.figure(figsize=(7, 5))
    plt.plot(Fz[:,0], Fz[:,1], 'red')
    plt.plot(Fz_optimized[:,0], Fz_optimized[:,1], 'green')
    plt.xlabel('Time [s]')
    plt.ylabel('Force [N]')
    plt.ylim((-1100,30))
    plt.title('Optimization of the external force on its key parameters ')
    plt.legend(['External Force Not-Optimized', 'External Force Optimized'], loc=4, ncol=1)

    
    #==============================================================================
    # 11. Analysis for different fall heights
    #==============================================================================
    
    # 11.1 Comput the max, average of joints on 3 fall heights
    #========================================================== 
    
    if Fall_All == True:
    
        norm_T1_T3_max = np.zeros((8,))
        norm_T1_T3_average = np.zeros((8,))
        
        # Correspond to one pick for each fall
        
        if fall == "1m":
              id_row_start = 0
              id_row_end = 1131
              
        if fall == "1.5m":
            id_row_start = 49
            id_row_end = 1608
            
        if fall == "2m":
            id_row_start = 214
            id_row_end = 1940
            
        # Average, Max of Norms
        
        for i in range(0,8):
            norm_T1_T3_max[i] = max(norm_T1_T3_opt[:,i])
            norm_T1_T3_average[i] = np.average(norm_T1_T3_opt[id_row_start:id_row_end,i])
        
        # Average, Max of Torques
        
        Optimised_Qa_smooth_abs = np.zeros((len(range(id_row_start,id_row_end)), len(Optimised_Qa_smooth[1,:])))
        Optimised_Qc_smooth_abs = np.zeros((len(range(id_row_start,id_row_end)), len(Optimised_Qc_smooth[1,:])))
        
        
        for i in range(0, len(Optimised_Qa_smooth[1,:])):
            for j in range(id_row_start, id_row_end):
                Optimised_Qa_smooth_abs[id_row_start,i] = abs(Optimised_Qa_smooth[j,i])
                Optimised_Qc_smooth_abs[id_row_start,i] = abs(Optimised_Qc_smooth[j,i])
        
        R2_max = np.zeros((8,))
        R2_average = np.zeros((8,))
        
        for i in range(1,9):
            if i == 2:
                R2_max[i-1] = max(Optimised_Qc_smooth_abs[:,(i+1)*3])
                R2_average[i-1] = np.average(Optimised_Qc_smooth_abs[:,(i+1)*3])
            else:
                R2_max[i-1] = max(Optimised_Qa_smooth_abs[:,(i+1)*3])
                R2_average[i-1] = np.average(Optimised_Qa_smooth_abs[:,(i+1)*3])
        
        
        
        if fall == "1m":
            norm_T1_T3_max_1M = norm_T1_T3_max
            norm_T1_T3_average_1M = norm_T1_T3_average
            R2_max_1M = R2_max
            R2_average_1M = R2_average
        
        if fall == "1.5m":
            norm_T1_T3_max_15M = norm_T1_T3_max
            norm_T1_T3_average_15M = norm_T1_T3_average
            R2_max_15M = R2_max      
            R2_average_15M = R2_average 
        
        if fall == "2m": 
            norm_T1_T3_max_2M = norm_T1_T3_max
            norm_T1_T3_average_2M = norm_T1_T3_average
            R2_max_2M = R2_max
            R2_average_2M = R2_average
            
        
        # 11.2 Plot the maxima on 1M, 1.5M, 2M
        #=====================================
        
        
        if fall == "2m":
            falls = ["1.0m", "1.5m", "2.0m"]
           
            plt.figure(figsize=(9, 4.5))
            barWidth = 0.2
            y1 = norm_T1_T3_max_1M
            y2 = norm_T1_T3_max_15M
            y3 = norm_T1_T3_max_2M
            
            y1 = list(y1[i] for i in [0,2,7])
            y2 = list(y2[i] for i in [0,2,7])
            y3 = list(y3[i] for i in [0,2,7])
            
            r1 = range(len(y1))
            r2 = [x + barWidth for x in r1]
            r3 = [x + barWidth for x in r2]    
            
            p1 = plt.bar(r1, y1, width = barWidth, color = ['Grey' for i in y1],
                        edgecolor = ['k' for i in y1], linewidth = 0.8, label='1 m')
            p2 = plt.bar(r2, y2, width = barWidth, color = ['blue' for i in y1],
                        edgecolor = ['k' for i in y1], linewidth = 0.8, label='1.5 m')
            p3 = plt.bar(r3, y3, width = barWidth, color = ['orange' for i in y1],
                      edgecolor = ['k' for i in y1], linewidth = 0.8, label= '2 m ',)
            
            
            plt.xticks(r2, ['Pelvis', 'Knee', 'Neck'], fontsize=13)
            plt.ylabel('Force [N]', fontsize=13)
            plt.title('Comparison of the resulting internal forces in different falls')
            plt.legend(handles=[p1, p2, p3], loc='upper right', fontsize=10)
            plt.ylim((0,1400))
            plt.show()

            increase_1_15_pelvis = (y2[0]-y1[0])/y1[0]
            increase_15_2_pelvis = (y3[0]-y2[0])/y2[0]
            
            
            
        #==============================================================================
        # 12. Energy anlysis
        #==============================================================================
            
        energie_cinetique = np.zeros(len(timeExp))
        energie_elastique = np.zeros(len(timeExp))
        energie_potentielle = np.zeros(len(timeExp))
        energie_totale = np.zeros(len(timeExp))
         
        for i in range(0,len(timeExp)):
        #1/2*m*v**2 with m=64kg
            energie_cinetique[i] = 0.5*64*((data_pelvis[i+1][2]-data_pelvis[i][2])/(data_pelvis[i+1][0]-data_pelvis[i][0]))**2
        #1.7 = gap in function of the pelvis coordinates (and not the F sensor)
        if data_pelvis[i][2] <= -1.7:
        #1/2*k*(gap-l)**2 with k=3149 and gap=1.7
            energie_elastique[i] = 0.5*3149*(1.7+data_pelvis[i][2])**2
        else:
            energie_elastique[i] = 0
        #m*g*h with m=64kg, g=9.81 m/s**2 and h in function of the lowest point of our motion
            energie_potentielle[i] = 64*9.81*abs(-data_pelvis[i][2]-2.31)
                 
        
        
        
        
                    
            
            
            
            
