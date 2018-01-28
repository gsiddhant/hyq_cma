import os, sys, rospy
from std_msgs.msg import Float64
from gazebo_msgs.srv import GetModelState
from sensor_msgs.msg import JointState
from tf.transformations import euler_from_quaternion, quaternion_from_euler
import numpy as np
import math

# Topic: /hyq/{leg}_{joint}_control/command | Msg: std_msgs/Float64
# Topic: /joint_states | Msg: sensor_msgs/JointState


# Joints present in the HyQ
JointNames = ['lf_haa_joint', 'lf_hfe_joint', 'lf_kfe_joint', \
            'lh_haa_joint', 'lh_hfe_joint', 'lh_kfe_joint', \
            'rf_haa_joint', 'rf_hfe_joint', 'rf_kfe_joint', \
            'rh_haa_joint', 'rh_hfe_joint', 'rh_kfe_joint']

# HyQ Joint Controllers
JointControls = ['lf_haa_control', 'lf_hfe_control', 'lf_kfe_control', \
                'lh_haa_control', 'lh_hfe_control', 'lh_kfe_control', \
                'rf_haa_control', 'rf_hfe_control', 'rf_kfe_control', \
                'rh_haa_control', 'rh_hfe_control', 'rh_kfe_control']

# Min and Max position of each of the Joints
JointLimits = [[-1.22, 0.44], [-0.87, 1.22], [-2.44, -0.36], \
                [-1.22, 0.44], [-1.22, 0.87], [0.36, 2.44], \
                [-1.22, 0.44], [-0.87, 1.22], [-2.44, -0.36], \
                [-1.22, 0.44], [-1.22, 0.87], [0.36, 2.44]]

# Mid value of the Joint Position Limits
JointMid = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
# (Max - Min)/2: Can be multiplied with a control function and added with the mid value to ensure limits
JointAmp = JointMid


evalTime = 3   # Total evaluation period (seconds)
conFreq = 100  # Rate of each iteration (Hz)
numOmega = 16     # Number of mean values considered

for i in range(len(JointLimits)):
    JointMid[i] = (JointLimits[i][0] + JointLimits[i][1])/2
    JointAmp[i] = abs((JointLimits[i][1] - JointLimits[i][0])/2)
    if i%3 == 0:
        JointMid[i] = 0
        JointAmp[i] = JointAmp[i]/5

JointLimits = np.matrix(JointLimits)

# HyQ Joint Positions
hyq_jp = [Float64() for controller in JointControls]

# HyQ Joint State
hyq_js = JointState()

# Experimentally obtained stable joint positions to put the hyq in 0 state
JointPos = [-0.3, 0.6, -1.4, \
            -0.1, -0.75, 1.4, \
            -0.3, 0.6, -1.4, \
            -0.1, -0.75, 1.4,]

# Joint State Subsriber Callback
def js_callback(msg):
    global hyq_js
    hyq_js = msg


# ---------------------------------------------------------------------------- #
# ------------------------          CMA               ------------------------ #
# ---------------------------------------------------------------------------- #


numCtrl = 12
# maxCtrl = 1
#
# nVar = numCtrl*numOmega
#
# xmean = np.random.randn(nVar,)
#
# la = (4 + int(round(3*np.log(nVar))))*3
# mu = int(round(la/2))
# w = np.log(mu + 0.5) - np.log(range(1,mu+1))
# w = w/np.sum(w)
# mueff = np.square(np.sum(w))/np.sum(np.square(w))
#
# sigma = 0.3*(maxCtrl*10)
#
# cs = (mueff + 2)/(nVar + mueff + 5)
# damps = 1 + cs + 2*max(np.sqrt((mueff - 1)/(nVar + 1)) - 1,0)
# chiN = np.sqrt(nVar)*(1 - 1/(4*nVar) + 1/(21*nVar**2))
#
# cc = (4 + mueff/nVar)/(4 + nVar + 2*mueff/nVar)
# c1 = 2/((nVar + 1.3)**2 + mueff)
# alpha_mu = 2
# cmu = min(1-c1, alpha_mu*(mueff - 2 + (1/mueff)/((nVar + 2)**2 + alpha_mu*mueff/2)))
# hth = ((1.4 + (2/(nVar + 1)))*chiN)
#
# pc = np.zeros((nVar,1))
# ps = np.zeros((nVar,1))
# B = np.eye(nVar)
# D = np.eye(nVar)
# C = B*D*np.transpose(B*D)
#
# eigeneval = 0
# counteval = 0
#
# arz = np.zeros((nVar,la))
# arx = np.zeros((nVar,la))


def evalFitness(p, r):
    return (p + r)


def finalCost(hyq_state, des_state):

    des_state = np.array(des_state)

    Q1 = 1000
    Q2 = 100

    if len(des_state) != 6:
        print('Length of des_state must be 6.')
        return None

    hyq_Pos = hyq_state.position
    hyq_x = hyq_Pos.x
    hyq_y = hyq_Pos.y
    hyq_z = hyq_Pos.z
    hyq_Qua = hyq_state.orientation
    (hyq_roll, hyq_pitch, hyq_yaw) = euler_from_quaternion([hyq_Qua.x, hyq_Qua.y, hyq_Qua.z, hyq_Qua.w])

    hyq_position = np.array([hyq_x, hyq_y, hyq_z])
    hyq_position = hyq_position.reshape((3,1))

    des_position = des_state[:3]
    des_position = des_position.reshape((3,1))

    hyq_orientation = np.array([hyq_roll, hyq_pitch, hyq_yaw])
    hyq_orientation = hyq_orientation.reshape((3,1))

    des_orientation = des_state[3:]
    des_orientation = des_orientation.reshape((3,1))

    pos_error = np.subtract(hyq_position, des_position)
    ori_error = np.subtract(hyq_orientation, des_orientation)

    p = Q1*np.matmul(np.transpose(pos_error), pos_error) \
    + Q2*np.matmul(np.transpose(ori_error), ori_error)

    return p


def recCost(prevJointState, curJointState, JointLimits):
    curJointState = np.array(curJointState)
    prevJointState = np.array(prevJointState)

    q_max = np.maximum.reduce(np.substract(np.transpose(curJointState), JointLimits[:,1]),np.zeros((len(curJointState),1)))
    q_min = np.minimum.reduce(np.substract(np.transpose(curJointState), JointLimits[:,0]),np.zeros((len(curJointState),1)))

    r1 = np.matmul(np.transpose(q_max),q_max) + np.matmul(np.transpose(q_min),q_min)
    r2 = np.substract(curJointState, prevJointState)
    r2 = np.matmul(np.transpose(r2),r2)

    return (r1 + 0.1*r2)


# ---------------------------------------------------------------------------- #
# ---------------------------------------------------------------------------- #
# ---------------------------------------------------------------------------- #


def RBFKernel(t, mu):
    sigma = 0.01
    return (np.exp(-((mu - t)*(mu - t))/(2*sigma**2)))


def controlSeq(weights):
    global evalTime, conFreq, numOmega

    T = np.linspace(0, evalTime, evalTime*conFreq, endpoint = True)
    control = np.zeros(evalTime*conFreq)

    mu = np.linspace(0, evalTime, numOmega, endpoint = True)

    ind = 0

    for t in T:             # t = t/100
        wSum = 0
        cSum = 0

        for i in range(numOmega):
            wSum += weights[i]*RBFKernel(t, mu[i])
            cSum += RBFKernel(t, mu[i])

        if cSum != 0:
            control[ind] = wSum/cSum
        else:
            control[ind] = 0

        ind += 1

    return control


def GenerateControlMat(weights):
    global evalTime, conFreq
    [rows, cols] = np.shape(weights)
    control = np.zeros((rows, evalTime*conFreq))

    for i in range(rows):
        control[i, :] = controlSeq(weights[i])

    return control


if __name__ == '__main__':

    weights = np.random.random((numCtrl, numOmega))*2 - 1

    rospy.init_node('JointControl')
    pubRate = rospy.Rate(conFreq)
    pub = [rospy.Publisher('/hyq/' + controller + '/command', Float64, queue_size=1) for controller in JointControls]
    rospy.Subscriber('joint_states', JointState, js_callback)
    hyq_stateService = rospy.ServiceProxy('/gazebo/get_model_state', GetModelState)

    theta = 0

    ControlMat = GenerateControlMat(weights)
    [rows, cols] = np.shape(ControlMat)
    #while not rospy.is_shutdown():
        # Get HyQ State after calling the 'get_model_state' service
    hyq_state = (hyq_stateService('hyq','')).pose

    theta += 1
    for i in range(cols):
        for controller in JointControls:
            ind = JointControls.index(controller)
            hyq_jp[ind] = ControlMat[ind, i]*JointAmp[ind] + JointPos[ind]
            pub[ind].publish(hyq_jp[ind])
            if theta > 628:
                theta = 0
        pubRate.sleep()
