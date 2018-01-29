import os, sys, rospy, time
from std_msgs.msg import Float64
from gazebo_msgs.srv import GetModelState
from sensor_msgs.msg import JointState
from tf.transformations import euler_from_quaternion, quaternion_from_euler
import numpy as np
import math
import scipy.stats as stats
from gazebo_ros import gazebo_interface
from std_srvs.srv import Empty


# Topic: /hyq/{leg}_{joint}_control/command | Msg: std_msgs/Float64
# Topic: /joint_states | Msg: sensor_msgs/JointState

param_name              = ""
model_name              = ""
robot_namespace         = ""
gazebo_namespace        = "/gazebo"
joint_names             = []
joint_positions         = []

# Setting the parameters
model_name = 'hyq'
param_name = 'robot_description'
# LF foot
joint_names.append('lf_haa_joint')
joint_positions.append(-0.2)
joint_names.append('lf_hfe_joint')
joint_positions.append(0.65)
joint_names.append('lf_kfe_joint')
joint_positions.append(-1.4)
# RF foot
joint_names.append('rf_haa_joint')
joint_positions.append(-0.2)
joint_names.append('rf_hfe_joint')
joint_positions.append(0.65)
joint_names.append('rf_kfe_joint')
joint_positions.append(-1.4)
# LH foot
joint_names.append('lh_haa_joint')
joint_positions.append(-0.2)
joint_names.append('lh_hfe_joint')
joint_positions.append(-0.65)
joint_names.append('lh_kfe_joint')
joint_positions.append(1.4)
# RH foot
joint_names.append('rh_haa_joint')
joint_positions.append(-0.2)
joint_names.append('rh_hfe_joint')
joint_positions.append(-0.65)
joint_names.append('rh_kfe_joint')
joint_positions.append(1.4)

joint_names.append('hokuyo_joint')
joint_positions.append(0.0)



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
JointMid = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
# (Max - Min)/2: Can be multiplied with a control function and added with the mid value to ensure limits
JointAmp = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]


evalTime = 4  # Total evaluation period (seconds)
conFreq = 100  # Rate of each iteration (Hz)
numOmega = 16     # Number of mean values considered

for i in range(len(JointLimits)):
    JointMid[i] = (JointLimits[i][0] + JointLimits[i][1])/2
    JointAmp[i] = abs((JointLimits[i][1] - JointLimits[i][0])/2)
    if i%3 == 0:
        JointMid[i] = -0.2
        JointAmp[i] = JointAmp[i]/1.5
    elif (i-2)%3 == 0:
        JointAmp[i] = JointAmp[i]/1.2
    elif (i-1)%3 == 0:
        JointMid[i] = (JointMid[i]*0.65)/abs(JointMid[i])
        JointAmp[i] = JointAmp[i]/1.01

JointLimits = np.matrix(JointLimits)
# HyQ Joint Positions
hyq_jp = [Float64() for controller in JointControls]

# HyQ Joint State
hyq_js = JointState()

# Experimentally obtained stable joint positions to put the hyq in 0 state
JointPos = [-0.3, 0.7, -1.4, \
            -0.2, -0.75, 1.4, \
            -0.3, 0.7, -1.4, \
            -0.2, -0.75, 1.4,]

# Joint State Subsriber Callback

def js_callback(msg):
    global hyq_js
    hyq_js = msg


des_state = [5.0, 0.0, 0.5, 0.0, 0.0, 0.0]


# ---------------------------------------------------------------------------- #
# ------------------------          CMA               ------------------------ #
# ---------------------------------------------------------------------------- #


numCtrl = 12
maxCtrl = 1

nVar = numCtrl*numOmega

xmean = stats.truncnorm.rvs((-10)/0.5,(10)/0.5,loc=0,scale=0.5,size=(nVar,))

la = (4 + int(round(3*np.log(nVar))))*3
mu = int(round(la/2))
w = np.log(mu + 0.5) - np.log(range(1,mu+1))
w = np.power(w,2)
w = w/np.sum(w)

mueff = np.square(np.sum(w))/np.sum(np.square(w))

sigma = 0.5

cs = (mueff + 2)/(nVar + mueff + 5)
damps = 1 + cs + 2*max(np.sqrt((mueff - 1)/(nVar + 1)) - 1,0)
chiN = np.sqrt(nVar)*(1 - 1/(4*nVar) + 1/(21*nVar**2))

cc = (4 + mueff/nVar)/(4 + nVar + 2*mueff/nVar)
c1 = 2/((nVar + 1.3)**2 + mueff)
alpha_mu = 2
cmu = min(1-c1, alpha_mu*(mueff - 2 + (1/mueff)/((nVar + 2)**2 + alpha_mu*mueff/2)))
hth = ((1.4 + (2/(nVar + 1)))*chiN)

pc = np.zeros((nVar,1))
ps = np.zeros((nVar,1))
B = np.eye(nVar)
D = np.eye(nVar)
C = B*D*np.transpose(B*D)

eigeneval = 0
counteval = 0

arz = np.zeros((nVar,la))
arx = np.zeros((nVar,la))


def finalCost(hyq_state, des_state):

    des_state = np.array(des_state)

    Q1 = 1000.0
    Q2 = 0.0

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


def recCost(prevJointState, curJointState, hyq_state, des_state):
    global JointLimits

    Q1 = 50
    Q2 = 50

    des_state = np.array(des_state)

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

    p1 = Q1*np.matmul(np.transpose(pos_error), pos_error)
    p2 = Q2*np.matmul(np.transpose(ori_error), ori_error)

    return (p1)


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
    ind = 0

    for t in T:             # t = t/100
        control[ind] = weights[0] + weights[1]*math.sin(t*weights[2]*math.pi + weights[3]) + weights[4]*math.cos(t*weights[5]*math.pi + weights[6]) \
                        - weights[7]*math.sin(t*weights[8]*math.pi + weights[9]) - weights[10]*math.cos(t*weights[11]*math.pi + weights[12]) + \
                        + weights[13]*math.sin(t*weights[14]*math.pi + weights[15])

        ind += 1

    return control


def GenerateControlMat(weights):
    global evalTime, conFreq
    [rows, cols] = np.shape(weights)
    control = np.zeros((rows, evalTime*conFreq))

    for i in range(rows):
        control[i, :] = controlSeq(weights[i])

    return control


def unpause(gazebo_namespace):
	rospy.wait_for_service(gazebo_namespace+'/unpause_physics')
	time.sleep(1)
	try:
		unpause_physics = rospy.ServiceProxy(gazebo_namespace+'/unpause_physics', Empty)
		resp = unpause_physics()
		return
	except rospy.ServiceException, e:
		print



if __name__ == '__main__':

    Cweights = np.random.random((numCtrl, numOmega))*2 - 1

    rospy.init_node('JointControl')
    pubRate = rospy.Rate(conFreq)
    pub = [rospy.Publisher('/hyq/' + controller + '/command', Float64, queue_size=1) for controller in JointControls]
    rospy.Subscriber('joint_states', JointState, js_callback)
    hyq_stateService = rospy.ServiceProxy('/gazebo/get_model_state', GetModelState)

    j = 0
    arfitness = np.zeros(la,)

    res_world = rospy.ServiceProxy('/gazebo/reset_world', Empty)
    res_sim = rospy.ServiceProxy('/gazebo/reset_simulation', Empty)
    pause_physics = rospy.ServiceProxy('/gazebo/pause_physics', Empty)
    unpause_physics = rospy.ServiceProxy('/gazebo/unpause_physics', Empty)

    generation = 0
    lastGenBest = 1000000

    while True:

        generation += 1
        arfitness *= 0
        arfitness += 1000000

        for k in range(la):

            pause_physics()
            res_world()
            res_sim()

            gazebo_namespace = '/gazebo'
            success = gazebo_interface.set_model_configuration_client(model_name, param_name, \
                                                                          joint_names, joint_positions, gazebo_namespace)

            success = unpause(gazebo_namespace)

            unpause_physics = rospy.ServiceProxy('/gazebo/unpause_physics', Empty)

            arz[:,k] = stats.truncnorm.rvs((-10)/0.5,(10)/0.5,loc=0,scale=0.5,size=(nVar,))
            arx[:,k] = (xmean + sigma*(np.matmul(B, np.matmul(D, arz[:,k]))))
            Cweights = arx[:,k].reshape((numCtrl,numOmega))

            ControlMat = GenerateControlMat(Cweights)
            [rows, cols] = np.shape(ControlMat)

            des_state_rec = [5, 0, 0, 0, 0, 0]
            shift = des_state_rec[0]/cols
            des_state_rec[0] = shift

            hyq_state = (hyq_stateService('hyq','')).pose
            des_state[0] = hyq_state.position.x + 5
            des_state[1] = hyq_state.position.y
            des_state[1] = hyq_state.position.z

            rcost = 0
            p_hyq_js_pos = np.zeros((numCtrl,))
            c_hyq_js_pos = p_hyq_js_pos

            for i in range(cols):
                c_hyq_js_pos = hyq_js.position

                for controller in JointControls:
                    ind = JointControls.index(controller)
                    hyq_jp[ind] = ControlMat[ind, i]
                    pub[ind].publish(hyq_jp[ind])

                hyq_state = (hyq_stateService('hyq','')).pose
                rcost += np.asscalar(recCost(p_hyq_js_pos[0:12], c_hyq_js_pos[0:12], hyq_state, des_state_rec))*0.01
                des_state_rec[0] += shift
                p_hyq_js_pos = hyq_js.position
                pubRate.sleep()

            hyq_state = (hyq_stateService('hyq','')).pose

            #np.asscalar(finalCost(hyq_state, des_state))

            hyq_Qua = hyq_state.orientation
            (hyq_roll, hyq_pitch, hyq_yaw) = euler_from_quaternion([hyq_Qua.x, hyq_Qua.y, hyq_Qua.z, hyq_Qua.w])

            arfitness[k] = 100*np.exp(-1*((hyq_state.position.x)**2)/2) + rcost
            print('Generation: ', generation, ' Genome: ', k + 1, ' Cost: ', arfitness[k], ' Prev Best Fitness: ', lastGenBest, ' Current Gen Best: ', arfitness.min())
            rcost = 0

        lastGenBest = min(lastGenBest,arfitness.min())

        arindex = np.argsort(arfitness)
        arfitness = np.sort(arfitness)

        t_xmean = np.zeros(nVar,)
        for k in range(mu):
            t_xmean += arx[:,arindex[k]]*w[k]
        t_xmean = t_xmean/mu

        xmean = t_xmean


        t_zmean = np.zeros(nVar,)
        for k in range(mu):
            t_zmean += arz[:,arindex[k]]*w[k]
        t_zmean = t_zmean/mu

        zmean = t_zmean

        ps = (1 - cs)*ps + (np.sqrt(cs*(2-cs))*mueff)*np.matmul(B,zmean)
        hsig = (np.linalg.norm(ps)/np.sqrt(1 - (1 - cs)**(2*counteval/la))/chiN) < (1.4 + (2/(nVar + 1)))

        pc = (1 - cc)*pc + hsig*np.sqrt(cc*(2 - cc)*mueff)*(np.matmul(B, np.matmul(D, zmean)))

        C = (1 - c1 - cmu)*C + c1*(np.matmul(pc, np.transpose(pc)) + (1 + hsig)*cc*(2-cc)*C) + \
            cmu*np.matmul((np.matmul(B, np.matmul(D, arz[:,arindex[0:mu]]))) ,  \
            np.matmul(np.diag(w),np.transpose((np.matmul(B, np.matmul(D, arz[:,arindex[0:mu]]))))))

        sigma = sigma*np.exp((cs/damps)*(np.linalg.norm(ps)/chiN - 1))

        if counteval - eigeneval > la/(c1 + cmu)/nVar/10:
            eigeneval = counteval
            C = np.triu(C) + np.transpose(np.triu(C,1))
            D, B = np.linalg.eig(C)
            D = np.diag(np.sqrt(D))

        np.savetxt('xmean_'+str(generation)+'.csv', xmean, delimiter=",")
        np.savetxt('C_'+str(generation)+'.csv', C, delimiter=",")
        np.savetxt('B_'+str(generation)+'.csv', B, delimiter=",")
        np.savetxt('D_'+str(generation)+'.csv', D, delimiter=",")
        np.savetxt('arfitness_'+str(generation)+'.csv', arfitness, delimiter=",")
        np.savetxt('sigma_'+str(generation)+'.csv', np.array([1, sigma]), delimiter=",")
        np.savetxt('ps_'+str(generation)+'.csv', np.array(ps), delimiter=",")
        np.savetxt('pc_'+str(generation)+'.csv', np.array(pc), delimiter=",")
        np.savetxt('zmean_'+str(generation)+'.csv', zmean, delimiter=",")
        np.savetxt('arx_'+str(generation)+'.csv', arx, delimiter=",")
