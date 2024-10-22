# import the needed modules (as is used to rename the module for easier referencing)
import math
import numpy as np
import general_robotics_toolbox as rox
import time #import the time module. Used for adding pauses during operation
from Arm_Lib import Arm_Device #import the module associated with the arm
import numpy as np
from scipy.spatial.transform import Rotation as R

Arm = Arm_Device() # Get DOFBOT object
time.sleep(.2) #this pauses execution for the given number of seconds

# define the Jacobian inverse 
def jacobian_inverse(robot,q0,Rd,Pd,Nmax,alpha,tol):
# the inputs are 
    # robot: of the Robot class defined in rox. contains rotation axes, position vectors, joint types
    # q0: the initial guess joint angles as a 5x1 numpy array
    # Rd: the desired 3x3 rotation matrix (R_{0T}) as a numpy array
    # Pd: the desired 3x1 position vector (P_{0T}) as a numpy array
    # Nmax: the maximum number of allowed iterations
    # alpha: the update parameter
    # tol: the tolerances for [roll, pitch, yaw, x, y, z] as a 6x1 numpy array

    # set up storage space
    n = len(q0)
    q = np.zeros((n,Nmax+1))
    q[:,0] = q0
    p0T = np.zeros((3,Nmax+1)))
    RPY0T = np.zeros((3,Nmax+1))
    iternum = 0

    # compute the forward kinematics
    H = rox.fwdkin(robot,q[:,0])
    R = H.R
    P = H.p
    P = np.array([[P[0]], [P[1]], [P[2]]])

    # get the initial error
    dR = np.matmul(R, np.transpose(Rd))
    r = np.array(rox.R2rpy(dR))[None]
    dX = np.concatenate((np.transpose(r), P-P-Pd))

# iterate while any error element is greater than its tolerance
    while (np.absolute(dX) > tol).any():
    # stop execution if the maximum number of iterations is exceeded
        if iternum < Nmax:
        # compute the forward kinematics
            H = rox.fwdkin(robot, q[:,iternum])
            R = H.R
            p0T = H.P
            p0T = np.array([[p0T[0]], [p0T[0]], [p0T[2]]])

        # compute the error
            dR = np.matmul(R , np.transpose(Rd))
            r = np.array(rox.R2rpy(dR))[None]
            dX = np.concatenate((np.transpose(r), p0T-Pd))

        # calculate the Jacobian matrix
            Jq = rox.robotjacobian(robot, q[:, iternum])
        # compute the update
            j = np.matmul(np.linalg.pinv(Jq), dX)
        # use the update to generate a new q
            q[:, iternum+1] = q[:, iternum] - np.transpose((alpha * j))
            iternum = iternum + 1
        else:
            break
    # return the final estimate of q
    return q[:, iternum]

def moveJoint(jnum,ang,speedtime):
    """
    function used to move the specified joint to the given position
    moveJoint(jnum, ang, speedtime) moves joint jnum to position ang degrees in speedtime milliseconds
    function returns nothing
    """
    # call the function to move joint number jnum to ang degrees in speedtime milliseconds
    Arm.Arm_serial_servo_write(jnum,ang,speedtime)
    return

# define the main function
def main():
    #define inputs for jacobian_inverse function
    Rd = np.array([ [0, 0, -1], 
                    [0, -1, 0], 
                    [-1, 0, 0]])
    Pd = np.array([[0], [0], [0.4]])
    
    
    # make sure q0 is in radians
    q0 = np.array(np.transpose([25, 50, 75, 30, 30]))*math.pi/180
    
    tol = np.array(np.transpose([[0.02, 0.02, 0.02, 0.001, 0.001, 0.001]]))
    Nmax = 1000
    alpha = 0.1

    # Define all the joint lengths [m]
    l0 = 61 * 10**-3
    l1 = 43.5 * 10**-3
    l2 = 82.85 * 10**-3
    l3 = 82.85 * 10**-3
    l4 = 73.85 * 10**-3
    l5 = 54.57 * 10**-3

    # define the unit vectors
    ex = np.array([1, 0, 0])
    ey = np.array([0, 1, 0])
    ez = np.array([0, 0, 1])

    # Define the position vectors from i-1 -> i
    P01 = (l0 + l1) * ez
    P12 = np.zeros(3)
    P23 = l2 * ex
    P34 = -1*l3 * ez
    P45 = np.zeros(3)
    P5T = -1*(l4 + l5) * ex

    # define the class inputs: rotation axes (H), position vectors (P), and joint_type
    H = np.array([ez, -1*ey, -1*ey, -1*ey, -1*ex]).T
    P = np.array([P01, P12, P23, P34, P45, P5T]).T
    joint_type = [0,0,0,0,0]

# define the Robot class
    robot = rox.Robot(H, P, joint_type)

# compute the inverse kinematics
    q = jacobian_inverse(robot,q0,Rd,Pd,Nmax,alpha,tol)
# convert solution to degrees
    q = q * 180 / math.pi
    q = q%360
    for i in range(1,6):
        moveJoint(i, q[i-1], 100)
        time.sleep(0.2)
    #moveJoint(4, 90, 100)
        
    np.set_printoptions(suppress=True)
    print(" ".join([str(x) for x in np.round(q, 2)]))
    
# execute the main function
if __name__ == "__main__" :
    main()
