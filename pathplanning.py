import numpy as np
from scipy.spatial.transform import Rotation as R
import csv
import time
from Arm_Lib import Arm_Device

def rotx(theta):
    return R.from_euler('x', theta, degrees=True).as_matrix()

def roty(theta):
    return R.from_euler('y', theta, degrees=True).as_matrix()

def rotz(theta):
    return R.from_euler('z', theta, degrees=True).as_matrix()

def fwdkin_Dofbot(q):
    ex = np.array([1, 0, 0])
    ey = np.array([0, 1, 0])
    ez = np.array([0, 0, 1])
    
    l0 = 0.061  # base to servo 1
    l1 = 0.0435  # servo 1 to servo 2
    l2 = 0.08285  # servo 2 to servo 3
    l3 = 0.08285  # servo 3 to servo 4
    l4 = 0.07385  # servo 4 to servo 5
    l5 = 0.05457  # servo 5 to gripper
    
    R01 = rotz(q[0])
    R12 = roty(-q[1])
    R23 = roty(-q[2])
    R34 = roty(-q[3])
    R45 = rotx(-q[4])
    R5T = roty(0)
    
    P01 = (l0 + l1) * ez
    P12 = np.zeros(3)
    P23 = l2 * ex
    P34 = -l3 * ez
    P45 = np.zeros(3)
    P5T = -(l4 + l5) * ex
    
    Rot = R01 @ R12 @ R23 @ R34 @ R45 @ R5T
    Pot = P01 + R01 @ (P12 + R12 @ (P23 + R23 @ (P34 + R34 @ (P45 + R45 @ P5T))))
    
    return Rot, Pot

def rotm2euler(R):
    sy = np.sqrt(R[0,0] * R[0,0] + R[1,0] * R[1,0])
    singular = sy < 1e-6
    
    if not singular:
        x = np.arctan2(R[2,1], R[2,2])
        y = np.arctan2(-R[2,0], sy)
        z = np.arctan2(R[1,0], R[0,0])
    else:
        x = np.arctan2(-R[1,2], R[1,1])
        y = np.arctan2(-R[2,0], sy)
        z = 0
    
    return np.array([z, y, x])

def wrap_to_180(angles):
    return ((angles + 180) % 360) - 180

def moveJoint(jnum, ang, speedtime):
    """
    Move the specified joint to the given position.
    """
    Arm.Arm_serial_servo_write(jnum, ang, speedtime)
    return

if __name__ == "__main__":
    # Initialize the robot arm
    Arm = Arm_Device()
    time.sleep(2)  # Wait for initialization
    
    try:
        # Define start and end configurations
        qstart = np.array([150., 45., 45., 45., 90.])
        qend = np.array([90., 45., -30., 20., 10.])
        
        N = 250
        lambda_vals = np.linspace(0, 1, N)
        
        # Pre-allocate arrays
        q = np.zeros((5, N))
        Rot = np.zeros((3, 3, N))
        eulerot = np.zeros((3, N))
        Pot = np.zeros((3, N))
        qset = []
        
        # Generate path
        for ii in range(N):
            q[:, ii] = (1 - lambda_vals[ii]) * qstart + lambda_vals[ii] * qend
            Rot[:, :, ii], Pot[:, ii] = fwdkin_Dofbot(q[:, ii])
            eulerot[:, ii] = wrap_to_180(rotm2euler(Rot[:, :, ii]) * 180 / np.pi)
        
        # Write to CSV
        with open('data.csv', mode='w', newline='') as output:
            writer = csv.writer(output)
            writer.writerow(['', 'λ', 'qdesired(λ)', 'q(λ)'])
            
            for i in range(N):
                qdesired = (1 - lambda_vals[i]) * qstart + lambda_vals[i] * qend
                qactual = q[:, i]
                
                qdesired_str = f"[{', '.join([str(round(x)) for x in qdesired])}]"
                qactual_str = f"[{', '.join([str(round(x)) for x in qactual])}]"
                
                writer.writerow([i+1, round(lambda_vals[i], 3), qdesired_str, qactual_str])
        
        print("Data written to data.csv")
        print("Starting robot movement...")
        
        # Read and execute the path
        with open('data.csv', 'r') as file:
            csv_reader = csv.reader(file)
            next(csv_reader)  # Skip header
            
            for row in csv_reader:
                # Parse the desired angles
                q_str = row[2].strip('[]').split(',')
                angles = [int(float(q)) for q in q_str]
                
                # Move each joint
                for jnum in range(5):
                    moveJoint(jnum + 1, angles[jnum], 500)  # 500ms movement time
                
                time.sleep(1)  # Wait between configurations
        
        print("Path execution completed!")
        
    except KeyboardInterrupt:
        print("\nProgram stopped by user")
    except Exception as e:
        print(f"An error occurred: {e}")
    finally:
        # Release servos when done
        Arm.release_all_servos()
        print("Servos released")
