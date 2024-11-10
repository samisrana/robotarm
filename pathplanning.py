import numpy as np
from scipy.spatial.transform import Rotation as R
import csv
import time
import matplotlib.pyplot as plt
from Arm_Lib import Arm_Device

Arm = Arm_Device()

# Define necessary functions...
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

# Function to move the specified joint to a given angle
def moveJoint(jnum, ang, speedtime):
    Arm.Arm_serial_servo_write(jnum, ang, speedtime)
    return

def readAllActualJointAngles(Arm):
    # Retrieve joint angles, replacing None values with 0.0 if they occur
    q = [Arm.Arm_serial_servo_read(jnum) for jnum in range(1, 6)]
    return np.array([angle if angle is not None else 0.0 for angle in q])

# Generate the path and move the robot accordingly
def move_robot_sequence(qstart, qend, N=250):
    lambda_vals = np.linspace(0, 1, N)
    q_path = np.zeros((5, N))  # Store calculated joint angles
    Tstart = time.time()
    Qmeasure = []  # Measured joint angles
    T = []  # Timestamps

    # Generate the path
    for ii in range(N):
        q_path[:, ii] = (1 - lambda_vals[ii]) * qstart + lambda_vals[ii] * qend

    # Move to initial joint configuration (first point)
    initial_angles = q_path[:, 0]
    for jnum, angle in enumerate(initial_angles):
        moveJoint(jnum + 1, angle, 800)
    time.sleep(2)  # Allow time for reaching initial position

    # Log initial measurement and time
    Qmeasure.append(readAllActualJointAngles(Arm))
    T.append(time.time() - Tstart)

    # Move through the generated path
    for idx, qdesired in enumerate(q_path.T[1:], start=1):  # Skip the first point
        for jnum, angle in enumerate(qdesired):
            moveJoint(jnum + 1, angle, 200)  # Set movement speed
        time.sleep(0.01)  # Small delay between movements

        # Log measured angles and time for each step
        Qmeasure.append(readAllActualJointAngles(Arm))
        T.append(time.time() - Tstart)

    return q_path, Qmeasure, T, lambda_vals

if __name__ == "__main__":
    # Define start and end configurations
    qstart = np.array([90., 90., 90., 90., 90.])
    qend = np.array([0., 210., -10., 120., 0.])
    N = 250  # Number of path points

    # Execute movement and capture data
    q_path, Qmeasure, T, lambda_vals = move_robot_sequence(qstart, qend, N)
    Qmeasure = np.array(Qmeasure).T  # Transpose for plotting

    # Save data to CSV
    with open('combined_data.csv', mode='w', newline='') as output:
        writer = csv.writer(output)
        writer.writerow(['', 'λ', 'qdesired(λ)', 'q(λ)', 'Timestamp'])
        
        for i, (timestamp, measured_angles, qdesired) in enumerate(zip(T, Qmeasure.T, q_path.T)):
            qdesired_str = f"[{', '.join([str(round(x)) for x in qdesired])}]"
            # Handle None values in measured angles by replacing with "0.0" if they occur
            measured_str = f"[{', '.join([str(round(x, 2)) if x is not None else '0.0' for x in measured_angles])}]"
            writer.writerow([i + 1, round(lambda_vals[i], 3), qdesired_str, measured_str, round(timestamp, 3)])

    print("Data saved to combined_data.csv")

    # Plot measured vs calculated angles for each joint
    for joint in range(5):
        plt.figure(figsize=(10, 6))
        plt.plot(lambda_vals, q_path[joint, :], label=f'Joint {joint + 1} Calculated Angle')
        plt.plot(lambda_vals, Qmeasure[joint, :], label=f'Joint {joint + 1} Measured Angle', linestyle='--')
        plt.xlabel("Lambda (Path Progress)")
        plt.ylabel("Angle (degrees)")
        plt.title(f"Measured vs Calculated Angles for Joint {joint + 1}")
        plt.legend()
        plt.grid()
        plt.show()
