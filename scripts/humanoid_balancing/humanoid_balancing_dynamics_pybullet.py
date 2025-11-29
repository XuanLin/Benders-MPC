import os
import inspect
import pybullet
import pybullet_data
import numpy as np
import scipy.io
import time
from termcolor import colored

currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))

class HumanoidBalancingDynamics:
    
    def __init__(self, m, h_com, h_arm, Icom, dR, dL, mu, tau_max, 
                 theta_ini=0.0, dtheta_ini=0.0):
        """
        Initialize the humanoid balancing simulation.
        
        Args:
            m: Mass [kg]
            h_com: CoM height from pivot [m]
            h_arm: Arm contact height from pivot [m]
            Icom: Centroidal inertia [kg·m²]
            dR: Right wall distance [m] (positive)
            dL: Left wall distance [m] (negative)
            kwall: Wall stiffness [N/m]
            mu: Friction coefficient
            tau_max: Max ankle torque [Nm]
            theta_ini: Initial angle [rad] (positive = rightward lean)
            dtheta_ini: Initial angular velocity [rad/s]
        """
        self.m = m
        self.h_com = h_com
        self.h_arm = h_arm
        self.Icom = Icom
        self.dR = dR
        self.dL = dL
        self.kwall = 800
        self.mu = mu
        self.tau_max = tau_max
        self.g = 9.81
        
        # Noise tracking
        self.read_noise_from_file = False
        self.save_noise_to_file = not self.read_noise_from_file
        self.list_noise = []
        
        self.t_total = 0.0
        self.list_t = []
        self.list_control = []
        
        # Connect to PyBullet
        pybullet.connect(pybullet.GUI)
        pybullet.configureDebugVisualizer(pybullet.COV_ENABLE_GUI, 0)
        pybullet.resetDebugVisualizerCamera(
            cameraDistance=2.5, 
            cameraYaw=0, 
            cameraPitch=-10, 
            cameraTargetPosition=[0, 0, h_com]
        )
        
        # Load ground plane
        pybullet.loadURDF(os.path.join(pybullet_data.getDataPath(), "plane.urdf"), 0, 0, 0)
        
        # Load the humanoid balancing URDF
        urdf_path = os.path.join(currentdir, 'humanoid_balancing.urdf')
        if not os.path.exists(urdf_path):
            raise FileNotFoundError(f"URDF not found at {urdf_path}")
        
        posX = 0.0
        posY = 0.0
        posZ = 0.01  # If this is 0.0, the pole is "inserted" into the ground and cannot move
        self.obj = pybullet.loadURDF(urdf_path, posX, posY, posZ)
        
        # Create collision walls
        mass = 0  # Static walls
        self.cuid_right = pybullet.createCollisionShape(
            pybullet.GEOM_BOX, 
            halfExtents=[0.05, 0.2, 0.3]
        )
        self.cuid_left = pybullet.createCollisionShape(
            pybullet.GEOM_BOX, 
            halfExtents=[0.05, 0.2, 0.3]
        )

        self.wall_right = pybullet.createMultiBody(
            mass, 
            self.cuid_right, 
            basePosition=[self.dR + 0.05, 0, self.h_arm], 
            baseOrientation=[0.0, 0.0, 0.0, 1.0]
        )
        self.wall_left = pybullet.createMultiBody(
            mass, 
            self.cuid_left, 
            basePosition=[self.dL - 0.05, 0, self.h_arm], 
            baseOrientation=[0.0, 0.0, 0.0, 1.0]
        )
        
        pybullet.changeVisualShape(self.wall_right, -1, rgbaColor=[0.0, 0.8, 0.0, 1.0])
        pybullet.changeVisualShape(self.wall_left, -1, rgbaColor=[0.0, 0.8, 0.0, 1.0])

        # Set contact properties for compliant walls
        pybullet.changeDynamics(
            self.wall_right, -1, 
            contactStiffness=self.kwall, 
            contactDamping=0.01, 
            restitution=0.9
        )
        pybullet.changeDynamics(
            self.wall_left, -1, 
            contactStiffness=self.kwall, 
            contactDamping=0.01, 
            restitution=0.9
        )
        
        # Store link indices for contact detection
        self.right_tip_link_id = None
        self.left_tip_link_id = None

        num_joints = pybullet.getNumJoints(self.obj)
        for i in range(num_joints):
            joint_info = pybullet.getJointInfo(self.obj, i)
            link_name = joint_info[12].decode('utf-8')  # Link name
            
            if link_name == "right_arm_tip":
                self.right_tip_link_id = i
            elif link_name == "left_arm_tip":
                self.left_tip_link_id = i

        # ============================================================================================
        # Disable default motor control on ankle joint
        maxForce = 0.0
        mode = pybullet.VELOCITY_CONTROL
        pybullet.setJointMotorControl2(self.obj, 1, controlMode=mode, force=maxForce)
        # ============================================================================================
        
        # Set physics parameters
        pybullet.setGravity(0, 0, -9.81)
        pybullet.setPhysicsEngineParameter(fixedTimeStep=0.0001)
        pybullet.setRealTimeSimulation(0)
        
        # Apply initial torque to set initial angle if needed
        if abs(theta_ini) > 1e-6:
            t_end = time.time() + 0.1
            torque_magnitude = 5.0 * np.sign(theta_ini)
            while time.time() < t_end:
                pybullet.applyExternalTorque(
                    self.obj, 1, [0, torque_magnitude, 0], 
                    flags=pybullet.WORLD_FRAME
                )
                pybullet.stepSimulation()
                time.sleep(0.0001)
        
        print(colored(f"Humanoid balancing initialized: dR={dR:.3f}m, dL={dL:.3f}m", 'green'))
    
    def start_logging(self, filename="humanoid_balancing_animation.mp4"):
        self.logging = pybullet.startStateLogging(pybullet.STATE_LOGGING_VIDEO_MP4, filename)
    
    def stop_logging(self):
        pybullet.stopStateLogging(self.logging)
    
    def read_sensor_output(self):
        """Read current state from sensors."""
        theta, dtheta, __, __ = pybullet.getJointState(self.obj, 1)
        
        contact_force_R = 0.0
        contact_force_L = 0.0

        # Right arm tip contacts with right wall
        contacts = pybullet.getContactPoints(
            bodyA=self.obj,
            bodyB=self.wall_right,
            linkIndexA=self.right_tip_link_id
        )
        for contact in contacts:
            contact_force_R += contact[9]  # Normal force
        
        # Left arm tip contacts with left wall
        contacts = pybullet.getContactPoints(
            bodyA=self.obj,
            bodyB=self.wall_left,
            linkIndexA=self.left_tip_link_id
        )
        for contact in contacts:
            contact_force_L += contact[9]  # Normal force

        
        return {'theta': theta,
                'dtheta': dtheta,
                'contact_force_R': contact_force_R,
                'contact_force_L': contact_force_L}
    
    def forward(self, tau_ankle, deltaT):
        
        # Generate noise
        noise = np.random.normal(0.0, 30.0)
        self.list_noise.append(noise)
        
        # Step the simulation for deltaT seconds
        t_begin = time.time()
        while True:
            # Apply disturbance torque
            pybullet.applyExternalTorque(
                self.obj, 1, [0, noise, 0], 
                flags=pybullet.WORLD_FRAME
            )

            # ============================================================================================
            # Apply ankle control torque
            pybullet.setJointMotorControl2(
                self.obj, 1, 
                controlMode=pybullet.TORQUE_CONTROL, 
                force=tau_ankle
            )
            # ============================================================================================

            pybullet.stepSimulation()
            time.sleep(0.0001)
            if time.time() >= (t_begin + deltaT):
                break
        
        t_end = time.time()
        self.t_total += t_end - t_begin
        self.list_control.append(tau_ankle)
        self.list_t.append([t_begin, t_end])
        
        # Read state
        # theta, dtheta, __, __ = pybullet.getJointState(self.obj, 1)
        sensor_read = self.read_sensor_output()      
        theta, dtheta, contact_force_R, contact_force_L = \
            sensor_read['theta'], sensor_read['dtheta'], sensor_read['contact_force_R'], sensor_read['contact_force_L']
                
        return {'theta': theta,
                'dtheta': dtheta,
                'contact_force_R': contact_force_R,
                'contact_force_L': contact_force_L}
    
    def __del__(self):
        pybullet.resetSimulation()
        pybullet.disconnect()
