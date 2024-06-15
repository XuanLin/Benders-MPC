import os, inspect
currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
print("current_dir=" + currentdir)
parentdir = os.path.join(currentdir, "../gym")
os.sys.path.insert(0, parentdir)

import pybullet
import pybullet_data

import numpy as np
import scipy.io
import time
import pdb
from termcolor import colored

from lxml import etree as ET

class cart_pole_dynamics:

    # Get current state x0, dx0, control input u, propagate dynamics forward by one step, and return x', dx'
    def __init__(self, mc, mp, ll, k1, k2, d_left, d_right, d_max, u_max, x_ini, theta_ini, dx_ini, dtheta_ini, cc):

        self.mc = mc
        self.mp = mp
        self.ll = ll
        self.k1 = k1
        self.k2 = k2
        self.d_left = d_left
        self.d_right = d_right
        self.d_max = d_max
        self.u_max = u_max
        self.x_ini = x_ini
        self.theta_ini = theta_ini
        self.dx_ini = dx_ini
        self.dtheta_ini = dtheta_ini
        self.g = 9.81

        self.read_noise_from_file = False
        self.save_noise_to_file = not self.read_noise_from_file

        if self.read_noise_from_file:
            self.c_noise = 0
            # self.list_noise = scipy.io.loadmat("Hz_contact_experiment/Hz_contact_noise/recorded_noise_1sec.mat")['noise'][0]
            self.list_noise = scipy.io.loadmat("Hz_contact_experiment/Hz_contact_noise/recorded_noise_100s.mat")['noise'][0]
            print(colored("Loading noise from file", 'red'))
            pdb.set_trace()
        else:
            self.list_noise = []

        self.t_total = 0.0
        self.list_t = []
        self.list_control = []

        # tree = ET.parse('/home/xuan/Desktop/cart_pole_soft_wall/cart_pole_moving_wall_MPC/pybullet_example/cartpole.urdf')
        # myroot = tree.getroot()
        # # TODO: Incorrect. Shifting origin is just not the right way.
        # myroot[2][1].attrib['xyz'] = str(x_ini) + " 0.0 0.0"  # "slider_to_cart" -> "origin"
        # # TODO: Incorrect. This will just lead to a constant offset.
        # # myroot[4][1].attrib['rpy'] = "0.0 " + str(-theta_ini) + " 0.0"  # "cart_to_pole" -> "origin" -> "rpy"
        # tree.write('/home/xuan/Desktop/cart_pole_soft_wall/cart_pole_moving_wall_MPC/pybullet_example/cartpole.urdf')

        #choose connection method: GUI, DIRECT, SHARED_MEMORY
        pybullet.connect(pybullet.GUI)
        pybullet.resetDebugVisualizerCamera(cameraDistance=2, cameraYaw=0, cameraPitch=0, cameraTargetPosition=[0,0,1])
        pybullet.loadURDF(os.path.join(pybullet_data.getDataPath(), "plane.urdf"), 0, 0, -1)
        #load URDF, given a relative or absolute file+path
        # TODO: Changing here for initial position is also incorrect, as it just shift the origin. At the end, what is the right way to do it ???
        posX = 0.0; posY = 0.0; posZ = 1

        self.obj = pybullet.loadURDF(currentdir + '/cartpole.urdf', posX, posY, posZ)

        # Set initial position - TODO: why is this not correct ???
        # pybullet.resetBasePositionAndOrientation(self.obj, posObj=[0.0, 0.0, 0.0], ornObj=[0.0, 0.0, 0.0, 1.0])
        # Set initial velocity
        # pybullet.resetBaseVelocity(self.obj, linearVelocity=[0.0, 0.0, 0.0], angularVelocity=[0.0, 0.0, 0.0])

        # Create collision boxes
        self.cuid_left = pybullet.createCollisionShape(pybullet.GEOM_BOX, halfExtents=[0.1, 0.1, 0.2])
        self.cuid_right = pybullet.createCollisionShape(pybullet.GEOM_BOX, halfExtents=[0.1, 0.1, 0.2])
        mass= 0 #static box
        self.box_left = pybullet.createMultiBody(mass, self.cuid_left, basePosition=[-self.d_left-0.1, 0, 1+self.ll], baseOrientation=[0.0, 0.0, 0.0, 1.0])
        self.box_right = pybullet.createMultiBody(mass, self.cuid_right, basePosition=[self.d_right+0.1, 0, 1+self.ll], baseOrientation=[0.0, 0.0, 0.0, 1.0])
        pybullet.changeDynamics(self.cuid_left, -1, contactStiffness=50, contactDamping=0.01, restitution=0.9)
        pybullet.changeDynamics(self.cuid_right, -1, contactStiffness=50, contactDamping=0.01, restitution=0.9)

        maxForce = 0.0
        mode = pybullet.VELOCITY_CONTROL
        pybullet.setJointMotorControl2(self.obj, 0, controlMode=mode, force=maxForce)
        pybullet.setJointMotorControl2(self.obj, 1, controlMode=mode, force=maxForce)

        # Set the gravity acceleration
        pybullet.setGravity(0, 0, -9.8)
        # self.tt = pybullet.addUserDebugText('Pole mass is {}, length is {}'.format(self.mp, self.ll), [-0.6, 0, 2])

        # Change time step
        pybullet.setPhysicsEngineParameter(fixedTimeStep=0.0001)

        pybullet.setRealTimeSimulation(0)

        # Enforce an initial torque to make an initial joint angle
        t_end = time.time() + 0.1
        while time.time() < t_end:
            pybullet.applyExternalTorque(self.obj, 1, [0, 8.0, 0], flags=pybullet.WORLD_FRAME)  # Note theta is opposite from the theta in optimization
            pybullet.stepSimulation()
            time.sleep(0.0001)

    def start_logging(self):
        self.logging = pybullet.startStateLogging(pybullet.STATE_LOGGING_VIDEO_MP4, "cart_pole_animation.mp4")

    def stop_logging(self):
        pybullet.stopStateLogging(self.logging)

    def begin_simulation(self):
        pybullet.setRealTimeSimulation(1)

    def stop_simulation(self):
        pybullet.setRealTimeSimulation(0)

    def apply_control_input(self, u, deltaT):
        # This loop (and the following) will take deltaT seconds if you run it in the same thread as your main thread. Hence, set up multi-thread and run!
        t_end = time.time() + deltaT
        while time.time() < t_end:
            pybullet.setJointMotorControl2(self.obj, 0, controlMode=pybullet.TORQUE_CONTROL, force=u)

    def apply_torque_noise(self, deltaT):
        t_end = time.time() + deltaT
        noise = np.random.normal(0.0, 2.0)
        while time.time() < t_end:
            pybullet.setJointMotorControl2(self.obj, 1, controlMode=pybullet.TORQUE_CONTROL, force=noise)

    def read_sensor_output(self):
        x1, dx1, __, __ = pybullet.getJointState(self.obj, 0)
        theta1, dtheta1, __, __ = pybullet.getJointState(self.obj, 1)
        # TODO: put force sensor to read lambda
        lam1 = 0.0
        lam2 = 0.0

        # Angle definition is negative
        return {'x': x1, 'dx': dx1, 'theta': -theta1, 'dtheta': -dtheta1, 'contact_force': np.array([lam1, lam2])}
        
    def forward(self, u, deltaT, delta_d_left, delta_d_right):

        pybullet.resetBasePositionAndOrientation(self.box_left, posObj=(-self.d_left-0.1+delta_d_left, 0, 1+self.ll), ornObj=(0.0, 0.0, 0.0, 1.0))
        pybullet.resetBasePositionAndOrientation(self.box_right, posObj=(self.d_right+0.1+delta_d_right, 0, 1+self.ll), ornObj=(0.0, 0.0, 0.0, 1.0))

        if self.read_noise_from_file:
            noise = self.list_noise[self.c_noise]
            self.c_noise += 1
        else:  # Generate new noise and save it
            noise = np.random.normal(0.0, 8.0)
            # Use this for long simulation duration
            # noise = np.random.normal(0.0, 4.0)
            self.list_noise.append(noise)

        # step the simulation for deltaT seconds
        t_begin = time.time()
        while True:
            pybullet.applyExternalTorque(self.obj, 1, [0, noise, 0], flags=pybullet.WORLD_FRAME)
            pybullet.setJointMotorControl2(self.obj, 0, controlMode=pybullet.TORQUE_CONTROL, force=u)
            pybullet.stepSimulation()
            time.sleep(0.0001)
            if time.time() >= (t_begin + deltaT): break

        t_end = time.time()
        self.t_total += t_end - t_begin
        self.list_control.append(u)
        self.list_t.append([t_begin, t_end])

        x1, dx1, __, __ = pybullet.getJointState(self.obj, 0)
        theta1, dtheta1, __, __ = pybullet.getJointState(self.obj, 1)
        # TODO: put force sensor to read lambda
        lam1 = 0.0
        lam2 = 0.0

        # Angle definition is negative
        return {'x': x1, 'dx': dx1, 'theta': -theta1, 'dtheta': -dtheta1, 'contact_force': np.array([lam1, lam2])}
    
    def change_pole_mass_length(self, mp_new, ll_new):
        pybullet.removeUserDebugItem(self.tt)
        pybullet.changeDynamics(self.obj, 1, mass=mp_new)  # This already changes the inertial matrix, no need to manually change it again.
        # TODO: joint length is not changed yet. - it seems like changeDynamics does not have an option to change CoM position
        self.tt = pybullet.addUserDebugText('Pole mass is changed to {:.2}, length is changed to {:.2}'.format(mp_new, ll_new), [-0.8, 0, 2])

    def __del__(self):
        
        if self.save_noise_to_file:
            scipy.io.savemat("saved_noise/recorded_noise.mat", mdict={'noise': np.array(self.list_noise), 
                                                                      'control': np.array(self.list_control),
                                                                      'time_stamp': np.array(self.list_t)})

        pybullet.resetSimulation()
        pybullet.disconnect()
        