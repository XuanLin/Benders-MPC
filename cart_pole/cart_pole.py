import numpy as np
import os, sys
from pybullet_dynamics.cart_pole_soft_wall_dynamics_pybullet import cart_pole_dynamics
import matplotlib.pyplot as plt
from termcolor import colored
import scipy.io
import copy
import time
import pdb

import GBD_cart_pole

from cart_pole_constants import N, dT, mc, mp, ll, k1, k2, d_left, d_right, d_max, u_max, x_ub, x_lb, lam_ub, lam_lb, Q, R, QN, g, x_ini, theta_ini, dx_ini, dtheta_ini

x0_GBD = np.array([x_ini, theta_ini, dx_ini, dtheta_ini])
x_goal = np.array([0.0, 0.0, 0.0, 0.0])
len_x = len(x0_GBD)
len_z = 2
D_max = x_ub[0] - x_lb[0]
lam_max = lam_ub[0] - lam_lb[0]

E = np.eye(len_x) + dT*np.array([[0.0,               0.0, 1.0, 0.0],
                                 [0.0,               0.0, 0.0, 1.0],
                                 [0.0,           g*mp/mc, 0.0, 0.0],
                                 [0.0, g*(mc+mp)/(ll*mc), 0.0, 0.0]])  

F = dT*np.array([[      0.0,        0.0,         0.0],
                 [      0.0,        0.0,         0.0],
                 [     1/mc,        0.0,         0.0],
                 [1/(ll*mc),  1/(ll*mp),  -1/(ll*mp)]])

G = np.zeros([len_x, len_z])

H1 = np.array([[ 0.0,  0.0,  0.0,  0.0],
               [ 0.0,  0.0,  0.0,  0.0],
               [-1.0,   ll,  0.0,  0.0],
               [ 1.0,  -ll,  0.0,  0.0],
               [ 1.0,  -ll,  0.0,  0.0],
               [-1.0,   ll,  0.0,  0.0],
               [ 1.0,  0.0,  0.0,  0.0],
               [-1.0,  0.0,  0.0,  0.0],
               [ 0.0,  1.0,  0.0,  0.0],
               [ 0.0, -1.0,  0.0,  0.0],
               [ 0.0,  0.0,  1.0,  0.0],
               [ 0.0,  0.0, -1.0,  0.0],
               [ 0.0,  0.0,  0.0,  1.0],
               [ 0.0,  0.0,  0.0, -1.0],
               [ 0.0,  0.0,  0.0,  0.0],
               [ 0.0,  0.0,  0.0,  0.0],
               [ 0.0,  0.0,  0.0,  0.0],
               [ 0.0,  0.0,  0.0,  0.0],
               [ 0.0,  0.0,  0.0,  0.0],
               [ 0.0,  0.0,  0.0,  0.0]])

H2 = np.array([[ 0.0,     1.0,     0.0],
               [ 0.0,     0.0,     1.0],
               [ 0.0,  1.0/k1,     0.0],
               [ 0.0, -1.0/k1,     0.0],
               [ 0.0,     0.0,  1.0/k2],
               [ 0.0,     0.0, -1.0/k2],
               [ 0.0,     0.0,     0.0],
               [ 0.0,     0.0,     0.0],
               [ 0.0,     0.0,     0.0],
               [ 0.0,     0.0,     0.0],
               [ 0.0,     0.0,     0.0],
               [ 0.0,     0.0,     0.0],
               [ 0.0,     0.0,     0.0],
               [ 0.0,     0.0,     0.0],
               [ 1.0,     0.0,     0.0],
               [-1.0,     0.0,     0.0],
               [ 0.0,     1.0,     0.0],
               [ 0.0,    -1.0,     0.0],
               [ 0.0,     0.0,     1.0],
               [ 0.0,     0.0,    -1.0]])

H3 = np.array([[-lam_max,      0.0],
               [     0.0, -lam_max],
               [   D_max,      0.0],
               [     0.0,      0.0],
               [     0.0,    D_max],
               [     0.0,      0.0],
               [     0.0,      0.0],
               [     0.0,      0.0],
               [     0.0,      0.0],
               [     0.0,      0.0],
               [     0.0,      0.0],
               [     0.0,      0.0],
               [     0.0,      0.0],
               [     0.0,      0.0],
               [     0.0,      0.0],
               [     0.0,      0.0],
               [     0.0,      0.0],
               [     0.0,      0.0],
               [     0.0,      0.0],
               [     0.0,      0.0]])

h_theta = np.array([[0.0],
                    [0.0],
                    [-d_right+D_max],
                    [ d_right],
                    [-d_left+D_max],
                    [ d_left],
                    [ x_ub[0]],  #  x_max
                    [-x_lb[0]],  # -x_min
                    [ x_ub[1]], #  theta_max
                    [-x_lb[1]], # -theta_min
                    [ x_ub[2]],  #  dx_max
                    [-x_lb[2]],  # -dx_min
                    [ x_ub[3]],  #  dtheta_max
                    [-x_lb[3]],  # -dtheta_min
                    [u_max],
                    [u_max],
                    [lam_max],
                    [0.0],  # If put 0 here, infeasible.
                    [lam_max],
                    [0.0]])

cc = 1
load_wall_motion = True
save_wall_motion = not load_wall_motion
if load_wall_motion:
    # wall_motion = scipy.io.loadmat('Hz_contact_experiment/Hz_contact_noise/wall_motion_1sec.mat')
    wall_motion = scipy.io.loadmat('Hz_contact_experiment/Hz_contact_noise/wall_motion_100s.mat')
    print(colored("Loading wall_motion from file", 'red'))
    pdb.set_trace()
else:
    delta_d_left_rand = 0.0
    delta_d_right_rand = 0.0
list_delta_d_left = []
list_delta_d_right = []

dynamics = cart_pole_dynamics(mc, mp, ll, k1, k2, d_left, d_right, d_max, u_max, x_ini, theta_ini, dx_ini, dtheta_ini, cc)

time_all = 0.0
time_spend_master = 0.0
num_iter_all = 0
time_traj = []
f_obj_traj = []
list_num_iter = []
list_time_spend_all = []; list_time_spend_real = []; list_time_spend_master = []

x1_animate = []
theta1_animate = []
x_traj = []; u_traj = []; z_traj = []
contact_force_traj = []
u_input = 0.0
x_planned = []
ll_traj = []
num_opt_cuts_traj = []; num_feas_cuts_traj = []

# Settings for GBD solver
full_mip_add_z_conflict_constraint = True
use_Gurobi = False
max_Benders_loop = 5  # 5 for N=10, 10 for N=15  TODO: For N>25, master problem is constantly infeasible. There should still be some issue, or maybe more numerical error due to many shifted cuts!
max_feas_cuts = 45  # 45 for N=10, 150 for N=15
lambda_th = 5000
ang_th = 15.0/180.0*np.pi  # For 45 feasibility cuts, this works better than 3deg
K_opt = 40; K_feas = 20; Lipshitz = 10
z_list = [[0, 0], [0, 1], [1, 0]]

h_d_theta = np.zeros([len_x, 1])

T_sim = 40.0
delta_T_dyn = 0.005  # Delta T for propagating dynamics. This is also assuming that MPC can be solved faster than 0.01s (100Hz)
num_MPC = int(T_sim/delta_T_dyn)

ct_delay = 0
ct_planned_contact = 0

gbd = GBD_cart_pole.GBD()

for i_loop in range(num_MPC):

    if not load_wall_motion:
        # Generate delta d trajectory
        delta_d_left_rand += np.random.normal(0.0, 0.2)*delta_T_dyn
        delta_d_right_rand += np.random.normal(0.0, 0.2)*delta_T_dyn
        delta_d_left = 0.03*np.sin(10*np.pi*i_loop/1000) + delta_d_left_rand
        delta_d_right = 0.03*np.sin(10*np.pi*i_loop/1000) + delta_d_right_rand
        
        # Use this for long simulation duration, less infeasible cases
        # delta_d_left_rand += np.random.normal(0.0, 0.05)*delta_T_dyn
        # delta_d_right_rand += np.random.normal(0.0, 0.05)*delta_T_dyn
        # delta_d_left = 0.01*np.sin(10*np.pi*i_loop/1000) + delta_d_left_rand
        # delta_d_right = 0.01*np.sin(10*np.pi*i_loop/1000) + delta_d_right_rand

        list_delta_d_left.append(delta_d_left)
        list_delta_d_right.append(delta_d_right)

    else:
        delta_d_left = wall_motion['delta_d_left'][0][i_loop]
        delta_d_right = wall_motion['delta_d_right'][0][i_loop]

    print("Number of MPC iterations is {}".format(i_loop))

    if i_loop == 0: dynamics.start_logging()
    ret = dynamics.forward(u=u_input, deltaT=delta_T_dyn, delta_d_left=delta_d_left, delta_d_right=delta_d_right)
    if i_loop == num_MPC-1: dynamics.stop_logging()

    x1 = ret['x']
    dx1 = ret['dx']
    theta1 = ret['theta']
    dtheta1 = ret['dtheta']
    contact_force = ret['contact_force']

    ll_traj.append(ll)
    x1_animate.append(x1)
    theta1_animate.append(theta1)

    x_traj.append(np.array([x1, theta1, dx1, dtheta1]))
    contact_force_traj.append(contact_force)
    
    x0_GBD = np.array([x1, theta1, dx1, dtheta1])
    c_left = d_left + delta_d_left
    c_right = d_right + delta_d_right

    h_theta = np.array([[0.0], [0.0], [-c_right+D_max], [c_right], [-c_left+D_max], [c_left], [ x_ub[0]],  # D_max is confusing with d_max
                    [-x_lb[0]], [ x_ub[1]], [-x_lb[1]],  [ x_ub[2]], [-x_lb[2]],  [ x_ub[3]], [-x_lb[3]], 
                    [u_max], [u_max], [lam_max], [0.0], [lam_max], [0.0]])

    t1 = time.time()
    sol = gbd.main_loop(x0_GBD, h_theta)
    solve_time = time.time() - t1
    print(colored("Speed " + str(1/(solve_time)) + " Hz", 'green'))

    u_input = sol['control']
    # x_planned.append(sol['x_sol'][0])

    u_traj.append([u_input])
    # z_traj.append(sol['z_sol'])

    # Include the case when problem is initially infeasible
    # if sol['max_loop_infeas'] or np.any(abs(sol['z_sol'] - 1.0) <= 1e-6):
    if sol['planned_contact']:

        time_all += solve_time
        num_iter_all += sol['num_iter']

        # Savings for plot
        list_num_iter.append(sol['num_iter'])  
        list_time_spend_all.append(solve_time)
        f_obj_traj.append(sol['cost'])  # Note since in the beginning there is no control due to max_loop, the overall cost is slightly worse for GBD
        time_traj.append((i_loop+1)*delta_T_dyn)  # Note as number of iterations with contact can be different, this doesn't mean anything after the first contact.
        num_opt_cuts_traj.append(sol['num_opt_cut'])
        num_feas_cuts_traj.append(sol['num_feas_cut'])

        ct_planned_contact += 1
        print(colored("MPC Spending on average {} ms, or {} Hz".format(1000*time_all/(ct_planned_contact), (ct_planned_contact)/time_all), 'green'))
        print(colored("The number of iterations are {}".format(list_num_iter), 'green'))
        print(colored("The average number of iterations is {}".format(1.0*num_iter_all/(ct_planned_contact)), 'green'))

# dynamics.__del__()

if use_Gurobi:
    # Save time trajectory
    scipy.io.savemat('saved_results/t_spend_Gurobi_' + str(cc) + '.mat', mdict={'time_traj': np.array(time_traj), 'cost_Gurobi': np.array(f_obj_traj), 
                                                                                'time_Gurobi': np.array(list_time_spend_all), 'num_iter_traj': np.array(list_num_iter),
                                                                                'x_traj': np.array(x_traj), 'N': N, 'dT': dT}) 
else:
    # Save time trajectory
    scipy.io.savemat('saved_results/t_spend_Benders_' + str(cc) + '.mat', mdict={'time_traj': np.array(time_traj), 'cost_Benders': np.array(f_obj_traj), 
                                                                                 'time_Benders': np.array(list_time_spend_all), 'time_Benders_real': np.array(list_time_spend_real), 'time_Benders_master': np.array(list_time_spend_master),
                                                                                 'num_iter_traj': np.array(list_num_iter), 'opt_cuts_traj': np.array(num_opt_cuts_traj), 'feas_cuts_traj': np.array(num_feas_cuts_traj),
                                                                                 'x_traj': np.array(x_traj), 'N': N, 'dT': dT})
    
# Save noise trajectory
if save_wall_motion:
    scipy.io.savemat('saved_noise/wall_motion.mat', mdict={'delta_d_left': np.array(list_delta_d_left), 'delta_d_right': np.array(list_delta_d_right)})
