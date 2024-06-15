import numpy as np
from class_GCS_maze import GCS
import matplotlib.pyplot as plt
from termcolor import colored
import scipy.io
import copy
import time
import pdb

import GBD_GCS_hybrid_N50 as GBD_MLD

x0_GBD = np.array([ -1.0,  -5, 0.0, 0.0])
x0_GCS = np.array([ -1.0,  -5, 0.0, 0.0, 0.0, 0.0])
x_goal = np.array([  8.5, 4.5, 0.0, 0.0])
nx = 4
nu = 2
nz = 11
nc = 60
N = 50
N_GCS = 51

h = np.zeros([nc, 1])
h_d_theta = np.zeros([nx, 1])

Mx = 15; My = 15

x_lim = [{'xl':  -2, 'xu': 0.0, 'yl': -5, 'yu': -4},  #0
         {'xl': 0.0, 'xu':   2, 'yl': -7, 'yu': -4},  #1
         {'xl':   0, 'xu':   1, 'yl': -4, 'yu':  3},  #2
         {'xl':   0, 'xu': 3.5, 'yl':  3, 'yu':  4},  #3
         {'xl':   3, 'xu':   4, 'yl': -4, 'yu':  3},  #4
         {'xl': 3.5, 'xu': 5.5, 'yl':  3, 'yu':  6},  #5
         {'xl':   3, 'xu':   7, 'yl': -5, 'yu': -4},  #6
         {'xl':   6, 'xu':   7, 'yl': -4, 'yu':  7},  #7
         {'xl':   6, 'xu':   9, 'yl':  4, 'yu':  5},  #8
         {'xl':   7, 'xu':   9, 'yl': -7, 'yu': -4},  #9
         {'xl':   8, 'xu':   9, 'yl': -4, 'yu':  2}]  #10

for ii in range(nz):
    h[(ii*4+0):(ii*4+4)] = np.array([[ x_lim[ii]['xu']+Mx],
                                     [-x_lim[ii]['xl']+Mx],
                                     [ x_lim[ii]['yu']+My],
                                     [-x_lim[ii]['yl']+My]])

# Constraints on particular a limits
dif = [1, 5]  # The set that has different limits
for ii in range(len(dif)):
    h[(nz+ii)*4:(nz+ii)*4+4] = np.array([[1.1],
                                         [1.1],
                                         [1.1],
                                         [1.1]])

h[(nz+len(dif))*4:(nz+len(dif)+1)*4] = np.array([[1],
                                                 [1],
                                                 [1],
                                                 [1]])

h[(nz+len(dif)+1)*4:(nz+len(dif)+2)*4] = np.array([[1],
                                                   [1],
                                                   [1],
                                                   [1]])

Q = np.diag([1, 1, 1/5, 1/5])
R = np.diag([1, 1])/10
QN = np.diag([1, 1, 1/5, 1/5])

def plot_result(sol_x, sol_x_gcs, color, linestyle):
    ax = plt.gca()
    ax.set_aspect('equal') 
    ax.set_facecolor("gray")

    for item in x_lim:
        rectangle = plt.Rectangle((item['xl'], item['yl']), (item['xu']-item['xl']), (item['yu']-item['yl']), fc='white', zorder=2)
        ax.add_patch(rectangle)

    ax.plot(x0_GBD[0], x0_GBD[1], marker='o', color='green', markersize=15)
    ax.plot(x_goal[0], x_goal[1], marker='*', color='red', markersize=15)

    sol_x_gcs = sol_x_gcs.transpose()

    # ax.plot(sol_x_gcs[:, 0], sol_x_gcs[:, 1], '.', color='blue', markersize=8)
    # ax.plot(sol_x[:, 0], sol_x[:, 1], marker='.', color=color, linewidth=2, markersize=12, ls=linestyle)
    ax.scatter(sol_x_gcs[:, 0], sol_x_gcs[:, 1], color='blue', s=8, zorder=3)
    ax.scatter(sol_x[:, 0], sol_x[:, 1], color=color, s=12, zorder=4)

    ang = np.arctan2(sol_x[:, 3], sol_x[:, 2])
    ang_gcs = np.arctan2(sol_x_gcs[:, 3], sol_x_gcs[:, 2])
    ax.plot([sol_x_gcs[:, 0]-0.15*np.cos(ang_gcs), sol_x_gcs[:, 0]+0.15*np.cos(ang_gcs)], [sol_x_gcs[:, 1]-0.15*np.sin(ang_gcs) , sol_x_gcs[:, 1]+0.15*np.sin(ang_gcs)], color='blue', linewidth=1)
    ax.plot([sol_x[:, 0]-0.15*np.cos(ang), sol_x[:, 0]+0.15*np.cos(ang)], [sol_x[:, 1]-0.15*np.sin(ang) , sol_x[:, 1]+0.15*np.sin(ang)], color=color, linewidth=2)

    # ax.quiver(sol_x[:, 0], sol_x[:, 1], sol_x[:, 2], sol_x[:, 3], color='red', scale=3)

    # plt.show()

GBD_solver = GBD_MLD.GBD()

# Setup GCS
gcs = GCS()
gcs.setup_problem()
Q_gcs = np.diag([1, 1, 1/5, 1/5, 1/10, 1/10])

time_consumed = []

num_loop = 200
dT_dyn = 0.2
costM = 100000  # was 100000. Smaller works worse.
cost_s = np.zeros([nz])
cost_t = np.zeros([nz])
cost_e = np.zeros([nz, nz, N_GCS-3])
gg = 4


for i_loop in range(num_loop):

    plt.figure()

    print("Iteration {}".format(i_loop))
    print("Initial conditions {}".format(x0_GCS))

    # Solve GCS every gg steps
    if i_loop % gg == 0:
        try:
            # Solve GCS
            sol_x_gcs, sol_binary, sol_z, sol_y = gcs.solve_problem(x0_GCS)

            # Construct cost map
            ys = sol_y['ys']; yt = sol_y['yt']; ye = sol_y['ye']
            zs_H = sol_z['zs_H']; zt_H = sol_z['zt_H']; ze_H = sol_z['ze_H']

            for vv in range(nz):
                if abs(ys[vv]) < 1e-4: cost_s[vv] = costM
                else:                  cost_s[vv] = zs_H[:, vv] @ Q_gcs @ zs_H[:, vv]/ys[vv]
                
                if abs(yt[vv]) < 1e-4: cost_t[vv] = costM
                else:                  cost_t[vv] = zt_H[:, vv] @ Q_gcs @ zt_H[:, vv]/yt[vv]

            for nn in range(N_GCS-3):
                for vv1 in range(nz):
                    for vv2 in range(nz):
                        if abs(ye[vv1, vv2, nn]) < 1e-4: cost_e[vv1, vv2, nn] = costM
                        else:                            cost_e[vv1, vv2, nn] = ze_H[:, vv1, vv2, nn] @ Q_gcs @ ze_H[:, vv1, vv2, nn]/ye[vv1, vv2, nn]
        
            GBD_solver.GCS_cost_map(cost_s, cost_t, cost_e)
        except:
            print(colored("GCS error caught! No GCS warm-start is used!", 'red'))

    # Solve MLD using GBD
    t1 = time.time()
    sol = GBD_solver.main_loop(x0_GBD, h)
    tc = time.time() - t1
    print(colored("Speed " + str(1/(tc)) + " Hz", 'green'))
    time_consumed.append(tc)
    # sol = GBD_solver.solve_full_MIP_problem(x0_GBD, h_d_theta, h)

    sol_x_GBD = np.zeros([N+1, nx])
    for ii in range(N+1):
        for ix in range(nx):
            sol_x_GBD[ii, ix] = sol['x_' + str(ii) + '_' + str(ix)]

    sol_u = np.array([sol['control_x'], sol['control_y']])
    # sol_u = sol_x_gcs.transpose()[1, 4:6]  # GCS only, no GBD

    # plot results
    plot_result(sol_x_GBD, sol_x_gcs, 'black', '--')
    plt.grid()
    plt.savefig(f'figures/MLD_{i_loop}', dpi=400, bbox_inches="tight")

    # Propagate dynamics
    curr_pos = x0_GCS[0:2]; curr_vel = x0_GCS[2:4]
    next_pos = curr_pos + curr_vel*dT_dyn
    next_vel = curr_vel + sol_u*dT_dyn

    x0_GBD = np.array([next_pos[0], next_pos[1], next_vel[0], next_vel[1]])
    x0_GCS = np.array([next_pos[0], next_pos[1], next_vel[0], next_vel[1], 0.0, 0.0])

print("Solving times are {}".format(time_consumed))
print("Averaged GBD solving time is {} ms, or {} Hz".format(1000*np.average(np.array(time_consumed)), 1/np.average(np.array(time_consumed))))
