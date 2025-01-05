import numpy as np
import os
import sys
from pybullet_dynamics.cart_pole_soft_wall_dynamics_pybullet import cart_pole_dynamics
import matplotlib.pyplot as plt
from termcolor import colored
import scipy.io
import time
import pdb

import cart_pole_cpp

# Constants
T_SIM = 40.0
DELTA_T_DYN = 0.005  # Hz control rate (200Hz)

class TrajectoryLogger:
    def __init__(self):
        self.time = []
        self.x_animate = []
        self.theta_animate = []
        self.states = []
        self.controls = []
        self.contact_forces = []
        self.ll_traj = []
        self.costs = []
        self.solver_stats = {
            'iterations': [],
            'solve_times': [],
            'opt_cuts': [],
            'feas_cuts': []
        }
        self.total_time = 0.0
        self.total_iterations = 0
        self.contact_count = 0
    
    def log_step(self, t, state, control, contact_force, ll, solve_info):
        """Log a single simulation step."""
        self.time.append(t)
        self.x_animate.append(state[0])
        self.theta_animate.append(state[1])
        self.states.append(state)
        self.controls.append([control])
        self.contact_forces.append(contact_force)
        self.ll_traj.append(ll)
        
        if solve_info['planned_contact']:
            self.costs.append(solve_info['cost'])
            self.solver_stats['iterations'].append(solve_info['num_iter'])
            self.solver_stats['solve_times'].append(solve_info['solve_time'])
            self.solver_stats['opt_cuts'].append(solve_info['num_opt_cut'])
            self.solver_stats['feas_cuts'].append(solve_info['num_feas_cut'])
            
            self.total_time += solve_info['solve_time']
            self.total_iterations += solve_info['num_iter']
            self.contact_count += 1
            
            self._print_stats()
    
    def _print_stats(self):
        """Print current solver statistics."""
        avg_time = self.total_time / self.contact_count
        print(colored(f"MPC Spending on average {1000*avg_time:.2f} ms, or {1/avg_time:.2f} Hz", 'green'))
        print(colored(f"The number of iterations are {self.solver_stats['iterations']}", 'green'))
        print(colored(f"The average number of iterations is {self.total_iterations/self.contact_count:.2f}", 'green'))
    
    def save_results(self, filename, params):
        """Save results to a .mat file."""
        results = {
            'time_traj': np.array(self.time),
            'cost_Benders': np.array(self.costs),
            'time_Benders': np.array(self.solver_stats['solve_times']),
            'num_iter_traj': np.array(self.solver_stats['iterations']),
            'opt_cuts_traj': np.array(self.solver_stats['opt_cuts']),
            'feas_cuts_traj': np.array(self.solver_stats['feas_cuts']),
            'x_traj': np.array(self.states),
            'N': params.N,
            'dT': params.dT
        }
        scipy.io.savemat(filename, mdict=results)


def main():
    # Load parameters
    params = cart_pole_cpp.CartPoleParams()
    param_dict = {name: getattr(params, name) for name in 
                  ['N', 'mc', 'mp', 'll', 'k1', 'k2', 'd_left', 'd_right', 'd_max', 'u_max', 'lam_max', 'dT', 'g', 'x_lb', 'x_ub']}
    globals().update(param_dict)  # Make parameters available in global scope
    
    # Initialize state and constraints
    initial_state = np.zeros(4)  # [x, theta, dx, dtheta]
    h_theta = np.copy(params.h_theta)
    
    # Setup simulation
    num_MPC = int(T_SIM / DELTA_T_DYN)
    load_wall_motion = True
    save_wall_motion = not load_wall_motion
    
    # Initialize wall motion
    if load_wall_motion:
        wall_motion = scipy.io.loadmat('Hz_contact_experiment/Hz_contact_noise/wall_motion_100s.mat')
        print(colored("Loading wall_motion from file", 'red'))
        pdb.set_trace()
    else:
        delta_d_left_rand = 0.0
        delta_d_right_rand = 0.0
        list_delta_d_left = []
        list_delta_d_right = []
    
    # Initialize dynamics and solver
    dynamics = cart_pole_dynamics(mc, mp, ll, k1, k2, d_left, d_right, d_max, u_max, *initial_state, 1)
    gbd_solver = cart_pole_cpp.CartPoleGBDSolver(params)
    logger = TrajectoryLogger()
    
    # Main simulation loop
    u_input = 0.0
    for i_loop in range(num_MPC):
        print(f"Number of MPC iterations is {i_loop}")
        t = i_loop * DELTA_T_DYN
        
        # Get wall positions
        if load_wall_motion:
            delta_d_left = wall_motion['delta_d_left'][0][i_loop]
            delta_d_right = wall_motion['delta_d_right'][0][i_loop]
        else:
            delta_d_left_rand += np.random.normal(0.0, 0.2) * DELTA_T_DYN
            delta_d_right_rand += np.random.normal(0.0, 0.2) * DELTA_T_DYN
            delta_d_left = 0.03 * np.sin(10 * np.pi * i_loop / 1000) + delta_d_left_rand
            delta_d_right = 0.03 * np.sin(10 * np.pi * i_loop / 1000) + delta_d_right_rand
            list_delta_d_left.append(delta_d_left)
            list_delta_d_right.append(delta_d_right)
        
        if i_loop == 0: dynamics.start_logging()
        # Forward step
        ret = dynamics.forward(u=u_input, deltaT=DELTA_T_DYN, delta_d_left=delta_d_left, delta_d_right=delta_d_right)
        if i_loop == num_MPC - 1: dynamics.stop_logging()

        # Update state and constraints
        current_state = np.array([ret['x'], ret['theta'], ret['dx'], ret['dtheta']])
        c_left = d_left + delta_d_left
        c_right = d_right + delta_d_right
        
        h_theta[2] = -c_right + x_ub[0] - x_lb[0]
        h_theta[3] = c_right
        h_theta[4] = -c_left + x_ub[0] - x_lb[0]
        h_theta[5] = c_left
        
        # Solve MPC
        solve_start = time.time()
        sol = gbd_solver.solve(current_state, h_theta)
        sol['solve_time'] = time.time() - solve_start
        print(colored(f"Speed {1/sol['solve_time']:.2f} Hz", 'green'))
        
        # Update control
        u_input = sol['control']
        
        # Log results
        logger.log_step(t, current_state, u_input, ret['contact_force'], ll, sol)
    
    # Save results
    if not globals().get('use_Gurobi', False):
        logger.save_results(f'saved_results/t_spend_Benders_{1}.mat', params)
    
    # Save wall motion if generated
    if save_wall_motion:
        scipy.io.savemat('saved_noise/wall_motion.mat', 
                        mdict={'delta_d_left': np.array(list_delta_d_left),
                              'delta_d_right': np.array(list_delta_d_right)})


if __name__ == '__main__':
    main()
