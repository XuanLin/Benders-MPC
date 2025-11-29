import numpy as np
import matplotlib.pyplot as plt
from termcolor import colored
import scipy.io
import time
import os

from humanoid_balancing_dynamics_pybullet import HumanoidBalancingDynamics
import humanoid_balancing_cpp

# Simulation constants
T_SIM = 20.0
DELTA_T_CONTROL = 0.02

class HumanoidLogger:
    """Logger for humanoid balancing simulation."""
    
    def __init__(self):
        self.time = []
        self.theta = []
        self.dtheta = []
        self.states = []
        self.tau_ankle = []
        self.f_right = []
        self.f_left = []
        self.f_ground = []
        self.contact_forces_R = []
        self.contact_forces_L = []
        self.costs = []
        self.solver_stats = {
            'iterations': [],
            'solve_times': [],
            'opt_cuts': [],
            'feas_cuts': []
        }
        self.contact_strategy = {
            'right_steps': [],
            'left_steps': [],
            'no_contact_steps': []
        }
        self.total_time = 0.0
        self.total_iterations = 0
        self.solve_count = 0
    
    def log_step(self, t, state, controls, solve_info):
        """Log a single simulation step."""
        self.time.append(t)
        self.theta.append(state[0])
        self.dtheta.append(state[1])
        self.states.append(state)
        
        self.tau_ankle.append(controls['tau_ankle'])
        self.f_right.append(controls['f_right'])
        self.f_left.append(controls['f_left'])
        self.f_ground.append(solve_info.get('f_ground', 0.0))
        
        self.contact_forces_R.append(controls.get('contact_force_R', 0.0))
        self.contact_forces_L.append(controls.get('contact_force_L', 0.0))
        
        self.costs.append(solve_info['cost'])
        self.solver_stats['iterations'].append(solve_info['num_iter'])
        self.solver_stats['solve_times'].append(solve_info['solve_time'])
        self.solver_stats['opt_cuts'].append(solve_info['num_opt_cut'])
        self.solver_stats['feas_cuts'].append(solve_info['num_feas_cut'])
        
        self.contact_strategy['right_steps'].append(solve_info['right_contact_steps'])
        self.contact_strategy['left_steps'].append(solve_info['left_contact_steps'])
        self.contact_strategy['no_contact_steps'].append(solve_info['no_contact_steps'])
        
        self.total_time += solve_info['solve_time']
        self.total_iterations += solve_info['num_iter']
        self.solve_count += 1
        
        self._print_stats(solve_info)
    
    def _print_stats(self, solve_info):
        """Print current solver statistics."""
        avg_time = self.total_time / self.solve_count
        avg_iter = self.total_iterations / self.solve_count
        
        print(colored(f"Step {self.solve_count}: {1000*solve_info['solve_time']:.2f}ms "
                     f"({1/solve_info['solve_time']:.1f}Hz), "
                     f"{solve_info['num_iter']} iter", 'cyan'))
        print(colored(f"  Average: {1000*avg_time:.2f}ms ({1/avg_time:.1f}Hz), "
                     f"{avg_iter:.1f} iter", 'green'))
        print(colored(f"  Contact plan: R={solve_info['right_contact_steps']}, "
                     f"L={solve_info['left_contact_steps']}, "
                     f"None={solve_info['no_contact_steps']}", 'yellow'))
        print(colored(f"  Cuts: Opt={solve_info['num_opt_cut']}, "
                     f"Feas={solve_info['num_feas_cut']}", 'magenta'))
    
    def save_results(self, filename, params):
        """Save results to a .mat file."""
        results = {
            'time_traj': np.array(self.time),
            'theta_traj': np.array(self.theta),
            'dtheta_traj': np.array(self.dtheta),
            'tau_ankle_traj': np.array(self.tau_ankle),
            'f_right_traj': np.array(self.f_right),
            'f_left_traj': np.array(self.f_left),
            'f_ground_traj': np.array(self.f_ground),
            'contact_force_R_traj': np.array(self.contact_forces_R),
            'contact_force_L_traj': np.array(self.contact_forces_L),
            'cost_traj': np.array(self.costs),
            'solve_time_traj': np.array(self.solver_stats['solve_times']),
            'num_iter_traj': np.array(self.solver_stats['iterations']),
            'opt_cuts_traj': np.array(self.solver_stats['opt_cuts']),
            'feas_cuts_traj': np.array(self.solver_stats['feas_cuts']),
            'right_contact_steps': np.array(self.contact_strategy['right_steps']),
            'left_contact_steps': np.array(self.contact_strategy['left_steps']),
            'no_contact_steps': np.array(self.contact_strategy['no_contact_steps']),
            'N': params.N,
            'dT': params.dT
        }
        scipy.io.savemat(filename, mdict=results)
        print(colored(f"\nResults saved to {filename}", 'green'))
    
    def plot_results(self):
        """Generate plots of the simulation results."""
        fig, axes = plt.subplots(4, 2, figsize=(14, 12))
        
        # State trajectory
        axes[0, 0].plot(self.time, np.rad2deg(self.theta), 'b-', linewidth=2)
        axes[0, 0].set_ylabel('Angle [deg]')
        axes[0, 0].set_xlabel('Time [s]')
        axes[0, 0].grid(True)
        axes[0, 0].set_title('Body Angle')
        
        axes[0, 1].plot(self.time, np.rad2deg(self.dtheta), 'r-', linewidth=2)
        axes[0, 1].set_ylabel('Angular Velocity [deg/s]')
        axes[0, 1].set_xlabel('Time [s]')
        axes[0, 1].grid(True)
        axes[0, 1].set_title('Angular Velocity')
        
        # Controls
        axes[1, 0].plot(self.time, self.tau_ankle, 'g-', linewidth=2)
        axes[1, 0].set_ylabel('Torque [Nm]')
        axes[1, 0].set_xlabel('Time [s]')
        axes[1, 0].grid(True)
        axes[1, 0].set_title('Ankle Torque')
        
        axes[1, 1].plot(self.time, self.f_right, 'b-', label='Right', linewidth=2)
        axes[1, 1].plot(self.time, self.f_left, 'r-', label='Left', linewidth=2)
        axes[1, 1].set_ylabel('Force [N]')
        axes[1, 1].set_xlabel('Time [s]')
        axes[1, 1].legend()
        axes[1, 1].grid(True)
        axes[1, 1].set_title('Wall Contact Forces')
        
        # Ground reaction force
        axes[2, 0].plot(self.time, self.f_ground, 'k-', linewidth=2)
        axes[2, 0].set_ylabel('Force [N]')
        axes[2, 0].set_xlabel('Time [s]')
        axes[2, 0].grid(True)
        axes[2, 0].set_title('Ground Reaction Force')
        
        # Solver performance
        axes[2, 1].plot(self.time, np.array(self.solver_stats['solve_times'])*1000, 'g-', linewidth=2)
        axes[2, 1].set_ylabel('Time [ms]')
        axes[2, 1].set_xlabel('Time [s]')
        axes[2, 1].grid(True)
        axes[2, 1].set_title('Solve Time')
        
        axes[3, 0].plot(self.time, self.solver_stats['iterations'], 'b-', linewidth=2)
        axes[3, 0].set_ylabel('Iterations')
        axes[3, 0].set_xlabel('Time [s]')
        axes[3, 0].grid(True)
        axes[3, 0].set_title('GBD Iterations')
        
        # Cuts
        axes[3, 1].plot(self.time, self.solver_stats['opt_cuts'], 'g-', label='Optimality', linewidth=2)
        axes[3, 1].plot(self.time, self.solver_stats['feas_cuts'], 'r-', label='Feasibility', linewidth=2)
        axes[3, 1].set_ylabel('Number of Cuts')
        axes[3, 1].set_xlabel('Time [s]')
        axes[3, 1].legend()
        axes[3, 1].grid(True)
        axes[3, 1].set_title('Stored Cuts')
        
        plt.tight_layout()
        plt.savefig('humanoid_balancing_results.png', dpi=150)
        print(colored("Plots saved to humanoid_balancing_results.png", 'green'))
        plt.show()


def main():
    """Main simulation loop."""
    print(colored("\n" + "="*60, 'cyan'))
    print(colored("  Humanoid Balancing with GBD-MPC and Wall Contacts", 'cyan', attrs=['bold']))
    print(colored("="*60 + "\n", 'cyan'))
    
    # Load parameters
    params = humanoid_balancing_cpp.HumanoidBalancingParams()
    
    print(colored("System Parameters:", 'yellow', attrs=['bold']))
    print(f"  Mass: {params.m:.1f} kg")
    print(f"  CoM height: {params.h_com:.2f} m")
    print(f"  Arm height: {params.h_arm:.2f} m")
    print(f"  Arm stretch length: {params.l_arm:.2f} m")
    print(f"  Wall distances: Left={params.dL:.3f}m, Right={params.dR:.3f}m")
    print(f"  Friction coef: {params.mu:.2f}")
    print(f"  Max ankle torque: {params.tau_max:.0f} Nm\n")
    
    print(colored("MPC Parameters:", 'yellow', attrs=['bold']))
    print(f"  Horizon: {params.N} steps")
    print(f"  Time step: {params.dT:.3f} s ({1/params.dT:.0f} Hz)")
    print(f"  Control loop: {DELTA_T_CONTROL:.3f} s ({1/DELTA_T_CONTROL:.0f} Hz)\n")
    
    # Initialize state - small perturbation from upright
    initial_theta = np.deg2rad(5.0)  # 5 degree initial lean
    initial_dtheta = 0.0
    initial_state = np.array([initial_theta, initial_dtheta])
    
    print(colored(f"Initial state: theta={np.rad2deg(initial_theta):.1f}°, "
                 f"dtheta={np.rad2deg(initial_dtheta):.1f}°/s\n", 'yellow'))
    
    # Initialize dynamics
    print(colored("Initializing PyBullet simulation...", 'cyan'))
    dynamics = HumanoidBalancingDynamics(
        m=params.m,
        h_com=params.h_com,
        h_arm=params.h_arm,
        Icom=params.Icom,
        dR=params.dR,
        dL=params.dL,
        mu=params.mu,
        tau_max=params.tau_max,
        theta_ini=initial_theta,
        dtheta_ini=initial_dtheta,
    )
    
    # Initialize solver
    print(colored("Initializing GBD solver...", 'cyan'))
    gbd_solver = humanoid_balancing_cpp.HumanoidBalancingGBDSolver(params)
    
    # Initialize logger
    logger = HumanoidLogger()
    
    # Get constraint vector
    h_theta = np.copy(params.h_theta)
    
    # Main simulation loop
    num_steps = int(T_SIM / DELTA_T_CONTROL)
    print(colored(f"\nStarting simulation for {T_SIM:.1f}s ({num_steps} steps)...\n", 'cyan', attrs=['bold']))
    
    # Initialize state
    current_state = initial_state
    controls = {'tau_ankle': 0.0, 'f_right': 0.0, 'f_left': 0.0}

    for i_step in range(num_steps):
        t = i_step * DELTA_T_CONTROL
        
        if i_step == 0:
            dynamics.start_logging()
        
        print(colored(f"\n{'='*60}", 'white'))
        print(colored(f"Step {i_step+1}/{num_steps} (t={t:.2f}s)", 'white', attrs=['bold']))
        print(colored(f"State: θ={np.rad2deg(current_state[0]):.2f}°, "
                    f"dθ={np.rad2deg(current_state[1]):.2f}°/s", 'white'))
        
        # Solve MPC FIRST (before forward step)
        solve_start = time.time()
        sol = gbd_solver.solve(current_state, h_theta)
        sol['solve_time'] = time.time() - solve_start
        
        # Extract controls
        controls = {
            'tau_ankle': sol['tau_ankle'],
            'f_right': sol['f_right'],
            'f_left': sol['f_left']
        }
        
        print(colored(f"Controls: τ={controls['tau_ankle']:.2f}Nm, "
              f"f_R={controls['f_right']:.2f}N, f_L={controls['f_left']:.2f}N", 'white'))
        
        # Apply control and step dynamics
        dynamics_result = dynamics.forward(
            tau_ankle=controls['tau_ankle'],
            deltaT=DELTA_T_CONTROL
        )
        
        # Update state for NEXT iteration
        current_state = np.array([dynamics_result['theta'], dynamics_result['dtheta']])
        
        # Contact forces
        controls['contact_force_R'] = dynamics_result['contact_force_R']
        controls['contact_force_L'] = dynamics_result['contact_force_L']

        # Print contact forces
        print(colored(f"Contact Forces: Right={controls['contact_force_R']:.2f}N, "
                    f"Left={controls['contact_force_L']:.2f}N", 'magenta'))

        # Log results
        logger.log_step(t, current_state, controls, sol)
        
        if i_step == num_steps - 1:
            dynamics.stop_logging()
    
    # Stop logging
    dynamics.stop_logging()
    
    # Print summary
    print(colored("\n" + "="*60, 'cyan'))
    print(colored("Simulation Complete!", 'green', attrs=['bold']))
    print(colored("="*60, 'cyan'))
    print(f"\nPerformance Summary:")
    print(f"  Average solve time: {1000*logger.total_time/logger.solve_count:.2f} ms")
    print(f"  Average frequency: {logger.solve_count/logger.total_time:.1f} Hz")
    print(f"  Average iterations: {logger.total_iterations/logger.solve_count:.1f}")
    print(f"  Total cuts - Opt: {logger.solver_stats['opt_cuts'][-1]}, "
          f"Feas: {logger.solver_stats['feas_cuts'][-1]}")
    
    # Save results
    os.makedirs('saved_results', exist_ok=True)
    logger.save_results('saved_results/humanoid_balancing_gbd.mat', params)
    
    # Generate plots
    print(colored("\nGenerating plots...", 'cyan'))
    logger.plot_results()
    
    print(colored("\nDone!", 'green', attrs=['bold']))


if __name__ == '__main__':
    main()