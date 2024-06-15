import numpy as np
import gurobipy as go
import pdb
from termcolor import colored
import matplotlib.pyplot as plt

np.set_printoptions(precision=3, suppress=True, linewidth=np.inf)

class GCS:

    def __init__(self) -> None:

        self.N = 51
        print(colored('Time step is {} now!!!'.format(self.N), 'red'))
        pdb.set_trace()
        self.nb = 11
        self.nx = 6
        self.nc = 12
        self.Q = np.diag([0, 0, 1/5, 1/5, 1, 1])
        self.inf = 1e4
        self.x_min = -15
        self.x_max = 15

        self.x_lim = [{'xl':-2, 'xu':  0,  'yl': -5, 'yu': -4},  #0
                      {'xl': 1, 'xu':  2,  'yl': -5, 'yu': -4},  #1
                      {'xl': 0, 'xu':  1,  'yl': -4, 'yu':  3},  #2
                      {'xl': 1, 'xu':  3,  'yl':  3, 'yu':  4},  #3
                      {'xl': 3, 'xu':  4,  'yl': -4, 'yu':  3},  #4
                      {'xl': 4, 'xu':  5,  'yl':  3, 'yu':  4},  #5
                      {'xl': 4, 'xu':  6,  'yl': -5, 'yu': -4},  #6
                      {'xl': 6, 'xu':  7,  'yl': -4, 'yu':  7},  #7
                      {'xl': 7, 'xu':  9,  'yl':  4, 'yu':  5},  #8
                      {'xl': 7, 'xu':  9,  'yl': -5, 'yu': -4},  #9
                      {'xl': 8, 'xu':  9,  'yl': -4, 'yu':  2}]  #10
        
        self.vu = 1; self.vl = -1
        self.au = 1; self.al = -1

        self.H = [np.array([[ 1,  0,  0,  0,  0,  0],
                            [-1,  0,  0,  0,  0,  0],
                            [ 0,  1,  0,  0,  0,  0],
                            [ 0, -1,  0,  0,  0,  0],
                            [ 0,  0,  1,  0,  0,  0],
                            [ 0,  0, -1,  0,  0,  0],
                            [ 0,  0,  0,  1,  0,  0],
                            [ 0,  0,  0, -1,  0,  0],
                            [ 0,  0,  0,  0,  1,  0],
                            [ 0,  0,  0,  0, -1,  0],
                            [ 0,  0,  0,  0,  0,  1],
                            [ 0,  0,  0,  0,  0, -1]]) for ii in range(self.nb)]
        
        self.h = []
        for ii in range(self.nb):
            self.h.append(np.array([  self.x_lim[ii]['xu'],   -self.x_lim[ii]['xl'],   self.x_lim[ii]['yu'],   -self.x_lim[ii]['yl'], 
                                                   self.vu,                -self.vl,                self.vu,                -self.vl,
                                                   self.au,                -self.al,                self.au,                -self.al]))
            
                 #0   #1   #2   #3   #4   #5   #6   #7   #8   #9   #10  #11
        gamma = [1.0, 0.1, 1.0, 1.0, 1.0, 0.1, 1.0, 1.0, 1.0, 1.0, 1.0]
        deltaT = 0.65
        self.E = [np.array([[1.0, 0.0, deltaT,    0.0,               0.0,              0.0], 
                            [0.0, 1.0,    0.0, deltaT,               0.0,              0.0],
                            [0.0, 0.0,    1.0,    0.0,  deltaT*gamma[ii],              0.0], 
                            [0.0, 0.0,    0.0,    1.0,               0.0, deltaT*gamma[ii]]]) for ii in range(self.nb)]
                               
        self.x0 = np.array([-1.0, -5.0, 0.0, 0.0, 0.0, 0.0])
        self.xT = np.array([ 8.5,  4.5, 0.0, 0.0, 0.0, 0.0])
            
    def setup_model(self):

        self.m = go.Model('GCS')
        # self.m.setParam('QCPDual', 1)
        self.m.setParam('BarHomogeneous', 1)
        self.m.setParam('NumericFocus', 1)
        self.m.setParam('OptimalityTol', 2e-3)
        self.m.setParam('FeasibilityTol', 2e-3)
        self.m.setParam('BarQCPConvTol', 2e-3)
        # If want strict binary, just put vtype=go.GRB.BINARY for y variables 
        # MIP=10.26  convex=8.36, relaxation gap ~= 20%
        self.x0_v = self.m.addMVar((self.nx,                           ), lb=-self.inf,  ub=self.inf,   name='x0')
        self.ye =   self.m.addMVar((         self.nb, self.nb, self.N-3), lb=0.0,        ub=1.0,        name='ye')
        self.ys =   self.m.addMVar((         self.nb,                  ), lb=0.0,        ub=1.0,        name='ys')
        self.yt =   self.m.addMVar((         self.nb,                  ), lb=0.0,        ub=1.0,        name='yt')
        self.se =   self.m.addMVar((         self.nb, self.nb, self.N-3), lb=0.0,        ub=self.inf,   name='se')
        self.ss =   self.m.addMVar((         self.nb,                  ), lb=0.0,        ub=self.inf,   name='ss')
        self.st =   self.m.addMVar((         self.nb,                  ), lb=0.0,        ub=self.inf,   name='st')
        self.ze_R = self.m.addMVar((self.nx, self.nb, self.nb, self.N-3), lb=self.x_min, ub=self.x_max, name='ze_R')
        self.zs_R = self.m.addMVar((self.nx, self.nb,                  ), lb=self.x_min, ub=self.x_max, name='zs_R')
        self.zt_R = self.m.addMVar((self.nx, self.nb,                  ), lb=self.x_min, ub=self.x_max, name='zt_R')
        self.ze_L = self.m.addMVar((self.nx, self.nb, self.nb, self.N-3), lb=self.x_min, ub=self.x_max, name='ze_L')
        self.zs_L = self.m.addMVar((self.nx, self.nb,                  ), lb=self.x_min, ub=self.x_max, name='zs_L')
        self.zt_L = self.m.addMVar((self.nx, self.nb,                  ), lb=self.x_min, ub=self.x_max, name='zt_L')

    def set_objective(self):
        self.obj = self.se[:, :, :].sum() + self.ss[:].sum() + self.st[:].sum()
        self.m.setObjective(self.obj, go.GRB.MINIMIZE)

        # Epigraph constraints
        for n in range(self.N-3):
            for vl in range(self.nb):
                for vr in range(self.nb):
                    self.m.addConstr(self.se[vl, vr, n]*self.ye[vl, vr, n] >= self.ze_L[:, vl, vr, n] @ self.Q @ self.ze_L[:, vl, vr, n])
    
        for v in range(self.nb):
            self.m.addConstr(self.ss[v]*self.ys[v] >= self.zs_L[:, v] @ self.Q @ self.zs_L[:, v])
            self.m.addConstr(self.st[v]*self.yt[v] >= self.zt_L[:, v] @ self.Q @ self.zt_L[:, v])

    def enforce_unit_flow(self):
        self.m.addConstr(self.ys[:].sum() == 1.0)
        self.m.addConstr(self.yt[:].sum() == 1.0)

    def enforce_degree(self):
        for n in range(self.N-3):
            for vl in range(self.nb):
                self.m.addConstr(self.ye[vl, :, n].sum() <= 1.0)

    def enforce_flow_conservation(self):
        for vm in range(self.nb):
            for n in range(self.N-4):  
                self.m.addConstr(self.ye[:, vm, n].sum() == self.ye[vm, :, n+1].sum())

            self.m.addConstr(self.ys[vm] == self.ye[vm, :, 0].sum())
            self.m.addConstr(self.yt[vm] == self.ye[:, vm, self.N-4].sum())

        for ix in range(self.nx):
            for vm in range(self.nb):
                for n in range(self.N-4):
                    self.m.addConstr(self.ze_R[ix, :, vm, n].sum() == self.ze_L[ix, vm, :, n+1].sum())

                self.m.addConstr(self.zs_R[ix, vm] == self.ze_L[ix, vm, :, 0].sum())
                self.m.addConstr(self.zt_L[ix, vm] == self.ze_R[ix, :, vm, self.N-4].sum())

    def enforce_set_membership(self):
        for n in range(self.N-3):
            for vl in range(self.nb):
                for vr in range(self.nb):
                    for ic in range(self.nc):
                        self.m.addConstr(self.H[vl][ic, :] @ self.ze_L[:, vl, vr, n] <= self.ye[vl, vr, n]*self.h[vl][ic])
                        self.m.addConstr(self.H[vr][ic, :] @ self.ze_R[:, vl, vr, n] <= self.ye[vl, vr, n]*self.h[vr][ic])
                    
        for v in range(self.nb):
            for ic in range(self.nc):
                self.m.addConstr(self.H[v][ic, :] @ self.zs_R[:, v] <= self.ys[v]*self.h[v][ic])
                self.m.addConstr(self.H[v][ic, :] @ self.zt_L[:, v] <= self.yt[v]*self.h[v][ic])

    def enforce_initial_terminal_conditions(self):
        for v in range(self.nb):
            self.m.addConstr(self.zs_L[:, v] == self.ys[v]*self.x0_v, name=f'init_{v}')
            self.m.addConstr(self.zt_R[:, v] == self.yt[v]*self.xT)

    def enforce_dynamics(self):
        for n in range(self.N-3):
            for vl in range(self.nb):
                for vr in range(self.nb):
                    self.m.addConstr(self.ze_R[0:4, vl, vr, n] == self.E[vl] @ self.ze_L[:, vl, vr, n])

        for v in range(self.nb):
            self.m.addConstr(self.zs_R[0:4, v] == self.E[1] @ self.zs_L[:, v])
            self.m.addConstr(self.zt_R[0:4, v] == self.E[6] @ self.zt_L[:, v])
        
    def setup_problem(self):
        self.setup_model()
        self.set_objective()
        self.enforce_unit_flow()
        self.enforce_degree()
        self.enforce_flow_conservation()
        self.enforce_set_membership()
        self.enforce_dynamics()
        self.enforce_initial_terminal_conditions()
        self.m.update()

    def solve_problem(self, x0_new):

        # Update x0 constraints
        self.x0_v.lb = self.x0_v.ub = x0_new

        # for v in range(self.nb):
        #     for ix in range(self.nx): 
        #         self.m.remove(self.m.getConstrByName(f'init_{v}[{ix}]'))
        #     self.m.addConstr(self.zs_L[:, v] == self.ys[v]*x0_new, name=f'init_{v}')

        self.m.update()

        self.m.optimize()
        
        if self.m.SolCount > 0:

            self.sol_ye = self.ye.X; self.sol_ys = self.ys.X; self.sol_yt = self.yt.X
            self.sol_se = self.se.X; self.sol_ss = self.ss.X; self.sol_st = self.st.X
            self.sol_ze_T = self.ze_R.X; self.sol_zs_T = self.zs_R.X; self.sol_zt_T = self.zt_R.X
            self.sol_ze_H = self.ze_L.X; self.sol_zs_H = self.zs_L.X; self.sol_zt_H = self.zt_L.X

            # Recover x
            xs = sum(self.sol_zs_H[:, ii] for ii in range(self.nb))
            xt = sum(self.sol_zt_T[:, ii] for ii in range(self.nb))

            sol_x = np.zeros([self.nx, self.N])
            sol_x[:, 0] = xs; sol_x[:, -1] = xt
            for n in range(1, self.N-2):
                sol_x[:, n] = sum(self.sol_ze_H[:, v1, v2, n-1] for v1 in range(self.nb) for v2 in range(self.nb))

            sol_x[:, self.N-2] = sum(self.sol_zt_H[:, v] for v in range(self.nb))

            # Recover z
            ye_portion = np.zeros([self.nb, self.N-3])  # The portion of each node carrying x
            for v1 in range(self.nb):
                for n in range(self.N-3):
                    ye_portion[v1, n] = self.sol_ye[v1, :, n].sum()

            sol_binary = np.zeros([self.nb, self.N-1])
            sol_binary[:, 0] = self.sol_ys; sol_binary[:, -1] = self.sol_yt
            sol_binary[:, 1:-1] = ye_portion

            sol_z = {'zs_H': self.sol_zs_H, 'zs_T': self.sol_zs_T, 
                     'zt_H': self.sol_zt_H, 'zt_T': self.sol_zt_T, 
                     'ze_H': self.sol_ze_H, 'ze_T': self.sol_ze_T}
            
            sol_y = {'ye': self.sol_ye, 'ys': self.sol_ys, 'yt': self.sol_yt}

            return sol_x, sol_binary, sol_z, sol_y

        else:
            print(colored("Problem infeasible!!", 'red'))     

    def plot_result(self, sol_x, color, linestyle):

        plt.figure()
        ax = plt.gca()
        ax.set_aspect('equal') 
        ax.set_facecolor("gray")

        for item in self.x_lim:
            rectangle = plt.Rectangle((item['xl'], item['yl']), (item['xu']-item['xl']), (item['yu']-item['yl']), fc='white')
            ax.add_patch(rectangle)

        ax.plot(self.x0[0], self.x0[1], marker='o', color='green', markersize=12)
        ax.plot(self.xT[0], self.xT[1], marker='*', color='red', markersize=12)

        ax.plot(sol_x[:, 0], sol_x[:, 1], marker='.', color=color, markersize=6, ls=linestyle)

        plt.show()

    def plot_scene(self):
            
        plt.figure()
        ax = plt.gca()
        ax.grid() 
        ax.set_aspect('equal') 
        ax.set_facecolor("gray")

        for item in self.x_lim:
            rectangle = plt.Rectangle((item['xl'], item['yl']), (item['xu']-item['xl']), (item['yu']-item['yl']), fc='white')
            ax.add_patch(rectangle)

        ax.plot(self.x0[0], self.x0[1], marker='o', color='green', markersize=12)
        ax.plot(self.xT[0], self.xT[1], marker='*', color='red', markersize=12)

        plt.show()

    def round_edges(self):

        # TODO: rounding needs to consider those y only constraints. It is not hard but you do need to consider those.
        # TODO: For example, use the algorithm proposed by that paper to do this well.
        ye_r = np.round(self.ye.X)
        ys_r = np.round(self.ys.X)
        yt_r = np.round(self.yt.X)

        return ye_r, ys_r, yt_r

def main():
    gcs = GCS()
    gcs.plot_scene()
    gcs.setup_problem()
    sol_x, sol_binary, sol_z, sol_y = gcs.solve_problem(gcs.x0)
    gcs.plot_result(sol_x.transpose(), 'blue', '--')

if __name__ == "__main__":
    main()
