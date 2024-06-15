import numpy as np
from scipy import linalg as la

# Define the problem
N = 10 # N=10 and dT=0.02 also makes the non shifted cut feasible. Otherwise, how to decrease the dual values?
dT = 0.02  # 0.1 is too large! 0.012 for N=15, 0.02 for N=10
mc = 1.0
mp = 0.4
ll = 0.6
k1 = 50
k2 = 50
d_left = 0.40  # Was 0.35 
d_right = 0.35
d_max = 0.6 # was 0.5
lam_max = 30.0
u_max = 20.0
g = 9.81
inf = 1e5

dim_x = 4
dim_u = 3

# States are x, theta, dx, dtheta
A = np.array([[0.0,               0.0, 1.0, 0.0],
              [0.0,               0.0, 0.0, 1.0],
              [0.0,           g*mp/mc, 0.0, 0.0],
              [0.0, g*(mc+mp)/(ll*mc), 0.0, 0.0]])  # This A is from dx = Ax + ..., the discretized system should be x[k+1] - x[k] = ()*dT, not x[k+1] = Ax[k] + ...
                                                    # Equivalently, A is changed
B = np.array([[      0.0,        0.0,         0.0], 
              [      0.0,        0.0,         0.0], 
              [     1/mc,        0.0,         0.0], 
              [1/(ll*mc),  1/(ll*mp),  -1/(ll*mp)]])

A_d = np.eye(dim_x) + A*dT
B_d = B*dT

# Q = np.array([[1.0, 0.0, 0.0, 0.0], 
#               [0.0, 2000.0, 0.0, 0.0], 
#               [0.0, 0.0, 100.0, 0.0],
#               [0.0, 0.0, 0.0, 10.0]])/10.0

# R = np.ones([dim_u, dim_u])/100.0

Q = np.array([[1.0,   0.0, 0.0,  0.0], 
              [0.0, 50.0,  0.0,  0.0], 
              [0.0,   0.0, 1.0,  0.0],
              [0.0,   0.0, 0.0, 50.0]])

R = np.array([[1.0, 0.0, 0.0],
              [0.0, 1.0, 0.0],
              [0.0, 0.0, 1.0]])/10

QN = la.solve_discrete_are(A_d, B_d, Q, R)

d = np.zeros(4)

x_ini = 0.0
theta_ini = 0.0/180.0*np.pi
dx_ini = 0.0 
dtheta_ini = 0.0

E = np.array([[-1.0,  ll, 0.0, 0.0], 
              [ 1.0, -ll, 0.0, 0.0]])
F = np.array([[1/k1, 0.0], 
              [0.0, 1/k2]])
H = np.array([[0.0],
              [0.0]])
c = np.array([d_right, d_left]) # To prevent bug, make this row vector

# Think about taking the x limit down, so your Benders won't have infeasibility issues
x_lb = np.array([-d_max, -np.pi/2, -2*d_max/dT, -np.pi/dT])
x_ub = np.array([ d_max,  np.pi/2,  2*d_max/dT,  np.pi/dT])

u_lb = np.array([-u_max])
u_ub = np.array([ u_max])

lam_lb = np.array([0.0, 0.0])
lam_ub = np.array([lam_max, lam_max])  # The peak has nothing to do with this upper value
