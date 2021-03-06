#Integrate a tetrahedral mesh forward in time with synchronous time updates according to the method presented by dormand and prince
#Tim Tyree
#12/9/2020

import numpy as np
from numba import njit
from ..model import *
from . import *


def get_one_step_explicit_dormand_prince_method(mu,lam,gamma):
	"""Explicit Runge-Kutta method of order 5(4).

	This uses the Dormand-Prince pair of formulas [1]_. The error is controlled
	assuming accuracy of the fourth-order method accuracy, but steps are taken
	using the fifth-order accurate formula (local extrapolation is done).
	A quartic interpolation polynomial is used for the dense output [2]_.

	Can be applied in the complex domain.

	Parameters
	----------
	mu: the first Lamé parameter. nonnegative float.
	lam: the second Lamé parameter. nonnegative float.
	gamma: the Rayleigh damping parameter. nonnegative float

	References
	----------
	.. [1] J. R. Dormand, P. J. Prince, "A family of embedded Runge-Kutta
		   formulae", Journal of Computational and Applied Mathematics, Vol. 6,
		   No. 1, pp. 19-26, 1980.
	.. [2] L. W. Shampine, "Some Practical Runge-Kutta Formulas", Mathematics
		   of Computation,, Vol. 46, No. 173, pp. 135-150, 1986.
	"""
	zero_mat = np.zeros((4,3))
	#one step implicit midpoint rule
	# local_one_step_implicit_midpoint_rule = get_local_one_step_implicit_midpoint_rule(mu,lam,gamma,num_iter = 30)
	#one step explicit Newmark method (misnamed, I know)
	# compute_one_step_forward_euler_method=get_compute_one_step_forward_euler_method(mu,lam,gamma)
	# step_to_time = compute_one_step_forward_euler_method
	compute_flow_map = get_flow_map(mu,lam,gamma)

	# @njit
	# def f(t,x,v,K_tau,tau_of_K,K_masses,Bm,zero_mat):
	#     x_out, v_out = compute_one_step_forward_euler_method(t,x,v,K_masses,K_tau,tau_of_K,Bm,zero_mat)
	#     return x_out, v_out
	@njit
	def f(t, x, v, Bm, K_masses):
		dxdt, dvdt = compute_flow_map(x, v, Bm, K_masses)
		return dxdt, dvdt

	#from print(inspect.getsource(scipy.integrate.RK45))
	C = np.array([0, 1/5, 3/10, 4/5, 8/9, 1])
	A = np.array([
		[0, 0, 0, 0, 0],
		[1/5, 0, 0, 0, 0],
		[3/40, 9/40, 0, 0, 0],
		[44/45, -56/15, 32/9, 0, 0],
		[19372/6561, -25360/2187, 64448/6561, -212/729, 0],
		[9017/3168, -355/33, 46732/5247, 49/176, -5103/18656]])
	B = np.array([35/384, 0, 500/1113, 125/192, -2187/6784, 11/84])
	E = np.array([-71/57600, 0, 71/16695, -71/1920, 17253/339200, -22/525,1/40])

	@njit
	def one_step_explicit_dormand_prince_method(h,x,v,K_masses,K_tau,tau_of_K,Bm):
		"""Explicit Runge-Kutta method of order 5(4).

			This uses the Dormand-Prince pair of formulas [1]_. The error is controlled
			assuming accuracy of the fourth-order method accuracy, but steps are taken
			using the fifth-order accurate formula (local extrapolation is done).
			A quartic interpolation polynomial is used for the dense output [2]_.

			Can be applied in the complex domain?

			Parameters
			----------
			mu: the first Lamé parameter. nonnegative float.
			lam: the second Lamé parameter. nonnegative float.
			gamma: the Rayleigh damping parameter. nonnegative float

			References
			----------
			.. [1] J. R. Dormand, P. J. Prince, "A family of embedded Runge-Kutta
				   formulae", Journal of Computational and Applied Mathematics, Vol. 6,
				   No. 1, pp. 19-26, 1980.
			.. [2] L. W. Shampine, "Some Practical Runge-Kutta Formulas", Mathematics
				   of Computation,, Vol. 46, No. 173, pp. 135-150, 1986.
			"""

		#synchronize the positions of the nodes to the same time
		#t = max(np.max(K_tau), tau_of_K)
		#TODO: try this --> t = tau_of_K + h
		t = tau_of_K

		#update all nodes using current velocity to the most recent time of the element
		#NOTE: consider updating node positions to final time using current velocity
		for a in range(4):
			x[a] += v[a] * (t - K_tau[a])


		# DT = h
		k1x, k1v = f(t,x,v, Bm, K_masses)
		# k1x = (k1x - x)/DT; k1v = (k1v - v)/DT

		c2 = C[1]; a21 = A[1,0];# DT = c2*h
		k2x, k2v = f(t, x+h*a21*k1x, v+h*a21*k1v, Bm, K_masses)
		# k2x = (k2x - x)/DT; k2v = (k2v - v)/DT

		c3 = C[2]; a31 = A[2,0]; a32 = A[2,1]; #DT = c3*h
		k3x, k3v = f(t+c3*h, x+h*(a31*k1x + a32*k2x), v+h*(a31*k1v + a32*k2v), Bm, K_masses)
		# k3x = (k3x - x)/DT; k3v = (k3v - v)/DT

		c4 = C[3]; a41 = A[3,0]; a42 = A[3,1]; a43 = A[3,2]; #DT = c4*h
		k4x, k4v = f(t+c4*h, x+h*(a41*k1x + a42*k2x + a43*k3x), v+h*(a41*k1v + a42*k2v + a43*k3v), Bm, K_masses)
		# k4x = (k4x - x)/DT; k4v = (k4v - v)/DT

		c5 = C[4]; a51 = A[4,0]; a52 = A[4,1]; a53 = A[4,2]; a54 = A[4,3]; #DT = c5*h
		k5x, k5v = f(t+c5*h, x+h*(a51*k1x + a52*k2x + a53*k3x + a54*k4x), v+h*(a51*k1v + a52*k2v + a53*k3v + a54*k4v), Bm, K_masses)
		# k5x = (k5x - x)/DT; k5v = (k5v - v)/DT

		c6 = C[5]; a61 = A[5,0]; a62 = A[5,1]; a63 = A[5,2]; a64 = A[5,3]; a65 = A[5,4]; #DT = c6*h
		k6x, k6v = f(t+c6*h, x+h*(a61*k1x + a62*k2x + a63*k3x + a64*k4x + a65*k5x), v+h*(a61*k1v + a62*k2v + a63*k3v + a64*k4v + a65*k5v), Bm, K_masses)
		# k6x = (k6x - x)/DT; k6v = (k6v - v)/DT

		#compute the result
		x_out = x + h*(B[0]*k1x + B[1]*k2x + B[2]*k3x + B[3]*k4x + B[4]*k5x + B[5]*k6x)
		v_out = v + h*(B[0]*k1v + B[1]*k2v + B[2]*k3v + B[3]*k4v + B[4]*k5v + B[5]*k6v)

		#estimate the error of the result
		x_err = x + h*(E[0]*k1x + E[1]*k2x + E[2]*k3x + E[3]*k4x + E[4]*k5x + E[5]*k6x)
		x_err = x_err - x_out
		v_err = v + h*(E[0]*k1v + E[1]*k2v + E[2]*k3v + E[3]*k4v + E[4]*k5v + E[5]*k6v)
		v_err = v_err - v_out

		max_err = np.max(np.abs(x_err))
		mav_err = np.max(np.abs(v_err))#*h

		return max_err, mav_err, x_out,v_out,x_err,v_err
	return one_step_explicit_dormand_prince_method

#######################################
# Example Usage:
#######################################
# one_step_explicit_dormand_prince_method = get_one_step_explicit_dormand_prince_method(mu,lam,gamma)
# # the stepsize
# h = 0.001
# # the local configuration of one tetrahedral element
# K_index = 7
# Bm = element_array_inverse_equilibrium_position[K_index]
# Ka = element_array_index[K_index]
# x = vertices[Ka].copy()
# v = velocities[Ka].copy()
# tau_of_K = element_array_time[K_index]
# K_tau = node_array_time[Ka].copy()
# K_masses = element_array_mass[Ka]
# max_err, mav_err, x_out,v_out = one_step_explicit_dormand_prince_method(h,x,v,K_masses,K_tau,tau_of_K,Bm,zero_mat)
