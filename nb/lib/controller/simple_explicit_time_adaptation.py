import numpy as np
from ..model import *
from . import *

def get_step_forward_and_learn_simple(mu,lam,gamma):
	zero_mat = np.zeros((4,3))
	mode='explicit'
	if mode=='explicit':
		def get_step_to_time(mu,lam,gamma,num_iter = 30):
			# step_to_time = get_local_one_step_implicit_midpoint_rule(mu,lam,gamma,num_iter = 30)
			step_to_time   = get_compute_one_step_forward_euler_method(mu,lam,gamma)
			# step_to_time = get_one_step_forward_euler_method(mu,lam,gamma)
			return step_to_time
	else:
		def get_step_to_time(mu,lam,gamma,num_iter = 30):
			step_to_time = get_local_one_step_implicit_midpoint_rule(mu,lam,gamma,num_iter = 30)
			# step_to_time   = get_compute_one_step_forward_euler_method(mu,lam,gamma)
			# step_to_time = get_one_step_forward_euler_method(mu,lam,gamma)
			return step_to_time
	step_to_time = get_step_to_time(mu,lam,gamma)

	@njit
	def step_forward_and_learn_simple(K_index, t_given, node_array_time, element_array_time, vertices, velocities, element_array_index,
	                           element_array_inverse_equilibrium_position, node_array_mass, atol_x, atol_v, btol_x, btol_v, learning_rate
	                          ):
	    Ka = element_array_index[K_index]
	    K_tau = node_array_time[Ka]
	    tau_of_K = element_array_time[K_index]
	    Bm  = element_array_inverse_equilibrium_position[K_index]
	    K_masses = node_array_mass[Ka]
	    x_current = vertices[Ka]
	    v_current = velocities[Ka]
	    #split the elemental configuration into three possibiities
	    x_A = x_current.copy()
	    x_B = x_current.copy()
	    v_A = v_current.copy()
	    v_B = v_current.copy()
	    tau_of_K_A = tau_of_K#.copy()
	    tau_of_K_B = tau_of_K#.copy()
	    K_tau_A = K_tau.copy()# K_tau_A[0] += 0.01
	    K_tau_B = K_tau.copy()
	    #define time steps options
	    DT_given   = t_given - tau_of_K
	    DT_lesser  = DT_given/2
	    DT_greater = DT_given*2
	    t_lesser   = DT_lesser  + tau_of_K
	    t_greater  = DT_greater + tau_of_K
	    #update config A to time t_lesser
	    x_A, v_A = step_to_time(t_lesser,
	            x_A,
	            v_A,K_masses,
	            K_tau_A,
	            tau_of_K_A,Bm,zero_mat)
	    #update times
	    K_tau_A    = t_lesser + 0.*K_tau_A
	    tau_of_K_A = t_lesser
	    #update config A to time t_given
	    x_A, v_A = step_to_time(t_given,
	            x_A,
	            v_A,K_masses,
	            K_tau_A,
	            tau_of_K_A,Bm,zero_mat)
	    #update times
	    K_tau_A    = t_given + 0.*K_tau_A
	    tau_of_K_A = t_given
	    #update config B to time t_given
	    x_B, v_B = step_to_time(t_given,
	            x_B,
	            v_B,K_masses,
	            K_tau_B,
	            tau_of_K_B,Bm,zero_mat)
	    #update times
	    K_tau_B    = t_given + 0.*K_tau_B
	    tau_of_K_B = t_given
	    # compute the max absolute difference (mad) in position and velocity between the two step sizes
	    mad_xBA = np.max(np.abs(x_B-x_A))
	    mad_vBA = np.max(np.abs(v_B-v_A))
	    madval = (mad_xBA, mad_vBA)
	    retval = (x_B.copy(), v_B.copy(), K_tau_B.copy(), tau_of_K_B)
	    # NOTE: if either of ^these values are above some threshold,
	    # then decrease the stepsize and restart the onestep method to be safe.
	    if (mad_xBA > atol_x) | (mad_vBA > atol_v):
	        #TODO: restart the algorithm with half the stepsize and return the result
	        # print('TODO: restart the algorithm with half the stepsize and return the result')
	        # retval = (x_A.copy(), v_A.copy(), K_tau_A.copy(), tau_of_K_A)
	        # next_stepsize = DT_lesser
	        # next_stepsize, retval, madval = step_forward_and_learn_simple(K_index, t_lesser, node_array_time, element_array_time, vertices, velocities, element_array_index,
	        #                    element_array_inverse_equilibrium_position, node_array_mass, atol_x, atol_v, btol_x, btol_v, learning_rate)
	        next_stepsize = DT_lesser
	    elif (mad_xBA < btol_x) & (mad_vBA < btol_v):
	        #store double the given step for later use
	        next_stepsize = DT_greater
	    else:
	    	next_stepsize = DT_given
	    return next_stepsize, retval, madval

	return step_forward_and_learn_simple
