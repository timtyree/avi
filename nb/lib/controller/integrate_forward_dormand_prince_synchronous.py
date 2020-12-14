#Integrate a tetrahedral mesh forward in time with synchronous time updates
#Tim Tyree
#12/6/2020

import numpy as np, heapq
from numba import njit
from ..model import *
from . import *

def get_integrate_system_dormand_prince_synchronous(mu,lam,gamma):
	one_step_explicit_dormand_prince_method = get_one_step_explicit_dormand_prince_method(mu,lam,gamma)
	@njit
	def integrate_system_dormand_prince_synchronous(tf, element_array_time, element_array_stepsize, node_array_time,
											 element_array_index, vertices, velocities,
										  node_array_mass, element_array_inverse_equilibrium_position, element_array_mass):#, minstepsize):
		"""
		Integrate up to time tf with synchronous forward integration using the dormand prince method.
		"""
		#initialize the queue by pushing all finite elements into the queue
		N_elements = element_array_stepsize.shape[0]
		queue = [(element_array_time[K_index],element_array_stepsize[K_index],K_index) for K_index in range(N_elements)]
		heapq.heapify(queue)

		tauK = element_array_time
		tau  = node_array_time
		#do until priority queue is empty
		while len(queue) > 0:
			t,h,K_index = heapq.heappop(queue)
			if t<=tf:
				tau_of_K = tauK[K_index]
				t_previous = tau_of_K#tauK[K_index]
				h = t - t_previous #the current stepsize
				# h = max(h,minstepsize)
				#get the element configuration
				Bm = element_array_inverse_equilibrium_position[K_index]
				Ka = element_array_index[K_index]
				x = vertices[Ka].copy()
				v = velocities[Ka].copy()
				tau_of_K = element_array_time[K_index]
				K_tau = tau[Ka].copy()
				K_masses = element_array_mass[Ka]
				max_err, mav_err, x_out,v_out,x_err,y_err = one_step_explicit_dormand_prince_method(h,x,v,K_masses,K_tau,tau_of_K,Bm)

				#update the element configuration
				vertices[Ka] = x_out
				velocities[Ka] = v_out
				tauK[K_index] = t
				tau[Ka] = t

				#TODO: make stepsize_next a function of max_err and/or mav_err
				stepsize_next = element_array_stepsize[K_index]#h
				t_next = t + stepsize_next
				element_array_stepsize[K_index] = stepsize_next

				#push the next task to the heap/priority queue
				heapq.heappush(queue, (t_next, stepsize_next, K_index))

		return True

	return integrate_system_dormand_prince_synchronous
