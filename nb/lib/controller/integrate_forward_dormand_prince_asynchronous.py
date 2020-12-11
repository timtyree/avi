#Integrate a tetrahedral mesh forward in time with synchronous time updates
#Tim Tyree
#12/6/2020

import numpy as np, heapq
from numba import njit
from ..model import *
from . import *
from ..measure import *

def get_compute_next_stepsize_neural_learning_rate(atol_x, atol_v, btol_x, btol_v, lasso_fraction = 0.5):

	@njit
	def compute_next_stepsize_neural_learning_rate(h, learning_rate, max_err,mav_err,energy_next, energy_prev):
		Delta_energy = energy_next - energy_prev
		loss = (1.-lasso_fraction) * Delta_energy**2 + lasso_fraction * np.abs(Delta_energy)
		booa = (max_err>atol_x)|(mav_err>atol_v)
		boob = (max_err<btol_x)&(mav_err<btol_v)
		# learning_exponent = learning_rate
		learning_exponent = learning_rate * loss # or some other function

		if booa: #if either error is too big,
			next_stepsize = h*np.exp(-learning_exponent) #decrease the stepsize
			#TODO: and repeat the step
		elif boob: #if both errors are too small,
			next_stepsize = h*np.exp(+learning_exponent) #increase the stepsize
		else: #else, keep the same stepsize
			next_stepsize = h
		return next_stepsize
	return compute_next_stepsize_neural_learning_rate


def get_compute_next_stepsize_fixed_learning_rate(atol_x, atol_v, btol_x, btol_v, learning_rate):

	@njit
	def compute_next_stepsize_fixed_learning_rate(h, learning_rate, max_err,mav_err,energy_next, energy_prev):
		booa = (max_err>atol_x)|(mav_err>atol_v)
		boob = (max_err<btol_x)&(mav_err<btol_v)
		learning_exponent = learning_rate
		#TODO: try learning_exponent = learning_rate*energy or some other function

		if booa: #if either error is too big,
			next_stepsize = h*np.exp(-learning_exponent) #decrease the stepsize
			#TODO: and repeat the step
		elif boob: #if both errors are too small,
			next_stepsize = h*np.exp(+learning_exponent) #increase the stepsize
		else: #else, keep the same stepsize
			next_stepsize = h
		return next_stepsize
	return compute_next_stepsize_fixed_learning_rate

def get_integrate_system_dormand_prince_asynchronous(mu,lam,gamma,atol_x, atol_v, btol_x, btol_v,learning_rate, lasso_fraction):
	one_step_explicit_dormand_prince_method = get_one_step_explicit_dormand_prince_method(mu,lam,gamma)
	compute_next_stepsize = get_compute_next_stepsize_neural_learning_rate(atol_x, atol_v, btol_x, btol_v,lasso_fraction)
	# compute_next_stepsize = get_compute_next_stepsize_fixed_learning_rate(atol_x, atol_v, btol_x, btol_v, learning_rate)
	
	@njit
	def integrate_system_dormand_prince_asynchronous(tf, element_array_time, element_array_stepsize, node_array_time,
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
				tau_of_K = tauK[K_index]
				K_tau = tau[Ka].copy()
				K_masses = element_array_mass[Ka]
				Ds = get_D_mat(x)
				W = get_element_volume(Ds)
				mass_of_K  = element_array_mass[K_index]

				#compute energy of configuration
				energy_prev = comp_element_energy ( mass_of_K, v, W, Bm, Ds, mu, lam)

				#perform the OneStep method
				max_err, mav_err, x_out,v_out = one_step_explicit_dormand_prince_method(h,x,v,K_masses,K_tau,tau_of_K,Bm)



				#update the element configuration
				vertices[Ka] = x_out
				velocities[Ka] = v_out
				tauK[K_index] = t
				tau[Ka] = t

				x = vertices[Ka].copy()
				v = velocities[Ka].copy()
				tau_of_K = tauK[K_index]
				K_tau = tau[Ka].copy()
				K_masses = element_array_mass[Ka]
				Ds = get_D_mat(x)
				W = get_element_volume(Ds)

				#compute energy of configuration
				energy_next = comp_element_energy ( mass_of_K, v, W, Bm, Ds, mu, lam)

				#choose the next stepsizeh, learning_rate, max_err,mav_err,energy_next, energy_prev
				stepsize_next = compute_next_stepsize(element_array_stepsize[K_index], learning_rate, max_err,mav_err,energy_next, energy_prev)
				t_next = t + stepsize_next
				element_array_stepsize[K_index] = stepsize_next

				#push the next task to the heap/priority queue
				heapq.heappush(queue, (t_next, stepsize_next, K_index))

		return True

	return integrate_system_dormand_prince_asynchronous
