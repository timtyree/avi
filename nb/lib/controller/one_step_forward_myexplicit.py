# copy of one_step_forward_euler.py
import numpy as np
from ..model import *

#TODO: implement K_accelerations in a simple movie making .ipynb (then script)
#TODO: try updating velocity before computing drag
#TODO: consider the backward error analysis of ^this.  Melvin might like this...
#TODO: consider different simple combinations of parameters while considering the mean value theorem and/or quadrature.
#TODO: consider optimizing in C or the warp-drive machinery provided by NVIDIA...

def get_compute_myexplicit_one_step_method(mu,lam,gamma):
	compute_force = get_compute_force(mu,lam,gamma)

	@njit
	def compute_myexplicit_one_step_method(t, K_vertices, K_velocities, K_accelerations, K_masses, K_tau, tau_of_K, Bm, zero_mat):
		'''integrates the inputed element configuration up tot time t.  element time is not updated by this function.
		also updates K_velocities, K_vertices, K_tau, to time t.
		returns K_vertices, K_velocities, K_accelerations.'''
		Na = 4 # number of nodes on a 3-simplex (tetrahedron), which is 4 vertices.
		# t, K_index, vertices, velocities, Ka, tau, tauK, elements, element_array_inverse_equilibrium_position, zero_mat, node_array_mass):
		#         Delta_x = np.multiply (K_velocities , (t - K_tau))
		#update a copy of the velocities to next time

		for a in range(Na):
			K_vertices[a] += K_velocities[a] * (t - K_tau[a]) #+ Delta_x[a]
			# K_vertices[a] = K_vertices[a] + K_velocities[a] * (t - K_tau[a]) #+ Delta_x[a]
		#compute the Ds matrix
		Ds  = get_D_mat(K_vertices)
		K_W = get_element_volume(Ds)
		# #update node times
		# for a in range(Na):
		# 	K_tau[a] = t

		v = K_velocities.copy()
		#TODO: update v to time t from time K_tau[a])
		#but with what acceleration do I define the rate of change of velocity? Which t^* is correct? Let's say the next one.
		#simplest solution = define an array called K_acceleration and keep it updated every function call
		K_velocities[a] += K_accelerations[a] * (t - K_tau[a])

		#compute the nodal forces for the tetrahedral element at the next time
		force = compute_force(v, Ds, K_W, Bm, zero_mat.copy())
		# force = compute_force(K_velocities, Ds, K_W, Bm, zero_mat) #is this faster? also doesn't update zero_mat?
		#        Delta_v = np.multiply ( (t - tau_of_K) / K_masses , force )
		#update node velocities
		for a in range(Na):
			K_velocities[a] += (t - tau_of_K) / K_masses[a] * force[a] #+ Delta_v[a]
			# K_velocities[a] = K_velocities[a] + (t - tau_of_K) / K_masses[a] * force[a] #+ Delta_v[a]
		#TODO(later): if node is not a boundary node, set velocity to zero
		#         return Delta_x, Delta_v
		return K_vertices, K_velocities, K_accelerations
	return compute_myexplicit_one_step_method

def get_one_step_forward_euler_method(mu,lam,gamma):
	compute_one_step_forward_euler_method=get_compute_one_step_forward_euler_method(mu,lam,gamma)
	zero_mat = np.zeros((4,3))
	@njit
	def one_step_forward_euler_method(t, K_index, element_array_index, tauK, tau,
									  vertices, velocities, node_array_mass, element_array_inverse_equilibrium_position):
		#compute the namespace of the force computation of the elemental configuration
		Ka = element_array_index[K_index]
		tau_of_K = tauK[K_index]
		K_tau = tau[Ka]
		K_vertices = vertices[Ka].copy()
		K_velocities = velocities[Ka].copy()
		K_masses = node_array_mass[Ka]
		Bm  = element_array_inverse_equilibrium_position[K_index]
		# K_W = compute_element_volume(node_array_position=vertices, element_array_index=element_array_index, K_index=K_index)
		#Ds  = get_D_mat(K_vertices)
		K_vertices, K_velocities = compute_one_step_forward_euler_method(t, K_vertices, K_velocities, K_masses, K_tau, tau_of_K, Bm, zero_mat) #updates nodal times
		# #update element's time
		# tauK[K_index] = t
		# #update node's time
		# tau[Ka] = t
		return K_vertices, K_velocities

	return one_step_forward_euler_method


# ##################################################################
# # Example Usage: update one element 50,000 times per second
# ##################################################################
# one_step_forward_euler_method = get_one_step_forward_euler_method(mu=1.,lam=1.,gamma=1.)
# K_index = 245
# for t in np.linspace(18, 19, 50000):
#     one_step_forward_euler_method(t, K_index, element_array_index, tauK, tau, vertices, velocities, node_array_mass, element_array_inverse_equilibrium_position)

# ##################################################################
# # Example Usage: one update task within an AVI
# ##################################################################
# #one elemental time update
# #update node positions
# Ka = elements[K_index]
# K_vertices = vertices[Ka]
# vertices, velocities = one_step_forward_euler_bulky(t, K_index, Ka, vertices, velocities, tau, tauK,
# 	elements, element_array_inverse_equilibrium_position, zero_mat, node_array_mass)
# #update node times
# for a in Ka:
# 	tau[a] = t
# #update element's time
# tauK[K_index] = t
# #compute next time for element's evaluation
# tKnext = t + stepsize #_compute_next_time(K, t, stepsize)

# ##################################################################
# # Example Usage: one update task within an AVI
# ##################################################################
# #given initialization as in explicit.py
# #simplified elemental time update
# Ka  = elements[K_index]
# K_vertices   = vertices[Ka]
# K_velocities = velocities[Ka]
# K_masses     = node_array_mass[Ka]
# K_tau        = tau[Ka]
# tau_of_K     = tauK[K_index]
# Ds  = get_D_mat(K_vertices)
# Bm  = element_array_inverse_equilibrium_position[K_index]
# K_W = get_element_volume(Ds)
# K_vertices, K_velocities = one_step_forward_euler_simplified(t, K_vertices, K_velocities,
# 	K_masses, K_tau, tau_of_K, Ds, Bm, K_W, zero_mat)
# #TODO(later): if node is a boundary node, set velocity to zero
# #update element's time
# tauK[K_index] = t
# #compute next time for element's evaluation
# tKnext = t + stepsize #_compute_next_time(K, t, stepsize)
