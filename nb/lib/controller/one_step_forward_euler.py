# one_step_forward_euler.py
import numpy as np
from ..model import *

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


def get_compute_one_step_forward_euler_method(mu,lam,gamma):
	compute_force = get_compute_force(mu,lam,gamma)

	@njit
	def compute_one_step_forward_euler_method(t, K_vertices, K_velocities, K_masses, K_tau, tau_of_K, Bm, zero_mat):
		'''integrates the inputed element configuration up tot time t.  element time is not updated by this function.
		also updates K_velocities, K_vertices, K_tau, to time t.'''
		Na = 4 # number of tetrahedral nodes, which is 4
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

		# #but with what acceleration do I define the rate of change of velocity? Which t^* is correct? Let's say the next one.
		# v = K_velocities.copy()
		# K_velocities[a] += K_accelerations[a] * (t - K_tau[a])

		#compute the nodal forces for the tetrahedral element at the next time
		force = compute_force(K_velocities, Ds, K_W, Bm, zero_mat.copy())
		# force = compute_force(K_velocities, Ds, K_W, Bm, zero_mat) #is this faster? also doesn't update zero_mat?
		#        Delta_v = np.multiply ( (t - tau_of_K) / K_masses , force )
		#update node velocities
		for a in range(Na):
			K_velocities[a] += (t - tau_of_K) / K_masses[a] * force[a] #+ Delta_v[a]
			# K_velocities[a] = K_velocities[a] + (t - tau_of_K) / K_masses[a] * force[a] #+ Delta_v[a]
		#TODO(later): if node is not a boundary node, set velocity to zero
		#         return Delta_x, Delta_v
		return K_vertices, K_velocities
	return compute_one_step_forward_euler_method


# ##################################################################
# # Example Usage: update one element 50,000 times per second
# ##################################################################
# one_step_forward_euler_method = get_one_step_forward_euler_method(mu=1.,lam=1.,gamma=1.)
# K_index = 245
# for t in np.linspace(18, 19, 50000):
#     one_step_forward_euler_method(t, K_index, element_array_index, tauK, tau, vertices, velocities, node_array_mass, element_array_inverse_equilibrium_position)





######################################################
# The following is deprecated
######################################################
def get_compute_one_step_map_forward_euler(mu,lam):
	comp_nodal_elastic_forces = get_comp_nodal_elastic_forces(mu, lam)
	@njit
	def compute_one_step_map_forward_euler(t, K_vertices, K_velocities, K_masses, K_tau, tau_of_K, Ds, Bm, K_W, zero_mat):
		'''returns Delta_x, Delta_v, which updated K_vertices, K_velocities to time t.
		also updates K_velocities, K_vertices, K_tau, tau_of_K to time t.'''
		Na = 4 # number of tetrahedral nodes, which is 4
		# t, K_index, vertices, velocities, Ka, tau, tauK, elements, element_array_inverse_equilibrium_position, zero_mat, node_array_mass):
		Delta_x = np.multiply (K_velocities[a] , (t - K_tau[a]))
		for a in range(Na):
			K_vertices[a] = K_vertices[a] + Delta_x[a]
		#update node times
		for a in range(Na):
			K_tau[a] = t
		#compute the nodal forces for each tetrahedral node
		f   = comp_nodal_elastic_forces(K_W, Bm, Ds, zero_mat.copy())
		#TODO(later): include any other forces, such as nodal forces, pressure forces, etc.
		#net nodal forces
		force = f
		Delta_v = np.multiply ( (t - tau_of_K) / K_masses , force )
		#update node velocities
		for a in range(Na):
			K_velocities[a] = K_velocities[a] + Delta_v[a]
		#update element's time
		tau_of_K = t
		#TODO(later): if node is not a boundary node, set velocity to zero
		return Delta_x, Delta_v
	return compute_one_step_map_forward_euler

def get_one_step_forward_euler_simplified(mu,lam):
	comp_nodal_elastic_forces = get_comp_nodal_elastic_forces(mu, lam)
	@njit
	def one_step_forward_euler_simplified(t, K_vertices, K_velocities, K_masses, K_tau, tau_of_K, Ds, Bm, K_W, zero_mat):
		'''returns K_vertices, K_velocities, having been updated to time t.'''
		Na = 4 # number of tetrahedral nodes, which is 4
		# t, K_index, vertices, velocities, Ka, tau, tauK, elements, element_array_inverse_equilibrium_position, zero_mat, node_array_mass):
		for a in range(Na):
			K_vertices[a] = K_vertices[a] + K_velocities[a] * (t - K_tau[a])
		#update node times
		for a in range(Na):
			K_tau[a] = t
		#compute the nodal forces for each tetrahedral node
		f   = comp_nodal_elastic_forces(K_W, Bm, Ds, zero_mat.copy())
		#TODO(later): include any other forces, such as nodal forces, pressure forces, etc.
		#net nodal forces
		force = f
		#update node velocities
		for a in range(Na):
			K_velocities[a] = K_velocities[a] + ( (t - tau_of_K) / K_masses[a]) * force[a]
		#TODO(later): if node is not a boundary node, set velocity to zero
		return K_vertices, K_velocities
	return one_step_forward_euler_simplified

# @njit
def one_step_forward_euler_bulky(t, K_index, vertices, velocities, Ka, tau, tauK, elements,
	element_array_inverse_equilibrium_position, zero_mat, node_array_mass):
	'''doesn't update times for nodes or elements'''
	for a in Ka:
		vertices[a] = vertices[a] + velocities[a] * (t - tau[a])
	#compute the nodal forces for each tetrahedral
	K_W = compute_element_volume(node_array_position=vertices, element_array_index=elements, K_index=K_index)
	Bm  = element_array_inverse_equilibrium_position[K_index]
	f   = compute_nodal_elastic_forces(K_vertices, K_W, Bm, f = zero_mat.copy())
	#TODO(later): include any other forces, such as nodal forces, pressure forces, etc.
	#net nodal forces
	force = f
	#update node velocities
	for j, a in enumerate(Ka):
		velocities[a] = velocities[a] + ( (t - tauK[K_index]) / node_array_mass[a]) * force[j]
	#TODO(later): if node is not a boundary node, set velocity to zero
	return vertices, velocities


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
