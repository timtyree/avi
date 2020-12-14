import numpy as np
from .geom import *
from numba import njit
# Tim Tyree
# 12.4.2020

# Following the methods described in `sifakis-courseNotes-TheoryAndDiscretization.pdf`
# --> ctrl+F for "Algorithm 1" >> elastic forces followed by force differentials (pg. 30 - 35)
#Nota Bene: ^they've got many constitutive models worked out

############################################################################
# Nodal Elastic Forces on tetrahedra
############################################################################
def get_calc_P(mu, lam):#, one, delta):
	return get_calc_P_neohookean(mu, lam)

#TODO: njit the value returned! 
def get_calc_P_neohookean(mu, lam):
	'''returns the first Piola-Kirchoff stress tensor ( times the constant membrane thickness, delta) 
	from the Neohookean constitutive model for elastic stress.
	Example Usage - given deformation matrix F = S.dot(R):
	mu = 1.; lam = 1.; delta = 0.1; 
	calc_P = get_calc_P(mu, lam)
	P = calc_P(F)'''
	@njit
	def calc_P(F):
		FT = np.linalg.inv(F.T)
		J = np.linalg.det(F)
		P = mu * (F - FT) + lam * np.log(J) * FT
		return P
	return calc_P

def get_compute_nodal_elastic_forces(mu, lam):
	calc_P = get_calc_P(mu, lam)
	@njit
	def compute_nodal_elastic_forces(K_vertices, K_W, Bm, f):
		Ds= get_D_mat(K_vertices)
		F = Ds @ Bm#np.matmul(Ds,Bm)#
		P = calc_P(F)
		H = -K_W * P @ Bm.T.copy()#np.matmul(P,Bm.T)#
		H = H.T
		f[0] += H[0]
		f[1] += H[1]
		f[2] += H[2]
		f[3] += (-H[0] -H[1] -H[2])
		return f
	return compute_nodal_elastic_forces

def get_comp_nodal_elastic_forces(mu, lam):
	calc_P = get_calc_P(mu, lam)
	@njit
	def comp_nodal_elastic_forces(K_W, Bm, Ds, f_out):
		F = Ds @ Bm#np.matmul(Ds,Bm)#
		P = calc_P(F)
		H = -K_W * P @ Bm.T.copy()#np.matmul(P,Bm.T)#
		H = H.T
		f_out[0] += H[0]
		f_out[1] += H[1]
		f_out[2] += H[2]
		f_out[3] += (-H[0] -H[1] -H[2])
		return f_out
	return comp_nodal_elastic_forces

# def get_nodal_elastic_forces(x, f, K, B, W, element_array_index):
# 	Ds = compute_D_mat(node_array_position=x, element_array_index=element_array_index, K_index=K)
# 	Bm = B[K]
# 	F = np.dot(Ds,Bm)
# 	P = calc_P(F)
# 	H = -W[K]*np.dot(P,Bm.T)
# 	f[0] += H[0]
# 	f[1] += H[1]
# 	f[2] += H[2]
# 	f[3] += (-H[0] -H[1] -H[2])
# 	return f


############################################################################
# Nodal Elastic Force Differentials on tetrahedra
############################################################################
def get_calc_delta_P(mu, lam):#, one, delta):
	return get_calc_delta_P_neohookean(mu, lam)

#TODO: njit the value returned! 
def get_calc_delta_P_neohookean(mu, lam):
	'''returns the first Piola-Kirchoff stress tensor ( times the constant membrane thickness, delta) 
	from the Neohookean constitutive model for elastic stress.
	Example Usage - given deformation matrix F = S.dot(R):
	mu = 1.; lam = 1.; delta = 0.1; 
	calc_P = get_calc_P(mu, lam)
	P = calc_P(F)'''
	@njit
	def calc_delta_P(delta_F, F):
		Finv = np.linalg.inv(F)
		FT = Finv.T
		J = np.linalg.det(F)
		# delta_P = mu * delta_F + (mu - lam * np.log(J)) * np.matmul(np.matmul(FT , delta_F.T) , FT ) + lam * np.trace(np.matmul(Finv , delta_F)) * FT
		delta_P = mu * delta_F + (mu - lam * np.log(J)) * FT @ delta_F.T.copy() @ FT + lam * np.trace(Finv @ delta_F) * FT
		return delta_P
	return calc_delta_P

#TODO(later): @njit
def get_compute_nodal_elastic_force_differentials(mu, lam):
	calc_delta_P = get_calc_delta_P(mu, lam)
	@njit
	def compute_nodal_elastic_force_differentials(delta_K_vertices, K_vertices, K_W, Bm, delta_f):
		delta_Ds= get_D_mat(K_vertices)
		delta_F = delta_Ds @ Bm#np.matmul(delta_Ds,Bm)#
		delta_P = calc_delta_P(delta_F)
		delta_H = -K_W * delta_P @ Bm.T.copy()#np.matmul(delta_P,Bm.T)#
		delta_H = delta_H.T
		delta_f[0] += delta_H[0]
		delta_f[1] += delta_H[1]
		delta_f[2] += delta_H[2]
		delta_f[3] -= delta_H[0] + delta_H[1] + delta_H[2]
		return delta_f
	return compute_nodal_elastic_force_differentials


############################################################################
# Nodal Rayleigh Damping on tetrahedra
############################################################################
def get_compute_nodal_elastic_force_differentials(mu,lam):
	calc_delta_P = get_calc_delta_P(mu, lam)
	@njit
	def compute_nodal_elastic_force_differentials(delta_Ds, Ds, K_W, Bm, delta_f):
		delta_F = delta_Ds @ Bm#np.matmul(delta_Ds,Bm)#
		F = Ds @ Bm#np.matmul(Ds , Bm)#
		delta_P = calc_delta_P(delta_F, F)
		delta_H = -K_W * delta_P @ Bm.T.copy()#np.matmul(delta_P,Bm.T)#
		delta_H = delta_H.T
		delta_f[0] = delta_H[0]
		delta_f[1] = delta_H[1]
		delta_f[2] = delta_H[2]
		delta_f[3] = -(delta_H[0] + delta_H[1] + delta_H[2])
		return delta_f
	return compute_nodal_elastic_force_differentials

def get_compute_nodal_damping_forces(mu,lam,gamma):
	compute_nodal_elastic_force_differentials = get_compute_nodal_elastic_force_differentials(mu,lam)
	@njit
	def compute_nodal_damping_forces(K_velocities, Ds, K_W, Bm,delta_f):
		delta_Ds = get_D_mat(-K_velocities)
		f_damping = -gamma * compute_nodal_elastic_force_differentials(delta_Ds, Ds, K_W, Bm, delta_f)
		return f_damping
	return compute_nodal_damping_forces

######################################
# Net force computation
######################################
def get_compute_force(mu,lam,gamma):
	'''nota bene, on my macbook, compute_force was able to compute over 50,000 times per second.'''
	compute_nodal_damping_forces  = get_compute_nodal_damping_forces(mu,lam,gamma)
	comp_nodal_elastic_forces = get_comp_nodal_elastic_forces(mu, lam)
	@njit
	def compute_force(K_velocities, Ds, K_W, Bm, zero_mat):
		"""returns the net nodal forces for an element"""
		fe   = comp_nodal_elastic_forces(K_W, Bm, Ds, f_out = zero_mat.copy())
		fd   = compute_nodal_damping_forces(K_velocities, Ds, K_W, Bm, delta_f=zero_mat.copy())
		force = fe + fd
		return force
	return compute_force

# ######################################
# # Example Usage
# ######################################

# #define Lamé parameters
# mu = 1; lam = 1;

# X_inv = element_array_inverse_equilibrium_position[0]
# zero_mat = np.zeros((4,3))
# x = node_array_position
# B = element_array_inverse_equilibrium_position
# W = element_array_volume
# calc_P = get_calc_P(mu, lam)

# #show a known test case yields a reasonable result for the nodal forces
# K_index = 0
# K = K_index
# f = zero_mat.copy()
# retval = get_nodal_elastic_forces(x, f, K, B, W, element_array_index)
# print(retval)

# #nontrivial test case: nodal forces when compressing in the x direction
# K_index = 0
# K_vertices = node_array_equilibrium_position[element_array_index[K_index]]
# com = np.mean(K_vertices,axis=0)
# K_vertices -= com
# K_vertices[:,0] *= 0.5
# K_W = element_array_volume[K_index]
# Bm  = element_array_inverse_equilibrium_position[K_index]
# f = compute_nodal_elastic_forces(K_vertices, K_W, Bm)

# # assert(np.isclose(f,0.).all())

# #assert net force is zero in the x direction
# assert(np.isclose(np.sum(f[:,0]),0.))

# Q = node_array_equilibrium_position[element_array_index[K_index]]
# com = np.mean(Q,axis=0)
# Q -= com
# q = K_vertices
