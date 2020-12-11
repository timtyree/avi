import numpy as np
from numba import njit
from . import *

def get_flow_map(mu,lam,gamma):
	zero_mat = np.zeros((4,3))
	compute_force = get_compute_force(mu,lam,gamma)
	@njit
	def compute_flow_map(x, v, Bm, K_masses):
		#assumes the positions of all nodes are synchronized with their element
		Ds = get_D_mat(x)
		W = get_element_volume(Ds)
		force = compute_force(v, Ds, W, Bm, zero_mat.copy())

		#return the rates of change of position and velocity
		dxdt =  v.copy()
		dvdt =  v.copy()
		for a in range(4):
			dvdt[a] = force[a]/K_masses[a]
		return dxdt, dvdt
	return compute_flow_map