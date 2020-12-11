import numpy as np
from numba import njit

@njit
def get_D_mat(K_vertices):
	x1  = K_vertices[0,0];y1 = K_vertices[0,1];z1 = K_vertices[0,2]
	x2  = K_vertices[1,0];y2 = K_vertices[1,1];z2 = K_vertices[1,2]
	x3  = K_vertices[2,0];y3 = K_vertices[2,1];z3 = K_vertices[2,2]
	x4  = K_vertices[3,0];y4 = K_vertices[3,1];z4 = K_vertices[3,2]
	D = np.array([
			  [x1-x4,x2-x4,x3-x4],
			  [y1-y4,y2-y4,y3-y4],
			  [z1-z4,z2-z4,z3-z4]])
	return D
@njit
def compute_D_mat(node_array_position, element_array_index, K_index):
	K_vertices = node_array_position[element_array_index[K_index]] 
	D = get_D_mat(K_vertices)
	return D
@njit
def compute_inverse_position(node_array_position, element_array_index, K_index):
	'''Example Usage: 
	X_inverse = compute_inverse_position(node_array_equilibrium_position, element_array_index, K_index)'''
	D_K = compute_D_mat(node_array_position, element_array_index, K_index)
	X_inverse = np.linalg.inv(D_K)
	#undeformed volume of element
	# W = np.linalg.det(D_K)/6.
	return X_inverse
@njit
def compute_element_volume(node_array_position, element_array_index, K_index):
	'''Example Usage: 
	X_inverse = compute_inverse_position(node_array_equilibrium_position, element_array_index, K_index)'''
	D_K = compute_D_mat(node_array_position, element_array_index, K_index)
	# X_inverse = np.linalg.inv(D_K)
	#undeformed volume of element
	W = np.abs(np.linalg.det(D_K)/6.)
	return W
@njit
def get_element_volume(D_K):
	'''undeformed volume of tetrahedral element'''
	W = np.abs(np.linalg.det(D_K)/6.)
	return W