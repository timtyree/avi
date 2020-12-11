# # Explicit AVI Algorithm
# # Tim Tyree
# # 9.21.2020
# import numpy as np
# import pandas as pd
# from numba import njit

def _compute_next_time(K, t, stepsize):
    '''compute next time for element's evaluation'''
    return t + stepsize





# #pop/push times
# def push(time_list, value):
# 	time_list.append(value)
# 	return time_list
# def pop(time_list):
# 	value = time_list[0]
# 	time_list.remove(value)
# 	return value
# def pop2(time_list):
# 	value1 = time_list[0]
# 	value2 = time_list[1]
# 	time_list.remove(value1)
# 	time_list.remove(value1)
# 	return value1, value2

# def get_substeps(t0 = 0, stepsize=1, num_substeps=4):
# 	'''make an even interval of s substep times'''
# 	h  = stepsize
# 	t1 = t0 + h
# 	s  = 4 #num_substeps
# 	DT = h/s
# 	time_list = [t0 + j*DT for j in range(s+1)]
# 	return sorted(time_list)

# # # @njit
# # def _compute_time(K, t0, stepsize, num_steps):
# # 	"""compute the time steps of the element"""
# # 	return t0
# # 	#the following is a list of steps
# # 	# tf = t0 + num_steps*stepsize
# # 	# return np.arange(t0,tf+stepsize,stepsize)

# # @njit
# def _compute_next_time(K, t, stepsize):
# 	'''compute next time for element's evaluation'''
# 	return t + stepsize

# def spring_force(X1,X2,x1,x2, k = 1.):
# 	'''spring force from 1 to 2'''
# 	D  = X2-X1
# 	d  = x2-x1
# 	x  = np.linalg.norm(d)
# 	x0 = np.linalg.norm(D)
# 	hat= d/x
# 	return -1.*k * (x-x0) * hat

# # @njit
# def _compute_dVdx(K,a,qK,t,q, k = 1.):
# 	pass
# 	# X1 = 
# 	# X2 = 
# 	# x1 = 
# 	# x2 = 
# 	# return spring_force(X1,X2,x1,x2, k = k)

# # def explicit(t0, tf, mesh, stepsize, num_steps, **kwargs)
# def explicit(t0, tf, q, v, tau, edge_list, M, stepsize, num_steps, **kwargs):
# 	'''solve by onestep method of explicit davi algorithm 
# 	t0, tf are floats for the initial and final times
# 	mesh.elements is a list of int primitives
# 	mesh.nodes is a list of int primitives
# 	mesh.x1 is a pandas dataframe. mesh.x32dot is a pandas dataframe.

# 	this algorithm implements the discrete Euler-Lagrange equations of
# 	 the action sum given by (29). in 'LaMaOrWe2003 2.pdf'
# 	'''
# 	#push all K finite elements into the queue at time t0
# 	queue = list()
# 	N = edge_list.shape[0]
# 	tauK = list(range(N))
# 	for K in elements:
# 		tauK[K] = t0
# 		# tK2     = t0 # _compute_time(K, t0, stepsize, num_steps)
# 		push( queue, (t0, K) )

# 	# TODO: njit the following into solve_local_action_explicitely()
# 	#iterate over the elements in time.  
# 	#do until priority queue is empty
# 	while len(queue) > 0:
# 		#extract next element
# 		t, K = pop(queue)

# 		#synchronous time updates
# 		#update node positions
# 		for a in edge_list[K]:
# 			q[a] = q[a] + v[a](t-tau[a])
# 		#update node times
# 		for a in edge_list[K]:
# 			tau[a] = t
# 		if t<tf:
# 			#compute element center (barycentric masses imply qK is the xyz coords of the barycenter)
# 			qK = 0.; count = 0;
# 			for a in edge_list[K]:
# 				qK += q[a]
# 				count += 1
# 			qK /= count
# 			#update node velocities 
# 			for a in edge_list[K]:
# 				v[a] = v[a] - ((t-tauK[K])/M[a])*_compute_dVdx(K, a,qK, t)
# 			#update element's time
# 			tauK[K] = t
# 			#compute next time for element's evaluation
# 			tKnext = _compute_next_time(K, t, stepsize)
# 			#Schedule K for next update
# 			push( queue, (tKnext, K) )
# 	return q, v


