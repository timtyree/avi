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

# 	this algorithm implements the discrete Euler-Lagrange equations of
# 	 the action sum given by (29). in 'LaMaOrWe2003 2.pdf'
# 	'''

def explicit_synchronous(t0, tf, stepsize, element_array_time, node_array_time, N_elements, N_vertices,
	vertices, elements, velocities, element_array_inverse_equilibrium_position, node_array_mass, node_array_momentum):
	'''element_array_time, node_array_time are reset to t0 and elements are stepped forward until tf in time increments of stepsize
	'''
	#push all K finite elements into the queue at initial time t0
	queue = PriorityQueue()# queue = list()
	#initialize all elemental times to the same time value, t0, and push all elements to queue
	tauK = t0+0.*element_array_time
	for K_index in list(range(N_elements)):
		tauK[K_index] = t0
		queue.put((t0, K_index))#     queue.append((t0, K_index)) 
		# tK2     = t0 # _compute_time(K, t0, stepsize, num_steps)

	# initialize nodal times to initial time t0
	tau = t0 + 0. * node_array_time

	#iterate over the elements in time.  
	#do until priority queue is empty
	# while len(queue) > 0:
	old_vertices = vertices.copy() 
	while not queue.empty():
		#extract next element
		t, K_index = queue.get()#pop(queue)
		if t<=tf:
			#synchronous time updates
			#update node positions
			Ka = elements[K_index]
			K_vertices = vertices[Ka]
			
			for a in Ka:
				vertices[a] = vertices[a] + velocities[a] * (t - tau[a])
			#update node times
			for a in Ka:
				tau[a] = t

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
			#update element's time
			tauK[K_index] = t
			#compute next time for element's evaluation
			tKnext = t + stepsize #_compute_next_time(K, t, stepsize)
			
			#Schedule K for next update
			queue.put((tKnext, K_index))#push( queue, (tKnext, K_index) )
			
			#if a new time has been observed, measure the old_mesh volume. record volume and tme. update the old_mesh and time
			if t>tme:
				volume = compute_net_volume(old_vertices, elements)
				volume_lst.append(volume)
				tme_lst.append(tme)
				#suppose synchronous time stepping
				tme = t
				old_vertices = vertices.copy()   
	#TODO: synchronize all nodal positions with current time, tf

	#update momentum
	for j in range(N_vertices):
		node_array_momentum[j] = node_array_mass[j] * velocities[j]
		
	#update synchronous time
	tme = tf

	#Bottom Line Up Front (BLUF)
	beep(3)
	print(tf-t0)
	print(stepsize)

	retval = (t0, tf, stepsize, element_array_time, node_array_time, N_elements, N_vertices,
		vertices, elements, velocities, element_array_inverse_equilibrium_position, node_array_mass, node_array_momentum)
	return retval


# ##################################################################
# # Example Usage: Initialization for Ibid
# ##################################################################

# os.chdir(nb_dir)
# # input_file_name = f'../data/spherical_meshes/spherical_mesh_64.stl'input_file_name = f'../data/spherical_meshes/spherical_mesh_64.stl'
# input_file_name = f'../data/spherical_meshes/spherical_mesh_100.stl'
# tme = 0.
# mass_density=1.
# dict_values_system = initialize_system(input_file_name, time_initial=tme, mass_density=mass_density)
# locals().update(dict_values_system)
# print(list(dict_values_system.keys()))

# #define Lamé parameters
# mu = 1; lam = 1;
# #get method of computing elastic forces 
# zero_mat = np.zeros((4,3))
# calc_P = get_calc_P(mu, lam)
# compute_nodal_elastic_forces= get_compute_nodal_elastic_forces(mu, lam)
# elements = element_array_index
# vertices = node_array_position
# N_elements = elements.shape[0]
# N_vertices = vertices.shape[0]
# momentum = node_array_momentum.copy()

# #perturb momentum in the outward x direction and let it run overnight with a small timestep
# velocities = momentum.copy()
# for j in range(N_vertices):
# 	velocities[j] /= node_array_mass[j]
# v_scale = 1
# velocities[:,0] = v_scale * vertices[:,0].copy()

# #initialize containers of measures
# volume_lst = []
# tme_lst = []


# ########################################################
# t0 = tme
# tf = 20.#0.001#.0019
# stepsize = 0.01#0.0.001 # 0.001 worked for tme to 2 but not 4
