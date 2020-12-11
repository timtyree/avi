import numpy as np, trimesh, tetgen
from ..model.geom import *

#################################
# Mesh Input
#################################
def load_trimesh(input_file_name):
	'''input_file_name is a .stl or .ply file'''
	mesh_trimesh = trimesh.load(input_file_name)
	t = 0.
	# mesh_trimesh.vertices -= mesh_trimesh.center_mass
	#normalize the mean radius to 1
	# mesh.vertices /= np.cbrt(mesh.volume*3/(4*np.pi))

	vertices_trimesh = np.array(mesh_trimesh.vertices)
	faces_trimesh = np.array(mesh_trimesh.faces)

	# assert(mesh_trimesh.is_winding_consistent)
	# assert(mesh_trimesh.is_watertight)
	# assert(mesh_trimesh.is_volume)
	return vertices_trimesh, faces_trimesh

def create_tetrahedral_mesh(vertices_trimesh, faces_trimesh):
	# create a tetrahedral mesh
	tet = tetgen.TetGen(vertices_trimesh,faces_trimesh)
	# #fault tolerant tetrahedralization
	# vertices_tet, elements_tet = tet.tetrahedralize(order=1, mindihedral=0., minratio=10., nobisect=False, steinerleft=100000)
	#high quality tetrahedralization
	vertices, elements = tet.tetrahedralize(order=1, mindihedral=20, minratio=1.5, nobisect=True, steinerleft=-1)
	return vertices, elements

def initialize(vertices,elements, time_initial = 0., mass_density=1.):
	'''elements is a list of indices of tetrahedra with vertices indicated by vertices.
	vertices,elements are both numpy arrays.
	mass_density has units of mass units per volume'''
	t=time_initial
		
	N_elements = elements.shape[0]
	N_vertices = vertices.shape[0]
	# print(f'initialized time to {t}.')

	#allocate memory for numpy arrays
	# initialize nodal arrays
	node_array_position = vertices.copy()
	node_array_initial_position = node_array_position.copy()
	node_array_time = t*(node_array_position[:,0])
	node_array_volume  = 0.*node_array_time

	# initialize elemental arrays
	element_array_index = elements.copy()
	element_array_volume  = 0.*elements.copy()[:,0]
	element_array_time  = t*elements.copy()[:,0]

	#compute the total volume of the elements (not the interior) and the barycentric nodal volumes
	net_volume = 0.
	affine_vec = np.array([1.,1.,1.,1.])
	for K_index in range(N_elements):
		#Behold! One tetrahedral finite volume, K
		K_vertices = vertices[elements[K_index]]

		#compute unsigned volume of one element
		K_X = np.vstack([K_vertices.T,affine_vec])
		K_volume = np.abs(np.linalg.det(K_X))/6.

		net_volume += K_volume
		
		#TODO: compute node_array_volume
		element_array_volume[K_index] = K_volume
		for vid in elements[K_index]:
			node_array_volume[vid] += K_volume/4.

	#assert the barycentric nodal volumes sum to the same value as the net volume of elements  
	assert ( np.isclose ( np.sum(node_array_volume) , np.sum(element_array_volume) ) )
	net_volume = np.sum(element_array_volume)

	net_mass = net_volume*mass_density

	#compute element_array_mass
	element_array_mass = mass_density*element_array_volume

	#compute barycentric nodal masses for the displacement invariant nodal masses
	node_array_mass  = mass_density*node_array_volume

	# initialize momentum to zero
	node_array_momentum = 0.*vertices.copy()
	
	#initialize strain to zero
	node_array_equilibrium_position = node_array_initial_position.copy()

	#precompute element_array_inverse_equilibrium_position
	inverse_equilibrium_position_lst = []
	for K_index in range(N_elements):
		X_inverse = compute_inverse_position(node_array_equilibrium_position, element_array_index, K_index)
		inverse_equilibrium_position_lst.append(X_inverse)
	element_array_inverse_equilibrium_position = np.stack(inverse_equilibrium_position_lst,axis=0)

	# #visual check showed everything was stored right
	# print(element_array_inverse_equilibrium_position[0])
	# print(compute_inverse_position(node_array_equilibrium_position, element_array_index, 0))

	dict_values_system = {
		'element_array_time':element_array_time,
		'element_array_index':element_array_index,
		'element_array_mass':element_array_mass,
		'element_array_volume':element_array_volume,
		'element_array_inverse_equilibrium_position': element_array_inverse_equilibrium_position,
		'node_array_equilibrium_position': node_array_equilibrium_position,
		'node_array_time':node_array_time,
		'node_array_position':node_array_position,
		'node_array_momentum':node_array_momentum,
		'node_array_mass':node_array_mass,
		'node_array_volume':node_array_volume
	}
	return dict_values_system

def initialize_system(input_file_name, time_initial=0., mass_density=1.):
	vertices_trimesh, faces_trimesh = load_trimesh(input_file_name);
	vertices, elements = create_tetrahedral_mesh(vertices_trimesh, faces_trimesh)
	dict_values_system = initialize(vertices,elements, time_initial = time_initial, mass_density=mass_density)
	return dict_values_system

#################################
# Mesh Output
#################################
#TODO: simply save the vertices and elements to txt files in a new folder, or use np.array.savez or something easy.


#################################
# Example Usage: import & initialization of a system
#################################
# os.chdir(nb_dir)
# # input_file_name = f'../data/spherical_meshes/spherical_mesh_64.stl'input_file_name = f'../data/spherical_meshes/spherical_mesh_64.stl'
# input_file_name = f'../data/spherical_meshes/spherical_mesh_1000.stl'
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
#     velocities[j] /= node_array_mass[j]
# v_scale = 1
# velocities[:,0] = v_scale * vertices[:,0].copy()

# #initialize containers of measures
# volume_lst = []
# tme_lst = []

