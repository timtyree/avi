import numpy as np
from numba import njit
from ..model.geom import *
from .elementary_measures import * 

@njit
def compute_net_volume(vertices, elements):
    '''compute the total volume of the elements (not the interior) and the barycentric nodal volumes'''
    N_elements = elements.shape[0]
    net_volume = 0.
    for K_index in range(N_elements):
        #Behold! One tetrahedral finite volume, K
        K_vertices = vertices[elements[K_index]]
        #compute unsigned volume of one element
        K_volume = compute_element_volume(vertices, elements, K_index)
        net_volume += K_volume
    return net_volume
@njit
def comp_element_array_energy(N_elements, element_array_mass, velocities, vertices,
                              elements, element_array_inverse_equilibrium_position, mu, lam):
    element_array_energy = np.zeros_like(element_array_mass)
    for K_index in range(N_elements):
        mass_of_K = element_array_mass[K_index]
        Ka = elements[K_index]
        K_velocities = velocities[Ka]
        K_vertices= vertices[Ka]
        K_W = compute_element_volume(node_array_position=vertices, element_array_index=elements, K_index=K_index)
        Bm  = element_array_inverse_equilibrium_position[K_index]
        Ds = get_D_mat(K_vertices)
        element_array_energy[K_index] = comp_element_energy(mass_of_K, K_velocities, K_W, Bm, Ds, mu, lam)
    return element_array_energy
@njit
def compute_net_energy(N_elements, element_array_mass, velocities, vertices,
                              elements, element_array_inverse_equilibrium_position, mu, lam):
    element_array_energy = comp_element_array_energy(N_elements, element_array_mass, velocities, vertices,
                              elements, element_array_inverse_equilibrium_position, mu, lam)                                                
    net_energy = np.sum(element_array_energy)
    return net_energy