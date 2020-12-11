import numpy as np
from numba import njit

#############################################################
# Elastic energy for a neohookean hyperelastic consitutive model
#############################################################
@njit
def comp_elastic_energy (  mass_of_K, K_W, Bm, Ds, mu, lam ):
    '''elastic potential energy of element according to the neohookean constitutive model.'''
    F = Ds @ Bm#np.matmul(Ds, Bm)#
    J  = np.linalg.det(F)
    I1 = np.trace(F.T @ F)#np.matmul(F.T, F))#
    energy_density = mu / 2 * (I1 - 3) - mu * np.log(J) + lam / 2 * np.log(J) ** 2 #alternative formulation not in use:  (J - 1)**2
    elastic_energy = K_W * energy_density
    return elastic_energy

#############################################################
# Kinetic energy
#############################################################
@njit
def comp_kinetic_energy ( mass_of_K, K_velocities):
    '''kinetic energy of element'''
    Na = 4
    kinetic_energy = 0.
    for a in range(Na):
        kinetic_energy += ( mass_of_K / Na ) * np.dot(K_velocities[a],K_velocities[a])
    kinetic_energy /= 2
    return kinetic_energy

#############################################################
# Energy
#############################################################
@njit
def comp_element_energy ( mass_of_K, K_velocities, K_W, Bm, Ds, mu, lam):
    return comp_kinetic_energy ( mass_of_K, K_velocities) + comp_elastic_energy (  mass_of_K, K_W, Bm, Ds, mu, lam )


def count_array(array, bins):
    counts, bins = np.histogram(array, bins=bins)
    return counts