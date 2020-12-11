# Measures for Finite Element (FE) extrinsic discrete measures
# Tim Tyree
# 9.23.2020
import numpy as np
import pandas as pd
from numba import njit

#TODO: fill this out/in with functions that make y-axis values

def discrete_action_sum (nodes_iterable, **kwargs) :
	pass

def discrete_kinetic_energy_sum (nodes_iterable, **kwargs) :
	return sum ( [discrete_kinetic_energy(node, **kwargs) for node in nodes_iterable])

def discrete_action_nodal(node, **kwargs) :
	neighbors_iterable = node.neighbors_iterable
	mass = node.mass

def volume_sum( faces_iterable, **kwargs) :
	pass