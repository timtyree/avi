# one_step_forward_euler.py
import numpy as np
from ..model import *
from . import *

def get_one_step_implicit_midpoint_rule(mu,lam,gamma,num_iter = 30):
    zero_mat = np.zeros((4,3))
    compute_one_step_forward_euler_method=get_compute_one_step_forward_euler_method(mu,lam,gamma)
    @njit
    def one_step_implicit_midpoint_rule(t, K_index, element_array_index, 
                                              tauK, tau, vertices, velocities, 
                                              node_array_mass, 
                                              element_array_inverse_equilibrium_position):
        #actionables per function call are as follow:
        t_previous = tauK[K_index]
        Ka = element_array_index[K_index]
        x_current, v_current = vertices[Ka], velocities[Ka]
        #compute the namespace of the force computation of the elemental configuration
        tau_of_K = tauK[K_index]
        K_tau = tau[Ka]
        K_vertices = vertices[Ka].copy()
        K_velocities = velocities[Ka].copy()
        K_masses = node_array_mass[Ka]
        Bm  = element_array_inverse_equilibrium_position[K_index]
        #initialize the first iterate of the soln to the implicit midpoint rule with the forward euler method (actually the explicit newmark method)
        K_vertices, K_velocities = compute_one_step_forward_euler_method(t, K_vertices, K_velocities, K_masses, K_tau, tau_of_K, Bm, zero_mat)
        x_next, v_next = K_vertices.copy(), K_velocities.copy()
        x_avg = x_current/2. + x_next/2.
        v_avg = v_current/2. + v_next/2.
        # madx_lst = []
        # madv_lst = []
        # k_lst = []
        # iter_count =0
        for k in range(num_iter):
            #compute the next iterate of the soln to the implicit midpoint rule num_iter times
            K_vertices, K_velocities = compute_one_step_forward_euler_method(t, x_avg, v_avg, K_masses, K_tau, tau_of_K, Bm, zero_mat)
            #     iter_count +=1
            #     delta_x = K_vertices - x_next
            #     delta_v = K_velocities - v_next
            x_next, v_next = K_vertices.copy(), K_velocities.copy()
            x_avg = x_current/2. + x_next/2.
            v_avg = v_current/2. + v_next/2.
            #     madx = np.max(np.abs(delta_x))
            #     madv = np.max(np.abs(delta_v))
            #     madx_lst.append(madx)
            #     madv_lst.append(madv)
            #     k_lst.append(k)
        return x_next, v_next
    return one_step_implicit_midpoint_rule

def get_local_one_step_implicit_midpoint_rule(mu,lam,gamma,num_iter = 30):
    zero_mat = np.zeros((4,3))
    compute_one_step_forward_euler_method=get_compute_one_step_forward_euler_method(mu,lam,gamma)
    @njit
    def local_one_step_implicit_midpoint_rule(t, x_current, v_current, K_masses, K_tau, tau_of_K, Bm, zero_mat):  
        #initialize the first iterate of the soln to the implicit midpoint rule
        x_next, v_next = compute_one_step_forward_euler_method(t, x_current, v_current, K_masses, K_tau, tau_of_K, Bm, zero_mat)
        x_avg = x_current/2. + x_next/2.
        v_avg = v_current/2. + v_next/2.
        for k in range(num_iter):
            #compute the next iterate of the soln to the implicit midpoint rule num_iter times
            K_vertices, K_velocities = compute_one_step_forward_euler_method(t, x_avg, v_avg, K_masses, K_tau, tau_of_K, Bm, zero_mat)
            x_next, v_next = K_vertices.copy(), K_velocities.copy()
            x_avg = x_current/2. + x_next/2.
            v_avg = v_current/2. + v_next/2.
        return x_next, v_next
    return local_one_step_implicit_midpoint_rule




