#Integrate a tetrahedral mesh forward in time with synchronous time updates
#Tim Tyree
#12/6/2020

import numpy as np, heapq
from numba import njit
from ..model import *
from . import *

def get_integrate_system_explicit_synchronous(mu,lam,gamma):
    one_step_forward_euler_method = get_one_step_forward_euler_method(mu,lam,gamma)
    @njit
    def integrate_system_explicit_synchronous(tf, element_array_time, element_array_stepsize, node_array_time,
                                             element_array_index, vertices, velocities, 
                                          node_array_mass, element_array_inverse_equilibrium_position):
        """integrate up to time tf with synchronous forward euler integration"""
        #initialize the queue by pushing all finite elements into the queue
        N_elements = element_array_stepsize.shape[0]
        queue = [(element_array_time[K_index],element_array_stepsize[K_index],K_index) for K_index in range(N_elements)]
        heapq.heapify(queue)

        tauK = element_array_time
        tau  = node_array_time
        #do until priority queue is empty
        while len(queue) > 0:
            t,h,K_index = heapq.heappop(queue)
            if t<=tf:
                t_previous = tauK[K_index]
                #TODO: for each stepcase: #(for the purpose of comparing each stepcase)
                #TODO: update the configuration arrays of each stepcase with the configuration for this element
                #timestep the configuration with forward euler integration
                K_vertices, K_velocities = one_step_forward_euler_method(t, K_index, element_array_index, 
                                              tauK, tau, vertices, velocities, 
                                              node_array_mass, 
                                              element_array_inverse_equilibrium_position)
                #update the elemental times
                tauK[K_index] = t
                Ka = element_array_index[K_index]
                #update node's time
                tau[Ka] = t
                #update node's position
                vertices[Ka] = K_vertices
                velocities[Ka] = K_velocities

                #TODO: compare results for the stepcases. choose an h_next for the next time for element's evaluation
                stepsize_next = h
                t_next = t + stepsize_next
                #TODO: update the actual configuration with the results of the best stepcase
                element_array_stepsize[K_index] = stepsize_next
                #TODO: push the next task to the heap/priority queue
                heapq.heappush(queue, (t_next, stepsize_next, K_index))

        #synchronize all nodes  to the current time, tf.
        for K_index in range(N_elements):
            Ka = element_array_index[K_index]
            for a in Ka:
                vertices[a] = vertices[a] + velocities[a] * (tf - tau[a])
        return True
    return integrate_system_explicit_synchronous