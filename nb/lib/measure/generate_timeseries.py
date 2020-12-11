from ..my_initialization import *

def generate_timeseries(mu,lam,gamma,salience,lasso_fraction,v_scale,mass_density,stepsize_init,atol_x,atol_v,btol_x,btol_v,mesh_size,tmax,data_fn, **kwargs):

    learning_rate = np.log(salience)
    #path to mesh
    input_file_name = f'../data/spherical_meshes/spherical_mesh_{mesh_size}.stl'
    # input_file_name = f'../data/spherical_meshes/spherical_mesh_1000.stl'

    # input_file_name = f'../data/spherical_meshes/spherical_mesh_400.stl'
    #where to save results
    data_folder =  os.path.join(nb_dir,'../data/mov_csv')
    data_fn = f"avi_es_fixed_lr_a_equal_b_{os.path.basename(input_file_name).replace('.stl',f'_mu_{mu}_lambda_{lam}_gamma_{gamma}_vscale_{v_scale}_stepsizeinit_{stepsize_init}')}_salience_{salience}_atolx_{atol_x}.csv"
    # data_fn = f"avi_ns_fixed_lr_a_equal_b_{os.path.basename(input_file_name).replace('.stl',f'_mu_{mu}_lambda_{lam}_gamma_{gamma}vscale_{v_scale}_stepsizeinit_{stepsize_init}')}_salience_{salience}_atolx_{atol_x}.csv"
    save_folder_vid = '../vid/tmp'
    folder_vid = '../vid'
    # data_fn_counts = data_fn.replace("s_","s_counts_").replace('.csv','.npz')
    data_fn_counts = data_fn.replace('.csv','.npy')

    os.chdir(nb_dir)
    # input_file_name = f'../data/spherical_meshes/spherical_mesh_64.stl'input_file_name = f'../data/spherical_meshes/spherical_mesh_64.stl'
    # input_file_name = f'../data/spherical_meshes/spherical_mesh_1000.stl'
    input_file_name = os.path.join(nb_dir,input_file_name)
    tme = 0.
    dict_values_system = initialize_system(input_file_name, time_initial=tme, mass_density=mass_density)
    locals().update(dict_values_system)
    N_elements = element_array_index.shape[0]
    N_vertices = node_array_position.shape[0]
    print(list(dict_values_system.keys()))


    integrate_system_explicit_asynchronous = get_integrate_system_explicit_asynchronous(mu,lam,gamma)



    #initialize system
    tauK = element_array_time
    tau = node_array_time

    #get method of computing elastic forces
    zero_mat = np.zeros((4,3))
    # calc_P = get_calc_P(mu, lam)
    # compute_nodal_damping_forces  = get_compute_nodal_damping_forces(mu,lam,gamma)
    # comp_nodal_elastic_forces = get_comp_nodal_elastic_forces(mu, lam)

    elements = element_array_index
    vertices = node_array_position

    #initialize stepsizes of simulation
    element_array_stepsize = np.zeros_like(element_array_time) + stepsize_init
    element_array_count_calls_one_step = np.zeros_like(element_array_time,dtype=int)
    element_array_count_config_updates = np.zeros_like(element_array_time,dtype=int)
    momentum = node_array_momentum.copy()
    velocities = momentum.copy()
    for j in range(N_vertices):
        velocities[j] /= node_array_mass[j]


    # #perturb momentum in the x direction and let it run overnight with a small timestep
    velocities[:,0] = -v_scale * vertices[:,0].copy()

    #initialize containers of measures
    volume_lst = []
    energy_lst = []
    tme_lst    = []
    stepsize_mean_lst   = []
    stepsize_std_lst    = []
    stepsize_median_lst = []
    stepsize_count_lst = []
    frac_lb = lambda i: np.exp(learning_rate*i)  ##np.exp((i+.5)*learning_rate/10)
    learning_bins = np.array([stepsize_init*frac_lb(i) for i in np.arange(-15,15)])

    # learning_bins = np.array([stepsize_init*np.exp((i+.5)*learning_rate) for i in np.arange(-3,3)])
    print(f"learning_bins are {learning_bins}")



    #prepare for video
    frameno = 1
    time_between_observations = 0.01
    time_end_recording = tmax
    time_of_next_observation = tme + time_between_observations

    #ready... get set... GO!
    os.chdir(nb_dir)
    os.chdir(save_folder_vid)

    while time_of_next_observation <= time_end_recording:
        tf = time_of_next_observation
        #integrate forward to the next time of observation
    #     integrate_system_explicit_synchronous(tf, element_array_time, element_array_stepsize, node_array_time,
    #                                              element_array_index, vertices, velocities,node_array_mass, element_array_inverse_equilibrium_position)
        #     integrate_system_implicit_synchronous(tf, element_array_time, element_array_stepsize, node_array_time,
        #                                          element_array_index, vertices, velocities,node_array_mass, element_array_inverse_equilibrium_position)
        #     integrate_system_dormand_prince_asynchronous(tf,element_array_time,element_array_stepsize,node_array_time,element_array_index,vertices,velocities,
        #         node_array_mass,element_array_inverse_equilibrium_position,element_array_mass)
        integrate_system_explicit_asynchronous(tf, element_array_time, element_array_stepsize, node_array_time,
                                                 element_array_index, vertices, velocities, node_array_mass, element_array_inverse_equilibrium_position, atol_x, atol_v, btol_x, btol_v, learning_rate)

        #update a copy of all positions to the observation time using the current velocity
        x = vertices.copy()
        for a in range(N_vertices):
            x[a] += velocities[a] * (tf - tau[a])

        #measure observables
        ##mesh measures
        net_volume = compute_net_volume(x, element_array_index)
        net_energy = compute_net_energy(N_elements, element_array_mass, velocities, x, #vertices,
                                      element_array_index, element_array_inverse_equilibrium_position, mu, lam)
        ##stepsize measures
        stepsize_mean   = np.mean(element_array_stepsize)
        stepsize_std    = np.std(element_array_stepsize)
        stepsize_median = np.median(element_array_stepsize)

        #record observables
        volume_lst.append(net_volume)
        energy_lst.append(net_energy)
        tme = tf
        tme_lst.append(tme)
        stepsize_mean_lst.append(stepsize_mean)
        stepsize_std_lst.append(stepsize_std)
        stepsize_median_lst.append(stepsize_median)
        stepsize_count_lst.append(count_array(array = element_array_stepsize,bins=learning_bins))
        #record image of system
    #     try:
    #         img = get_img_of_system(vertices, input_file_name, darkmode = False, text=f'time={tf:.2f}')
    #     except:
    #         pass
    #     save_fn_img = f'img{frameno:09}.png'
    #     frameno += 1
    #     Img = Image.fromarray(img)
    # #     Img.save(save_fn_img)


    #     del Img
        #increment time_of_next_observation
        time_of_next_observation += time_between_observations
        #TODO(later): do something that makes a boring video less boring
        #TODO(later): def element_array_count_calls_one_step to the call of integrate_system_.... i.e. Keep track of how many times each element is called to time step
        #TODO(later): keep track of how many times each element config is updated with element_array_count_config_updates
    print(stepsize_mean, stepsize_std, stepsize_median)
    # beep(3)


    df = pd.DataFrame({
        't':tme_lst,
        'volume':volume_lst,
        'energy':energy_lst,
        'stepsize_mean':stepsize_mean_lst,
        'stepsize_std':stepsize_std_lst,
        'stepsize_median':stepsize_median_lst,
    })


    os.chdir(data_folder)
    df.to_csv(data_fn, index=False)

    np.array(stepsize_count_lst)
    # np.savez(data_fn_counts)
    np.save(data_fn_counts, np.array(stepsize_count_lst))

    return data_fn

# ####################################
# # Example Usage
# ####################################


# kwargs= {
# 'mu' : 1.,
# 'lam' : 10.,
# 'gamma' : .05,#1.#1.;
# 'salience' : 2,#128#32
# 'lasso_fraction' : 0.5,
# 'v_scale' : 2.,
# 'mass_density':1.,
# 'stepsize_init' : 0.002,#0.005#0.00001#0.0001
# 'atol_x' : 0.00001,#0.001#1e-7;
# 'atol_v' : 0.00001,#0.001#1e-7;
# 'btol_x' : 0.000001,#0.001#1e-10;
# 'btol_v' : 0.000001,#0.001#1e-10;
# 'mesh_size' : 64,
#     'tmax':,
# }
# #input triangular mesh
# input_file_name = f'../data/spherical_meshes/spherical_mesh_{mesh_size}.stl'
# #where to save results
# data_folder =  os.path.join(nb_dir,'../data/mov_csv')
# data_fn = f"avi_es_fixed_lr_a_equal_b_{os.path.basename(input_file_name).replace('.stl',f'_mu_{mu}_lambda_{lam}_gamma_{gamma}_vscale_{v_scale}_stepsizeinit_{stepsize_init}')}_salience_{salience}_atolx_{atol_x}.csv"
# #TODO: update kwargs
# generate_timeseries(**kwargs)
# beep(1)
