compute_next_time

def compute_next_time(K_index, t, stepsize):
    '''compute next time for element's evaluation'''
    return _compute_next_time(K_index, t, stepsize)

def _compute_next_time(K, t, stepsize):
    '''compute next time for element's evaluation'''
    return t + stepsize