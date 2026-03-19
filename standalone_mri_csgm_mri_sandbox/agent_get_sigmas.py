import numpy as np


# --- Extracted Dependencies ---

def get_sigmas(sigmas_config):
    if sigmas_config['sigma_dist'] == 'geometric':
        sigmas = np.exp(np.linspace(np.log(sigmas_config['sigma_begin']), np.log(sigmas_config['sigma_end']), sigmas_config['num_steps']))
    elif sigmas_config['sigma_dist'] == 'uniform':
        sigmas = np.linspace(sigmas_config['sigma_begin'], sigmas_config['sigma_end'], sigmas_config['num_steps'])
    else:
        raise NotImplementedError('sigma distribution not supported')
    return sigmas
