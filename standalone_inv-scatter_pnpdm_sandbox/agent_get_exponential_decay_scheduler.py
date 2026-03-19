

# --- Extracted Dependencies ---

def get_exponential_decay_scheduler(num_steps, sigma_max, sigma_min, rho=0.9):
    sigma_steps = []
    sigma = sigma_max
    for i in range(num_steps):
        sigma_steps.append(sigma)
        sigma = max(sigma_min, sigma * rho)
    return sigma_steps
