import numpy as np
import torch
from scipy.special import hankel1
from scipy.integrate import dblquad


# --- Extracted Dependencies ---

def generate_em_functions(p):
    XPix, YPix = np.meshgrid(p['x'], p['y'])
    hank_fun = lambda x: 1j * 0.25 * hankel1(0, x)
    transmitter_angles = np.linspace(0, 359, p['numTrans']) * np.pi / 180
    x_transmit = p['sensorRadius'] * np.cos(transmitter_angles)
    y_transmit = p['sensorRadius'] * np.sin(transmitter_angles)
    receiver_angles = np.linspace(0, 359, p['numRec']) * np.pi / 180
    x_receive = p['sensorRadius'] * np.cos(receiver_angles)
    y_receive = p['sensorRadius'] * np.sin(receiver_angles)
    p['receiverMask'] = np.ones((p['numTrans'], p['numRec']))
    
    diff_x_rp = np.tile(XPix[:, :, np.newaxis], (1, 1, p['numRec'])) - \
                np.tile(x_receive[np.newaxis, np.newaxis, :], (p['Ny'], p['Nx'], 1))
    diff_y_rp = np.tile(YPix[:, :, np.newaxis], (1, 1, p['numRec'])) - \
                np.tile(y_receive[np.newaxis, np.newaxis, :], (p['Ny'], p['Nx'], 1))
    distance_rec_to_pix = np.sqrt(diff_x_rp**2 + diff_y_rp**2)

    diff_x_tp = np.tile(XPix[:, :, np.newaxis], (1, 1, p['numTrans'])) - \
                np.tile(x_transmit[np.newaxis, np.newaxis, :], (p['Ny'], p['Nx'], 1))
    diff_y_tp = np.tile(YPix[:, :, np.newaxis], (1, 1, p['numTrans'])) - \
                np.tile(y_transmit[np.newaxis, np.newaxis, :], (p['Ny'], p['Nx'], 1))
    distance_trans_to_pix = np.sqrt(diff_x_tp**2 + diff_y_tp**2)
    
    p['uincDom'] = hank_fun(p['kb'] * distance_trans_to_pix)
    sensor_greens_function = hank_fun(p['kb'] * distance_rec_to_pix)
    p['sensorGreensFunction'] = (p['kb']**2) * sensor_greens_function
    
    x_green = np.arange(-p['Nx'], p['Nx']) * p['dx']                        
    y_green = np.arange(-p['Ny'], p['Ny']) * p['dy']
    XGreen, YGreen = np.meshgrid(x_green, y_green)
    R = np.sqrt(XGreen**2 + YGreen**2)
    domain_greens_function = hank_fun(p['kb'] * R)
    
    def integrand_real(x, y):
            if x == 0 and y == 0:
                return 0.0
            return np.abs(hank_fun(p['kb'] * np.sqrt(x**2 + y**2)).real)
    def integrand_imag(x, y):
        if x == 0 and y == 0:
            return 0.0
        return np.abs(hank_fun(p['kb'] * np.sqrt(x**2 + y**2)).imag)
    
    Ny = p['Ny']
    Nx = p['Nx']
    dx = p['dx']
    dy = p['dy']
    domain_greens_function[Ny, Nx] = dblquad(
            integrand_real,
            -dx/2, dx/2, -dy/2, dy/2
        )[0] / (dx * dy)
    domain_greens_function[Ny, Nx] += (dblquad(
        integrand_imag,
        -dx/2, dx/2, -dy/2, dy/2
    )[0] / (dx * dy)) * 1j
    
    p['domainGreensFunction'] = (p['kb']**2) * domain_greens_function
    return p

def construct_parameters(Lx=0.18, Ly=0.18, Nx=128, Ny=128, wave=6, numRec=360, numTrans=60, sensorRadius=1.6,
                         device='cuda'):
    em = {}
    em['Lx'] = Lx
    em['Ly'] = Ly
    em['Nx'] = Nx
    em['Ny'] = Ny
    em['dx'] = em['Lx'] / em['Nx']
    em['dy'] = em['Ly'] / em['Ny']
    em['x'] = np.linspace(-em['Nx']/2, em['Nx']/2 - 1, em['Nx']) * em['dx']
    em['y'] = np.linspace(-em['Ny']/2, em['Ny']/2 - 1, em['Ny']) * em['dy']
    em['c'] = 299792458
    em['lambda'] = em['dx'] * wave
    em['freq'] = em['c'] / em['lambda'] / 1e9
    em['numRec'] = numRec
    em['numTrans'] = numTrans
    em['sensorRadius'] = sensorRadius
    em['kb'] = 2 * np.pi / em['lambda']
    em = generate_em_functions(em)
    return torch.from_numpy(em['domainGreensFunction']).to(device).unsqueeze(-1), torch.from_numpy(em['sensorGreensFunction']).to(device).unsqueeze(-1), torch.from_numpy(em['uincDom']).to(device).unsqueeze(-1), torch.from_numpy(em['receiverMask']).unsqueeze(-1)
