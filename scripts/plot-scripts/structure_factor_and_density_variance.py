import sys
import os
sys.path.append('../utils')
import Utils
import numpy as np
import matplotlib.pyplot as plt
import json
from scipy import stats
from scipy.optimize import curve_fit, fsolve
import csv
from numpy import genfromtxt
import pandas as pd
from scipy.special import jv, gamma
from scipy.integrate import simps
import math

def save_data(k,s_k,output_file):
    if isinstance(k,np.ndarray):
        k = k.tolist()
    if isinstance(s_k,np.ndarray):
        s_k = s_k.tolist()
    data = {'k':k,'s_k':s_k}
    with open(output_file,'w') as fs:
        fs.write(json.dumps(data))

def read_data(loc):
    if not os.path.isfile(loc):
        return None
    with open(loc,'r') as fs:
        data = json.loads(fs.read())
    k = np.array(data['k'])
    s_k = np.array(data['s_k'])
    return (k,s_k)

def density_variance(R, d, k_vals, S_vals, rho=1.0):
    """
    Compute the d-dimensional number variance sigma_N^2(R) given:
      - R:      radius of the d-dimensional sphere
      - d:      dimension
      - k_vals: array of wavenumbers (0 <= k <= k_max)
      - S_vals: array of S(k) values corresponding to k_vals
      - rho:    number density (default = 1.0)

    Returns:
      sigma2: the number variance sigma_N^2(R).
    """

    # --- 1) Volume of the d-dimensional ball of radius R
    #     v1(R) = (pi^(d/2) / Gamma(d/2 + 1)) * R^d
    v1_R = (np.pi**(d/2) / gamma(d/2 + 1)) * R**d

    # --- 2) Surface area of the unit d-sphere
    #     s1(1) = 2 * pi^(d/2) / Gamma(d/2)
    s1_1 = 2.0 * np.pi**(d/2) / gamma(d/2)

    # --- 3) Define alpha_2_tilde(k; R)
    #     alpha_2_tilde(k;R) = 2^d * pi^(d/2) * Gamma(1 + d/2) * [J_{d/2}(kR)]^2 / k^d
    k_vals = np.asarray(k_vals)
    S_vals = np.asarray(S_vals)

    alpha_2 = (
        2.0**d
        * np.pi**(d/2)
        * gamma(1.0 + d/2.0)
        * (jv(d/2.0, R * k_vals) ** 2)
        / (k_vals**d)
    )

    # --- 4) Construct the integrand: k^(d-1) * S(k) * alpha_2_tilde(k; R)
    integrand = (k_vals ** (d - 1)) * S_vals * alpha_2

    # --- 5) Perform numerical integration
    integral_val = simps(integrand, k_vals)

    # --- 6) Multiply by prefactor: rho * v1(R) * s1(1) / (2 pi)^d
    # prefactor = rho * v1_R * s1_1 / ((2.0 * np.pi) ** d)            #number variance
    prefactor = rho * v1_R * s1_1 / ((2.0 * np.pi * v1_R) ** d)       #density variance
    # prefactor = s1_1 / ((2.0 * np.pi) ** d)                         #number variance/mean
    sigma2 = prefactor * integral_val 

    return sigma2

def get_radial_sk_and_den_fluc(simulation_dir:str,N = 300,n_frames=5,num_dia=100):
    dl = Utils.DataLoader(simulation_dir,mode="range",rng=[-n_frames,-1])
    iters,x_list = dl.get_data_list('x')
    x = [dl.get_pbc_position(np.array(_x)) for _x in x_list]
    sk_list = []
    for _x in x:
        if dl.info['dim'] == 1:
            kx,s = dl.get_strucutre_factor(_x,N)
            k,sk = Utils.get_radial_profile(s)
            rho = dl.info['N']/(dl.info['box_size'][0])      
        elif dl.info['dim'] == 2:
            kx,ky,s = dl.get_strucutre_factor(_x,N)
            k,sk = Utils.get_radial_profile(s)
            rho = dl.info['N']/(dl.info['box_size'][0]*dl.info['box_size'][1])      
        elif dl.info['dim'] == 3:
            kx,ky,kz,s = dl.get_strucutre_factor(_x,N)
            k,sk = Utils.get_radial_profile(s)
            rho = dl.info['N']/(dl.info['box_size'][0]*dl.info['box_size'][1]*dl.info['box_size'][2])      
        sk_list.append(sk)
        
    # k = k.astype(float)*np.max(kx)/np.max(k)*dl.info["r"]*2.0/(2.0*np.pi)
    k = (k.astype(float)*np.max(kx)/np.max(k))
    avg_sk = np.mean(np.array(sk_list),axis=0)
    method_str = dl.info['optimization_method']
    if method_str == "inversepower_probabilistic_pairwise_sgd":
        method_str = "pairwise SGD, phi=" + str(dl.info['phi'])
    elif method_str == "inversepower_probabilistic_particlewise_sgd":
        method_str = "particle-wise SGD, phi=" + str(dl.info['phi'])


    #calculate density fluctuations using averaged S(k)
    radii = np.logspace(np.log10(dl.info['r']), np.log10(0.5*dl.info['box_size'][0]), num=num_dia)

    # Compute sigma_N^2(R) for each R
    sigma_vals = []
    for R in radii:
        sigma2 = density_variance(R, dl.info['dim'], k, avg_sk, rho=rho)
        sigma_vals.append(sigma2)

    return k,avg_sk,2.0*radii,sigma_vals


data_dir_list = ["/scratch/sa7483/sig-python/sips/scripts/testing/correlated_pairwise_ro_scan_dim/dir0#eps_1.000000E+00#corr_mag_0.000000E+00#runID_0",
                "/scratch/sa7483/sig-python/sips/scripts/testing/correlated_pairwise_ro_scan_dim/dir1#eps_1.000000E+00#corr_mag_1.000000E-01#runID_0",
                "/scratch/sa7483/sig-python/sips/scripts/testing/correlated_pairwise_ro_scan_dim/dir2#eps_1.000000E+00#corr_mag_2.000000E-01#runID_0",
                "/scratch/sa7483/sig-python/sips/scripts/testing/correlated_pairwise_ro_scan_dim/dir3#eps_1.000000E+00#corr_mag_3.000000E-01#runID_0",
                "/scratch/sa7483/sig-python/sips/scripts/testing/correlated_pairwise_ro_scan_dim/dir4#eps_1.000000E+00#corr_mag_4.000000E-01#runID_0",
                "/scratch/sa7483/sig-python/sips/scripts/testing/correlated_pairwise_ro_scan_dim/dir5#eps_1.000000E+00#corr_mag_5.000000E-01#runID_0",
                "/scratch/sa7483/sig-python/sips/scripts/testing/correlated_pairwise_ro_scan_dim/dir6#eps_1.000000E+00#corr_mag_6.000000E-01#runID_0",
                "/scratch/sa7483/sig-python/sips/scripts/testing/correlated_pairwise_ro_scan_dim/dir7#eps_1.000000E+00#corr_mag_7.000000E-01#runID_0",
                "/scratch/sa7483/sig-python/sips/scripts/testing/correlated_pairwise_ro_scan_dim/dir8#eps_1.000000E+00#corr_mag_8.000000E-01#runID_0",
                "/scratch/sa7483/sig-python/sips/scripts/testing/correlated_pairwise_ro_scan_dim/dir9#eps_1.000000E+00#corr_mag_9.000000E-01#runID_0",
                "/scratch/sa7483/sig-python/sips/scripts/testing/correlated_pairwise_ro_scan_dim/dir10#eps_1.000000E+00#corr_mag_1.000000E+00#runID_0"]

figure_size = [8.0, 4.0]
fig, (ax1, ax2) = plt.subplots(1, 2)
fig.set_size_inches(figure_size[0], figure_size[1])
colors = ['red','blue','orange','green','black','brown', 'purple','gray', 'cyan', 'pink', 'royalblue']
force_calculation = False
c = 0

var_list = []
len_list = []
k_list = []
s_list = []

for data_dir in data_dir_list:
    # for _dir in os.listdir(data_dir):
    #     if _dir.find('run') < 0:
    #         continue
    # simulation_dir = os.path.join(data_dir,_dir)

    simulation_dir = data_dir
    output_file = os.path.join(simulation_dir,'structure_factor.json')
    if os.path.isfile(output_file) and (not force_calculation):
        k,s_k = read_data(output_file)
    else:
        # k,s_k,lengths,var = get_radial_sk_and_den_fluc(simulation_dir, N = 1000, n_frames=100, num_dia=100)    #use for 2d
        k,s_k,lengths,var = get_radial_sk_and_den_fluc(simulation_dir, N = 100, n_frames=50, num_dia=100)       #use for 3d

    k_list.append(k)
    s_list.append(s_k)
    len_list.append(lengths)
    var_list.append(var)

    ax1.errorbar(lengths,var,fmt='.',c=colors[c],alpha=0.6)
    ax1.set_xscale('log')
    ax1.set_yscale('log')
    ax2.plot(k, s_k, '.', c=colors[c], alpha=0.6)
    ax2.set_xscale('log')
    ax2.set_yscale('log')
    c+=1

ax1.tick_params(axis='x',labelsize = 14)
ax1.tick_params(axis = 'y', which = 'both', labelsize = 14)
ax1.set_xlabel(r"$L$",fontsize=14)
ax1.set_ylabel("$Var(L)$",fontsize=14)

ax2.tick_params(axis='x',labelsize = 14)
ax2.tick_params(axis = 'y', which = 'both', labelsize = 14)
ax2.set_xlabel(r"$k$",fontsize=14)
ax2.set_ylabel("$S(k)$",fontsize=14)

fig.tight_layout()
fig.savefig('strucutre_factor.png', dpi=300)
plt.show()


#save the structure factor and density variance
data_dir_save = "/scratch/sa7483/sig-python/sips/scripts/testing/correlated_pairwise_ro_scan_dim"

file_path_1 = os.path.join(data_dir_save, 'var.json')
file_path_2 = os.path.join(data_dir_save, 'lengths.json')
file_path_3 = os.path.join(data_dir_save, 's.json')
file_path_4 = os.path.join(data_dir_save, 'k.json')
os.makedirs(data_dir_save, exist_ok=True)
with open(file_path_1, 'w') as file:
    json.dump(np.array(var_list).tolist(), file)
with open(file_path_2, 'w') as file:
    json.dump(np.array(len_list).tolist(), file)
with open(file_path_3, 'w') as file:
    json.dump(np.array(s_list).tolist(), file)
with open(file_path_4, 'w') as file:
    json.dump(np.array(k_list).tolist(), file)

