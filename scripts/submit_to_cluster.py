import sys
import os
import matplotlib.pyplot as plt
import itertools
import subprocess
import json
import time
import numpy as np
import copy
import math


class BatchJobManager:
    def __init__(self, input_config, comp_config,root_dir,rm = False):
        self.input_config = input_config
        self._input_config = input_config
        self.comp_config = comp_config
        self.root_dir = root_dir
        self.check_nthreads(float(comp_config["NTHREADS"]),input_config["N"],input_config["phi"])
        self.count = 0
        self.check_input()
        if os.path.isdir(root_dir) and rm:
            subprocess.run(['rm','-r',root_dir])
        subprocess.run(['mkdir',root_dir])
        
    def check_nthreads(self,Nth,N,phi):
        if Nth*Nth*Nth > 3.14159*N/(48*phi):
            raise RuntimeError("#Threads check fails \n Increase the number of particles or decrease the number of threads ")
    def check_input(self):
        if type(self.input_config['n_save']) == str:
            raise RuntimeError("n_save should be a int ")
    #scan two parameters
    def scan_params(self,scan_dict,parent_param = None,n_repeat = 1):
        '''
        root_dir--root directory to store the data
        scan_dict--the dictionary for parameters e,g {"phi":[0.1,0.2],"N":[100,200],"batch_szie":[1,2]}
        n_repeat--runs n_repeat computaion(with the same input parameters but different initial configuration) 
        parent_param -- a new directory will be created for each value of the parent_param 
        '''
        if parent_param:
            parent_param_values = scan_dict[parent_param]
            scan_dict.pop(parent_param)
            for ppv in parent_param_values:
                ppv_str = self.parameters2str({parent_param:ppv})
                sub_dir_name = os.path.join(self.root_dir,ppv_str)
                subprocess.run(['mkdir',sub_dir_name])
                count = 0
                for run_id in range(n_repeat):
                    self.input_config["runID"] = run_id
                    for pv_dict in self.get_unzip_dict(scan_dict):
                        data_dir_name = "dir{}#".format(count) + self.parameters2str(pv_dict) + "#runID_"+str(run_id) 
                        subprocess.run(['mkdir',os.path.join(sub_dir_name,data_dir_name)])
                        self.input_config[parent_param] = ppv
                        self.update_input_config(pv_dict)
                        self.write_input(os.path.join(sub_dir_name,data_dir_name))
                        self.submit_config(os.path.join(sub_dir_name,data_dir_name))
                        self.restore_input_config()
                        count += 1
        else:
            count = 0
            for run_id in range(n_repeat):
                self.input_config["runID"] = run_id
                for pv_dict in self.get_unzip_dict(scan_dict):
                    data_dir_name = "dir{}#".format(count) + self.parameters2str(pv_dict) + "#runID_"+str(run_id) 
                    subprocess.run(['mkdir',os.path.join(self.root_dir,data_dir_name)])
                    self.update_input_config(pv_dict)
                    self.write_input(os.path.join(self.root_dir,data_dir_name))
                    self.submit_config(os.path.join(self.root_dir,data_dir_name))
                    self.restore_input_config()
                    count += 1
            
                    
    def parameters2str(self,pv_dict):
        rtn = ""
        for p in pv_dict.keys():
            if type(pv_dict[p]) == float or type(pv_dict[p]) == np.float64:
                rtn +="{}_{:.6E}#".format(p,pv_dict[p])
            else:
                rtn +="{}_{}#".format(p,pv_dict[p])
        return rtn[:-1]
        
        
    def get_unzip_dict(self,scan_dict):
        '''
        scan_dict = {'a':[a1,a2..an],'b':[b1,b2..bn]...}
                  where a and b are parameter names in input_config.keys()
        return unzipped dicts, {'a':a1,'b':b1 ...},{'a':a1,'b':b2 ...}.
        '''
        value_lists=  list(itertools.product(*scan_dict.values())) 
        param_name_list = list(scan_dict.keys())
        for v_list in value_lists:
            param_value_dict = {param_name_list[i]:v_list[i] for i in range(len(param_name_list))}
            yield param_value_dict

    def update_input_config(self,config):
        for k in config.keys():
            self.input_config[k] = config[k]

    def restore_input_config(self):
        self.input_config = self._input_config

    def write_input(self,loc):
        with open(os.path.join(loc,"info.json"),'w') as fs:
            fs.write(json.dumps(self.input_config,indent=4))
    
    def submit_config(self,args,nsleep = 0.05):
        with open('./submit_batch.INI', 'r') as file:
            fstring = file.read()
        for key in self.comp_config:
            fstring = fstring.replace(key, self.comp_config[key])
        
        if type(args) != str:
            raise NotImplementedError("args must be a string")
        fstring = fstring.replace("ARGS", args)
        with open('./submit_batch.sbatch', 'w') as file:
            file.write(fstring)
        subprocess.run(["sbatch", "submit_batch.sbatch"])
        time.sleep(nsleep)
        self.count += 1
        subprocess.run(["rm", "submit_batch.sbatch"])
    

input_config3D = {
    "runID": 0,
    "dim": 3,
    "N": 64,
    "box_size": [
        100.0,
        100.0,
        100.0
    ],
    "phi": 0.15,
    "n_steps": 100000,
    "n_save": 1000,
    "n_rec":1000,
    "save_mode": "concise",
    "optimization_method": "particlewise_bro",
    "init_config": "random",
}

input_config2D = {
    "runID": 0,
    "dim": 2,
    "N": 64,
    "box_size": [
        100.0,
        100.0
    ],
    "phi": 0.15,
    "n_steps": 100000,
    "n_save": 1000,
    "n_rec":1000,
    "save_mode": "concise",
    "optimization_method": "particlewise_bro",
    "init_config": "random",
}

input_config_corr2D = {
    "runID": 0,
    "dim": 2,
    "N": 64,
    "r": 1.0,
    "box_size": [
        100.0,
        100.0
    ],
    "phi": 0.15,
    "n_steps": 100000,
    "n_save": 1000,
    "n_rec":1000,
    "corr": 0.0,
    "save_mode": "concise",
    "optimization_method": "particlewise_bro",
    "init_config": "random",
}

input_config_corr3D = {
    "runID": 0,
    "dim": 3,
    "N": 64,
    "box_size": [
        100.0,
        100.0,
        100.0
    ],
    "phi": 0.15,
    "n_steps": 100000,
    "n_save": 1000,
    "n_rec":1000,
    "corr": 0.0,
    "save_mode": "concise",
    "optimization_method": "particlewise_bro",
    "init_config": "random",
}

input_config_RO_corr2D = {
    "runID": 0,
    "dim": 2,
    "N": 64,
    "box_size": [
        100.0,
        100.0
    ],
    "phi": 0.15,
    "n_steps": 100000,
    "n_save": 1000,
    "n_rec":1000,
    "corr_mag": 0.0,
    "corr_dir": 0.0,
    "save_mode": "concise",
    "optimization_method": "particlewise_bro",
    "init_config": "random",
}

input_config_RO_corr3D = {
    "runID": 0,
    "dim": 3,
    "N": 64,
    "box_size": [
        100.0,
        100.0,
        100.0
    ],
    "phi": 0.15,
    "n_steps": 100000,
    "n_save": 1000,
    "n_rec":1000,
    "corr_mag": 0.0,
    "corr_dir": 0.0,
    "save_mode": "concise",
    "optimization_method": "particlewise_bro",
    "init_config": "random",
}

input_config_corr_undir_2D = {
    "runID": 0,
    "dim": 2,
    "N": 64,
    "box_size": [
        100.0,
        100.0
    ],
    "phi": 0.15,
    "n_steps": 100000,
    "n_save": 1000,
    "n_rec":1000,
    "a0":1.0,
    "mpow":1.0,
    "corr": 0.0,
    "save_mode": "concise",
    "optimization_method": "particlewise_bro",
    "init_config": "random",
}

comp_config = {
    "NTHREADS":"1",
    "PYTHON_EXE":"run.py",
    # "MEMORY":"12GB",
    "MEMORY":"4GB",
    "TIME":"00:05:00",
    "CONDA_ENV":"hyperalg"
}


# # calculating S(k) and density variance
# root_dir = '/scratch/sa7483/sig-python/sips/scripts/testing/correlated_pairwise_ro_scan_dim'
# comp_config["PYTHON_EXE"] = "/home/sa7483/sips_project/sips-main/scripts/plot-scripts/structure_factor_and_density_variance.py"
# comp_config["TIME"] = "00:05:00"
# comp_config["NTHREADS"] = "1"

# with open('./submit_batch.INI', 'r') as file:
#     fstring = file.read()
# for key in comp_config:
#     fstring = fstring.replace(key, comp_config[key])
# with open('./submit_batch.sbatch', 'w') as file:
#     file.write(fstring)
# subprocess.run(["sbatch", "submit_batch.sbatch"])
# time.sleep(0.05)
# subprocess.run(["rm", "submit_batch.sbatch"])


root_dir = '/scratch/sa7483/sig-python/sips/scripts/testing/thermal_noise'
subprocess.run(['mkdir',root_dir])

# # 1 particlewise BRO
# job_root_dir = os.path.join(root_dir,'particlewise_bro')
# input_config = copy.deepcopy(input_config3D)
# input_config["optimization_method"] = "particlewise_bro"
# input_config["eps"] = 0.001
# input_config["N"] = 40000
# input_config["n_steps"] = 180000
# input_config["n_save"] = 3600
# input_config["init_config"] = "random"
# input_config["zoom_start"] = 0
# input_config["zoom_end"] = 14400
# input_config["zoom_rate"] = 20
# scan_dict = {'phi':[0.63,0.65]}
# comp_config["TIME"] = "48:00:00"
# comp_config["NTHREADS"] = "1"
# jobs = BatchJobManager(input_config,comp_config,job_root_dir)
# jobs.scan_params(scan_dict,"phi",n_repeat= 10)

# # 2 reciprocal pairwise BRO
# job_root_dir = os.path.join(root_dir,'reciprocal_pairwise_bro')
# input_config = copy.deepcopy(input_config3D)
# input_config["optimization_method"] = "reciprocal_pairwise_bro"
# # input_config["eps"] = 0.01
# input_config["N"] = 100000
# input_config["n_steps"] = 200000
# input_config["n_save"] = 10000
# input_config["init_config"] = "random"
# input_config["zoom_start"] = 0
# input_config["zoom_end"] = 1
# input_config["zoom_rate"] = 1
# scan_dict = {'phi':[1.0, 1.5], 'eps':[0.5, 1.0]}
# comp_config["TIME"] = "04:00:00"
# comp_config["NTHREADS"] = "4"
# jobs = BatchJobManager(input_config,comp_config,job_root_dir)
# jobs.scan_params(scan_dict,"phi",n_repeat= 1)


# # 2 non-reciprocal pairwise BRO
# job_root_dir = os.path.join(root_dir,'nonreciprocal_pairwise_bro')
# input_config = copy.deepcopy(input_config3D)
# input_config["optimization_method"] = "nonreciprocal_pairwise_bro"
# # input_config["eps"] = 0.01
# input_config["N"] = 100000
# input_config["n_steps"] = 200000
# input_config["n_save"] = 10000
# input_config["init_config"] = "random"
# input_config["zoom_start"] = 0
# input_config["zoom_end"] = 1
# input_config["zoom_rate"] = 1
# scan_dict = {'phi':[2.0], 'eps':[1.0]}
# comp_config["TIME"] = "04:00:00"
# comp_config["NTHREADS"] = "4"
# jobs = BatchJobManager(input_config,comp_config,job_root_dir)
# jobs.scan_params(scan_dict,"phi",n_repeat= 1)

# # 3 particlewise stodyn match bro
# job_root_dir = os.path.join(root_dir,'inversepower_particlewise_stodyn_match_bro')
# input_config = copy.deepcopy(input_config3D)
# input_config["optimization_method"] = "inversepower_particlewise_stodyn_match_bro"
# input_config["eps"] = 0.01
# input_config["N"] = 40000
# input_config["n_steps"] = 180000
# input_config["n_save"] = 3600
# input_config["init_config"] = "random"
# input_config["zoom_start"] = 0
# input_config["zoom_end"] = 14400
# input_config["zoom_rate"] = 20
# scan_dict = {'phi':[0.63,0.65]}
# comp_config["TIME"] = "48:00:00"
# comp_config["NTHREADS"] = "1"
# jobs = BatchJobManager(input_config,comp_config,job_root_dir)
# jobs.scan_params(scan_dict,"phi",n_repeat= 10)

# # 4 reciprocal pairwise stodyn match bro
# job_root_dir = os.path.join(root_dir,'inversepower_reciprocal_pairwise_stodyn_match_bro')
# input_config = copy.deepcopy(input_config3D)
# input_config["optimization_method"] = "inversepower_reciprocal_pairwise_stodyn_match_bro"
# # input_config["eps"] = 0.1
# input_config["N"] = 100000
# input_config["n_steps"] = 500000
# input_config["n_save"] = 1000
# input_config["init_config"] = "random"
# input_config["zoom_start"] = 0
# input_config["zoom_end"] = 1
# input_config["zoom_rate"] = 1
# scan_dict = {'phi':[0.68, 0.7, 1.0], 'eps':[0.5]}
# comp_config["TIME"] = "05:00:00"
# comp_config["NTHREADS"] = "4"
# jobs = BatchJobManager(input_config,comp_config,job_root_dir)
# jobs.scan_params(scan_dict,"phi",n_repeat= 1)


# # reciprocal pairwise sgd
# job_root_dir = os.path.join(root_dir,'inversepower_probabilistic_pairwise_sgd')
# input_config = copy.deepcopy(input_config2D)
# input_config["optimization_method"] = "inversepower_probabilistic_pairwise_sgd"
# input_config["dim"] = 2        #3
# input_config["r"] = 1.0
# input_config["a0"] = 1.0
# input_config["mpow"] = 1.0     #1.25

# # input_config["lr"] = 0.5       #0.5

# input_config["prob"] = 0.5

# input_config["N"] = 100000                                 #100000                           #318309  
# input_config["box_size"] = [1000.0,1000.0]                 #[1000.0,1000.0,1000.0]
# input_config["n_steps"] = 200000                           #200000                           #30000
# input_config["n_save"] = 1000
# input_config["n_rec"] = 1000
# input_config["init_config"] = "random"
# input_config["zoom_start"] = 0
# input_config["zoom_end"] = 1
# input_config["zoom_rate"] = 1

# # scan_dict = {'phi':[1.0], 'prob':[0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]} 

# # scan_dict = {'phi':[1.0], 'mpow':[1.0, 1.25], 'prob':[0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]} 

# # scan_dict = {'phi':[1.0], 'lr':[0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 1.1, 1.2, 1.3, 1.4, 1.5]} 

# scan_dict = {'phi':[1.0], 'lr':[0.5, 0.6, 0.7, 0.8, 0.9, 1.0]} 

# # scan_dict = {'phi':[1.0], 'lr':[0.25], 'prob':[0.1, 0.3, 0.7, 0.9, 1.0]} 

# comp_config["TIME"] = "47:00:00"
# # comp_config["TIME"] = "15:00:00"
# comp_config["NTHREADS"] = "4"
# jobs = BatchJobManager(input_config,comp_config,job_root_dir)
# jobs.scan_params(scan_dict,"phi",n_repeat= 1)



# #  correlated pairwise sgd
# job_root_dir = os.path.join(root_dir,'inversepower_correlated_probabilistic_pairwise_sgd')
# input_config = copy.deepcopy(input_config2D)
# input_config["optimization_method"] = "inversepower_correlated_probabilistic_pairwise_sgd"
# input_config["dim"] = 2
# input_config["r"] = 1.0
# input_config["a0"] = 1.0
# input_config["mpow"] = 1.0                                    
# input_config["lr"] = 0.5                                      
# input_config["prob"] = 0.5                                      
# input_config["N"] = 318309                                           #2546472
# input_config["box_size"] = [1000.0,1000.0]                    # [1000.0,1000.0,1000.0]
# input_config["n_steps"] = 200000
# input_config["n_save"] = 100
# input_config["n_rec"] = 200000                 
# input_config["init_config"] = "random"
# input_config["zoom_start"] = 0
# input_config["zoom_end"] = 1
# input_config["zoom_rate"] = 1

# # scan_dict = {'phi':[1.0], 'prob':[0.5], 'corr':[0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]}
# # scan_dict = {'phi':[1.0], 'corr':[0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]}

# scan_dict = {'phi':[0.5535], 'corr':[0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]}

# comp_config["TIME"] = "02:00:00"       #use when n_rec = 200000
# # comp_config["TIME"] = "47:00:00"         #use when n_rec = 1000
# # comp_config["TIME"] = "120:00:00"     
# comp_config["NTHREADS"] = "4"
# jobs = BatchJobManager(input_config,comp_config,job_root_dir)
# # jobs.scan_params(scan_dict,"phi",n_repeat= 1)
# jobs.scan_params(scan_dict,"phi",n_repeat= 10)



# # 4 particlewise sgd
# job_root_dir = os.path.join(root_dir,'inversepower_probabilistic_particlewise_sgd')
# input_config = copy.deepcopy(input_config2D)
# input_config["optimization_method"] = "inversepower_probabilistic_particlewise_sgd"
# input_config["dim"] = 2
# input_config["r"] = 1.0
# input_config["a0"] = 1.0
# input_config["mpow"] = 1.0
# input_config["lr"] = 1.0
# input_config["N"] = 100000
# input_config["box_size"] = [500.0,500.0]
# input_config["n_steps"] = 200000
# input_config["n_save"] = 1000
# input_config["init_config"] = "random"
# input_config["zoom_start"] = 0
# input_config["zoom_end"] = 1
# input_config["zoom_rate"] = 1
# scan_dict = {'phi':[1.0], 'prob':[0.95, 0.99]}
# comp_config["TIME"] = "04:00:00"
# comp_config["NTHREADS"] = "4"
# jobs = BatchJobManager(input_config,comp_config,job_root_dir)
# jobs.scan_params(scan_dict,"phi",n_repeat= 1)


# 4 correlated pairwise stodyn match sgd
job_root_dir = os.path.join(root_dir,'inversepower_correlated_pairwise_stodyn_match_sgd')
input_config = copy.deepcopy(input_config2D)
input_config["optimization_method"] = "inversepower_correlated_pairwise_stodyn_match_sgd"
input_config["dim"] = 2
input_config["r"] = 1.0
input_config["a0"] = 1.0
input_config["mpow"] = 1.0
input_config["lr"] = 0.5
input_config["box_size"] = [1000.0,1000.0]
input_config["N"] = 318309
input_config["n_steps"] = 200000
input_config["n_save"] = 1000
input_config["n_rec"] = 200000
input_config["init_config"] = "random"
input_config["zoom_start"] = 0
input_config["zoom_end"] = 1
input_config["zoom_rate"] = 1
# scan_dict = {'phi':[1.0], 'prob':[0.5], 'corr':[0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]}
scan_dict = {'phi':[1.0], 'prob':[0.5], 'Dtherm':[0.001, 0.005], 'corr':[0.0, 0.9, 1.0]}
comp_config["TIME"] = "20:00:00"
comp_config["NTHREADS"] = "4"
jobs = BatchJobManager(input_config,comp_config,job_root_dir)
jobs.scan_params(scan_dict,"phi",n_repeat= 1)


# # 5 particlewise sgd match bro
# job_root_dir = os.path.join(root_dir,'prob_particlewise_sgd match_bro')
# input_config = copy.deepcopy(input_config3D)
# input_config["optimization_method"] = "inversepower_probabilistic_particlewise_sgd_match_bro"
# input_config["eps"] = 0.01
# input_config["N"] = 40000
# input_config["n_steps"] = 180000
# input_config["n_save"] = 3600
# input_config["init_config"] = "random"
# input_config["zoom_start"] = 0
# input_config["zoom_end"] = 14400
# input_config["zoom_rate"] = 20
# scan_dict = {'phi':[0.63,0.65]}
# comp_config["TIME"] = "48:00:00"
# comp_config["NTHREADS"] = "1"
# jobs = BatchJobManager(input_config,comp_config,job_root_dir)
# jobs.scan_params(scan_dict,"phi",n_repeat= 10)


# # 6 pairwise sgd match bro
# job_root_dir = os.path.join(root_dir,'prob_pairwise_sgd_match_bro')
# input_config = copy.deepcopy(input_config2D)
# input_config["optimization_method"] = "inversepower_probabilistic_pairwise_sgd_match_bro"
# # input_config["eps"] = 0.01
# input_config["N"] = 100000
# input_config["n_steps"] = 200000
# input_config["n_save"] = 1000
# input_config["init_config"] = "random"
# input_config["zoom_start"] = 0
# input_config["zoom_end"] = 1
# input_config["zoom_rate"] = 1
# scan_dict = {'phi':[1.0], 'eps':[0.5, 1.0]}
# comp_config["TIME"] = "05:00:00"
# comp_config["NTHREADS"] = "4"
# jobs = BatchJobManager(input_config,comp_config,job_root_dir)
# jobs.scan_params(scan_dict,"phi",n_repeat= 1)



# 4 correlated pairwise stodyn match bro
job_root_dir = os.path.join(root_dir,'inversepower_correlated_pairwise_stodyn_match_bro')
input_config = copy.deepcopy(input_config_corr2D)
input_config["optimization_method"] = "inversepower_correlated_pairwise_stodyn_match_bro"
input_config["dim"] = 2
input_config["r"] = 1.0
input_config["N"] = 318309          
input_config["box_size"] = [1000.0,1000.0]
input_config["n_steps"] = 200000             
input_config["n_save"] = 1000
input_config["n_rec"] = 200000
input_config["init_config"] = "random"
input_config["zoom_start"] = 0
input_config["zoom_end"] = 0
input_config["zoom_rate"] = 0
scan_dict = {'phi':[1.0], 'eps':[1.0], 'Dtherm':[0.001, 0.005], 'corr':[0.0, 0.9, 1.0]}
comp_config["TIME"] = "20:00:00"
comp_config["NTHREADS"] = "4"
jobs = BatchJobManager(input_config,comp_config,job_root_dir)
jobs.scan_params(scan_dict,"phi",n_repeat= 1)


# # 4 correlated pairwise bro
# job_root_dir = os.path.join(root_dir,'correlated_pairwise_bro')
# input_config = copy.deepcopy(input_config_corr2D)
# input_config["optimization_method"] = "correlated_pairwise_bro"
# input_config["dim"] = 2
# input_config["r"] = 1.0
# input_config["N"] = 318309                                   #100000
# input_config["box_size"] = [1000.0,1000.0]
# input_config["n_steps"] = 200000
# input_config["n_save"] = 100              #1000
# input_config["init_config"] = "random"
# input_config["zoom_start"] = 0
# input_config["zoom_end"] = 0
# input_config["zoom_rate"] = 0
# # scan_dict = {'phi':[1.0], 'eps':[1.0], 'corr':[0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]}
# # scan_dict = {'phi':[1.0], 'eps':[0.5], 'corr':[0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]}
# scan_dict = {'phi':[0.3555], 'eps':[1.0], 'corr':[0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]}
# # comp_config["TIME"] = "08:00:00"
# comp_config["TIME"] = "02:00:00"
# comp_config["NTHREADS"] = "4"
# jobs = BatchJobManager(input_config,comp_config,job_root_dir)
# # jobs.scan_params(scan_dict,"phi",n_repeat= 1)
# jobs.scan_params(scan_dict,"phi",n_repeat= 10)


# # 4 correlated pairwise ro
# job_root_dir = os.path.join(root_dir,'correlated_pairwise_ro')
# input_config = copy.deepcopy(input_config_RO_corr2D)
# input_config["optimization_method"] = "correlated_pairwise_ro"
# input_config["dim"] = 2     
# input_config["r"] = 1.0
# input_config["N"] = 318309                #318309                    #100000
# input_config["box_size"] = [1000.0,1000.0]       
# input_config["n_steps"] = 200000
# input_config["n_save"] = 100        #1000
# input_config["init_config"] = "random"
# input_config["zoom_start"] = 0
# input_config["zoom_end"] = 0
# input_config["zoom_rate"] = 0
# input_config["corr_dir"] = 1.0
# # scan_dict = {'phi':[1.0], 'eps':[1.0], 'corr_mag':[0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]}
# # scan_dict = {'phi':[1.0], 'eps':[0.5], 'corr_mag':[0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]}
# scan_dict = {'phi':[0.3375], 'eps':[1.0], 'corr_mag':[0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]}
# # comp_config["TIME"] = "15:00:00"
# comp_config["TIME"] = "02:00:00"
# comp_config["NTHREADS"] = "4"
# jobs = BatchJobManager(input_config,comp_config,job_root_dir)
# # jobs.scan_params(scan_dict,"phi",n_repeat= 1)
# jobs.scan_params(scan_dict,"phi",n_repeat= 10)



# 4 correlated pairwise undirected stodyn match ro
job_root_dir = os.path.join(root_dir,'inversepower_correlated_pairwise_undirected_stodyn_match_ro')
input_config = copy.deepcopy(input_config_corr_undir_2D)
input_config["optimization_method"] = "inversepower_correlated_pairwise_undirected_stodyn_match_ro"
input_config["r"] = 1.0
input_config["box_size"] = [1000.0,1000.0]
input_config["N"] = 318309
input_config["n_steps"] = 200000
input_config["n_save"] = 1000
input_config["n_rec"] = 200000
input_config["init_config"] = "random"
input_config["zoom_start"] = 0
input_config["zoom_end"] = 0
input_config["zoom_rate"] = 0
# scan_dict = {'phi':[1.0], 'eps':[1.0], 'corr':[0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]}
scan_dict = {'phi':[1.0], 'eps':[1.0], 'Dtherm':[0.001, 0.005], 'corr':[0.0, 0.9, 1.0]}
comp_config["TIME"] = "20:00:00"
comp_config["NTHREADS"] = "4"
jobs = BatchJobManager(input_config,comp_config,job_root_dir)
jobs.scan_params(scan_dict,"phi",n_repeat= 1)



# # 4 correlated pairwise stodyn (not matching RO, BRO, or SGD)
# job_root_dir = os.path.join(root_dir,'inversepower_correlated_pairwise_stodyn')
# input_config = copy.deepcopy(input_config_corr2D)
# input_config["optimization_method"] = "inversepower_correlated_pairwise_stodyn"
# input_config["dim"] = 2
# input_config["r"] = 1.0
# input_config["N"] = 100000

# input_config["alpha"] = 1.0
# input_config["D0"] = 4.0/3.0
# input_config["a0"] = 1.0
# input_config["mpow"] = 2.0

# input_config["box_size"] = [500.0,500.0]
# input_config["n_steps"] = 200000
# input_config["n_save"] = 1000
# input_config["init_config"] = "random"
# input_config["zoom_start"] = 0
# input_config["zoom_end"] = 0
# input_config["zoom_rate"] = 0
# scan_dict = {'phi':[1.0], 'eps':[1.0], 'corr':[0.0, 0.9, 1.0]}
# comp_config["TIME"] = "05:00:00"
# comp_config["NTHREADS"] = "4"
# jobs = BatchJobManager(input_config,comp_config,job_root_dir)
# jobs.scan_params(scan_dict,"phi",n_repeat= 1)


# # 4 reciprocal pairwise sgd quad potential
# job_root_dir = os.path.join(root_dir,'inversepower_probabilistic_pairwise_sgd')
# input_config = copy.deepcopy(input_config2D)
# input_config["optimization_method"] = "inversepower_probabilistic_pairwise_sgd"
# input_config["dim"] = 2
# input_config["r"] = 1.0
# input_config["a0"] = 1.0
# # input_config["mpow"] = 2.0
# input_config["lr"] = 1.0
# input_config["N"] = 100000
# input_config["box_size"] = [500.0,500.0]
# input_config["n_steps"] = 200000
# input_config["n_save"] = 1000
# input_config["init_config"] = "random"
# input_config["zoom_start"] = 0
# input_config["zoom_end"] = 1
# input_config["zoom_rate"] = 1
# scan_dict = {'phi':[1.0, 1.25, 1.5, 2.0], 'prob':[0.5], 'mpow':[1.0, 1.5, 2.0, 2.5]}
# comp_config["TIME"] = "05:00:00"
# comp_config["NTHREADS"] = "4"
# jobs = BatchJobManager(input_config,comp_config,job_root_dir)
# jobs.scan_params(scan_dict,"phi",n_repeat= 1)



# # #test for small systems

# # 4 correlated pairwise bro
# job_root_dir = os.path.join(root_dir,'correlated_pairwise_bro')
# input_config = copy.deepcopy(input_config_corr2D)
# input_config["optimization_method"] = "correlated_pairwise_bro"
# input_config["dim"] = 2
# input_config["r"] = 1.0
# input_config["N"] = 1000
# input_config["box_size"] = [50.0,50.0]
# input_config["n_steps"] = 2000
# input_config["n_save"] = 1
# input_config["init_config"] = "random"
# input_config["zoom_start"] = 0
# input_config["zoom_end"] = 0
# input_config["zoom_rate"] = 0
# scan_dict = {'phi':[0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0], 'eps':[1.0], 'corr':[0.0]}
# comp_config["TIME"] = "00:30:00"
# comp_config["NTHREADS"] = "4"
# jobs = BatchJobManager(input_config,comp_config,job_root_dir)
# jobs.scan_params(scan_dict,"phi",n_repeat= 1)



# # finding critical volume fraction RO
# # correlated pairwise ro
# job_root_dir = os.path.join(root_dir,'correlated_pairwise_ro')
# input_config = copy.deepcopy(input_config_RO_corr2D)
# input_config["optimization_method"] = "correlated_pairwise_ro"
# input_config["dim"] = 2
# input_config["r"] = 1.0
# input_config["N"] = 318309
# input_config["box_size"] = [1000.0,1000.0]
# input_config["n_save"] = 1000
# input_config["init_config"] = "random"
# input_config["zoom_start"] = 0
# input_config["zoom_end"] = 0
# input_config["zoom_rate"] = 0
# input_config["corr_dir"] = 1.0

# input_config["n_steps"] = 200000
# # scan_dict = {'phi':[0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0], 'eps':[1.0], 'corr_mag':[0.0]}
# scan_dict = {'phi':[0.31, 0.32, 0.33, 0.34, 0.35, 0.36, 0.37, 0.38, 0.39, 0.41, 0.42, 0.43, 0.44, 0.45, 0.46, 0.47, 0.48, 0.49], 'eps':[1.0], 'corr_mag':[0.0]}
# comp_config["TIME"] = "12:00:00"

# comp_config["NTHREADS"] = "4"
# jobs = BatchJobManager(input_config,comp_config,job_root_dir)
# jobs.scan_params(scan_dict,"phi",n_repeat= 1)



# # finding critical volume fraction BRO
# # correlated pairwise bro
# job_root_dir = os.path.join(root_dir,'correlated_pairwise_bro')
# input_config = copy.deepcopy(input_config_corr2D)
# input_config["optimization_method"] = "correlated_pairwise_bro"
# input_config["dim"] = 2
# input_config["r"] = 1.0
# input_config["N"] = 318309
# input_config["box_size"] = [1000.0,1000.0]
# input_config["n_save"] = 1000
# input_config["init_config"] = "random"
# input_config["zoom_start"] = 0
# input_config["zoom_end"] = 0
# input_config["zoom_rate"] = 0

# input_config["n_steps"] = 200000
# # scan_dict = {'phi':[0.3, 0.31, 0.32, 0.33, 0.34, 0.35, 0.36, 0.37, 0.38, 0.39, 0.40, 0.41, 0.42, 0.43, 0.44, 0.45, 0.46, 0.47, 0.48, 0.49, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0], 'eps':[1.0], 'corr':[0.0]}
# scan_dict = {'phi':[0.1, 0.2, 0.3, 0.31, 0.32, 0.33, 0.34, 0.35, 0.36, 0.37, 0.38, 0.39, 0.4, 0.41, 0.42, 0.43, 0.44, 0.45, 0.46, 0.47, 0.48, 0.49, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0], 'eps':[1.0], 'corr':[0.0]}
# # comp_config["TIME"] = "02:30:00"
# comp_config["TIME"] = "08:00:00"

# comp_config["NTHREADS"] = "4"
# jobs = BatchJobManager(input_config,comp_config,job_root_dir)
# jobs.scan_params(scan_dict,"phi",n_repeat= 1)



# # finding critical volume fraction SGD
# # reciprocal correlated pairwise sgd
# job_root_dir = os.path.join(root_dir,'inversepower_correlated_probabilistic_pairwise_sgd')
# input_config = copy.deepcopy(input_config2D)
# input_config["optimization_method"] = "inversepower_correlated_probabilistic_pairwise_sgd"
# input_config["dim"] = 2
# input_config["r"] = 1.0
# input_config["a0"] = 1.0
# input_config["mpow"] = 1.0
# input_config["lr"] = 0.5
# input_config["N"] = 318309
# input_config["box_size"] = [1000.0,1000.0]
# input_config["n_save"] = 1000
# input_config["n_rec"] = 1000
# input_config["init_config"] = "random"
# input_config["zoom_start"] = 0
# input_config["zoom_end"] = 1
# input_config["zoom_rate"] = 1

# input_config["n_steps"] = 200000

# # scan_dict = {'phi':[0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.61, 0.62, 0.63, 0.64, 0.65, 0.66, 0.67, 0.68, 0.69, 0.70, 0.71, 0.72, 0.73, 0.74, 0.75, 0.76, 0.77, 0.78, 0.79, 0.8, 0.9, 1.0, 1.1, 1.2, 1.3, 1.4, 1.5, 1.6], 'prob':[0.5], 'corr':[0.0]}

# # comp_config["TIME"] = "15:00:00"
# comp_config["TIME"] = "47:00:00"     #use when n_rec = 1000

# comp_config["NTHREADS"] = "4"
# jobs = BatchJobManager(input_config,comp_config,job_root_dir)
# jobs.scan_params(scan_dict,"phi",n_repeat= 1)



# #############################################################################################################################################
