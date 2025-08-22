# import zipfile
# import os
# import json
# import re
# import numpy as np
# import subprocess
# import matplotlib.pyplot as plt
# import matplotlib.patches as mpatches
# from matplotlib import animation 
# import finufft
# from itertools import product
# from numba import jit
# from sklearn.neighbors import NearestNeighbors

# class DataLoader():
#     def __init__(self,dir_name,mode = "all",rng=[]):
#         self.info_file = os.path.join(dir_name,"info.json")
#         self.dir_name = dir_name
#         self.data = []
#         if not os.path.isfile(self.info_file):
#             raise Exception("Info file does not exist!")

#         with open(self.info_file,'r') as fs:
#             self.info = json.loads(fs.read())
        
#         if not "r" in self.info:
#             v = np.prod(self.info["box_size"])
#             if self.info['dim'] == 3:   
#                 self.info["r"] = np.power(3.0/4.0*v*self.info['phi']/np.pi/self.info['N'],1.0/3.0)
#             elif self.info['dim'] == 2:
#                 self.info["r"] = np.power(v*self.info['phi']/self.info['N']/np.pi,1.0/2.0)
        
#         if mode == "all":
#             for file in os.listdir(self.dir_name):
#                 #iter = re.search(".+iter(\w+)[.].+",file)
#                 if file.endswith(".zip") and file.find('iter')>0:
#                     #print(file)
#                     self.data.append(self.read_zipped_json(os.path.join(dir_name,file)))
#             self.data = sorted(self.data,key=lambda data:data['iter'])
#         elif mode =="last":
#             for file in os.listdir(self.dir_name):
#                 if file.endswith(str(self.info['last_iteration'])+".zip"):
#                     self.data.append(self.read_zipped_json(os.path.join(dir_name,file)))
#                 elif file.endswith(str(self.info['last_iteration']-1)+".zip"):
#                     self.data.append(self.read_zipped_json(os.path.join(dir_name,file)))
#         elif mode =="last_existed":
#             iters = []
#             for file in os.listdir(self.dir_name):
#                 if file.endswith(".zip") and file.find('iter')>0:
#                     m = re.search(".+iter(\w+)[.].+",file).group(1)
#                     _iter = int(re.search(".+iter(\w+)[.].+",file).group(1))
#                     iters.append(_iter)
#             max_iter = max(iters)
#             for file in os.listdir(self.dir_name):
#                 if file.endswith(str(max_iter)+".zip"):
#                     self.data.append(self.read_zipped_json(os.path.join(dir_name,file)))
#                 elif file.endswith(str(max_iter)+".zip"):
#                     self.data.append(self.read_zipped_json(os.path.join(dir_name,file)))  
#         elif mode == "range":
#             iters = []
#             for file in os.listdir(self.dir_name):
#                 if file.endswith(".zip") and file.find('iter')>0:
#                     m = re.search(".+iter(\w+)[.].+",file).group(1)
#                     iter = int(re.search(".+iter(\w+)[.].+",file).group(1))
#                     if iter>= rng[0] and iter <= rng[1]:
#                         self.data.append(self.read_zipped_json(os.path.join(dir_name,file)))
#             self.data = sorted(self.data,key=lambda data:data['iter'])
                        
#     def read_data_by_index(self,idx):
#          file_name = "data_iter{}.json".format(idx*self.info['n_iter'])
#          return self.read_zipped_json(os.path.join(self.dir_name,file_name))

#     def read_zipped_json(self,file_name):
#         with zipfile.ZipFile(file_name) as myzip:
#             with myzip.open('data.json') as myfile:
#                 return json.loads(myfile.read())
    
#     def get_data_list(self,key):
#         iter = [d['iter'] for d in self.data]
#         value = [d[key] for d in self.data]
#         return iter,value

#     def get_last_iter_data(self,key):
#         return self.data[-1]['iter'],self.data[-1][key]
    
#     def get_pair_corr(self,x,dr = None):
#         x = np.reshape(x,(-1,self.info["dim"]))
#         if not dr :
#             dr = self.info["r"]*0.02
#         r_max = self.info["r"]*30
#         V = 1.0
#         for d in self.info["box_size"]:
#             V *= d
#         n_slots = int(r_max/dr + 0.5)
#         r = np.linspace(0.5*dr,r_max,n_slots)
#         count = np.zeros(r.shape)
#         for i in range(self.info["N"]):
#             dx = x[i,:] - x
#             # periodic boundary condition
#             for d in range(self.info["dim"]):
#                 dx[np.where(dx[:,d] < -0.5*self.info["box_size"][d]),d] += self.info["box_size"][d]
#                 dx[np.where(dx[:,d] > 0.5*self.info["box_size"][d]),d] -= self.info["box_size"][d]
#             dx_norm = np.sqrt(np.sum(dx*dx,axis=1))
#             indices = (dx_norm/dr).astype(int)
#             indices = indices[np.where(indices<n_slots)]
#             count[indices] += 1.0/self.info["N"]
#         count[0] = 0.0 # the zero distance should be excluded
#         # normalize the count
#         if self.info["dim"] == 3:
#             count /= (4.0*np.pi*r*r*dr + 1e-8)
#         elif self.info["dim"] == 2:
#             count /= (2.0*np.pi*r*dr + 1e-8)
#         density = self.info["N"]/V
#         return (r/(self.info["r"]*2.0),count/density)
    
#     def get_pair_corr_along_axes2d(self,x,inbr,dr = None,dt = 0.1):
#         x = np.reshape(x,(-1,self.info["dim"]))
#         if not dr :
#             dr = self.info["r"]*0.01
#         r_max = self.info["r"]*30
#         n_slots = int(r_max/dr + 0.5)
#         r = np.linspace(0.5*dr,r_max,n_slots)
#         para_count = np.zeros(r.shape)
#         perp_count = np.zeros(r.shape)
#         n_pairs = 0.0
#         for i in range(self.info["N"]):
#             if len(inbr[i]) > 0:
#                 dx = x - x[i,:]
#                 # periodic boundary condition
#                 for d in range(self.info["dim"]):
#                     dx[np.where(dx[:,d] < -0.5*self.info["box_size"][d]),d] += self.info["box_size"][d]
#                     dx[np.where(dx[:,d] > 0.5*self.info["box_size"][d]),d] -= self.info["box_size"][d]
#                 dx_norm = np.sqrt(np.sum(dx*dx,axis=1))
#                 for j in inbr[i]:
#                     n_pairs += 1.0
#                     theta_parallel = np.arctan2(dx[j,1],dx[j,0])
#                     dtheta = np.arctan2(dx[:,1],dx[:,0]) - theta_parallel
#                     dtheta[np.where(dtheta > np.pi)] -= 2.0*np.pi
#                     dtheta[np.where(dtheta < -np.pi)] += 2.0*np.pi
#                     select_para = np.where( np.logical_or(np.abs(dtheta) < 0.5*dt, np.abs(np.abs(dtheta) - np.pi) < 0.5*dt))
#                     select_perp = np.where( np.logical_or(np.abs(dtheta - 0.5*np.pi) < 0.5*dt, np.abs(dtheta + 0.5*np.pi) < 0.5*dt))

#                     indices_para = (dx_norm[select_para]/dr).astype(int)
#                     indices_para = indices_para[np.where(indices_para<n_slots)]
#                     indices_perp= (dx_norm[select_perp]/dr).astype(int)
#                     indices_perp = indices_perp[np.where(indices_perp<n_slots)]
                    
#                     para_count[indices_para] += 1.0
#                     perp_count[indices_perp] += 1.0

#         para_count[0] = 0.0 # the zero distance should be excluded
#         perp_count[0] = 0.0
#         para_count /= (2.0*dt*r*dr + 1e-8)
#         perp_count /= (2.0*dt*r*dr + 1e-8)
#         V = 1.0
#         for d in self.info["box_size"]:
#             V *= d
#         density = self.info["N"]/V
#         print(n_pairs)
#         return (r/(self.info["r"]*2.0),para_count/(density*n_pairs),perp_count/(density*n_pairs))


    
#     def get_strucutre_factor(self,x,N = None):
#         # x: numpy array for particle positions
#         # N: resolvsion in k-space
#         #    the output is a N^dim array
#         x = np.reshape(x,(-1,self.info["dim"]))
#         if N is None:
#             N = 512
#         if self.info["dim"] == 2:
#             x1 = (x[:,0]/self.info["box_size"][0] - 0.5)*2.0*np.pi 
#             x2 = (x[:,1]/self.info["box_size"][1] - 0.5)*2.0*np.pi
#             c = np.ones(self.info["N"]).astype(np.complex128)
#             f = finufft.nufft2d1(x1, x2, c, (N, N),modeord=0)
#             s = np.abs(f*np.conjugate(f))/N
#             kx = np.fft.fftshift(np.fft.fftfreq(N)*N*2*np.pi/self.info["box_size"][0])
#             ky = np.fft.fftshift(np.fft.fftfreq(N)*N*2*np.pi/self.info["box_size"][1])
#             return (kx,ky,s)
#         elif self.info["dim"] == 3:
#             x1 = (x[:,0]/self.info["box_size"][0] - 0.5)*2.0*np.pi 
#             x2 = (x[:,1]/self.info["box_size"][1] - 0.5)*2.0*np.pi
#             x3 = (x[:,2]/self.info["box_size"][2] - 0.5)*2.0*np.pi
#             c = np.ones(self.info["N"]).astype(np.complex128)
#             f = finufft.nufft3d1(x1, x2, x3, c, (N, N, N),modeord=0)
#             s = np.abs(f*np.conjugate(f))/N
#             kx = np.fft.fftshift(np.fft.fftfreq(N)*N*2*np.pi/self.info["box_size"][0])
#             ky = np.fft.fftshift(np.fft.fftfreq(N)*N*2*np.pi/self.info["box_size"][1])
#             kz = np.fft.fftshift(np.fft.fftfreq(N)*N*2*np.pi/self.info["box_size"][2])
#             return (kx,ky,kz,s)
    
# def get_nearast_distances(self,x,box_size,n_nbrs=1,pbc = True):
#     '''
#     return the nearest neighbours and the distances
#             e.g.ind[i] the nearest neighbour of particle i
#                 dist[i] the distance between particle i and its nearest neighbour
#     '''
#     dim = len(box_size)
#     x = x.reshape((-1,dim))
#     N = x.shape[0]
#     if pbc:
#         augmented_x = [x]
#         for d in range(dim):
#             dx = [0.0 for d in range(dim)]
#             dx[d] = box_size[d]
#             augmented_x.append(x + dx)
#             augmented_x.append(x - dx)
#         x = np.concatenate(augmented_x)
#     nbrs = NearestNeighbors(n_neighbors=n_nbrs+1, algorithm='auto').fit(x)
#     distances, indices = nbrs.kneighbors(x)
#     dist,ind = distances[:,1],indices[:,1]
#     if pbc:
#         for _ in range(2*dim):
#             ind[np.where(ind>=N)] -= N
#     return ind,dist

# @jit(nopython=True)
# def get_radial_structure_factor2d(kx,ky,s,n_bins=100,cut_off = 0.7):
#     k_max = np.sqrt(kx[-1]**2 + ky[-1]**2)*cut_off 
#     s_k = np.zeros(n_bins)
#     dA = (kx[2]-kx[1])*(ky[2]-ky[1])
#     delta_k = k_max/n_bins
#     for i in range(s.shape[0]):
#         for j in range(s.shape[1]):
#             k = np.sqrt(kx[i]*kx[i] + ky[j]*ky[j])
#             bin_num = int(k/delta_k) 
#             if bin_num < n_bins:
#                 s_k[bin_num] += s[i,j]*dA
#     k = np.arange(n_bins)*delta_k + 0.5*delta_k
#     s_k /= 2.0*np.pi*k*delta_k
#     return k,s_k

# @jit(nopython=True)
# def get_radial_structure_factor3d(kx,ky,kz,s,n_bins=100,cut_off = 0.6):
#     k_max = np.sqrt(kx[-1]**2 + ky[-1]**2 + kz[-1]**2)*cut_off 
#     s_k = np.zeros(n_bins)
#     dV = (kx[2]-kx[1])*(ky[2]-ky[1])*(kz[2]-kz[1])
#     delta_k = k_max/n_bins
#     for i in range(s.shape[0]):
#         for j in range(s.shape[1]):
#             for l in range(s.shape[2]):
#                 k = np.sqrt(kx[i]*kx[i] + ky[j]*ky[j] + kz[l]*kz[l])
#                 bin_num = int(k/delta_k)
#                 if bin_num < n_bins:
#                     s_k[bin_num] += s[i,j,l]*dV
#     k = np.arange(n_bins)*delta_k + 0.5*delta_k
#     s_k /= 4.0*np.pi*k*k*delta_k
#     return k[1:],s_k[1:]/s_k[-1]


# #load data for multiple calculations in a floder
# class DataSetLoader():
#     def __init__(self,dir_name,mode = "last"):
#         self.dir_name = dir_name
#         self.loader_list = []
#         for dir in os.listdir(self.dir_name):
#             if os.path.isdir(os.path.join(dir_name,dir)) and dir.find('#')>-1:
#                 my_data = DataLoader(os.path.join(dir_name,dir),mode = mode)
#                 self.loader_list.append(my_data)
            
#     def get_data_list(self,iter_idx,key):
#         if key in self.loader_list[0].info:
#             data_list = [(d.info[key],d.data[iter_idx]) for d in self.loader_list]
#             data_list = sorted(data_list,key = lambda data:data[0])
#         else:
#             data_list = [(d.data[iter_idx][key],d.data[iter_idx]) for d in self.loader_list]
#             data_list = sorted(data_list,key = lambda data:data[0])
#         val = [d[0] for d in data_list]
#         data = [d[1] for d in data_list]
#         return val,data

# def save_data(loc,iter,x,e_tot,g_tot,inbr,n_active,clusters,e_batch,g_batch,batch_pairs,dx,rm=True,compress=True):
#     if not os.path.isdir(loc):
#         subprocess.run(["mkdir", loc])
#     file_name = "data_iter{}.json".format(iter)
#     f = os.path.join(loc,file_name)
#     if os.path.exists(f) and rm:
#         subprocess.run(["rm", "-r",f])
#     # wrap the data
#     data = dict()
#     data['iter'] = iter
#     data['x'] = x.tolist()
#     data['total_energy'] = e_tot
#     data['total_gradient'] = g_tot
#     data['neighbor'] = inbr
#     data['n_active'] = n_active
#     data['n_clusters'] = len(clusters)
#     cluster_sizes = [len(clusters[i]) for i in range(len(clusters))]
#     data['avg_cluster_size'] = float(np.mean(cluster_sizes))
#     data['batch_energy'] = e_batch
#     data['batch_gradient'] = g_batch
#     data['batch_pairs'] = batch_pairs
#     data['dx'] = dx.tolist()
#     fzip = f.replace(".json",".zip")
#     # save the data as json a file

#     if compress:
#         fzip = f.replace(".json",".zip")
#         with zipfile.ZipFile(fzip, 'w') as myzip:
#             myzip.writestr("data.json",json.dumps(data),compress_type=zipfile.ZIP_DEFLATED)
#     else:
#         with open(f,'w') as fs:
#             fs.write(json.dumps(data))


# class SpherePlot():
#     def __init__(self,loader_or_str):
#         if isinstance(loader_or_str,str):
#             self.loader = DataLoader(loader_or_str)
#         else:
#             self.loader = loader_or_str
#         if not 'radii' in self.loader.info:
#             self.loader.info['radii'] = np.full(self.loader.info['N'],self.loader.info['r']) 
#         self.npts = 10
#         self.active_color='r'
#         self.atom_color='b'
#         self.batch_color='b'
#         self.bond_color=None
    
#     def set_npts4spheres(self,npts):
#         self.npts = npts
    
#     def plot_circle(self,ax,r,x,c='b',alpha=0.9):
#         circle = mpatches.Circle(x, r, facecolor = c,edgecolor='k')
#         ax.add_patch(circle)

#     def plot_sphere(self,ax,r,xx,c = 'b',alpha = 0.9):
#         if not ax:
#             raise Exception("axis for plotting is None")
#         p = np.linspace(0.0,2*np.pi,2*self.npts)
#         t = np.linspace(0.0,np.pi,self.npts)
#         theta, phi = np.meshgrid(t,p)
#         x = r*np.sin(theta)*np.cos(phi) + xx[0]
#         y = r*np.sin(theta)*np.sin(phi) + xx[1]
#         z = r*np.cos(theta) + xx[2]
#         ax.plot_surface(x, y, z,color = c,alpha = alpha)
    
#     def plot_line(self,ax,p1,p2,c = 'r',alpha = 0.5):
#         if self.loader.info['dim'] ==2:
#             ax.plot([p1[0],p2[0]],[p1[1],p2[1]],color= c)
#         elif self.loader.info['dim'] ==3:
#             ax.plot3D([p1[0],p2[0]],[p1[1],p2[1]],[p1[2],p2[2]],color= c)
        
#     def plot_arrow(self,ax,x,dx,c = 'k',alpha = 0.5):
#         if self.loader.info['dim'] ==2:
#             ax.arrow(x[0],x[1],dx[0],dx[1],color= c,alpha=alpha)
#         elif self.loader.info['dim'] ==3:
#             ax.plot3D([p1[0],p2[0]],[p1[1],p2[1]],[p1[2],p2[2]],color= c)
        
#     def plot_atom(self,ax,r,x,c = 'b',alpha = 0.9):
#         if self.loader.info['dim'] == 2:
#             self.plot_circle(ax,r,x,c = c,alpha = alpha)
#         elif self.loader.info['dim'] == 3:
#             self.plot_sphere(ax,r,x,c = c,alpha = alpha)
 
#     def plot_atoms(self,ax,data,alpha=0.5,pbc = True):
#         x = np.array(data['x'])
#         box = self.loader.info['box_size']
#         if pbc:
#             rx = x.reshape(self.loader.info['N'],self.loader.info['dim'])
#             for d in range(self.loader.info['dim']):
#                 rx[np.where(rx[:,d] > box[d]),d] -= box[d] 
#                 rx[np.where(rx[:,d] < 0.0),d]+= box[d]
#         radii = self.loader.info['radii']
#         for i in range(self.loader.info['N']):
#             x_i = x[i*self.loader.info['dim']:(i+1)*self.loader.info['dim']]
#             if len(data['neighbor'][i]) > 0:
#                 self.plot_atom(ax,radii[i],x_i,c = self.active_color,alpha=alpha)
#             else:
#                 self.plot_atom(ax,radii[i],x_i,c = self.atom_color,alpha=alpha)
#     def plot_dx(self,ax,data,alpha=0.5,scale = 1.0,pbc=True):
#         x = np.array(data['x'])
#         dx = np.array(data['dx'])*scale
#         box = self.loader.info['box_size']
#         if pbc:
#             rx = x.reshape(self.loader.info['N'],self.loader.info['dim'])
#             for d in range(self.loader.info['dim']):
#                 rx[np.where(rx[:,d] > box[d]),d] -= box[d] 
#                 rx[np.where(rx[:,d] < 0.0),d]+= box[d]
#         if self.loader.info['dim'] ==2:
#             for i in range(self.loader.info['N']):
#                 x_i = x[i*self.loader.info['dim']:(i+1)*self.loader.info['dim']]
#                 dx_i = dx[i*self.loader.info['dim']:(i+1)*self.loader.info['dim']]
#                 ax.arrow(x_i[0],x_i[1],dx_i[0],dx_i[1],color= 'k',alpha=alpha)
#         elif self.loader.info['dim'] ==3:
#             rx = x.reshape(self.loader.info['N'],self.loader.info['dim'])
#             rdx = dx.reshape(self.loader.info['N'],self.loader.info['dim'])
#             ax.quiver(x[:,0],x[:,1],x[:,2],rdx[:,0],rdx[:,1],rdx[:,2])
     
#     def plot_gradient(self,ax,data,alpha=0.5,scale = 1.0,pbc=True):
#         x = np.array(data['x'])
#         dx = np.array(data['total_gradient'])*scale
#         box = self.loader.info['box_size']
#         if pbc:
#             rx = x.reshape(self.loader.info['N'],self.loader.info['dim'])
#             for d in range(self.loader.info['dim']):
#                 rx[np.where(rx[:,d] > box[d]),d] -= box[d] 
#                 rx[np.where(rx[:,d] < 0.0),d]+= box[d]
#         if self.loader.info['dim'] ==2:
#             for i in range(self.loader.info['N']):
#                 x_i = x[i*self.loader.info['dim']:(i+1)*self.loader.info['dim']]
#                 dx_i = dx[i*self.loader.info['dim']:(i+1)*self.loader.info['dim']]
#                 ax.arrow(x_i[0],x_i[1],dx_i[0],dx_i[1],color= 'k',alpha=alpha)
#         elif self.loader.info['dim'] ==3:
#             rx = x.reshape(self.loader.info['N'],self.loader.info['dim'])
#             rdx = dx.reshape(self.loader.info['N'],self.loader.info['dim'])
#             ax.quiver(x[:,0],x[:,1],x[:,2],rdx[:,0],rdx[:,1],rdx[:,2])
#     def plot_batch(self,ax,data):
#         x = data['x']
#         radii = self.loader.info['radii']
#         for pair in data['batch_pairs']:
#             i,j =pair[0],pair[1]
#             x_i = x[i*self.loader.info['dim']:(i+1)*self.loader.info['dim']]
#             x_j = x[j*self.loader.info['dim']:(j+1)*self.loader.info['dim']]
#             self.plot_atom(ax,radii[i],x_i,c=self.batch_color)
#             self.plot_atom(ax,radii[j],x_j,c=self.batch_color)
#             if self.bond_color:
#                 self.plot_line(ax,x_i,x_j,c=self.bond_color)
    
#     def plot_box(self,ax,c='b'):
#         box = self.loader.info['box_size']
#         if self.loader.info['dim'] ==3: 
#             vertices = [[0,0,0],[0,0,box[2]],[0,box[1],0],[0,box[1],box[2]],[box[0],0,0],[box[0],0,box[2]],[box[0],box[1],0],box]
#         elif self.loader.info['dim'] ==2:
#             vertices = [[0,0],[0,box[1]],[box[0],0],box]
#         for i in range(len(vertices)):
#             for j in range(i+1,len(vertices)):
#                 if np.abs(L - np.linalg.norm(vertices[i],vertices[j])) < 1e-8:
#                     self.plot_line(vertices[i],vertices[j])

         
#     def animate(self,fig,fn,rng,inter,show=True,output_file=None):
#         # the local animation function
#         def animate_fn(iter):
#             # we want a fresh figure everytime
#             fig.clf()
#             # call the global function
#             fn(iter)

#         ani = animation.FuncAnimation(fig, animate_fn,frames=np.arange(rng[0], rng[1]),
#                              interval=inter, blit=False)
#         if show:
#             return plt.show()
#         else:
#             return ani
    
#     def save(self,ani,output_file,fps = 10,dpi = 100):
#         if output_file.find('.gif') > 0:    
#             ani.save(output_file, writer='imagemagick',fps=fps,dpi=dpi)
#         elif output_file.find('.mp4') > 0:  
#             ani.save(output_file, writer='ffmpeg',fps=fps,dpi=dpi)
#         else:
#             raise Exception("unknown type for producing move")


# class FileManager:
#     def __init__(self,folder_list = []):
#         self.folder_list = folder_list

#     def update_folder_list_by_root_dir(self,root_dir):
#         self.folder_list = []
#         for folder,_,_ in os.walk(root_dir):
#             if "runID" in folder:
#                 self.folder_list.append(folder)

#     def sorted_by_value(self,keyword:str,eps:float = 1e-6):
#         fstr_dict = {}
#         pattern = keyword + r"_([+-]?\d*\.?\d+(?:[Ee][+-]?\d+)?)"
#         for fstr in self.folder_list:
#             match = re.search(pattern, fstr)
#             if match:
#                 value = float(match.group(1))
#                 value_str = match.group(1)
#             else:
#                 raise(FileNotFoundError("no pattern is matched"))
#             fstr_dict[fstr] = (value,value_str)
#         sorted_items = sorted(fstr_dict.items(),key = lambda k:k[1][0])
#         sorted_str_dict = {}
#         for fstr,v in sorted_items:
#             _,val_str = v
#             if val_str in sorted_str_dict:
#                 sorted_str_dict[val_str].append(fstr)
#             else:
#                 sorted_str_dict[val_str] = [fstr]
#         return sorted_str_dict
    
#     def filter_by_dict(self,filter_dict:dict,eps:float=1e-6,update_list = False):
#         fstrs = []
#         for fstr in self.folder_list:
#             satisfied = True
#             for k,v in filter_dict.items():
#                 pattern = k + r"_([+-]?\d*\.?\d+(?:[Ee][+-]?\d+)?)"
#                 match = re.search(pattern, fstr)
#                 value = float(match.group(1))
#                 if abs(v - value) > eps:
#                     satisfied = False
#                     break
#             if satisfied:
#                 fstrs.append(fstr)
#         if update_list:
#             self.folder_list = fstrs
#         return fstrs
    
#     def value_larger_than(self,keyword,threshold,update_list = False):
#         folder_list = []
#         for fstr in self.folder_list:
#             pattern = keyword + r"_([+-]?\d*\.?\d+(?:[Ee][+-]?\d+)?)"
#             match = re.search(pattern, fstr)
#             value = float(match.group(1))
#             if value > threshold:
#                 folder_list.append(fstr)
#         if update_list:
#             self.folder_list = folder_list
#         return folder_list
    
#     def value_less_than(self,keyword,threshold,update_list = False):
#         folder_list = []
#         for fstr in self.folder_list:
#             pattern = keyword + r"_([+-]?\d*\.?\d+(?:[Ee][+-]?\d+)?)"
#             match = re.search(pattern, fstr)
#             value = float(match.group(1))
#             if value < threshold:
#                 folder_list.append(fstr)
#         if update_list:
#             self.folder_list = folder_list
#         return folder_list

 
# def extract_value_from_filename(fstr,keyword):
#     pattern = keyword + r"_([+-]?\d*\.?\d+(?:[Ee][+-]?\d+)?)"
#     match = re.search(pattern, fstr)
#     value = float(match.group(1))
#     if match:
#         value = float(match.group(1))
#     else:
#         raise(FileNotFoundError("no pattern is matched in {}".format(fstr)))
#     return value
        


import zipfile
import os
import json
import re
import numpy as np
import subprocess
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib import animation 
import finufft
from itertools import product
from numba import jit
from sklearn.neighbors import NearestNeighbors
from scipy.special import gamma

class DataLoader():
    def __init__(self,dir_name,mode = "all",rng=[]):
        self.info_file = os.path.join(dir_name,"info.json")
        self.dir_name = dir_name
        self.data = []
        if not os.path.isfile(self.info_file):
            raise Exception("Info file does not exist!")

        with open(self.info_file,'r') as fs:
            self.info = json.loads(fs.read())
        
        if not "r" in self.info:
            v = np.prod(self.info["box_size"])
            if self.info['dim'] == 3:   
                self.info["r"] = np.power(3.0/4.0*v*self.info['phi']/np.pi/self.info['N'],1.0/3.0)
            elif self.info['dim'] == 2:
                self.info["r"] = np.power(v*self.info['phi']/self.info['N']/np.pi,1.0/2.0)
        
        if mode == "all":
            for file in os.listdir(self.dir_name):
                #iter = re.search(".+iter(\w+)[.].+",file)
                if file.endswith(".zip") and file.find('iter')>0:
                    #print(file)
                    self.data.append(self.read_zipped_json(os.path.join(dir_name,file)))
            self.data = sorted(self.data,key=lambda data:data['iter'])
        elif mode =="last":
            for file in os.listdir(self.dir_name):
                if file.endswith(str(self.info['last_iteration'])+".zip"):
                    self.data.append(self.read_zipped_json(os.path.join(dir_name,file)))
                elif file.endswith(str(self.info['last_iteration']-1)+".zip"):
                    self.data.append(self.read_zipped_json(os.path.join(dir_name,file)))
        elif mode =="last_existed":
            iters = []
            for file in os.listdir(self.dir_name):
                if file.endswith(".zip") and file.find('iter')>0:
                    m = re.search(".+iter(\w+)[.].+",file).group(1)
                    _iter = int(re.search(".+iter(\w+)[.].+",file).group(1))
                    iters.append(_iter)
            max_iter = max(iters)
            for file in os.listdir(self.dir_name):
                if file.endswith(str(max_iter)+".zip"):
                    self.data.append(self.read_zipped_json(os.path.join(dir_name,file))) 
        elif mode == "range":
            iters = []
            for file in os.listdir(self.dir_name):
                if file.endswith(".zip") and file.find('iter')>0:
                    m = re.search(".+iter(\w+)[.].+",file).group(1)
                    _iter = int(re.search(".+iter(\w+)[.].+",file).group(1))
                    iters.append(_iter)
            iters.sort()
            rng[0] = iters[rng[0]]
            rng[1] = iters[rng[1]]
            for file in os.listdir(self.dir_name):
                if file.endswith(".zip") and file.find('iter')>0:
                    m = re.search(".+iter(\w+)[.].+",file).group(1)
                    _iter = int(re.search(".+iter(\w+)[.].+",file).group(1))   
                    if _iter>= rng[0] and _iter <= rng[1]:
                        self.data.append(self.read_zipped_json(os.path.join(dir_name,file)))
            self.data = sorted(self.data,key=lambda data:data['iter'])
        
                        
    def read_data_by_index(self,idx):
         file_name = "data_iter{}.json".format(idx*self.info['n_iter'])
         return self.read_zipped_json(os.path.join(self.dir_name,file_name))

    def read_zipped_json(self,file_name):
        with zipfile.ZipFile(file_name) as myzip:
            with myzip.open('data.json') as myfile:
                return json.loads(myfile.read())
    
    def get_data_list(self,key):
        iter = [d['iter'] for d in self.data]
        value = [d[key] for d in self.data]
        return iter,value

    def get_last_iter_data(self,key):
        return self.data[-1]['iter'],self.data[-1][key]
    
    def get_pair_corr(self,x,dr = None):
        x = np.reshape(x,(-1,self.info["dim"]))
        if not dr :
            dr = self.info["r"]*0.02
        r_max = self.info["r"]*30
        V = 1.0
        for d in self.info["box_size"]:
            V *= d
        n_slots = int(r_max/dr + 0.5)
        r = np.linspace(0.5*dr,r_max,n_slots)
        count = np.zeros(r.shape)
        for i in range(self.info["N"]):
            dx = x[i,:] - x
            # periodic boundary condition
            for d in range(self.info["dim"]):
                dx[np.where(dx[:,d] < -0.5*self.info["box_size"][d]),d] += self.info["box_size"][d]
                dx[np.where(dx[:,d] > 0.5*self.info["box_size"][d]),d] -= self.info["box_size"][d]
            dx_norm = np.sqrt(np.sum(dx*dx,axis=1))
            indices = (dx_norm/dr).astype(int)
            indices = indices[np.where(indices<n_slots)]
            count[indices] += 1.0/self.info["N"]
        count[0] = 0.0 # the zero distance should be excluded
        # normalize the count
        if self.info["dim"] == 3:
            count /= (4.0*np.pi*r*r*dr + 1e-8)
        elif self.info["dim"] == 2:
            count /= (2.0*np.pi*r*dr + 1e-8)
        density = self.info["N"]/V
        return (r/(self.info["r"]*2.0),count/density)
    def get_pbc_position(self,x):
        if x.shape[-1] != self.info["dim"]:
            x = np.reshape(x,(-1,self.info["dim"]))
        for d in range(self.info["dim"]):
            while np.any(x[:,d] < 0.0):
                x[np.where(x[:,d] < 0.0),d] += self.info["box_size"][d]
            while np.any(x[:,d] > self.info["box_size"][d]):
                x[np.where(x[:,d] > self.info["box_size"][d]),d] -= self.info["box_size"][d]
        return x

    def get_pair_corr_along_axes2d(self,x,inbr,dr = None,dt = 0.1):
        x = np.reshape(x,(-1,self.info["dim"]))
        if not dr :
            dr = self.info["r"]*0.01
        r_max = self.info["r"]*30
        n_slots = int(r_max/dr + 0.5)
        r = np.linspace(0.5*dr,r_max,n_slots)
        para_count = np.zeros(r.shape)
        perp_count = np.zeros(r.shape)
        n_pairs = 0.0
        for i in range(self.info["N"]):
            if len(inbr[i]) > 0:
                dx = x - x[i,:]
                # periodic boundary condition
                for d in range(self.info["dim"]):
                    dx[np.where(dx[:,d] < -0.5*self.info["box_size"][d]),d] += self.info["box_size"][d]
                    dx[np.where(dx[:,d] > 0.5*self.info["box_size"][d]),d] -= self.info["box_size"][d]
                dx_norm = np.sqrt(np.sum(dx*dx,axis=1))
                for j in inbr[i]:
                    n_pairs += 1.0
                    theta_parallel = np.arctan2(dx[j,1],dx[j,0])
                    dtheta = np.arctan2(dx[:,1],dx[:,0]) - theta_parallel
                    dtheta[np.where(dtheta > np.pi)] -= 2.0*np.pi
                    dtheta[np.where(dtheta < -np.pi)] += 2.0*np.pi
                    select_para = np.where( np.logical_or(np.abs(dtheta) < 0.5*dt, np.abs(np.abs(dtheta) - np.pi) < 0.5*dt))
                    select_perp = np.where( np.logical_or(np.abs(dtheta - 0.5*np.pi) < 0.5*dt, np.abs(dtheta + 0.5*np.pi) < 0.5*dt))

                    indices_para = (dx_norm[select_para]/dr).astype(int)
                    indices_para = indices_para[np.where(indices_para<n_slots)]
                    indices_perp= (dx_norm[select_perp]/dr).astype(int)
                    indices_perp = indices_perp[np.where(indices_perp<n_slots)]
                    
                    para_count[indices_para] += 1.0
                    perp_count[indices_perp] += 1.0

        para_count[0] = 0.0 # the zero distance should be excluded
        perp_count[0] = 0.0
        para_count /= (2.0*dt*r*dr + 1e-8)
        perp_count /= (2.0*dt*r*dr + 1e-8)
        V = 1.0
        for d in self.info["box_size"]:
            V *= d
        density = self.info["N"]/V
        print(n_pairs)
        return (r/(self.info["r"]*2.0),para_count/(density*n_pairs),perp_count/(density*n_pairs))


    
    def get_strucutre_factor(self,x,N = None):
        # x: numpy array for particle positions
        # N: resolvsion in k-space
        #    the output is a N^dim array
        x = np.reshape(x,(-1,self.info["dim"]))
        x = self.get_pbc_position(x) # position betwee 0 and L
        if N is None:
            N = 512
        if self.info["dim"] == 1:
            x1 = (x[:,0]/self.info["box_size"][0] - 0.5)*2.0*np.pi 
            c = np.ones(self.info["N"]).astype(np.complex128)
            f = finufft.nufft1d1(x1, c, N, modeord=0)
            s = np.abs(f*np.conjugate(f))/self.info["N"]
            kx = np.fft.fftshift(np.fft.fftfreq(N)*N*2*np.pi/self.info["box_size"][0])
            return (kx,s)
        elif self.info["dim"] == 2:
            x1 = (x[:,0]/self.info["box_size"][0] - 0.5)*2.0*np.pi 
            x2 = (x[:,1]/self.info["box_size"][1] - 0.5)*2.0*np.pi
            c = np.ones(self.info["N"]).astype(np.complex128)
            f = finufft.nufft2d1(x1, x2, c, (N, N),modeord=0)
            s = np.abs(f*np.conjugate(f))/self.info["N"]
            kx = np.fft.fftshift(np.fft.fftfreq(N)*N*2*np.pi/self.info["box_size"][0])
            ky = np.fft.fftshift(np.fft.fftfreq(N)*N*2*np.pi/self.info["box_size"][1])
            return (kx,ky,s)
        elif self.info["dim"] == 3:
            x1 = (x[:,0]/self.info["box_size"][0] - 0.5)*2.0*np.pi 
            x2 = (x[:,1]/self.info["box_size"][1] - 0.5)*2.0*np.pi
            x3 = (x[:,2]/self.info["box_size"][2] - 0.5)*2.0*np.pi
            c = np.ones(self.info["N"]).astype(np.complex128)
            f = finufft.nufft3d1(x1, x2, x3, c, (N, N, N),modeord=0)
            s = np.abs(f*np.conjugate(f))/self.info["N"]
            kx = np.fft.fftshift(np.fft.fftfreq(N)*N*2*np.pi/self.info["box_size"][0])
            ky = np.fft.fftshift(np.fft.fftfreq(N)*N*2*np.pi/self.info["box_size"][1])
            kz = np.fft.fftshift(np.fft.fftfreq(N)*N*2*np.pi/self.info["box_size"][2])
            return (kx,ky,kz,s)
        

    # def get_density_fluc(self,x,num_dia=100,num_samples=100):
    #     # x: numpy array for particle positions
    #     x = np.reshape(x,(-1,self.info["dim"]))
    #     x = self.get_pbc_position(x) # position betwee 0 and L

    #     # diameters = np.linspace(self.info["R"] * 1e-1, self.info["box_size"][0], num_dia)
    #     diameters = np.logspace(np.log10(self.info["r"] * 5e-1), 0.99*np.log10(self.info["box_size"][0]), num_dia)

    #     # counts = np.zeros(num_samples)
    #     # means = np.zeros(len(diameters))
    #     # variances = np.zeros(len(diameters))

    #     # for i in range(len(diameters)):
    #     #     for j in range(num_samples):
    #     #         center = self.info["box_size"][0] * np.random.rand(self.info["dim"])  # Random center
    #     #         # Compute periodic distances from points to center
    #     #         diff = np.abs(x - center)
    #     #         diff = np.minimum(diff, self.info["box_size"][0] - diff)  # Wrap distances
    #     #         distances = np.sqrt(np.sum(diff**2, axis=1))
    #     #         radius = diameters[i] / 2.0
    #     #         counts[j] = np.sum(distances <= radius)
    #     #     means[i] = np.mean(counts, axis=0)
    #     #     variances[i] = np.var(counts, axis=0)

    #     counts = np.zeros((num_samples, num_dia))

    #     for i, D in enumerate(diameters):
    #         radius = D / 2.0
    #         centers = self.info["box_size"][0] * np.random.rand(num_samples, self.info["dim"])  # M random centers
            
    #         # Compute periodic distances from points to all centers
    #         diff = x[:, np.newaxis, :] - centers[np.newaxis, :, :]
    #         diff = np.abs(diff)
    #         diff = np.minimum(diff, self.info["box_size"][0] - diff)  # Periodic boundaries
    #         distances = np.sqrt(np.sum(diff ** 2, axis=2))            # Shape: (N, num_samples)
    #         counts[:, i] = np.sum(distances <= radius, axis=0)        # Counts for each center

    #     means = np.mean(counts, axis=0)
    #     variances = np.var(counts, axis=0)

    #     return diameters, means, variances

    
def get_nearast_distances(self,x,box_size,n_nbrs=1,pbc = True):
    '''
    return the nearest neighbours and the distances
            e.g.ind[i] the nearest neighbour of particle i
                dist[i] the distance between particle i and its nearest neighbour
    '''
    dim = len(box_size)
    x = x.reshape((-1,dim))
    N = x.shape[0]
    if pbc:
        augmented_x = [x]
        for d in range(dim):
            dx = [0.0 for d in range(dim)]
            dx[d] = box_size[d]
            augmented_x.append(x + dx)
            augmented_x.append(x - dx)
        x = np.concatenate(augmented_x)
    nbrs = NearestNeighbors(n_neighbors=n_nbrs+1, algorithm='auto').fit(x)
    distances, indices = nbrs.kneighbors(x)
    dist,ind = distances[:,1],indices[:,1]
    if pbc:
        for _ in range(2*dim):
            ind[np.where(ind>=N)] -= N
    return ind,dist

@jit(nopython=True)
def get_radial_structure_factor2d(kx,ky,s,n_bins=800,cut_off = 0.7):
    k_max = np.sqrt(kx[-1]**2 + ky[-1]**2)*cut_off 
    s_k = np.zeros(n_bins)
    dA = (kx[2]-kx[1])*(ky[2]-ky[1])
    delta_k = k_max/n_bins
    for i in range(s.shape[0]):
        for j in range(s.shape[1]):
            k = np.sqrt(kx[i]*kx[i] + ky[j]*ky[j])
            bin_num = int(k/delta_k) 
            if bin_num < n_bins:
                s_k[bin_num] += s[i,j]*dA
    k = np.arange(n_bins)*delta_k + 0.5*delta_k
    s_k /= 2.0*np.pi*k*delta_k
    return k,s_k

def get_radial_profile(data, center=None,rm_npts=1):
    ndim = len(data.shape)
    R = int(data.shape[0]/2)
    if center is None:
        center = np.array(data.shape)/2
    for i in range(ndim):
        center = np.expand_dims(center, axis=-1)
    idx = np.indices((data.shape))
    r = np.sqrt(np.sum((idx - center)**2,axis=0))
    r = r.astype(int)
    r = np.where(r <= R, r, R+1)
    tbin = np.bincount(r.ravel(),data.ravel())
    nr = np.bincount(r.ravel())
    radialprofile = tbin/nr
    return np.unique(r).ravel()[rm_npts:-1], radialprofile[rm_npts:-1]

@jit(nopython=True)
def get_radial_structure_factor3d(kx,ky,kz,s,n_bins=100,cut_off = 0.7):
    s[0,0,0] = 0.0
    k_max = np.sqrt(kx[-1]**2 + ky[-1]**2 + kz[-1]**2)*cut_off 
    s_k = np.zeros(n_bins)
    dV = (kx[2]-kx[1])*(ky[2]-ky[1])*(kz[2]-kz[1])
    delta_k = k_max/n_bins
    for i in range(s.shape[0]):
        for j in range(s.shape[1]):
            for l in range(s.shape[2]):
                k = np.sqrt(kx[i]*kx[i] + ky[j]*ky[j] + kz[l]*kz[l])
                bin_num = int(k/delta_k)
                if bin_num < n_bins:
                    s_k[bin_num] += s[i,j,l]*dV
    k = np.arange(n_bins)*delta_k + 0.5*delta_k
    s_k /= 4.0*np.pi*k*k*delta_k
    return k[1:],s_k[1:]/s_k[-1]

@jit(nopython=True)
def get_density_fluc(x,dim=2,box_size=500,r=1.0,num_dia=100,num_samples=100):

    diameters = np.linspace(r*5e-1, 0.99*box_size, num_dia)
    counts = np.zeros(num_samples)
    means = np.zeros(len(diameters))
    variances = np.zeros(len(diameters))

    for i in range(len(diameters)):
        for j in range(num_samples):
            center = box_size * np.random.rand(dim)  # Random center
            # Compute periodic distances from points to center
            diff = np.abs(x - center)
            diff = np.minimum(diff, box_size - diff)  # Wrap distances
            distances = np.sqrt(np.sum(diff**2, axis=1))
            radius = diameters[i] / 2.0
            counts[j] = np.sum(distances <= radius)
        means[i] = np.mean(counts)
        variances[i] = np.var(counts)

    return diameters, means, variances


#load data for multiple calculations in a floder
class DataSetLoader():
    def __init__(self,dir_name,mode = "last"):
        self.dir_name = dir_name
        self.loader_list = []
        for dir in os.listdir(self.dir_name):
            if os.path.isdir(os.path.join(dir_name,dir)) and dir.find('#')>-1:
                my_data = DataLoader(os.path.join(dir_name,dir),mode = mode)
                self.loader_list.append(my_data)
            
    def get_data_list(self,iter_idx,key):
        if key in self.loader_list[0].info:
            data_list = [(d.info[key],d.data[iter_idx]) for d in self.loader_list]
            data_list = sorted(data_list,key = lambda data:data[0])
        else:
            data_list = [(d.data[iter_idx][key],d.data[iter_idx]) for d in self.loader_list]
            data_list = sorted(data_list,key = lambda data:data[0])
        val = [d[0] for d in data_list]
        data = [d[1] for d in data_list]
        return val,data

def save_data(loc,iter,x,e_tot,g_tot,inbr,n_active,clusters,e_batch,g_batch,batch_pairs,dx,rm=True,compress=True):
    if not os.path.isdir(loc):
        subprocess.run(["mkdir", loc])
    file_name = "data_iter{}.json".format(iter)
    f = os.path.join(loc,file_name)
    if os.path.exists(f) and rm:
        subprocess.run(["rm", "-r",f])
    # wrap the data
    data = dict()
    data['iter'] = iter
    data['x'] = x.tolist()
    data['total_energy'] = e_tot
    data['total_gradient'] = g_tot
    data['neighbor'] = inbr
    data['n_active'] = n_active
    data['n_clusters'] = len(clusters)
    cluster_sizes = [len(clusters[i]) for i in range(len(clusters))]
    data['avg_cluster_size'] = float(np.mean(cluster_sizes))
    data['batch_energy'] = e_batch
    data['batch_gradient'] = g_batch
    data['batch_pairs'] = batch_pairs
    data['dx'] = dx.tolist()
    fzip = f.replace(".json",".zip")
    # save the data as json a file

    if compress:
        fzip = f.replace(".json",".zip")
        with zipfile.ZipFile(fzip, 'w') as myzip:
            myzip.writestr("data.json",json.dumps(data),compress_type=zipfile.ZIP_DEFLATED)
    else:
        with open(f,'w') as fs:
            fs.write(json.dumps(data))


class SpherePlot():
    def __init__(self,loader_or_str):
        if isinstance(loader_or_str,str):
            self.loader = DataLoader(loader_or_str)
        else:
            self.loader = loader_or_str
        if not 'radii' in self.loader.info:
            self.loader.info['radii'] = np.full(self.loader.info['N'],self.loader.info['r']) 
        self.npts = 10
        self.active_color='r'
        self.atom_color='b'
        self.batch_color='b'
        self.bond_color=None
    
    def set_npts4spheres(self,npts):
        self.npts = npts
    
    def plot_circle(self,ax,r,x,c='b',alpha=0.9):
        circle = mpatches.Circle(x, r, facecolor = c,edgecolor='k')
        ax.add_patch(circle)

    def plot_sphere(self,ax,r,xx,c = 'b',alpha = 0.9):
        if not ax:
            raise Exception("axis for plotting is None")
        p = np.linspace(0.0,2*np.pi,2*self.npts)
        t = np.linspace(0.0,np.pi,self.npts)
        theta, phi = np.meshgrid(t,p)
        x = r*np.sin(theta)*np.cos(phi) + xx[0]
        y = r*np.sin(theta)*np.sin(phi) + xx[1]
        z = r*np.cos(theta) + xx[2]
        ax.plot_surface(x, y, z,color = c,alpha = alpha)
    
    def plot_line(self,ax,p1,p2,c = 'r',alpha = 0.5):
        if self.loader.info['dim'] ==2:
            ax.plot([p1[0],p2[0]],[p1[1],p2[1]],color= c)
        elif self.loader.info['dim'] ==3:
            ax.plot3D([p1[0],p2[0]],[p1[1],p2[1]],[p1[2],p2[2]],color= c)
        
    def plot_arrow(self,ax,x,dx,c = 'k',alpha = 0.5):
        if self.loader.info['dim'] ==2:
            ax.arrow(x[0],x[1],dx[0],dx[1],color= c,alpha=alpha)
        elif self.loader.info['dim'] ==3:
            ax.plot3D([p1[0],p2[0]],[p1[1],p2[1]],[p1[2],p2[2]],color= c)
        
    def plot_atom(self,ax,r,x,c = 'b',alpha = 0.9):
        if self.loader.info['dim'] == 2:
            self.plot_circle(ax,r,x,c = c,alpha = alpha)
        elif self.loader.info['dim'] == 3:
            self.plot_sphere(ax,r,x,c = c,alpha = alpha)
 
    def plot_atoms(self,ax,data,alpha=0.5,pbc = True):
        x = np.array(data['x'])
        box = self.loader.info['box_size']
        if pbc:
            rx = x.reshape(self.loader.info['N'],self.loader.info['dim'])
            for d in range(self.loader.info['dim']):
                rx[np.where(rx[:,d] > box[d]),d] -= box[d] 
                rx[np.where(rx[:,d] < 0.0),d]+= box[d]
        radii = self.loader.info['radii']
        for i in range(self.loader.info['N']):
            x_i = x[i*self.loader.info['dim']:(i+1)*self.loader.info['dim']]
            if len(data['neighbor'][i]) > 0:
                self.plot_atom(ax,radii[i],x_i,c = self.active_color,alpha=alpha)
            else:
                self.plot_atom(ax,radii[i],x_i,c = self.atom_color,alpha=alpha)
    def plot_dx(self,ax,data,alpha=0.5,scale = 1.0,pbc=True):
        x = np.array(data['x'])
        dx = np.array(data['dx'])*scale
        box = self.loader.info['box_size']
        if pbc:
            rx = x.reshape(self.loader.info['N'],self.loader.info['dim'])
            for d in range(self.loader.info['dim']):
                rx[np.where(rx[:,d] > box[d]),d] -= box[d] 
                rx[np.where(rx[:,d] < 0.0),d]+= box[d]
        if self.loader.info['dim'] ==2:
            for i in range(self.loader.info['N']):
                x_i = x[i*self.loader.info['dim']:(i+1)*self.loader.info['dim']]
                dx_i = dx[i*self.loader.info['dim']:(i+1)*self.loader.info['dim']]
                ax.arrow(x_i[0],x_i[1],dx_i[0],dx_i[1],color= 'k',alpha=alpha)
        elif self.loader.info['dim'] ==3:
            rx = x.reshape(self.loader.info['N'],self.loader.info['dim'])
            rdx = dx.reshape(self.loader.info['N'],self.loader.info['dim'])
            ax.quiver(x[:,0],x[:,1],x[:,2],rdx[:,0],rdx[:,1],rdx[:,2])
     
    def plot_gradient(self,ax,data,alpha=0.5,scale = 1.0,pbc=True):
        x = np.array(data['x'])
        dx = np.array(data['total_gradient'])*scale
        box = self.loader.info['box_size']
        if pbc:
            rx = x.reshape(self.loader.info['N'],self.loader.info['dim'])
            for d in range(self.loader.info['dim']):
                rx[np.where(rx[:,d] > box[d]),d] -= box[d] 
                rx[np.where(rx[:,d] < 0.0),d]+= box[d]
        if self.loader.info['dim'] ==2:
            for i in range(self.loader.info['N']):
                x_i = x[i*self.loader.info['dim']:(i+1)*self.loader.info['dim']]
                dx_i = dx[i*self.loader.info['dim']:(i+1)*self.loader.info['dim']]
                ax.arrow(x_i[0],x_i[1],dx_i[0],dx_i[1],color= 'k',alpha=alpha)
        elif self.loader.info['dim'] ==3:
            rx = x.reshape(self.loader.info['N'],self.loader.info['dim'])
            rdx = dx.reshape(self.loader.info['N'],self.loader.info['dim'])
            ax.quiver(x[:,0],x[:,1],x[:,2],rdx[:,0],rdx[:,1],rdx[:,2])
    def plot_batch(self,ax,data):
        x = data['x']
        radii = self.loader.info['radii']
        for pair in data['batch_pairs']:
            i,j =pair[0],pair[1]
            x_i = x[i*self.loader.info['dim']:(i+1)*self.loader.info['dim']]
            x_j = x[j*self.loader.info['dim']:(j+1)*self.loader.info['dim']]
            self.plot_atom(ax,radii[i],x_i,c=self.batch_color)
            self.plot_atom(ax,radii[j],x_j,c=self.batch_color)
            if self.bond_color:
                self.plot_line(ax,x_i,x_j,c=self.bond_color)
    
    def plot_box(self,ax,c='b'):
        box = self.loader.info['box_size']
        if self.loader.info['dim'] ==3: 
            vertices = [[0,0,0],[0,0,box[2]],[0,box[1],0],[0,box[1],box[2]],[box[0],0,0],[box[0],0,box[2]],[box[0],box[1],0],box]
        elif self.loader.info['dim'] ==2:
            vertices = [[0,0],[0,box[1]],[box[0],0],box]
        for i in range(len(vertices)):
            for j in range(i+1,len(vertices)):
                if np.abs(L - np.linalg.norm(vertices[i],vertices[j])) < 1e-8:
                    self.plot_line(vertices[i],vertices[j])

         
    def animate(self,fig,fn,rng,inter,show=True,output_file=None):
        # the local animation function
        def animate_fn(iter):
            # we want a fresh figure everytime
            fig.clf()
            # call the global function
            fn(iter)

        ani = animation.FuncAnimation(fig, animate_fn,frames=np.arange(rng[0], rng[1]),
                             interval=inter, blit=False)
        if show:
            return plt.show()
        else:
            return ani
    
    def save(self,ani,output_file,fps = 10,dpi = 100):
        if output_file.find('.gif') > 0:    
            ani.save(output_file, writer='imagemagick',fps=fps,dpi=dpi)
        elif output_file.find('.mp4') > 0:  
            ani.save(output_file, writer='ffmpeg',fps=fps,dpi=dpi)
        else:
            raise Exception("unknown type for producing move")


class FileManager:
    def __init__(self,folder_list = []):
        self.folder_list = folder_list

    def update_folder_list_by_root_dir(self,root_dir):
        self.folder_list = []
        for folder,_,_ in os.walk(root_dir):
            if "runID" in folder:
                self.folder_list.append(folder)

    def sorted_by_value(self,keyword:str,eps:float = 1e-6):
        fstr_dict = {}
        pattern = keyword + r"_([+-]?\d*\.?\d+(?:[Ee][+-]?\d+)?)"
        for fstr in self.folder_list:
            match = re.search(pattern, fstr)
            if match:
                value = float(match.group(1))
                value_str = match.group(1)
            else:
                raise(FileNotFoundError("no pattern is matched"))
            fstr_dict[fstr] = (value,value_str)
        sorted_items = sorted(fstr_dict.items(),key = lambda k:k[1][0])
        sorted_str_dict = {}
        for fstr,v in sorted_items:
            _,val_str = v
            if val_str in sorted_str_dict:
                sorted_str_dict[val_str].append(fstr)
            else:
                sorted_str_dict[val_str] = [fstr]
        return sorted_str_dict
    
    def filter_by_dict(self,filter_dict:dict,eps:float=1e-6,update_list = False):
        fstrs = []
        for fstr in self.folder_list:
            satisfied = True
            for k,v in filter_dict.items():
                pattern = k + r"_([+-]?\d*\.?\d+(?:[Ee][+-]?\d+)?)"
                match = re.search(pattern, fstr)
                value = float(match.group(1))
                if abs(v - value) > eps:
                    satisfied = False
                    break
            if satisfied:
                fstrs.append(fstr)
        if update_list:
            self.folder_list = fstrs
        return fstrs
    
    def value_larger_than(self,keyword,threshold,update_list = False):
        folder_list = []
        for fstr in self.folder_list:
            pattern = keyword + r"_([+-]?\d*\.?\d+(?:[Ee][+-]?\d+)?)"
            match = re.search(pattern, fstr)
            value = float(match.group(1))
            if value > threshold:
                folder_list.append(fstr)
        if update_list:
            self.folder_list = folder_list
        return folder_list
    
    def value_less_than(self,keyword,threshold,update_list = False):
        folder_list = []
        for fstr in self.folder_list:
            pattern = keyword + r"_([+-]?\d*\.?\d+(?:[Ee][+-]?\d+)?)"
            match = re.search(pattern, fstr)
            value = float(match.group(1))
            if value < threshold:
                folder_list.append(fstr)
        if update_list:
            self.folder_list = folder_list
        return folder_list

 
def extract_value_from_filename(fstr,keyword):
    pattern = keyword + r"_([+-]?\d*\.?\d+(?:[Ee][+-]?\d+)?)"
    match = re.search(pattern, fstr)
    value = float(match.group(1))
    if match:
        value = float(match.group(1))
    else:
        raise(FileNotFoundError("no pattern is matched in {}".format(fstr)))
    return value
        
