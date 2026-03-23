import numpy as np
import math
from scipy import interpolate
from skimage.metrics import mean_squared_error
import matplotlib.pyplot as plt
import h5py

from scipy.interpolate import RegularGridInterpolator
from hilbert_toolkit import hilbert_fft_marple as dht
from hilbert_toolkit import hilbert_pad_simple
#https://github.com/usnistgov/Hilbert



def get_pixelposition(width):
    # a_f: pixel position (pixel at the detecter center, u=a_f*pixelsize)  
    #[ -4.5  -3.5  -2.5  -1.5  -0.5 0.5 1.5 2.5 3.5 4.5]
    return np.arange(width)-(width-1)/2

def recon_3(proj,cb_para):
  

    dht_pad = lambda x: hilbert_pad_simple(x, dht, 1)


    Volumen_num_xz=cb_para['Volumen_num_xz']
    Volumen_num_y=cb_para['Volumen_num_y']
    detector_width=cb_para['detector_width']
    detector_height=cb_para['detector_height']
    num_projs=cb_para['num_projs']
    voxelSize = cb_para['voxelSize']   
    SDD = cb_para['SDD']    
    SOD= cb_para['SOD'] 
    pixelSize = cb_para['pixelSize'] 



    ## weight 1
    u=-get_pixelposition(detector_width)* pixelSize   # detector position width
    v=-get_pixelposition(detector_height)* pixelSize   # detector position height
    xu,yu = np.meshgrid(u,v, indexing='ij')
    A=np.sqrt(SDD**2+xu**2+yu**2)
    cone_weight=np.tile(SOD/A,(num_projs,1,1))   #num_projs, width, height   3d
    proj=proj*cone_weight


    ## derivative
    dx=1
    for i in range(num_projs): 
        proj[i]=np.array([np.gradient((proj[i])[:,ii], dx)  for ii in  range(detector_height)]).T/pixelSize
    
    ## weight 2
    weigth_2=np.tile(A**2,(num_projs,1,1))   #num_projs, width, height   3d
    proj=proj*weigth_2

  

    ## Reconstruct image by interpolation
    reconstructed = np.zeros((Volumen_num_xz, Volumen_num_xz,Volumen_num_y))
    radius = Volumen_num_xz // 2-0.5
    radius_z = Volumen_num_y // 2-0.5
    xpr,ypr,zpr = np.meshgrid((np.arange(Volumen_num_xz)-radius)*voxelSize,(np.arange(Volumen_num_xz)-radius)*voxelSize,(np.arange(Volumen_num_y)-radius_z)*voxelSize , indexing='ij')
    

    ## rotation angle
    angle_shift=-3*np.pi/2 #degree offset
    theta = np.linspace(0.0+angle_shift,np.pi*2+angle_shift, num_projs, endpoint=False)
    

    for one_proj, angle in zip(proj, theta):   
        t = ypr * np.cos(angle) - xpr * np.sin(angle)
        U=SOD+ypr * np.sin(angle)  + np.cos(angle) *xpr
        ai=SDD*t/U
        bi=zpr*SDD/U
        weight_3=U**2+t**2+zpr**2
        weight_sin=np.sign(np.sin(angle+np.arctan(ai/SDD)))

        interpolant= RegularGridInterpolator((u, v), one_proj, bounds_error=False, fill_value=0)
        
        reconstructed += interpolant((ai,bi)) * weight_sin /  weight_3
    reconstructed=-reconstructed*np.pi  /num_projs
    

 
    for j in range(Volumen_num_xz):   #y
        for k in range(Volumen_num_y):   #z
            reconstructed[:,j,k]=dht_pad(reconstructed[:,j,k]) 
       
                
    return reconstructed/(-2*np.pi)



if __name__ == '__main__':
    
    path_data='/lgrp/edu-2025-2-gpulab/Data/proj_shepplogan128.hdf5'

    with h5py.File(path_data,'r') as f:

        voxelSize=f['voxelSize'][()] 
        Volumen_num_xz=int(f['Volumen_num_xz'][()] )
        Volumen_num_y=int(f['Volumen_num_y'][()] )
        SDD = f['SDD'][()] 
        SOD =f['SOD'][()]  
        magnification=SDD/SOD
        pixelSize =f['pixelSize'][()] 
  
        num_projs=int(f['num_projs'][()] )
        detector_width=int(f['detector_width'][()] )
        detector_height=int(f['detector_height'][()] )
    
        
        cb_para={
            'num_projs' :num_projs,
            'pixelSize' :pixelSize ,
            'voxelSize' : voxelSize,  
            'Volumen_num_xz':Volumen_num_xz, 
            'Volumen_num_y':Volumen_num_y, 
            'SDD':  SDD,   
            'SOD' :SOD,   
            'detector_width':detector_width,
            'detector_height':detector_height,
            }

        projection=f['Projection'][:,:,:]

        volume=recon_3(projection,cb_para)
        print(volume.shape)
  




    
