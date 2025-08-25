import os 
import time
import statistics
import random
from multiprocessing import Pool
import pathlib
import ctypes
import glob
import statsmodels.api as sm
import csv
import neurokit2 as nk
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from multiprocessing import Pool
from scipy.stats import wasserstein_distance
from scipy import integrate
from sklearn.manifold import MDS

start = time.time()
fftw_lib=ctypes.CDLL('./fftw_wrapper_lach.so')
#fftw_lib=ctypes.CDLL('./fftw_wrapper_lach.so')
fftw_lib.fftw_wrapper.argtypes=[ctypes.POINTER(ctypes.c_double),ctypes.POINTER(ctypes.c_double),ctypes.c_int]
fftw_lib.fftw_wrapper.restype =None

def permutate_vals (yvals_perm_list):
    yvals_perm_list_copy=[]
    for i in range(0,len(yvals_perm_list)):
        yvals_perm_list_copy.append(yvals_perm_list[i])
    k=len(yvals_perm_list_copy)
    new_perm_list=[]
    for i in range (0,k):
        yvals_perm_list_copy[i]=yvals_perm_list_copy[random.randint(0,(k-1))]
        new_perm_list=yvals_perm_list_copy
        return new_perm_list

def fftw_wrapper(input_array):
    size = len(input_array)
    output_array=np.zeros(2*size-2,dtype=np.float64)
    fftw_lib.fftw_wrapper(input_array.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),output_array.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),ctypes.c_int(size))
    output_array_tolist=output_array.tolist()
    return output_array_tolist

def fft_list_abs_sorted_two_lists (ylist1,ylist2):
    ylist1copy=[]
    for i in range(len(ylist1)):
        ylist1copy.append(ylist1[i])
        ylist2copy=[]
    for i in range(len(ylist2)):
        ylist2copy.append(ylist2[i])
    yarray1=np.array(ylist1copy)
    yarray2=np.array(ylist2copy)
    coef_array_1=fftw_wrapper(yarray1)
    coef_array_2=fftw_wrapper(yarray2)
    coef_array_1.sort(reverse=True)
    coef_array_2.sort(reverse=True)
    return coef_array_1, coef_array_2

def wasserstein_list(orig_list,perm_list):
    iterations=500
    list_wasserstein_case1=[]
    list_wasserstein_case2=[]
    orig_list_copy=[]
    for i in range(len(orig_list)):
        orig_list_copy.append(orig_list[i])
    perm_list_copy=[]
    for i in range(len(perm_list)):
        perm_list_copy.append(perm_list[i])
    for j in range(0,iterations):
        perm_list_1=permutate_vals(perm_list_copy)
        perm_list_1copy=[]
        for i in range(0,len(perm_list_1)):
            perm_list_1copy.append(perm_list_1[i])
        fft1,fft2 = fft_list_abs_sorted_two_lists(orig_list_copy, perm_list_1copy)
        list_wasserstein_case1.append(wasserstein_distance(fft1,fft2))
        perm_list_2=permutate_vals(perm_list_1copy)
        perm_list_2copy=[]
        for i in range(0,len(perm_list_2)):
            perm_list_2copy.append(perm_list_2[i])
        fft3,fft4=fft_list_abs_sorted_two_lists(perm_list_1copy, perm_list_2copy)
        list_wasserstein_case2.append(wasserstein_distance(fft3,fft4))
    return list_wasserstein_case1, list_wasserstein_case2

def extractMAG(path_string):
    
    mag_milli = []
    csv_files = glob.glob(f"{path_string}/*.csv")
    ne_csv_files = [f for f in csv_files if os.path.getsize(f)>0]
    
    df_list = (pd.read_csv(file) for file in ne_csv_files)
    data_milli = list(df_list)
    for i in range(0,len(data_milli)):
        mag_milli.append(   ( ne_csv_files[i]  ,   ((data_milli[i]['mag']).tolist()) ) )
    mag_milli_pop = [(s,l) for s,l in mag_milli if len(l)>=20]
    return mag_milli_pop

def extractMJD(path_string):
    mjd_milli= []
    csv_files = glob.glob(f"{path_string}/*.csv")
    df_list = (pd.read_csv(file) for file in csv_files)
    data_milli = list(df_list)
    for i in range(0,len(data_milli)):
        mjd_milli.append( ((data_milli[i]['mag']).tolist()))
    mjd_milli_pop = [ele for ele in mjd_milli if ele != []]
    return mjd_milli_pop

def lsof_acf(tseries_list):

    indiv_delta=[]
    for j in range (0, len(tseries_list)-1):
        indiv_delta.append(tseries_list[j+1]-tseries_list[j])
    
    
    lsofls_acf=sm.tsa.acf(indiv_delta,nlags=10)
    return lsofls_acf

def multi_thread_wasslist(filt_ornot_maglist):
    list2=filt_ornot_maglist[:]
    list2_copy = list2[:]
    wasslistR1P,wasslistR2P=wasserstein_list(list2,list2_copy)

    wass_list_modifiedDoubPerm=[]
    for i in range(0,len(wasslistR1P)):
        wass_list_modifiedDoubPerm.append(wasslistR1P[i]/statistics.mean(wasslistR2P))
    return statistics.mean(wass_list_modifiedDoubPerm)

def integration_entropy(y_list4romb):
    ffty1=fft_list_abs_sorted_two_lists(y_list4romb,y_list4romb)
    integral = integrate.trapz(ffty1)
    return integral[0]



def process_tuples(full_mag_list_QSO):
    string,floats=full_mag_list_QSO 
#-----------------------------------------------------------------------------------
#Id list

    QSOfilenames = os.path.basename(string)
#------------------------------------------------------------------------------------------
#creation of con non identifier
#con_ones_list = [1] * len(confilenames)
#-------------------------------------------------------------------------------------------------
# Sample Entropy
    sampEnt_list_QSO=((nk.entropy_sample(floats)[0]))
#-----------------------------------------------------------------------------------------------
#Multis-scale Entropy
    lstCarr=np.array(floats)
    multEnt_list_QSO=(nk.entropy_multiscale(lstCarr)[0])
#---------------------------------------------------------------------------------------
#Integral fourier coefficient entropy
   
    intEnt_list_QSO=(integration_entropy(floats))
#---------------------------------------------------------------------------------------
#Fourier Coeff WD Normalized
    #print('How many FP threads for wasserstein Fcoeff')
    FouWassEnt_list_QSO=multi_thread_wasslist(floats)
#---------------------------------------------------------------------------------------
#ACF multiscale Entropy
  
    onlymags_acf_l=lsof_acf(floats)
    ACFEnt_list_QSO=(nk.entropy_sample(onlymags_acf_l)[0])

    return (QSOfilenames,ACFEnt_list_QSO,FouWassEnt_list_QSO,intEnt_list_QSO,multEnt_list_QSO,sampEnt_list_QSO)
#-------------------------------------------------------------------------------

def parallel_process(data_list,num_processes):
    if num_processes is None:
        num_processes=os.cpu_count()
    return results
def write_to_csv(results,filename,mode):
    header=['ID','ACF_ENT','FOURIER_ENT','INTEGRAL_ENT','MULTISCALE_ENT','SAMPLE_ENT']
    with open(filename,mode,newline='') as file:
        writer=csv.writer(file)
        if mode =='w':
            writer.writerow(header)
        for result in results:
            writer.writerow(result)



def batch_process(data_list,batch_size):
    num_batches=(len(data_list)+batch_size-1)//batch_size
    for i in range(num_batches):
        batch=data_list[i*batch_size:(i+1)*batch_size]

        with Pool(192) as pool:
            results=pool.map(process_tuples,data_list)
        write_to_csv(results,'results_mult_QSO.csv',mode='a'if i>0 else'w')

def get_first_n_files(directory,n):
    all_files=[os.path.join(directory,file) for file in os.listdir(directory) if file.endswith('.csv')]
    return all_files[:n]

def main():
    Milli_paths_QSO = '/sharedata/fastdisk/perkeys/spentropy/data/alberto_QSO'

    full_mag_list_QSO_main=extractMAG(Milli_paths_QSO)
    batch_process(full_mag_list_QSO_main,1000)
main()





end = time.time()
print(f"Run time of the program is {end - start} seconds")
print('')
