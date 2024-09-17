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
#import seaborn as sns
import pandas as pd
import numpy as np
from scipy.stats import wasserstein_distance
from scipy import integrate
from sklearn.manifold import MDS

start = time.time()
fftw_lib=ctypes.CDLL('./fftw_wrapper_lach.so')
fftw_lib.fftw_wrapper.argtypes=[ctypes.POINTER(ctypes.c_double),ctypes.POINTER(ctypes.c_double),ctypes.c_int]
fftw_lib.fftw_wrapper.restype =None

#def get_paths(path_name):
 #   return_path = []
  #  my_path = pathlib.Path(path_name)
   # for currentPath in my_path.iterdir():
    #    if currentPath.is_dir():
     #       get_paths(currentPath)
      #  return_path.append(str(currentPath))
    #return return_path

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
    iterations=50
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

#def lengthSlicing(sliceList):
#    while True:
 #       sliceListCopy=sliceList.copy()
  #      print('Type xMIN and xMax of your choosing. Respectively.')
   #     print('xMIN = ')
    #    xMIN=int(input())
     #   print('xMAX = ')
      #  xMAX=int(input())
       # if xMIN>xMAX:
        #    print('xMIN has to be lower than xMAX dumbfuck')
         #   continue
       # lengths=[len(lst) for lst in sliceListCopy]
       # filteredLengths=[num for num in lengths if xMIN<=num<=xMAX]
       # break
    #filteredListFull=[lest for lest, size in zip(sliceListCopy,lengths)if size in filteredLengths]
    #return filteredListFull

#def crossmatch( lsofls ):
 #   lsoflscopy=lsofls.copy()
  #  lsofls_total_array=[]
   # for j in range(0,len(lsoflscopy)):
       # lsofls_wd_ind=[]
       # for i in range(0,len(lsoflscopy)):
        #    lsofls_wd_ind.append(wasserstein_distance( lsoflscopy[j],lsoflscopy[i] ) )
        #lsofls_total_array.append(lsofls_wd_ind)
    #return lsofls_total_array

#def create_plot_doub(  plot_lsoflsL,plot_lsoflsN  ,let ):
    #modelL = MDS(n_components=2,dissimilarity='precomputed', random_state=10)
    #distance_df_2dL=modelL.fit_transform(plot_lsoflsL)
    #xL = distance_df_2dL[:,0]
    #yL = distance_df_2dL[:,1]
    #modelN = MDS(n_components=2,dissimilarity='precomputed', random_state=10)
    #distance_df_2dN=modelN.fit_transform(plot_lsoflsN)
    #3xN = distance_df_2dN[:,0]
    #yN = distance_df_2dN[:,1]
    #dbscan=DBSCAN(eps=0.02,min_samples=50)
    #clustering_results=dbscan.fit(distance_df_2d)
    #plt.figure(figsize=(8,8),dpi=500    )
    #cluster_members=clustering_results.components_
    #x = distance_df_2d[:,0]
    #y = distance_df_2d[:,1]
    #plt.scatter(xL,yL,c='g',alpha=0.2,label='con+non_set1')
    #plt.scatter(xN,yN,c='purple',alpha=0.2,label='con+non_set2')
    #plt.legend()
    #plt.scatter(
     #   cluster_members[:,0],
      #  cluster_members[:,1],
       # c='r', s=100, alpha=0.1)
    #plt.suptitle(f'{let}', fontsize=16)
    #plt.savefig(f'/home/kali/Desktop/{let}.png')

#def create_plot(  plot_lsoflsL,let ):
    #modelL = MDS(n_components=2,dissimilarity='precomputed', random_state=10)
    #distance_df_2dL=modelL.fit_transform(plot_lsoflsL3)
    #xL = distance_df_2dL[:,0]
    #yL = distance_df_2dL[:,1]
    #dbscan=DBSCAN(eps=0.02,min_samples=50)
    #clustering_results=dbscan.fit(distance_df_2d)
    #plt.figure(figsize=(8,8),dpi=500    )
    #cluster_members=clustering_results.components_
    #x = distance_df_2d[:,0]
    #y = distance_df_2d[:,1]
   # plt.scatter(xL,yL,c='g',alpha=0.2,label='oh fuck')
   # plt.legend()
    #plt.scatter(
     #   cluster_members[:,0],
      #  cluster_members[:,1],
       # c='r', s=100, alpha=0.1)
    #plt.suptitle(f'{let}', fontsize=16)
    #plt.savefig(f'/home/kali/Desktop/{let}.png')

#def create_plot_heatmap(  plot_lsofls  ,let ):
    #plt.figure(figsize=[10,8], dpi=500)
    #sns.heatmap(
    #    plot_lsofls,
    #    xticklabels=False,
    #    yticklabels=False)
    #plt.suptitle(f'{let}', fontsize=16)
    #plt.savefig(f'/home/kali/Desktop/{let}.png')

def extractMAG(path_string):
    mag_milli = []
    csv_files = glob.glob(f"{path_string}/*.csv")
    ne_csv_files = [f for f in csv_files if os.path.getsize(f)>0]
    df=(pd.read_csv(file) for file in ne_csv_files  )
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
    lsofls_delta=[]
    for i in  range(0,len(tseries_list)):
        indiv_delta=[]
        for j in range (0, len(tseries_list[i])-1):
            indiv_delta.append(tseries_list[i][j+1]-tseries_list[i][j])
        lsofls_delta.append(indiv_delta)
    lsofls_acf=[]
    for x in range(0,len(lsofls_delta)):
        lsofls_acf.append(sm.tsa.acf(lsofls_delta[x],nlags=10))
    return lsofls_acf

def multi_thread_wasslist(filt_ornot_maglist,threads):
    list2=filt_ornot_maglist[:]
    list2_copy = list2[:]
    if __name__=='__main__':
        with Pool(threads) as pool:
            lists_to_process = [(list2,list2_copy)]*threads
            resultsSingPerm= pool.starmap(wasserstein_list,lists_to_process)
            wasslistR1P,wasslistR2P=zip(*resultsSingPerm)
    concatenatedR1P = [item for sublist in wasslistR1P for item in sublist]
    concatenatedR2P = [item for sublist in wasslistR2P for item in sublist]
    wass_list_modifiedDoubPerm=[]
    for i in range(0,len(wasslistR1P)):
        wass_list_modifiedDoubPerm.append(concatenatedR1P[i]/statistics.mean(concatenatedR2P))
    return statistics.mean(wass_list_modifiedDoubPerm)

def integration_entropy(y_list4romb):
    ffty1=fft_list_abs_sorted_two_lists(y_list4romb,y_list4romb)
    integral = integrate.trapezoid(ffty1)
    return integral[0]


Milli_paths_QSO = '/sharedata/fastdisk/perkeys/spentropy/data/alberto_QSO'

#----------------------------------------------------------------------------
#These are the lists will all of the magnitudes
full_mag_list_QSO=extractMAG(Milli_paths_QSO)
#-----------------------------------------------------------------------------------
#Id list
QSOIDlist=[]
for _,lstC in full_mag_list_QSO:
    QSOIDlist.append(_)
QSOfilenames = [os.path.basename(path) for path in QSOIDlist]
#------------------------------------------------------------------------------------------
#creation of con non identifier
#con_ones_list = [1] * len(confilenames)
#-------------------------------------------------------------------------------------------------
# Sample Entropy
sampEnt_list_QSO=[]
for _,lstC in full_mag_list_QSO:
    sampEnt_list_QSO.append((nk.entropy_sample(lstC)[0]))
#-----------------------------------------------------------------------------------------------
#Multis-scale Entropy
multEnt_list_QSO=[]
for _,lstC in full_mag_list_QSO:
    lstCarr=np.array(lstC)
    multEnt_list_QSO.append(nk.entropy_multiscale(lstCarr)[0])
#---------------------------------------------------------------------------------------
#Integral fourier coefficient entropy
intEnt_list_QSO=[]
for _,lstC in full_mag_list_QSO:
    intEnt_list_QSO.append(integration_entropy(lstC))
#---------------------------------------------------------------------------------------
#Fourier Coeff WD Normalized
#print('How many FP threads for wasserstein Fcoeff')
num_threads_fcoef=32
FouWassEnt_list_QSO=[]
for _,lstC in full_mag_list_QSO:
    FouWassEnt_list_QSO.append(multi_thread_wasslist(lstC,num_threads_fcoef))
#---------------------------------------------------------------------------------------
#ACF multiscale Entropy
onlymags_l=[]
for _,lstC in full_mag_list_QSO:
    onlymags_l.append(lstC)
    onlymags_acf_l=lsof_acf(onlymags_l)
ACFEnt_list_QSO=[]
for mag_l in onlymags_acf_l:
    ACFEnt_list_QSO.append((nk.entropy_sample(mag_l)[0]))
#-------------------------------------------------------------------------------


# Name of the CSV file
csv_file = '/sharedata/fastdisk/perkeys/spentropy/code'

# Header row
header = ['ID','ACF_ENT','FOURIER_ENT','INTEGRAL_ENT','MULTISCALE_ENT','SAMPLE_ENT']

# Open the CSV file in write mode
with open(csv_file, 'w', newline='') as file:
    writer = csv.writer(file)
    # Write the header row:
    writer.writerow(header)
    # Zip the lists together and iterate over the zipped lists
    for row in zip(QSOfilenames, ACFEnt_list_QSO,FouWassEnt_list_QSO,intEnt_list_QSO,multEnt_list_QSO,sampEnt_list_QSO):
        writer.writerow(row)
#---------------------------------------------------------------------------------------------------------------
#LENGTH SLICING OPTIOINAL SECTION

#filtered_total_range_L=lengthSlicing(full_mag_list_L)
#filtered_total_range_N=lengthSlicing(full_mag_list_N)
#print('There are', len(filtered_total_range_L), 'light curves in this Sampling Range for confirmed-Lenses')
#print('There are', len(filtered_total_range_N), 'light curves in this Sampling Range for non-Lenses')
#-----------------------------------------------------------------------------------------------------------------
#GRAPHING OPTIONAL SECTION

#plt.rcParams['figure.dpi']=300
#fig, ax = plt.subplots()
#fig.suptitle('Confirmed vs Non confirmed Lenses Double Perm Norm')
#sns.kdeplot(nonlense_single_double_perm, ax=ax, bw_method=0.45,color='green', fill=True,label='Non Lensed')#
#sns.kdeplot(lense_single_double_perm, ax=ax, bw_method=0.45,color='blue', fill=True,label='Confirmed Lensed')
#sns.kdeplot(meanQUAD, ax=ax, bw_method=0.15,color='purple', fill=True,label='Quad ')
#ax.legend(bbox_to_anchor=(1.02, 1.02), loc='upper left')
#plt.tight_layout()
#plt.savefig("CN300-400CFFTW.png")
#create_plot(distmatL,'Confirmed Lens MJD Delta ACF WD MDS Distance Matrix D2')
#create_plot_doub(distmatL,distmatN,'Confirmed Non-Lens vs Lens MAG Norm Delta ACF WD MDS')
#create_plot_heatmap(distmatL,'Confirmed Lens MJD Delta ACF WD Distance Matrix ')
#create_plot_heatmap(distmatN,'Confirmed Non-Lens MJD Delta ACF WD Distance Matrix ')
#-------------------------------------------------------------------------------------------------------------------
end = time.time()
print(f"Run time of the program is {end - start} seconds")
print('')
