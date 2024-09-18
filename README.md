# Purpose
This is a python code using an fftw3 wrapper in C to be able to compute a newly defined value of entropy. \
This is can be used on any time series but in this case it is used on light curves within the ZTF catalouge.

# Summary of Code
## C Requirements
In order to be able to run this code you must have **FFTW3**, which is a library to calcualte fourier transforms written in C. The purpose of this is to largely reduce computation time as this code is being used on millions of light curves, the computation time without multithreading can easily reach into time scaled involving weeks.\
For the C file, you can use your choice of compiler, it makes little difference. I personally have used clang and gcc with similar results. Running: 
```bash
sudo pacman -S fftw
or
sudo apt update
sudo apt install libfftw3-dev
```
will properly install fftw3 on your machine. 
**Make sure to compile the C file on your machine as C is architecture dependant**

```bash
gcc -shared -o fftw_wrapper.so -fPIC fftw_wrapper.c -lfftw3
```
This command should work on almost any UNIX system.
Included for brevity are the .so files. These are not necesary for your distro as they will be created when you compile.  

## Python Requirementes 
This code was run using command python3.7. It has not been tested on any other version of python although using python3 should suffice.
The python requirements to install using pip are:
```python
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
```
Install as needed.
## How to Run
To run this code you need full_int_mat_4LAC.py, and fftw_wrapper.c, and a directory containing ZTF csv's. Included are example files. Any other file is used as testing.

### Proper .so naming 
On line 21 you will see 
```python
fftw_lib=ctypes.CDLL('./fftw_wrapper_lach.so')
```
You can change the name of the .so that it fits whatever naming convention suits you. In this example provided of 
```bash
gcc -shared -o fftw_wrapper.so -fPIC fftw_wrapper.c -lfftw3
```
you would change fftw_wrapper.so to fftw_wrapper_lach.so, or whatever name you want, just make sure to change the name in the compile command and on line 21.

### Multi-Threading
On line 187 you should see: 
```python
with Pool(192) as pool:
```
This was ran on a server with 96 cores (192 threads) so, it was multithreaded to optimize computation time. You do not have to multi-thread, as cores I imagine your cores are much more limited than 96. You can simply change 192 to 1 and it will only use one core. You can modify the code to your needs to not include the batch_process at all if you desire.
This part is flexible. 

### Input and Output 
The input that you need will be a directory containing ZTF csv's. It will read to get the columns of ID and the associated light curves, and create a tuple of them. It will create 5 different entropy definitions on the given light curves. Included is 1.5 thousand example light curves.
The FOURIER_ENT is what my new entropy definition is calcualted as. The 5 different entropy calculations are used as post-comparative tools to find potentinal lensed quasar candidates.   

Line 196 contains the path that you will change to your needs:
```python
Milli_paths_QSO = '/sharedata/fastdisk/perkeys/spentropy/data/alberto_QSO'
```


# Summary of Math
See paper : [Using Fourier Coefficients and Wasserstein Distances to Estimate Entropy in Time Series] \
DOI: 10.1109/e-Science58273.2023.10254949

# For Contact
I am currently a Master's student at CSULB (graduating spring 2025) and you can email me at scott.perkey01@student.csulb.edu. \
My other academic email is perkeys@uci.edu. \
These are the main ways to contact me through email.


