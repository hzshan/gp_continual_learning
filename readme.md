## The script below can be used to compress result files from one folder into one file (much much faster for FTP transfer)
```commandline
cd /n/home11/haozheshan/ContinualLearning2022/outputs/
find $foldername | tar -c -f zipped_results.tar.gz -T -
cd ~/
```




Nov 1 2022

## Quick facts about gradient descent simulations
1. The main files to look at are `gradient.py` and `grad_utils.py`
3. 


## Debugging problems with theoretical calculations
1. **Numerical precision**. When computing predictor dynamics using the full GP theory, it is important to use 64-bit 
   precision. 32-bit precision is known to cause inaccuracies and numerical instability in cases.
2. **Make sure that all input vectors are normalized**. To save computation time, all the functions that compute 
   cross kernels do not check whether all input vectors have the same L2 norm, but this is assumed. If there are 
   weird numerical results, always checke whether all input vectors have been normalized. Problems may arise when, 
   for example, two datasets are mixed to generated intermediate-similarity datasets.
