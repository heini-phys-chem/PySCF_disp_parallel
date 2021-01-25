# PySCF_disp_parallel
PySCF script to run dispersion scan in parallel

Requirements:
- pyscf (pip install pyscf)
- dftd3 library (https://github.com/cuanto/libdftd3)
- joblib (pip install joblib)

How to use:
- specify the xyz files (python list)
- specify the methods (python list)
- specify number of cores (n_jobs=... in line 160)

then simply:
```
./dispersion_energies.py
```

The energies wil be saved in the file `energies.log`

WARNING: PySCF uses all cores available if not limited using `export OMP_NUM_THREADS=...`
