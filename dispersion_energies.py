#!/usr/bin/env python3

import os
import time
import numpy as np

from pyscf import gto, scf, dft, cc, dftd3
from pyscf.cc import ccsd_t

from joblib import Parallel, delayed

#from numba import jit

#_loaderpath = '/home/heinen/workcopies/libdftd3/lib'
_loaderpath = '/home/heinen/anaconda3/lib/python3.8/site-packages/pyscf/lib'
libdftd3 = np.ctypeslib.load_library('libdftd3.so', _loaderpath)

os.system("rm -rf energies.log")

filenames = [
#    'h2_1.xyz', 'h2_2.xyz', 'h2_3.xyz'
'xyz/freq_nm_mode02_00001.xyz',
'xyz/freq_nm_mode02_00002.xyz',
'xyz/freq_nm_mode02_00003.xyz',
'xyz/freq_nm_mode02_00004.xyz',
'xyz/freq_nm_mode02_00005.xyz',
'xyz/freq_nm_mode02_00006.xyz',
'xyz/freq_nm_mode02_00007.xyz',
'xyz/freq_nm_mode02_00008.xyz',
'xyz/freq_nm_mode02_00009.xyz',
'xyz/freq_nm_mode02_00011.xyz',
'xyz/freq_nm_mode02_00012.xyz',
'xyz/freq_nm_mode02_00013.xyz',
'xyz/freq_nm_mode02_00014.xyz',
'xyz/freq_nm_mode02_00015.xyz',
'xyz/freq_nm_mode02_00016.xyz',
'xyz/freq_nm_mode02_00017.xyz',
'xyz/freq_nm_mode02_00018.xyz',
'xyz/freq_nm_mode02_00019.xyz',
'xyz/freq_nm_mode02_00020.xyz',
'xyz/freq_nm_mode02_00021.xyz',
]

methods = ['B3LYP', 'B3LYP-D3', 'PBE0', 'PBE0-D3', 'M06-2X', 'WB97X-V', 'CCSD(T)']

class bcolors:
    BLUE = '\033[94m'
    GREEN = '\033[92m'
    WARNING = '\033[93m'
    FAIL = '\033[91m'
    ENDC = '\033[0m'


def get_mol(f, mol):
    labels = np.array([])
    coords = np.array([])

    lines = open(f, 'r').readlines()
    numAtoms = int(lines[0])

    for i, line in enumerate(lines):
        if i == 0 or i == 1: continue
        tokens = line.split()

        label = tokens[0]
        x     = float(tokens[1])
        y     = float(tokens[2])
        z     = float(tokens[3])

        labels = np.append(labels, label)
        coords = np.append(coords, [x, y, z])

    coords = coords.reshape(numAtoms,3)

    mol.atom     = [ [  labels[0], (coords[0][0], coords[0][1], coords[0][2]) ] ]
    mol.atom.extend( [ [labels[i], (coords[i][0], coords[i][1], coords[i][2])] for i in range(1,len(labels))]  )

def get_basis(mol):
    mol.basis = 'def2tzvpri'

def get_dft(mol, method):
    mf = dft.RKS(mol)
    mf.xc = method

    return mf

def get_dft_d3(mol, method):
    method = method[:-3]
    mf = dftd3.dftd3(dft.RKS(mol, xc=method))

    return mf


def get_cc(mol, method):
    mf = scf.RHF(mol).run()
    HF_energy = mf.kernel()
    ccsd = cc.CCSD(mf)

    return ccsd, HF_energy

def build_molecule(filename):
    mol = gto.Mole()

    # read in molecule & define basis set
    get_mol(filename, mol)
    get_basis(mol)

    # make molecule
    mol.build()

    mol.verbose = 0

    return mol

def run_calculation(method, filename, energies):

   print("{}[-]{} Starting {} calculation for molecule {}".format(bcolors.BLUE, bcolors.ENDC, method, filename))

   # create molecule
   mol = build_molecule(filename)

   # set up & run calculations
   if "B3LYP-D3" in method:
       mf     = get_dft_d3(mol, method)
       energy = mf.kernel()
   elif "PBE0-D3" in method:
       mf     = get_dft_d3(mol, method)
       energy = mf.kernel()
   elif "B3LYP" in method:
       mf     = get_dft(mol, method)
       energy = mf.kernel()
   elif "PBE0" in method:
       mf     = get_dft(mol, method)
       energy = mf.kernel()
   elif "M06-2X" in method:
       mf     = get_dft(mol, method)
       energy = mf.kernel()
   elif "WB97X-V" in method:
       mf     = get_dft(mol, method)
       energy = mf.kernel()
   elif "CCSD" in method:
       ccsd, HF_energy = get_cc(mol, method)
       calc            = ccsd.kernel()
       ccsd_energy     = ccsd.energy()
       ccsd_t_energy   = ccsd_t.kernel(ccsd, ccsd.ao2mo(), verbose=0)
       energy = ccsd_t_energy + ccsd_energy + HF_energy
   else:
       print("{}Method {} not encoded{}".format(bcolors.WARNING, method, bcolors.ENDC))
       exit(1)

   print("{}[+]{} {} calculation for molecule {} finished".format(bcolors.GREEN, bcolors.ENDC, method, filename))

   return [filename, energy]

if __name__ == '__main__':
    energies = dict()

    for method in methods:
        energies[method] = []
        energies[method].append(Parallel(n_jobs=20)(delayed(run_calculation)(method, filename, energies) for filename in filenames))

    print()

    fout = open("energies.log", 'w')
    for key, values in energies.items():
        fout.write("{}\t{}\n".format(key, *values))
    fout.close()

