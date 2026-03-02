import os,csv,re
import pandas as pd
import numpy as np
import scanpy as sc
import math
import SpaGCN as spg
from scipy.sparse import issparse
import random, torch
import warnings
warnings.filterwarnings("ignore")
import matplotlib.colors as clr
import matplotlib.pyplot as plt
import SpaGCN as spg
#In order to read in image data, we need to install some package. Here we recommend package "opencv"
#inatll opencv in python
#!pip3 install opencv-python
import cv2

# Paths and variables
person = "CRC_P2"
bin_size = "002um"
os.makedirs(f"../data/{person}", exist_ok=True)
local_data_path = f"../data/{person}"

genomosaic_input_path = f"../../../data/{person}/input"
genomosaic_output_path = f"../../../data/{person}/output/binned_outputs/square_{bin_size}/spatial"
cell_seg_path = f"../../../Cell_Segmentation/output/{person}/{bin_size}"

imagepath = os.path.join(genomosaic_input_path, "Visium_HD_Human_Colon_Cancer_P2_tissue_image.btf")
os.makedirs(f"../results/{person}", exist_ok=True)
results_path = f"../results/{person}"

#adj=np.loadtxt(os.path.join(local_data_path, 'adj.csv'), delimiter=',')
adj = np.load(os.path.join(local_data_path,'adj.npy'), mmap_mode='r')
#adj = sparse.csr_matrix(adj) # make sparse to save mem
adata = sc.read_h5ad(os.path.join(local_data_path, "nuclei_centroids.h5ad"))
adata.var_names_make_unique()
spg.prefilter_genes(adata,min_cells=3) # avoiding all genes are zeros
spg.prefilter_specialgenes(adata)

print("Sanity Check - Adjacency Stats: ")
print("min:", np.nanmin(adj))
print("max:", np.nanmax(adj))
print("has NaN:", np.isnan(adj).any())
print("has inf:", np.isinf(adj).any())

p=0.5
with open(os.path.join(local_data_path, "l.txt"), 'r') as f:
    l = float(f.read().strip())

#If the number of clusters known, we can use the spg.search_res() fnction to search for suitable resolution(optional)
#For this toy data, we set the number of clusters=7 since this tissue has 7 layers
n_clusters=7 # based on Web Summary CRC P1
#Set seed
r_seed=t_seed=n_seed=100
#Search for suitable resolution

print("l:", l)
print("type:", type(l))
try:
    arr = np.asarray(l)
    print("np.shape:", arr.shape)
    print("np.dtype:", arr.dtype)
    print("isnan (any):", np.isnan(arr).any())
    print("as float:", float(arr.ravel()[0]))
except Exception as e:
    print("could not coerce l to float:", e)

res=spg.search_res(adata, adj, l, n_clusters, start=0.7, step=0.1, tol=5e-3, lr=0.05, max_epochs=20, r_seed=r_seed, t_seed=t_seed, n_seed=n_seed)

file_path = f'../data/{person}/res_cluster_{n_clusters}.txt'
with open(file_path, 'w') as file:
    file.write(str(res))
