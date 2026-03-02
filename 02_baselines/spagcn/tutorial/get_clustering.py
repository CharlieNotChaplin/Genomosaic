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

# ---------------------------------------------
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

# ----------------------------------------------

#adj=np.loadtxt(os.path.join(local_data_path, 'adj.csv'), delimiter=',')
adj = np.load(os.path.join(local_data_path,'adj.npy'), mmap_mode='r')
adata = sc.read_h5ad(os.path.join(local_data_path, "nuclei_centroids.h5ad"))
adata.var_names_make_unique()
spg.prefilter_genes(adata,min_cells=3) # avoiding all genes are zeros
spg.prefilter_specialgenes(adata)

r_seed=t_seed=n_seed=100
n_clusters=7 

# load l, res, set p 
p=0.5
with open(os.path.join(local_data_path, "l.txt"), 'r') as f1:
    l = float(f1.read().strip())
with open(os.path.join(local_data_path, f"res_cluster_{n_clusters}.txt"), 'r') as f2:
    res = float(f2.read().strip())


####

clf=spg.SpaGCN()
clf.set_l(l)
#Set seed
random.seed(r_seed)
torch.manual_seed(t_seed)
np.random.seed(n_seed)
#Run
clf.train(adata,adj,init_spa=True,init="louvain",res=res, tol=5e-3, lr=0.05, max_epochs=200)
y_pred, prob=clf.predict()
adata.obs["pred"]= y_pred
adata.obs["pred"]=adata.obs["pred"].astype('category')
#Do cluster refinement(optional)
#shape="hexagon" for Visium data, "square" for ST data.
x_array=adata.obs["x_array"].tolist()
y_array=adata.obs["y_array"].tolist()
adj_2d=spg.calculate_adj_matrix(x=x_array,y=y_array, histology=False)
refined_pred=spg.refine(sample_id=adata.obs.index.tolist(), pred=adata.obs["pred"].tolist(), dis=adj_2d, shape="hexagon")
adata.obs["refined_pred"]=refined_pred
adata.obs["refined_pred"]=adata.obs["refined_pred"].astype('category')

adata.write_h5ad("temp.h5ad", compression="gzip")  # write temporarily
import os
print(f"Resulting file size of the anndata object: {os.path.getsize('temp.h5ad')/1e6:.2f} MB")
os.remove("temp.h5ad")

#Save results
adata.write_h5ad(os.path.join(results_path, f"results_cluster_{n_clusters}.h5ad"))