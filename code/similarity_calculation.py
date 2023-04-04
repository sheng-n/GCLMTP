import numpy as np
import copy
import pandas as pd

'''Recalculating similarity'''
def Preproces_Data(A, test_id):

    copy_A = A / 1
    for i in range(test_id.shape[0]):
        copy_A[int(test_id[i][0])][int(test_id[i][1])] = 0
    return copy_A

'''GIPK '''
def calculate_kernel_bandwidth(A):
    IP_0 = 0
    for i in range(A.shape[0]):
        IP = np.square(np.linalg.norm(A[i]))
        # print(IP)
        IP_0 += IP
    lambd = 1/((1/A.shape[0]) * IP_0)
    return lambd

def calculate_GaussianKernel_sim(A):

    kernel_bandwidth = calculate_kernel_bandwidth(A)
    gauss_kernel_sim = np.zeros((A.shape[0],A.shape[0]))
    for i in range(A.shape[0]):
        for j in range(A.shape[0]):
            gaussianKernel = np.exp(-kernel_bandwidth * np.square(np.linalg.norm(A[i] - A[j])))
            gauss_kernel_sim[i][j] = gaussianKernel

    return gauss_kernel_sim



'''Functional similarity'''
def PBPA(RNA_i, RNA_j, di_sim, rna_di):
    diseaseSet_i = rna_di[RNA_i] > 0

    diseaseSet_j = rna_di[RNA_j] > 0
    diseaseSim_ij = di_sim[diseaseSet_i][:, diseaseSet_j]
    ijshape = diseaseSim_ij.shape
    if ijshape[0] == 0 or ijshape[1] == 0:
        return 0
    return (sum(np.max(diseaseSim_ij, axis=0)) + sum(np.max(diseaseSim_ij, axis=1))) / (ijshape[0] + ijshape[1])


def getRNA_functional_sim(RNAlen, diSiNet, rna_di):
    RNASiNet = np.zeros((RNAlen, RNAlen))
    for i in range(RNAlen):
        for j in range(i + 1, RNAlen):
            RNASiNet[i, j] = RNASiNet[j, i] = PBPA(i, j, diSiNet, rna_di)
    RNASiNet = RNASiNet + np.eye(RNAlen)
    return RNASiNet


def RNA_fusion_sim (G1, G2, F):
    fusion_sim = np.zeros((len(G1),len(G2)))
    G = (G1+G2)/2
    for i in range (len(G1)):
        for j in range(len(G1)):
            if F[i][j] > 0 :
                fusion_sim[i][j] = F[i][j]
            else:
                fusion_sim[i][j] = G[i][j]
    return fusion_sim

def dis_fusion_sim (G1, G2, SD):

    D_fuse = (SD + (G1 + G2)/2)/2

    return D_fuse








