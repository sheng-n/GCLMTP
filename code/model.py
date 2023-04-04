import numpy as np
import torch
import torch.nn as nn
import pandas as pd


'''Read Dataset1'''
def read_file1():

    '''one fold is used here as an example'''
    lnc_mi = np.loadtxt("data/yuguoxian_lnc_mi.txt")
    mi_lnc = lnc_mi.T
    lnc_dis = np.loadtxt("data/lnc_dis_association.txt")
    mi_dis = np.loadtxt("data/mi_dis.txt")
    dis_sim = np.loadtxt("data/dis_fusion_sim.txt")
    lnc_sim = np.loadtxt("data/lnc_fusion_sim.txt")
    mi_sim = np.loadtxt("data/mi_fusion_sim.txt")
    lnc_dis_test_id = np.loadtxt("data/lnc_dis_test_id1.txt")
    mi_dis_test_id = np.loadtxt("data/mi_dis_test_id1.txt")
    mi_lnc_test_id = np.loadtxt("data/mi_lnc_test_id1.txt")
    return mi_lnc, lnc_dis, mi_dis, dis_sim, lnc_sim, mi_sim, lnc_dis_test_id, mi_dis_test_id, mi_lnc_test_id

'''Read dataset2'''
def read_file2():

    '''one fold is used here as an example'''

    di_lnc = pd.read_csv('dataset2/di_lnc_intersection.csv', index_col='Unnamed: 0')
    di_mi = pd.read_csv('dataset2/di_mi_intersection.csv', index_col='Unnamed: 0')
    mi_lnc = pd.read_csv('dataset2/mi_lnc_intersection.csv', index_col='Unnamed: 0')

    lnc_dis = di_lnc.values.T
    mi_dis = di_mi.values.T
    mi_lnc = mi_lnc.values

    dis_sim = np.loadtxt("dataset2/dis_fusion_sim.txt")
    lnc_sim = np.loadtxt("dataset2/lnc_fusion_sim.txt")
    mi_sim = np.loadtxt("dataset2/mi_fusion_sim.txt")

    lnc_dis_test_id = np.loadtxt("dataset2/lnc_dis_test_id1.txt")
    mi_dis_test_id = np.loadtxt("dataset2/mi_dis_test_id1.txt")
    mi_lnc_test_id = np.loadtxt("dataset2/mi_lnc_test_id1.txt")        #
    return mi_lnc, lnc_dis, mi_dis, dis_sim, lnc_sim, mi_sim, lnc_dis_test_id, mi_dis_test_id, mi_lnc_test_id

'''Zeroing of positive samples in the test set'''
def Preproces_Data (A, test_id):
    copy_A = A / 1
    for i in range(test_id.shape[0]):
        copy_A[int(test_id[i][0])][int(test_id[i][1])] = 0
    return copy_A

'''Constructing adjacency matrix'''
def construct_graph(lncRNA_disease,  miRNA_disease, miRNA_lncRNA, lncRNA_sim, miRNA_sim, disease_sim ):
    lnc_dis_sim = np.hstack((lncRNA_sim, lncRNA_disease, miRNA_lncRNA.T))

    dis_lnc_sim = np.hstack((lncRNA_disease.T,disease_sim, miRNA_disease.T))

    mi_lnc_dis = np.hstack((miRNA_lncRNA,miRNA_disease,miRNA_sim))

    matrix_A = np.vstack((lnc_dis_sim,dis_lnc_sim,mi_lnc_dis))          #
    return matrix_A

'''Normalisation'''
def lalacians_norm(adj):
    # adj += np.eye(adj.shape[0])
    degree = np.array(adj.sum(1))
    D = []
    for i in range(len(degree)):
        if degree[i] != 0:
            de = np.power(degree[i], -0.5)
            D.append(de)
        else:
            D.append(0)
    degree = np.diag(np.array(D))
    norm_A = degree.dot(adj).dot(degree)
    return norm_A

'''gcn'''
class GCNConv(nn.Module):
    def __init__(self, in_size, out_size,):
        super(GCNConv, self).__init__()
        self.in_size = in_size
        self.out_size = out_size
        self.weight = nn.Parameter(torch.FloatTensor(in_size, out_size))
        nn.init.xavier_uniform_(self.weight.data, gain=1.414)

    def forward(self, adj, features):
       out = torch.mm(adj, features)      # A*X，
       out = torch.mm(out,self.weight)    # A*X*W
       return out

'''Encoder'''
class Encoder(nn.Module):
    def __init__(self, input_dim, hidden_dim, n_calss):
        super(Encoder, self).__init__()
        self.gcn1 = GCNConv(input_dim, hidden_dim)
        self.prelu1 = nn.PReLU(hidden_dim)

        self.gcn2 = GCNConv(hidden_dim,n_calss)
        self.prelu2 = nn.PReLU(n_calss)

        self.last_linear = torch.nn.Linear(hidden_dim + n_calss, n_calss)

    def forward(self, x, adj, corrupt=True):
        if corrupt:
            perm = torch.randperm(x.shape[0])
            x = x[perm]

        x1 = self.gcn1(adj, x)
        x1 = self.prelu1(x1)
        x2 = self.gcn2(adj,x1)
        x2 = self.prelu2(x2)

        return x2

'''Discriminator'''
class Discriminator(nn.Module):
    def __init__(self, hidden_dim):
        super(Discriminator, self).__init__()
        self.weight = nn.Parameter(torch.Tensor(hidden_dim, hidden_dim))
        nn.init.xavier_uniform_(self.weight.data, gain=1.414)

    def forward(self, x, summary):
        x = torch.matmul(x, torch.matmul(self.weight, summary))
        return x

'''Model'''

class GCLMTP(nn.Module):
    def __init__(self, input_dim, hidden_dim,output_dim):
        super(GCLMTP, self).__init__()
        self.encoder = Encoder(input_dim, hidden_dim,output_dim)
        self.discriminator = Discriminator(output_dim)
        self.loss = nn.BCEWithLogitsLoss()

    def forward(self, edge_index, x ):
        positive = self.encoder(x, edge_index, corrupt=False)
        negative = self.encoder(x, edge_index, corrupt=True)

        summary = torch.sigmoid(positive.mean(dim=0))

        # print("summary: ",summary.shape)
        positive_D = self.discriminator(positive, summary)
        negative_D = self.discriminator(negative, summary)

        l1 = self.loss(positive_D, torch.ones_like(positive_D))
        l2 = self.loss(negative_D, torch.zeros_like(negative_D))

        L = l1 + l2
        return L, positive


def train(la_A, Epoch, in_features, N_HID, out_features, LR):

    G = GCLMTP(input_dim=in_features, hidden_dim=N_HID, output_dim=out_features)
    G_optimizer = torch.optim.Adam(G.parameters(), lr=LR)
    A_laplacians = torch.from_numpy(la_A).float()
    X = torch.from_numpy(la_A).float()
    # print(A_laplacians.shape,X.shape)
    ################################GPU#############################
    if torch.cuda.is_available():
        G = G.cuda()

        A_laplacians = A_laplacians.cuda()
        X = X.cuda()

    ###########################model train###########################
    for epoch in range(Epoch):
        G_loss, embedding = G(A_laplacians, X)

        G_optimizer.zero_grad()
        G_loss.backward()
        G_optimizer.step()
        print('Epoch: ', epoch, '| train G_loss: %.10f' % G_loss.item())
    # np.savetxt("dataset1_result/low_A_256.txt", embedding.detach().cpu().numpy())
    # np.savetxt("dataset2_result/low_A_256.txt", embedding.detach().cpu().numpy())


if __name__ == '__main__':

    #Hyperparameters
    Epoch = 500
    in_features = 1140
    # in_features = 1276  # num_lnc + num_dis + num_mi
    N_HID = 512
    out_features = 256
    LR = 0.0001

    mi_lnc, lnc_dis, mi_dis, dis_sim, lnc_sim, mi_sim, lnc_dis_test_id, mi_dis_test_id, lnc_mi_test_id = read_file1()
    # mi_lnc, lnc_dis, mi_dis, dis_sim, lnc_sim, mi_sim, lnc_dis_test_id, mi_dis_test_id, lnc_mi_test_id = read_file2()

    lnc_dis = Preproces_Data(lnc_dis,lnc_dis_test_id)
    mi_dis = Preproces_Data(mi_dis,mi_dis_test_id)
    mi_lnc = Preproces_Data(mi_lnc,lnc_mi_test_id)

    matrix_A = construct_graph(lnc_dis,mi_dis,mi_lnc,lnc_sim,mi_sim,dis_sim)
    # print(matrix_A.shape)
    la_A = lalacians_norm(matrix_A)
    # print(la_A.shape)
    train(la_A, Epoch, in_features, N_HID, out_features, LR)



