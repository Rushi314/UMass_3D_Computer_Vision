import numpy as np
import torch
from torch.nn import Sequential as Seq, Linear as Lin, LeakyReLU, GroupNorm
import torch.nn.functional as f


# the "MLP" block that you will use the in the PointNet and CorrNet modules you will implement
# This block is made of a linear transformation (FC layer), 
# followed by a Leaky RelU, a Group Normalization (optional, depending on enable_group_norm)
# the Group Normalization (see Wu and He, "Group Normalization", ECCV 2018) creates groups of 32 channels
def MLP(channels, enable_group_norm=True):
    if enable_group_norm:
        num_groups = [0]
        for i in range(1, len(channels)):
            if channels[i] >= 32:
                num_groups.append(channels[i] // 32)
            else:
                num_groups.append(1)
        return Seq(*[
            Seq(Lin(channels[i - 1], channels[i]), LeakyReLU(negative_slope=0.2), GroupNorm(num_groups[i], channels[i]))
            for i in range(1, len(channels))])
    else:
        return Seq(*[Seq(Lin(channels[i - 1], channels[i]), LeakyReLU(negative_slope=0.2))
                     for i in range(1, len(channels))])


# PointNet module for extracting point descriptors
# num_input_features: number of input raw per-point or per-vertex features 
# 		 			  (should be 3, since we have 3D point positions in this assignment)
# num_output_features: number of output per-point descriptors (should be 32 for this assignment)
# this module should include
# - a MLP that processes each point i into a 128-dimensional vector f_i
# - another MLP that further processes these 128-dimensional vectors into h_i (same number of dimensions)
# - a max-pooling layer that collapses all point features h_i into a global shape representaton g
# - a concat operation that concatenates (f_i, g) to create a new per-point descriptor that stores local+global information
# - a MLP that transforms this concatenated descriptor into the output 32-dimensional descriptor x_i
# **** YOU SHOULD CHANGE THIS MODULE, CURRENTLY IT IS INCORRECT ****
class PointNet(torch.nn.Module):
    def __init__(self, num_input_features, num_output_features):
        super(PointNet, self).__init__()
        self.mlp1 = MLP([num_input_features, 32, 64, 128])
        self.mlp2 = MLP([128, 128])
        # self.max = torch.max_pool2d(kernel_size=)
        self.mlp3 = MLP([256, 128, 64])
        self.linear = torch.nn.Linear(64, num_output_features)

    def forward(self, x):
        f_i = self.mlp1(x)
        h_i = self.mlp2(f_i)
        g, idx = torch.max(h_i, dim=0)  # Max pool to create a 128 dimensional vector
        new = torch.cat(len(f_i)*[g])
        new = torch.reshape(new, (len(f_i), len(g)))
        concat = torch.cat((f_i,new), dim=1)
        f_g_1 = self.mlp3(concat)
        out = self.linear(f_g_1)
        return out


# CorrNet module that serves 2 purposes:  
# (a) uses the PointNet module to extract the per-point descriptors of the point cloud (out_pts)
#     and the same PointNet module to extract the per-vertex descriptors of the mesh (out_vtx)
# (b) if self.train_corrmask=1, it outputs a correspondence mask
# The CorrNet module should
# - include a (shared) PointNet to extract the per-point and per-vertex descriptors 
# - normalize these descriptors to have length one
# - when train_corrmask=1, it should include a MLP that outputs a confidence 
#   that represents whether the mesh vertex i has a correspondence or not
#   Specifically, you should use the cosine similarity to compute a similarity matrix NxM where
#   N is the number of mesh vertices, M is the number of points in the point cloud
#   Each entry encodes the similarity of vertex i with point j
#   Use the similarity matrix to find for each mesh vertex i, its most similar point n[i] in the point cloud 
#   Form a descriptor matrix X = NxF whose each row stores the point descriptor of n[i] (from the point cloud descriptors)
#   Form a vector S = Nx1 whose each entry stores the similarity of the pair (i, n[i])
#   From the PointNet, you also have the descriptor matrix Y = NxF storing the per-vertex descriptors
#   Concatenate [X Y S] into a N x (2F + 1) matrix
#   Transform this matrix into the correspondence mask Nx1 through a MLP followed by a linear transformation
# **** YOU SHOULD CHANGE THIS MODULE, CURRENTLY IT IS INCORRECT ****
class CorrNet(torch.nn.Module):
    def __init__(self, num_output_features, train_corrmask):
        super(CorrNet, self).__init__()
        self.train_corrmask = train_corrmask
        self.pointnet_share = PointNet(3, num_output_features)
        self.mlp = MLP([2*num_output_features+1, 64])  # you won't use this, delete it, this is there just for the code to run
        self.l = torch.nn.Linear(64, 1)


    def forward(self, vtx, pts):
        out_vtx = self.pointnet_share(vtx)
        out_pts = self.pointnet_share(pts)
        norm_vtx = f.normalize(out_vtx, dim=1, p=2)  # L2 Normalization
        norm_pts = f.normalize(out_pts, dim=1, p=2)  # L2 Normalization
        if self.train_corrmask:
            sim_matrix = torch.matmul(norm_vtx,norm_pts.T)
            #print("sim_matrix:: ", sim_matrix.shape)
            pt_s,idx = torch.max(sim_matrix, dim=1)
            pt_s = torch.reshape(pt_s, (len(pt_s),1))
            #print("pts_matrix:: ", pt_s.shape)
            sim_pts = norm_pts[idx,:]
            #print("sim_pts_matrix:: ", sim_pts.shape)
            concat = torch.cat((norm_vtx, sim_pts, pt_s), dim=1)
            #print(concat.shape)
            out = self.mlp(concat)
            out = self.l(out)
            #print("out:shape::: ", out.shape)
            return norm_vtx, norm_pts, out
        else:
            out_corrmask = None
            return norm_vtx, norm_pts, out_corrmask
