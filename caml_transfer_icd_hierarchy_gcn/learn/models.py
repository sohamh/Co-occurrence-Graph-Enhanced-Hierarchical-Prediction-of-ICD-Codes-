"""
    Holds PyTorch models
"""
from gensim.models import KeyedVectors
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.init import xavier_uniform_ as xavier_uniform
from torch.autograd import Variable
import pickle
from icdcodex import icd2vec, hierarchy
from torch_geometric.nn import Sequential, GCNConv, GATConv
import math

from datasets import diff

import numpy as np

from math import floor
import random
import sys
import time
from scipy.spatial import distance

from constants import *
from dataproc import extract_wvs
from tools import get_target
# -*- coding: utf-8 -*-
import networkx as nx
from tools import normalize_adjacency_matrix

# from __future__ import absolute_import
# from __future__ import unicode_literals
# from __future__ import division
# from __future__ import print_function

import math
import torch
import torch.nn as nn
import torch.nn.init as init
from torch.nn.parameter import Parameter
from torch.nn.modules.module import Module
import torch.nn.functional as F
from sklearn.metrics.pairwise import cosine_similarity
# from torch_sparse import spmm
# from utils import *
from torch_geometric.data import Data, Batch
from itertools import combinations
import os
os.environ["CUDA_VISIBLE_DEVICES"] = str(1)

class BaseModel(nn.Module):

    def __init__(self, Y, embed_file, dicts, lmbda=0, dropout=0.5, gpu=True, embed_size=100):
        super(BaseModel, self).__init__()
        torch.manual_seed(1337)
        self.gpu = gpu
        self.Y = Y
        self.embed_size = embed_size
        self.embed_drop = nn.Dropout(p=dropout)
        self.lmbda = lmbda

        # make embedding layer
        if embed_file:
            print("loading pretrained embeddings...")
            W = torch.Tensor(extract_wvs.load_embeddings(embed_file))

            self.embed = nn.Embedding(W.size()[0], W.size()[1], padding_idx=0)
            self.embed.weight.data = W.clone()
        else:
            # add 2 to include UNK and PAD
            vocab_size = len(dicts['ind2w'])
            self.embed = nn.Embedding(vocab_size + 2, embed_size, padding_idx=0)

    def _get_loss(self, y4t,y4, y3,y2,y1, target4, target3,target2,target1, diffs=None):
        # calculate the BCE
        loss4t = F.binary_cross_entropy_with_logits(y4t + 1e-10, target4)
        # loss4t=0
        loss4 = F.binary_cross_entropy_with_logits(y4 + 1e-10, target4)
        # loss3 = F.binary_cross_entropy_with_logits(y3 + 1e-10, target3)
        # loss2 = F.binary_cross_entropy_with_logits(y2 + 1e-10, target2)
        # loss1 = F.binary_cross_entropy_with_logits(y1 + 1e-10, target1)
        # reg=0.01*att_w-dist
        # loss=loss4t+loss4+0.1*loss3+0.001*loss2+0.0001*loss1+reg
        loss = loss4t + loss4 #+ 0.1 * loss3 + 0.001 * loss2 #+ reg


        # add description regularization loss if relevant
        if self.lmbda > 0 and diffs is not None:
            diff = torch.stack(diffs).mean()
            loss = loss + diff
        return loss

    def embed_descriptions(self, desc_data, gpu):
        # label description embedding via convolutional layer
        # number of labels is inconsistent across instances, so have to iterate over the batch
        b_batch = []
        for inst in desc_data:
            if len(inst) > 0:
                if gpu:
                    lt = Variable(torch.cuda.LongTensor(inst))
                else:
                    lt = Variable(torch.LongTensor(inst))
                d = self.desc_embedding(lt)
                d = d.transpose(1, 2)
                d = self.label_conv(d)
                d = F.max_pool1d(torch.tanh(d), kernel_size=d.size()[2])
                d = d.squeeze(2)
                b_inst = self.label_fc1(d)
                b_batch.append(b_inst)
            else:
                b_batch.append([])
        return b_batch

    def _compare_label_embeddings(self, target, b_batch, desc_data):


        # description regularization loss
        # b is the embedding from description conv
        # iterate over batch because each instance has different # labels
        diffs = []
        for i, bi in enumerate(b_batch):
            ti = target[i]
            inds = torch.nonzero(ti.data).squeeze().cpu().numpy()

            zi = self.final.weight[inds, :]

            diff = (zi - bi).mul(zi - bi).mean()

            # multiply by number of labels to make sure overall mean is balanced with regard to number of labels
            diffs.append(self.lmbda * diff * bi.size()[0])
        return diffs

    def _compare_label_ont_embeddings(self, target, b_batch, desc_data,gamma=0.0):


        # description regularization loss
        # b is the embedding from description conv
        # iterate over batch because each instance has different # labels
        diffs = []
        for i, bi in enumerate(b_batch):
            ti = target[i]
            inds = torch.nonzero(ti.data).squeeze().cpu().numpy()

            zi = self.final.weight[inds, :]
            ##### SOHA
            # di = self.emb[inds, :]
            # diff = (zi - di).mul(zi - di).mean()
            ##### SOHA

            diff = (zi - bi).mul(zi - bi).mean()

            # multiply by number of labels to make sure overall mean is balanced with regard to number of labels
            diffs.append(gamma * diff * bi.size()[0])
        return diffs


class BOWPool(BaseModel):
    """
        Logistic regression model over average or max-pooled word vector input
    """

    def __init__(self, Y, embed_file, lmbda, gpu, dicts, pool='max', embed_size=100, dropout=0.5, code_emb=None):
        super(BOWPool, self).__init__(Y, embed_file, dicts, lmbda, dropout=dropout, gpu=gpu, embed_size=embed_size)
        self.final = nn.Linear(embed_size, Y)
        if code_emb:
            self._code_emb_init(code_emb, dicts)
        else:
            xavier_uniform(self.final.weight)
        self.pool = pool

    def _code_emb_init(self, code_emb, dicts):
        code_embs = KeyedVectors.load_word2vec_format(code_emb)
        weights = np.zeros(self.final.weight.size())
        for i in range(self.Y):
            code = dicts['ind2c'][i]
            weights[i] = code_embs[code]
        self.final.weight.data = torch.Tensor(weights).clone()

    def forward(self, x, target, desc_data=None, get_attention=False):
        # get embeddings and apply dropout
        x = self.embed(x)
        x = self.embed_drop(x)
        x = x.transpose(1, 2)
        if self.pool == 'max':
            import pdb;
            pdb.set_trace()
            x = F.max_pool1d(x)
        else:
            x = F.avg_pool1d(x)
        logits = F.sigmoid(self.final(x))
        loss = self._get_loss(logits, target, diffs)
        return yhat, loss, None


class GraphConvLayer(Module):
    def __init__(self, in_features, out_features, bias=True):
        super(GraphConvLayer, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight = Parameter(torch.Tensor(in_features, out_features))
        init.xavier_uniform_(self.weight)

        if bias:
            self.bias = Parameter(torch.Tensor(out_features))
            stdv = 1. / math.sqrt(self.bias.size(0))
            self.bias.data.uniform_(-stdv, stdv)
        else:
            self.register_parameter('bias', None)

    def forward(self, feature, adj):
        support = torch.matmul(feature, self.weight)
        output = torch.matmul(adj.float(), support)

        if self.bias is not None:
            return output + self.bias
        else:
            return output

    def __repr__(self):
        return self.__class__.__name__ + ' (' \
               + str(self.in_features) + ' -> ' \
               + str(self.out_features) + ')'


class ConvAttnPool(BaseModel):

    def __init__(self, Y, embed_file, kernel_size, num_filter_maps, lmbda, gpu, dicts, ontology=False, embed_size=100,
                 dropout=0.5, code_emb=None, transfer=False, transfer_10=False,cooccurence_graph=None):
        super(ConvAttnPool, self).__init__(Y, embed_file, dicts, lmbda, dropout=dropout, gpu=gpu, embed_size=embed_size)

        # initialize conv layer as in 2.1
        self.dicts=dicts
        self.conv = nn.Conv1d(self.embed_size, num_filter_maps, kernel_size=kernel_size,
                              padding=int(floor(kernel_size / 2)))
        xavier_uniform(self.conv.weight)

        ###### define Y

        # self.Y1 = max(dicts['L1'].values()) + 1
        #
        # self.Y2 = max(dicts['L2'].values()) + 1
        #
        # self.Y3 = max(dicts['L3'].values()) + 1

        self.Y4 = len(dicts['ind2c'])

        # context vectors for computing attention as in 2.2
        self.U4 = nn.Linear(num_filter_maps, self.Y4)
        # self.U3 = nn.Linear(num_filter_maps, self.Y3)
        # self.U2 = nn.Linear(num_filter_maps, self.Y2)
        # self.U1 = nn.Linear(num_filter_maps, self.Y1)

        if (ontology):
            # Define your layers
            self.graph_conv_1 = GraphConvLayer(num_filter_maps*1, num_filter_maps*1)#GCNConv(num_filter_maps*4, num_filter_maps*4)#GATConv(num_filter_maps*4, num_filter_maps*4)##
            # self.graph_conv_2 = GraphConvLayer(num_filter_maps * 3, num_filter_maps * 3)
            # self.OA = nn.Linear(num_filter_maps, Y)

        # xavier_uniform(self.U1.weight)
        # xavier_uniform(self.U2.weight)
        # xavier_uniform(self.U3.weight)
        xavier_uniform(self.U4.weight)

        ###Soha first lear how to lower the dimension from num_filter_maps to 1
        # self.reduction = nn.Linear(2*num_filter_maps, 1).cuda()
        # final layer: create a matrix to use for the L binary classifiers as in 2.3
        self.final4t = nn.Linear(num_filter_maps*1 , self.Y4)
        self.final4 = nn.Linear(num_filter_maps * 2, self.Y4)
        # self.final3 = nn.Linear(num_filter_maps*2, self.Y3)
        # self.final3 = nn.Linear(num_filter_maps, self.Y3)
        # self.final2 = nn.Linear(num_filter_maps*1, self.Y2)
        # self.final1 = nn.Linear(num_filter_maps, self.Y1)

        # self.final = nn.Linear(Y, Y)
        # xavier_uniform(self.final1.weight)
        # xavier_uniform(self.final2.weight)
        # xavier_uniform(self.final3.weight)
        xavier_uniform(self.final4.weight)

        # initialize with trained code embeddings if applicable
        if code_emb:
            self._code_emb_init(code_emb, dicts)
            # also set conv weights to do sum of inputs
            weights = torch.eye(self.embed_size).unsqueeze(2).expand(-1, -1, kernel_size) / kernel_size
            self.conv.weight.data = weights.clone()
            self.conv.bias.data.zero_()

        # conv for label descriptions as in 2.5
        # description module has its own embedding and convolution layers
        if lmbda > 0:
            W = self.embed.weight.data
            self.desc_embedding = nn.Embedding(W.size()[0], W.size()[1], padding_idx=0)
            self.desc_embedding.weight.data = W.clone()

            self.label_conv = nn.Conv1d(self.embed_size, num_filter_maps, kernel_size=kernel_size,
                                        padding=int(floor(kernel_size / 2)))
            xavier_uniform(self.label_conv.weight)

            self.label_fc1 = nn.Linear(num_filter_maps, num_filter_maps)
            xavier_uniform(self.label_fc1.weight)

        ### Soha
        self.ontology = ontology
        if (self.ontology == True ):

            # if (transfer == True):
            #     if (transfer_10):
            #         icd10 = self.covert_icd9_to_10(dicts['c2ind'])
            #         emb = torch.from_numpy(self.ont_emb(icd10, version=10))
            #     else:
            #         emb = torch.from_numpy(self.ont_emb(dicts['c2ind'].keys(), version=9))
            # else:
            #     emb = torch.from_numpy(self.ont_emb(dicts['c2ind'].keys(), version=9))
            #     # self.emb= torch.from_numpy(data['embedding'])
            # emb = emb.to("cuda:0")
            # emb.requires_grad = False
            # emb = torch.nn.functional.normalize(emb, p=2.0, dim=0)
            # # self.dist = torch.from_numpy(distance.cdist(self.emb.cpu(), self.emb.cpu(), metric='euclidean')).cuda()
            # # self.dist= torch.from_numpy(cosine_similarity(emb.cpu())).cuda()
            # with open('/home/mahdi/codes/caml_transfer_icd_hierarchy_gcn/ontology_dist.pkl', 'rb') as file:
            #     self.dist = (8-pickle.load(file)).cuda()
            self.dist= normalize_adjacency_matrix(torch.from_numpy(nx.to_numpy_matrix(cooccurence_graph)).float()).cuda()
            self.dist.requires_grad = False
            # self.dist=torch.nn.functional.normalize(self.dist, p=2.0, dim=0)


        # else:
        #     emb = torch.from_numpy(self.ont_emb(dicts['c2ind'].keys(), version=9))
        #     emb = emb.to("cuda:0")
        #     emb.requires_grad = False
        #     self.emb = torch.nn.functional.normalize(emb, p=2.0, dim=0)
        #     # self.U.weight.data = torch.Tensor(emb).clone()
        #     # self.ontweight = torch.rand()

    def covert_icd9_to_10(self, icd9_dict):
        #### now we use the saved dictionary to convert ICD 9 to ICD 10
        with open('/mapping/GEM/diagnosis_gems_2018/SOHA-GEMS9to10.pkl',
                  "rb") as f:  # Python 3: open(..., 'rb')
            dict3, label = pickle.load(f)

        d = list(icd9_dict.keys())
        icd9 = [s.replace('.', '') for s in d]
        count = 0
        icd10 = []
        for key in icd9:
            val = dict3.get(key)
            if (val is None):
                count = count + 1
            else:
                icd10.append(val)
        return icd10

    def ont_emb(self, codes, version=9):
        d = list(codes)
        codes = [s.replace('.', '') for s in d]
        embedder = icd2vec.Icd2Vec(num_embedding_dimensions=50, workers=-1)
        if (version == 9):
            embedder.fit(*hierarchy.icd9())
            codes_of_interest = codes  # ["0330", "0340", "9101"]
            embedding = embedder.to_vec(codes_of_interest)
        else:
            #     embedder.fit(*hierarchy.icd10cm("2020"))
            with open("/material/embedding_ind10.pkl",
                      "rb") as fp:  # Unpickling
                data10 = pickle.load(fp)
            embedding = data10['embedding']

        # G = hierarchy.icd9()
        # tree = G[0].to_undirected()
        # leaf_nodes = codes_of_interest
        # subgraph = tree.subgraph(leaf_nodes)
        # dist_matrix = [[0] * len(leaf_nodes) for _ in range(len(leaf_nodes))]
        # for i, node1 in enumerate(leaf_nodes):
        #     for j, node2 in enumerate(leaf_nodes):
        #         if i != j:
        #             dist = nx.shortest_path_length(tree, node1, node2)
        #             dist_matrix[i][j] = dist

        icd_9_hierarchy, icd_9_codes = hierarchy.icd9()
        # icd_10_cm_hierarchy, icd_10_cm_codes = hierarchy.icd10cm("2020")
        G = nx.relabel_nodes(icd_9_hierarchy, {"root": "ICD-9"})
        # X = dfs_predecessors(G)
        G = hierarchy.icd9()
        tree = G[0]
        # x=[]
        # for i,node in enumerate(codes_of_interest):
        #     x.append({node: nx.predecessor(tree,node)})
        T = nx.dfs_tree(tree, 'root')

        leaf_nodes = codes_of_interest
        dist_matrix = [[0] * len(leaf_nodes) for _ in range(len(leaf_nodes))]

        paths = {}
        for node in leaf_nodes:
            paths[node] = nx.shortest_path(tree.reverse(), node, 'root')
        for i, node1 in enumerate(leaf_nodes):
            d1 = paths[node1]
            for j, node2 in enumerate(leaf_nodes):
                d2 = paths[node2]
                common_element_sum = -1  # Default value for no common element found
                for index1, element1 in enumerate(d1):
                    if element1 in d2:
                        index2 = d2.index(element1)
                        common_element_sum = index1 + index2
                        break
                dist_matrix[i][j] = common_element_sum

        return embedding

    def _code_emb_init(self, code_emb, dicts):
        code_embs = KeyedVectors.load_word2vec_format(code_emb)
        weights = np.zeros(self.final.weight.size())
        for i in range(self.Y):
            code = dicts['ind2c'][i]
            weights[i] = code_embs[code]
        self.U.weight.data = torch.Tensor(weights).clone()
        self.final.weight.data = torch.Tensor(weights).clone()

    def forward(self, x, target, desc_data=None, get_attention=True):
        # get embeddings and apply dropout
        x = self.embed(x)
        x = self.embed_drop(x)
        x = x.transpose(1, 2)

        ####SOHA
        if (self.ontology):
            # apply convolution and nonlinearity (tanh)
            hp = torch.tanh(self.conv(x).transpose(1, 2))
            # apply attention
            alpha4 = F.softmax(self.U4.weight.matmul(hp.transpose(1, 2)), dim=2)
            # alpha3 = F.softmax(self.U3.weight.matmul(hp.transpose(1, 2)), dim=2)
            # alpha2 = F.softmax(self.U2.weight.matmul(hp.transpose(1, 2)), dim=2)
            # alpha1 = F.softmax(self.U1.weight.matmul(hp.transpose(1, 2)), dim=2)
            # document representations are weighted sums using the attention. Can compute all at once as a matmul

            m4 = alpha4.matmul(hp)  # K
            # m3 = alpha3.matmul(hp)
            # m2 = alpha2.matmul(hp)
            # m1 = alpha1.matmul(hp)

            # dic1 = list(self.dicts['L1'].values())
            # XX = torch.FloatTensor(dic1)
            # m1ppp = m4.clone()
            # m1ppp[:, torch.arange(m4.shape[1]), :] = m1[:, XX.squeeze().long(), :]
            #
            # dic2 = list(self.dicts['L2'].values())
            # ## ADDING HERE
            # tmp = len(set(dic2))
            # dicp = torch.zeros((m2.shape[1])).cuda()
            # for i in range(tmp):
            #     ind = dic2.index(dic2[i])
            #     dicp[i] = (dic1[ind])
            # m1p = m2.clone()
            # m1p[:, torch.arange(m2.shape[1]), :] = m1[:, dicp.squeeze().long(), :]
            # m22 = torch.cat((m1p, m2), dim=2)
            # # m22=m2.clone()
            ## ADDING END
            # XX = torch.FloatTensor(dic2)
            # m2pp = m4.clone()
            # m2pp[:, torch.arange(m4.shape[1]), :] = m2[:, XX.squeeze().long(), :]
            #
            # dic3 = list(self.dicts['L3'].values())
            # ## ADDING HERE
            # tmp = len(set(dic3))
            # dicp = torch.zeros((m3.shape[1])).cuda()
            # for i in range(tmp):
            #     ind = dic3.index(dic3[i])
            #     dicp[i] = (dic2[ind])
            # m2p = m3.clone()
            # m2p[:, torch.arange(m3.shape[1]), :] = m2[:, dicp.squeeze().long(), :]
            #
            # m2X = torch.zeros(m2p.shape[0], m2p.shape[1], m2p.shape[2] * 1).cuda()  # m4.clone()
            # m2X[:, torch.arange(1750), :] = m22[:, dicp.squeeze().long(), :]
            # m33 = torch.cat((m2X, m3), dim=2)
            # ## ADDING END
            # XX = torch.FloatTensor(dic3)
            # m3p = m4.clone()
            # m3p[:, torch.arange(m4.shape[1]), :] = m3[:, XX.squeeze().long(), :]
            # # m4t = torch.cat((m1ppp, m2pp, m3p, m4.clone()), dim=2)
            # m4t = torch.cat(( m2pp, m3p, m4.clone()), dim=2)

            # x_split = torch.split(m4t, 1, dim=0)
            # data_list = []
            # Convert the list of tuples to a torch tensor
            # edge_indices_tensor = torch.tensor(list(combinations(list(range(m4t.shape[1])), 2))).t()
            # for i in range(16):
            #     data = Data(x=x_split[i], edge_index=edge_indices_tensor)
            #     data_list.append(data)
            # batch = Batch.from_data_list(data_list)
            m4t=m4.clone()
            out1= self.graph_conv_1(m4t, self.dist)#(batch.x, edge_index=batch.edge_index,return_attention_weights=True)
            # out1 = F.leaky_relu(out1, negative_slope=0.2)
            # out1 = self.graph_conv_2(out1, self.dist)
            # # Create the edge_index tensor
            # edge_index = torch.nonzero(self.dist, as_tuple=False).t()
            # # Create the edge_attr tensor
            # edge_attr = self.dist[edge_index[0], edge_index[1]]
            # out1=self.graph_conv_1(m4t,edge_index,edge_weight=edge_attr)
            out1 = F.leaky_relu(out1, negative_slope=0.2)
            # m4=out1
            m4=torch.cat((m4t,out1),dim=2)


        else:

            # apply convolution and nonlinearity (tanh)
            hp = torch.tanh(self.conv(x).transpose(1, 2))
            # apply attention
            alpha4 = F.softmax(self.U4.weight.matmul(hp.transpose(1, 2)), dim=2)
            alpha3 = F.softmax(self.U3.weight.matmul(hp.transpose(1, 2)), dim=2)
            alpha2 = F.softmax(self.U2.weight.matmul(hp.transpose(1, 2)), dim=2)
            alpha1 = F.softmax(self.U1.weight.matmul(hp.transpose(1, 2)), dim=2)
            # document representations are weighted sums using the attention. Can compute all at once as a matmul

            m4 = alpha4.matmul(hp)  # K
            m3 = alpha3.matmul(hp)
            m2 = alpha2.matmul(hp)
            m1 = alpha1.matmul(hp)

            dic1 = list(self.dicts['L1'].values())
            XX = torch.FloatTensor(dic1)
            m1ppp = m4.clone()
            m1ppp[:, torch.arange(m4.shape[1]), :] = m1[:, XX.squeeze().long(), :]

            dic2 = list(self.dicts['L2'].values())
            ## ADDING HERE
            tmp = len(set(dic2))
            dicp = torch.zeros((m2.shape[1])).cuda()
            for i in range(tmp):
                ind = dic2.index(dic2[i])
                dicp[i] = (dic1[ind])
            m1p = m2.clone()
            m1p[:, torch.arange(m2.shape[1]), :] = m1[:, dicp.squeeze().long(), :]
            m22 = torch.cat((m1p, m2), dim=2)
            ## ADDING END
            XX = torch.FloatTensor(dic2)
            m2pp=m4.clone()
            m2pp[:, torch.arange(m4.shape[1]), :] = m2[:, XX.squeeze().long(), :]

            dic3 = list(self.dicts['L3'].values())
            ## ADDING HERE
            tmp = len(set(dic3))
            dicp = torch.zeros((m3.shape[1])).cuda()
            for i in range(tmp):
                ind = dic3.index(dic3[i])
                dicp[i] = (dic2[ind])
            m2p = m3.clone()
            m2p[:, torch.arange(m3.shape[1]), :] = m2[:, dicp.squeeze().long(), :]

            m2X = torch.zeros(m2p.shape[0], m2p.shape[1], m2p.shape[2]*2).cuda()  # m4.clone()
            m2X[:, torch.arange(1750), :] = m22[:, dicp.squeeze().long(), :]
            m33 = torch.cat((m2X, m3), dim=2)
            ## ADDING END
            XX = torch.FloatTensor(dic3)
            m3p = m4.clone()
            m3p[:, torch.arange(m4.shape[1]), :] = m3[:, XX.squeeze().long(), :]
            m4 = torch.cat((m1ppp, m2pp, m3p, m4), dim=2)

            # linear map from hp to the dimensions of e_i at use it as V
        # final layer classification
        # y1 = self.final1.weight.mul(m1).sum(dim=2).add(self.final1.bias)
        y1=0
        y2=0
        y3=0
        # y2 = self.final2.weight.mul(m22).sum(dim=2).add(self.final2.bias)
        # y3 = self.final3.weight.mul(m33).sum(dim=2).add(self.final3.bias)
        # y3= self.final3.weight.mul(torch.cat((m2p, m3), dim=2)).sum(dim=2).add(self.final3.bias)
        y4 = self.final4.weight.mul(m4).sum(dim=2).add(self.final4.bias)
        # y4t=0
        y4t = self.final4t.weight.mul(m4t).sum(dim=2).add(self.final4t.bias)
        # y = self.final(m)

        if desc_data is not None:
            # run descriptions through description module
            b_batch = self.embed_descriptions(desc_data, self.gpu)
            # get l2 similarity loss
            diffs = self._compare_label_embeddings(target, b_batch, desc_data)
            gamma=1000
            diffs_ont = self._compare_label_ont_embeddings(target, b_batch, desc_data, gamma)
            diffs=0*diffs+diffs_ont
        else:
            diffs = None

        # final sigmoid to get predictions
        # yhat = y#.unsqueeze(dim=0)

        # Copy values from targets to targetp using X indices
        target1 = torch.from_numpy(get_target(target, self.dicts, level=1)).cuda()
        target2 = torch.from_numpy(get_target(target, self.dicts, level=2)).cuda()
        target3=torch.from_numpy(get_target(target, self.dicts, level=3)).cuda()

        loss = self._get_loss(y4t,y4,y3,y2,y1, target, target3,target2,target1, diffs)
        if (torch.isnan(loss)):
            print("hello")
        return y4t, loss, alpha4


class VanillaConv(BaseModel):

    def __init__(self, Y, embed_file, kernel_size, num_filter_maps, gpu=True, dicts=None, embed_size=100, dropout=0.5):
        super(VanillaConv, self).__init__(Y, embed_file, dicts, dropout=dropout, embed_size=embed_size)
        # initialize conv layer as in 2.1
        self.conv = nn.Conv1d(self.embed_size, num_filter_maps, kernel_size=kernel_size)
        xavier_uniform(self.conv.weight)

        # linear output
        self.fc = nn.Linear(num_filter_maps, Y)
        xavier_uniform(self.fc.weight)

    def forward(self, x, target, desc_data=None, get_attention=False):
        # embed
        x = self.embed(x)
        x = self.embed_drop(x)
        x = x.transpose(1, 2)

        # conv/max-pooling
        c = self.conv(x)
        if get_attention:
            # get argmax vector too
            x, argmax = F.max_pool1d(torch.tanh(c), kernel_size=c.size()[2], return_indices=True)
            attn = self.construct_attention(argmax, c.size()[2])
        else:
            x = F.max_pool1d(torch.tanh(c), kernel_size=c.size()[2])
            attn = None
        x = x.squeeze(dim=2)

        # linear output
        x = self.fc(x)

        # final sigmoid to get predictions
        yhat = x
        loss = self._get_loss(yhat, target)
        return yhat, loss, attn

    def construct_attention(self, argmax, num_windows):
        attn_batches = []
        for argmax_i in argmax:
            attns = []
            for i in range(num_windows):
                # generate mask to select indices of conv features where max was i
                mask = (argmax_i == i).repeat(1, self.Y).t()
                # apply mask to every label's weight vector and take the sum to get the 'attention' score
                weights = self.fc.weight[mask].view(-1, self.Y)
                if len(weights.size()) > 0:
                    window_attns = weights.sum(dim=0)
                    attns.append(window_attns)
                else:
                    # this window was never a max
                    attns.append(Variable(torch.zeros(self.Y)).cuda())
            # combine
            attn = torch.stack(attns)
            attn_batches.append(attn)
        attn_full = torch.stack(attn_batches)
        # put it in the right form for passing to interpret
        attn_full = attn_full.transpose(1, 2)
        return attn_full


class VanillaRNN(BaseModel):
    """
        General RNN - can be LSTM or GRU, uni/bi-directional
    """

    def __init__(self, Y, embed_file, dicts, rnn_dim, cell_type, num_layers, gpu, embed_size=100, bidirectional=False):
        super(VanillaRNN, self).__init__(Y, embed_file, dicts, embed_size=embed_size, gpu=gpu)
        self.gpu = gpu
        self.rnn_dim = rnn_dim
        self.cell_type = cell_type
        self.num_layers = num_layers
        self.num_directions = 2 if bidirectional else 1

        # recurrent unit
        if self.cell_type == 'lstm':
            self.rnn = nn.LSTM(self.embed_size, floor(self.rnn_dim / self.num_directions), self.num_layers,
                               bidirectional=bidirectional)
        else:
            self.rnn = nn.GRU(self.embed_size, floor(self.rnn_dim / self.num_directions), self.num_layers,
                              bidirectional=bidirectional)
        # linear output
        self.final = nn.Linear(self.rnn_dim, Y)

        # arbitrary initialization
        self.batch_size = 16
        self.hidden = self.init_hidden()

    def forward(self, x, target, desc_data=None, get_attention=False):
        # clear hidden state, reset batch size at the start of each batch
        self.refresh(x.size()[0])

        # embed
        embeds = self.embed(x).transpose(0, 1)
        # apply RNN
        out, self.hidden = self.rnn(embeds, self.hidden)

        # get final hidden state in the appropriate way
        last_hidden = self.hidden[0] if self.cell_type == 'lstm' else self.hidden
        last_hidden = last_hidden[-1] if self.num_directions == 1 else last_hidden[-2:].transpose(0,
                                                                                                  1).contiguous().view(
            self.batch_size, -1)
        # apply linear layer and sigmoid to get predictions
        yhat = self.final(last_hidden)
        loss = self._get_loss(yhat, target)
        return yhat, loss, None

    def init_hidden(self):
        if self.gpu:
            h_0 = Variable(torch.cuda.FloatTensor(self.num_directions * self.num_layers, self.batch_size,
                                                  floor(self.rnn_dim / self.num_directions)).zero_())
            if self.cell_type == 'lstm':
                c_0 = Variable(torch.cuda.FloatTensor(self.num_directions * self.num_layers, self.batch_size,
                                                      floor(self.rnn_dim / self.num_directions)).zero_())
                return (h_0, c_0)
            else:
                return h_0
        else:
            h_0 = Variable(torch.zeros(self.num_directions * self.num_layers, self.batch_size,
                                       floor(self.rnn_dim / self.num_directions)))
            if self.cell_type == 'lstm':
                c_0 = Variable(torch.zeros(self.num_directions * self.num_layers, self.batch_size,
                                           floor(self.rnn_dim / self.num_directions)))
                return (h_0, c_0)
            else:
                return h_0

    def refresh(self, batch_size):
        self.batch_size = batch_size
        self.hidden = self.init_hidden()
