"""
    Various utility methods
"""
import csv
import json
import math
import os
import pickle

import torch
from torch.autograd import Variable

from learn import models #model_0 as models
from constants import *
import datasets
import persistence
import numpy as np
import networkx as nx
from tqdm import tqdm

def get_target(target,dicts,level):
    target=target.cpu().data.numpy()
    ind2c =  dicts['ind2c']
    if (level == 1):
        Ldict = dicts['L1']
    elif (level == 2):
        Ldict = dicts['L2']
    elif (level == 3):
        Ldict = dicts['L3']
    else:
        Ldict = ind2c
    target2 = np.zeros([target.shape[0], max(Ldict.values()) + 1])
    for i in range(0, target.shape[0]):
        ind = np.where(target[i, :] == 1)[0]
        for j in range(0, len(ind)):
            target2[i, Ldict.get(ind[j])] = 1
    target = target2
    return target

def normalize_adjacency_matrix(adjacency_matrix):
    # Add self-loops to the adjacency matrix
    adjacency_matrix = adjacency_matrix + torch.eye(adjacency_matrix.size(0)).to(adjacency_matrix.device)

    # Calculate the degree matrix
    degree_matrix = torch.sum(adjacency_matrix, dim=1)

    # Calculate the inverse of the square root of the degree matrix
    degree_matrix_sqrt_inv = torch.pow(degree_matrix, -0.5)

    # Calculate the symmetrically normalized adjacency matrix
    normalized_adjacency_matrix = torch.mm(torch.mm(torch.diag(degree_matrix_sqrt_inv), adjacency_matrix),
                                           torch.diag(degree_matrix_sqrt_inv))

    return normalized_adjacency_matrix

def build_cooccurrence_graph(targets):
    cooccurrence_matrix = np.matmul(targets.transpose(1, 0), targets)
    graph = nx.DiGraph()
    num_codes = targets.shape[1]
    graph.add_nodes_from(range(num_codes))
    for i in range(num_codes):
        for j in range(num_codes):
            if i != j and cooccurrence_matrix[i, j] > 0:
                weight = cooccurrence_matrix[i, j]
                graph.add_edge(i, j, weight=weight)
    return graph

def pick_model(args, dicts):
    """
        Use args to initialize the appropriate model
    """
    num_labels = len(dicts['ind2c'])
    desc_embed = args.lmbda > 0
    gen = datasets.data_generator(args.data_path, dicts, 50000, num_labels, version=args.version, desc_embed=desc_embed)
    for batch_idx, tup in tqdm(enumerate(gen)):
        data, target, hadm_ids, _, descs = tup

    cooccurrence_graph = build_cooccurrence_graph(target)

    Y = len(dicts['ind2c'])
    if args.model == "rnn":
        model = models.VanillaRNN(Y, args.embed_file, dicts, args.rnn_dim, args.cell_type, args.rnn_layers, args.gpu, args.embed_size,
                                  args.bidirectional)
    elif args.model == "cnn_vanilla":
        filter_size = int(args.filter_size)
        model = models.VanillaConv(Y, args.embed_file, filter_size, args.num_filter_maps, args.gpu, dicts, args.embed_size, args.dropout)
    elif args.model == "conv_attn":
        filter_size = int(args.filter_size)
        model = models.ConvAttnPool(Y, args.embed_file, filter_size, args.num_filter_maps, args.lmbda, args.gpu, dicts, ontology=args.ontology,
                                    embed_size=args.embed_size, dropout=args.dropout, code_emb=args.code_emb,transfer=args.transfer,transfer_10=args.transfer_10,cooccurence_graph=cooccurrence_graph)
    elif args.model == "logreg":
        model = models.BOWPool(Y, args.embed_file, args.lmbda, args.gpu, dicts, args.pool, args.embed_size, args.dropout, args.code_emb)
    if args.test_model:
        sd = torch.load(args.test_model)
        model.load_state_dict(sd)
    if args.gpu:
        model.cuda()
    return model

def make_param_dict(args):
    """
        Make a list of parameters to save for future reference
    """
    param_vals = [args.Y, args.filter_size, args.dropout, args.num_filter_maps, args.rnn_dim, args.cell_type, args.rnn_layers, 
                  args.lmbda, args.command, args.weight_decay, args.version, args.data_path, args.vocab, args.embed_file, args.lr,
                  args.ontology,args.transfer,args.transfer_10,args.subcode, args.from_checkpoint,args.checkpoint]
    param_names = ["Y", "filter_size", "dropout", "num_filter_maps", "rnn_dim", "cell_type", "rnn_layers", "lmbda", "command",
                   "weight_decay", "version", "data_path", "vocab", "embed_file", "lr", "ontology", "transfer","transfer_10","subcode",
                   "from_checkpoint","checkpoint"]
    params = {name:val for name, val in zip(param_names, param_vals) if val is not None}
    return params

def build_code_vecs(code_inds, dicts):
    """
        Get vocab-indexed arrays representing words in descriptions of each *unseen* label
    """
    code_inds = list(code_inds)
    ind2w, ind2c, dv_dict = dicts['ind2w'], dicts['ind2c'], dicts['dv']
    vecs = []
    for c in code_inds:
        code = ind2c[c]
        if code in dv_dict.keys():
            vecs.append(dv_dict[code])
        else:
            #vec is a single UNK if not in lookup
            vecs.append([len(ind2w) + 1])
    #pad everything
    vecs = datasets.pad_desc_vecs(vecs)
    return (torch.cuda.LongTensor(code_inds), vecs)

