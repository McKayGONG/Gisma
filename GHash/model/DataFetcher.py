# -*- coding: utf-8 -*-
"""
Created in Tue Jul 16 09:20:31 2019

@author: Zongyue Qin
"""
from random import shuffle, sample, randint, choice
from glob import glob
import networkx as nx
import scipy.sparse as sp
from sklearn.preprocessing import OneHotEncoder
import numpy as np
from multiprocessing.dummy import Pool as ThreadPool
import itertools
import time
import subprocess
import os
import platform
import tensorflow as tf

# Platform-specific GED executable path
if platform.system() == 'Windows':
    _ged_exe = os.path.join(os.path.dirname(__file__), 'BSS-GED', 'ged.exe')
else:
    _ged_exe = os.path.join(os.path.dirname(__file__), 'BSS-GED', 'ged')
import xml
import copy

from nx_to_gxl import nx_to_gxl
from config import FLAGS
from utils import sorted_nicely, sample_from_polynomial
from MyGraph import *

class DataFetcher:
    """ Represents a set of data """
    
    """ read training graphs and test graphs """
    def __init__(self, dataset, max_graph_num = 10000000,
                 exact_ged = False, wrp_train_graph = True,
                 label_transform = lambda x: x):
        self.max_graph_num = max_graph_num
        self.exact_ged = exact_ged
        self.label_transform = label_transform
        # a helper object to map features to consecutive integer
        self.type_hash={}
        self.typeCnt = 0
        
        self.dataset = dataset
        self.valid_percentage = FLAGS.valid_percentage
        self.node_feat_type = FLAGS.node_feat_encoder
        self.node_feat_name = FLAGS.node_feat_name
        self.node_label_name = FLAGS.node_label_name
        # the data directory should be ../data/$dataset/train(test)
        self.data_dir = os.path.join('..','data',dataset)
        self.train_data_dir = os.path.join(self.data_dir,'train')
        self.test_data_dir = os.path.join(self.data_dir, 'test')
        # read graphs
        self.train_file = os.path.join(self.train_data_dir, 'graphs.bss')
        self.test_file = os.path.join(self.test_data_dir, 'graphs.bss')

        train_graphs_raw, max_label = self._readGraphs(self.train_file)
        #train_graphs = train_graphs[0:FLAGS.batchsize*(1+FLAGS.k)]

        test_graphs, max_label_2 = self._readGraphs(self.test_file)
        self.max_label = max(max_label, max_label_2)

        #self.node_feat_encoder = self._create_node_feature_encoder(
        #    train_graphs + test_graphs)

        self.gid2graph = {}

        # Always keep raw train graphs for lightweight access
        self.train_graphs_raw = train_graphs_raw

        if wrp_train_graph == True:
            self.train_graphs = self.create_wrapper_graph(train_graphs_raw, 'train')
            self.train_graphs, self.valid_graphs = self._split_train_valid(self.train_graphs)
        else:
            self.train_graphs = []

        self.test_graphs = self.create_wrapper_graph(test_graphs, 'test')
        
        # pointers help to sample graphs
        self.cur_train_sample_ptr = 0
        self.cur_valid_sample_ptr = 0
        self.cur_test_sample_ptr = 0

        if len(self.test_graphs) > 0:
            #raise RuntimeError('train_graphs is empty, can\'t get feature dim')
            self.node_feature_dim = self.test_graphs[0].sparse_node_inputs.shape[1]
        elif len(self.train_graphs) > 0:
            self.node_feature_dim = self.train_graphs[0].sparse_node_inputs.shape[1]
        else:
            raise RuntimeError('train_graphs is empty, can\'t get feature dim')

        if FLAGS.sample_by_proximity == True:
            # The following variables need to be maintained outside the DataFetcher with 
            # model during training, see train.py
            self.positive_cum_p = np.ones([FLAGS.sample_pool_size, FLAGS.sample_pool_size])
            self.negative_cum_p = np.ones([FLAGS.sample_pool_size, FLAGS.sample_pool_size])
            self.sample_ptr = 0
            self.sample_bias = 0

    def getGraphByGid(self, gid):
        return self.gid2graph[gid]

    def get_pos_by_gid(self, gid, tvt):
        if tvt == 'train':
            lis = self.train_graphs
        elif tvt == 'valid':
            lis = self.valid_graphs
        elif tvt == 'test':
            lis = self.test_graphs
        else:
            raise RuntimeError('unrecognized tvt: {}'.format(tvt))

        for pos, g in enumerate(lis):
            if g.nxgraph.graph['gid'] == gid:
                return pos

        return -1

    """ return a training graph's gid """
    def get_train_graph_gid(self, pos):
        return self.train_graphs[pos].ori_graph['gid']

    def get_test_graph_gid(self, pos):
        return self.test_graphs[pos].ori_graph['gid']

    """ return dimension of features """
    def get_node_feature_dim(self):
        return self.node_feature_dim #self.train_graphs[0].sparse_node_inputs.shape[1]

    def get_train_data(self):
        for i in range(FLAGS.epochs*2):
            feat, lap, sizes, labels, gen_labels =  self.sample_train_data(FLAGS.batchsize)
            yield np.array(feat[0]), np.array(feat[1]), np.array(feat[2]),\
            np.array(lap[0]), np.array(lap[1]), np.array(lap[2]),\
            np.array(sizes), np.array(labels), np.array(gen_labels)

  
    def sampleByProximity(self):
        # Positive Sample
        # self.positive_cum_p is a n*n matrix
        n = FLAGS.sample_pool_size
        N = len(self.train_graphs)
        cum_p = np.zeros((n-1))
        if self.sample_ptr > 0:
            cum_p[0:self.sample_ptr] = self.positive_cum_p[self.sample_ptr,0:self.sample_ptr]
        if self.sample_ptr < n-1:
            cum_p[self.sample_ptr:] = self.positive_cum_p[self.sample_ptr,self.sample_ptr+1:]
        cum_p = cum_p / np.sum(cum_p)
        cum_p = np.cumsum(cum_p)

        positive_sample_idx = sample_from_polynomial(cum_p, FLAGS.positive_sample_num)
        positive_sample_idx[positive_sample_idx>=self.sample_ptr]= positive_sample_idx[positive_sample_idx>=self.sample_ptr]+1
        positive_sample_idx = positive_sample_idx + self.sample_bias
        positive_sample_idx = positive_sample_idx % N

        # Negative Sample
        if self.sample_ptr > 0:
            cum_p[0:self.sample_ptr] = self.negative_cum_p[self.sample_ptr,0:self.sample_ptr]
        if self.sample_ptr < n-1:
            cum_p[self.sample_ptr:] = self.negative_cum_p[self.sample_ptr,self.sample_ptr+1:]
        cum_p = cum_p / np.sum(cum_p)
        cum_p = np.cumsum(cum_p)

        negative_sample_idx = sample_from_polynomial(cum_p, FLAGS.batchsize-FLAGS.positive_sample_num-1)
        negative_sample_idx[negative_sample_idx>=self.sample_ptr] = negative_sample_idx[negative_sample_idx>=self.sample_ptr]+1
        negative_sample_idx = negative_sample_idx + self.sample_bias
        negative_sample_idx = negative_sample_idx % N

        sample_idx = [(self.sample_bias+self.sample_ptr)%N]+list(positive_sample_idx)+list(negative_sample_idx)
        #print(sample_idx)
        self.sample_ptr = self.sample_ptr + 1
        if self.sample_ptr == self.positive_cum_p.shape[0]:
            self.sample_ptr = 0
#        print('cum_p')
#        print(cum_p)
#        print('sample_ptr=%d'%self.sample_ptr)
#        print('sample_bias=%d'%self.sample_bias)
#        print(sample_idx)
        return sample_idx

       
    """ Sample training data """
    def sample_train_data(self, batchsize):
        """ we would First sample $batchsize graphs from train_graphs and 
            compute label between each pair. Then for each sampled graph we 
            randomly generate $k similar graphs as positive data. Finally we 
            merge features and laplacian matrices of batchsize*(k+1) graphs
            together and return feature matrix, laplacian matrix, sizes of each 
            graph and labels of each pair of sampled graphs """
        if FLAGS.sample_by_proximity == True:
            sample_idx = self.sampleByProximity()
        else:
            sample_idx = None
        self.get_train_graphs_and_compute_label(batchsize, sample_idx)
        
        # generate k similar graphs for each graph in self.sample_graphs
        """
        generated_graphs = []
        generated_labels = []
        for g in self.sample_graphs:
            graphs, labels = self.generate_similar_graphs(g, k)
            generated_graphs = generated_graphs + graphs
            generated_labels.append(labels)
        self.sample_graphs = self.sample_graphs + generated_graphs
        """
        self.gen_graphs = [[] for g in self.sample_graphs]
        self.gen_labels = [[] for g in self.sample_graphs]
        generated_labels = []
        
        pool = ThreadPool()
        pairs = [(i, g) for i,g in enumerate(self.sample_graphs)]        
        pool.map(self.gen_sim_graph_wrp, pairs)
        pool.close()
        pool.join()
        
        for graphs, labs in zip(self.gen_graphs, self.gen_labels):
            self.sample_graphs = self.sample_graphs + graphs
            generated_labels.append(labs)
        
        
        # get features of each graph and stack them to one sparse matrix
        features = sp.vstack([g.sparse_node_inputs for g in self.sample_graphs])
        features = self._sparse_to_tuple(features)
        
        # get laplacian matrix of each graph and diagnolize them into one
        # big matrix
        laplacians = sp.block_diag([g.laplacian for g in self.sample_graphs])
        laplacians = self._sparse_to_tuple(laplacians)
        
        # get size of each graph
        sizes = [len(g.ori_graph['nodes']) for g in self.sample_graphs]         
        #for g in self.sample_graphs:
        #    print(len(g.ori_graph['nodes']))
        #    print(g.ori_graph['adj_mat'].shape)
        
        self.labels = self.label_transform(self.labels)
        generated_labels = self.label_transform(generated_labels)

        return features, laplacians, sizes, self.labels, generated_labels
        
    """ get train_graphs and compute label between each pair of graphs """
    def get_train_graphs_and_compute_label(self, batchsize, sample_idx = None):
        if sample_idx is None:
            # sample $batchsize graphs
            start = self.cur_train_sample_ptr
            #else:
            #    start = batchsize*iteration%len(self.train_graphs)
            end = start + batchsize
            if end > len(self.train_graphs):
                end = len(self.train_graphs)
        
            sample_graphs = self.train_graphs[start:end]
            self.cur_train_sample_ptr = end
            # set pointer to the beginning and shuffle if pointer is at end
            if self.cur_train_sample_ptr == len(self.train_graphs):
                shuffle(self.train_graphs)
                self.cur_train_sample_ptr = 0        
            if len(sample_graphs) < batchsize:
                sample_graphs = sample_graphs +\
                self.train_graphs[0:batchsize-len(sample_graphs)]
            self.sample_graphs = sample_graphs
        else:
            self.sample_graphs = [self.train_graphs[idx] for idx in sample_idx]


        # Compute Label of every pair
        if not self.exact_ged:
            self.labels = self.getApproxGEDForEachPair(sample_graphs)
        else:
            fname = self.writeSampledGraphList2TempFile()
            g_cnt = str(len(self.sample_graphs))
            try:
                ret = subprocess.check_output([_ged_exe, fname, g_cnt, fname, g_cnt,
                                       str(FLAGS.GED_threshold - 1),
                                       str(FLAGS.beam_width)])

                geds = [float(ged) for ged in ret.split()][0:-1]
                geds = np.array(geds)
                geds[geds==-1] = FLAGS.GED_threshold
                self.labels = np.resize(geds, (batchsize, batchsize))
            except subprocess.TimeoutExpired:
                print('BSS-GED timeout, use Approxmate GED instead')
                self.labels = self.getApproxGEDForEachPair(self.sample_graphs)
            os.remove(fname)

        return
 
    # get certain graphs in self.training_graphs according to idx_list
    # return their features and laplacian matrices.
    def get_data_without_label(self, idx_list, tvt):
        graphs = []
        if tvt == 'train':
            target_list = self.train_graphs
        elif tvt == 'test':
            target_list = self.test_graphs
        elif tvt == 'valid':
            target_list = self.valid_graphs
        else:
            raise RuntimeError('unrecognized tvt label '+str(tvt))
            
        for idx in idx_list:
        #    print('without label')
        #    print(target_list[idx].nxgraph.graph['gid'])
            graphs.append(target_list[idx])
            """
            gg = target_list[idx]
            print(gg.nxgraph.graph['gid'])
            print(gg.sparse_node_inputs)
            print(gg.laplacian)
            """

        features = sp.vstack([g.sparse_node_inputs for g in graphs])
        features = self._sparse_to_tuple(features)
        
        laplacians = sp.block_diag([g.laplacian for g in graphs])
        laplacians = self._sparse_to_tuple(laplacians)
        
        sizes = [len(g.ori_graph['nodes']) for g in graphs]
        self.batch_node_num = sum(sizes)
                
        return features, laplacians, sizes

    def get_train_graphs_num(self):
        return len(self.train_graphs)
    
    def get_test_graphs_num(self):
        return len(self.test_graphs)
    
    """ read *.gexf in graph_dir and return a list of networkx graph """
    def _readGraphs(self, graph_file):
        from tqdm import tqdm

        graphs = []
        err_cnt = 0
        f = open(graph_file, 'r')

        # Read all lines and count graphs by detecting the pattern:
        # graph_id (single int), then "num_nodes num_edges" (two ints)
        lines = f.readlines()
        f.close()

        # Count actual graphs by looking for the ID + count pattern
        total_graphs = 0
        i = 0
        while i < len(lines) - 1:  # Need at least 2 lines for ID + count
            line = lines[i].strip()
            next_line = lines[i+1].strip() if i+1 < len(lines) else ""
            parts = next_line.split()
            # Check if this is a graph ID line (single int) followed by node/edge count (two ints)
            if line and line.lstrip('-').isdigit() and len(parts) == 2:
                if parts[0].isdigit() and parts[1].isdigit():
                    total_graphs += 1
                    if total_graphs >= self.max_graph_num:
                        break
            i += 1

        pbar = tqdm(total=total_graphs, desc='Reading graphs', unit='graph')

        status = 'id'
        node_left = 0
        edge_left = 0
        max_node_label = 0
        for line in lines:
            if status == 'id':
                if len(graphs) == self.max_graph_num:
                    break
                graph = {}
                graph['gid'] = int(line)
                graph['nodes'] = []
                graph['edges'] = []
                status = 'cnt'
            elif status == 'cnt':
                line = line.split()
                node_left = int(line[0])
                edge_left = int(line[1])
                status = 'node'
            elif status == 'node':
                node_label = int(line)
                if node_label > max_node_label:
                    max_node_label = node_label
                graph['nodes'].append(node_label)
                node_left = node_left - 1
                if node_left == 0:
                    node_cnt = len(graph['nodes'])
                    graph['adj_mat'] = np.zeros((node_cnt, node_cnt))
                    if edge_left > 0:
                        status = 'edge'
                    else:
                        graphs.append(graph)
                        status = 'id'
                        pbar.update(1)
            elif status == 'edge':
                line = line.split()
                f_node = int(line[0])
                t_node = int(line[1])
                graph['edges'].append((f_node, t_node))
                graph['adj_mat'][f_node,t_node]=1
                graph['adj_mat'][t_node,f_node]=1
                edge_left = edge_left - 1
                if edge_left == 0:
                    graphs.append(graph)
                    status = 'id'
                    pbar.update(1)
            else:
                raise RuntimeError('Unrecognized status: '+status)
        pbar.close()
        # print('%d error occured'%err_cnt)
        return graphs, max_node_label
#        for file in sorted_nicely(glob(graph_dir+'/*.gexf')):
#            if len(graphs) >= self.max_graph_num:
#                print('reach maximal graph num: %d'%self.max_graph_num)
#                break
#            gid = int(os.path.basename(file).split('.')[0])
#            try:
#                g = nx.read_gexf(file)
#            except xml.etree.ElementTree.ParseError:
#                print('err in nx.read_fexf', end = ' ')
#                print(file)
#                err_cnt = err_cnt + 1
#                continue
#            g.graph['gid'] = gid
#            graphs.append(g)
#            if len(graphs) == FLAGS.batchsize*(1+FLAGS.k):
#                break
#            if not nx.is_connected(g):
#                print('{} not connected'.format(gid))
#        return graphs
    
    """ split train_graphs into training set and validation set """
    def _split_train_valid(self, graphs):
        if self.valid_percentage < 0 or self.valid_percentage > 1:
            raise RuntimeError('valid_percentage {} must be in [0,1]'.format(
                    self.valid_percentage))
        shuffle(graphs)
        sp = int(len(graphs) * self.valid_percentage)
        valid_graphs = graphs[0:sp]
        train_graphs = graphs[sp:]
        
#        self._check_graphs_num(train_graphs, 'train')
#        self._check_graphs_num(valid_graphs, 'validation')
        return train_graphs, valid_graphs

    """ make sure the number of graphs > 2 """
    def _check_graphs_num(self, graphs, label):
        if len(graphs) <= 2:
            raise RuntimeError('Insufficient {} graphs {}'.format(
                label, len(graphs)))
    
    def _create_node_feature_encoder(self, gs):
        if self.node_feat_type == 'onehot':
            return NodeFeatureOneHotEncoder(gs, self.node_feat_name)
        elif 'constant' in self.node_feat_type:
            return NodeFeatureConstantEncoder(gs, self.node_feat_name)
        else:
            raise RuntimeError('Unknown node_feat_encoder {}'.format(
                self.node_feat_encoder))

    """ create a wrapper object (MyGraph class) for each nxgraph """
    def create_wrapper_graph(self, graphs, tvt):
        from tqdm import tqdm

        rtn = []
        for g in tqdm(graphs, desc=f'{tvt} MyGraph', unit='graph'):
            mg = MyGraph(g, self.max_label)
            rtn.append(mg)
            self.gid2graph[mg.ori_graph['gid']] = mg
        return rtn

    """Convert sparse matrix to tuple representation."""
    def _sparse_to_tuple(self, sparse_mx):
        def to_tuple(mx):
            if not sp.isspmatrix_coo(mx):
                mx = mx.tocoo()
            coords = np.vstack((mx.row, mx.col)).transpose()
            values = mx.data
            shape = mx.shape
            return coords, values, shape

        if isinstance(sparse_mx, list):
            for i in range(len(sparse_mx)):
                sparse_mx[i] = to_tuple(sparse_mx[i])
        else:
            sparse_mx = to_tuple(sparse_mx)

        return sparse_mx

    def getApproxGEDForEachPair(self, graphs):
        pass

    def getGEDForPair(self, g1, g2):
        
        g1_fname = self.writeGraph2TempFile(g1)
        g2_fname = self.writeGraph2TempFile(g2)

        ged = subprocess.check_output([_ged_exe, g1_fname, '1', g2_fname, '1', 
                                       str(FLAGS.GED_threshold),
                                       str(FLAGS.beam_width)])
        ged = ged.split()[0]
        # remove temporary files
        os.remove(g1_fname)
        os.remove(g2_fname)
        
        return int(ged)
        
    def getLabelsForSampledGraphs(self, pair):
        id1 = pair[0]
        id2 = pair[1]

        if id2 >= id1:
            return

        ged = self.getGEDForPair(self.sample_graphs[id1], 
                                   self.sample_graphs[id2])
        
        if ged > -1:
            self.labels[id1, id2] = ged
            self.labels[id2, id1] = ged
        else:
            self.labels[id1, id2] = FLAGS.GED_threshold
            self.labels[id2, id1] = FLAGS.GED_threshold
        return

    def writeSampledGraphList2TempFile(self):

        fname = 'tmpfile/'+str(time.time()) +  '.tmpfile'
        f = open(fname, 'w')

        for graph in self.sample_graphs:
            ori_graph = graph.ori_graph
            string = '{:d}\n{:d} {:d}\n'.format(ori_graph['gid'], 
                                          len(ori_graph['nodes']), 
                                          len(ori_graph['edges']))
            f.write(string)
            #label2node = {}
        
            #for i,n in enumerate(nxgraph.nodes(data=True)):
            #    if self.node_feat_name is None:
            #        label2node[n[1][self.node_label_name]] = i
            #        f.write('1\n')
            #    else:
            #        if n[1][self.node_feat_name] not in self.type_hash.keys():
            #            self.type_hash[n[1][self.node_feat_name]] = self.typeCnt
            #            self.typeCnt = self.typeCnt + 1
            #        if self.node_label_name != 'none':
            #            label2node[n[1][self.node_label_name]] = i
            #        f.write(str(self.type_hash[n[1][self.node_feat_name]])+'\n')
            for n in ori_graph['nodes']:
                f.write('%d\n'%n)

            for e in ori_graph['edges']:
                f.write('%d %d 0\n'%(e[0],e[1]))
#            for e in nxgraph.edges():
#                if self.node_label_name == 'none':
#                    f.write(str(e[0])+' '+str(e[1])+' 0\n')                
#                else:
#                    f.write(str(label2node[e[0]]) + ' ' + str(label2node[e[1]]) + ' 0\n')

        f.close()
        return fname

 
    def writeGraphList2TempFile(self, gid_list):
        raise RuntimeError('writeGraphList2TempFile unimplemented')
        #fname = 'tmpfile/'+str(time.time()) +  '.tmpfile'
        #f = open(fname, 'w')

        #for gid in gid_list:
        #    graph = self.gid2graph[gid]
        #    nxgraph = graph.nxgraph
        #    string = '{:d}\n{:d} {:d}\n'.format(nxgraph.graph['gid'], 
        #                                  len(nxgraph.nodes()), 
        #                                  len(nxgraph.edges()))
        #    f.write(string)
        #    label2node = {}
        
        #    for i,n in enumerate(nxgraph.nodes(data=True)):
        #        if self.node_feat_name is None:
        #            label2node[n[1][self.node_label_name]] = i
        #            f.write('1\n')
        #        else:
        #            if n[1][self.node_feat_name] not in self.type_hash.keys():
        #                self.type_hash[n[1][self.node_feat_name]] = self.typeCnt
        #                self.typeCnt = self.typeCnt + 1
        #            if self.node_label_name != 'none':
        #                label2node[n[1][self.node_label_name]] = i
        #            f.write(str(self.type_hash[n[1][self.node_feat_name]])+'\n')
        #
        #    for e in nxgraph.edges():
        #        if self.node_label_name == 'none':
        #            f.write(str(e[0])+' '+str(e[1])+' 0\n')                
        #        else:
        #            f.write(str(label2node[e[0]]) + ' ' + str(label2node[e[1]]) + ' 0\n')

        #f.close()
        #return fname
 

    def writeGraph2TempFile(self, graph):
        nxgraph = graph.ori_graph
        fname = 'tmpfile/'+str(time.time()) + '_'+str(nxgraph.graph['gid']) + '.tmpfile'
        f = open(fname, 'w')
        string = '{:d}\n{:d} {:d}\n'.format(ori_graph['gid'], 
                                          len(ori_graph['nodes']), 
                                          len(ori_graph['edges']))
        f.write(string)
        label2node = {}
 
        for n in ori_graph['nodes']:
            f.write('%d\n'%n)

        
        for e in ori_graph['edges']:
            f.write('%d %d 0\n'%(e[0],e[1]))
#       
#        for i,n in enumerate(nxgraph.nodes(data=True)):
#            if self.node_feat_name is None:
#                label2node[n[1][self.node_label_name]] = i
#                f.write('1\n')
#            else:
#                if n[1][self.node_feat_name] not in self.type_hash.keys():
#                    self.type_hash[n[1][self.node_feat_name]] = self.typeCnt
#                    self.typeCnt = self.typeCnt + 1
#                if self.node_label_name != 'none':
#                    label2node[n[1][self.node_label_name]] = i
#                f.write(str(self.type_hash[n[1][self.node_feat_name]])+'\n')
#        
#        for e in nxgraph.edges():
#            if self.node_label_name == 'none':
#                f.write(str(e[0])+' '+str(e[1])+' 0\n')                
#            else:
#                f.write(str(label2node[e[0]]) + ' ' + str(label2node[e[1]]) + ' 0\n')

        f.close()
        return fname
    
    def gen_sim_graph_wrp(self, pair):
        idx = pair[0]
        g = pair[1]
        gs, ls = self.generate_similar_graphs(g, FLAGS.k)
        self.gen_graphs[idx] = self.gen_graphs[idx] + gs
        self.gen_labels[idx] = self.gen_labels[idx] + ls
        return
    
    def generate_similar_graphs(self, g, k):
        generated_graphs = []
        geds = []
        for i in range(k):
            tmp_g = copy.deepcopy(g.ori_graph)
            # sample how many edit operation to perform
        #    op_num = randint(1,7)#FLAGS.GED_threshold-2)
        #    if op_num  == 7:
        #        op_num = 1
            op_num = randint(1,FLAGS.max_op)
            # though not accurate, may be good enough
            geds.append(op_num)
            j = 0
            op_cannot_be_1 = False
            op_cannot_be_2 = False
            while j < op_num:
                # randomly select a operation and do it
                # 0: change node label, 1: insert edge, 2: delete edge
                # 3: insert node; 4: delete node
                can_delete_node = self.has_isolate_node(tmp_g)
                # couldn't delete edge when only one node left
                op_cannot_be_2 = len(tmp_g['edges']) == 0
                
                
                op = randint(0, 4)
                # Skip op=0 (change label) if only one label exists or using constant encoding
                op_cannot_be_0 = (self.max_label == 0) or ('constant' in self.node_feat_type)
                while (can_delete_node is False and op == 4) or\
                (op == 0 and op_cannot_be_0) or\
                (op_cannot_be_1 and op == 1) or\
                (op_cannot_be_2 and op == 2):
                    op = randint(0, 4)
                 
                if op == 0:
                    if not self.random_change_node_label(tmp_g):
                        # Cannot change label (only one label exists), retry
                        j = j - 1
                    
                elif op == 1:
                    if not self.random_insert_edge(tmp_g):
                        op_cannot_be_1 = True
                        # insert edge fail, this op doesn't count
                        j = j - 1 
                    else:
                        op_cannot_be_2 = True
                        
                elif op == 2:
                    if not self.random_delete_edge(tmp_g):
                        op_cannot_be_2 = True
                        # delete fail, this op doesn't count
                        j = j - 1
                    else:
                        op_cannot_be_1 = False
                        
                elif op == 3:
                    self.random_insert_node(tmp_g)
                    # insert node takes 2 ops, so add an extra one here
                    j = j + 1
                    op_cannot_be_1 = False
                    
                else:
                    self.random_delete_node(tmp_g)
                    # delete node takes 2 ops
                    j = j + 1
                    
                    
                j = j + 1
            generated_graphs.append(MyGraph(tmp_g, self.max_label))   

        return generated_graphs, geds
    
    def has_isolate_node(self, g):
        non_isolate_nodes = set()
        for e in g['edges']:
            non_isolate_nodes.add(e[0])
            non_isolate_nodes.add(e[1])
        return len(non_isolate_nodes) != len(g['nodes'])

    def random_change_node_label(self, g):
        # If only one label exists, cannot change to a different label
        if self.max_label == 0:
            return False
        idx = randint(0, len(g['nodes'])-1)
        old_feat = g['nodes'][idx]
        new_feat = old_feat
        while new_feat == old_feat:
            new_feat = randint(0, self.max_label)
        g['nodes'][idx] = new_feat

        return True
    
    def random_insert_edge(self, g):
        old_edge_num = len(g['edges'])
        n = len(g['nodes'])
        if old_edge_num >= n*(n-1)/2:
            return False

        f_node = randint(0,n-1)
        t_node = randint(0,n-1)
        while f_node == t_node or g['adj_mat'][f_node,t_node]!=0:
            f_node = randint(0,n-1)
            t_node = randint(0,n-1)
        g['edges'].append((f_node, t_node))
        g['adj_mat'][f_node,t_node]=1
        g['adj_mat'][t_node,f_node]=1
            
        return True
    
    def random_delete_edge(self, g):
        e = choice(g['edges'])
        g['edges'].remove(e)
        g['adj_mat'][e[0],e[1]]=0
        g['adj_mat'][e[1],e[0]]=0
        """
        while not nx.is_connected(g):
            g.add_edge(*e)
            sample_cnt = sample_cnt + 1
            if sample_cnt > 100:
                break
            e = sample(g.edges(), 1)[0]
            g.remove_edge(*e)"""
        return True
    
    def random_insert_node(self, g):
        # randomly select a node to add edge to
        node_label = randint(0, self.max_label)
        g['nodes'].append(node_label)
        n = len(g['nodes'])
        newMat = np.zeros((n,n))
        newMat[0:n-1,0:n-1]=g['adj_mat']
        g['adj_mat']=newMat
        assert(len(g['nodes'])==g['adj_mat'].shape[0])
#        node = sample(g.node.keys(),1)[0]
#        node_label = str(time.time())
#        attri = {}
#        if self.node_feat_type == 'onehot':
#            attri[self.node_feat_name] = sample(\
#                 self.node_feat_encoder.feat_idx_dic.keys(),1)[0]
#        if self.node_label_name != 'none':
#            attri[self.node_label_name] = node_label
#        g.add_node(node_label, attr_dict=attri)
#        g.add_edge(node, node_label)
        return
    
    def random_delete_node(self, g):
        # find all nodes with 0 degree
#        deg = g.degree()
#        candidate_nodes = []
#        for n in deg.keys():
#            if deg[n] == 1:
#                candidate_nodes.append(n)
        non_isolate_nodes = set()

        for e in g['edges']:
            non_isolate_nodes.add(e[0])
            non_isolate_nodes.add(e[1])

        candidate_nodes = set(range(len(g['nodes'])))-non_isolate_nodes
        if len(candidate_nodes) == 0:
            raise RuntimeError('All nodes\' degree larger than 0, cannot delete')
        i = choice(list(candidate_nodes))
        del g['nodes'][i]
        newMat = np.delete(g['adj_mat'],i,axis=0)
        newMat = np.delete(newMat, i, axis=1)
        g['adj_mat'] = newMat
        return

