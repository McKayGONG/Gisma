# neighborhood prediction model


import torch
import torch.nn as nn
import torch.nn.functional as F
from dgl import DGLGraph
import dgl.function as fn
from functools import partial
import dgl
from torch.utils.data import DataLoader
import numpy as np
import time
import random
from dgl.nn.pytorch.conv import GINConv, SGConv, SAGEConv
import os
import networkx as nx
from dgl.data import DGLDataset
from tqdm import tqdm




def read_and_split_to_individual_graph(fname, gsizeNoLessThan=0, gsizeLessThan=10000000, graph_num=100000000):
    """
    Read graphs from file. Supports two formats:
      - Gisma format: "ID <id>" header
      - AIDS format: "t # <id>" header
    Both use "v <id> <label>" and "e <src> <dst> [<label>]" lines.
    """
    with open(fname) as f:
        content = f.read()

    # Detect format
    if content.lstrip().startswith('ID '):
        # Gisma format: split by "ID "
        blocks = content.split('\nID ')
        # First block might start with "ID " (no leading newline)
        if blocks[0].startswith('ID '):
            blocks[0] = blocks[0][3:]
        else:
            blocks = blocks[1:]  # skip empty first
    else:
        # AIDS format: split by "t # "
        content = content.replace('\\n', '\n')
        blocks = content.split("t # ")[1:]

    glist = []
    for block in blocks:
        lines = block.strip().split('\n')
        if not lines:
            continue

        gid = lines[0].strip().split(' ')[0]
        g = nx.Graph(id=gid)

        for line in lines[1:]:
            parts = line.strip().split(' ')
            if parts[0] == 'v':
                g.add_node(parts[1], att="0")
            elif parts[0] == 'e':
                g.add_edge(parts[1], parts[2], att="0")

        if gsizeNoLessThan <= g.number_of_nodes() < gsizeLessThan:
            glist.append(g)
            if len(glist) > graph_num:
                return glist

    return glist




class GINDataset(DGLDataset):

    def __init__(self, name, trainFileName, gid2dgmap, gID2InitEmbMap, gID2InitTensorIndexMap, neighNum, margin, isTrain,
                 st=0, ed=10, padding_index=40000,
                 self_loop=False, degree_as_nlabel=False,
                 raw_dir=None, force_reload=False, verbose=False):

        self._name = name
        gin_url = ""

        self.gid2dgmap = gid2dgmap
        self.gID2InitEmbMap = gID2InitEmbMap
        self.gID2InitTensorIndexMap = gID2InitTensorIndexMap
        self.isTrain = isTrain
        self.neighNum = neighNum
        self.margin = margin
        self.trainFileName = trainFileName
        self.st = st
        self.ed = ed
        self.padding_index = padding_index

        self.qList = []
        self.pgNodeEmbList = []
        self.ground_truth = []
        self.neighInitEmbIndexList = []
        self.mask_of_1_list = []
        self.mask_of_0_list = []
        self.class_weight_list = []

        super(GINDataset, self).__init__(name=name, url=gin_url, hash_key=(name, self_loop, degree_as_nlabel),
                                         raw_dir=raw_dir, force_reload=force_reload, verbose=verbose)

    @property
    def raw_path(self):
        return os.path.join(".", self.raw_dir)


    def download(self):
        pass


    def __len__(self):
        return len(self.qList)


    def __getitem__(self, idx):
        return self.qList[idx], self.pgNodeEmbList[idx], self.neighInitEmbIndexList[idx], self.ground_truth[idx], self.mask_of_1_list[idx], self.mask_of_0_list[idx], self.class_weight_list[idx]


    def _file_path(self):
        return self.file


    def process(self):
        if self.isTrain:
            f = open(self.trainFileName) # do not forget to set this address
        else:
            f = open('test.data') # do not forget to set this address

        lines = f.read()
        f.close()
        lines = lines.strip().split('\n')

        for line in lines:
            tmp = line.strip().split(' ')

            self.qList.append(self.gid2dgmap[tmp[0]])
            self.pgNodeEmbList.append(self.gID2InitEmbMap[tmp[1]])
            q_g_ged = float(tmp[2])
            gt = [int(ele) for ele in tmp[3:3+self.neighNum]]
            gt = gt[self.st:self.ed]
            self.ground_truth.append(gt)
            mask_of_1 = []
            mask_of_0 = []
            for i in range(0, len(gt)):
                if gt[i] == 1:
                    mask_of_1.append(1)
                    mask_of_0.append(0)
                else:
                    mask_of_0.append(1)
                    mask_of_1.append(0)

            mask_of_1 = torch.tensor(mask_of_1).bool()
            mask_of_0 = torch.tensor(mask_of_0).bool()

            self.mask_of_1_list.append(mask_of_1)
            self.mask_of_0_list.append(mask_of_0)

            neighIDs_and_neighDists = tmp[3+self.neighNum:]
            neighIDs = neighIDs_and_neighDists[0 : int(len(neighIDs_and_neighDists)/2)]
            neighDists = neighIDs_and_neighDists[int(len(neighIDs_and_neighDists)/2) : ]
            if len(neighIDs) != len(neighDists):
                print('ERROR! len(neighIDs) != len(neighDists)')
                print("neighIDs", len(neighIDs))
                print("neighDists", len(neighDists))
                exit(-1)


            neighIDs = neighIDs[self.st:self.ed]
            posInInitTensor = []
            for neighID in neighIDs:
                posInInitTensor.append(self.gID2InitTensorIndexMap.get(neighID, self.padding_index))
            while len(posInInitTensor) < (self.ed-self.st):
                posInInitTensor.append(self.padding_index)

            class_weight = []
            neighDists = neighDists[self.st:self.ed]
            for idx in range(0, len(neighDists)):
                neighDist = float(neighDists[idx])
                dist_diff = neighDist - q_g_ged - self.margin # the self.margin here needs to be consistent with the margin in train.data generation
                class_weight.append(abs(dist_diff))
            while len(class_weight) < (self.ed-self.st):
                class_weight.append(0.0)
            class_weight = np.array(class_weight)
            class_weight = class_weight/(100.0)
            class_weight = np.exp(class_weight)


            self.class_weight_list.append(class_weight)
            self.neighInitEmbIndexList.append(posInInitTensor)



    def save(self):
        pass

    def load(self):
        pass

    def has_cache(self):
        pass




class Model(nn.Module):
    def __init__(self, hdim, outputNum):
        super(Model, self).__init__()

        self.hdim = hdim
        self.outputNum = outputNum

        self.RELU = torch.nn.ReLU(inplace=True)

        self.fc_init = nn.Linear(20, self.hdim, bias=True)
        self.conv1_for_g = GINConv(None, 'mean')
        self.conv2_for_g = GINConv(None, 'mean')
        self.gnn_bn = torch.nn.BatchNorm1d(hdim)
        self.gnn_bn2 = torch.nn.BatchNorm1d(hdim)

        self.fc = nn.Linear(self.hdim*3, 256, bias=True)
        self.fc2 = nn.Linear(256, 256, bias=True)
        self.fc3 = nn.Linear(256, 256, bias=True)
        self.fc4 = nn.Linear(256, 1, bias=True)
        self.bn = torch.nn.BatchNorm1d(256)
        self.bn2 = torch.nn.BatchNorm1d(256)
        self.bn3 = torch.nn.BatchNorm1d(256)

        self.dp = torch.nn.Dropout(0.5)



    def forward(self, qList, pgNodeEmbList, neighEmbList, classWeightList):

        batch_size = len(pgNodeEmbList)
        number_of_outputs = self.outputNum   # do not forget to set it for different dataset

        qList.ndata['h'] = self.fc_init(qList.ndata['h'])
        qList.ndata['h'] = self.RELU(self.gnn_bn(self.conv1_for_g(qList, qList.ndata['h'])))
        qList.ndata['h'] = self.RELU(self.gnn_bn2(self.conv2_for_g(qList, qList.ndata['h'])))
        qemb = dgl.mean_nodes(qList, 'h')

        a = torch.cat([qemb, pgNodeEmbList], 1)
        a = a.repeat(1, number_of_outputs).view(-1, self.hdim*2)

        b = torch.cat([a, neighEmbList], 1)

        H = self.RELU(self.bn(self.fc(b)))
        H2 = self.RELU(self.bn2(self.fc2(H)))
        H3 = self.RELU(self.bn3(self.fc3(H2)))
        preds = torch.sigmoid(self.fc4(H3))

        preds = preds.view(batch_size, number_of_outputs)

        return preds, H3


bceLoss = nn.BCELoss()
mseLoss = nn.MSELoss()


def weighted_binary_cross_entropy(output, target, weights=None):
    output = torch.clamp(output, min=1e-6, max=1-1e-6)

    if weights is not None:
        assert len(weights) == 2
        loss = weights[1] * (target * torch.log(output)) + \
               weights[0] * ((1 - target) * torch.log(1 - output))
    else:
        loss = target * torch.log(output) + (1 - target) * torch.log(1 - output)

    return torch.neg(torch.mean(loss))


def myloss(epoch, preds, gts, mask_of_1_list, mask_of_0_list, classWeightList):
    bce = weighted_binary_cross_entropy(preds, gts, [10.0, 1.0])
    return bce


def perf_measure(y_actual, y_hat):
    TP = 0
    FP = 0
    TN = 0
    FN = 0

    if len(y_actual) == 0 or len(y_hat) == 0:
        return TP, FP, TN, FN

    for i in range(len(y_hat)):
        if y_actual[i]==y_hat[i]==1:
           TP += 1
        if y_hat[i]==1 and y_actual[i]!=y_hat[i]:
           FP += 1
        if y_actual[i]==y_hat[i]==0:
           TN += 1
        if y_hat[i]==0 and y_actual[i]!=y_hat[i]:
           FN += 1

    return TP, FP, TN, FN


def myloss_for_test(preds, gts, thr):
    fpList = []
    fnList = []
    tpList = []
    tnList = []
    fprList = []
    fnrList = []
    tprList = []
    for i in range(0, preds.shape[0]):
        pred = preds[i]
        gt = gts[i]
        pred = pred.view(-1,1).cpu().detach().numpy()
        gt = gt.view(-1,1).cpu().detach().numpy()
        y = (pred > thr)
        y = y.astype(int)

        TP, FP, TN, FN = perf_measure(gt, y)
        fpList.append(FP)
        fnList.append(FN)
        tpList.append(TP)
        tnList.append(TN)

        # Sensitivity, hit rate, recall, or true positive rate
        TPR = TP/(TP+FN+0.000001)
        # Specificity or true negative rate
        TNR = TN/(TN+FP+0.000001)
        # Precision or positive predictive value
        PPV = TP/(TP+FP+0.000001)
        # Negative predictive value
        NPV = TN/(TN+FN+0.000001)
        # Fall out or false positive rate
        FPR = FP/(FP+TN+0.000001)
        # False negative rate
        FNR = FN/(TP+FN+0.000001)
        # False discovery rate
        FDR = FP/(TP+FP+0.000001)

        # Overall accuracy
        ACC = (TP+TN)/(TP+FP+FN+TN)

        fprList.append(FPR)
        fnrList.append(FNR)
        tprList.append(TPR)

    fpList = np.array(fpList)
    fnList = np.array(fnList)
    tpList = np.array(tpList)
    tnList = np.array(tnList)
    fprList = np.array(fprList)
    fnrList = np.array(fnrList)
    tprList = np.array(tprList)

    return np.mean(fprList), np.mean(fnrList), np.mean(tprList), np.mean(fpList), np.mean(fnList), np.mean(tpList), np.mean(tnList)



def _read_one_gemb(args):
    """Worker function for parallel embedding reading."""
    addr, gfile = args
    gID = gfile[1:-4]
    with open(os.path.join(addr, gfile)) as f:
        lines = f.read().strip().split('\n')[1:]
    nodeEmbList = []
    for line in lines:
        tmp = line.strip().split(' ')
        nodeEmbList.append([float(ele) for ele in tmp[1:]])
    # Return plain list (picklable), convert to tensor in main process
    n = len(nodeEmbList)
    dim = len(nodeEmbList[0]) if n > 0 else 0
    mean = [sum(nodeEmbList[i][j] for i in range(n)) / n for j in range(dim)] if n > 0 else []
    return gID, mean


def read_initial_gemb(addr, pkl_path=None):
    """Load graph embeddings. Prefer pkl if available, fallback to txt files."""
    # Try pkl first
    if pkl_path and os.path.exists(pkl_path):
        import pickle
        print(f"  Loading embeddings from pkl: {pkl_path}")
        with open(pkl_path, 'rb') as f:
            raw_map = pickle.load(f)
        gEmbMap = {}
        for k, v in raw_map.items():
            if isinstance(v, torch.Tensor):
                gEmbMap[str(k)] = v
            else:
                gEmbMap[str(k)] = torch.from_numpy(v).float()
        print(f"  Loaded {len(gEmbMap)} embeddings from pkl")
        return gEmbMap

    # Fallback to txt files
    from multiprocessing import Pool
    num_workers = max(1, int(os.cpu_count() * 0.7))
    gfileList = os.listdir(addr)
    args_list = [(addr, gfile) for gfile in gfileList]
    print(f"  Using {num_workers} workers ({os.cpu_count()} CPUs)")

    gEmbMap = {}
    with Pool(num_workers) as pool:
        for gID, emb in tqdm(pool.imap_unordered(_read_one_gemb, args_list, chunksize=100),
                             total=len(args_list),
                             desc="  Reading embeddings", unit="file"):
            gEmbMap[gID] = torch.tensor(emb)
    return gEmbMap



def make_a_dglgraph(g):

    max_deg = 20 # largest node degree of q
    ones = torch.eye(max_deg)

    nodes = list(g.nodes())
    if len(nodes) == 0:
        dg = dgl.graph(([], []))
        dg.ndata['h'] = torch.zeros(0, max_deg)
        return dg

    node_map = {node: i for i, node in enumerate(nodes)}

    edges = [[],[]]
    for edge in g.edges():
        end1 = node_map[edge[0]]
        end2 = node_map[edge[1]]
        edges[0].append(int(end1))
        edges[1].append(int(end2))
        edges[0].append(int(end2))
        edges[1].append(int(end1))

    if len(edges[0]) == 0:
        dg = dgl.graph(([], []), num_nodes=len(nodes))
    else:
        dg = dgl.graph((torch.tensor(edges[0]), torch.tensor(edges[1])), num_nodes=len(nodes))

    h0 = dg.in_degrees().view(1,-1).squeeze()
    if h0.dim() == 0:
        h0 = h0.unsqueeze(0)
    h0 = torch.clamp(h0, 0, max_deg-1)
    h0 = ones.index_select(0, h0).float()
    dg.ndata['h'] = h0

    return dg


def _make_one_dgl(g):
    """Worker function for parallel DGL cache building."""
    gid = g.graph.get('id')
    dg = make_a_dglgraph(g)
    dg = dgl.add_self_loop(dg)
    return gid, dg


def _build_dgl_cache(entire_dataset):
    """Build DGL cache sequentially."""
    print(f"  Building DGL cache ({len(entire_dataset)} graphs)...")
    gid2dgmap = {}
    for g in tqdm(entire_dataset, desc="  Building DGL cache", unit="graph"):
        gid = g.graph.get('id')
        dg = make_a_dglgraph(g)
        dg = dgl.add_self_loop(dg)
        gid2dgmap[gid] = dg
    return gid2dgmap


def readQ2GDistBook(fname, validNodeIDSet=None):
    '''
    store distance from the query to a data graph
    '''
    f = open(fname)
    lines = f.read()
    f.close()
    lines = lines.strip()
    lines = lines.split('\n')
    distBook = {}
    for line in lines:
        tmp = line.split(" ")
        if validNodeIDSet != None and tmp[1] in validNodeIDSet:
            if tmp[0] in distBook:
                distBook[tmp[0]][tmp[1]] = float(tmp[2])
            else:
                distBook[tmp[0]] = {}
                distBook[tmp[0]][tmp[1]] = float(tmp[2])
    return distBook


def make_big_init_emb_tensor(gID2InitEmbMap, hdim):
    gid2posMap = {}
    embList = []
    for k,v in gID2InitEmbMap.items():
        embList.append(v)
        gid2posMap[k] = len(embList)-1
    embList.append(torch.zeros(hdim))

    return torch.stack(embList), gid2posMap


def get_exact_answer(topk, Q2GDistBook):
    answer = {}
    for query in Q2GDistBook.keys():
        distToGList = list(Q2GDistBook[query].items())
        distToGList.sort(key=lambda x: x[1])
        dist_thr = -1
        if topk-1 < len(distToGList):
            dist_thr = distToGList[topk-1][1]
        else:
            dist_thr = 1000000.0
        a = []
        for ele in distToGList:
            if ele[1] <= dist_thr:
                a.append(ele)
            else:
                break
        answer[query] = a
    return answer




def my_collate_fn(samples, gInitEmbTensor):
    qEmbs, pgNodes, neighIndexLists, gts, mask_of_1_list, mask_of_0_list, classWeightList = map(list, zip(*samples))

    neighIndexLists = torch.tensor(neighIndexLists).view(1,-1).squeeze()
    neighInitEmbs = torch.index_select(gInitEmbTensor, 0, neighIndexLists)

    return dgl.batch(qEmbs), torch.stack(pgNodes), neighInitEmbs, torch.tensor(gts), torch.stack(mask_of_1_list), torch.stack(mask_of_0_list), torch.tensor(classWeightList)




if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description='Train Mrk neighbor pruning models')
    parser.add_argument('--dataset', type=str, default='AIDS',
                        choices=['AIDS', 'PubChem', 'SYN', 'Chemical1M'],
                        help='Dataset name (default: AIDS)')
    parser.add_argument('--epochs', type=int, default=1000)
    parser.add_argument('--lr', type=float, default=0.05)
    parser.add_argument('--batch_size', type=int, default=1000)
    parser.add_argument('--emb_dim', type=int, default=512)
    parser.add_argument('--lr_step', type=int, default=10,
                        help='LR decay step size in epochs (default: 10)')
    parser.add_argument('--lr_gamma', type=float, default=0.95,
                        help='LR decay factor (default: 0.95)')
    args = parser.parse_args()

    SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
    dataset = args.dataset
    ds_lower = dataset.lower()

    D_of_pg = 80  # max degree of PG
    model_prediction_num = 80  # Process all 80 neighbors at once
    prune_margin = 1

    n_epochs = args.epochs
    lr = args.lr
    l2norm = 0
    batch_size = args.batch_size
    emb_dim = args.emb_dim

    PERCS = [20, 40, 60, 80]  # 4 independent Mrk rankers

    model_save_dir = os.path.join(SCRIPT_DIR, 'models', 'mrk', ds_lower)
    os.makedirs(model_save_dir, exist_ok=True)

    graph_file = os.path.join(SCRIPT_DIR, '..', 'Gisma', 'datasets', dataset, 'db.txt')
    emb_dir = os.path.join(SCRIPT_DIR, 'emb', ds_lower + str(emb_dim))

    #####################################################################################################
    ### Load shared data (once for all 4 models)
    #####################################################################################################

    print("Loading Node2Vec embeddings...")
    pkl_path = os.path.join(SCRIPT_DIR, 'emb', f'{ds_lower}{emb_dim}_graph_emb.pkl')
    gID2InitEmbMap = read_initial_gemb(emb_dir, pkl_path=pkl_path)
    gInitEmbBigTensor, gID2InitTensorIndexMap = make_big_init_emb_tensor(gID2InitEmbMap, emb_dim)
    PADDING_INDEX = len(gInitEmbBigTensor) - 1  # last row is zero vector
    print('read g init emb done.')
    print("gInitEmbBigTensor.shape", gInitEmbBigTensor.shape)
    print("PADDING_INDEX", PADDING_INDEX)

    print("Loading graphs and building DGL cache...")
    entire_dataset = read_and_split_to_individual_graph(graph_file, 0, 10000000)
    print('entire_dataset len: ', len(entire_dataset))
    gid2dgmap = _build_dgl_cache(entire_dataset)

    #####################################################################################################
    ### Train 4 independent Mrk models (perc=20, 40, 60, 80)
    #####################################################################################################

    for perc in PERCS:
        print(f"\n{'='*70}")
        print(f"  Training Mrk model: perc={perc}%")
        print(f"{'='*70}")

        train_data_file = os.path.join(SCRIPT_DIR, 'training_data', f'{ds_lower}_train.perc{perc}.data')
        if not os.path.exists(train_data_file):
            print(f"  ERROR: {train_data_file} not found. Run gen_ori_data.py first.")
            continue

        print("  Creating dataset (80 neighbors per sample)...")
        full_dataset = GINDataset(ds_lower, train_data_file, gid2dgmap, gID2InitEmbMap,
                                  gID2InitTensorIndexMap, D_of_pg, prune_margin, isTrain=True,
                                  st=0, ed=80, padding_index=PADDING_INDEX)

        # Train/Val split (80/20)
        from torch.utils.data import random_split
        val_size = int(len(full_dataset) * 0.2)
        train_size = len(full_dataset) - val_size
        train_dataset, val_dataset = random_split(full_dataset, [train_size, val_size],
                                                  generator=torch.Generator().manual_seed(42))
        print(f"  Train: {train_size}, Val: {val_size}")

        train_loader = DataLoader(
            train_dataset,
            batch_size=batch_size,
            collate_fn=partial(my_collate_fn, gInitEmbTensor=gInitEmbBigTensor),
            num_workers=0,
            drop_last=False,
            shuffle=True)
        val_loader = DataLoader(
            val_dataset,
            batch_size=batch_size,
            collate_fn=partial(my_collate_fn, gInitEmbTensor=gInitEmbBigTensor),
            num_workers=0,
            drop_last=False,
            shuffle=False)

        # Create a fresh model + optimizer + scheduler for each perc
        model = Model(hdim=emb_dim, outputNum=model_prediction_num)
        optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=l2norm)
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=args.lr_step, gamma=args.lr_gamma)

        print(f"  Training: epochs={n_epochs}, lr={lr}, batch_size={batch_size}")

        # Early stopping
        import copy
        early_stop_patience = 100  # Stop if no improvement for 100 epochs
        best_val_loss = float('inf')
        best_epoch = -1
        best_model_state = None
        epochs_without_improvement = 0

        total_start_time = time.time()

        # Outer progress bar: epoch level
        epoch_bar = tqdm(range(n_epochs), desc=f"  perc{perc} epochs", unit="ep", position=0)

        for epoch in epoch_bar:
            old_lr = optimizer.param_groups[0]['lr']

            # ---- Train ----
            model.train()
            epoch_loss = 0.0
            train_correct = 0
            train_total = 0
            batch_count = 0

            # Inner progress bar: batch level
            batch_bar = tqdm(train_loader, desc=f"    train", unit="batch",
                             position=1, leave=False)
            for qids, pgNodes, neighEmbList, gts, mask_of_1_list, mask_of_0_list, classWeightList in batch_bar:
                preds, H3 = model(qids, pgNodes, neighEmbList, classWeightList)
                loss = myloss(epoch, preds, gts.float(), mask_of_1_list, mask_of_0_list, classWeightList)
                epoch_loss += loss.item()
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                batch_count += 1

                # Training accuracy
                pred_labels = (preds > 0.5).int()
                train_correct += (pred_labels == gts).sum().item()
                train_total += gts.numel()

                batch_bar.set_postfix(loss=f"{loss.item():.4f}")

            avg_loss = epoch_loss / max(batch_count, 1)
            train_acc = train_correct / max(train_total, 1)

            # ---- Validation (every 10 epochs) ----
            if (epoch + 1) % 10 == 0:
                model.eval()
                val_loss = 0.0
                val_correct = 0
                val_total = 0
                val_batch_count = 0
                with torch.no_grad():
                    for qids, pgNodes, neighEmbList, gts, mask_of_1_list, mask_of_0_list, classWeightList in val_loader:
                        preds, H3 = model(qids, pgNodes, neighEmbList, classWeightList)
                        loss = myloss(epoch, preds, gts.float(), mask_of_1_list, mask_of_0_list, classWeightList)
                        val_loss += loss.item()
                        val_batch_count += 1
                        pred_labels = (preds > 0.5).int()
                        val_correct += (pred_labels == gts).sum().item()
                        val_total += gts.numel()
                val_avg_loss = val_loss / max(val_batch_count, 1)
                val_acc = val_correct / max(val_total, 1)

                # Early stopping check
                if val_avg_loss < best_val_loss:
                    best_val_loss = val_avg_loss
                    best_epoch = epoch
                    best_model_state = copy.deepcopy(model.state_dict())
                    epochs_without_improvement = 0
                else:
                    epochs_without_improvement += 10

                epoch_bar.set_postfix(
                    loss=f"{avg_loss:.4f}", acc=f"{train_acc:.4f}",
                    v_loss=f"{val_avg_loss:.4f}", v_acc=f"{val_acc:.4f}",
                    best=f"e{best_epoch}", lr=f"{old_lr:.5f}")

                if epochs_without_improvement >= early_stop_patience:
                    epoch_bar.close()
                    print(f"\n  [perc{perc}] Early stopping at epoch {epoch}, "
                          f"best val_loss={best_val_loss:.4f} at epoch {best_epoch}")
                    break
            else:
                epoch_bar.set_postfix(
                    loss=f"{avg_loss:.4f}", acc=f"{train_acc:.4f}",
                    lr=f"{old_lr:.5f}")

            scheduler.step()

        total_time = time.time() - total_start_time
        print(f"  [perc{perc}] Training done. Time: {total_time:.1f}s")

        # Save best model
        if best_model_state is not None:
            model.load_state_dict(best_model_state)
        save_path = os.path.join(model_save_dir, f"best.perc{perc}.pkl")
        torch.save(model.state_dict(), save_path)
        print(f"  Saved best model (epoch {best_epoch}) to {save_path}")

    print(f"\nAll 4 Mrk models trained and saved to {model_save_dir}/")
