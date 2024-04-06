import torch
import numpy as np
import torch.nn.functional as F
from torch_scatter import scatter_add
from torch_geometric.data import Data

from collections import deque

import os


def allowed_file(filename, ext_list):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ext_list


def build_test_graph(num_nodes, num_rels, triplets):
    src, rel, dst = triplets.transpose(0, 1)

    src, dst = torch.cat((src, dst)), torch.cat((dst, src))
    rel = torch.cat((rel, rel + num_rels))

    edge_index = torch.stack((src, dst))
    edge_index = edge_index.to(torch.int64)
    edge_type = rel

    data = Data(edge_index=edge_index)
    data.entity = torch.from_numpy(np.arange(num_nodes))
    data.edge_type = edge_type
    data.edge_norm = edge_normalization(edge_type, edge_index, num_nodes, num_rels)

    return data


def edge_normalization(edge_type, edge_index, num_entity, num_relation):
    '''
        Edge normalization trick
        - one_hot: (num_edge, num_relation)
        - deg: (num_node, num_relation)
        - index: (num_edge)
        - deg[edge_index[0]]: (num_edge, num_relation)
        - edge_norm: (num_edge)
    '''
    edge_index = edge_index.to(torch.int64)
    one_hot = F.one_hot(edge_type.to(torch.int64), num_classes=2 * num_relation).to(torch.float)
    deg = scatter_add(one_hot, edge_index[0], dim=0, dim_size=num_entity)
    index = edge_type + torch.arange(len(edge_index[0])) * (2 * num_relation)
    edge_norm = 1 / deg[edge_index[0]].view(-1)[index]

    return edge_norm


def load_data(file_path):
    print("load data from {}".format(file_path))

    with open(os.path.join(file_path, 'entities.dict')) as f:
        entity2id = dict()
        id2entity = dict()

        for line in f:
            eid, entity = line.strip().split('\t')
            entity2id[entity] = int(eid)
            id2entity[int(eid)] = entity

    with open(os.path.join(file_path, 'relations.dict')) as f:
        relation2id = dict()
        id2relation = dict()

        for line in f:
            rid, relation = line.strip().split('\t')
            relation2id[relation] = int(rid)
            id2relation[int(rid)] = relation

    train_triplets = read_triplets(os.path.join(file_path, 'train.txt'), entity2id, relation2id)
    valid_triplets = read_triplets(os.path.join(file_path, 'valid.txt'), entity2id, relation2id)
    test_triplets = read_triplets(os.path.join(file_path, 'test.txt'), entity2id, relation2id)

    print('num_entity: {}'.format(len(entity2id)))
    print('num_relation: {}'.format(len(relation2id)))
    print('num_train_triples: {}'.format(len(train_triplets)))
    print('num_valid_triples: {}'.format(len(valid_triplets)))
    print('num_test_triples: {}'.format(len(test_triplets)))

    all_triplets = train_triplets + valid_triplets + test_triplets

    return id2entity, entity2id, id2relation, relation2id, all_triplets


def read_triplets(file_path, entity2id, relation2id):
    triplets = []

    with open(file_path) as f:
        for line in f:
            head, relation, tail = line.strip().split('\t')
            triplets.append((entity2id[head], relation2id[relation], entity2id[tail]))

    return triplets


def triples_to_adj(triples):
    adj_list = {}
    for triple in triples:
        s, r, o = triple
        if s not in adj_list:
            adj_list[s] = [(r, o)]
        else:
            adj_list[s].append((r, o))
    return adj_list


def get_k_hop_subgraph(adj_list, nodes, k):
    subgraph = {}
    visited = set()
    queue = [(node, 0) for node in nodes]

    while queue:
        current_node, current_hop = queue.pop(0)

        if current_node in visited or current_hop > k:
            continue

        subgraph[current_node] = adj_list.get(current_node, [])

        visited.add(current_node)

        for relation, neighbor in adj_list.get(current_node, []):
            if neighbor not in visited:
                queue.append((neighbor, current_hop + 1))

    return subgraph


def find_meta_paths(adj_list, start_node, end_node, metapath):
    queue = deque([[start_node]])
    paths = []
    metalen = len(metapath)

    while queue:
        current_path = queue.popleft()
        current_node = current_path[-1]
        current_index = len(current_path) - 1

        if current_node == end_node:
            paths.append(current_path)
            continue

        if current_index > metalen - 1:
            continue

        if current_node in adj_list:
            for relation, neighbor in adj_list[current_node]:
                if (neighbor not in current_path) and relation == metapath[current_index]:
                    queue.append(current_path + [neighbor])

    return paths


def find_paths(adj_list, start_node, end_node, max_steps):
    queue = deque([[start_node]])
    paths = []

    while queue:
        current_path = queue.popleft()
        current_node = current_path[-1]
        current_index = len(current_path) - 1

        if current_node == end_node:
            paths.append(current_path)
            continue

        if current_index > max_steps - 1:
            continue

        if current_node in adj_list:
            for relation, neighbor in adj_list[current_node]:
                if neighbor not in current_path:
                    queue.append(current_path + [neighbor])

    return paths


def calculate_in_out_degree(adj_list):
    in_degree = {}
    out_degree = {}
    for node, neighbors in adj_list.items():
        out_degree[node] = len(neighbors)
        for _, neighbor in neighbors:
            neighbor_node = neighbor
            if neighbor_node in in_degree:
                in_degree[neighbor_node] += 1
            else:
                in_degree[neighbor_node] = 1
    return in_degree, out_degree


def predict_link(embedding, w, subject, obj, num_relations):
    triplets = np.array([(subject, r, obj) for r in range(num_relations)])
    s = embedding[triplets[:, 0]]
    o = embedding[triplets[:, 2]]
    r = w[triplets[:, 1]]

    score = torch.sum(s * r * o, dim=1)

    indices = [(idx, val.item()) for idx, val in enumerate(score) if val > 0]

    if len(indices) != 0:
        indices = sorted(indices, key=lambda x: x[1], reverse=True)

    return indices
