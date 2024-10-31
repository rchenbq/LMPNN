from src.structure.knowledge_graph_index import KGIndex
from src.structure.knowledge_graph import KnowledgeGraph
from src.structure.nbp_complex import ComplEx
from typing import List, Tuple, Union
from collections import defaultdict
import torch
import torch.nn as nn
import random
import json
import heapq
import argparse


import time

Triple = Tuple[int, int, int]

def get_edge_score(edge: Triple, model: ComplEx):
    # get the score of the edge

    head_emb = model._entity_embedding.weight.data[edge[0]]
    rel_emb = model._relation_embedding.weight.data[edge[1]]
    tail_emb = model._entity_embedding.weight.data[edge[2]]
    
    head_emb = head_emb.reshape(1, -1)
    rel_emb = rel_emb.reshape(1, -1)
    tail_emb = tail_emb.reshape(1, -1)

    score = model.embedding_score(head_emb, rel_emb, tail_emb)
    return score

def get_subgraph_triples(kg: KnowledgeGraph, model: ComplEx, num_edges: int):
    # get a subgraph triple with a fixed number of edges
    entities = defaultdict(list)
    entities_flag = defaultdict(bool)

    for triples in kg.triples:
        entities[triples[0]].append(triples)
        entities[triples[2]].append(triples)
        entities_flag[triples[0]] = 0
        entities_flag[triples[2]] = 0

    heap = []
    heapq.heapify(heap)
    v = random.choice(list(entities.keys()))
    # print(v)
    # print(entities[v])
    # print(type(entities[v]))
    # print(entities_flag[v])

    subgraph_triples = []
    for i in range(num_edges):
        if entities_flag[v] == 0:
            for edge in list(entities[v]):
                if edge[0] == v and entities_flag[edge[2]] == 1:
                    continue
                if edge[2] == v and entities_flag[edge[0]] == 1:
                    continue
                edge_score = get_edge_score(edge, model)
                heapq.heappush(heap, (edge_score, edge))
            entities_flag[v] = 1

        (subgraph_edge_score, subgraph_edge) = heapq.heappop(heap)
        # print(subgraph_edge)
        subgraph_triples.append(subgraph_edge)

        if entities_flag[subgraph_edge[0]] == 0:
            v = subgraph_edge[0]
        else:
            v = subgraph_edge[2]
        
    return subgraph_triples

if __name__ == "__main__":
    
    checkpoint_path = 'pretrain/cqd/FB15k-237-model-rank-100-epoch-50-1602503352.pt'
    checkpoint = torch.load(checkpoint_path)
    rank = 100

    kgidx_path = 'data/FB15k-237-betae/kgindex.json'
    kgidx = KGIndex.load(kgidx_path)

    triple_path = 'data/FB15k-237-betae/train_kg.tsv'
    kg = KnowledgeGraph.create(triple_files = triple_path, kgindex = kgidx)

    # print(checkpoint['_entity_embedding.weight'].shape) # 14505*200
    # print(checkpoint['_relation_embedding.weight'].shape) # 474*200
    # print(kgidx.num_entities) # 14505
    # print(kgidx.num_relations) # 474

    my_model = ComplEx(num_entities=kgidx.num_entities, num_relations=kgidx.num_relations, embedding_dim=rank)
    my_model._entity_embedding.weight.data = checkpoint['_entity_embedding.weight']
    my_model._relation_embedding.weight.data = checkpoint['_relation_embedding.weight']

    num_edges = 10

    # triples = random.choice(kg.triples)
    # print(triples)
    # score = get_edge_score(triples, my_model)
    # print(score)


    t0 = time.time()
    subgraph_triples = get_subgraph_triples(kg, my_model, num_edges)
    delta_t = time.time() - t0
    print(subgraph_triples)

    print('Time used: ', delta_t)