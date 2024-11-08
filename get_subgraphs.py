from src.structure.knowledge_graph_index import KGIndex
from src.structure.knowledge_graph import KnowledgeGraph
from src.structure.nbp_complex import ComplEx
from typing import Tuple
from collections import defaultdict
from tqdm import tqdm

import torch
import random
import json
import heapq
import argparse
import os

import time

Triple = Tuple[int, int, int]
kgs = ['FB15k-237', 'FB15k', 'NELL']

parser = argparse.ArgumentParser()
parser.add_argument("--checkpoint_rank", type = int, default = 1000, choices = [100, 200, 500, 1000])
parser.add_argument("--checkpoint_epoch", type = int, default = 100, choices = [50, 100])
parser.add_argument("--max_num_edges", type = int, default = 10)
parser.add_argument("--num_subgraphs", type = int, default = 1000000)
parser.add_argument("--p", type = float, default = 0.9)


def get_subgraph_triples(entities: defaultdict, num_edges: int, p: float):
    # get a subgraph triple with a fixed number of edges
    
    heap = []
    heapq.heapify(heap)
    v = random.choice(list(entities.keys()))
    visited = set()

    subgraph_triples = []
    pr = 1.0
    for _ in range(num_edges):
        if v not in visited:
            for edge in list(entities[v]):
                
                if edge[0] == v and (edge[2] in visited):
                    continue
                if edge[2] == v and (edge[0] in visited):
                    continue
                # edge_score = edges_score[edge]
                heapq.heappush(heap, (random.random(), edge))

            visited.add(v)
        
        if len(heap) >0 and random.random() < pr:
            subgraph_edge = heapq.heappop(heap)[1]
            pr *= p
        else:
            break
        subgraph_triples.append(subgraph_edge)

        if subgraph_edge[0] not in visited:
            v = subgraph_edge[0]
        else:
            v = subgraph_edge[2]
        
    return subgraph_triples

if __name__ == "__main__":
    
    arg = parser.parse_args()
    rank = arg.checkpoint_rank
    epoch = arg.checkpoint_epoch
    max_num_edges = arg.max_num_edges
    num_subgraphs = arg.num_subgraphs
    p = arg.p

    for kgname in kgs:
        print('Processing ' + kgname)
        # triple_path = os.path.join('data', kgname + '-betae', 'train_kg.tsv')
        # kgidx_path = os.path.join('data', kgname + '-betae', 'kgindex.json')
        # kgidx = KGIndex.load(kgidx_path) # load kgindex
        # kg = KnowledgeGraph.create(triple_files = triple_path, kgindex = kgidx) # load knowledge graph

        print("Loading edges score for " + kgname)
        edges_score = torch.load('score/complex/' + kgname + '-rank' + str(rank) + '-epoch' + str(epoch) + '-edges_score.pt') # load edges score
        # sorted_edges_score = sorted(edges_score.items(), key = lambda x: x[1])
        print("Edges score loaded")

        entities = defaultdict(list)

        for triple in edges_score.keys():
            if len(entities[triple[0]]) < max_num_edges:
                entities[triple[0]].append(triple)
            else:
                for recorded_triple in entities[triple[0]]:
                    if edges_score[triple] < edges_score[recorded_triple]:
                        entities[triple[0]].remove(recorded_triple)
                        entities[triple[0]].append(triple)
                        break
            if len(entities[triple[2]]) < max_num_edges:
                entities[triple[2]].append(triple)
            else:
                for recorded_triple in entities[triple[2]]:
                    if edges_score[triple] < edges_score[recorded_triple]:
                        entities[triple[2]].remove(recorded_triple)
                        entities[triple[2]].append(triple)
                        break
        
        subgraphs = []
        for i in tqdm(range(num_subgraphs)):
            subgraph_triples = get_subgraph_triples(entities, max_num_edges, p)
            subgraphs.append(subgraph_triples)
        
        print(len(subgraphs))
        count = defaultdict(int)
        average_num_edges = 0
        for subgraph in subgraphs:
            average_num_edges += len(subgraph)
            count[len(subgraph)] += 1
        average_num_edges /= len(subgraphs)
        print(average_num_edges)
        print(count)
        print("--------------------")
        n = random.randint(0, len(subgraphs)-1)
        print(n)
        print(subgraphs[n])

        save_path = os.path.join('subgraphs/complex', kgname + '-rank' + str(rank) + '-epoch' + str(epoch) + '-subgraphs.pt')
        torch.save(subgraphs, save_path)
