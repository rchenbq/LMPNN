from src.structure.knowledge_graph_index import KGIndex
from src.structure.knowledge_graph import KnowledgeGraph
from src.structure.nbp_complex import ComplEx
from typing import Tuple
from collections import defaultdict
from tqdm import tqdm
import argparse
import torch
import os

Triple = Tuple[int, int, int]
kgs = ['FB15k-237', 'FB15k', 'NELL']


parser = argparse.ArgumentParser()
parser.add_argument("--checkpoint_rank", type = int, default = 1000, choices = [100, 200, 500, 1000])
parser.add_argument("--checkpoint_epoch", type = int, default = 100, choices = [50, 100])

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

if __name__ == "__main__":

    args = parser.parse_args()
    rank = args.checkpoint_rank
    epoch = args.checkpoint_epoch

    for kgname in kgs:

        print('Processing ' + kgname)

        triple_path = os.path.join('data', kgname + '-betae', 'train_kg.tsv')
        kgidx_path = os.path.join('data', kgname + '-betae', 'kgindex.json')
        kgidx = KGIndex.load(kgidx_path) # load kgindex
        kg = KnowledgeGraph.create(triple_files = triple_path, kgindex = kgidx) # load knowledge graph

        checkpoint_path = ''
        for par, dirs, files in os.walk('pretrain/cqd'):
            for fname in files:
                if fname.endswith('.pt') and fname.startswith(kgname):
                    
                    rank_flag = False
                    epoch_flag = False

                    terms = fname.split('-')
                    for i, t in enumerate(terms):
                        if t == 'rank' and int(terms[i+1]) == rank:
                            rank_flag = True
                        if t == 'epoch' and int(terms[i+1]) == epoch:
                            epoch_flag = True

                    if rank_flag and epoch_flag:
                        checkpoint_path = os.path.join(par, fname)
                        break

        checkpoint = torch.load(checkpoint_path) # load checkpoint

        my_model = ComplEx(num_entities=kgidx.num_entities, num_relations=kgidx.num_relations, embedding_dim=rank)
        my_model._entity_embedding.weight.data = checkpoint['_entity_embedding.weight']
        my_model._relation_embedding.weight.data = checkpoint['_relation_embedding.weight']

        edges_score = defaultdict(torch.Tensor)
        for triple in tqdm(kg.triples):
            edge_score = get_edge_score(triple, my_model)
            edges_score[triple] = edge_score
        
        
        save_path = os.path.join('score/complex', kgname + '-rank' + str(rank) + '-epoch' + str(epoch) + '-edges_score.pt')
        torch.save(edges_score, save_path)

        
