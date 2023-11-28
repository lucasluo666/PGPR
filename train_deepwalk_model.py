import random
import numpy as np
from gensim.models import Word2Vec
from utils import *
from data_utils import AmazonDataset, AmazonDataLoader
from transe_model import KnowledgeEmbedding
from data_utils import AmazonDataset
from knowledge_graph import KnowledgeGraph
import argparse


class DeepWalk(object):
    def __init__(self, kg, walk_length=10, num_walks=64, window_size=5, embedding_size=128, workers=3, epochs=5):
        self.kg = kg
        self.walk_length = walk_length
        self.num_walks = num_walks
        self.window_size = window_size
        self.embedding_size = embedding_size
        self.workers = workers
        self.epochs = epochs
        self.sentences = None
        self.model = None

    def _generate_walks(self):
        print('Generating walks...')
        walks = []
        for etype in self.kg.degrees:
            for eid in self.kg.degrees[etype]:
                for _ in range(self.num_walks):
                    walks.append(self._generate_random_walk(str(etype) + '+' + str(eid)))
        return walks

    def _generate_random_walk(self, start_node):
        walk = [start_node]
        while len(walk) < self.walk_length:
            curr_node = walk[-1]
            curr_etype, curr_eid = curr_node.split('+')
            neighbors = []
            for r in self.kg.G[curr_etype][int(curr_eid)].keys():
                for neighbor in self.kg.G[curr_etype][int(curr_eid)][r]:
                    tail_type = KG_RELATION[curr_etype][r]
                    neighbors.append((r, str(tail_type) + '+' + str(neighbor)))
            if len(neighbors) == 0:
                break
            r, next_node = random.choice(neighbors)
            walk.append(r)
            walk.append(next_node)
        return walk

    def train(self):
        self.sentences = self._generate_walks()
        print('Training DeepWalk model...')
        self.model = Word2Vec(sentences=self.sentences,
                              vector_size=self.embedding_size,
                              epochs=self.epochs,
                              window=self.window_size,
                              min_count=0,
                              sg=1,
                              workers=self.workers,
                              )

    def save_embeddings(self):
        if self.model is None:
            raise Exception('Model has not been trained yet!')
        embeddings = {}
        for etype in self.kg.degrees:
            embeddings[etype] = np.zeros((len(self.kg.degrees[etype]), self.embedding_size))
            for eid in self.kg.degrees[etype]:
                embeddings[etype][eid] = self.model.wv[str(etype) + '+' + str(eid)]
            relation_types = get_relations(etype)
            for relation in relation_types:
                embeddings[relation] = np.zeros((1, self.embedding_size))
                embeddings[relation][0] = self.model.wv[relation]
                print(relation,self.model.wv[relation])
        embed_file = '{}/deepwalk_embed.pkl'.format(TMP_DIR[args.dataset])
        pickle.dump(embeddings, open(embed_file, 'wb'))

parser = argparse.ArgumentParser()
parser.add_argument('--dataset', type=str, default=BEAUTY, help='One of {beauty, cd, cell, clothing}.')
parser.add_argument('--name', type=str, default='train_transe_model', help='model name.')
parser.add_argument('--seed', type=int, default=123, help='random seed.')
parser.add_argument('--gpu', type=str, default='1', help='gpu device.')
parser.add_argument('--epochs', type=int, default=30, help='number of epochs to train.')
parser.add_argument('--batch_size', type=int, default=64, help='batch size.')
parser.add_argument('--lr', type=float, default=0.5, help='learning rate.')
parser.add_argument('--weight_decay', type=float, default=0, help='weight decay for adam.')
parser.add_argument('--l2_lambda', type=float, default=0, help='l2 lambda')
parser.add_argument('--max_grad_norm', type=float, default=5.0, help='Clipping gradient.')
parser.add_argument('--embed_size', type=int, default=100, help='knowledge embedding size.')
parser.add_argument('--num_neg_samples', type=int, default=5, help='number of negative samples.')
parser.add_argument('--steps_per_checkpoint', type=int, default=200, help='Number of steps for checkpoint.')
args = parser.parse_args()
print('Create', '', 'knowledge graph from dataset...')
dataset = load_dataset(args.dataset)
kg = KnowledgeGraph(dataset)
kg.compute_degrees()
deepwalk = DeepWalk(kg)
deepwalk.train()
deepwalk.save_embeddings()