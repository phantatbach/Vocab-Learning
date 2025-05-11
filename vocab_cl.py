import argparse
import pandas as pd
import numpy as np
from gensim.models import KeyedVectors
from gensim.models.fasttext import load_facebook_model
from sklearn_extra.cluster import KMedoids
from sklearn.metrics import pairwise_distances
import os

class VocabCluster:
    def __init__(self, embedding_path, n_clusters, input_file, output_file, lang='en', vocab_column='Vocab'):
        self.embedding_path = embedding_path
        self.n_clusters = n_clusters
        self.input_file = input_file
        self.output_file = output_file
        self.lang = lang.lower()
        self.embedding = None
        self.df = None
        self.vocab_list = []
        self.vocab_column = vocab_column
        self.words = []
        self.vectors = []

    def load_vocab(self):
        try:
            with open(self.input_file, encoding='utf-8') as f:
                self.df = pd.read_csv(f)
        except UnicodeDecodeError as e:
            print(f"❌ Unicode error when reading file: {e}")
            raise

        if self.vocab_column not in self.df.columns:
            raise ValueError(f"Column '{self.vocab_column}' not found in input file!")

        self.vocab_list = self.df[self.vocab_column].astype(str).str.strip().str.lower().tolist()

    def load_embedding(self):
        if self.embedding_path.endswith('.bin'):
            model = load_facebook_model(self.embedding_path)
            self.embedding = model.wv
        else:
            self.embedding = KeyedVectors.load_word2vec_format(self.embedding_path, binary=False, encoding='latin-1')

    def get_phrase_vector(self, phrase):
        tokens = phrase.lower().split()
        
        # ✅ Nếu là tiếng Hà Lan, bỏ "de" và "het"
        if self.lang == 'nl':
            tokens = [t for t in tokens if t not in {'de', 'het'}]

        vecs = [self.embedding[w] for w in tokens if w in self.embedding]
        return np.mean(vecs, axis=0) if vecs else None

    def vectorize_vocab(self):
        for word in self.vocab_list:
            vec = self.get_phrase_vector(word)
            if vec is not None:
                self.words.append(word)
                self.vectors.append(vec)
        if not self.vectors:
            raise ValueError("No valid vectors found for any vocab entries.")
        self.vectors = np.vstack(self.vectors)

    def cluster_vocab(self):
        dist = pairwise_distances(self.vectors, metric='cosine')
        kmedoids = KMedoids(n_clusters=self.n_clusters, metric='precomputed', random_state=42)
        labels = kmedoids.fit_predict(dist)
        label_map = dict(zip(self.words, labels))
        self.df['Group'] = self.df[self.vocab_column].map(label_map)

    def save_output(self):
        self.df.to_csv(self.output_file, index=False, encoding='utf-8-sig')
        print(f"✅ Saved to {self.output_file}")

    def run(self):
        self.load_vocab()
        self.load_embedding()
        self.vectorize_vocab()
        self.cluster_vocab()
        self.save_output()

if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description="Clustering Vocabs"
    )
    parser.add_argument('input_file',  help="Path to input CSV")
    parser.add_argument('output_file', help="Path to output CSV")
    parser.add_argument('n_clusters',   type=int, required=True, help='Specify the number of clusters!')
    parser.add_argument('--embedding_path',     required=True, help='Specify the embedding path!')
    parser.add_argument('--lang',     required=True, help='Specify your language!')
    parser.add_argument('--vocab_column', default='Vocab')
    args = parser.parse_args()

    clusterer = VocabCluster(
        input_file      = args.input_file,
        output_file     = args.output_file,
        n_clusters      = args.n_clusters,
        embedding_path  = args.embedding_path,
        lang            = args.lang,
        vocab_column    = args.vocab_column
    )
    clusterer.run()
