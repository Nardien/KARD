import os
import math
import pickle
from typing import List, Optional
from sklearn.preprocessing import normalize

from scipy.sparse import csr_matrix

from pyserini import index, search
from pyserini.analysis import Analyzer, get_lucene_analyzer
from tqdm import tqdm

class Vectorizer:
    """Base class for vectorizer implemented on top of Pyserini.
    Parameters
    ----------
    lucene_index_path : str
        Path to lucene index folder
    min_df : int
        Minimum acceptable document frequency
    verbose : bool
        Whether to print out debugging information
    """

    def __init__(self, lucene_index_path: str, min_df: int = 1, verbose: bool = False):
        self.min_df: int = min_df
        self.verbose: bool = verbose
        if lucene_index_path in ["enwiki-paragraphs", "beir-v1.0.0-bioasq-flat"]:
            print("Load reader...")
            self.index_reader = index.IndexReader.from_prebuilt_index(lucene_index_path)
            print("Load searcher...")
            self.searcher = search.lucene.LuceneSearcher.from_prebuilt_index(lucene_index_path)
        else:
            self.index_reader = index.IndexReader(lucene_index_path)
            self.searcher = search.lucene.LuceneSearcher(lucene_index_path)
            lucene_index_path = os.path.basename(lucene_index_path)
        self.num_docs: int = self.searcher.num_docs
        self.stats = self.index_reader.stats()
        self.analyzer = Analyzer(get_lucene_analyzer())

        # build vocabulary
        print("Build Vocab...")
        vocab_dump = f"vocab_dump_{lucene_index_path}.pkl"

        if os.path.exists(vocab_dump):
            with open(vocab_dump, 'rb') as f:
                self.vocabulary_ = pickle.load(f)
        else:
            self.vocabulary_ = set()
            for term in self.index_reader.terms():
                if term.df > self.min_df:
                    self.vocabulary_.add(term.term)
            self.vocabulary_ = sorted(self.vocabulary_)
            with open(vocab_dump, 'wb') as f:
                pickle.dump(self.vocabulary_, f)

        print("Build Term to index mapping...")
        # build term to index mapping
        self.term_to_index = {}
        for i, term in enumerate(self.vocabulary_):
            self.term_to_index[term] = i
        self.vocabulary_size = len(self.vocabulary_)

        if self.verbose:
            print(f'Found {self.vocabulary_size} terms with min_df={self.min_df}')

    def get_query_vector(self, query: str):
        matrix_row, matrix_col, matrix_data = [], [], []
        tokens = self.analyzer.analyze(query)
        for term in tokens:
            if term in self.vocabulary_:
                matrix_row.append(0)
                matrix_col.append(self.term_to_index[term])
                matrix_data.append(1)
        vectors = csr_matrix((matrix_data, (matrix_row, matrix_col)), shape=(1, self.vocabulary_size))
        return vectors

class BM25Vectorizer(Vectorizer):
    """Wrapper class for BM25 vectorizer implemented on top of Pyserini.
    Parameters
    ----------
    lucene_index_path : str
        Path to lucene index folder
    min_df : int
        Minimum acceptable document frequency
    verbose : bool
        Whether to print out debugging information
    """

    def __init__(self, lucene_index_path: str, min_df: int = 1, verbose: bool = False):
        super().__init__(lucene_index_path, min_df, verbose)

    def get_vectors(self, docids: List[str], norm: Optional[str] = 'l2'):
        """Get the BM25 vectors given a list of docids
        Parameters
        ----------
        norm : str
            Normalize the sparse matrix
        docids : List[str]
            The piece of text to analyze.
        Returns
        -------
        csr_matrix
            Sparse matrix representation of BM25 vectors
        """
        matrix_row, matrix_col, matrix_data = [], [], []
        num_docs = len(docids)

        for index, doc_id in enumerate(docids):

            # Term Frequency
            tf = self.index_reader.get_document_vector(doc_id)
            if tf is None:
                continue

            # Filter out in-eligible terms
            tf = {t: tf[t] for t in tf if t in self.term_to_index}

            # Convert from dict to sparse matrix
            for term in tf:
                bm25_weight = self.index_reader.compute_bm25_term_weight(doc_id, term, analyzer=None)
                matrix_row.append(index)
                matrix_col.append(self.term_to_index[term])
                matrix_data.append(bm25_weight)

        vectors = csr_matrix((matrix_data, (matrix_row, matrix_col)), shape=(num_docs, self.vocabulary_size))

        # if norm:
        #     return normalize(vectors, norm=norm)
        return vectors

if __name__ == "__main__":
    vectorizer = BM25Vectorizer('enwiki-paragraphs')