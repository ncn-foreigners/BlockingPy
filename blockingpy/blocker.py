import numpy as np
import pandas as pd
import networkx as nx
from scipy import sparse
from typing import Optional, Union, List, Dict, Any
import logging
import itertools
from collections import OrderedDict
import sys

from .controls import controls_ann, controls_txt
from .annoy_blocker import AnnoyBlocker
from .hnsw_blocker import HNSWBlocker
from .mlpack_blocker import MLPackBlocker
from .nnd_blocker import NNDBlocker
from .voyager_blocker import VoyagerBlocker
from .faiss_blocker import FaissBlocker
from .helper_functions import validate_input, validate_true_blocks, create_sparse_dtm
from .blocking_result import BlockingResult

class Blocker:
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        console_handler = logging.StreamHandler(sys.stdout)
        self.logger.addHandler(console_handler)
        self.eval_metrics = None
        self.confusion = None
        self.x_colnames = None
        self.y_colnames = None
        self.control_ann = {}
        self.control_txt = {}

    def block(self, 
              x: Union[pd.Series, sparse.csr_matrix, np.ndarray, np.array, List[str]],
              y: Optional[Union[np.ndarray, pd.Series, sparse.csr_matrix, List[str]]] = None,
              deduplication: bool = True,
              ann: str = "annoy",
              ann_write: Optional[str] = None,
              true_blocks: Optional[pd.DataFrame] = None,
              verbose: int = 0,
              graph: bool = False,
              control_txt: Dict[str, Any] = {},
              control_ann: Dict[str, Any] = {},
              x_colnames: Optional[List[str]] = None,
              y_colnames: Optional[List[str]] = None):
        
        self.logger.setLevel(logging.INFO if verbose else logging.WARNING)

        self.x_colnames = x_colnames
        self.y_colnames = y_colnames
        self.control_ann = controls_ann(control_ann)
        self.control_txt = controls_txt(control_txt)
        
        if deduplication:
            self.y_colnames = self.x_colnames

        if ann == 'nnd':
            distance = self.control_ann.get('nnd').get('metric')
        elif ann in ['annoy', 'voyager', 'hnsw', 'faiss']:
            distance = self.control_ann.get(ann).get('distance')
        else:
            distance = None

        if distance is None:
            distance = {  
                "nnd": "cosine",
                "hnsw": "cosine",
                "annoy": "angular",
                'voyager': "cosine",
                'faiss': "euclidean",
                "lsh": None,
                "kd": None
            }.get(ann)

        validate_input(x, ann, distance, ann_write)

        if y is not None:
            deduplication = False
            y_default = False
            k = 1
        else :
            y_default = y
            y = x
            k = 2
        
        validate_true_blocks(true_blocks, deduplication)

        #TOKENIZATION
        if sparse.issparse(x):
            if self.x_colnames is None or self.y_colnames is None:
                raise ValueError("Input is sparse, but x_colnames or y_colnames is None.")
            
            x_dtm = pd.DataFrame.sparse.from_spmatrix(x, columns=self.x_colnames)
            y_dtm = pd.DataFrame.sparse.from_spmatrix(y, columns=self.y_colnames)
        elif isinstance(x, np.ndarray):
            if self.x_colnames is None or self.y_colnames is None:
                raise ValueError("Input is np.ndarray, but x_colnames or y_colnames is None.")
            
            x_dtm = pd.DataFrame(x, columns=self.x_colnames)
            y_dtm = pd.DataFrame(y, columns=self.y_colnames)
        else:  
            self.logger.info("===== creating tokens =====\n")
            x_dtm = create_sparse_dtm(x,
                                      self.control_txt,
                                      verbose=True if verbose == 3 else False)
            y_dtm = create_sparse_dtm(y,
                                      self.control_txt,
                                      verbose=True if verbose == 3 else False)  
        #TOKENIZATION

        colnames_xy = np.intersect1d(x_dtm.columns, y_dtm.columns)
        
        self.logger.info(f"===== starting search ({ann}, x, y: {x_dtm.shape[0]}, {y_dtm.shape[0]}, t: {len(colnames_xy)}) =====")
    
        if ann == 'nnd':
            blocker = NNDBlocker()
        elif ann == 'hnsw':
            blocker = HNSWBlocker()
        elif ann in ['lsh', 'kd']:
            blocker = MLPackBlocker()
        elif ann == 'annoy':
            blocker = AnnoyBlocker()
        elif ann == 'voyager':
            blocker = VoyagerBlocker()
        elif ann == 'faiss':
            blocker = FaissBlocker()

        x_df = blocker.block(
            x=x_dtm[colnames_xy],
            y=y_dtm[colnames_xy],
            k=k,
            verbose=True if verbose in [2,3] else False,
            controls=self.control_ann
            )
    
        self.logger.info("===== creating graph =====\n")
        
        if deduplication:
            x_df = x_df[x_df['y'] > x_df['x']]

            x_df['query_g'] = 'q' + x_df['y'].astype(str)
            x_df['index_g'] = 'q' + x_df['x'].astype(str)
        else:
            x_df['query_g'] = 'q' + x_df['y'].astype(str)
            x_df['index_g'] = 'i' + x_df['x'].astype(str)

        ### IGRAPH PART IN R
        x_gr = nx.from_pandas_edgelist(x_df, source='query_g', target='index_g', create_using=nx.Graph())
        components = nx.connected_components(x_gr)
        x_block = {}
        for component_id, component in enumerate(components):
            for node in component:
                x_block[node] = component_id

        unique_query_g = x_df['query_g'].unique()
        unique_index_g = x_df['index_g'].unique()
        combined_keys = list(unique_query_g) + [node for node in unique_index_g if node not in unique_query_g]

        sorted_dict = OrderedDict()
        for key in combined_keys:
            if key in x_block:
                sorted_dict[key] = x_block[key]

        x_df['block'] = x_df['query_g'].apply(lambda x: x_block[x] if x in x_block else None)
        ###

        if true_blocks is not None:
            if not deduplication:
                pairs_to_eval = x_df[x_df['y'].isin(true_blocks['y'])][['x','y','block']]
                pairs_to_eval = pairs_to_eval.merge(true_blocks[['x','y']],
                                                    on=['x','y'],
                                                    how='left',
                                                    indicator='both')
                pairs_to_eval['both'] = np.where(pairs_to_eval['both'] == 'both',0,-1)

                true_blocks = true_blocks.merge(pairs_to_eval[['x', 'y']], 
                                                on=['x', 'y'], 
                                                how='left', 
                                                indicator='both')
                true_blocks['both'] = np.where(true_blocks['both'] == 'both', 0, 1)
                true_blocks['block'] = true_blocks['block'] + pairs_to_eval['block'].max()

                to_concat = true_blocks[true_blocks['both'] == 1][['x', 'y', 'block', 'both']]
                pairs_to_eval = pd.concat([pairs_to_eval, to_concat], ignore_index=True)
                pairs_to_eval['row_id'] = range(len(pairs_to_eval))
                pairs_to_eval['x2'] = pairs_to_eval['x'] + pairs_to_eval['y'].max()

                pairs_to_eval_long = pd.melt(pairs_to_eval[['y', 'x2', 'row_id', 'block', 'both']],
                                            id_vars=['row_id', 'block', 'both'],
                                            )
                pairs_to_eval_long = pairs_to_eval_long[pairs_to_eval_long['both'] == 0]
                pairs_to_eval_long['block_id'] = pairs_to_eval_long.groupby('block').ngroup()
                pairs_to_eval_long['true_id'] = pairs_to_eval_long['block_id']

                block_id_max = pairs_to_eval_long['block_id'].max(skipna=True)
                pairs_to_eval_long.loc[pairs_to_eval_long['both'] == -1, 'block_id'] = block_id_max + pairs_to_eval_long.groupby('row_id').ngroup() + 1 
                block_id_max = pairs_to_eval_long['block_id'].max(skipna=True)
                # recreating R's rleid function
                pairs_to_eval_long['rleid'] = (pairs_to_eval_long['row_id'] != pairs_to_eval_long['row_id'].shift(1)).cumsum()
                pairs_to_eval_long.loc[(pairs_to_eval_long['both'] == 1) & (pairs_to_eval_long['block_id'].isna()), 'block_id'] = block_id_max + pairs_to_eval_long['rleid']

                true_id_max = pairs_to_eval_long['true_id'].max(skipna=True)
                pairs_to_eval_long.loc[pairs_to_eval_long['both'] == 1, 'true_id'] = true_id_max + pairs_to_eval_long.groupby('row_id').ngroup() + 1
                true_id_max = pairs_to_eval_long['treu_id'].max(skipna=True)
                # recreating R's rleid function again
                pairs_to_eval_long['rleid'] = (pairs_to_eval_long['row_id'] != pairs_to_eval_long['row_id'].shift(1)).cumsum()
                pairs_to_eval_long.loc[(pairs_to_eval_long['both'] == -1) & (pairs_to_eval_long['true_id'].isna()), 'true_id'] = true_id_max + pairs_to_eval_long['rleid']

                pairs_to_eval_long.drop('rleid', inplace=True)

            else:
                pairs_to_eval_long = (pd.melt(x_df[['x', 'y', 'block']], id_vars=['block'])
                      [['block', 'value']]
                      .drop_duplicates()
                      .rename(columns={'block': 'block_id', 'value': 'x'})
                      .merge(true_blocks[['x', 'block']], on='x', how='left')
                      .rename(columns={'block': 'true_id'}))
            
            candidate_pairs = np.fromiter(itertools.combinations(range(pairs_to_eval_long.shape[0]), 2), dtype=int).reshape(-1, 2)

            block_id_array = pairs_to_eval_long['block_id'].values
            true_id_array = pairs_to_eval_long['true_id'].values
            same_block = block_id_array[candidate_pairs[:, 0]] == block_id_array[candidate_pairs[:, 1]]
            same_truth = true_id_array[candidate_pairs[:, 0]] == true_id_array[candidate_pairs[:, 1]]

            self.confusion = pd.crosstab(same_block, same_truth)
            
            fp = self.confusion.loc[True, False]   
            fn = self.confusion.loc[False, True]   
            tp = self.confusion.loc[True, True]    
            tn = self.confusion.loc[False, False]  

            recall = tp / (fn + tp) if (fn + tp) != 0 else 0 
            precision = tp / (tp + fp) if (tp + fp) != 0 else 0
            f1_score = 2 * (precision * recall) / (precision + recall) if (precision + recall) != 0 else 0
            accuracy = (tp + tn) / (tp + tn + fp + fn)
            specificity = tn / (tn + fp) if (tn + fp) != 0 else 0
            fpr = fp / (fp + tn) if (fp + tn) != 0 else 0
            fnr = fn / (fn + tp) if (fn + tp) != 0 else 0

            self.eval_metrics = {
                'recall': recall,
                'precision': precision,
                'fpr': fpr,
                'fnr': fnr,
                'accuracy': accuracy,
                'specificity': specificity,
                'f1_score': f1_score,
            }
            self.eval_metrics = pd.Series(self.eval_metrics)
        
        x_df = x_df.sort_values(['x', 'y', 'block']).reset_index(drop=True)

        return BlockingResult(x_df=x_df,
                              ann=ann,
                              deduplication=deduplication,
                              true_blocks=true_blocks,
                              eval_metrics=self.eval_metrics,
                              confusion=self.confusion,
                              colnames_xy=colnames_xy,
                              graph=graph)