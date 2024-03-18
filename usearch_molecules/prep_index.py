"""Indexes fingerprints using USearch."""

import os
import logging
from typing import List, Callable
from multiprocessing import Process, cpu_count

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

from usearch.index import Index 

from dataset import FingerprintedDataset


logger = logging.getLogger(__name__)



def mono_index_mol2vec(dataset: FingerprintedDataset):
    index_path_rdkit = os.path.join(dataset.dir, "index-mol2vec.usearch")
    os.makedirs(os.path.join(dataset.dir), exist_ok=True)

    index_mol2vec = Index(ndim=300, dtype='f32', metric='l2sq')

    try:
        for shard_idx, shard in enumerate(dataset.shards):
            print('shard.first_key: ', shard.first_key  )
            if shard.first_key in index_mol2vec:
                logger.info(f"Skipping {shard_idx + 1} / {len(dataset.shards)}")
                continue

            logger.info(f"Starting {shard_idx + 1} / {len(dataset.shards)}")
            table = shard.load_table(["mol2vec"])
            n = len(table)
            print("n:",n)

            # No need to shuffle the entries as they already are:
            keys = np.arange(shard.first_key, shard.first_key + n)
            mol2vec_fingerprints = [table["mol2vec"][i] for i in range(n)]
            print("fp:",mol2vec_fingerprints) #.to_numpy()

            # First construct the index just for MACCS representations
            vectors = np.vstack([np.array(mol2vec_fingerprints[i].as_py(), dtype=np.float32) for i in range(n)])
            print(vectors)
            
            index_mol2vec.add(keys, vectors, log=f"Building {index_path_mol2vec}")

            # Optional self-recall evaluation:
            # stats: SearchStats = self_recall(index_maccs, sample=1000)
            # logger.info(f"Self-recall: {100*stats.mean_recall:.2f} %")
            # logger.info(f"Efficiency: {100*stats.mean_efficiency:.2f} %")
            if shard_idx % 100 == 0: index_mol2vec.save(index_path_mol2vec)

            # Discard the objects to save some memory
            dataset.shards[shard_idx].table_cached = None
            dataset.shards[shard_idx].index_cached = None

        index_mol2vec.save(index_path_mol2vec)
        index_mol2vec.reset()
    except KeyboardInterrupt:
        pass



if __name__ == "__main__":
    logging.basicConfig(level=logging.DEBUG)
    logger.info("Time to index some molecules!")

    mono_index_mol2vec(FingerprintedDataset.open("data/pubchem"))
