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



def mono_index_rdkit(dataset: FingerprintedDataset):
    index_path_rdkit = os.path.join(dataset.dir, "index-rdkit.usearch")
    os.makedirs(os.path.join(dataset.dir), exist_ok=True)

    index_rdkit = Index(ndim=16, dtype='f16', metric='l2sq')

    try:
        for shard_idx, shard in enumerate(dataset.shards):
            print('shard.first_key: ', shard.first_key  )
            if shard.first_key in index_rdkit:
                logger.info(f"Skipping {shard_idx + 1} / {len(dataset.shards)}")
                continue

            logger.info(f"Starting {shard_idx + 1} / {len(dataset.shards)}")
            table = shard.load_table(["rdkit"])
            n = len(table)
            print("n:",n)

            # No need to shuffle the entries as they already are:
            keys = np.arange(shard.first_key, shard.first_key + n)
            #rdkit_fingerprints = [table["rdkit"][i].as_buffer() for i in range(n)]
            rdkit_fingerprints = [table["rdkit"][i] for i in range(n)]
            print("fp:",rdkit_fingerprints) #.to_numpy()

            # First construct the index just for MACCS representations
            #vectors = np.vstack([np.zeros(16, dtype=np.uint8) for i in range(n)])
            vectors = np.vstack([np.array(rdkit_fingerprints[i].as_py(), dtype=np.float16) for i in range(n)])
            print(vectors)
            
            index_rdkit.add(keys, vectors, log=f"Building {index_path_rdkit}")

            # Optional self-recall evaluation:
            # stats: SearchStats = self_recall(index_maccs, sample=1000)
            # logger.info(f"Self-recall: {100*stats.mean_recall:.2f} %")
            # logger.info(f"Efficiency: {100*stats.mean_efficiency:.2f} %")
            if shard_idx % 100 == 0: index_rdkit.save(index_path_rdkit)

            # Discard the objects to save some memory
            dataset.shards[shard_idx].table_cached = None
            dataset.shards[shard_idx].index_cached = None

        index_rdkit.save(index_path_rdkit)
        index_rdkit.reset()
    except KeyboardInterrupt:
        pass



if __name__ == "__main__":
    logging.basicConfig(level=logging.DEBUG)
    logger.info("Time to index some molecules!")


    #mono_index_maccs(FingerprintedDataset.open("data/pubchem"))
    mono_index_rdkit(FingerprintedDataset.open("data/pubchem"))
