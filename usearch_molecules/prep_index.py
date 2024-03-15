"""Indexes fingerprints using USearch."""

import os
import logging
from typing import List, Callable
from multiprocessing import Process, cpu_count

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

from usearch.index import Index, CompiledMetric, MetricKind, MetricSignature, ScalarKind
from usearch.eval import self_recall, SearchStats

from metrics_numba import (tanimoto_maccs) 
from dataset import (FingerprintedDataset,FingerprintedEntry) 
from to_fingerprint import (shape_maccs) 

logger = logging.getLogger(__name__)



def mono_index_rdkit(dataset: FingerprintedDataset):
    index_path_rdkit = os.path.join(dataset.dir, "index-rdkit.usearch")
    os.makedirs(os.path.join(dataset.dir), exist_ok=True)

    index_rdkit = Index(ndim=16, dtype='i8', metric='l2sq')

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
            print("fp:",rdkit_fingerprints)

            # First construct the index just for MACCS representations
            vectors = np.vstack([np.zeros(16, dtype=np.uint8) for i in range(n)])
                                
            """
            vectors = np.vstack(
                [
                    FingerprintedEntry.from_parts(
                        None,
                        rdkit_fingerprints[i],
                        None,
                        None,
                        shape_rdkit,
                    ).fingerprint
                    for i in range(n)
                ]
            )
            """
            
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



def mono_index_maccs(dataset: FingerprintedDataset):
    index_path_maccs = os.path.join(dataset.dir, "index-maccs.usearch")
    os.makedirs(os.path.join(dataset.dir), exist_ok=True)

    index_maccs = Index(
        ndim=shape_maccs.nbits,
        dtype=ScalarKind.B1,
        metric=CompiledMetric(
            pointer=tanimoto_maccs.address,
            kind=MetricKind.Tanimoto,
            signature=MetricSignature.ArrayArray,
        ),
        # path=index_path_maccs,
    )

    try:
        for shard_idx, shard in enumerate(dataset.shards):
            if shard.first_key in index_maccs:
                logger.info(f"Skipping {shard_idx + 1} / {len(dataset.shards)}")
                continue

            logger.info(f"Starting {shard_idx + 1} / {len(dataset.shards)}")
            table = shard.load_table(["maccs"])
            n = len(table)

            # No need to shuffle the entries as they already are:
            keys = np.arange(shard.first_key, shard.first_key + n)
            maccs_fingerprints = [table["maccs"][i].as_buffer() for i in range(n)]

            # First construct the index just for MACCS representations
            vectors = np.vstack(
                [
                    FingerprintedEntry.from_parts(
                        None,
                        maccs_fingerprints[i],
                        None,
                        None,
                        shape_maccs,
                    ).fingerprint
                    for i in range(n)
                ]
            )

            index_maccs.add(keys, vectors, log=f"Building {index_path_maccs}")

            # Optional self-recall evaluation:
            # stats: SearchStats = self_recall(index_maccs, sample=1000)
            # logger.info(f"Self-recall: {100*stats.mean_recall:.2f} %")
            # logger.info(f"Efficiency: {100*stats.mean_efficiency:.2f} %")
            if shard_idx % 100 == 0:
                index_maccs.save(index_path_maccs)

            # Discard the objects to save some memory
            dataset.shards[shard_idx].table_cached = None
            dataset.shards[shard_idx].index_cached = None

        index_maccs.save(index_path_maccs)
        index_maccs.reset()
    except KeyboardInterrupt:
        pass



if __name__ == "__main__":
    logging.basicConfig(level=logging.DEBUG)
    logger.info("Time to index some molecules!")


    #mono_index_maccs(FingerprintedDataset.open("data/pubchem"))
    mono_index_rdkit(FingerprintedDataset.open("data/pubchem"))
