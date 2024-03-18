"""Fingerprints the molecules, encoding them with 4 techniques, producing 28 Billions fingerprints for 7 Billion molecules."""

import os
import logging
from typing import List, Callable
from multiprocessing import Process, cpu_count

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

#from usearch.index import Index #, CompiledMetric, MetricKind, MetricSignature, ScalarKind
#from usearch.eval import self_recall, SearchStats

from to_fingerprint import smiles_to_rdkit
from dataset import write_table

logger = logging.getLogger(__name__)
    
    
def augment_with_rdkit(parquet_path: os.PathLike):
    meta = pq.read_metadata(parquet_path)
    column_names: List[str] = meta.schema.names
    if "rdkit" in column_names: return
    
    logger.info(f"Starting file {parquet_path}")
    table: pa.Table = pq.read_table(parquet_path)
    rdkit_list = []
    for smiles in table["smiles"]:
        try:
            fingers = smiles_to_mol2vec(str(smiles))
            fingers = pa.array(fingers, type=pa.float32())
            #print(type(fingers))
            rdkit_list.append(fingers) 
        except Exception:
            fingers = np.zeros(300, dtype=np.float32)
            fingers = pa.array(fingers, type=pa.float32())
            rdkit_list.append(fingers)  
            
    #print("Tab: ", table["smiles"][0], " ", table["cid"][0])
    
    mol2vec_list = pa.array(mol2vec_list, pa.list_(pa.float32())) 
    mol2vec_field = pa.field("mol2vec", pa.list_(pa.float32()), nullable=False) 
    table = table.append_column(rdkit_field, rdkit_list)
    write_table(table, parquet_path)



def augment_parquets_shard(parquet_dir: os.PathLike,):
    shard_index, shards_count = 0, 1
    filenames: List[str] = sorted(os.listdir(parquet_dir))
    files_count = len(filenames)
    try:
        for file_idx in range(shard_index, files_count, shards_count):
            try:
                filename = filenames[file_idx]
                augment_with_rdkit(os.path.join(parquet_dir, filename))
                logger.info( "Augmented shard {}. Process # {} / {}".format(filename, shard_index, shards_count))
            except KeyboardInterrupt as e:
                raise e

    except KeyboardInterrupt as e:
        logger.info(f"Stopping shard {shard_index} / {shards_count}")
        raise e



if __name__ == "__main__":
    logging.basicConfig(level=logging.DEBUG)
    logger.info("Time to encode some molecules!")

    augment_parquets_shard("data/pubchem/parquet/")
 