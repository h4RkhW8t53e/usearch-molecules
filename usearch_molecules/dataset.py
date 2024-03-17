from __future__ import annotations
import os
import random
from dataclasses import dataclass
from typing import Tuple, List, Optional

from tqdm import tqdm
import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

from usearch.index import Index, Matches, Key
import stringzilla as sz

#from usearch_molecules.to_fingerprint import (
from to_fingerprint import smiles_to_rdkit

SEED = 42  # For reproducibility
SHARD_SIZE = 1_000_000  # This would result in files between 150 and 300 MB
BATCH_SIZE = 100_000  # A good threshold to split insertions



def shard_name(dir: str, from_index: int, to_index: int, kind: str):
    return os.path.join(dir, kind, f"{from_index:0>10}-{to_index:0>10}.{kind}")


def write_table(table: pa.Table, path_out: os.PathLike):
    return pq.write_table(
        table,
        path_out,
        # Without compression the file size may be too large.
        # compression="NONE",
        write_statistics=False,
        store_schema=True,
        use_dictionary=False,
    )


@dataclass
class FingerprintedShard:
    """Potentially cached table and smiles path containing up to `SHARD_SIZE` entries."""

    first_key: int
    name: str

    table_path: os.PathLike
    smiles_path: os.PathLike
    table_cached: Optional[pa.Table] = None
    smiles_caches: Optional[sz.Strs] = None

    @property
    def is_complete(self) -> bool:
        return os.path.exists(self.table_path) and os.path.exists(self.smiles_path)

    @property
    def table(self) -> pa.Table:
        return self.load_table()

    @property
    def smiles(self) -> sz.Strs:
        return self.load_smiles()

    def load_table(self, columns=None, view=False) -> pa.Table:
        if not self.table_cached:
            self.table_cached = pq.read_table(self.table_path, memory_map=view, columns=columns,)
        return self.table_cached

    def load_smiles(self) -> sz.Strs:
        if not self.smiles_caches:
            self.smiles_caches = sz.Str(sz.File(self.smiles_path)).splitlines()
        return self.smiles_caches


@dataclass
class FingerprintedDataset:
    dir: os.PathLike
    shards: List[FingerprintedShard]
    index: Optional[Index] = None

    @staticmethod
    def open(dir: os.PathLike, max_shards: Optional[int] = None,) -> FingerprintedDataset:
        """Gather a list of files forming the dataset."""

        if dir is None: return FingerprintedDataset(dir=None, shards=[])

        shards = []
        filenames = sorted(os.listdir(os.path.join(dir, "parquet")))
        if max_shards:
            filenames = filenames[:max_shards]

        for filename in tqdm(filenames, unit="shard"):
            if not filename.endswith(".parquet"): continue

            filename = filename.replace(".parquet", "")
            first_key = int(filename.split("-")[0])
            table_path = os.path.join(dir, "parquet", filename + ".parquet")
            smiles_path = os.path.join(dir, "smiles", filename + ".smi")

            shard = FingerprintedShard(first_key=first_key, name=filename, table_path=table_path, smiles_path=smiles_path,)
            shards.append(shard)

        print(f"Fetched {len(shards)} shards")

        index = None
        index_path = os.path.join(dir, 'index-rdkit.usearch')
        if os.path.exists(index_path): index = Index.restore(index_path)

        return FingerprintedDataset(dir=dir, shards=shards, index=index)


    def shard_containing(self, key: int) -> FingerprintedShard:
        for shard in self.shards:
            if shard.first_key <= key and key <= (shard.first_key + SHARD_SIZE): return shard


    def search(self, smiles: str, count: int = 10, log: bool = False,) -> List[Tuple[int, str, float]]:
        """Search for similar molecules in the whole dataset."""

        fingers = smiles_to_rdkit(smiles)
        entry_fingerprint = np.array(fingers, dtype=np.float16)
        
        results: Matches = self.index.search(entry_fingerprint, count, log=log)

        filtered_results = []
        for match in results:
            shard = self.shard_containing(match.key)
            row = int(match.key - shard.first_key)
            #result = str(shard.smiles[row])
            #table = shard.load_table(["smiles"])
            #result = str(table["smiles"][row])
            result = str(shard.load_table(["smiles"])["smiles"][row])   
            filtered_results.append((match.key, result, match.distance))

        return filtered_results

    def __len__(self) -> int:
        return len(self.index)

    def random_smiles(self) -> str:
        shard_idx = random.randint(0, len(self.shards) - 1)
        shard = self.shards[shard_idx]
        row = random.randint(0, len(shard.smiles) - 1)
        return str(shard.smiles[row])


if __name__ == "__main__":
    dataset = FingerprintedDataset.open("data/pubchem")
    dataset.search("C")
