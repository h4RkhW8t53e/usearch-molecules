"""Exports all molecules from the PubChem, GDB13 and Enamine REAL datasets into Parquet shards, with up to 1 Million molecules in every granule."""
import os
import logging
from dataclasses import dataclass
from typing import List, Callable, Optional, Tuple
from multiprocessing import Process, cpu_count

import pyarrow as pa
from stringzilla import File, Strs, Str

from dataset import shard_name, write_table, SHARD_SIZE, SEED

logger = logging.getLogger(__name__)


@dataclass
class RawDataset:
    lines: Strs
    extractor: Callable


    def count_lines(self) -> int:
        return len(self.lines)

    def cid(self, row_idx: int) -> Optional[str]:
        return self.extractor(str(self.lines[row_idx]))

    def smiles(self, row_idx: int) -> Optional[str]:
        return self.extractor(str(self.lines[row_idx]))

    def smiles_slice(self, count_to_skip: int, max_count: int) -> List[Tuple[int, str, str]]:
        result = []
        count_lines = len(self.lines)
        for row_idx in range(count_to_skip, count_lines):
            smiles_cid = self.smiles(row_idx)
            if smiles_cid:
                smiles, cid = smiles_cid
                result.append((row_idx, smiles, cid))
                if len(result) >= max_count:
                    return result

        return result

    

def pubchem(dir: os.PathLike) -> RawDataset:
    """
    gzip -d CID-SMILES.gz
    """
    
    file = Str(File(os.path.join(dir, "test_1k.smi"))) 
    lines = file.splitlines()
    
    # Let's shuffle across all the files
    #lines.shuffle(SEED)

    def extractor(row: str) -> Optional[str]:
        row = row.strip("\n")
        if " " in row:
            row = row.split(" ") #row.split(" ")[0]
            return row
        return None

    return RawDataset(
        lines=lines,
        extractor=extractor,
    )




def export_parquet_shard(dataset: RawDataset, dir: os.PathLike, shard_index: int, shards_count: int, rows_per_part: int = SHARD_SIZE,):
    os.makedirs(os.path.join(dir, "parquet"), exist_ok=True)

    try:
        lines_count = dataset.count_lines()
        logger.info(f"Loaded {lines_count:,} lines")
        first_epoch_offset = shard_index * rows_per_part
        epoch_size = shards_count * rows_per_part

        for start_row in range(first_epoch_offset, lines_count, epoch_size):
            end_row = start_row + rows_per_part

            rows_and_smiles = dataset.smiles_slice(start_row, rows_per_part)
            path_out = shard_name(dir, start_row, end_row, "parquet")
            if os.path.exists(path_out): continue

            try:
                dicts, dicts_cid = [], []
                for _, smiles, cid in rows_and_smiles:
                    try:
                        dicts.append({"smiles": smiles})
                        dicts_cid.append(cid)
                    except Exception:
                        continue

                schema = pa.schema([pa.field("smiles", pa.string(), nullable=False)])
                table = pa.Table.from_pylist(dicts, schema=schema)
                
                dicts_cid = pa.array(dicts_cid, pa.string())
                schema_cid = pa.field("cid", pa.string(), nullable=False) 
                table = table.append_column(schema_cid, dicts_cid)
                write_table(table, path_out)
                

            except KeyboardInterrupt as e:
                raise e

            shard_description = "Molecules {:,}-{:,} / {:,}. Process # {} / {}".format(start_row, end_row, lines_count, shard_index, shards_count)
            logger.info(f"Passed {shard_description}")

    except KeyboardInterrupt as e:
        logger.info(f"Stopping shard {shard_index} / {shards_count}")
        raise e



if __name__ == "__main__":
    logging.basicConfig(level=logging.DEBUG)
    logger.info("Time to pre-process some molecules!")


    processes = 1

    export_parquet_shard(pubchem("data/pubchem"), "data/pubchem", 0, 1)
