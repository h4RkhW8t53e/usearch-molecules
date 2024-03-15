
from __future__ import annotations
import os
from dataclasses import dataclass
from typing import Tuple, List

import numpy as np


try:
    from rdkit import Chem
    from rdkit.Chem import AllChem, MACCSkeys, rdFingerprintGenerator
except ImportError:
    print("Can't fingerprint molecules without RDKit and JPype")



# RDKit fingerprints

_fpgen = rdFingerprintGenerator.GetRDKitFPGenerator(fpSize=16, countSimulation=False)
#smi = "CC(C)(C)c1ccc2occ(CC(=O)Nc3ccccc3F)c2c1"
#print(fpgen.GetCountFingerprintAsNumPy(Chem.MolFromSmiles(smi)))


def molecule_to_rdkit(x):
    global _fpgen
    return _fpgen.GetCountFingerprintAsNumPy(x)


def smiles_to_rdkit(smiles: str,) -> np.ndarray:
    """Uses RDKit to compute RDKit Count representation."""

    molecule = Chem.MolFromSmiles(smiles)
    return molecule_to_rdkit(molecule)
    
    """
    bitset = np.zeros(16, dtype=np.uint32)
    bitset = molecule_to_rdkit(molecule)
    bitset = np.packbits(bitset)
    return (bitset,)
    """




def molecule_to_maccs(x):
    return MACCSkeys.GenMACCSKeys(x)


def molecule_to_ecfp4(x):
    return AllChem.GetMorganFingerprintAsBitVect(x, 2, nBits=2048)


def molecule_to_fcfp4(x):
    return AllChem.GetMorganFingerprintAsBitVect(x, 2, nBits=2048, useFeatures=True)


    
def smiles_to_maccs_ecfp4_fcfp4(smiles: str,) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Uses RDKit to simultaneously compute MACCS, ECFP4, and FCFP4 representations."""

    molecule = Chem.MolFromSmiles(smiles)
    return (
        np.packbits(molecule_to_maccs(molecule)),
        np.packbits(molecule_to_ecfp4(molecule)),
        np.packbits(molecule_to_fcfp4(molecule)),
    )




@dataclass
class FingerprintShape:
    """Represents the shape of a hybrid fingerprint, potentially containing multiple concatenated bit-vectors."""

    include_maccs: bool = False
    include_ecfp4: bool = False
    include_fcfp4: bool = False
    nbytes_padding: int = 0

    @property
    def nbytes(self) -> int:
        return (
            self.include_maccs * 21
            + self.nbytes_padding
            + self.include_ecfp4 * 256
            + self.include_fcfp4 * 256
        )

    @property
    def nbits(self) -> int:
        return self.nbytes * 8

    @property
    def index_name(self) -> str:
        parts = ["index"]
        if self.include_maccs: parts.append("maccs")
        if self.include_ecfp4: parts.append("ecfp4")
        if self.include_fcfp4: parts.append("fcfp4")
        return "-".join(parts) + ".usearch"


shape_maccs = FingerprintShape(
    include_maccs=True,
    nbytes_padding=3,
)

shape_mixed = FingerprintShape(
    include_maccs=True,
    include_ecfp4=True,
    nbytes_padding=3,
)


