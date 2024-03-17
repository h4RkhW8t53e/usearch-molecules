
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
    


