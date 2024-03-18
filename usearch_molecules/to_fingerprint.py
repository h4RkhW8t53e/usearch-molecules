
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


# pip install git+https://github.com/samoturk/mol2vec
from mol2vec.features import mol2alt_sentence, mol2sentence, MolSentence, DfVec #, sentences2vec
from gensim.models import word2vec


# mol2vec
# #Redefine function for gensim v.4

def sentences2vec(sentences, model, unseen=None):
    keys, vec = set(model.wv.index_to_key), [] #keys = set(model.wv.vocab.keys())
    if unseen: unseen_vec = model.wv.get_vector(unseen)
    for sentence in sentences:
        if unseen:
            vec.append(sum([model.wv.get_vector(y) if y in set(sentence) & keys else unseen_vec for y in sentence]))
        else:
            vec.append(sum([model.wv.get_vector(y) for y in sentence if y in set(sentence) & keys]))
    return np.array(vec)



# Load mol2vec 300-bit model

m2v_fname = dir + "data/mol2vec/model_300dim.pkl"
print("Mol2vec model: ",m2v_fname)
m2v_model = word2vec.Word2Vec.load(m2v_fname)


def molecule_to_mol2vec(x):
    sentences = MolSentence(mol2alt_sentence(x, 1))
    vec = sentences2vec([sentences], m2v_model, unseen='UNK')[0]
    return vec


def smiles_to_mol2vec(smiles: str,) -> np.ndarray:
    """Uses mol2vec representation."""

    molecule = Chem.MolFromSmiles(smiles)
    return molecule_to_mol2vec(molecule)
    


