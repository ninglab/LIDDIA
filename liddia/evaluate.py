import sys
import copy
import time
import re
import random
import pandas as pd
import os
import traceback
import numpy as np
import contextlib
import io
from tqdm import tqdm
from anthropic import Anthropic
from typing import Dict, List, Tuple
from pprint import pprint

from rdkit.Contrib.SA_Score import sascorer
from rdkit import Chem
from rdkit.Chem import QED
from rdkit.Chem import Descriptors
from rdkit.Chem import AllChem
from rdkit import DataStructs

_cache = {}
def tanimoto(smi1, smi2):
    if f"{smi1}.{smi2}" not in _cache:
        mol1 = Chem.MolFromSmiles(smi1)
        mol2 = Chem.MolFromSmiles(smi2)
        # Use 2048-bit Morgan fingerprints of radius 2
        fp1 = AllChem.GetMorganFingerprintAsBitVect(mol1, 2)
        fp2 = AllChem.GetMorganFingerprintAsBitVect(mol2, 2)
        # print(fp1.GetNumOnBits(), fp2.GetNumOnBits(), len(fp1), len(fp2))
        _cache[f"{smi1}.{smi2}"] = DataStructs.TanimotoSimilarity(fp1, fp2)
        _cache[f"{smi2}.{smi1}"] = DataStructs.TanimotoSimilarity(fp1, fp2)
    return _cache[f"{smi1}.{smi2}"]

def get_diversity(predicted : List[str]) -> Tuple[float, float]:
    diversity_scores = []
    for i in range(len(predicted)):
        for j in range(i + 1, len(predicted)):
            diversity_scores.append(1 - tanimoto(predicted[i], predicted[j]))
    return np.mean(diversity_scores), np.std(diversity_scores)

def get_novelty(smi: str, ligands : List[str]) -> float:
    max_tanimoto = max([tanimoto(smi, ligand) for ligand in ligands])
    return 1 - max_tanimoto

def get_sascore(smi):
    mol = Chem.MolFromSmiles(smi)
    return sascorer.calculateScore(mol)

def get_qed(smi):
    mol = Chem.MolFromSmiles(smi)
    return QED.qed(mol)

def get_vina(smi: str, target_pdb: str) -> float:
    from .docking.vina_docking import docking_score
    try:
        with contextlib.redirect_stderr(io.StringIO()):
            score = docking_score(ligand_smiles=smi, target_pdb=target_pdb, verbose=0)
    except Exception as e:
        traceback.print_exc()
        score = 0
    return score

def get_lipinski(smi : str):
    mol = Chem.MolFromSmiles(smi)
    score = 0
    if Descriptors.MolWt(mol) <= 500:
        score += 1
    if Chem.Crippen.MolLogP(mol) <= 5:
        score += 1
    if Chem.rdMolDescriptors.CalcNumHBD(mol) <= 5:
        score += 1
    if Chem.rdMolDescriptors.CalcNumHBA(mol) <= 10:
        score += 1
    return score

def evaluate(data: pd.DataFrame, metrics: Dict[str, str], **kwargs):
    return_metrics = {}
    smiles = data["SMILES"].tolist()
    for key, key_data in metrics.items():
        if key == "size":
            return_metrics[key] = len(data)
        elif key == "diversity":
            return_metrics[key] = get_diversity(smiles)[0]
        elif key == "novelty":
            if key_data not in data.columns:
                values = [get_novelty(smi, kwargs["drugs"]) for smi in smiles]
                data[key_data] = values
            return_metrics[key] = {"min": np.min(data[key_data]), "max": np.max(data[key_data]), "median": np.median(data[key_data])}
        elif key == "sascore":
            if key_data not in data.columns:
                values = [get_sascore(smi) for smi in smiles]
                data[key_data] = values
            return_metrics[key] = {"min": np.min(data[key_data]), "max": np.max(data[key_data]), "median": np.median(data[key_data])}
        elif key == "qed":
            if key_data not in data.columns:
                values = [get_qed(smi) for smi in smiles]
                data[key_data] = values
            return_metrics[key] = {"min": np.min(data[key_data]), "max": np.max(data[key_data]), "median": np.median(data[key_data])}
        elif key == "lipinski":
            if key_data not in data.columns:
                values = [get_lipinski(smi) for smi in smiles]
                data[key_data] = values
            return_metrics[key] = {"min": np.min(data[key_data]), "max": np.max(data[key_data]), "median": np.median(data[key_data])}
        elif key == "vina":
            path_to_pdb = os.path.join(kwargs["env_dir"], kwargs["target_pdb"])
            if key_data not in data.columns:
                values = [get_vina(smi, path_to_pdb) for smi in smiles]
                data[key_data] = values
            return_metrics[key] = {"min": np.min(data[key_data]), "max": np.max(data[key_data]), "median": np.median(data[key_data])}
        else:
            raise NotImplementedError
    return data, return_metrics