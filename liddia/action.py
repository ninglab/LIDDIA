import sys
import copy
import time
import re
import random
import pandas as pd
import os
import traceback
import numpy as np
import yaml
import shutil
from tqdm import tqdm
from anthropic import Anthropic
from typing import Dict, List, Tuple
from pprint import pprint

from .evaluate import *
from .prompt_template import *
from .utils import *

def sample_zinc(input_pdb: str, n_samples: int) -> pd.DataFrame:
    from tdc.generation import MolGen
    data = MolGen(name = 'ZINC')
    split = data.get_split()
    zinc = split["train"]["smiles"].tolist() + split["test"]["smiles"].tolist()
    metadata = {}
    return pd.DataFrame(random.sample(zinc, n_samples), columns=["SMILES"]), metadata

# def sample_pocket2mol(input_pdb: str, n_samples: int) -> pd.DataFrame:
#     from .apptainer.pocket2mol import pocket2mol
#     seed = np.random.randint(1, 1000000000)
#     outputs = pocket2mol(n_samples=n_samples, pocket_protein_pdb=input_pdb, seed=seed)
#     metadata = {"seed": seed}
#     return pd.DataFrame(outputs, columns=["SMILES"]), metadata

def run_code(action_input: Dict[str, pd.DataFrame], input_desc: str, agent, input_fmt: str = input_code_fmt, max_iter: int = 3) -> pd.DataFrame:
    _D = action_input
    for i in range(max_iter):
        response, _ = get_response(input_fmt.format(input_desc=input_desc), agent, "You are a Python coding expert")
        try:
            code = re.findall(r'```python\n([\w\W+]*)```', response)[0]
            print(code)
            exec(code, globals())
            metadata = {"code": code, "response": response}
            outputs = eval("_function(_D)")
            return outputs, metadata
        except Exception as e:
            traceback.format_exc()
            print(e)
            continue
    return pd.DataFrame(), metadata

def graph_ga_optimizer(input_df: pd.DataFrame, property: str, n_outputs: int, output_dir: str, **kwargs) -> pd.DataFrame:
    from molopt.graph_ga import GraphGA
    if property == "QED":
        _f = lambda x: get_qed(x)
        max_oracle_calls = 500
    elif property == "SAScore":
        _f = lambda x: 1-((get_sascore(x)-1)/9)
        max_oracle_calls = 300
    elif property == "Vina Score":
        _f = lambda x: -get_vina(x, os.path.join(kwargs["env_dir"], kwargs["target_pdb"]))
        max_oracle_calls = 300
    else:
        raise NotImplementedError
    def reward(smi):
        return _f(smi)
    seed = np.random.randint(1, 1000000000)
    smi_list = input_df["SMILES"].tolist()
    optimizer = GraphGA(smi_file=smi_list, n_jobs=-1, max_oracle_calls=max_oracle_calls, freq_log=100, output_dir = output_dir, log_results=False) 
    _buffer = {}
    if property in input_df.columns.tolist():
        for i, (_, row) in enumerate(input_df.iterrows()):
            if property == "Vina Score":
                _buffer[row["SMILES"]] = [-row[property], i+1]
            elif property == "SAScore":
                _buffer[row["SMILES"]] = [1-((row[property]-1)/9), i+1]
            else:
                _buffer[row["SMILES"]] = [row[property], i+1]
    optimizer.oracle.mol_buffer = _buffer
    optimizer.optimize(oracle=reward, patience=5, seed=seed)
    with open(os.path.join(output_dir, f"results_graph_ga_reward_{seed}.yaml"), "r") as f:
        content = yaml.safe_load(f)
    outputs = list(content.keys())[:n_outputs]
    metadata = {"seed": seed}
    if property == "Vina Score":
        scores = [-item[0] for item in list(content.values())][:n_outputs]
        return pd.DataFrame.from_dict({"SMILES": outputs, "Vina Score": scores}), metadata  
    else:
        return pd.DataFrame(outputs, columns=["SMILES"]), metadata