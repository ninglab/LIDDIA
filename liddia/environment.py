import sys
import copy
import time
import re
import random
import pandas as pd
import os
import traceback
import numpy as np
import ast
from tqdm import tqdm
from anthropic import Anthropic
from typing import Dict, List, Tuple
from pprint import pprint

from .evaluate import *
from .utils import *
from .prompt_template import *
from .action import *

def run_action(action_id, action_input, memory, agent, metrics, **kwargs):
    #run existing action
    action_block = memory.stream[action_id]
    if action_block["type"] == "GENERATE":
        input_block = memory.stream[action_input[0]]
        _input = input_block["filename"]
        input_pdb = os.path.join(kwargs["env_dir"], _input)
        outputs, metadata = action_block["func"](input_pdb=input_pdb, n_samples=100)
    elif action_block["type"] == "OPTIMIZE":
        input_block = memory.stream[action_input[0]]
        _input = input_block["data"]
        _property = action_input[1]
        output_dir = os.path.join(kwargs["log_dir"], "log_optimize")
        outputs, metadata = action_block["func"](_input, _property, n_outputs=100, output_dir=output_dir, **kwargs)   
    elif action_block["type"] == "CODE":
        _input = {}
        for key in action_input[:-1]:
                _input[key] = memory.stream[key]["data"].copy()
        _desc = action_input[-1]     
        if "input_code_fmt" in kwargs.keys():
            outputs, metadata = action_block["func"](_input, _desc, agent=agent, input_fmt=kwargs["input_code_fmt"])
        outputs, metadata = action_block["func"](_input, _desc, agent=agent)
    else:
        raise NotImplementedError
    
    if len(outputs) > 0:
        #run evaluation
        data, outputs_metrics = evaluate(outputs, metrics, **kwargs)
        #add mol to memory
        mol_id = memory.add_mol(data=data, action=action_id, metrics=outputs_metrics, return_id=True)
    else:
        mol_id = "EMPTY SET"
    #calculate cost
    cost = get_cost(action_block["cost"])
    return mol_id, int(cost), metadata