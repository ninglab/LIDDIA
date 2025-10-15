import sys
import copy
import time
import re
import random
import pandas as pd
import os
import traceback
import numpy as np
from tqdm import tqdm
from anthropic import Anthropic
from typing import Dict, List, Tuple
from pprint import pprint

def get_response(input_prompt: str, agent, sys_msg: str = "You are a chemist expert"):
    messages = []
    messages.append({"role": "system", "content": sys_msg})
    messages.append({"role": "user", "content": input_prompt})
    return agent.get_response(messages), messages

def get_goal_answer_response(response: str):
    answer = re.findall(r'Answer: (.+)', response)[0]
    reason = re.findall(r'Reason: (.+)', response)[0]
    return answer, reason

def get_metadata_from_response(response: str):
    import ast
    action = re.findall(r'Action: (.+\d\d\d)', response)[0]
    action_input = ast.literal_eval(re.findall(r'Input: (.+)', response)[0])
    # plan = re.findall(r'Plan: (.+)', response)[0]
    # reason = re.findall(r'Reasoning: (.+)', response)[0]
    return action, action_input

def get_desc_from_response(response: str):
    desc = re.findall(r'Desc: (.+)', response)[0]
    return desc

def get_cost(text: str):
    return re.findall(r'(\d+)', text)[-1]

def get_mol_str(mol_dicts, mol_fmt: str):
    mol_str = ""
    for item in mol_dicts:
        _to_metrics_str = {}
        for key, val in item["metrics"].items():
            if isinstance(val, dict):
                for val_key, val_val in val.items():
                    _to_metrics_str[key + "_" + val_key] = val_val
            else:
                _to_metrics_str[key] = val
        mol_str += mol_fmt.format(id=item["id"], **_to_metrics_str)
    return mol_str

def get_pocket_str(pocket_dicts, pocket_fmt: str):
    pocket_str = ""
    for item in pocket_dicts:
        pocket_str += pocket_fmt.format(id=item["id"], desc=item["desc"]) + "\n"
    return pocket_str

def get_action_str(action_dicts, action_fmt: str):
    action_str = ""
    for item in action_dicts:
        action_str += action_fmt.format(id=item["id"], desc=item["desc"], cost=item["cost"]) + "\n"
    return action_str

def get_req_str(reqs, req_fmt: str):
    req_str = ""
    for req in reqs:
        req_str += req_fmt.format(req=req) + "\n"
    return req_str

def get_history_str(memory):
    history_str = ""
    for item in memory.history:
        if "CODE" not in item["action_id"]:
            history_str += f"- Run {item['action_id']} on {', '.join(item['action_input'])}. The output is {item['action_output']}\n"
        else:
            history_str += f"- Run {item['action_id']} on {', '.join(item['action_input'][:-1])}. The output is {item['action_output']}. The code description: {item['action_input'][-1]}\n"
    return history_str