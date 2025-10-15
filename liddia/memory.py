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

class Memory(object):
    static_id = 0
    history_id = 0
    def __init__(self):
        self.stream = {}
        self.history = []

    def _generate_id(self, identifier: str):
        val = identifier + f"{self.static_id:03}"
        self.static_id += 1
        return val

    def _add_block(self, mem_id: str, block):
        self.stream[mem_id] = block

    def add_mol(self, data: pd.DataFrame, return_id: bool = False, **kwargs):
        mem_id = self._generate_id("MOL")
        block = {"type": "MOL", "data": data}
        for key, val in kwargs.items():
            block[key] = val
        self._add_block(mem_id, block)
        if return_id:
            return mem_id

    def add_pocket(self, filename: str, return_id: bool = False, **kwargs):
        mem_id = self._generate_id("POCKET")
        block = {"type": "POCKET", "filename": filename}
        for key, val in kwargs.items():
            block[key] = val
        self._add_block(mem_id, block)
        if return_id:
            return mem_id

    def add_action(self, action_id: str, desc: str, return_id: bool = False, **kwargs):
        if "id" in kwargs.keys():
            mem_id = kwargs["id"]
        else:
            mem_id = self._generate_id(action_id)  
        block = {"type": action_id, "desc": desc}    
        for key, val in kwargs.items():
            block[key] = val  
        self._add_block(mem_id, block)
        if return_id:
            return mem_id

    def add_history(self, **kwargs):
        block = {"id": self.history_id}
        for key, val in kwargs.items():
            block[key] = val
        self.history.append(block)  
        self.history_id += 1