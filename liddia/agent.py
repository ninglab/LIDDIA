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

from .action import *

class Claude():
    def __init__(self, key, model="claude-3-5-sonnet-20241022"):
        self.timestamp = 0
        self.model = model
        self.key = key
        self.history = []
        
    def get_response(self, messages):
        client = Anthropic(api_key=self.key)
        while self.timestamp > (time.time() - 30):
            time.sleep(5)
        self.timestamp = time.time()
        if messages[0]["role"] == "system":
            self.history.append(messages)
            return "\n".join([r.text for r in client.messages.create(
                max_tokens=2048,
                system=messages[0]["content"],
                messages=messages[1:],
                model=self.model
            ).content])
        raise Exception("messages must start with system")

    def count_tokens(self, text):
        client = Anthropic(api_key=self.key)
        return client.count_tokens(text)

from openai import OpenAI
import time

class DeepSeekR1():
    def __init__(self, key, model="deepseek-reasoner"):
        self.timestamp = 0
        self.model = model
        self.key = key
        self.history = []
    
    def get_response(self, messages, retry=True):
        client = OpenAI(api_key=self.key, base_url="https://api.deepseek.com")
        self.timestamp = time.time()
        try:
            result = client.chat.completions.create(
                stop=["<PAUSE>"],
                messages=messages,
                model=self.model
            ).choices[0].message.content
            return result
        except Exception as e:
            if retry:
                time.sleep(0.5)
                return self.get_response(messages, retry=False)
            else:
                raise e

    def count_tokens(self, text):
        return 0

class OpenAIAgent():
    def __init__(self, key, model):
        self.timestamp = 0
        self.model = model
        self.key = key
        self.history = []
    
    def get_response(self, messages, retry=True):
        client = OpenAI(api_key=self.key)
        self.timestamp = time.time()
        try:
            result = client.chat.completions.create(
                stop=["<PAUSE>"],
                messages=messages,
                model=self.model
            ).choices[0].message.content
            return result
        except Exception as e:
            if retry:
                time.sleep(0.5)
                return self.get_response(messages, retry=False)
            else:
                raise e

    def count_tokens(self, text):
        return 0