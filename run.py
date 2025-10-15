import sys
import copy
import time
import re
import random
import pandas as pd
import os
import traceback
import numpy as np
import json
import fire
import pickle
from datetime import datetime
from tqdm import tqdm
from anthropic import Anthropic
from typing import Dict, List, Tuple
from pprint import pprint

from liddia.action import *
from liddia.environment import *
from liddia.evaluate import *
from liddia.memory import *
from liddia.prompt_template import *
from liddia.utils import *
from liddia.agent import *

with open("my-anthropic-key.txt", "r") as f:
    ANTHROPIC_API_KEY = f.read()

def main(target: str = "ABCC8",
         log_dir: str = "log",
         max_iter: int = 10,
         model: str = "claude-3-5-sonnet-20241022",
         env_dir: str = "./dataset/pdb",
         drug_dir: str = "./dataset/drugs_out.csv",):
    #region Task 
    drugs = pd.read_csv(drug_dir)
    drugs_exp = drugs[["NAME", "QED", "SAScore", "Lipinski Rules Followed", "Vina Score"]].groupby("NAME").mean().reset_index()
    drugs_exp["Lipinski"] = drugs_exp["Lipinski Rules Followed"]
    task = {}
    for _, row in drugs_exp.iterrows():
        if row["NAME"] == target:
            reqs = []
            reqs.append(f"At least {5} molecules")
            reqs.append(f"Vina Score must be lower than {row['Vina Score']:.2f}")
            reqs.append(f"Novelty must be at least {0.8:.2f}")
            reqs.append(f"Diversity must be at least {0.8:.2f}")
            reqs.append(f"QED must be better than {row['QED']:.2f}")
            reqs.append(f"SAScore must be better than {row['SAScore']:.2f}")
            reqs.append(f"Lipinski must be better than or at least {row['Lipinski']:.2f}")
            task = {"target": row["NAME"], "requirements": reqs, "pocket": f"{row['NAME']}.pdb", "drugs": drugs[drugs["NAME"] == row["NAME"]]["SMILES"].tolist()}
            break
    if len(task) == 0:
        raise Exception(f"Cannot found {target} on existing drugs")
    task["resource"] = max_iter
    task["metrics"] = {"size": "Size", "qed": "QED", "sascore": "SAScore", "lipinski": "Lipinski", "diversity": "Diversity", "novelty": "Novelty", "vina": "Vina Score"} #make sure to change prompt_template if changing metrics (mol_fmt and input_code_fmt)
    #endregion

    #region Agent
    agent = Claude(key=ANTHROPIC_API_KEY, model=model)
    #endregion

    #region loop
    os.makedirs(log_dir, exist_ok=True)
    run_id = str(datetime.now().strftime("%y-%m-%d-%H-%M-%S")) + f"_{target}"
    os.makedirs(os.path.join(log_dir, run_id), exist_ok=True)
    path_to_log = os.path.join(log_dir, run_id)

    mol_dicts = []
    pocket_dicts = [{"filename": task["pocket"],
                     "desc": f"This is a pocket for the target {task['target']}"}]
    action_dicts = [{"action_id": "GENERATE",
                     "desc": "This action runs a structured-based drug design model to generate at least 100 molecules using the target's pocket. The input is a pocket, such as ['POCKET001'].",
                    #  "func": sample_pocket2mol,
                     "func": sample_zinc,
                     "cost": "1"},
                    {"action_id": "OPTIMIZE",
                     "desc": "This action use a genetic algorithm with mutation and crossover operations on the starting population. The action uses the graph representations of the compounds to optimize one property. The action will generate at least 100 molecules. The input is a molecule set and the property, such as ['MOL001', 'QED'] or ['MOL001', 'SAScore']. Property must be one of ['QED', 'SAScore', 'Vina Score'].",
                     "func": graph_ga_optimizer,
                     "cost": "1"},
                    {"action_id": "CODE",
                     "desc": "This action runs a python code on existing molecules. This action can be used to search or combine existing molecules based on some criterias. You need to provide the criteria yourself. The input can be a molecule set or a list of molecule sets, such as ['MOL001'] or ['MOL001', 'MOL002']. The output should be a molecule set.",
                     "func": run_code,
                     "cost": "1"}]

    memory = Memory()
    for mol in mol_dicts:
        raise NotImplementedError
    for pocket in pocket_dicts:
        memory.add_pocket(filename=pocket["filename"], desc=pocket["desc"])
    for action in action_dicts:
        memory.add_action(**action)

    #Agent Loop
    logger = {"model": model}
    resource = task["resource"]
    pbar = tqdm(range(max_iter))
    eval_str = ""
    try:
        for n_iter in pbar:
            logger[n_iter] = {}
            
            #CREATE CONTEXT
            context_mol_dicts, context_pocket_dicts, context_action_dicts = [], [], []
            for key, val in memory.stream.items():
                if val["type"] == "POCKET":
                    context_pocket_dicts.append({"id": key, "desc": val["desc"]})
                elif val["type"] == "MOL":
                    context_mol_dicts.append({"id": key, "metrics": val["metrics"]})
                else:
                    context_action_dicts.append({"id": key, "desc": val["desc"], "cost": val["cost"]})
            mol_str = get_mol_str(context_mol_dicts, mol_fmt)
            pocket_str = get_pocket_str(context_pocket_dicts, pocket_fmt)
            action_str = get_action_str(context_action_dicts, action_fmt)
            req_str = get_req_str(task["requirements"], req_fmt)
            history_str = get_history_str(memory=memory)
            resource_str = str(resource) + " action"
            input_prompt = input_fmt.format(mol_str=mol_str, pocket_str=pocket_str, action_str=action_str, req_str=req_str, resource_str=resource_str, history_str=history_str, eval_str=eval_str)
            logger[n_iter]["input_prompt"] = input_prompt
            
            #RUN AGENT
            response, _ = get_response(input_prompt, agent)
            logger[n_iter]["response"] = response

            #RUN ACTION
            action_id, action_input = get_metadata_from_response(response)
            logger[n_iter]["action"] = (action_id, action_input)
            if "CODE" in action_id:
                desc = get_desc_from_response(response)
                action_input += [desc]
            action_output, cost, metadata = run_action(action_id, action_input, memory=memory, agent=agent, metrics=task["metrics"], target_pdb=task["pocket"],  drugs=task["drugs"], env_dir=env_dir, log_dir=path_to_log)
            logger[n_iter]["action_output"] = action_output
            resource = resource - cost
            memory.add_history(action_id=action_id, action_input=action_input, action_output=action_output, metadata=metadata)
    
            #EVALUATE
            if action_output in memory.stream.keys():
                #stop or not
                goal_mol_str = get_mol_str([{"id": action_output, "metrics": memory.stream[action_output]["metrics"]}], mol_fmt)
                input_goal_prompt = check_goal_fmt.format(mol_str=goal_mol_str, req_str=req_str)
                goal_response, _ = get_response(input_goal_prompt, agent)
                logger[n_iter]["input_goal_prompt"] = input_goal_prompt
                answer, goal_reason = get_goal_answer_response(goal_response)
                logger[n_iter]["goal_response"] = goal_response
                if answer == "YES":
                    logger["success"] = True
                    break
            
            #UPDATE
            if action_output in memory.stream.keys():
                eval_str = f"{action_output}: {goal_reason}"

            #OTHER
            if resource <= 0:
                logger["success"] = False
                break
        if "success" not in logger.keys():
            logger["success"] = False
    except Exception as e:
        logger["success"] = False
        logger["error_message"] = traceback.format_exc() + str(e)
    
    logger["task"] = task

    with open(os.path.join(log_dir, run_id, f"{task['target']}.json"), "w") as file:
        json.dump(logger, file, indent=4)  # indent for pretty formatting
    with open(os.path.join(log_dir, run_id, f"{task['target']}_memory.pkl"), 'wb') as f:
        pickle.dump(memory, f)
    with open(os.path.join(log_dir, run_id, f"{task['target']}_agent_messages.pkl"), 'wb') as f:
        pickle.dump(agent.history, f)
    #endregion


if __name__ == "__main__":
    fire.Fire(main)