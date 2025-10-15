input_fmt = """You have access to the following molecules and pockets:
{pocket_str}{mol_str}
You also have access to a set of actions:
{action_str}
Your job is to find molecules that satisfy these requirements:
{req_str}
Here is a history of actions you have taken and the results:
{history_str}
Here is the evaluation result from previous iteration:
{eval_str}

Let's think step by step and take your time before you answer the question. What is the best action to take and what is the input of the action?

Remember that you currently have {resource_str} left to solve the task.
Remember that you can only use one action.

Your answer must follow this format:

Action: [name of action]
Input: [input of the action, should be the identifier like ['MOL001'] or ['POCKET001']]

If you plan to use "CODE" action, you need to include this additional format:

Desc: [explain what do you want to do with input of the action. Be as verbose and descriptive as possible but at most three sentences. Always refer the identifier of the action input. ]
"""

mol_fmt = """- Molecule Set {id}:
    Size: {size}
    Diversity: {diversity:.2f}
    Vina Score: Range {vina_min:.2f} to {vina_max:.2f}, Median {vina_median:.2f}
    Novelty: Range {novelty_min:.2f} to {novelty_max:.2f}, Median {novelty_median:.2f}
    Lipinski: Range {lipinski_min:.2f} to {lipinski_max:.2f}, Median {lipinski_median:.2f}
    QED: Range {qed_min:.2f} to {qed_max:.2f}, Median {qed_median:.2f}
    SAScore: Range {sascore_min:.2f} to {sascore_max:.2f}, Median {sascore_median:.2f}
"""

mol_fmt_1 = """- Molecule Set {id}:
    Size: {size}
    Vina Score: Range {vina_min:.2f} to {vina_max:.2f}, Median {vina_median:.2f}
    Novelty: Range {novelty_min:.2f} to {novelty_max:.2f}, Median {novelty_median:.2f}
    Lipinski: Range {lipinski_min:.2f} to {lipinski_max:.2f}, Median {lipinski_median:.2f}
    QED: Range {qed_min:.2f} to {qed_max:.2f}, Median {qed_median:.2f}
    SAScore: Range {sascore_min:.2f} to {sascore_max:.2f}, Median {sascore_median:.2f}
"""

mol_fmt_2 = """- Molecule Set {id}:
    Size: {size}
    Diversity: {diversity:.2f}
    Vina Score: Range {vina_min:.2f} to {vina_max:.2f}, Median {vina_median:.2f}
    Lipinski: Range {lipinski_min:.2f} to {lipinski_max:.2f}, Median {lipinski_median:.2f}
    QED: Range {qed_min:.2f} to {qed_max:.2f}, Median {qed_median:.2f}
    SAScore: Range {sascore_min:.2f} to {sascore_max:.2f}, Median {sascore_median:.2f}
"""

pocket_fmt = """- Pocket {id}:
    Description: {desc}"""

action_fmt = "- {id}: {desc}  Cost: {cost}"

req_fmt = "- {req}"

check_goal_fmt = """
You have access to the following pool of molecules:
{mol_str}
Your job is to find molecules that satisfy these requirements:
{req_str}
Does this pool of molecules satisfy the requirements?
Remember that all molecules in the pool must satisfy the requirements.

Let's think step by step and answer with the following format:
Reason: (a compact and brief one-sentence reasoning)
Answer: (YES or NO)
"""

eval_fmt = """
You have access to the following pool of molecules:
{mol_str}
Your job is to provide evaluation based on these requirements:
{req_str}
What do you think about the molecules?

Let's think step by step and answer with the following format:
Reason: (a compact and brief one-sentence reasoning)
Answer: (1, 2, 3, 4 or 5, with 5 being very good and 1 being very poor)
"""

input_code_fmt = """Your job is to make a Python function called _function.
The input is a Dict[str, pd.DataFrame] the following columns: 
The dataframe has these columns: [\"SMILES\", \"QED\", \"SAScore\", \"Lipinski\", \"Novelty\", \"Vina Score\"].
The output must be pandas DataFrame with the same columns as input.
The function should be able to do the following task: {input_desc}

Your output must follow the following format:

```python
import pandas as pd

def _function(Dict[str, pd.DataFrame]) -> pd.DataFrame:
    #---IMPORT LIBRARIES HERE---#
    #---IMPORT LIBRARIES HERE---#
    
    #---CODE HERE---#
    #---CODE HERE---#
    
    output_df = ...
    return output_df
```python

Make sure you import the necessary libraries.
"""

input_code_fmt_2 = """Your job is to make a Python function called _function.
The input is a Dict[str, pd.DataFrame] the following columns: 
The dataframe has these columns: [\"SMILES\", \"QED\", \"SAScore\", \"Lipinski\", \"Vina Score\"].
The output must be pandas DataFrame with the same columns as input.
The function should be able to do the following task: {input_desc}

Your output must follow the following format:

```python
import pandas as pd

def _function(Dict[str, pd.DataFrame]) -> pd.DataFrame:
    #---IMPORT LIBRARIES HERE---#
    #---IMPORT LIBRARIES HERE---#
    
    #---CODE HERE---#
    #---CODE HERE---#
    
    output_df = ...
    return output_df
```python

Make sure you import the necessary libraries.
"""