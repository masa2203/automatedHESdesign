import os
import random
import itertools
from typing import Optional, Union, Tuple, Any, Deque, List, Dict

import numpy as np
import torch


def set_seeds(seed):
    """
    Fixes the random seed for all relevant packages.

    :param seed: An integer that represents the seed to be set.
    """
    os.environ['PYTHONHASHSEED'] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    # torch.backends.cudnn.deterministic = True
    # torch.backends.cudnn.benchmark = True


def start_count(lst_of_actions_taken):
    """
    Calculates the number of GT starts from a list of GT dispatches.

    :param lst_of_actions_taken: A list that represents the GT dispatch history.
    :type lst_of_actions_taken: list
    :return: An integer that represents the number of GT starts.
    """
    count = 0
    count += 1 if lst_of_actions_taken[0] else 0
    for i in range(len(lst_of_actions_taken) - 1):
        if lst_of_actions_taken[i] == 0 and lst_of_actions_taken[i + 1] != 0:
            count += 1
    return count


def generate_discrete_actions(
        gt_specs: Optional[List[Dict[str, float]]] = None,
        bes_specs: Optional[List[Dict[str, float]]] = None
) -> List[np.ndarray]:
    """
    Generate discrete action space for GTs and/or BESs based on provided specifications.
    Can also generate actions for exclusively GTs or BESs if only one of them is provided.

    Parameters:
    gt_specs (Optional[List[Dict[str, float]]]): List of dictionaries, each containing
                                                 'start', 'stop', and 'num' for np.linspace for each GT.
    bes_specs (Optional[List[Dict[str, float]]]): List of dictionaries, each containing
                                                 'start', 'stop', and 'num' for np.linspace for each BES.

    Returns:
    List[np.ndarray]: A list containing numpy arrays of all possible action combinations.

    Example:
        gt_specs = [{'start': -1, 'stop': 1, 'num': 9}]
        bes_specs = [{'start': -1, 'stop': 1, 'num': 9}]
        discrete_actions = generate_discrete_actions(gt_specs, bes_specs)
    """
    # Initialize action lists
    gt_actions, bes_actions = [], []

    if gt_specs is not None:
        gt_actions = [np.linspace(**spec) for spec in gt_specs]

    if bes_specs is not None:
        bes_actions = [np.linspace(**spec) for spec in bes_specs]

    # Combine the action sets based on provided specs
    all_actions = []
    if gt_actions:
        all_actions += gt_actions
    if bes_actions:
        all_actions += bes_actions

    # Ensure there's at least one action set provided
    if not all_actions:
        raise ValueError("At least one of gt_specs or bes_specs must be provided.")

    # Get all combinations
    combinations = list(itertools.product(*all_actions))

    # Convert each combination tuple to a numpy array
    discrete_actions = [np.array(combination) for combination in combinations]

    return discrete_actions


def plant_config_train_test_split_res():
    """
    VERSION FOR HIGH RESOLUTION ENV
    Returns 2 pre-defined lists of dictionaries with different plant configurations;
    - training
    - testing (interpolation only)
    """
    set_seeds(22)  # Fix random seed to ensure receiving the same train/test combo

    config = {
        'num_wt': [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20],
        'rate_gas_price': [0.25, 0.5, 0.75, 1.0, 1.25, 1.5, 2.0, 2.5, 3.0],
        'penalty': [250, 500, 750, 1000, 1500, 2000, 2500, 3000],
        'bes_cap': [1, 25, 50, 75, 100, 125, 150],
        'bes_rate': [5, 10, 15, 20, 25, 30, 35, 40],
    }

    # Use itertools.product to generate all combinations
    all_combinations = list(itertools.product(*config.values()))

    dict_combinations = [
        dict(zip(config.keys(), combination)) for combination in all_combinations
    ]

    # Sample 10 unique dicts without replacement
    test_dicts = random.sample(dict_combinations, 10)

    # Form a list with the rest (excluding the 10 sampled)
    train_dicts = [d for d in dict_combinations if d not in test_dicts]

    # print(f"Number of sampled dicts: {len(test_dicts)}")
    # print(f"Number of dicts in the rest: {len(train_dicts)}")

    return train_dicts, test_dicts
