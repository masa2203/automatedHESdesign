import os
import pickle
import time
import warnings
from typing import Optional

import optuna
import pandas as pd
import json
import numpy as np

from stable_baselines3.common.vec_env import VecNormalize, DummyVecEnv

from envs.gt_model import A35
from envs.storage_model import BESS, DODDegradingBESS

from utils.utilities import start_count


def get_env_log_data(env, mean_reward, start_time):
    """
    Gets the data to be logged for a given environment.

    :param env: A gym environment.
    :param mean_reward: A float that represents the mean reward.
    :param start_time: A float that represents the start time.
    :return: A dictionary that contains the data to be logged.
    """
    gt_classes = [A35]
    bes_classes = [BESS, DODDegradingBESS]
    gts_used = []  # To be populated with GT types present in the environment

    # Environment wrapped for stable-baslines3
    if isinstance(env, VecNormalize) or isinstance(env, DummyVecEnv):
        episode_info = env.unwrapped.envs[0].unwrapped.return_episode_info()
        # Check for GTs and collect their types
        if (hasattr(env.unwrapped.envs[0].unwrapped, 'gt') and
                isinstance(env.unwrapped.envs[0].unwrapped.gt, tuple(gt_classes))):
            gts_used.append(env.unwrapped.envs[0].unwrapped.gt)
        elif hasattr(env.unwrapped.envs[0].unwrapped, 'gts'):
            for gt in env.unwrapped.envs[0].unwrapped.gts:
                if isinstance(gt, tuple(gt_classes)):
                    gts_used.append(gt)
        has_gt = len(gts_used) > 0

        has_bes = (hasattr(env.unwrapped.envs[0].unwrapped, 'storage') and
                   isinstance(env.unwrapped.envs[0].unwrapped.storage, tuple(bes_classes)))

    # Unwrapped environment
    else:
        episode_info = env.return_episode_info()
        # GT
        if hasattr(env, 'gt') and isinstance(env.gt, tuple(gt_classes)):
            gts_used.append(env.gt)
        if hasattr(env, 'gts'):
            for gt in env.gts:
                if isinstance(gt, tuple(gt_classes)):
                    gts_used.append(gt)
        has_gt = len(gts_used) > 0
        has_bes = hasattr(env, 'storage') and isinstance(env.storage, tuple(bes_classes))

    stats = {
        'reward_sum': mean_reward,
        'compute_time': time.time() - start_time,
    }

    # If GT(s) is/are part of the env, order of actions: [GT1, GT2, ..., BES]
    if has_gt:
        non_gt_actions = has_bes
        if non_gt_actions > 0:
            gt_actions = np.array([a[:-non_gt_actions] for a in episode_info['actions']])
        else:
            gt_actions = np.array(episode_info['actions'])

        # Ensure gt_actions is at least 2D (where the second dimension is GTs)
        if gt_actions.ndim == 1:
            gt_actions = np.expand_dims(gt_actions, axis=-1)  # Make single GT actions 2D

        # Get list of GT powers
        if isinstance(episode_info['gt_powers'][0], list):
            gt_powers = list(map(list, zip(*episode_info['gt_powers'])))
        else:
            gt_powers = [episode_info['gt_powers']]

        # Calculate average loads by action and power
        # (can differ significantly if gas is limited, else minor differences possible)
        avg_gt_action = [np.mean(gt) for gt in gt_actions.T]
        avg_gt_action_when_on = [np.mean([action for action in gt if action > 0]) for gt in gt_actions.T]
        avg_gt_load_by_power = [sum(powers) / len(powers) / gt.max_power for powers, gt in zip(gt_powers, gts_used)]
        op_steps = [sum(1 for t in gt if t > 0) for gt in gt_powers]
        avg_gt_load_by_power_when_on = [
            sum(powers) / ops / gt.max_power if ops > 0 else None
            for powers, ops, gt in zip(gt_powers, op_steps, gts_used)
        ]

        # Calculate operating hours and starts
        # gt_oper_time = [sum(1 for action in gt if action > 0) for gt in gt_actions.T]  # purely action based
        # number_of_starts = [start_count(gt) for gt in gt_actions.T]  # purely action based
        gt_oper_time = [sum(1 for action in gt if action > 0) for gt in gt_powers]  # power based
        number_of_starts = [start_count(gt) for gt in gt_powers]  # power based

        # Get list of sums by GT for fuel, carbon tax, maintenance
        if isinstance(episode_info['fuel_costs'][0], list):
            gt_fuel_costs = list(map(list, zip(*episode_info['fuel_costs'])))
            fuel_cost_sum = [sum(gt) for gt in gt_fuel_costs]
        else:
            fuel_cost_sum = [sum(episode_info['fuel_costs'])]

        if isinstance(episode_info['carbon_taxes'][0], list):
            gt_carbon_taxes = list(map(list, zip(*episode_info['carbon_taxes'])))
            carbon_tax_sum = [sum(gt) for gt in gt_carbon_taxes]
        else:
            carbon_tax_sum = [sum(episode_info['carbon_taxes'])]

        if isinstance(episode_info['maintenance_costs'][0], list):
            gt_maintenance_costs = list(map(list, zip(*episode_info['maintenance_costs'])))
            maint_cost_sum = [sum(gt) for gt in gt_maintenance_costs]
        else:
            maint_cost_sum = [sum(episode_info['maintenance_costs'])]

        gt_stats = {
            'fuel_cost_sum': fuel_cost_sum,
            'carbon_tax_sum': carbon_tax_sum,
            'maint_cost_sum': maint_cost_sum,
            'avg_gt_action': avg_gt_action,
            'avg_gt_action_when_on': avg_gt_action_when_on,
            'avg_gt_load_by_power': avg_gt_load_by_power,
            'avg_gt_load_by_power_when_on': avg_gt_load_by_power_when_on,
            'operating_time_GT': gt_oper_time,
            'number_of_starts': number_of_starts,
        }
        stats = stats | gt_stats

    # If BES is part of the env
    if has_bes:
        discharge_count = len(list(filter(lambda x: (x > 0), episode_info['bes_power_flows'])))
        charge_count = len(list(filter(lambda x: (x < 0), episode_info['bes_power_flows'])))

        bes_stats = {
            'degr_cost_sum': sum(episode_info['degr_costs']),
            'avg_soc': sum(episode_info['socs']) / len(episode_info['socs']),
            'num_charging': charge_count,
            'num_discharging': discharge_count,
        }
        stats = stats | bes_stats

    # If balances were tracked
    if episode_info['e_balances']:
        oversupply_values = list(filter(lambda x: (x > 0), episode_info['e_balances']))
        num_oversupply = len(oversupply_values)
        avg_oversupply = sum(oversupply_values) / num_oversupply if num_oversupply > 0 else 0

        undersupply_values = list(filter(lambda x: (x < 0), episode_info['e_balances']))
        num_undersupply = len(undersupply_values)
        avg_undersupply = sum(undersupply_values) / num_undersupply if num_undersupply > 0 else 0

        demand_balancing_stats = {
            'num_oversupply': num_oversupply,
            'avg_oversupply': avg_oversupply,
            'num_undersupply': num_undersupply,
            'avg_undersupply': avg_undersupply,
        }
        stats = stats | demand_balancing_stats

    # Add tracked time-series
    log_data = stats | episode_info

    return log_data


def sb3_create_stats_file(path, exp_params, style: str = 'eval'):
    """
    Creates a CSV file that contains the training and evaluation statistics of multiple independent runs
    based on stable-baseline3's monitor-files.

    :param path: A string that represents the path to the directory where the CSV file will be created.
    :param exp_params: A dictionary that contains the experiment parameters.
    :param style: A string that can be 'eval' or 'train' to indicate which file to create.
    """
    assert style == 'eval' or style == 'train', 'Valid styles are "eval" and "train"!'

    def make_stats_frame(monitor_df: pd.DataFrame):
        """Returns a pd.Dataframe with episodes and steps to be populated with rewards."""
        len_mon = len(monitor_df) - 1  # -1 due to monitor header
        n_eps = exp_params['n_episodes']
        data = {}
        if style == 'train':
            if n_eps + 1 != len_mon:  # +1 due to 'mandatory' final eval episode
                warnings.warn('Mismatch between chosen and conducted episodes.')
            n_cols = len_mon
            data = {
                'episodes': [i for i in range(n_cols)],
                'steps': [i * exp_params['len_episode'] for i in range(n_cols)]
            }
        elif style == 'eval':
            if int(n_eps * exp_params['len_episode'] / exp_params['eval_freq']) + 1 != len_mon:
                print('In here')
                warnings.warn('Mismatch between chosen and conducted episodes.')
            n_cols = len_mon
            data = {
                'episodes': [i for i in range(n_cols)],
                'steps': [i * exp_params['eval_freq'] for i in range(n_cols)]
            }
        else:
            warnings.warn('Unsupported stats style chosen. Created stats file with no columns. '
                          'Supported styles are "train" and "valid"')

        frame = pd.DataFrame(data).T
        return frame

    stats_frame = None
    count = 0
    for subdir, dirs, files in os.walk(path):
        for file in files:
            if file.endswith('monitor.csv') and style in file:
                df = pd.read_csv(os.path.join(subdir, file))
                stats_frame = make_stats_frame(df) if stats_frame is None else stats_frame
                stats_frame.loc[count] = [float(i) for i in df.index.values.tolist()[1:]]
                count += 1

    numerical_rows = stats_frame.iloc[2:]  # This excludes the first two non-numerical rows

    # Calculate mean and standard deviation for the numerical part
    mean_row = numerical_rows.mean(axis=0)
    std_row = numerical_rows.std(axis=0)

    # Add these as new rows to the DataFrame
    stats_frame.loc['mean'] = mean_row
    stats_frame.loc['std'] = std_row

    stats_frame.to_csv(os.path.join(path, style+'_stats.csv'))

    if style == 'eval':
        # Print evaluation stats if available
        print('Mean episodic rewards over all evaluation runs: ')
        print(mean_row)


def get_reward_stats_from_output(
        path: str,
):
    """
    Saves and prints mean and standard deviation of demo env and eval env rewards across output.json files.

    :param path: A string representing the path to the experiment folder.
    """
    eval_reward_sums = []
    demo_reward_sums = []
    # Iterate through all files in the folder and subfolders
    for root, _, files in os.walk(path):
        for file_name in files:
            if file_name.endswith("output.json"):  # Check if the file is a .json file
                file_path = os.path.join(root, file_name)
                # Open and read the JSON file
                with open(file_path, 'r') as json_file:
                    data = json.load(json_file)
                    # Extract the reward_sum value if it exists
                    if "reward_sum" in data:
                        eval_reward_sums.append(data["reward_sum"])
                    # Extract the demo_env_reward_sum value if it exists
                    if "demo_env_reward_sum" in data:
                        demo_reward_sums.append(data["demo_env_reward_sum"])

    stats = {
        'eval_env_reward_mean': np.mean(eval_reward_sums),
        'eval_env_reward_std': np.std(eval_reward_sums),
        'demo_env_reward_mean': np.mean(demo_reward_sums),
        'demo_env_reward_std': np.std(demo_reward_sums),
    }

    print('\nStats on demo and eval environments:')
    print('\t', stats)

    # Save the dictionary as a JSON file
    with open(os.path.join(path, 'reward_stats.json'), 'w') as json_file:
        json.dump(stats, json_file)


def print_and_save_tune_logs(study: optuna.study,
                             save_path: Optional[str],
                             run_id: str
                             ):
    """
    Prints and saves the results of an Optuna study.

    :param study: An optuna study object.
    :param save_path: A string that represents the path to the directory where the results will be saved.
    :param run_id: A string that represents the ID of the run.
    """
    print("Number of finished trials: ", len(study.trials))

    print("Best trial:")
    best_trial = study.best_trial

    print(f"  Value: {best_trial.value}")

    print("  Params: ")
    for key, value in best_trial.params.items():
        print(f"    {key}: {value}")

    print("  User attrs:")
    for key, value in best_trial.user_attrs.items():
        print(f"    {key}: {value}")

    if save_path is not None:
        # Write report
        study.trials_dataframe().to_csv(os.path.join(save_path, "results.csv"))
        # Save sampler
        with open(os.path.join(save_path, "sampler.pkl"), "wb") as fout:
            pickle.dump(study.sampler, fout)