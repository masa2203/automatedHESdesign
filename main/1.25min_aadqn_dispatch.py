import os
import sys
import json
import random
import socket
import time
from typing import Any

from stable_baselines3 import DQN
from stable_baselines3.common.callbacks import EvalCallback
from stable_baselines3.common.logger import configure

# Get source directory
file_dir = os.path.dirname(os.path.abspath(__file__))
src_dir = os.path.abspath(os.path.join(file_dir, os.pardir))
sys.path.append(src_dir)

from envs.environments import *
from envs.env_params import *
from utils.callbacks import ProgressBarManager
from utils.make_env import make_env
from utils.net_design import *
from utils.scheduler import linear_scheduler_sb3
from utils.utilities import set_seeds, plant_config_train_test_split_res
from utils.logger import sb3_create_stats_file, get_env_log_data

# PLANT PARAMS
ENV = GasTurbineBatteryRenewablesDemandEnv
ENV_KWARGS = on_2016_1min
TEST_ENV_KWARGS = on_3y_test_1min

# LOG
CREATE_LOG = True
SAVE_MODEL = True
VERBOSE = 0
LOGGER_TYPE = ["csv"]
SAVE_PATH = os.path.join(src_dir, 'log', ENV_KWARGS['env_name'], 'aadqn', 'run', 'varied_configs',
                         f'{ENV_KWARGS["env_name"]}_to_{TEST_ENV_KWARGS["env_name"]}_test') if CREATE_LOG else None

# ACTIONS - USES PREDEFINED DISCRETE ACTIONS
discretization_params = 'predefined'
DISCRETE_ACTIONS = None

# EXP PARAMS
EXP_PARAMS = {
    'n_runs': 1,
    'n_episodes': 200,
    'num_train_envs': 400,
    # 'random_episodes': False,
    'random_episodes': {'mode': 'month', 'num': 20},
    'len_episode': int(ENV_KWARGS['modeling_period_h'] / ENV_KWARGS['resolution_h']),
    'seed': 22,
    # Env
    'use_predefined_action_wrapper': 'on_default',  # USES PREDEFINED DISCRETE ACTIONS
    'flatten_obs': True,
    'frame_stack': 18,
    # Normalization
    'minmax_scaling': True,
    'norm_obs': False,
    'norm_reward': True,
    # Evaluation
    'eval_while_training': True,
    'eval_freq': int(ENV_KWARGS['modeling_period_h'] / ENV_KWARGS['resolution_h']) * 50,
    # Penalties/Reward Modifiers
    'bes_soc_penalty': (100, 0.25),  # Tuple (Penalty weight, upper SOC limit)
    'complete_gt_start': True,
    'delay_gt_shutdown': 4,
    'forecasts_from_file': dict(
        forecasted_vars=['pred_demand', 'pred_wind_power']
    )
}

RL_PARAMS: dict[str, Any] = {
    'policy': "MlpPolicy" if EXP_PARAMS['flatten_obs'] else 'MultiInputPolicy',
    'learning_rate': 9e-5,  # Default: 1e-4
    'use_lr_sched': True,
    'buffer_size': 1_000_000,  # Default: 1e6
    'learning_starts': 1_000,  # Default: 50_000
    'batch_size': 128,  # Default: 32
    'tau': 0.35,  # Default: 1.0
    'gamma': 0.993,  # Default: 0.99
    'train_freq': 1,  # Default: 4
    'gradient_steps': -1,  # Default: 1
    'target_update_interval': 1_000,  # Default: 1e4
    'exploration_fraction': 0.5,  # Default: 0.1
    'exploration_initial_eps': 0.8,  # Default: 1.0
    'exploration_final_eps': 0.01,  # Default: 0.05
    'max_grad_norm': 0.2,  # Default: 10

    'policy_kwargs': {
        # Defaults reported for MultiInputPolicy
        'net_arch': 'extra_large',  # Default: None
        'activation_fn': 'tanh',  # Default: tanh
    }
}

train, test = plant_config_train_test_split_res()

if SAVE_PATH is not None:
    os.makedirs(SAVE_PATH, exist_ok=True)
    with open(os.path.join(SAVE_PATH, 'inputs.json'), 'w') as f:
        json.dump({
            'DISCRETE_ACTIONS': discretization_params,
            'EXP_PARAMS': EXP_PARAMS,
            'RL_PARAMS': str(RL_PARAMS),
            'PLANT_PARAMS': ENV_KWARGS,
            'TEST_ENV_PARAMS': TEST_ENV_KWARGS,
            'TEST_CONFIGS': test,
            'PC_NAME': socket.gethostname(),
        }, f)

# Handling of schedulers (remove from dict and apply function)
if 'use_lr_sched' in RL_PARAMS:
    use_lr_sched = RL_PARAMS.pop('use_lr_sched')
    if use_lr_sched:
        RL_PARAMS['learning_rate'] = linear_scheduler_sb3(RL_PARAMS['learning_rate'])

RL_PARAMS['policy_kwargs']['net_arch'] = net_arch_dict[RL_PARAMS['policy_kwargs']['net_arch']]['qf']
RL_PARAMS['policy_kwargs']['activation_fn'] = activation_fn_dict[RL_PARAMS['policy_kwargs']['activation_fn']]

print('|| Number of envs: {} ||'.format(EXP_PARAMS['num_train_envs']))

# Training logic for single run
for run in range(EXP_PARAMS['n_runs']):
    # total timesteps
    tt = int(EXP_PARAMS['n_episodes'] * ENV_KWARGS['modeling_period_h'] / ENV_KWARGS['resolution_h'])

    # Create path for logging
    run_path = os.path.join(SAVE_PATH, f'run_{run}') if SAVE_PATH is not None else None
    if SAVE_PATH is not None:
        os.makedirs(run_path, exist_ok=True)

    # Update seed
    seed = EXP_PARAMS['seed'] + run
    set_seeds(seed)
    print('\n|| Run #{} | Seed #{} ||'.format(run, seed))

    start = time.time()

    # CREATE ENVIRONMENT
    train_env = make_env(env=ENV,
                         env_kwargs=ENV_KWARGS,
                         path=os.path.join(run_path, "train_monitor.csv") if run_path is not None else None,
                         forecasts_from_file=EXP_PARAMS.get('forecasts_from_file'),
                         use_random_episodes=EXP_PARAMS['random_episodes'],
                         update_plant_config=random.choices(train, k=EXP_PARAMS['num_train_envs']),
                         # update_plant_config=[
                         #     {'num_wt': 13, 'rate_gas_price': 1.0, 'penalty': 500, 'bes_cap': 75,
                         #      'bes_rate': 10}],  # default for ON
                         minmax_scaling=EXP_PARAMS['minmax_scaling'],
                         use_predefined_action_wrapper=EXP_PARAMS['use_predefined_action_wrapper'],
                         complete_gt_start=EXP_PARAMS.get('complete_gt_start', False),
                         delay_gt_shutdown=EXP_PARAMS.get('delay_gt_shutdown', None),
                         bes_soc_penalty=EXP_PARAMS.get('bes_soc_penalty', None),
                         flatten_obs=EXP_PARAMS['flatten_obs'],
                         discrete_actions=DISCRETE_ACTIONS,
                         frame_stack=EXP_PARAMS['frame_stack'],
                         norm_obs=EXP_PARAMS['norm_obs'],
                         norm_reward=EXP_PARAMS['norm_reward'],
                         gamma=RL_PARAMS['gamma'],
                         verbose=bool(VERBOSE),
                         )

    # DEFINE MODEL
    model = DQN(env=train_env, verbose=VERBOSE, seed=seed, **RL_PARAMS)
    if run_path is not None:
        logger = configure(run_path, LOGGER_TYPE)
        model.set_logger(logger)

    with ProgressBarManager(total_timesteps=tt) as callback:
        # Evaluation during training
        if EXP_PARAMS['eval_while_training']:
            eval_env = make_env(env=ENV,
                                env_kwargs=TEST_ENV_KWARGS,
                                path=os.path.join(run_path,
                                                  'eval_monitor.csv') if run_path is not None else None,
                                forecasts_from_file=EXP_PARAMS.get('forecasts_from_file'),
                                use_random_episodes=False,
                                update_plant_config=test,
                                minmax_scaling=EXP_PARAMS['minmax_scaling'],
                                use_predefined_action_wrapper=EXP_PARAMS['use_predefined_action_wrapper'],
                                complete_gt_start=EXP_PARAMS.get('complete_gt_start', False),
                                delay_gt_shutdown=EXP_PARAMS.get('delay_gt_shutdown', None),
                                flatten_obs=EXP_PARAMS['flatten_obs'],
                                discrete_actions=DISCRETE_ACTIONS,
                                frame_stack=EXP_PARAMS['frame_stack'],
                                norm_obs=EXP_PARAMS['norm_obs'],
                                norm_reward=EXP_PARAMS['norm_reward'],
                                gamma=RL_PARAMS['gamma'],
                                )

            eval_callback = EvalCallback(eval_env=eval_env,
                                         n_eval_episodes=len(test),
                                         eval_freq=EXP_PARAMS['eval_freq'],
                                         deterministic=True,
                                         # best_model_save_path=run_path,
                                         best_model_save_path=None,
                                         verbose=VERBOSE)

            model.learn(total_timesteps=tt, callback=[eval_callback, callback])
        # Evaluation only after training
        else:
            model.learn(total_timesteps=tt, callback=callback)

    if run_path is not None:
        if SAVE_MODEL:
            model.save(os.path.join(run_path, 'model'))  # Save best model instead through callback
            train_env.save(os.path.join(run_path, 'env.pkl'))

    train_env.close()

    print()
    print(f'Execution time = {time.time() - start}s')

# GET STATISTICS FROM MULTIPLE RUNS
if CREATE_LOG and EXP_PARAMS['n_runs'] > 1:
    # Does not make sense for current logger when multiple episodes are evaluated
    # if EXP_PARAMS['eval_while_training']:
    #     sb3_create_stats_file(path=num_path, exp_params=EXP_PARAMS, style='eval')
    sb3_create_stats_file(path=SAVE_PATH, exp_params=EXP_PARAMS, style='train')
