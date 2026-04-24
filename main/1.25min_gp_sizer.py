import json
import pickle
import socket
from typing import Dict, Any

import optuna
import os
from config import src_dir
from envs.environments import *
from envs.env_params import *
from sizing.score_config import ScoreConfig
from sizing.query_dispatch import DispatchAgent
from utils.logger import print_and_save_tune_logs

# Fixed parameters
ENV = GasTurbineBatteryRenewablesDemandEnv
ENV_KWARGS = on_3y_test_1min
agent = 'aadqn'

env_years = {
    'on_2016_1.25min': 1,
    'on_3y_test_1.25min': 3,
}

# LOG
CREATE_LOG = True
CONTINUE_RUN = False
SAVE_PATH = os.path.join(src_dir, 'log', ENV_KWARGS['env_name'], 'sizing', 'gp', input('Save in folder: ')) \
    if CREATE_LOG else None
RUN_ID = input('Define run ID for existing run: ') if CONTINUE_RUN and SAVE_PATH else '0'

# EXP PARAMS
EXP_PARAMS = {
    'path': os.path.join(src_dir, 'models', 'saved_dispatch_models', agent, '1.25min_200ep_m400env'),
    'penalty': 500,
    'rate_gas_price': 1.0,
    'bes_capex_factor': 1.0,

    'n_trials': 40,
    'n_startup_trials': 25,
    'n_jobs': 1,
    'timeout': int(3600) * 24,  # in seconds
}

if SAVE_PATH is not None:
    os.makedirs(SAVE_PATH, exist_ok=CONTINUE_RUN)
    with open(os.path.join(SAVE_PATH, f'inputs_{RUN_ID}.json'), 'w') as f:
        json.dump({
            'EXP_PARAMS': EXP_PARAMS,
            'PLANT_PARAMS': ENV_KWARGS,
            'PC_NAME': socket.gethostname(),
        }, f)

# Initialize the dispatch agent and sizer
dispatcher = DispatchAgent(
    env=ENV,
    env_kwargs=ENV_KWARGS,
    agent=agent,
    path=EXP_PARAMS['path'],
)
sizer = ScoreConfig(
    num_years=env_years[ENV_KWARGS['env_name']],
    bes_capex_per_mwh=270_000 * EXP_PARAMS.get('bes_capex_factor', 1),
    bes_capex_rate_per_mw=120_000 * EXP_PARAMS.get('bes_capex_factor', 1),
)


def sample_params(trial: optuna.Trial) -> Dict[str, Any]:
    """
    Sampler for hyperparameters.

    :param trial: Optuna trial object
    :return: The sampled hyperparameters for the given trial.
    """
    num_wt = trial.suggest_int('num_wt', 0, 20)  # Number of wind turbines
    bes_cap = trial.suggest_int('bes_cap', 1, 150)  # BES capacity (MWh)
    bes_rate = trial.suggest_int('bes_rate', 1, 40)  # BES charge/discharge rate (MW)

    return {
        'num_wt': num_wt,
        'bes_cap': bes_cap,
        'bes_rate': bes_rate,
    }


def objective(trial):
    """
    Optuna objective function to minimize TOTEX or maximize reward.

    Parameters:
        trial (optuna.Trial): Optuna trial object to suggest parameter values.

    Returns:
        float: Value to optimize (TOTEX or negative reward).
    """
    sampled_params = sample_params(trial)

    # Define the configuration
    test_config = dict(
        num_wt=sampled_params['num_wt'],
        bes_cap=sampled_params['bes_cap'],
        bes_rate=sampled_params['bes_rate'],
        penalty=EXP_PARAMS['penalty'],
        rate_gas_price=EXP_PARAMS['rate_gas_price'],
    )

    # Query the dispatch agent
    log = dispatcher.query(test_config)

    # Compute TOTEX
    totex = sizer.get_totex(
        numWT=test_config['num_wt'],
        capBES=test_config['bes_cap'],
        rateBES=test_config['bes_rate'],
        fuel_cost=log['fuel_cost_sum'][0],
        variable_om=log['maint_cost_sum'][0],
        degradation_cost=log['degr_cost_sum'],
    )

    # Log and return the value to optimize
    # trial.set_user_attr("reward_sum", log['reward_sum'])
    # trial.set_user_attr("fuel_cost_sum", log['fuel_cost_sum'][0])
    # trial.set_user_attr("maint_cost_sum", log['maint_cost_sum'][0])
    # trial.set_user_attr("degr_cost_sum", log['degr_cost_sum'])
    trial.set_user_attr("totex (no pen)", totex)

    total_penalty = ((EXP_PARAMS['penalty'] * abs(log['avg_undersupply']) * log['num_undersupply'] * ENV_KWARGS['resolution_h']) /
                     env_years[ENV_KWARGS['env_name']])
    totex += total_penalty

    return totex


# Select the sampler, can be random, TPESampler, CMAES, ...
if CONTINUE_RUN:
    try:
        sampler = pickle.load(open(os.path.join(SAVE_PATH, 'sampler.pkl'), 'rb'))
    except FileNotFoundError:
        sampler = optuna.samplers.GPSampler(
            n_startup_trials=EXP_PARAMS['n_startup_trials'],  # Default = 10 (random unless other ind. sampler chosen)
            deterministic_objective=True,
        )
else:
    sampler = optuna.samplers.GPSampler(
        n_startup_trials=EXP_PARAMS['n_startup_trials'],  # Default = 10 (random unless other ind. sampler chosen)
        deterministic_objective=True,
    )

storage = f"sqlite:///{os.path.join(SAVE_PATH, 'database.db')}" if SAVE_PATH is not None else None

# Create Optuna study
study = optuna.create_study(
    direction='minimize',
    sampler=sampler,
    storage=storage,
    study_name='tbd',
    load_if_exists=True,
)

# Run optimization
try:
    study.optimize(
        objective,
        n_trials=EXP_PARAMS['n_trials'],
        n_jobs=EXP_PARAMS['n_jobs'],
        timeout=EXP_PARAMS['timeout']
    )
except KeyboardInterrupt:
    pass


print_and_save_tune_logs(study=study, save_path=SAVE_PATH, run_id=RUN_ID)
