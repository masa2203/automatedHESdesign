import json
import time
from stable_baselines3 import PPO, SAC, DDPG, DQN, A2C, TD3
from stable_baselines3.common.vec_env import VecNormalize
from stable_baselines3.common.evaluation import evaluate_policy

from envs.env_params import *
from utils.make_env import make_env
from utils.utilities import generate_discrete_actions
from utils.logger import get_env_log_data

import warnings

# Suppress all FutureWarnings
warnings.simplefilter(action='ignore', category=FutureWarning)


"""LOAD A SAVED MODEL AND ENV -> TEST ON ANOTHER ENV"""
AGENTS = {
    'ppo': PPO,
    'sac': SAC,
    'a2c': A2C,
    'dqn': DQN,
    'ddpg': DDPG,
    'td3': TD3,
    'aadqn': DQN,
    'aappo': PPO,
}


class DispatchAgent:
    """
        The `DispatchAgent` class is responsible for loading a saved RL model and environment,
        configuring the test environment, and querying the agent with specific configurations.

        This class supports various RL algorithms (PPO, SAC, A2C, DQN, etc.) and handles dynamic environment
        updates for evaluation. It loads the agent and environment from a specified path and provides
        functionality to evaluate the agent using a test environment configuration.

        Attributes:
            env (str): Name of the environment to be used for evaluation.
            env_kwargs (dict): Environment configuration parameters.
            agent (str): RL algorithm to be used (e.g., 'ppo', 'sac').
            path (str): Path to the saved model and environment files.
            inputs (dict): Loaded experiment inputs from the saved `inputs.json`.
            saved_model_path (str): Path to the saved RL model file.
            saved_env_path (str): Path to the saved VecNormalize environment file.
            disc_actions (list or None): Predefined discrete actions, if applicable.
        """
    def __init__(
            self,
            env,
            env_kwargs: dict,
            agent: str,
            path: str,
    ):
        """
        Initialize the DispatchAgent instance.

        This method initializes the agent with the environment and configuration parameters, loads
        the saved experiment inputs and model, and optionally generates discrete actions.

        Parameters:
            env (str): Name of the environment to be used for evaluation.
            env_kwargs (dict): Environment configuration parameters.
            agent (str): RL algorithm to be used (e.g., 'ppo', 'sac').
            path (str): Path to the directory containing the saved model and experiment inputs.

        Raises:
            FileNotFoundError: If the specified path does not exist or is not a directory.

        Example:
            >>> agent = DispatchAgent(
                    env='CustomEnv',
                    env_kwargs={'param1': 'value1'},
                    agent='ppo',
                    path='./saved_model'
                )
        """

        self.env = env
        self.env_kwargs = env_kwargs
        self.agent = AGENTS[agent]

        # Load inputs from saved experiment/agent
        with open(os.path.join(path, 'inputs.json')) as f:
            self.inputs = json.load(f)

        if os.path.isdir(path):
            self.saved_model_path = os.path.join(path, 'run_0', 'model.zip')
            self.saved_env_path = os.path.join(path, 'run_0', 'env.pkl')
        else:
            FileNotFoundError('No file found!')

        # ACTIONS
        self.disc_actions = None
        discretization_params = self.inputs.get('DISCRETE_ACTIONS', None)
        if discretization_params != 'predefined' and discretization_params is not None:
            self.disc_actions = generate_discrete_actions(**discretization_params)

    def query(
            self,
            config: dict,
    ):
        """
        Evaluate the saved RL model on a test environment configuration.

        This method configures the test environment based on the provided configuration, loads the
        saved RL model and environment, and evaluates the model on one or more episodes. The method
        returns the mean reward and logs evaluation data.

        Parameters:
            config (dict): Test environment configuration. This is used to update specific plant
                parameters for evaluation.

        Returns:
            dict: A dictionary containing logged data, including:
                - mean_reward (float): Mean reward achieved during evaluation.
                - evaluation_time (float): Time taken for evaluation.
                - environment_metrics (dict): Metrics tracked by the environment during evaluation.

        Example:
            >>> config = {'param1': value1, 'param2': value2}
            >>> log_data = agent.query(config)
            >>> print(log_data['mean_reward'])

        Raises:
            FileNotFoundError: If the saved model or environment files are not found.
        """
        start_time = time.time()

        env = make_env(
            env=self.env,
            env_kwargs=self.env_kwargs,
            update_plant_config=[config],
            flatten_obs=self.inputs['EXP_PARAMS']['flatten_obs'],
            discrete_actions=self.disc_actions,
            frame_stack=self.inputs['EXP_PARAMS']['frame_stack'],
            forecasts_from_file=self.inputs['EXP_PARAMS'].get('forecasts_from_file', None),
            minmax_scaling=self.inputs['EXP_PARAMS'].get('minmax_scaling', False),
            use_predefined_action_wrapper=self.inputs['EXP_PARAMS'].get(
                'use_predefined_action_wrapper',  # New argument
                self.inputs['EXP_PARAMS'].get('use_predefined_discrete_actions', None)  # Fallback to old argument
            ),
            complete_gt_start=self.inputs['EXP_PARAMS'].get('complete_gt_start', False),
            delay_gt_shutdown=self.inputs['EXP_PARAMS'].get('delay_gt_shutdown', None),
            use_vec_env=False,  # Don't call VecNormalize (called with .load() below)
        )

        env = VecNormalize.load(self.saved_env_path, env)
        env.training = False
        env.norm_reward = False
        env.unwrapped.envs[0].unwrapped.start_tracking()  # Start tracking env variables for evaluation
        model = self.agent.load(self.saved_model_path, env=env)

        mean_reward, std_reward = evaluate_policy(model, model.get_env(), n_eval_episodes=1)
        log_data = get_env_log_data(env=env, mean_reward=mean_reward, start_time=start_time)

        # print(f"mean_reward: {mean_reward:,.2f} +/- {std_reward:,.2f}")

        return log_data
