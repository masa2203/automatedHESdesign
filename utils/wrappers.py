import os
import copy
from typing import Dict, List
import numpy as np
import pandas as pd

import gymnasium as gym
from gymnasium import spaces

from config import src_dir

import warnings


class MinMaxScaler(gym.ObservationWrapper):
    """
    A gym observation wrapper that scales the observation to the range [0, 1].

    :param env: A gym environment.
    """

    def __int__(self, env):
        """
        Initializes the MinMaxScaler class.

        :param env: A gym environment.
        """
        super().__init__(env)

    def observation(self, obs):
        """
        Scales the observation to the range [0, 1].

        :param obs: A numpy array that represents the observation.
        :return: A numpy array that represents the scaled observation.
        """
        scaled_obs = {}
        if obs is not None:
            for key in obs:
                low = self.observation_space[key].low
                high = self.observation_space[key].high
                scaled_obs[key] = (obs[key] - low) / (high - low)
        return scaled_obs


class DiscreteActions(gym.ActionWrapper):
    """
    A gym action wrapper that converts discrete actions to continuous actions.

    :param env: A gym environment.
    :param disc_to_cont: A list that represents the discrete actions.
    """

    def __init__(self, env, disc_to_cont):
        """
        Initializes the DiscreteActions class.

        :param env: A gym environment.
        :param disc_to_cont: A list that represents the discrete actions.
        """
        super().__init__(env)
        self.disc_to_cont = disc_to_cont

        # Check on correct action dimension
        assert env.action_space.shape[0] == len(disc_to_cont[0]), \
            "Number action dimension after discretization must match environment's action dimensions."

        self.action_space = gym.spaces.Discrete(len(disc_to_cont))

    def action(self, act):
        """
        Converts the discrete action to a continuous action.

        :param act: An integer that represents the discrete action.
        :return: A numpy array that represents the continuous action.
        """
        return np.array(self.disc_to_cont[act]).astype(self.env.action_space.dtype)

    def reverse_action(self, action):
        """
        Raises a NotImplementedError.

        :param action: A numpy array that represents the action.
        :raises: NotImplementedError.
        """
        raise NotImplementedError


class RescaleActionSpace(gym.ActionWrapper):
    """
    A gym action wrapper that rescales the action space, taking into account non-zero lower bounds.
    """

    def __init__(self, env):
        """
        Initializes the RescaleActionSpace class.

        :param env: A gym environment.
        """
        super(RescaleActionSpace, self).__init__(env)
        self.orig_action_space = self.env.action_space
        # Calculate the scale and offset for each action dimension based on original bounds
        self.scale = (self.orig_action_space.high - self.orig_action_space.low) / 2.0
        self.offset = (self.orig_action_space.high + self.orig_action_space.low) / 2.0
        # Define the new action space as [-1, 1] for all dimensions
        self.action_space = spaces.Box(low=-1, high=1, shape=self.orig_action_space.shape,
                                       dtype=self.orig_action_space.dtype)

    def action(self, action):
        """
        Rescales the action from [-1,1] to the original action space.

        :param action: A numpy array that represents the action in the [-1,1] space.
        :return: A numpy array that represents the rescaled action in the original action space.
        """
        # Rescale actions to the original space
        rescaled_action = action * self.scale + self.offset
        return rescaled_action

    def reverse_action(self, action):
        """
        Reverses the rescaling of the action from the original action space to [-1,1].

        :param action: A numpy array that represents the action in the original action space.
        :return: A numpy array that represents the reversed action in the [-1,1] space.
        """
        # Reverse scaling from original space to [-1,1]
        reversed_action = (action - self.offset) / self.scale
        return reversed_action


class PreDefinedDiscreteActions(gym.ActionWrapper):
    """
    Wrapper that defines a discrete action space with adaptive actions
    (based on the then-statement of if-then-else rules).

    Basic version with 6 actions for standard GT-BES-RE-Demand env.
    """

    def __init__(self, env):
        super().__init__(env)
        assert env.unwrapped.action_space.shape[0] == 2, 'Action space not fitting to this pre-defined action wrapper!'
        assert len(self.env.get_wrapper_attr('gts')) == 1, 'Env for this wrapper should have one GT!'

        # Define a new discrete action space with 6 actions
        self.action_space = gym.spaces.Discrete(6)
        self.num_gts = len(self.env.get_wrapper_attr('gts'))

        self.bes = self.env.unwrapped.storage

        self.avg_gt_max = 35  # in MW
        self.gt_tolerance = 0.00  # Increase GT action on [0,1] scale by this amount to compensate for amb. conditions

    def action(self, action):
        """
        Takes a discrete action and maps it to a continuous action.
        """
        # Map the discrete action to a continuous action
        continuous_action = self.map_action_to_continuous(action)
        # Return the continuous action to be taken in the environment
        return continuous_action

    def map_action_to_continuous(self, action):
        """
        Maps the discrete action to the continuous action space of the environment.
        Assumes that the original continuous action space is a Box with the first
        dimensions corresponding to GTs and the last dimension to the Battery.
        """
        continuous_action = np.zeros(self.num_gts + 1)  # +1 for the battery

        demand = self.env.unwrapped.obs['demand'].item()
        re_power = self.env.unwrapped.obs['re_power'].item()

        diff = demand - re_power  # positive diff => additional energy needed

        # Keep GT and BES off/idle
        if action == 0:
            pass

        # Charge BES with surplus REs (no GT usage)
        elif action == 1:
            if diff >= 0:  # If no surplus REs
                pass  # Leave BES idle
            else:  # Surplus REs
                bes_action = max(diff / self.bes.max_charge_rate, -1.0)
                continuous_action[-1] = bes_action  # Charge BES

        # Meet deficient power supply with BES (no GT usage)
        elif action == 2:
            if diff <= 0:  # If no deficiency
                pass  # Leave BES idle
            else:
                bes_action = min(diff / self.bes.max_discharge_rate / self.bes.discharge_eff, 1.0)
                continuous_action[-1] = bes_action

        # Meet deficient power supply with GT (no BES usage)
        elif action == 3:
            if diff <= 0:  # If no deficiency
                pass  # Leave/turn GT off
            else:
                gt_action = min((diff / self.avg_gt_max) + self.gt_tolerance, 1.0)
                continuous_action[0] = gt_action

        # Meet deficient power supply with BES + GT (Prioritizing BES)
        elif action == 4:
            if diff <= 0:  # If no deficiency
                pass  # Leave/turn GT off
            else:
                # Note: This doesn't account for insufficient SOC
                # First, use as much BES power as possible/necessary
                bes_action = min(diff / self.bes.max_discharge_rate / self.bes.discharge_eff, 1.0)
                continuous_action[-1] = bes_action
                # Meet difference from GT
                bes_flow = bes_action * self.bes.max_discharge_rate * self.bes.discharge_eff
                gt_action = min(((diff - bes_flow) / self.avg_gt_max) + self.gt_tolerance, 1.0)
                continuous_action[0] = max(0, gt_action)

        # Use GT for both deficient power supply + BES charging
        elif action == 5:
            if diff <= 0:  # If no deficiency
                pass  # Leave/turn GT off
            else:
                # Note: This doesn't account for full SOC
                gt_action_needed = min((diff / self.avg_gt_max) + self.gt_tolerance, 1.0)
                gt_action = min(gt_action_needed + 0.32, 1.0)  # 0.32 ~= 10 MW
                continuous_action[0] = gt_action

                surplus_gt_power = (gt_action - gt_action_needed) * self.avg_gt_max
                bes_action = max(-surplus_gt_power / self.bes.max_charge_rate, -1.0)
                continuous_action[-1] = bes_action

        # Correct for GT startup (less power produced due to ramping)
        if continuous_action[0] != 0 and self.env.unwrapped.gts[0].GT_state == 0:
            # Note: Start time is saved in hour-fraction, e.g. 0.25 = 15min.
            start_time = self.env.unwrapped.gts[0].start_reg_h
            if ('t2m' in self.env.unwrapped.obs and
                    self.env.unwrapped.gts[0].start_long_h is not None and
                    self.env.unwrapped.obs['t2m'] < 273.15):
                start_time = self.env.unwrapped.gts[0].start_long_h

            new_gt_actions = min(continuous_action[0] * (1 / (1 - start_time)), 1)
            continuous_action[0] = new_gt_actions

        return continuous_action.astype(self.env.get_wrapper_attr('precision')['float'])

    def reset(self, **kwargs):
        """
        Resets the environment and updates 'self.bes'.
        """
        obs = self.env.reset(**kwargs)  # Reset the underlying environment
        self.bes = self.env.unwrapped.storage  # Update BES  after reset
        return obs


class AddForecastsFromFile(gym.ObservationWrapper):
    """
    A gymnasium observation wrapper that adds forecasts from a pre-trained model to the observation space.

    This wrapper dynamically loads a CSV file containing forecasts for specified variables (e.g., wind power, demand),
    checks the compatibility of the forecast file with the environment's dataset, and incorporates the forecasts into
    the observation space.

    :param env: The gymnasium environment whose observation space is to be enhanced with forecasts.
    :param forecasted_vars: A list of variable names from the forecast file to be added to the observation space.
    :param forecast_filepath: A string of the path to the forecast CSV file or `None` to use pre-saved paths.
    """

    # Pre-saved filepaths for common environments (static attribute)
    ENV_FORECAST_PATHS = {
        'on_2016': os.path.join(src_dir, 'data', '1h', 'forecast', 'on_2016_pred.csv'),
        'on_3y_test': os.path.join(src_dir, 'data', '1h', 'forecast', 'on_3ytest_pred.csv'),
        'on_2016_1.25min': os.path.join(src_dir, 'data', 'fine_res', 'forecast', 'on_2016_res_pred.csv'),
        'on_3y_test_1.25min': os.path.join(src_dir, 'data', 'fine_res', 'forecast', 'on_3ytest_res_pred.csv'),
    }

    def __init__(
            self,
            env: gym.Env,
            forecasted_vars: list[str],
            forecast_filepath: str = None,
    ):
        """
        Initializes the AddForecastsToObservation wrapper.

        :param env: A gym environment.
        :param forecasted_vars: A list of variables to add from the forecast file
                        (e.g., ['pred_wind_power', 'pred_demand']).
        :param forecast_filepath: A string of the path to the forecast CSV file or `None` to use pre-saved paths.
        """
        super().__init__(env)
        self.forecasted_vars = forecasted_vars

        # Resolve forecast_filepath
        if not forecast_filepath:
            env_name = self.env.unwrapped.env_name
            if env_name in self.ENV_FORECAST_PATHS:
                forecast_filepath = self.ENV_FORECAST_PATHS[env_name]
            else:
                raise ValueError(
                    f"No forecast filepath provided and no pre-saved path found for environment '{env_name}'."
                )

        self.forecast_filepath = forecast_filepath

        self.default_num_wt = self.env.unwrapped.num_wt
        self.pred_wind_power_default = None

        # Specify ranges for forecasted variables
        ranges = {
            'pred_wind_power': (self.observation_space['re_power'].low,
                                self.observation_space['re_power'].high),
            'pred_demand': (self.observation_space['demand'].low,
                            self.observation_space['demand'].high),
        }

        # Load the forecast file
        usecols = ['Date'] + self.forecasted_vars
        self.forecast_data = pd.read_csv(self.forecast_filepath, index_col=0, usecols=usecols)

        # Check compatibility of forecast file with the environment's dataset
        if len(self.forecast_data) != len(self.env.unwrapped.data['Date']):
            raise ValueError(
                f"Forecast data length ({len(self.forecast_data)}) does not match environment data length "
                f"({len(self.env.unwrapped.data)})."
            )

        # Add forecasts to the environment's `data` dictionary and extend observation space
        for var in self.forecasted_vars:
            if var not in self.forecast_data.columns:
                raise ValueError(f"Forecast variable '{var}' not found in the forecast file.")
            # Add the forecast column to `data`
            if var == 'pred_wind_power':  # multiply by number of turbines and convert to MW for consistency
                self.env.unwrapped.data[var] = self.forecast_data[var].to_numpy() * self.default_num_wt / 1000
                self.pred_wind_power_default = self.env.unwrapped.data[var]
            else:
                self.env.unwrapped.data[var] = self.forecast_data[var].to_numpy()
            # Extend the observation space dynamically
            low, high = ranges[var]
            self.observation_space[var] = spaces.Box(low=low, high=high, shape=(1,))

    def observation(self, obs: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
        """
        Updates the observation dictionary by adding forecasted values for specified variables.

        :param obs: Original observation dictionary from the environment.
        :return: Updated observation dictionary including forecasted values.
        """
        current_index = self.env.get_wrapper_attr('count')
        for var in self.forecasted_vars:
            try:
                forecast_value = self.env.unwrapped.data[var][current_index]
            except IndexError:  # Take same value as current time-step if end of data file reached
                forecast_value = self.env.unwrapped.data[var][current_index-1]
            obs[var] = np.array([forecast_value], dtype=np.float32)
        return obs


class ForceGTStartupActionWrapper(gym.ActionWrapper):
    """
    An ActionWrapper that forces GT startup to complete by modifying the agent's actions.
    Relevant for fine temporal resolutions.

    If a GT is in the startup process (start_up_remaining_h > 0), the wrapper overrides the agent's action
    to ensure that the GT continues the startup process until it is fully operational.
    """

    def __init__(self, env):
        super().__init__(env)
        assert hasattr(env.unwrapped, "gts"), "Environment must have GT objects."
        self.gts = env.unwrapped.gts  # Access GTs in the environment

    def action(self, action):
        """
        Modifies the agent's action to enforce GT startup completion.

        :param action: The original action chosen by the agent.
        :return: The modified action to ensure GT startup continues.
        """
        for i, gt in enumerate(self.gts):
            if gt.start_up_remaining_h > 0 and action[i] < gt.operating_threshold:
                # Note: Rounding issue -> at least one timestep with power output of GT needed
                action[i] = gt.operating_threshold + 1e-5  # Force GT to continue start-up
        return action


class GTDelayShutDownWrapper(gym.ActionWrapper):
    """
    Wrapper that delays the GT shut down until the agent sets GT power to zero for a
    predefined number of consecutive timesteps.

    If the agent sets GT power to zero, the wrapper will enforce the minimum operating
    threshold until the zero action is repeated for N consecutive timesteps.
    """

    def __init__(self, env, consecutive_shutdown_steps):
        """
        Initialize the wrapper.

        :param env: The environment to wrap.
        :param consecutive_shutdown_steps: Number of consecutive steps where the agent must
                                           set GT power to zero to allow shutdown.
        """
        super().__init__(env)
        assert hasattr(env.unwrapped, "gts"), "Environment must have GT objects."
        self.gts = env.unwrapped.gts  # Access GTs in the environment
        self.consecutive_shutdown_steps = consecutive_shutdown_steps  # Steps needed for shutdown
        self.consecutive_zero_steps = [0] * len(self.gts)  # Track consecutive zero actions for each GT

    def action(self, action):
        """
        Modify the agent's action to enforce minimum operating threshold logic.

        :param action: The original action chosen by the agent.
        :return: The modified action to enforce minimum operating threshold.
        """
        for i, gt in enumerate(self.gts):
            # Check if the agent is trying to set GT power to zero while the GT is operating
            if gt.GT_state != 0 and action[i] == 0:
                # Increment the zero action counter for this GT
                self.consecutive_zero_steps[i] += 1

                # Check if the zero action has been sustained for enough timesteps
                if self.consecutive_zero_steps[i] < self.consecutive_shutdown_steps:
                    # Enforce minimum operating threshold
                    action[i] = gt.operating_threshold + 1e-5

            else:
                # Reset zero action counter if the agent sets a non-zero action
                self.consecutive_zero_steps[i] = 0

        return action


class BESSOCPenalty(gym.RewardWrapper):
    """
    A gym reward wrapper that applies a penalty to the reward based on the SOC of the BES.

    This wrapper reduces the reward given to the agent by a penalty that increases as the SOC of the BES
    decreases. This encourages the agent to maintain a higher SOC in the BES.

    :param env: A gymnasium environment.
    :param penalty: A float representing the penalty applied for each unit below the max SOC.
    :param max_soc: A float representing the maximum SOC value, default is 1.
    """

    def __init__(self, env, penalty, max_soc=1):
        """
        Initializes the BESSOCPenalty class.

        :param env: A gym environment.
        :param penalty: A float representing the penalty applied for each unit below the max SOC.
        :param max_soc: A float representing the maximum SOC value, default is 1.
        """
        super().__init__(env)
        self.penalty = penalty
        self.max_soc = max_soc

    def reward(self, reward):
        """
        Adjusts the reward based on the current SOC of the BES.

        Applies a penalty to the reward if the SOC of the BES is below the max_soc. The penalty is proportional
        to the difference between the current SOC and max_soc.

        :param reward: A float representing the original reward for the current step.
        :return: A float representing the adjusted reward after applying the penalty.
        """
        p = max((self.max_soc - self.env.get_wrapper_attr('obs')['soc'].item()), 0) / self.max_soc * self.penalty
        reward -= p
        return reward


class RandomEpisodesWrapper(gym.Wrapper):
    """
    A gym observation wrapper that samples random subsets from the underlying time-series data based on the specified
     mode ('day', 'week', or 'month') and iterates over these subsets at every new episode.

    :param env: A gymnasium environment.
    :param mode: A string specifying the episode length to pick from ['day', 'week', 'month'].
    :param num: An integer representing the number of episodes to sample.
    """

    def __init__(
            self,
            env,
            mode: str = 'day',  # Episode length to pick
            num: int = 1,  # Number of episodes to sample.
    ):
        """
        Initializes the RandomEpisodesWrapper class.

        :param env: A gymnasium environment.
        :param mode: A string specifying the subset selection mode ('day', 'week', or 'month').
        :param num: An integer representing the number of subsets (episodes) to be randomly sampled.
        """
        super().__init__(env)

        def get_random_subset(mode: str = 'day'):
            """Returns a sampled subset of the entire dataset."""
            while True:  # Keep trying until a suitable subset is found
                if mode == 'day':
                    date_only_np = np.array([pd.to_datetime(date).date() for date in dates_np])
                    unique_days = np.unique(date_only_np)
                    random_day = np.random.choice(unique_days)
                    random_day_indices = np.where(date_only_np == random_day)[0]
                    if len(random_day_indices) >= 24:  # Check if at least 24 hours of data
                        random_day_data = {var: self.org_data[var][random_day_indices] for var in self.org_data}
                        print('Day picked: ', random_day)
                        return random_day_data
                elif mode == 'week':
                    week_only_np = np.array([pd.to_datetime(date).strftime('%Y-%U') for date in dates_np])
                    unique_weeks = np.unique(week_only_np)
                    random_week = np.random.choice(unique_weeks)
                    random_week_indices = np.where(week_only_np == random_week)[0]
                    if len(random_week_indices) >= 168:  # Check if at least 168 hours of data
                        random_week_data = {var: self.org_data[var][random_week_indices] for var in self.org_data}
                        print('Week picked: ', random_week)
                        return random_week_data
                elif mode == 'month':
                    month_year_np = np.array([pd.to_datetime(date).strftime('%Y-%m') for date in dates_np])
                    unique_months = np.unique(month_year_np)
                    random_month = np.random.choice(unique_months)
                    random_month_indices = np.where(month_year_np == random_month)[0]
                    if len(random_month_indices) >= 28 * 24:  # At least 28 days
                        random_month_data = {var: self.org_data[var][random_month_indices] for var in self.org_data}
                        print('Month picked: ', random_month)
                        return random_month_data
                elif mode == 'year':
                    unique_years = np.unique(dates_np.astype('datetime64[Y]'))
                    random_year = np.random.choice(unique_years)
                    random_year_indices = np.where(dates_np.astype('datetime64[Y]') == random_year)[0]
                    if len(random_year_indices) >= 365 * 24:  # At least 365 days
                        random_year_data = {var: self.org_data[var][random_year_indices] for var in self.org_data}
                        print('Year picked: ', random_year)
                        return random_year_data
                else:
                    raise NotImplementedError('Mode not supported!')

        # Save original dataset
        self.org_data = self.env.unwrapped.data

        # Remove timezone and convert to datetime
        mod_data = [dt.split('+')[0] for dt in self.org_data['Date']]
        dates_np = np.array([np.datetime64(date) for date in mod_data])

        # Placeholder for dataset subsets
        self.datasets = []

        # Create desired number of subsets
        for i in range(num):
            subset = get_random_subset(mode=mode)
            self.datasets.append(subset)

        # Iterator
        self.i = 0

    def reset(self, **kwargs):
        """
        Resets the environment with a new randomly selected subset of data based on the specified mode
        and updates the dataset for the next episode.

        This method ensures that each episode the agent experiences is based on a different subset,
        cycling through the pre-generated list of subsets.

        :param kwargs: Additional keyword arguments passed to the environment's reset method.
        :return: The initial observation and info from the environment's reset method.
        """
        # Reset env variables with new episode length and dataset
        self.env.unwrapped.len_episode = self.datasets[self.i]['Date'].shape[0]
        self.env.unwrapped.data = self.datasets[self.i]

        # Update iterator
        self.i += 1
        self.i %= len(self.datasets)

        obs, info = self.env.reset(**kwargs)
        return obs, info


class UpdateEnvConfig(gym.ObservationWrapper):
    """
    A gymnasium observation wrapper that modifies the environment's configuration and observation space dynamically
    based on predefined configurations. This wrapper allows for customization of environment parameters such as
    renewable energy power, gas prices, penalties, and battery storage capacities, while also expanding the
    observation space.

    :param env: A gymnasium environment.
    :param configs: A list of dictionaries, where each dictionary contains configuration values for the environment.
                    These configurations are applied sequentially at the start of each new episode.
    """

    def __init__(
            self,
            env,
            configs: list[dict]
    ):
        """
        Initializes the UpdateEnvConfig class.

        :param env: A gymnasium environment.
        :param configs: A list of dictionaries containing environment configuration values. Each dictionary should
                        specify parameters such as 'num_wt', 'rate_gas_price', 'penalty', 'bes_cap', and 'bes_rate'.
        """
        super().__init__(env)

        if self.env.unwrapped.pv_cap_mw != 0:
            warnings.warn('Wrapper must be adapted to work with PV power.')

        # Save original dataset
        self.org_data = copy.deepcopy(self.env.unwrapped.data)

        # Step 1: Pre-saved information
        self.defaults = {'num_wt': self.env.unwrapped.num_wt}
        # self.configs = [{'num_wt': 10, 'rate_gas_price': 0.5, 'penalty': 250, 'bes_cap': 45, 'bes_rate': 5}]
        self.configs = configs
        self.new_state_vars = dict(
            penalty=(0, 5000),
            bes_cap=(0, 200),
            bes_rate=(0, 50),
        )

        # Step 2: Expand state space
        for var in self.new_state_vars:
            self.observation_space[var] = spaces.Box(low=self.new_state_vars[var][0],
                                                     high=self.new_state_vars[var][1],
                                                     shape=(1,))

        # Iterator
        self.i = -1  # Start with config 0 when calling reset() for the first time

        # print('\nPlant configurations: ', self.configs)

    def observation(self, obs: Dict[str, np.ndarray]) -> Dict[str, np.ndarray]:
        """
        Updates the observation space with the current configuration values.

        :param obs: A dictionary containing the original observation from the environment.
        :return: A dictionary containing the updated observation, with additional variables representing the current
                 configuration values.
        """
        for var in self.new_state_vars:
            obs[var] = np.array([self.configs[self.i][var]])
        return obs

    def reset(self, **kwargs):
        """
        Resets the environment with updated configuration values based on the current episode index and modifies the
        environment's underlying dataset. It also ensures that the observation space is updated accordingly.

        :param kwargs: Additional keyword arguments passed to the environment's reset method.
        :return: The modified initial observation and info from the environment's reset method.
        """
        # Update iterator
        self.i += 1  # Note, must be done before env update to avoid mismatch with observation()-method.
        self.i %= len(self.configs)

        # Update the env's dataset
        # Note: Computationally cheap solution, avoid re-reading of csv files
        self.env.unwrapped.data['re_power'] = self.org_data['re_power'] * (
                self.configs[self.i]['num_wt'] / self.defaults['num_wt'])
        self.env.unwrapped.data['gas_price'] = self.org_data['gas_price'] * self.configs[self.i]['rate_gas_price']

        # Update the env's components
        self.env.unwrapped.grid.penalty = self.configs[self.i]['penalty']

        self.env.unwrapped.storage.total_cap = self.configs[self.i]['bes_cap']
        self.env.unwrapped.storage.investment_cost = (self.configs[self.i]['bes_cap'] *
                                                      self.env.unwrapped.storage_dict['degradation']['battery_capex'])

        self.env.unwrapped.storage.max_charge_rate = self.configs[self.i]['bes_rate']
        self.env.unwrapped.storage.max_discharge_rate = self.configs[self.i]['bes_rate']

        # Call reset and update obs space
        obs, info = self.env.reset(**kwargs)
        obs = self.observation(obs)

        # print('\tNext config: ', self.configs[self.i])

        return obs, info
