import os

from config import src_dir


# -------------------------------- ONTARIO HOURLY RESOLUTION --------------------------------

# GT PARAMETERS
on_gt_default = [dict(
        gt_class='A35',
        num_gts=1,
        gt_specs=dict(
            price_natural_gas=0,  # CAD per GJ - saved as time-series data
            carbon_tax={'rate': 0, 'style': 'fuel'},  # CAD per cubic meter (m^3) - saved as time-series data
            fixed_hourly_insp_cost=0,
            cost_overhaul=33_000_000,  # CAD, over 25 years lifetime value from literature
            total_cycles=26_000,  # value from literature
            total_hours=200_000,  # value from literature
            mech_idle_fuel_flow=1200,  # pph, obtained for 2018
            start_long_h=None,  # time long start in hours (e.g. 35/60 for 35min)
            start_reg_h=20/60,  # time regular start in hours (e.g. 35/60 for 35min)
            init_strategy='zero',  # 'zero' or 'random'
            operating_threshold=0.005,  # if action smaller, GT is kept off
            ),
    )]

# BES PARAMETERS
on_bes_default = dict(
    total_cap=75,  # MWh
    max_soc=0.9,  # fraction of total capacity
    min_soc=0.1,  # fraction of total capacity
    max_charge_rate=10,  # MW
    max_discharge_rate=10,  # MW
    charge_eff=0.92,  # fraction
    discharge_eff=0.92,  # fraction
    aux_equip_eff=1.0,  # fraction, applied to charge & discharge
    self_discharge=0.0,  # fraction, applied to every step (0 = no self-discharge)
    init_strategy='min',  # 'min', 'max', 'half', or 'random'
    degradation={
        'type': 'DOD',
        'battery_capex': 300_000,  # CAD/MWh
        'k_p': 1,  # Peukert lifetime constant, degradation parameter
        'N_fail_100': 6_000,  # number of cycles at DOD=1 until battery is useless
        'add_cal_age': False,  # adds fixed cost for calendar ageing if True via MAX-operator
        'battery_life': 20,  # expected battery life in years
    },
)

# dict of data columns to serve as state var, value is (lower bound, upper bound) -> used for minmax scaling
on_state_vars = dict(
    re_power=(0.0, 30.0),
    sin_h=(-1.0, 1.0),
    cos_h=(-1.0, 1.0),
    sin_w=(-1.0, 1.0),
    cos_w=(-1.0, 1.0),
    sin_m=(-1.0, 1.0),
    cos_m=(-1.0, 1.0),
    workday=(0.0, 1.0),
    gas_price=(0.0, 10.0)
)

# GRID PARAMETERS
on_grid = dict(
    demand_profile='industry',
    sell_surplus=False,
    buy_deficit=False,
    spread=0,  # CAD per MWh added to the pool price of bought power
    penalty=500,  # CAD per MWh for missing electricity
)

on_3y_train = {
    'env_name': 'on_3y_train',  # used for saving path
    'data_file': os.path.join(src_dir, 'data', '1h', 'on_all_train.csv'),
    'demand_file': os.path.join(src_dir, 'data', '1h', 'on_all_train.csv'),
    'state_vars': on_state_vars,
    'grid': on_grid,
    'gt': on_gt_default,
    'storage': on_bes_default,
    'num_wt': 13,  # number of wind turbines, 1 WT = 2.3 MW
    'pv_cap_mw': 0,  # PV capacity in MW
    'resolution_h': 1.0,  # resolution in hours
    'modeling_period_h': 26_304,  # modeling period duration in hours
}

on_3y_test = {
    'env_name': 'on_3y_test',  # used for saving path
    'data_file': os.path.join(src_dir, 'data', '1h', 'on_all_test.csv'),
    'demand_file': os.path.join(src_dir, 'data', '1h', 'on_all_test.csv'),
    'state_vars': on_state_vars,
    'grid': on_grid,
    'gt': on_gt_default,
    'storage': on_bes_default,
    'num_wt': 13,  # number of wind turbines, 1 WT = 2.3 MW
    'pv_cap_mw': 0,  # PV capacity in MW
    'resolution_h': 1.0,  # resolution in hours
    'modeling_period_h': 26_304,  # modeling period duration in hours
}

on_2016 = {
    'env_name': 'on_2016',  # used for saving path
    'data_file': os.path.join(src_dir, 'data', '1h', 'on_2016.csv'),
    'demand_file': os.path.join(src_dir, 'data', '1h', 'on_2016.csv'),
    'state_vars': on_state_vars,
    'grid': on_grid,
    'gt': on_gt_default,
    'storage': on_bes_default,
    'num_wt': 13,  # number of wind turbines, 1 WT = 2.3 MW
    'pv_cap_mw': 0,  # PV capacity in MW
    'resolution_h': 1.0,  # resolution in hours
    'modeling_period_h': 8784,  # modeling period duration in hours
}


# -------------------------------- ONTARIO FINE RESOLUTIONS --------------------------------

on_2016_30min = {
    'env_name': 'on_2016_30min',  # used for saving path
    'data_file': os.path.join(src_dir, 'data', 'fine_res', 'on_2016_30min.csv'),
    'demand_file': os.path.join(src_dir, 'data', 'fine_res', 'on_2016_30min.csv'),
    'state_vars': on_state_vars,
    'grid': on_grid,
    'gt': on_gt_default,
    'storage': on_bes_default,
    'num_wt': 13,  # number of wind turbines, 1 WT = 2.3 MW
    'pv_cap_mw': 0,  # PV capacity in MW
    'resolution_h': 0.5,  # resolution in hours
    'modeling_period_h': 8784,  # modeling period duration in hours
}

on_2016_15min = {
    'env_name': 'on_2016_15min',  # used for saving path
    'data_file': os.path.join(src_dir, 'data', 'fine_res', 'on_2016_15min.csv'),
    'demand_file': os.path.join(src_dir, 'data', 'fine_res', 'on_2016_15min.csv'),
    'state_vars': on_state_vars,
    'grid': on_grid,
    'gt': on_gt_default,
    'storage': on_bes_default,
    'num_wt': 13,  # number of wind turbines, 1 WT = 2.3 MW
    'pv_cap_mw': 0,  # PV capacity in MW
    'resolution_h': 0.25,  # resolution in hours
    'modeling_period_h': 8784,  # modeling period duration in hours
}

on_2016_5min = {
    'env_name': 'on_2016_5min',  # used for saving path
    'data_file': os.path.join(src_dir, 'data', 'fine_res', 'on_2016_5min.csv'),
    'demand_file': os.path.join(src_dir, 'data', 'fine_res', 'on_2016_5min.csv'),
    'state_vars': on_state_vars,
    'grid': on_grid,
    'gt': on_gt_default,
    'storage': on_bes_default,
    'num_wt': 13,  # number of wind turbines, 1 WT = 2.3 MW
    'pv_cap_mw': 0,  # PV capacity in MW
    'resolution_h': 5/60,  # resolution in hours
    'modeling_period_h': 8784,  # modeling period duration in hours
}

on_2016_5min_1w = {
    'env_name': 'on_2016_5min_1w',  # used for saving path
    'data_file': os.path.join(src_dir, 'data', 'fine_res', 'on_2016_5min_1w.csv'),
    'demand_file': os.path.join(src_dir, 'data', 'fine_res', 'on_2016_5min_1w.csv'),
    'state_vars': on_state_vars,
    'grid': on_grid,
    'gt': on_gt_default,
    'storage': on_bes_default,
    'num_wt': 13,  # number of wind turbines, 1 WT = 2.3 MW
    'pv_cap_mw': 0,  # PV capacity in MW
    'resolution_h': 5/60,  # resolution in hours
    'modeling_period_h': 168,  # modeling period duration in hours
}

on_2016_5min_1d = {
    'env_name': 'on_2016_5min_1d',  # used for saving path
    'data_file': os.path.join(src_dir, 'data', 'fine_res', 'on_2016_5min_1d.csv'),
    'demand_file': os.path.join(src_dir, 'data', 'fine_res', 'on_2016_5min_1d.csv'),
    'state_vars': on_state_vars,
    'grid': on_grid,
    'gt': on_gt_default,
    'storage': on_bes_default,
    'num_wt': 13,  # number of wind turbines, 1 WT = 2.3 MW
    'pv_cap_mw': 0,  # PV capacity in MW
    'resolution_h': 5/60,  # resolution in hours
    'modeling_period_h': 24,  # modeling period duration in hours
}

on_2016_1min = {
    'env_name': 'on_2016_1.25min',  # used for saving path
    'data_file': os.path.join(src_dir, 'data', 'fine_res', 'on_2016_1.25min.csv'),
    'demand_file': os.path.join(src_dir, 'data', 'fine_res', 'on_2016_1.25min.csv'),
    'state_vars': on_state_vars,
    'grid': on_grid,
    'gt': on_gt_default,
    'storage': on_bes_default,
    'num_wt': 13,  # number of wind turbines, 1 WT = 2.3 MW
    'pv_cap_mw': 0,  # PV capacity in MW
    'resolution_h': 1.25/60,  # resolution in hours
    'modeling_period_h': 8784,  # modeling period duration in hours
}

on_3y_test_1min = {
    'env_name': 'on_3y_test_1.25min',  # used for saving path
    'data_file': os.path.join(src_dir, 'data', 'fine_res', 'on_3y_test_1.25min.csv'),
    'demand_file': os.path.join(src_dir, 'data', 'fine_res', 'on_3y_test_1.25min.csv'),
    'state_vars': on_state_vars,
    'grid': on_grid,
    'gt': on_gt_default,
    'storage': on_bes_default,
    'num_wt': 13,  # number of wind turbines, 1 WT = 2.3 MW
    'pv_cap_mw': 0,  # PV capacity in MW
    'resolution_h': 1.25/60,  # resolution in hours
    'modeling_period_h': 26_304,  # modeling period duration in hours
}

on_3y_test_5min = {
    'env_name': 'on_3y_test_5min',  # used for saving path
    'data_file': os.path.join(src_dir, 'data', 'fine_res', 'on_3y_test_5min.csv'),
    'demand_file': os.path.join(src_dir, 'data', 'fine_res', 'on_3y_test_5min.csv'),
    'state_vars': on_state_vars,
    'grid': on_grid,
    'gt': on_gt_default,
    'storage': on_bes_default,
    'num_wt': 13,  # number of wind turbines, 1 WT = 2.3 MW
    'pv_cap_mw': 0,  # PV capacity in MW
    'resolution_h': 5/60,  # resolution in hours
    'modeling_period_h': 26_304,  # modeling period duration in hours
}