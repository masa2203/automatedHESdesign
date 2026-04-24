import json
import os
import pickle
import socket
import time

import torch
from torch import optim
from torch.utils.data import DataLoader

from config import src_dir
from forecasting.process_data import open_dataset, split_dataset, scale_data, sliding_window
from forecasting.training import train_model, validation, LOSS_FUNCTIONS_MAP
from models import forecasters
from utils.utilities import set_seeds

# Check if GPU is available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using {device} device.")

# LOG
CREATE_LOG = False
SAVE_PATH = os.path.join(src_dir, 'log', 'forecast', 'wind', '1h', 'run', input('Save in folder: ')) \
    if CREATE_LOG else None

file_path = os.path.join(src_dir, 'data', '1h', 'on_all_train.csv')

PARAMS = {
    'data_ranges': dict(
        train_range=('2017-01', '2017-12'),
        val_range=('2015-01', '2015-12'),
        test_range=('2016-01', '2016-12'),
    ),
    'target_column': 'wind_power',
    'feature_columns': [
        'sin_h', 'cos_h',
        'sin_w', 'cos_w',
        'sin_m', 'cos_m',
        # 'workday',
        # 'nextday_workday'
        'wind speed at 100m (m/s)',
        'air temperature at 2m (C)',
        'relative humidity at 2m (%)',
        'surface air pressure (Pa)'
    ],
    # 'columns_to_scale': None,
    'columns_to_scale': [
        'wind speed at 100m (m/s)',
        'air temperature at 2m (C)',
        'relative humidity at 2m (%)',
        'surface air pressure (Pa)'
    ],
    'scaler_type': 'min-max',
    'hours_ahead': 1,
    'window_size': 12,
    'lr': 0.007,
    'lr_decay_factor': 0.4,
    'lr_patience': 10,
    'min_lr': 1e-5,
    'n_epochs': 150,
    'early_stopping_threshold': 40,
    'batch_size': 32,
    'model_params': {
        'cnn_net_shape': [16],
        'cnn_kernel_size': 3,
        'cnn_stride': 1,
        'ann_net_shape': [128],
    },
    'seed': 22,
}

if SAVE_PATH is not None:
    os.makedirs(SAVE_PATH, exist_ok=True)
    with open(os.path.join(SAVE_PATH, 'inputs.json'), 'w') as f:
        json.dump({
            'DATA': file_path,
            'PARAMS': PARAMS,
            'PC_NAME': socket.gethostname(),
        }, f)

set_seeds(22)

data = open_dataset(
    file_path=file_path,
    target=PARAMS['target_column'],
    features=PARAMS['feature_columns'],
)

train_df, val_df, test_df = split_dataset(
    df=data,
    **PARAMS['data_ranges']
)

train_df, val_df, test_df, feature_scaler, target_scaler = scale_data(
    train_df=train_df,
    val_df=val_df,
    test_df=test_df,
    columns_to_scale=PARAMS['columns_to_scale'],  # Features to scale
    target_column=PARAMS['target_column'],
    scaler_type=PARAMS['scaler_type']
)


train_x, train_y = sliding_window(
    features=train_df.values,
    labels=train_df[PARAMS['target_column']].values,
    window_size=PARAMS['window_size'],
    label_distance=PARAMS['hours_ahead'],
    multihorizon=False,
    return_tensor=True,
)

val_x, val_y = sliding_window(
    features=val_df.values,
    labels=val_df[PARAMS['target_column']].values,
    window_size=PARAMS['window_size'],
    label_distance=PARAMS['hours_ahead'],
    multihorizon=False,
    return_tensor=True,
)

test_x, test_y = sliding_window(
    features=test_df.values,
    labels=test_df[PARAMS['target_column']].values,
    window_size=PARAMS['window_size'],
    label_distance=PARAMS['hours_ahead'],
    multihorizon=False,
    return_tensor=True,
)

train = torch.utils.data.TensorDataset(train_x.to(device), train_y.to(device))
val = torch.utils.data.TensorDataset(val_x.to(device), val_y.to(device))
test = torch.utils.data.TensorDataset(test_x.to(device), test_y.to(device))

train = DataLoader(dataset=train, batch_size=PARAMS['batch_size'], shuffle=True)
val = DataLoader(dataset=val, batch_size=val_y.shape[0], shuffle=False)
test = DataLoader(dataset=test, batch_size=test_y.shape[0], shuffle=False)

model = forecasters.CNN(
    in_dim=train_df.shape[1],
    out_dim=1,
    window_size=PARAMS['window_size'],
    **PARAMS['model_params']
).to(device)

optimizer = torch.optim.Adam(
    model.parameters(),
    lr=PARAMS['lr']
)

scheduler = optim.lr_scheduler.ReduceLROnPlateau(
    optimizer,
    mode='min',
    factor=PARAMS["lr_decay_factor"],
    patience=PARAMS["lr_patience"],
    min_lr=PARAMS['min_lr']
)

start_time = time.time()

train_losses, vali_losses, vali_counter = train_model(
    model=model,
    optimizer=optimizer,
    scheduler=scheduler,
    train_loader=train,
    vali_loader=val,
    target_scaler=target_scaler,
    n_epochs=PARAMS['n_epochs'],
    early_stopping_threshold=PARAMS['early_stopping_threshold'],
    loss_func='mse',
    save_path=SAVE_PATH,
)

test_pred, test_labels = validation(
    model=model,
    vali_loader=test,
    vali_losses=None,
    target_scaler=target_scaler,
    loss_func='rmse',
    verbose=False,
    testing=True,
)

outputs = {}

for loss_name in LOSS_FUNCTIONS_MAP:
    loss_fn = LOSS_FUNCTIONS_MAP[loss_name]
    loss = loss_fn(test_labels[0].squeeze(), test_pred[0].squeeze())
    print(f'{loss_name} test loss: {loss}')
    outputs[loss_name] = loss.item()

if SAVE_PATH is not None:
    outputs['train_losses'] = train_losses
    outputs['vali_losses'] = vali_losses
    outputs['vali_counter'] = vali_counter
    outputs['testset_labels'] = test_labels[0].squeeze().cpu().tolist()
    outputs['testset_pred'] = test_pred[0].squeeze().cpu().tolist()
    with open(os.path.join(SAVE_PATH, 'outputs.json'), 'w') as f:
        json.dump(outputs, f)
    with open(os.path.join(SAVE_PATH, 'scaler.pkl'), 'wb') as f:
        pickle.dump(feature_scaler, f)
    with open(os.path.join(SAVE_PATH, 'target_scaler.pkl'), 'wb') as f:
        pickle.dump(target_scaler, f)
