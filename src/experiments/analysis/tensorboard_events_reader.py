from tbparse import SummaryReader
import pandas as pd
import numpy as np

def get_dfs_from_events(tensorboard_file: str):
    reader = SummaryReader(tensorboard_file)
    df = reader.scalars

    total_steps = df['step'].max() 
    total_epochs = df.loc[len(df) - 1, 'step']
    steps_per_epoch = total_steps // total_epochs
    steps_per_record = df[df['tag']=='TRAIN/ce']['step'].diff().iloc[1]
    records = total_steps // steps_per_record + 1
    records_per_epoch = np.ceil(steps_per_epoch / steps_per_record)

    df_train = pd.DataFrame(columns=['epoch', 'epoch_step', 'ce', 'loss', 'mse', 'lr_main', 'lr_slate_encoder'],
                                 index=np.arange(records), dtype=float)
    df_train['ce'] = np.array(df['value'][df['tag'] == 'TRAIN/ce'])
    df_train['loss'] = np.array(df['value'][df['tag'] == 'TRAIN/loss'])
    df_train['mse'] = np.array(df['value'][df['tag'] == 'TRAIN/mse'])
    df_train['lr_main'] = np.array(df['value'][df['tag'] == 'TRAIN/lr_main'])
    df_train['lr_slate_encoder'] = np.array(df['value'][df['tag'] == 'TRAIN/lr_slate_encoder'])
    for k in range(len(df_train)):
        df_train.loc[k, 'epoch'] = int(k // records_per_epoch)
        df_train.loc[k, 'epoch_step'] = int(k * steps_per_record)

    df_val = pd.DataFrame(columns=['epoch', 'ce', 'best_loss', 'loss', 'mse', 'gen_mse', 'exact_match', 'pixel_acc'],
                                 index=np.arange(total_epochs), dtype=float)
    df_val['ce'] = np.array(df['value'][df['tag'] == 'VAL/ce'])
    df_val['loss'] = np.array(df['value'][df['tag'] == 'VAL/loss'])
    df_val['best_loss'] = np.array(df['value'][df['tag'] == 'VAL/best_loss'])
    df_val['mse'] = np.array(df['value'][df['tag'] == 'VAL/mse'])
    df_val['gen_mse'] = np.array(df['value'][df['tag'] == 'VAL/gen_mse'])
    df_val['exact_match'] = np.array(df['value'][df['tag'] == 'VAL/exact_match'])
    df_val['pixel_acc'] = np.array(df['value'][df['tag'] == 'VAL/pixel_acc'])
    for k in range(len(df_val)):
        df_val.loc[k, 'epoch'] = int(k)

    df_test = pd.DataFrame(columns=['epoch', 'd0_exact_match', 'd1_exact_match', 'd2_exact_match', 'd0_mse', 'd1_mse', 'd2_mse', 'd0_pixel_acc', 'd1_pixel_acc', 'd2_pixel_acc'],
                          index=np.arange(total_epochs), dtype=float)
    df_test['d0_exact_match'] = np.array(df['value'][df['tag'] == 'TEST_d0/exact_match'])
    df_test['d0_mse'] = np.array(df['value'][df['tag'] == 'TEST_d0/mse'])
    df_test['d0_pixel_acc'] = np.array(df['value'][df['tag'] == 'TEST_d1/pixel_acc'])
    df_test['d1_exact_match'] = np.array(df['value'][df['tag'] == 'TEST_d1/exact_match'])
    df_test['d1_mse'] = np.array(df['value'][df['tag'] == 'TEST_d1/mse'])
    df_test['d2_exact_match'] = np.array(df['value'][df['tag'] == 'TEST_d2/exact_match'])
    df_test['d2_mse'] = np.array(df['value'][df['tag'] == 'TEST_d2/mse'])
    df_test['d2_pixel_acc'] = np.array(df['value'][df['tag'] == 'TEST_d2/pixel_acc'])
    for k in range(len(df_test)):
        df_test.loc[k, 'epoch'] = int(k)

    return df_train, df_val, df_test