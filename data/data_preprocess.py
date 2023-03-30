# encoding=utf-8
import h5py
import pandas as pd
import numpy as np
import os
import scipy.io as scio

frame = pd.read_table('annotations.txt')
dim = 2048
train_fraction = 0.8
signals_tr = []
labels_tr = []
signals_tt = []
labels_tt = []
count = 0
for idx in range(len(frame)):
    mat_name = os.path.join('raw_data', frame['file_name'][idx])
    raw_data = scio.loadmat(mat_name)
    for key, value in raw_data.items():
            if key[5:7] == 'DE':
                signal = value
                sample_num = signal.shape[0]//dim
                train_num = int(sample_num*train_fraction)
                test_num = sample_num - train_num
                signal = signal[0:dim*sample_num]
                signals = np.array(np.split(signal, sample_num))
                
                signals_tr.append(signals[0:train_num, :])
                signals_tt.append(signals[train_num:sample_num, :])
                labels_tr.append(idx*np.ones(train_num)) 
                labels_tt.append(idx*np.ones(test_num)) 
signals_tr_np = np.concatenate(signals_tr).squeeze()
labels_tr_np = np.concatenate(np.array(labels_tr)).astype('uint8')
signals_tt_np = np.concatenate(signals_tt).squeeze()
labels_tt_np = np.concatenate(np.array(labels_tt)).astype('uint8')
print(signals_tr_np.shape, labels_tr_np.shape, signals_tt_np.shape, labels_tt_np.shape)

f = h5py.File('DE.h5', 'w')
f.create_dataset('X_train', data=signals_tr_np)
f.create_dataset('y_train', data=labels_tr_np)
f.create_dataset('X_test', data=signals_tt_np)
f.create_dataset('y_test', data=labels_tt_np)
f.close()