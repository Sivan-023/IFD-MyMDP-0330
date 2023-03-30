# encoding=utf-8

import numpy as np
from stable_baselines3 import DQN
from data import create_dataset
from environment import ClassifyEnv

# Load the CWRU dataset
train_data = create_dataset('data/DE.h5', train=True).X[:]
train_label = create_dataset('data/DE.h5', train=True).y[:]
test_data = create_dataset('data/DE.h5', train=False).X[:]
test_label = create_dataset('data/DE.h5', train=False).y[:]
print('#train_label_num = %d' % len(np.unique(train_label)))
print('#train_data_num = %d' % len(train_data))

print('#train_label_num = %d' % len(np.unique(test_label)))
print('#train_data_num = %d' % len(test_data))

# Create an instance of the ClassifyEnv environment
train_env = ClassifyEnv(mode='train', trainx=train_data, trainy=train_label)
test_env = ClassifyEnv(mode='test', trainx=test_data, trainy=test_label)

# Create a DQN agent
model = DQN(policy='MlpPolicy',
            env=train_env,
            learning_rate=1e-3,
            buffer_size=10000,  # 最多积累N步最新的数据,旧的删除
            learning_starts=500,  # 积累了N步的数据以后再开始训练
            batch_size=64,  # 每次采样N步
            tau=0.8,  # 软更新的比例,1就是硬更新
            gamma=0.9,
            train_freq=(1, 'step'),  # 训练的频率
            target_update_interval=1000,  # target网络更新的频率
            policy_kwargs={},  # 网络参数
            verbose=1,
            device='cpu')

# Train the DQN agent
model.learn(total_timesteps=int(1e1), progress_bar=True)

# Evaluate the agent on the test dataset
correct, total = 0, 0
for i in range(len(test_data)):
    obs = test_data[i]
    true_label = test_label[i]
    action, _states = model.predict(obs)
    if action == true_label:
        correct += 1
    total += 1

accuracy = correct / total

print("Total:", total)
print("Accuracy:", accuracy)