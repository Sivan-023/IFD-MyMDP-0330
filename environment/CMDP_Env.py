# encoding=utf-8

import gym
import numpy as np
from gym import spaces
from gym.utils import seeding
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, precision_score, recall_score, f1_score


class ClassifyEnv(gym.Env):
    metadata = {'render.modes': ['human']}

    def __init__(self, mode, trainx, trainy):  # mode means training or testing
        self.mode = mode
        self.Env_data = trainx
        self.Answer = trainy
        self.id = np.arange(trainx.shape[0])

        self.game_len = self.Env_data.shape[0]

        self.num_classes = len(set(self.Answer))
        self.action_space = spaces.Discrete(self.num_classes)
        self.observation_space = gym.spaces.Box(low=-1, high=1, shape=(2048,))
        self.step_ind = 0
        self.y_pred = []

    def seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def step(self, a):
        self.y_pred.append(a)
        y_true_cur = []
        info = {}
        terminal = False
        if a == self.Answer[self.id[self.step_ind]]:
            reward = 1.
        else:
            reward = -1.
            if self.mode == 'train':  ##训练过程中，只要智能体猜错了，游戏就终止
                terminal = True

        self.step_ind += 1

        if self.step_ind == self.game_len-1:
            y_true_cur = self.Answer[self.id]
            terminal = True

        return self.Env_data[self.id[self.step_ind]], reward, terminal, info

    # return: (states, observations)
    def reset(self):
        if self.mode == 'train':
            np.random.shuffle(self.id)
        self.step_ind = 0
        self.y_pred = []
        return self.Env_data[self.id[self.step_ind]]

    def My_metrics(self, y_pre, y_true):
        confusion_mat = confusion_matrix(y_true, y_pre)
        print('\n')
        print(classification_report(y_true, y_pre))
        Acc = accuracy_score(y_true, y_pre) * 100  # OA
        Pre = precision_score(y_true, y_pre, average='weighted') * 100  # Precision
        Recall = recall_score(y_true, y_pre, average='weighted') * 100  # Recall_score
        F1  = f1_score(y_true, y_pre, average='weighted') * 100  # F1_score
        print(Acc)
        print(Pre)
        print(Recall)
        print(F1)
        return Acc, Pre,Recall,F1