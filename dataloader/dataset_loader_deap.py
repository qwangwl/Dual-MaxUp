import numpy as np
import time
import scipy
import scipy.signal
import scipy.io
# import self defined functions
import torch
from torch.utils.data import Dataset
import random
import scipy.io as sio

#load the samples that have been splited
def get_data_label(save_path, input_file="s01"):
    data_file = save_path + input_file + ".npy"
    samples = np.load(data_file, allow_pickle=True)

    data = samples.item()['Data']
    label = samples.item()['Label']

    return data, label
#shuffle the samples
def shuffle_samples(data, label):
    idx = list(range(len(label)))
    np.random.shuffle(idx)
    data = data[idx]
    label = label[idx]
    print(len(label))
    print(label)
    return data,label

class DatasetLoader_Deap_subjects(Dataset):

    def __init__(self, setname, test_subject_id, partition_val_test, oneHot=False):
        save_path = "E:\\zq_codes\\data\\static\\"

        train_X = []
        train_y = []
        test_X = []
        test_y = []

        subjects = ["s{:02d}".format(i) for i in range(1, 33)]
        for sub_idx, sub in enumerate(subjects):
            dual_data, dual_label = get_data_label(save_path, input_file=sub)
            if (sub_idx == test_subject_id-1):
                test_X.append(dual_data)
                test_y.append(dual_label)
            else:
                train_X.append(dual_data)
                train_y.append(dual_label)

        train_X = np.array(train_X)
        train_y = np.array(train_y)
        test_X = np.array(test_X)
        test_y = np.array(test_y)

        train_X = train_X.reshape((-1, 32, 128))
        train_y = train_y.reshape((-1,))
        test_X = test_X.reshape((-1, 32, 128))
        test_y = test_y.reshape((-1,))

        #将测试受试者的部分影片的数据划分为验证集，这样做的目的是为了测试集中样本分布均衡
        split_id1 = int(len(np.where(test_y == 0)[0]) / partition_val_test)
        split_id2 = int(2400*(partition_val_test-1)/partition_val_test) + split_id1

        val_X = test_X[:split_id1]
        val_X = np.append(val_X, test_X[split_id2:], axis=0)
        val_y = test_y[:split_id1]
        val_y = np.append(val_y, test_y[split_id2:], axis=0)
        # 剩下的样本作为测试集
        test_X = test_X[split_id1:split_id2]
        test_y = test_y[split_id1:split_id2]

        train_X = train_X.astype('float32')
        val_X = val_X.astype('float32')
        test_X = test_X.astype('float32')

        self.X_val = val_X
        self.y_val = val_y

        self.X_test = test_X
        self.y_test = test_y
        
        if setname == 'train':
            self.data =  torch.from_numpy(train_X).type(torch.Tensor)
            self.label =  torch.from_numpy(train_y).type(torch.Tensor)
        elif setname == 'val':
            self.data =  torch.from_numpy(val_X).type(torch.Tensor)
            self.label =  torch.from_numpy(val_y).type(torch.Tensor)
        elif setname == 'test':
            # 把测试集打乱
            #test_X, test_y = shuffle_samples(test_X, test_y)
            self.data =  torch.from_numpy(test_X).type(torch.Tensor)
            self.label =  torch.from_numpy(test_y).type(torch.Tensor)

        #print(train_X.shape,train_y.shape)
        #print(val_X.shape, val_y.shape)
        #print(test_X.shape, test_y.shape)

        if oneHot :
            self.label = self.label.type(torch.LongTensor)
            one_hot_label = torch.zeros(self.label.size(0), 2)
            one_hot_label[torch.arange(self.label.size(0)), self.label] = 1
            self.label = one_hot_label
        
        self.num_class = 2

    def __len__(self):
        return len(self.data)

    def __getitem__(self, i):
        data, label = self.data[i], self.label[i]
        return data, label

    def get_labels(self):
        return self.label


