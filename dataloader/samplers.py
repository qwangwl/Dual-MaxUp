##+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
## Modified from: https://github.com/Sha-Lab/FEAT
##
## This source code is licensed under the MIT-style license found in the
## LICENSE file in the root directory of this source tree
##+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
""" Sampler for dataloader. """
import torch
import numpy as np
import random

class CategoriesSampler():
    """The class to generate episodic data"""
    def __init__(self, label, n_batch, n_cls, n_per):
        self.n_batch = n_batch
        self.n_cls = n_cls
        self.n_per = n_per

        label = np.array(label)
        self.m_ind = []
        for i in range(n_cls):
            ind = np.argwhere(label == i).reshape(-1)
            ind = torch.from_numpy(ind)
            self.m_ind.append(ind)

            #查看每一类的样本有多少
            #print(len(ind))

            #查看每一类样本的编号
            #print(ind)

    def __len__(self):
        return self.n_batch
    def __iter__(self):
        for i_batch in range(self.n_batch):
            batch = []
            for c in range(self.n_cls):
                l = self.m_ind[c]
                pos = torch.randperm(len(l))[:self.n_per]
                batch.append(l[pos])
            batch = torch.stack(batch).t().reshape(-1)
            #查看一个batch中的样本编号
            #print('i_batch',i_batch)
            #print(batch)
            yield batch

class CategoriesSampler_Different_Subjects():
    """The class to generate episodic data"""

    def __init__(self, label, train_shot_subject, n_batch, n_cls, n_shot, n_query):
        self.n_batch = n_batch
        self.n_cls = n_cls
        self.n_shot = n_shot
        self.n_query = n_query

        self.train_shot_subject = train_shot_subject
        self.train_query_subject = random.choice(np.arange(31))
        while self.train_shot_subject == self.train_query_subject:
            self.train_query_subject = random.choice(np.arange(31))

        label = np.array(label)
        self.m_ind_shot = []
        self.m_ind_query = []
        for i in range(n_cls):
            ind = np.argwhere(label == i).reshape(-1)
            shot_id = np.argwhere(
                (ind > (self.train_shot_subject*2400)-1) * (ind < ((self.train_shot_subject+1)*2400))).reshape(-1)
            query_id = np.argwhere(
                (ind > (self.train_query_subject*2400)-1) * (ind < ((self.train_query_subject+1)*2400))).reshape(-1)
            ind_shot = torch.from_numpy(ind[shot_id])
            ind_query = torch.from_numpy(ind[query_id])
            self.m_ind_shot.append(ind_shot)
            self.m_ind_query.append(ind_query)
            # 查看每一类的样本有多少
            # print(len(ind))

            # 查看每一类样本的编号
            # print(ind)



    def __len__(self):
        return self.n_batch

    def __iter__(self):
        for i_batch in range(self.n_batch):
            batch = []
            for c in range(self.n_cls):
                l_shot = self.m_ind_shot[c]
                l_query = self.m_ind_query[c]
                pos_shot = torch.randperm(len(l_shot))[:self.n_shot]
                pos_query = torch.randperm(len(l_query))[:self.n_query]
                pos = torch.cat([l_shot[pos_shot], l_query[pos_query]], 0)
                batch.append(pos)
            batch = torch.stack(batch).t().reshape(-1)
            # 查看一个batch中的样本编号
            # print('i_batch',i_batch)
            # print(batch)
            yield batch

class CategoriesSampler_Different_Groups():
    """The class to generate episodic data"""

    def __init__(self, num_group, label_group, label, train_shot_group, n_batch, n_cls, n_shot, n_query):
        self.n_batch = n_batch
        self.n_cls = n_cls
        self.n_shot = n_shot
        self.n_query = n_query

        train_query_group = random.choice(np.arange(num_group))
        while train_shot_group == train_query_group:
            train_query_group = random.choice(np.arange(num_group))

        label = np.array(label)
        self.m_ind_shot = []
        self.m_ind_query = []
        for i in range(n_cls):
            ind = np.argwhere(label == i).reshape(-1)
            shot_id = ind[np.argwhere(label_group[i] == train_shot_group).reshape(-1)]
            query_id = ind[np.argwhere(label_group[i] == train_query_group).reshape(-1)]
            self.m_ind_shot.append(torch.from_numpy(shot_id))
            self.m_ind_query.append(torch.from_numpy(query_id))
            # 查看每一类的样本有多少
            # print(len(ind))

            # 查看每一类样本的编号
            # print(ind)

    def __len__(self):
        return self.n_batch

    def __iter__(self):
        #print(self.n_batch)
        for i_batch in range(self.n_batch):
            batch = []
            for c in range(self.n_cls):
                l_shot = self.m_ind_shot[c]
                l_query = self.m_ind_query[c]
                pos_shot = torch.randperm(len(l_shot))[:self.n_shot]
                pos_query = torch.randperm(len(l_query))[:self.n_query]
                pos = torch.cat([l_shot[pos_shot], l_query[pos_query]], 0)
                batch.append(pos)
            batch = torch.stack(batch).t().reshape(-1)
            # 查看一个batch中的样本编号
            #print('i_batch',i_batch)
            #print(batch)
            yield batch

#从一个受试者的不同影片中进行采样
class CategoriesSampler_Different_Trails():
    """The class to generate episodic data"""

    def __init__(self, label, train_subject, n_batch, n_cls, n_shot, n_query):
        self.n_batch = n_batch
        self.n_cls = n_cls
        self.n_shot = n_shot
        self.n_query = n_query

        label = np.array(label)
        self.m_ind_shot = []
        self.m_ind_query = []
        for i in range(n_cls):
            ind = np.argwhere(label == i).reshape(-1)
            #1.取出train_subject对应的ind
            sub_ind = np.argwhere(
                (ind > (train_subject*2400)-1) * (ind < ((train_subject+1)*2400))).reshape(-1)
            #2.对半划分为shot_id和query_id
            shot_id = sub_ind[:len(sub_ind)//2]
            query_id = sub_ind[len(sub_ind)//2:]

            ind_shot = torch.from_numpy(ind[shot_id])
            ind_query = torch.from_numpy(ind[query_id])
            self.m_ind_shot.append(ind_shot)
            self.m_ind_query.append(ind_query)
            # 查看每一类的样本有多少
            # print(len(ind))

            # 查看每一类样本的编号
            # print(ind)

    def __len__(self):
        return self.n_batch

    def __iter__(self):
        for i_batch in range(self.n_batch):
            batch = []
            for c in range(self.n_cls):
                l_shot = self.m_ind_shot[c]
                l_query = self.m_ind_query[c]
                pos_shot = torch.randperm(len(l_shot))[:self.n_shot]
                pos_query = torch.randperm(len(l_query))[:self.n_query]
                pos = torch.cat([l_shot[pos_shot], l_query[pos_query]], 0)
                batch.append(pos)
            batch = torch.stack(batch).t().reshape(-1)
            # 查看一个batch中的样本编号
            # print('i_batch',i_batch)
            # print(batch)
            yield batch
#meta_test
class No_Duplicate_CategoriesSampler():
    """The class to generate episodic data"""

    def __init__(self, label, n_batch, n_cls, n_per):
        self.n_batch = n_batch
        self.n_cls = n_cls
        self.n_per = n_per

        label = np.array(label)
        self.m_ind = []
        for i in range(n_cls):
            ind = np.argwhere(label == i).reshape(-1)
            ind = torch.from_numpy(ind)
            self.m_ind.append(ind)

            # 查看每一类的样本有多少
            # print(len(ind))

            # 查看每一类样本的编号
            # print(ind)

    def __len__(self):
        return self.n_batch

    def __iter__(self):
        m_ind_temp = self.m_ind
        for i_batch in range(self.n_batch):
            batch = []
            for c in range(self.n_cls):
                l = m_ind_temp[c]
                pos = torch.randperm(len(l))[:self.n_per]
                batch.append(l[pos])
                #不返回取样
                l_new = []
                for i, j in enumerate(l):
                    if i not in pos:
                        l_new = np.append(l_new, j)
                #检验l_new是否为空，为空时类型是list
                #if not isinstance(l_new,list):
                l_new = l_new.astype('int64')
                l_new = torch.from_numpy(l_new)
                m_ind_temp[c] = l_new

            batch = torch.stack(batch).t().reshape(-1)
            # 查看一个batch中的样本编号
            # print('i_batch',i_batch)
            # print(batch)
            yield batch

