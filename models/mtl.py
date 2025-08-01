##+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
## Modified from: https://github.com/yaoyao-liu/meta-transfer-learning
##
## This source code is licensed under the MIT-style license found in the
## LICENSE file in the root directory of this source tree
##+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
""" Meta Learner """
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from models.feature_extractor import Deap_FeatureExtractor

def maxup_criterion(logits, label_shot, num_of_method):
    if num_of_method == 1:
        loss = F.cross_entropy(logits, label_shot)
        return loss
    else:
        loss = F.cross_entropy(logits, label_shot, reduction = 'none') 
        loss = loss.view(num_of_method+1, -1) 
        loss, loc = loss.max(dim=0)
        return loss.mean()
    

class BaseLearner(nn.Module):
    """The class for inner loop."""
    def __init__(self, args, z_dim):
        super().__init__()
        self.classify = nn.Sequential(
            nn.Linear(in_features=z_dim, out_features=128),
            nn.BatchNorm1d(num_features=128),
            nn.ReLU(True),
            nn.Linear(in_features=128, out_features=2),
            nn.LogSoftmax(dim=1)
        )

    def forward(self, input_x, the_vars=None):
        return self.classify(input_x)

    def parameters(self):
        return list(self.classify.parameters())

class MtlLearner(nn.Module):
    """The class for outer loop."""
    def __init__(self, args, mode='meta', num_cls=2,n_channels=32, d_model=128,
                 ratio=2, dropout=0.5, num_of_method=1):
        super().__init__()
        self.args = args
        self.mode = mode
        self.update_lr = args.base_lr
        self.update_step = args.update_step
        self.encoder = Deap_FeatureExtractor(args.model_type, n_channels=n_channels, d_model=d_model,
                                             ratio=ratio, dropout=dropout)
        self.z_dim = self.encoder.out_features
        self.num_of_method = num_of_method

        #print("z_dim"+str(self.z_dim))
        self.base_learner = BaseLearner(args, self.z_dim)

        if self.mode != 'meta':
            self.pre_fc = nn.Sequential(nn.Linear(self.encoder.out_features, num_cls))

    def forward(self, inp):
        #这两种情况没有进行内层的参数更新
        if self.mode=='pre' or self.mode=='origval':
            return self.pretrain_forward(inp)
        elif self.mode=='preval':
            data_shot, label_shot, data_query = inp
            return self.preval_forward(data_shot, label_shot, data_query)
        elif self.mode=='group':
            return self.encoder_forward(inp)
        #meta_train阶段，base_learner是迭代更新的
        elif self.mode=='meta_train':
            data_shot, label_shot, data_query = inp
            return self.meta_forward(data_shot, label_shot, data_query)
        #meta_train阶段，引入最大化损失并更新
        elif self.mode=='meta_train_aug':
            data_shot, label_shot, data_query = inp
            return self.meta_forward_aug(data_shot, label_shot, data_query)
        #meta_test阶段，base_learner是不迭代更新的
        elif self.mode=='meta_test':
            data_shot, label_shot, data_query = inp
            return self.meta_forward_test(data_shot, label_shot, data_query)
        #meta_test阶段，引入最大化损失并更新
        elif self.mode=='meta_test_aug':
            data_shot, label_shot, data_query = inp
            return self.meta_forward_test_aug(data_shot, label_shot, data_query)
        elif self.mode == 'zeroShotTest':
            return self.base_learner(self.encoder(inp))
        else:
            raise ValueError('Please set the correct mode.')

    def pretrain_forward(self, inp):
        return F.softmax(self.pre_fc(self.encoder(inp)), dim=1)

    def preval_forward(self, data_shot, label_shot, data_query):
        embedding_shot = self.encoder(data_shot)
        embedding_query = self.encoder(data_query)
        params = self.base_learner.parameters()
        optimizer = optim.Adam(params)

        logits = self.base_learner(embedding_shot)
        loss = F.cross_entropy(logits, label_shot)
        loss.backward(retain_graph=True)
        optimizer.step()

        logits_q = self.base_learner(embedding_query)

        for _ in range(2):
            optimizer.zero_grad()
            logits = self.base_learner(embedding_shot)
            loss = F.cross_entropy(logits, label_shot)
            loss.backward(retain_graph=True)
            optimizer.step()

            logits_q = self.base_learner(embedding_query)

        return logits_q

    #生成encoder提取的中间特征向量
    def encoder_forward(self, inp):
        return self.encoder(inp)
    
    def meta_forward(self, data_shot, label_shot, data_query):
        embedding_shot=self.encoder(data_shot)
        embedding_query = self.encoder(data_query)
        params=self.base_learner.parameters()
        optimizer=optim.Adam(params)

        logits = self.base_learner(embedding_shot)
        loss = F.cross_entropy(logits, label_shot)
        loss.backward(retain_graph=True)
        optimizer.step()

        logits_q = self.base_learner(embedding_query)

        for _ in range(10):
            optimizer.zero_grad()
            logits = self.base_learner(embedding_shot)
            loss = F.cross_entropy(logits, label_shot)
            loss.backward(retain_graph=True)
            optimizer.step()

            logits_q = self.base_learner(embedding_query)

        return logits_q
    
    def meta_forward_test(self, data_shot, label_shot, data_query):
        embedding_shot=self.encoder(data_shot)
        embedding_query = self.encoder(data_query)

        torch.save(dict(params=self.base_learner.state_dict()), 'base_learner.pth')
        new_classify = BaseLearner(self.args, self.z_dim)
        new_classify.load_state_dict(torch.load('base_learner.pth')['params'])

        if torch.cuda.is_available():
            torch.backends.cudnn.benchmark = True
            new_classify = new_classify.cuda()

        params=new_classify.parameters()
        optimizer=optim.Adam(params)

        logits = new_classify(embedding_shot)

        """if label_shot.ndimension() == 2:
            loss = F.binary_cross_entropy_with_logits(logits, label_shot)
        else:
            loss = F.cross_entropy(logits, label_shot)"""
        loss = F.cross_entropy(logits, label_shot)

        loss.backward(retain_graph=True)
        optimizer.step()

        logits_q = new_classify(embedding_query)

        for _ in range(self.args.update_step):
            optimizer.zero_grad()
            logits = new_classify(embedding_shot)
            """if label_shot.ndimension() == 2:
            loss = F.binary_cross_entropy_with_logits(logits, label_shot)
            else:
                loss = F.cross_entropy(logits, label_shot)"""
            loss = F.cross_entropy(logits, label_shot)
            loss.backward(retain_graph=True)
            optimizer.step()

            logits_q = new_classify(embedding_query)

        return logits_q,logits
    
    def meta_forward_aug(self, data_shot, label_shot, data_query):
        embedding_shot=self.encoder(data_shot)
        embedding_query = self.encoder(data_query)
        params=self.base_learner.parameters()
        optimizer=optim.Adam(params)

        logits = self.base_learner(embedding_shot)
        # TODO maxup loss
        loss = maxup_criterion(logits, label_shot, self.num_of_method)
        # loss = F.cross_entropy(logits, label_shot)

        loss.backward(retain_graph=True)
        optimizer.step()

        logits_q = self.base_learner(embedding_query)

        for _ in range(10):
            optimizer.zero_grad()
            logits = self.base_learner(embedding_shot)

            # TODO maxup loss
            loss = maxup_criterion(logits, label_shot, self.num_of_method)
            # loss = F.cross_entropy(logits, label_shot)

            loss.backward(retain_graph=True)
            optimizer.step()

            logits_q = self.base_learner(embedding_query)

        return logits_q
    
    def meta_forward_test_aug(self, data_shot, label_shot, data_query):
        embedding_shot=self.encoder(data_shot)
        embedding_query = self.encoder(data_query)

        torch.save(dict(params=self.base_learner.state_dict()), 'base_learner.pth')
        new_classify = BaseLearner(self.args, self.z_dim)
        new_classify.load_state_dict(torch.load('base_learner.pth')['params'])

        if torch.cuda.is_available():
            torch.backends.cudnn.benchmark = True
            new_classify = new_classify.cuda()

        params=new_classify.parameters()
        optimizer=optim.Adam(params)

        logits = new_classify(embedding_shot)
        # TODO maxup loss
        loss = maxup_criterion(logits, label_shot, self.num_of_method)
        # loss = F.cross_entropy(logits, label_shot)
        loss.backward(retain_graph=True)
        optimizer.step()

        logits_q = new_classify(embedding_query)

        for _ in range(self.args.update_step):
            optimizer.zero_grad()
            logits = new_classify(embedding_shot)
            # TODO maxup loss
            loss = maxup_criterion(logits, label_shot, self.num_of_method)
            # loss = F.cross_entropy(logits, label_shot)
            loss.backward(retain_graph=True)
            optimizer.step()

            logits_q = new_classify(embedding_query)

        return logits_q,logits