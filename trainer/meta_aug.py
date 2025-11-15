# +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
## Modified from: https://github.com/yaoyao-liu/meta-transfer-learning
##
## This source code is licensed under the MIT-style license found in the
## LICENSE file in the root directory of this source tree
##+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
""" Trainer for meta-train phase. """
import os.path as osp
import os
import tqdm
import numpy as np
import time
import random
from sklearn.metrics import roc_auc_score, precision_score, recall_score, accuracy_score, f1_score
from sklearn.preprocessing import LabelBinarizer
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from tensorboardX import SummaryWriter

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.autograd import Variable

from utils.misc import Averager, Timer, count_acc, compute_confidence_interval, ensure_path
from dataloader.dataset_loader_deap import DatasetLoader_Deap_subjects as Dataset
from dataloader.samplers import CategoriesSampler,\
                                CategoriesSampler_Different_Subjects,CategoriesSampler_Different_Groups,\
                                CategoriesSampler_Different_Trails
from models.mtl import MtlLearner
from trainer.augmentations import data_aug


def data_aug_(data_support, label_support, data_query, label_query, method):
    new_data_s, new_label_s, new_data_q, new_label_q = \
        data_support.clone(), label_support.clone(), data_query.clone(), label_query.clone()
    if method == "s_ma":
        new_data_s, new_label_s = data_aug(data_support, label_support, 'ma')
    elif method == "s_antts":
        new_data_s, new_label_s = data_aug(data_support, label_support, 'antts')
    elif method == "s_fda":
        new_data_s, new_label_s = data_aug(data_support, label_support, 'fda')
    elif method == "q_ma":
        new_data_q, new_label_q = data_aug(data_query, label_query, 'ma')
    elif method == "q_rcs":
        new_data_q, new_label_q = data_aug(data_query, label_query, 'rcs')
    elif method == "q_an":
        new_data_q, new_label_q = data_aug(data_query, label_query, 'an')
    elif method == "s_ma+s_antts":
        new_data_s, new_label_s = data_aug(data_support, label_support, 'ma')
        new_data_s, new_label_s = data_aug(new_data_s, new_label_s, 'antts')
    elif method == "s_ma+s_fda":
        new_data_s, new_label_s = data_aug(data_support, label_support, 'ma')
        new_data_s, new_label_s = data_aug(new_data_s, new_label_s, 'fda')
    elif method == "q_ma+q_rcs":
        new_data_q, new_label_q = data_aug(data_query, label_query, 'ma')
        new_data_q, new_label_q = data_aug(new_data_q, new_label_q, 'rcs')
    elif method == "q_ma+q_an":
        new_data_q, new_label_q = data_aug(data_query, label_query, 'ma')
        new_data_q, new_label_q = data_aug(new_data_q, new_label_q, 'an')
    elif method == "s_ma+q_ma":
        new_data_s, new_label_s = data_aug(data_support, label_support, 'ma')
        new_data_q, new_label_q = data_aug(data_query, label_query, 'ma')
    elif method == "s_ma+q_rcs":
        new_data_s, new_label_s = data_aug(data_support, label_support, 'ma')
        new_data_q, new_label_q = data_aug(data_query, label_query, 'rcs')
    elif method == "s_antts+q_ma":
        new_data_s, new_label_s = data_aug(data_support, label_support, 'antts')
        new_data_q, new_label_q = data_aug(data_query, label_query, 'ma')
    elif method == "s_antts+q_rcs":
        new_data_s, new_label_s = data_aug(data_support, label_support, 'antts')
        new_data_q, new_label_q = data_aug(data_query, label_query, 'rcs')   
    
    return new_data_s, new_label_s, new_data_q, new_label_q


def data_aug_maxup(data_support, label_support, data_query, label_query, method_pol, num_of_method=2):
    new_data_s = torch.zeros([num_of_method] + list(data_support.size()))
    new_data_q = torch.zeros([num_of_method] + list(data_query.size()))
    new_label_s = torch.zeros([num_of_method] + list(label_support.size()), dtype=torch.long)
    new_label_q = torch.zeros([num_of_method] + list(label_query.size()), dtype=torch.long)

    for i in range(num_of_method):
        method = random.sample(method_pol, 1)[0]   
        new_data_s[i], new_label_s[i], new_data_q[i], new_label_q[i] = data_aug_(data_support, label_support, data_query, label_query, method)


    return new_data_s, new_label_s, new_data_q, new_label_q

def maxup_criterion(logits, label_query, num_of_method):
    if num_of_method == 1:
        loss = F.cross_entropy(logits, label_query.float())
        return loss
    else:
        loss = F.cross_entropy(logits, label_query.float(), reduction = 'none') 
        loss = loss.view(num_of_method+1, -1) 
        loss, loc = loss.max(dim=0)
        return loss.mean()

def mean_criterion(logits, label_query, num_of_method):
    if num_of_method == 1:
        loss = F.cross_entropy(logits, label_query.float())
        return loss
    else:
        loss = F.cross_entropy(logits, label_query.float(), reduction='none')
        loss = loss.view(num_of_method + 1, -1)
        loss = loss.mean(dim=0)  # 取平均而不是最大
        return loss.mean()


def maxup_accuracy(pred_query, label_query):
    # 获取预测值的最大值索引
    pred = torch.argmax(pred_query, dim=-1).view(-1)
    # 将 one-hot 标签转换为索引
    label = torch.argmax(label_query, dim=-1).view(-1)
    # 比较预测值与标签是否相等
    correct = pred.eq(label) 
    # 计算准确率
    accuracy = correct.sum().float() / label_query.size(0) * 100  
    return accuracy


class MetaTrainer_aug(object):
    """The class that contains the code for the meta-train phase and meta-eval phase."""

    def __init__(self, args): 

        if args.aug_pool:
            if args.aug_pool == 'only':
                self.da_pool = ["s_ma", "q_ma"]
            elif args.aug_pool == 'single':
                self.da_pool = ["s_ma", "s_antts", "s_fda", "q_ma", "q_rcs", "q_an"]
            elif args.aug_pool == 'medium':
                self.da_pool = ["s_ma+s_antts", "s_ma+s_fda", "q_ma+q_rcs", "q_ma+q_an", \
                                "s_ma+q_ma", "s_ma+q_rcs", "s_antts+q_ma", "s_antts+q_rcs"]
            elif args.aug_pool == 'large':
                self.da_pool = ["s_ma", "s_antts", "q_ma", "q_rcs", "s_ma+s_antts", "s_ma+s_fda",\
                                 "q_ma+q_rcs",  "q_ma+q_an", "s_ma+q_ma", "s_ma+q_rcs", \
                                 "s_antts+q_ma", "s_antts+q_rcs" ]
        else:
            self.da_pool = None

        self.num_of_methods = args.m
        self.model_train_mode = args.model_train_mode
        self.model_test_mode = args.model_test_mode

        # Set the folder to save the records and checkpoints
        # 一级目录
        #log_base_dir = './logs/'
        #将一半的影片作为验证集
        log_base_dir = args.log_base_dir

        if not osp.exists(log_base_dir):
            os.mkdir(log_base_dir)
        # 二级目录
        base_dir = '_'.join([args.dataset, args.model_type])
        base_dir = osp.join(log_base_dir, base_dir)
        if not osp.exists(base_dir):
            os.mkdir(base_dir)

        # 三级目录
        if args.support_aug :
            save_path1 = 'support_aug_' + str(args.support_aug) + '_m_' + str(self.num_of_methods)
        elif args.query_aug :
            save_path1 = 'query_aug_' + str(args.query_aug) + '_m_' + str(self.num_of_methods)
        elif args.aug_pool :
            save_path1 = 'aug_pool_' + str(args.aug_pool) + '_m_' + str(self.num_of_methods) + \
            '__' + str(self.model_train_mode) 
        else:
            if args.meta_task_sampler == 'DG':
                save_path1 = 'meta' +  '_meta_task_sampler_' + str(args.meta_task_sampler) + '_k_' + str(args.k)
            else:
                save_path1 = 'meta' + '_meta_task_sampler_' + str(args.meta_task_sampler)

        meta_base_dir = osp.join(base_dir, save_path1)
        if not osp.exists(meta_base_dir):
            os.mkdir(meta_base_dir)

        save_path2 = str(args.test_subject_id)
        args.save_path = meta_base_dir + '/test_subject_' + save_path2
        ensure_path(args.save_path)
        self.save_path = args.save_path

        # Set args to be shareable in the class
        self.args = args

        # Load meta-train set
        # print("Preparing dataset loader")
        self.trainset = Dataset('train', test_subject_id=self.args.test_subject_id,
                                partition_val_test=self.args.partition_val_test)
        # AS_Sampling from all samples
        if self.args.meta_task_sampler == 'AS':
            self.train_sampler = CategoriesSampler(self.trainset.label, self.args.num_batch, self.args.way,
                                               self.args.train_shot + self.args.train_query)
            self.train_loader = DataLoader(dataset=self.trainset, batch_sampler=self.train_sampler,
                                           num_workers=3, pin_memory=True)

        # Load meta-val set
        self.valset = Dataset('val', test_subject_id=self.args.test_subject_id,
                              partition_val_test=self.args.partition_val_test)
        self.val_sampler = CategoriesSampler(self.valset.label, 20, self.args.way, self.args.val_shot + self.args.val_query)
        self.val_loader = DataLoader(dataset=self.valset, batch_sampler=self.val_sampler, num_workers=4,
                                     pin_memory=True)

        # Load meta-test set
        self.testset = Dataset('test', test_subject_id=self.args.test_subject_id,
                              partition_val_test=self.args.partition_val_test)
        self.testData = self.testset.X_test
        self.testLabel = self.testset.y_test

        # Build meta-transfer learning model
        # num_samples
        num_channels, num_points = self.trainset.data.shape[1], self.trainset.data.shape[2]
        self.model = MtlLearner(self.args, mode='meta', num_cls=2, n_channels=num_channels, 
                                d_model=num_points, ratio=2, dropout=self.args.dropout,
                                num_of_method=self.num_of_methods)

        # Set optimizer
        self.optimizer = torch.optim.Adam(
            [{'params': filter(lambda p: p.requires_grad, self.model.encoder.parameters())}, \
             {'params': self.model.base_learner.parameters(), 'lr': self.args.meta_lr2}], lr=self.args.meta_lr1)
        # Set learning rate scheduler
        self.lr_scheduler = torch.optim.lr_scheduler.StepLR(self.optimizer, step_size=self.args.step_size,
                                                            gamma=self.args.gamma)

        # load pretrained model without FC classifier
        self.model_dict = self.model.state_dict()
        if self.args.init_weights is not None:
            pretrained_dict = torch.load(self.args.init_weights)['params']
        else:
            pre_base_dir = osp.join(base_dir, 'pre')
            pre_save_path2 = str(args.test_subject_id)
            pre_save_path = pre_base_dir + '/test_subject_' + pre_save_path2
            pretrained_dict = torch.load(osp.join(pre_save_path, 'max_acc.pth'))['params']
        pretrained_dict = {'encoder.' + k: v for k, v in pretrained_dict.items()}
        pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in self.model_dict}
        self.model_dict.update(pretrained_dict)
        self.model.load_state_dict(self.model_dict)

        # Set model to GPU
        if torch.cuda.is_available():
            torch.backends.cudnn.benchmark = True
            self.model = self.model.cuda()

    def generate_label_group(self):
        # data = self.trainset.data
        # label = self.trainset.label
        #if torch.cuda.is_available():
            #data = data.cuda()
        # print(data.shape)
        self.model.mode = 'group'
        #1.生成中间特征向量

        trainloader = DataLoader(self.trainset, batch_size = 1024)
        data = []
        label = []
        for x, y in trainloader:
            x = x.cuda()
            out_features = self.model(x)
            data.append(out_features.detach().cpu().numpy())
            label.append(y.numpy())

        out_features = np.concatenate(data)
        label = np.concatenate(label)

        # out_features = self.model.cpu()(data)
        #print(out_features.shape)

        #2.PCA降维
        # out_features = out_features.detach().cpu().numpy()
        # 创建PCA对象，设置降维后的维度
        pca = PCA(n_components=100)
        # 对数据进行降维转换
        reduced_features = pca.fit_transform(out_features)
        # 输出降维后的结果
        #print(reduced_features.shape)

        #3.分别对data_0和data_1进行聚类
        label_0 = np.argwhere(label == 0).reshape(-1)
        label_1 = np.argwhere(label == 1).reshape(-1)
        data_0 = reduced_features[label_0]
        data_1 = reduced_features[label_1]

        kmeans_model = KMeans(n_clusters=self.args.k)

        kmeans_model.fit(data_0)
        label_0_group = kmeans_model.predict(data_0)

        kmeans_model.fit(data_1)
        label_1_group = kmeans_model.predict(data_1)

        label_group = []
        label_group.append(label_0_group)
        label_group.append(label_1_group)

        return label_group

    def save_model(self, name):
        """The function to save checkpoints.
        Args:
          name: the name for saved checkpoint
        """
        torch.save(dict(params=self.model.state_dict()), osp.join(self.args.save_path, name + '.pth'))

    def train(self):
        """The function for the meta-train phase."""
        # Set the meta-train log
        trlog = {}
        trlog['args'] = vars(self.args)
        trlog['train_loss'] = []
        trlog['val_loss'] = []
        trlog['train_acc'] = []
        trlog['val_acc'] = []
        trlog['max_acc'] = 0.0
        trlog['max_acc_epoch'] = 0

        # Set the timer
        timer = Timer()
        # Set global count to zero
        global_count = 0
        # Set tensorboardX
        print("summarywritter")
        writer = SummaryWriter(comment=self.args.save_path)
        print("llsummarywritter")

        # Generate the labels for train set of the episodes
        label_shot = torch.arange(self.args.way).repeat(self.args.train_shot)
        one_hot_label_shot = torch.zeros(label_shot.size(0), self.args.way)
        one_hot_label_shot[torch.arange(label_shot.size(0)), label_shot] = 1
        label_shot = one_hot_label_shot

        if torch.cuda.is_available():
            label_shot = label_shot.type(torch.cuda.LongTensor)
        else:
            label_shot = label_shot.type(torch.LongTensor)

        val_label_shot = torch.arange(self.args.way).repeat(self.args.val_shot)
        one_hot_val_label_shot = torch.zeros(val_label_shot.size(0), self.args.way)
        one_hot_val_label_shot[torch.arange(val_label_shot.size(0)), val_label_shot] = 1
        val_label_shot = one_hot_val_label_shot
        if torch.cuda.is_available():
            val_label_shot = val_label_shot.type(torch.cuda.LongTensor)
        else:
            val_label_shot = val_label_shot.type(torch.LongTensor)

        #生成分组
        #这里在训练循环的外面，所以分组不会改变（可以试一下adaptive的分组生成）
        label_group = self.generate_label_group()

        if self.args.meta_task_sampler == 'AS':
            train_loader = self.train_loader

        # Start meta-train
        # start_time = time.time()
        #self.model.mode = self.model_train_mode
        for epoch in range(1, self.args.max_epoch + 1):

            self.model.mode = self.model_train_mode
            start_time = time.time()
            # Update learning rate
            self.lr_scheduler.step()
            # Set the model to train mode
            self.model.train()
            # Set averager classes to record training losses and accuracies
            train_loss_averager = Averager()
            train_acc_averager = Averager()

            # DS_Sampling from different subjects
            if self.args.meta_task_sampler == 'DS':
                #print("current sampler: DS")
                train_shot_subject = epoch % 31
                train_sampler = CategoriesSampler_Different_Subjects(self.trainset.label, train_shot_subject,
                                                                     self.args.num_batch, self.args.way,
                                                                     self.args.train_shot, self.args.train_query)
                train_loader = DataLoader(dataset=self.trainset, batch_sampler=train_sampler, num_workers=3,
                                          pin_memory=True)
            # DG_Sampling from different groups
            elif self.args.meta_task_sampler == 'DG':
                train_shot_group = epoch%self.args.k
                train_sampler = CategoriesSampler_Different_Groups(self.args.k, label_group,
                                                               self.trainset.label, train_shot_group,
                            self.args.num_batch, self.args.way, self.args.train_shot, self.args.train_query)
                train_loader = DataLoader(dataset=self.trainset, batch_sampler=train_sampler, num_workers=3,
                                           pin_memory=True)
            elif self.args.meta_task_sampler == 'DT':
                train_subject = epoch % 31
                train_sampler = CategoriesSampler_Different_Trails(self.trainset.label, train_subject,
                                                                     self.args.num_batch, self.args.way,
                                                                     self.args.train_shot, self.args.train_query)
                train_loader = DataLoader(dataset=self.trainset, batch_sampler=train_sampler, num_workers=3,
                                          pin_memory=True)
                
            # Generate the labels for test set of the episodes during meta-train updates
            label = torch.arange(self.args.way).repeat(self.args.train_query)
            one_hot_label = torch.zeros(label.size(0), self.args.way)
            one_hot_label[torch.arange(label.size(0)), label] = 1
            label = one_hot_label

            if torch.cuda.is_available():
                label = label.type(torch.cuda.LongTensor)
            else:
                label = label.type(torch.LongTensor)
            
            # print(label)
            
            # Using tqdm to read samples from train loader
            print('Epoch {}---------------train-----------------'.format(epoch))
            tqdm_gen = tqdm.tqdm(train_loader)
            for i, batch in enumerate(tqdm_gen, 1):
                # Update global count number
                global_count = global_count + 1
                if torch.cuda.is_available():
                    data, _ = [_.cuda() for _ in batch]
                else:
                    data = batch[0]
                p = self.args.train_shot * self.args.way
                data_shot, data_query = data[:p], data[p:]

                # data_shot (20, 32, 128) data_query (40, 32, 128)
                # print(data_shot.shape, data_query.shape)
                # print(label_shot.shape) # 20
                # print(label.shape) # 40 

                # TODO data augment
                #生成样本
                new_data_shot, new_label_shot, new_data_query, new_label = \
                    data_aug_maxup(data_shot, label_shot, data_query, label, self.da_pool, self.num_of_methods)
                
                # TODO reshape
                new_data_shot = new_data_shot.reshape(-1, *data_shot.shape[-2:]).to(data_shot.device)
                new_label_shot = new_label_shot.reshape(-1, *label_shot.shape[-1:]).to(label_shot.device)
                new_data_query = new_data_query.reshape(-1, *data_query.shape[-2:]).to(data_query.device)
                new_label = new_label.reshape(-1, *label.shape[-1:]).to(label.device)

                #连接生成样本和原始样本
                merge_data_shot = torch.cat((data_shot, new_data_shot), dim=0)
                merge_label_shot = torch.cat((label_shot, new_label_shot), dim=0)
                merge_data_query = torch.cat((data_query, new_data_query), dim=0)
                merge_label = torch.cat((label, new_label), dim=0)

                # TODO compute logits
                logits_q = self.model((merge_data_shot, merge_label_shot.float(), merge_data_query))

                # TODO compute loss and acc
                # loss = maxup_criterion(logits_q, merge_label.float(), self.num_of_methods)
                loss = mean_criterion(logits_q, merge_label.float(), self.num_of_methods)
                acc = maxup_accuracy(logits_q, merge_label)

                # Write the tensorboardX records
                writer.add_scalar('data/loss', float(loss), global_count)
                writer.add_scalar('data/acc', float(acc), global_count)

                # Add loss and accuracy for the averagers
                train_loss_averager.add(loss.item())
                train_acc_averager.add(acc)

                # Loss backwards and optimizer updates
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

            # Update the averagers
            train_loss_averager = train_loss_averager.item()
            train_acc_averager = train_acc_averager.item()

            print('Epoch {}, Train, Loss={:.4f} Acc={:.4f}'.format(epoch, train_loss_averager, train_acc_averager))
            print("--- %s seconds ---" % (time.time() - start_time))
            trlog['train_loss'].append(train_loss_averager)
            trlog['train_acc'].append(train_acc_averager)

            # Start validation for this epoch, set model to eval mode
            self.model.eval()

            # Set averager classes to record validation losses and accuracies
            val_loss_averager = Averager()
            val_acc_averager = Averager()

            # Generate the labels for test set of the episodes during meta-val for this epoch
            label = torch.arange(self.args.way).repeat(self.args.val_query)
            one_hot_label = torch.zeros(label.size(0), self.args.way)
            one_hot_label[torch.arange(label.size(0)), label] = 1
            label = one_hot_label

            if torch.cuda.is_available():
                label = label.type(torch.cuda.LongTensor)
            else:
                label = label.type(torch.LongTensor)

            # Print previous information
            if epoch % 10 == 0:
                self.model.mode = 'meta_train'
                print('Best Epoch {}, Best Val Acc={:.4f}'.format(trlog['max_acc_epoch'], trlog['max_acc']))

                # Run meta-validation
                print('Epoch {}---------------valid----------------'.format(epoch))
                for i, batch in enumerate(self.val_loader, 1):
                    if torch.cuda.is_available():
                        data, _ = [_.cuda() for _ in batch]
                    else:
                        data = batch[0]
                    p = self.args.val_shot * self.args.way
                    data_shot, data_query = data[:p], data[p:]
                    logits = self.model((data_shot, val_label_shot.float(), data_query))
                    #loss = F.binary_cross_entropy_with_logits(logits.float(), label.float())
                    loss = F.cross_entropy(logits, label.float())
                    acc = count_acc(logits, label)

                    val_loss_averager.add(loss.item())
                    val_acc_averager.add(acc)

                # Update validation averagers
                val_loss_averager = val_loss_averager.item()
                val_acc_averager = val_acc_averager.item()
                # Write the tensorboardX records
                writer.add_scalar('data/val_loss', float(val_loss_averager), epoch)
                writer.add_scalar('data/val_acc', float(val_acc_averager), epoch)
                # Print loss and accuracy for this epoch
                print('Epoch {}, Val, Loss={:.4f} Acc={:.4f}'.format(epoch, val_loss_averager, val_acc_averager))

                # Update best saved model
                if val_acc_averager > trlog['max_acc']:
                    trlog['max_acc'] = val_acc_averager
                    trlog['max_acc_epoch'] = epoch
                    self.save_model('max_acc')
                # Save model every 10 epochs
                if epoch % 10 == 0:
                    self.save_model('epoch' + str(epoch))

                # Update the logs
                #trlog['train_loss'].append(train_loss_averager)
                #trlog['train_acc'].append(train_acc_averager)
                trlog['val_loss'].append(val_loss_averager)
                trlog['val_acc'].append(val_acc_averager)

                # Save log
                torch.save(trlog, osp.join(self.args.save_path, 'trlog'))

                print('Running Time: {}, Estimated Time: {}'.format(timer.measure(),
                                                                    timer.measure(epoch / self.args.max_epoch)))

        print('Total Running Time: {}'.format(timer.measure()))

        time_trlog = {}
        time_trlog['time'] = timer.measure()
        # Save log
        torch.save(time_trlog, osp.join(self.args.save_path, "time_trlog"))

        writer.close()


    def eval(self):
        """The function for the meta-eval phase."""

        # Load the logs
        def multiclass_roc_auc_score(y_test, y_pred, average="macro"):
            lb = LabelBinarizer()
            lb.fit(y_test)
            y_test = lb.transform(y_test)
            y_pred = lb.transform(y_pred)
            return roc_auc_score(y_test, y_pred, average=average)

        trlog = torch.load(osp.join(self.args.save_path, 'trlog'))

        self.model.mode = self.model_test_mode
        # Load meta-test set
        test_set = Dataset('test', test_subject_id=self.args.test_subject_id,
                           partition_val_test=self.args.partition_val_test)
        
        test_num_batch = self.args.test_num_batch
        sampler = CategoriesSampler(test_set.label, test_num_batch, self.args.way,
                                    self.args.val_shot + self.args.val_query)
        loader = DataLoader(test_set, batch_sampler=sampler, num_workers=8, pin_memory=True)

        # Set test accuracy recorder
        test_acc_record = np.zeros((test_num_batch,))
        test_f1_record = np.zeros((test_num_batch,))
        test_auc_record = np.zeros((test_num_batch,))

        # shot acc
        shot_acc_record = np.zeros((test_num_batch,))

        # Load model for meta-test phase
        if self.args.eval_weights is not None:
            self.model.load_state_dict(torch.load(self.args.eval_weights)['params'])
        else:
            self.model.load_state_dict(torch.load(osp.join(self.args.save_path, 'max_acc' + '.pth'))['params'])
        # Set model to eval mode
        self.model.eval()

        # Set accuracy averager
        ave_acc = Averager()

        shot_ave_acc = Averager()

        # Generate labels

        label = torch.arange(self.args.way).repeat(self.args.val_query)
        one_hot_label = torch.zeros(label.size(0), self.args.way)
        one_hot_label[torch.arange(label.size(0)), label] = 1
        if torch.cuda.is_available():
            label = label.type(torch.cuda.LongTensor)
            one_hot_label = one_hot_label.type(torch.cuda.LongTensor)
        else:
            label = label.type(torch.LongTensor)
            one_hot_label = one_hot_label.type(torch.LongTensor)

        label_shot = torch.arange(self.args.way).repeat(self.args.val_shot)
        one_hot_label_shot = torch.zeros(label_shot.size(0), self.args.way)
        one_hot_label_shot[torch.arange(label_shot.size(0)), label_shot] = 1

        if torch.cuda.is_available():
            label_shot = label_shot.type(torch.cuda.LongTensor)
            one_hot_label_shot = one_hot_label_shot.type(torch.cuda.LongTensor)
        else:
            label_shot = label_shot.type(torch.LongTensor)
            one_hot_label_shot = one_hot_label_shot.type(torch.LongTensor)


        Y = label.data.cpu().numpy()
        # Start meta-test
        for i, batch in enumerate(loader, 1):
            if torch.cuda.is_available():
                data, _ = [_.cuda() for _ in batch]
            else:
                data = batch[0]
            k = self.args.way * self.args.val_shot
            data_shot, data_query = data[:k], data[k:]

            ## data augmentation for support data
            if self.args.test_support_aug:
                if self.args.test_support_aug == 'ma':
                    #mix-up使用的是one-hot编码
                    new_shot_data, new_shot_label = data_aug(data_shot, one_hot_label_shot, 
                                                    self.args.test_support_aug)

                    merged_label_shot = torch.cat((one_hot_label_shot, new_shot_label), dim=0)
                else:
                    new_shot_data, new_shot_label = data_aug(data_shot, label_shot, 
                                                    self.args.test_support_aug)

                    merged_label_shot = torch.cat((label_shot, new_shot_label), dim=0)
                merged_shot = torch.cat((data_shot, new_shot_data), dim=0)   
                logits_q,logits = self.model((merged_shot, merged_label_shot, data_query))      
            else:
                logits_q,logits = self.model((data_shot, label_shot, data_query))


            #data_query上的准确率
            acc = count_acc(logits_q, label)
            logits_q = logits_q.data.cpu().numpy()
            predicted = np.argmax(logits_q, axis=1)
            f1 = f1_score(Y, predicted, average='macro')
            auc = multiclass_roc_auc_score(Y, predicted)
            ave_acc.add(acc)
            test_acc_record[i - 1] = acc
            test_f1_record[i - 1] = f1
            test_auc_record[i - 1] = auc

            #data_shot上的准确率
            #shot_acc = count_acc(logits, merged_label)
            #shot_ave_acc.add(shot_acc)


            if i % 100 == 0:
                print('batch {}: {:.2f}({:.2f})'.format(i, ave_acc.item() * 100, acc * 100))

        # Calculate the confidence interval, update the logs
        m, pm = compute_confidence_interval(test_acc_record)
        f1_m, f1_pm = compute_confidence_interval(test_f1_record)
        auc_m, auc_pm = compute_confidence_interval(test_auc_record)

        print('Val Best Epoch {}, Acc {:.4f}, Test Acc {:.4f}'.format(trlog['max_acc_epoch'], trlog['max_acc'],
                                                                      ave_acc.item()))
        test_trlog = {}
        test_trlog['args'] = vars(self.args)
        test_trlog['ave_acc'] = ave_acc.item()
        test_trlog['Test_Acc'] = m
        test_trlog['Test_Acc_variance'] = pm
        test_trlog['Test_f1'] = f1_m
        test_trlog['Test_f1_variance'] = f1_pm
        test_trlog['Test_auc'] = auc_m
        test_trlog['Test_auc_variance'] = auc_pm

        test_trlog['shot_ave_acc'] = shot_ave_acc.item()
        # Save log
        trlog_name = 'test_trlog_shot'+str(self.args.val_shot)+'_aug_'+str(self.args.test_support_aug)\
                    +'__'+str(self.model_test_mode)
        torch.save(test_trlog, osp.join(self.args.save_path, trlog_name))
