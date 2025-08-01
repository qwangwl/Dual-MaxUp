##+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
## Modified from: https://github.com/yaoyao-liu/meta-transfer-learning
##
## This source code is licensed under the MIT-style license found in the
## LICENSE file in the root directory of this source tree
##+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
""" Trainer for pretrain phase. """
import os.path as osp
import os
import tqdm
import numpy as np
from sklearn.metrics import roc_auc_score, precision_score, recall_score, accuracy_score
from tensorboardX import SummaryWriter

import torch
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.autograd import Variable

from dataloader.samplers import CategoriesSampler
from dataloader.dataset_loader_deap import DatasetLoader_Deap_subjects as Dataset
from models.mtl import MtlLearner
from utils.misc import Averager, Timer, count_acc, ensure_path

class PreTrainer(object):
    """The class that contains the code for the pretrain phase."""

    def __init__(self, args):
        # Set the folder to save the records and checkpoints
        #一级目录
        #log_base_dir = './logs/'
        #将一半的影片作为验证集
        log_base_dir = args.log_base_dir
        #将dropout=0.5->0.2
        #log_base_dir = './logs_dropout_0.2/'
        if not osp.exists(log_base_dir):
            os.mkdir(log_base_dir)
        # 二级目录
        base_dir = '_'.join([args.dataset, args.model_type])
        base_dir = osp.join(log_base_dir, base_dir)
        if not osp.exists(base_dir):
            os.mkdir(base_dir)

        # 三级目录
        save_path1 = 'pre'
        pre_base_dir = osp.join(base_dir, save_path1)
        if not osp.exists(pre_base_dir):
            os.mkdir(pre_base_dir)

        save_path2 = str(args.test_subject_id)
        args.save_path = pre_base_dir + '/test_subject_' + save_path2
        ensure_path(args.save_path)
        self.save_path = args.save_path

        # Set args to be shareable in the class
        self.args = args

        # Load pretrain set
        print("Preparing dataset loader")
        self.trainset = Dataset('train', test_subject_id=self.args.test_subject_id,
                                partition_val_test=self.args.partition_val_test)
        self.train_loader = DataLoader(dataset=self.trainset, batch_size=args.pre_batch_size, shuffle=True,
                                       num_workers=4, pin_memory=True)

        # Load meta-val set
        self.valset = Dataset('val', test_subject_id=self.args.test_subject_id,
                              partition_val_test=self.args.partition_val_test)
        self.val_sampler = CategoriesSampler(self.valset.label, 20, self.args.way, self.args.val_shot + self.args.val_query)
        self.val_loader = DataLoader(dataset=self.valset, batch_sampler=self.val_sampler, num_workers=4,
                                     pin_memory=True)

        # Set pretrain class number
        num_class_pretrain = self.trainset.num_class

        # Build pretrain model
        num_channels, num_points = self.trainset.data.shape[1], self.trainset.data.shape[2]
        self.model = MtlLearner(self.args, mode='pre', num_cls=2,
                                n_channels=num_channels, d_model=num_points, ratio=2, dropout=self.args.dropout)
        # self.model=self.model.float()

        # Set optimizer
        params = list(self.model.encoder.parameters()) + list(self.model.pre_fc.parameters())
        self.optimizer = optim.Adam(params)

        # Set model to GPU
        if torch.cuda.is_available():
            torch.backends.cudnn.benchmark = True
            self.model = self.model.cuda()

    def save_model(self, name):
        """The function to save checkpoints.
        Args:
          name: the name for saved checkpoint
        """
        torch.save(dict(params=self.model.encoder.state_dict()), osp.join(self.args.save_path, name + '.pth'))

    def train(self):
        """The function for the pre-train phase."""

        # Set the pretrain log
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
        writer = SummaryWriter(comment=self.args.save_path)

        # Start pretrain
        for epoch in range(1, self.args.pre_max_epoch + 1):
            # Set the model to train mode

            print('Epoch {}'.format(epoch))
            self.model.train()
            self.model.mode = 'pre'
            # Set averager classes to record training losses and accuracies
            train_loss_averager = Averager()
            train_acc_averager = Averager()

            # Using tqdm to read samples from train loader
            tqdm_gen = tqdm.tqdm(self.train_loader)
            # for i, batch in enumerate(self.train_loader):
            for i, batch in enumerate(tqdm_gen, 1):
                # Update global count number
                global_count = global_count + 1
                if torch.cuda.is_available():
                    data, _ = [_.cuda() for _ in batch]
                else:
                    data = batch[0]
                label = batch[1]
                if torch.cuda.is_available():
                    label = label.type(torch.cuda.LongTensor)
                else:
                    label = label.type(torch.LongTensor)
                logits = self.model(data)
                loss = F.cross_entropy(logits, label)
                # Calculate train accuracy
                acc = count_acc(logits, label)

                # Write the tensorboardX records
                writer.add_scalar('data/loss', float(loss), global_count)
                writer.add_scalar('data/acc', float(acc), global_count)
                # Print loss and accuracy for this step
                train_loss_averager.add(loss.item())
                train_acc_averager.add(acc)

                # Loss backwards and optimizer updates
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

            # Update the averagers
            train_loss_averager = train_loss_averager.item()
            train_acc_averager = train_acc_averager.item()

            print('epoch:{} , train_loss_averager: {:.4f}, train_acc_averager: {:.4f}'.format(epoch, train_loss_averager,
                                                                                      train_acc_averager))

            # start the original evaluation
            self.model.eval()
            self.model.mode = 'origval'

            _, valid_results = self.val_orig(self.valset.X_val, self.valset.y_val)
            print('origval:validation accuracy : {:.4f} '.format(valid_results[0]))



            # Start validation for this epoch, set model to eval mode
            self.model.eval()
            self.model.mode = 'preval'

            # Set averager classes to record validation losses and accuracies
            val_loss_averager = Averager()
            val_acc_averager = Averager()

            # Generate the labels for test
            label = torch.arange(self.args.way).repeat(self.args.val_query)
            if torch.cuda.is_available():
                label = label.type(torch.cuda.LongTensor)
            else:
                label = label.type(torch.LongTensor)
            label_shot = torch.arange(self.args.way).repeat(self.args.val_shot)
            if torch.cuda.is_available():
                label_shot = label_shot.type(torch.cuda.LongTensor)
            else:
                label_shot = label_shot.type(torch.LongTensor)

            # Run meta-validation
            for i, batch in enumerate(self.val_loader, 1):
                if torch.cuda.is_available():
                    data, _ = [_.cuda() for _ in batch]
                else:
                    data = batch[0]
                # data=data.float()
                p = self.args.val_shot * self.args.way
                data_shot, data_query = data[:p], data[p:]
                logits = self.model((data_shot, label_shot, data_query))
                loss = F.cross_entropy(logits, label)
                acc = count_acc(logits, label)
                val_loss_averager.add(loss.item())
                val_acc_averager.add(acc)

            # Update validation averagers
            val_loss_averager = val_loss_averager.item()
            val_acc_averager = val_acc_averager.item()
            # Write the tensorboardX records
            writer.add_scalar('data/val_loss', float(val_loss_averager), epoch)
            writer.add_scalar('data/val_acc', float(val_acc_averager), epoch)
            print('epoch:{} , val_loss_averager: {:.4f}, val_acc_averager: {:.4f}'.format(epoch, val_loss_averager,
                                                                                      val_acc_averager))
            # Update best saved model
            if val_acc_averager > trlog['max_acc']:
                trlog['max_acc'] = val_acc_averager
                trlog['max_acc_epoch'] = epoch
                self.save_model('max_acc')
            # Save model every 10 epochs
            if epoch % 10 == 0:
                self.save_model('epoch' + str(epoch))

            # Update the logs
            trlog['train_loss'].append(train_loss_averager)
            trlog['train_acc'].append(train_acc_averager)
            trlog['val_loss'].append(val_loss_averager)
            trlog['val_acc'].append(val_acc_averager)

            # Save log
            torch.save(trlog, osp.join(self.args.save_path, 'trlog'))

            if epoch % 10 == 0:
                print('Running Time: {}, Estimated Time: {}'.format(timer.measure(),
                                                                    timer.measure(epoch / self.args.max_epoch)))
        writer.close()

    def val_orig(self, X_val, y_val):
        predicted_loss = []
        inputs = torch.from_numpy(X_val)
        labels = torch.FloatTensor(y_val * 1.0)
        inputs, labels = Variable(inputs), Variable(labels)

        results = []
        predicted = []

        self.model.eval()
        self.model.mode = 'origval'

        if torch.cuda.is_available():
            inputs = inputs.type(torch.cuda.FloatTensor)
        else:
            inputs = inputs.type(torch.FloatTensor)

        predicted = self.model(inputs)
        predicted = predicted.data.cpu().numpy()

        Y = labels.data.numpy()
        predicted = np.argmax(predicted, axis=1)
        for param in ["acc", "auc", "recall", "precision", "fmeasure"]:
            if param == 'acc':
                results.append(accuracy_score(Y, np.round(predicted)))
            if param == "recall":
                results.append(recall_score(Y, np.round(predicted), average='micro'))
            if param == "fmeasure":
                precision = precision_score(Y, np.round(predicted), average='micro')
                recall = recall_score(Y, np.round(predicted), average='micro')
                results.append(2 * precision * recall / (precision + recall))

        return predicted, results
