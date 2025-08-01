# +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
## Modified from: https://github.com/yaoyao-liu/meta-transfer-learning
##
## This source code is licensed under the MIT-style license found in the
## LICENSE file in the root directory of this source tree
# +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
""" Main function for this repo. """
import argparse
import torch
from utils.misc import pprint
from utils.gpu_tools import set_gpu
from trainer.meta import MetaTrainer
from trainer.pre import PreTrainer
import os
os.environ["LOKY_MAX_CPU_COUNT"] = "10"

from trainer.meta_aug_2 import MetaTrainer_aug

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    # Basic parameters
    parser.add_argument('--model_type', type=str, default='CNN',
                        choices=['MLP','CNN','Transformer','Dual-Attention'])  # The network architecture
    parser.add_argument('--dataset', type=str, default='DEAP') # Dataset
    parser.add_argument('--num_subjects', type=int, default=32) 
    parser.add_argument('--phase', type=str, default='meta_train',
                        choices=['pre_train', 'meta_train', 'meta_eval', 'meta_maxup', 'meta_maxup_eval'])  # Phase
    # Manual seed for PyTorch, "0" means using random seed
    parser.add_argument('--seed', type=int, default=3407)
    parser.add_argument('--gpu', default='0')  # GPU id
    parser.add_argument('--dataset_dir', type=str,
                        default='./data/')  # Dataset folder
    parser.add_argument('--label_mode', type=str, default='val')
    parser.add_argument('--test_subject_id', type=int, default=1)
    parser.add_argument('--partition_val_test', type=int, default=2)
    parser.add_argument('--dropout', type=float, default=0.5)
    parser.add_argument('--log_base_dir', type=str, default='./logs/')

    # Parameters for pretain phase
    # Epoch number for pre-train phase
    parser.add_argument('--pre_max_epoch', type=int, default=10)
    # Batch size for pre-train phase
    parser.add_argument('--pre_batch_size', type=int, default=100)
    # embedding size
    parser.add_argument('--embed_size', type=int, default=200)
    # Learning rate for pre-train phase
    parser.add_argument('--pre_lr', type=float, default=0.05)
    # Gamma for the pre-train learning rate decay
    parser.add_argument('--pre_gamma', type=float, default=0.5)
    # The number of epochs to reduce the pre-train learning rate
    parser.add_argument('--pre_step_size', type=int, default=25)
    # Momentum for the optimizer during pre-train
    parser.add_argument('--pre_custom_momentum', type=float, default=0.9)
    # Weight decay for the optimizer during pre-train
    parser.add_argument('--pre_custom_weight_decay',
                        type=float, default=0.0005)

    # Parameters for meta-train phase
    # Epoch number for meta-train phase
    parser.add_argument('--max_epoch', type=int, default=100)
    # The number for different tasks used for meta-train
    parser.add_argument('--num_batch', type=int, default=30)
    parser.add_argument('--test_num_batch', type=int, default=20)
    # Shot number, how many samples for one class in a task
    parser.add_argument('--train_shot', type=int, default=10)
    # Way number, how many classes in a task
    parser.add_argument('--way', type=int, default=3)
    # The number of training samples for each class in a task
    parser.add_argument('--train_query', type=int, default=10)
    # The number of test samples for each class in a task
    parser.add_argument('--val_shot', type=int, default=5)
    parser.add_argument('--val_query', type=int, default=15)
    # Learning rate for SS weights
    parser.add_argument('--meta_lr1', type=float, default=0.0001)
    # Learning rate for FC weights
    parser.add_argument('--meta_lr2', type=float, default=0.005)
    # Learning rate for the inner loop
    parser.add_argument('--base_lr', type=float, default=0.005)
    # The number of updates for the inner loop
    parser.add_argument('--update_step', type=int, default=20)
    # The number of epochs to reduce the meta learning rates
    parser.add_argument('--step_size', type=int, default=3)
    # Gamma for the meta-train learning rate decay
    parser.add_argument('--gamma', type=float, default=0.8)
    # The pre-trained weights for meta-train phase
    parser.add_argument('--init_weights', type=str, default=None)
    # The meta-trained weights for meta-eval phase
    parser.add_argument('--eval_weights', type=str, default=None)
    # Additional label for meta-train
    parser.add_argument('--meta_label', type=str, default='exp1')

    # Meta_task sampling methods
    # AS_Sampling from all samples
    # DS_Sampling from different subjects
    # DG_Sampling from different groups
    parser.add_argument('--meta_task_sampler', type=str, default='AS',
                        choices=['AS','DS','DG','DT'])

    #Kmeans参数
    parser.add_argument('--k', type=int, default=25)


    #增强方法
    #下面三个是针对meta_adapt阶段的
    parser.add_argument('--support_aug', '-saug', default=None, type=str,
                        help='If use support level data augmentation.')
    parser.add_argument('--support_aug2', '-saug2', default=None, type=str,
                        help='If use combine support level data augmentation.')
    parser.add_argument('--shot_aug', '-shotaug', default=[], nargs='+', type=str,
                        help='If use shot level data augmentation.')
    parser.add_argument('--query_aug', '-qaug', default=None, type=str,
                        help='If use query level data augmentation.')
    
    # 这个是针对meta_test阶段的
    parser.add_argument('--test_support_aug', '-tsaug', default=None, type=str,
                        help='If use test support level data augmentation.')
    
    #关于MaxUp的参数设置
    #增强池的类型
    parser.add_argument('--aug_pool', default=None, type=str,
                         choices=['only', 'single', 'medium', 'large'])
    #增强次数m
    parser.add_argument('--m', type=int, default=1)

    #引入support_maxup的参数设置
    #meta_train阶段是否support_maxup
    parser.add_argument('--model_train_mode', default='meta_train', type=str,
                         choices=['meta_train', 'meta_train_aug'])
    #meta_test阶段是否support_maxup
    parser.add_argument('--model_test_mode', default='meta_test', type=str,
                         choices=['meta_test', 'meta_test_aug'])


    # Set the parameters
    args = parser.parse_args()
    # pprint(vars(args))

    # Set the GPU id
    set_gpu(args.gpu)

    # Set manual seed for PyTorch
    if args.seed == 0:
        print('Using random seed.')
        torch.backends.cudnn.benchmark = True
    else:
        print('Using manual seed:', args.seed)
        torch.manual_seed(args.seed)
        torch.cuda.manual_seed(args.seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

    #print('test subject id:', args.test_subject_id)
    #print('test_num_batch:', args.test_num_batch)
    #print('current phase:', args.phase)

    # Start trainer for pre-train, meta-train or meta-eval
    if args.phase == 'pre_train':
        trainer = PreTrainer(args)
        trainer.train()
    elif args.phase == 'meta_train':
        trainer = MetaTrainer(args)
        trainer.train()  
    elif args.phase == 'meta_eval':
        trainer = MetaTrainer(args)
        trainer.eval()   
    elif args.phase == 'meta_maxup':
        trainer = MetaTrainer_aug(args)
        trainer.train()  
    elif args.phase == 'meta_maxup_eval':
        trainer = MetaTrainer_aug(args)
        trainer.eval()
    else:
        raise ValueError('Please set correct phase.')