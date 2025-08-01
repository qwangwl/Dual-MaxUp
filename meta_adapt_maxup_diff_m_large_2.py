##+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
## Modified from: https://github.com/yaoyao-liu/meta-transfer-learning

## This source code is licensed under the MIT-style license found in the
## LICENSE file in the root directory of this source tree
##+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
""" Generate commands for meta-adaptation phase. """
import os


def run_exp(test_subject_id = 1, val_shot = 1, aug_pool = 'only',
             m = 12, model_train_mode = 'meta_train'):
    
    model_type = 'Dual-Attention'
    meta_task_sampler='DG'

    train_shot = 10
    train_query = 20
    way = 2

    #val_shot = 5
    test_num_batch = 50
    log_base_dir = './logs_2025.07.31_maxup/'

    the_command = 'python main.py' \
                  + ' --model_type=' + str(model_type) \
                  + ' --train_shot=' + str(train_shot) \
                  + ' --train_query=' + str(train_query) \
                  + ' --way=' + str(way) \
                  + ' --val_shot=' + str(val_shot) \
                  + ' --test_num_batch=' + str(test_num_batch) \
                  + ' --test_subject_id=' + str(test_subject_id) \
                  + ' --meta_task_sampler=' + str(meta_task_sampler) \
                  + ' --log_base_dir=' + str(log_base_dir) \
                  + ' --m=' + str(m) \
                  + ' --aug_pool=' + str(aug_pool) \
                  + ' --model_train_mode=' + str(model_train_mode) 
                  
    os.system(the_command + ' --phase=meta_maxup')

# val_shot_set = [5]
# aug_pool_choices = ['only', 'single', 'medium', 'large']
# # aug_pool_choices = ['large']
# #关于MaxUp的参数设置
# #m_choices = [16, 20, 24]
# m_choices = [20]

# model_train_mode_choices = ['meta_train_aug']

# for aug_pool in aug_pool_choices:
#     print('aug_pool:' + str(aug_pool))
#     for model_train_mode in model_train_mode_choices:
#         print('model_train_mode:' + str(model_train_mode))
#         for m in m_choices:
#             print('m:' + str(m))
#             for i in range(20,33):
#                 print('test subject id:' + str(i))
#                 for val_shot in val_shot_set:
#                     run_exp(test_subject_id=i, val_shot=val_shot, aug_pool=aug_pool, m=m, 
#                             model_train_mode=model_train_mode)

val_shot_set = [5]
# aug_pool_choices = ["only", 'single', 'medium', 'large']
aug_pool_choices = ['large']
#关于MaxUp的参数设置
m_choices = [20]
# m_choices = [1, 2, 4, 8, 12, 16, 20, 24, 28, 32, 36, 40]

model_train_mode_choices = ['meta_train_aug', "meta_train"]

for aug_pool in aug_pool_choices:
    print('aug_pool:' + str(aug_pool))
    for model_train_mode in model_train_mode_choices:
        print('model_train_mode:' + str(model_train_mode))
        for m in m_choices:
            print('m:' + str(m))
            for i in range(1, 17):
                print('test subject id:' + str(i))
                for val_shot in val_shot_set:
                    run_exp(test_subject_id=i, val_shot=val_shot, aug_pool=aug_pool, m=m, 
                            model_train_mode=model_train_mode)