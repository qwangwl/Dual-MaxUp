##+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
## Modified from: https://github.com/yaoyao-liu/meta-transfer-learning
##
## This source code is licensed under the MIT-style license found in the
## LICENSE file in the root directory of this source tree
##+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
""" Generate commands for pre-train phase. """
import os


def run_exp(model_type = 'MLP',test_subject_id=1):
    train_shot = 10
    train_query = 20
    way = 2

    val_shot = 5
    log_base_dir = './logs_2025.6.21_maxup/'

    the_command = 'python main.py' \
                  + ' --model_type=' + str(model_type) \
                  + ' --train_shot=' + str(train_shot) \
                  + ' --train_query=' + str(train_query) \
                  + ' --way=' + str(way) \
                  + ' --val_shot=' + str(val_shot) \
                  + ' --test_subject_id=' + str(test_subject_id) \
                  + ' --log_base_dir=' + str(log_base_dir) \
                  + ' --phase=pre_train'

    os.system(the_command)

#models = ['MLP','CNN','Transformer']
models = ['Dual-Attention']
for model in models:
    print("current model: " + model)
    for i in range(1,33):
        run_exp(model_type = model, test_subject_id=i)