# -*- coding: utf-8 -*-
"""
=======================================================================
 Copyright (c) 2019 PolyU CBS LLT Group. All Rights Reserved

@Author      :  Rong XIANG
@Contect     :  xiangrong0302@gmail.com
@Time        :  2020/07/01
@Description :
=======================================================================
"""

import time
import os
import logging
import numpy as np
import main as Affective_main

logger = logging.getLogger()

# os.environ["CUDA_VISIBLE_DEVICES"] = "0,1"

def load_custom_setting(args, para_custom):
    print('---------- load custom settings ----------')

    if para_custom.MODEL_NAME == r'bert':
        args.model_type = r'bert'
        # '/uncased_L-12_H-768_A-12/' '/scibert_scivocab_uncased/'
        temp = '/uncased_L-12_H-768_A-12/'
        args.model_name_or_path = r'./model/BERT/torch' + temp
        args.config_name = r'./model/BERT/torch' + temp + 'config.json'  # 'bert_config.json', 'config.json'
        args.tokenizer_name = r'./model/BERT/torch' + temp + 'vocab.txt'

    elif para_custom.MODEL_NAME == r'xlnet':
        args.model_type = r'xlnet'
        args.model_name_or_path = r'xlnet-base-cased'
        args.config_name = r''
        args.tokenizer_name = r''

    elif para_custom.MODEL_NAME == 'xlm':
        args.model_type = r'xlm'
        args.model_name_or_path = r'xlm-mlm-tlm-xnli15-1024'
        args.config_name = r''
        args.tokenizer_name = r''
        args.per_gpu_train_batch_size = 16
        args.per_gpu_eval_batch_size = 16

    elif para_custom.MODEL_NAME == r'roberta':
        args.model_type = r'roberta'
        args.model_name_or_path = r'roberta-base'
        args.config_name = r''
        args.tokenizer_name = r''

    # # Required parameters
    args.data_dir = para_custom.DATA_DIR
    args.task_name = para_custom.DATASET
    args.output_dir = r'./tmp/' + para_custom.MODEL_NAME + '_' + para_custom.DATASET

    # optional parameters
    args.per_gpu_train_batch_size = para_custom.BATCH_SIZE
    args.per_gpu_eval_batch_size = para_custom.BATCH_SIZE

    args.cache_dir = r''
    args.max_seq_length = para_custom.SEQUENCE_LEN
    args.do_train = para_custom.DO_TRAIN
    args.do_eval = para_custom.DO_EVAL
    args.do_test = para_custom.DO_TEST
    args.evaluate_during_training = False
    args.do_lower_case = True

    args.gradient_accumulation_steps = 1
    args.learning_rate = 5e-5
    args.weight_decay = 0.0
    args.adam_epsilon = 0.000001
    args.max_grad_norm = 1.0
    args.num_train_epochs = para_custom.EPOCH
    args.max_steps = -1
    args.warmup_steps = para_custom.WARMUP_STEPS
    args.warmup_proportion = para_custom.WARMUP_PROPORTION

    args.logging_steps = para_custom.LOGGING_STEP
    args.save_steps = para_custom.EVALUATE_STEP
    args.eval_all_checkpoints = True
    args.no_cuda = False
    args.overwrite_output_dir = True
    args.overwrite_cache = para_custom.REBUILD_FEATURE
    args.seed = 42

    args.fp16 = False
    args.fp16_opt_level = 'O1'
    args.local_rank = -1
    args.server_ip = ''
    args.server_port = ''

    # additional parameters

    # general
    args.FORCE_CPU = para_custom.FORCE_CPU
    args.local_rank = para_custom.LOCAL_RANK

    # train para
    args.RESUME_STEP = para_custom.RESUME_STEP
    args.DEBUG_TRAIN_INFO = para_custom.DEBUG_TRAIN_INFO
    args.DEBUG_TENSORBOARD_WRITER = para_custom.DEBUG_TENSORBOARD_WRITER

    # eval para
    args.DEBUG_EVAL_DUMP = para_custom.DEBUG_EVAL_DUMP

    # test para
    args.TEST_STEP = para_custom.TEST_STEP

    return args


class Args_Custom:

    def __init__(self):
        # mandatory
        self.MODEL_NAME = r'bert'  # bert    xlnet    xlm    roberta
        self.DATASET = 'None'

        self.DATA_DIR = './data/datasets'
        self.DO_TRAIN = True
        self.DO_EVAL = True
        self.DO_TEST = False

        # optional
        self.REBUILD_FEATURE = True

        self.BATCH_SIZE = 32
        self.WARMUP_STEPS = -1  # overwrited by WARMUP_PROPORTION
        self.WARMUP_PROPORTION = 0.01

        self.EPOCH = 1
        self.SEQUENCE_LEN = 32
        self.LOGGING_STEP = 100
        self.EVALUATE_STEP = 100
        self.RESUME_STEP = -1  # -1 for invalid
        self.TEST_STEP = -1  # -1 for invalid

        # debug mode, using CPU to trace torch code
        self.FORCE_CPU = False
        self.DEBUG_TRAIN_INFO = False
        self.DEBUG_EVAL_DUMP = False
        self.DEBUG_TENSORBOARD_WRITER = True

        # only for log
        self.DATASET_NAME = 'SSST2'

        # os.environ['MASTER_ADDR'] = 'localhost'
        # os.environ['MASTER_PORT'] = '12355'
        self.LOCAL_RANK = -1


MODEL_NAME = r'bert'  # bert    xlnet    xlm    roberta
# CUR_DATAS = ['SSST2', 'AirRecord', 'MR', 'spr2018', 'Twitter']
CUR_DATAS = [ 'SSST2' ]

AFFECTIVE_RES = 'EPA.pkl'  # EPA.pkl    NRC-VAD.pkl


if __name__ == "__main__":
    global CUR_DATA

    # load default settings
    args = Affective_main.load_default_settings()

    # overwrite cmd settings

    # other settings
    args.AFFECTIVE_RES = './data/AffectiveLexicon/EPA.pkl'  # EPA.pkl    NRC-VAD.pkl

    # for thres in np.arange( 0.1, 1.0, 0.1 ):
    for item in CUR_DATAS:
        CUR_DATA = item
        print(CUR_DATA)

        # customize parameters
        args_custom = Args_Custom()
        args_custom.DATASET = item
        args = load_custom_setting(args, args_custom)

        args.task_name = item
        args.data_dir = args.data_dir + '/' + item

        LOG_FP = './log/' + time.strftime('%Y-%m-%d_%H-%M-%S', time.localtime(time.time())) + '_' + \
                 args_custom.MODEL_NAME + '_' + \
                 args_custom.DATASET + '_' + \
                 '.txt'
        Affective_main.logger = Affective_main.set_logger(logger, LOG_FP, True)

        # experimentally weights
        if args.AFFECTIVE_RES == 'EPA.pkl':
            rng = np.arange(0.5, 2.0, 0.2)
        else:
            rng = np.arange(0.5, 4.0, 0.5)

        for al in rng:

            for name, value in vars(args_custom).items():
                Affective_main.logger.info('$$$$$ custom para {}: {}'.format(name, value))

            for name, value in vars(args).items():
                Affective_main.logger.info('$$$$$ final para {}: {}'.format(name, value))

            Affective_main.main(args)
