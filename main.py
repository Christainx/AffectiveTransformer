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
import argparse
import glob
import pprint
import logging
import os
import random
import math
import numpy as np
import pathlib
import pickle
import torch
import json
import operator
from scipy.sparse import vstack, hstack, coo_matrix
from tqdm import tqdm, trange
from torch.utils.data import (Dataset, DataLoader, RandomSampler, SequentialSampler, TensorDataset)
from torch.utils.data.distributed import DistributedSampler
from tensorboardX import SummaryWriter
from tqdm import tqdm, trange

from pytorch_transformers import (WEIGHTS_NAME, BertConfig,
                                  BertForSequenceClassification, BertTokenizer,
                                  RobertaConfig,
                                  RobertaForSequenceClassification,
                                  RobertaTokenizer,
                                  XLMConfig, XLMForSequenceClassification,
                                  XLMTokenizer, XLNetConfig,
                                  XLNetForSequenceClassification,
                                  XLNetTokenizer)

from pytorch_transformers import AdamW, WarmupLinearSchedule

from examples.utils_glue import (compute_metrics, convert_examples_to_features,
                        output_modes, processors)

ALL_MODELS = sum((tuple(conf.pretrained_config_archive_map.keys()) for conf in (BertConfig, XLNetConfig,
                                                                                RobertaConfig)), ())

MODEL_CLASSES = {
    ## bert ernie bert_wwm bert_wwwm_ext
    'bert': (BertConfig, BertForSequenceClassification, BertTokenizer),
    'xlnet': (XLNetConfig, XLNetForSequenceClassification, XLNetTokenizer),
    'roberta': (BertConfig, BertForSequenceClassification, BertTokenizer),
}

def set_seed(args):
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if args.n_gpu > 0:
        torch.cuda.manual_seed_all(args.seed)

def seed_everything(seed=1029):
    """
    设置整个开发环境的seed
    :param seed:
    :param device:
    :return:
    """
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    # some cudnn methods can be random even after fixing the seed
    # unless you tell it to be deterministic
    torch.backends.cudnn.deterministic = True

class EPAInfo:
    WEIGHT_E = 0.5
    WEIGHT_P = 0.3
    WEIGHT_A = 0.2
    ALPHA = 0.7

def load_and_cache_examples(args, task, tokenizer, data_type='train'):

    fpout = open( args.AFFECTIVE_RES, 'rb' )
    [wordlist, dict_e, dict_p, dict_a] = pickle.load( fpout )
    epainfos = EPAInfo()
    epainfos.wordlist = wordlist
    epainfos.dict_e = dict_e
    epainfos.dict_p = dict_p
    epainfos.dict_a = dict_a

    # if args.local_rank not in [-1, 0] and not evaluate:
    if args.local_rank not in [-1, 0]:
        torch.distributed.barrier()  # Make sure only the first process in distributed training process the dataset,
        # and the others will use the cache

    processor = processors[task]()
    output_mode = output_modes[task]
    # Load data features from cache or dataset file
    cached_features_file = os.path.join(args.data_dir,
                                        'cached_{}_{}_{}_{}'.format(
                                            data_type,
                                            list(filter(None, args.model_name_or_path.split('/'))).pop(),
                                            str(args.max_seq_length),
                                            str(task)))
    if os.path.exists(cached_features_file) and not args.overwrite_cache:
        logger.info("Loading features from cached file %s", cached_features_file)
        features = torch.load(cached_features_file)
    else:
        logger.info("Creating features from dataset file at %s", args.data_dir)
        label_list = processor.get_labels()
        # if task in ['mnli', 'mnli-mm'] and args.model_type in ['roberta']:
        #     label_list[1], label_list[2] = label_list[2], label_list[1]

        if data_type == 'train':
            examples = processor.get_train_examples(args.data_dir)
        elif data_type == 'dev':
            examples = processor.get_dev_examples(args.data_dir)
        else:
            examples = processor.get_test_examples(args.data_dir)

        features = convert_examples_to_features(examples, label_list, args.max_seq_length, tokenizer, output_mode,
                                                cls_token_at_end=bool(args.model_type in ['xlnet']),
                                                # xlnet has a cls token at the end
                                                cls_token=tokenizer.cls_token,
                                                cls_token_segment_id=2 if args.model_type in ['xlnet'] else 0,
                                                sep_token=tokenizer.sep_token,
                                                sep_token_extra=bool(args.model_type in ['roberta']),
                                                # roberta uses an extra separator b/w pairs of sentences, cf. github.com/pytorch/fairseq/commit/1684e166e3da03f5b600dbb7855cb98ddfcd0805
                                                pad_on_left=bool(args.model_type in ['xlnet']),
                                                # pad on the left for xlnet
                                                pad_token=tokenizer.convert_tokens_to_ids([tokenizer.pad_token])[0],
                                                pad_token_segment_id=4 if args.model_type in ['xlnet'] else 0,
                                                epa_infos=epainfos,
                                                )
        if args.local_rank in [-1, 0]:
            logger.info("Saving features into cached file %s", cached_features_file)
            torch.save(features, cached_features_file)

    # if args.local_rank == 0 and not evaluate:
    if args.local_rank == 0:
        torch.distributed.barrier()  # Make sure only the first process in distributed training process the dataset,
        # and the others will use the cache

    # Convert to Tensors and build dataset
    all_input_ids = torch.tensor([f.input_ids for f in features], dtype=torch.long)
    all_input_mask = torch.tensor([f.input_mask for f in features], dtype=torch.long)
    all_segment_ids = torch.tensor([f.segment_ids for f in features], dtype=torch.long)
    if output_mode == "classification":
        all_label_ids = torch.tensor([f.label_id for f in features], dtype=torch.long)
    elif output_mode == "regression":
        all_label_ids = torch.tensor([f.label_id for f in features], dtype=torch.float)
    epa_vec = torch.tensor( [f.epa_vec for f in features], dtype = torch.float )

    dataset = TensorDataset( all_input_ids, all_input_mask, all_segment_ids, all_label_ids, epa_vec )
    return dataset


def train(args, train_dataset, model, tokenizer):
    """ Train the model """
    if args.local_rank in [-1, 0] and args.DEBUG_TENSORBOARD_WRITER:
        tb_writer = SummaryWriter()

    args.train_batch_size = args.per_gpu_train_batch_size * max(1, args.n_gpu)
    train_sampler = RandomSampler(train_dataset) if args.local_rank == -1 else DistributedSampler(train_dataset)
    train_dataloader = DataLoader(train_dataset, sampler=train_sampler, batch_size=args.train_batch_size)

    if args.max_steps > 0:
        t_total = args.max_steps
        args.num_train_epochs = args.max_steps // (len(train_dataloader) // args.gradient_accumulation_steps) + 1
    else:
        t_total = len(train_dataloader) // args.gradient_accumulation_steps * args.num_train_epochs
    args.warmup_steps = int(t_total * args.warmup_proportion)
    logger.info('warmup steps: ' + str(args.warmup_steps))

    # Prepare optimizer and schedule (linear warmup and decay)
    no_decay = ['bias', 'LayerNorm.weight']
    optimizer_grouped_parameters = [
        {'params': [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)],
         'weight_decay': args.weight_decay},
        {'params': [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}
    ]
    optimizer = AdamW(optimizer_grouped_parameters, lr=args.learning_rate, eps=args.adam_epsilon)
    scheduler = WarmupLinearSchedule(optimizer, warmup_steps=args.warmup_steps, t_total=t_total)
    # logger.info('learning rate: ' + str(optimizer.param_groups[0]["lr"]))

    if args.fp16:
        try:
            from apex import amp
        except ImportError:
            raise ImportError("Please install apex from https://www.github.com/nvidia/apex to use fp16 training.")
        model, optimizer = amp.initialize(model, optimizer, opt_level=args.fp16_opt_level)

    # multi-gpu training (should be after apex fp16 initialization)
    if args.n_gpu > 1:
        model = torch.nn.DataParallel(model)
    # Distributed training (should be after apex fp16 initialization)
    if args.local_rank != -1:
        model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.local_rank],
                                                          output_device=args.local_rank,
                                                          find_unused_parameters=True)
    # Train!
    print('')
    logger.info("***** Running training *****")
    logger.info("===> Training Examples Num = %d", len(train_dataset[0]))
    logger.info("===> Training Epochs Num = %d", args.num_train_epochs)
    logger.info("===> Instantaneous batch size per GPU = %d", args.per_gpu_train_batch_size)
    logger.info("===> Total train batch size (w. parallel, distributed & accumulation) = %d",
                args.train_batch_size * args.gradient_accumulation_steps * (
                    torch.distributed.get_world_size() if args.local_rank != -1 else 1))
    logger.info("===> Gradient Accumulation steps = %d", args.gradient_accumulation_steps)
    logger.info("===> Total optimization steps = %d", t_total)

    global_step = 0
    tr_loss, logging_loss = 0.0, 0.0
    model.zero_grad()
    train_iterator = trange(int(args.num_train_epochs), desc="Epoch", disable=args.local_rank not in [-1, 0])
    set_seed(args)  # Added here for reproductibility (even between python 2 and 3)
    for _ in train_iterator:
        epoch_iterator = tqdm(train_dataloader, desc="Iteration", disable=args.local_rank not in [-1, 0])
        for step, batch in enumerate(epoch_iterator):
            model.train()
            batch = tuple(t.to(args.device) for t in batch)
            inputs = {
                'input_ids': batch[0],
                'attention_mask': batch[1],
                'token_type_ids': batch[2] if args.model_type in ['bert', 'xlnet'] else None,
                # XLM and RoBERTa don't use segment_ids
                'labels': batch[3],
                'epa_matrix': batch[4],
            }
            outputs = model(**inputs)
            loss = outputs[0]  # model outputs are always tuple in pytorch-transformers (see doc)

            if args.n_gpu > 1:
                loss = loss.mean()  # mean() to average on multi-gpu parallel training
            if args.gradient_accumulation_steps > 1:
                loss = loss / args.gradient_accumulation_steps

            if args.fp16:
                with amp.scale_loss(loss, optimizer) as scaled_loss:
                    scaled_loss.backward()
                torch.nn.utils.clip_grad_norm_(amp.master_params(optimizer), args.max_grad_norm)
            else:
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)

            tr_loss += loss.item()
            if (step + 1) % args.gradient_accumulation_steps == 0:
                scheduler.step()  # Update learning rate schedule
                optimizer.step()
                model.zero_grad()
                global_step += 1

                if args.local_rank in [-1, 0] and args.logging_steps > 0 and global_step % args.logging_steps == 0:
                    # Log metrics
                    if args.local_rank == -1 and args.evaluate_during_training:  # Only evaluate when single GPU otherwise metrics may not average well
                        results = evaluate(args, model, tokenizer)
                        for key, value in results.items():
                            tb_writer.add_scalar('eval_{}'.format(key), value, global_step)
                    tb_writer.add_scalar('lr', scheduler.get_lr()[0], global_step)
                    tb_writer.add_scalar('loss', (tr_loss - logging_loss) / args.logging_steps, global_step)
                    logging_loss = tr_loss

                if args.local_rank in [-1, 0] and args.save_steps > 0 and global_step % args.save_steps == 0:
                    # Save model checkpoint
                    output_dir = os.path.join(args.output_dir, 'checkpoint-{}'.format(global_step))
                    if not os.path.exists(output_dir):
                        os.makedirs(output_dir)
                    model_to_save = model.module if hasattr(model,
                                                            'module') else model  # Take care of distributed/parallel training
                    model_to_save.save_pretrained(output_dir)
                    torch.save(args, os.path.join(output_dir, 'training_args.bin'))
                    logger.info("Saving model checkpoint to %s", output_dir)

            if args.max_steps > 0 and global_step > args.max_steps:
                epoch_iterator.close()
                break
        if args.max_steps > 0 and global_step > args.max_steps:
            train_iterator.close()
            break

    if args.local_rank in [-1, 0]:
        tb_writer.close()

    return global_step, tr_loss / global_step


def evaluate(global_step, args, model, tokenizer, prefix=""):
    # Loop to handle MNLI double evaluation (matched, mis-matched)
    eval_task_names = ("mnli", "mnli-mm") if args.task_name == "mnli" else (args.task_name,)
    eval_outputs_dirs = (args.output_dir, args.output_dir + '-MM') if args.task_name == "mnli" else (args.output_dir,)

    results = {}
    for eval_task, eval_output_dir in zip(eval_task_names, eval_outputs_dirs):
        eval_dataset = load_and_cache_examples(args, eval_task, tokenizer, data_type='dev')
        if not os.path.exists(eval_output_dir) and args.local_rank in [-1, 0]:
            os.makedirs(eval_output_dir)

        args.eval_batch_size = args.per_gpu_eval_batch_size * max(1, args.n_gpu)
        # Note that DistributedSampler samples randomly
        eval_sampler = SequentialSampler(eval_dataset) if args.local_rank == -1 else DistributedSampler(eval_dataset)
        eval_dataloader = DataLoader(eval_dataset, sampler=eval_sampler, batch_size=args.eval_batch_size)

        # Eval!
        logger.info("***** Running evaluation {} *****".format(prefix))
        logger.info("  Num examples = %d", len(eval_dataset))
        logger.info("  Batch size = %d", args.eval_batch_size)
        eval_loss = 0.0
        nb_eval_steps = 0
        preds = None
        out_label_ids = None
        for batch in tqdm(eval_dataloader, desc="Evaluating"):
            model.eval()
            batch = tuple(t.to(args.device) for t in batch)

            with torch.no_grad():
                inputs = {
                    'input_ids': batch[0],
                    'attention_mask': batch[1],
                    'token_type_ids': batch[2] if args.model_type in ['bert', 'xlnet'] else None,
                    # XLM and RoBERTa don't use segment_ids
                    'labels': batch[3],
                    'epa_matrix': batch[4],
                }
                outputs = model(**inputs)
                tmp_eval_loss, logits = outputs[:2]

                eval_loss += tmp_eval_loss.mean().item()
            nb_eval_steps += 1
            if preds is None:
                preds = logits.detach().cpu().numpy()
                out_label_ids = inputs['labels'].detach().cpu().numpy()
            else:
                preds = np.append(preds, logits.detach().cpu().numpy(), axis=0)
                out_label_ids = np.append(out_label_ids, inputs['labels'].detach().cpu().numpy(), axis=0)

        eval_loss = eval_loss / nb_eval_steps
        if args.output_mode == "classification":
            preds = np.argmax(preds, axis=1)
        elif args.output_mode == "regression":
            preds = np.squeeze(preds)
        result = compute_metrics(eval_task, preds, out_label_ids)
        #         print('preds:')
        #         for item in preds:
        #             fp_pred.writelines( str( item ) + '\n' )
        #         print('true:')
        #         for item in out_label_ids:
        #             fp_true.writelines( str( item ) + '\n' )
        results.update(result)

        output_eval_file = os.path.join(eval_output_dir, "eval_results.txt")
        with open(output_eval_file, "w") as writer:
            logger.info("***** Eval results {} *****".format(prefix))
            for key in sorted(result.keys()):
                logger.info("  %s = %s", key, str(result[key]))
                writer.write("%s = %s\n" % (key, str(result[key])))

    return results


def predict(args, model, tokenizer, prefix=""):
    # Loop to handle MNLI double evaluation (matched, mis-matched)
    pred_task_names = ("mnli", "mnli-mm") if args.task_name == "mnli" else (args.task_name,)
    pred_outputs_dirs = (args.output_dir, args.output_dir + '-MM') if args.task_name == "mnli" else (args.output_dir,)

    results = {}
    for pred_task, pred_output_dir in zip(pred_task_names, pred_outputs_dirs):
        pred_dataset = load_and_cache_examples(args, pred_task, tokenizer, data_type='test')
        if not os.path.exists(pred_output_dir) and args.local_rank in [-1, 0]:
            os.makedirs(pred_output_dir)

        args.pred_batch_size = args.per_gpu_eval_batch_size * max(1, args.n_gpu)
        # Note that DistributedSampler samples randomly
        pred_sampler = SequentialSampler(pred_dataset) if args.local_rank == -1 else DistributedSampler(pred_dataset)
        eval_dataloader = DataLoader(pred_dataset, sampler=pred_sampler, batch_size=args.eval_batch_size)

        logger.info("***** Running prediction {} *****".format(prefix))
        logger.info("  Num examples = %d", len(pred_dataset))
        logger.info("===> Prediction examples Num = %d", len(pred_dataset[0]))
        logger.info("  Batch size = %d", args.pred_batch_size)
        nb_pred_steps = 0
        preds = None
        gold_output = None

        for batch in tqdm(eval_dataloader, desc="Evaluating"):
            model.eval()
            # batch = tuple(t.to(args.device) for t in batch)
            batch = tuple(t.to(args.device) for t in batch)[:-1]
            with torch.no_grad():
                inputs = {
                    'input_ids': batch[0],
                    'attention_mask': batch[1],
                    'token_type_ids': batch[2] if args.model_type in ['bert', 'xlnet'] else None,
                    # XLM and RoBERTa don't use segment_ids
                    'labels': batch[3],
                    'epa_matrix': batch[4],
                }
                if args.model_type != 'distilbert':
                    inputs['token_type_ids'] = batch[2] if (
                            'bert' in args.model_type or 'xlnet' in args.model_type) else None  # XLM, DistilBERT and
                    # RoBERTa don't use segment_ids
                outputs = model(**inputs)
                tmp_eval_loss, logits = outputs[:2]

                eval_loss += tmp_eval_loss.mean().item()

            nb_pred_steps += 1
            if preds is None:
                preds = logits.detach().cpu().numpy()
                out_label_ids = inputs['labels'].detach().cpu().numpy()
            else:
                preds = np.append(preds, logits.detach().cpu().numpy(), axis=0)
                out_label_ids = np.append(out_label_ids, inputs['labels'].detach().cpu().numpy(), axis=0)


        eval_loss = eval_loss / nb_pred_steps
        if args.output_mode == "classification":
            preds = np.argmax(preds, axis=1)
        elif args.output_mode == "regression":
            preds = np.squeeze(preds)
        result = compute_metrics(pred_task, preds, out_label_ids)
        #         print('preds:')
        #         for item in preds:
        #             fp_pred.writelines( str( item ) + '\n' )
        #         print('true:')
        #         for item in out_label_ids:
        #             fp_true.writelines( str( item ) + '\n' )
        results.update(result)

        output_eval_file = os.path.join(pred_output_dir, "eval_results.txt")
        with open(output_eval_file, "w") as writer:
            logger.info("***** Eval results {} *****".format(prefix))
            for key in sorted(result.keys()):
                logger.info("  %s = %s", key, str(result[key]))
                writer.write("%s = %s\n" % (key, str(result[key])))
    return results


def main(args):
    pprint.pprint(args.__dict__)  # print args parameters

    if os.path.exists(args.output_dir) and os.listdir(
            args.output_dir) and args.do_train and not args.overwrite_output_dir:
        raise ValueError(
            "Output directory ({}) already exists and is not empty. Use --overwrite_output_dir to overcome.".format(
                args.output_dir))

    # Setup distant debugging if needed
    if args.server_ip and args.server_port:
        # Distant debugging - see https://code.visualstudio.com/docs/python/debugging#_attach-to-a-local-script
        import ptvsd
        print("Waiting for debugger attach")
        ptvsd.enable_attach(address=(args.server_ip, args.server_port), redirect_output=True)
        ptvsd.wait_for_attach()

    # Setup CUDA, GPU & distributed training
    if args.FORCE_CPU:
        device = torch.device("cpu")
        args.n_gpu = 1
    elif args.local_rank == -1 or args.no_cuda:
        device = torch.device("cuda" if torch.cuda.is_available() and not args.no_cuda else "cpu")
        args.n_gpu = torch.cuda.device_count()
    else:  # Initializes the distributed backend which will take care of sychronizing nodes/GPUs
        torch.cuda.set_device(args.local_rank)
        device = torch.device("cuda", args.local_rank)
        torch.distributed.init_process_group(backend='nccl')
        args.n_gpu = 1
    args.device = device

    # Setup logging
    logger.warning("Process rank: %s, device: %s, n_gpu: %s, distributed training: %s, 16-bits training: %s",
                   args.local_rank, device, args.n_gpu, bool(args.local_rank != -1), args.fp16)

    # Set seed
    seed_everything(args.seed)
    # Prepare GLUE task
    args.task_name = args.task_name.lower()
    if args.task_name not in processors:
        raise ValueError("Task not found: %s" % (args.task_name))
    processor = processors[args.task_name]()
    args.output_mode = output_modes[args.task_name]
    label_list = processor.get_labels()

    num_labels = len(label_list)

    # Load pretrained model and tokenizer
    if args.local_rank not in [-1, 0]:
        torch.distributed.barrier()  # Make sure only the first process in distributed training will download model &
        # vocab

    args.model_type = args.model_type.lower()
    config_class, model_class, tokenizer_class = MODEL_CLASSES[args.model_type]
    config = config_class.from_pretrained(args.config_name if args.config_name else args.model_name_or_path,
                                          num_labels=num_labels, finetuning_task=args.task_name)
    tokenizer = tokenizer_class.from_pretrained(args.tokenizer_name if args.tokenizer_name else args.model_name_or_path,
                                                do_lower_case=args.do_lower_case)
    model = model_class.from_pretrained(args.model_name_or_path, from_tf=bool('.ckpt' in args.model_name_or_path),
                                        config=config)

    if args.local_rank == 0:
        torch.distributed.barrier()  # Make sure only the first process in distributed training will download model &
        # vocab
    model.to(args.device)
    print('')
    logger.info("Training/evaluation parameters:")
    for name, value in vars(args).items():
        if name != 'LOSS_PARA':
            logger.info('Training/evaluation para {}: {}'.format(name, value))
        else:
            for name, value in args.LOSS_PARA.items():
                if name != 'graph':
                    logger.info('Training/evaluation loss para {}: {}'.format(name, value))

    # Training
    if args.do_train:
        if args.RESUME_STEP != -1:
            global_step = args.RESUME_STEP
            resumefile = args.output_dir + '/checkpoint-' + str(global_step) + '/'
            model = model_class.from_pretrained(resumefile)
            model.to(args.device)

        train_dataset = load_and_cache_examples(args, args.task_name, tokenizer, data_type='train')
        global_step, tr_loss = train(args, train_dataset, model, tokenizer)
        logger.info(" global_step = %s, average loss = %s", global_step, tr_loss)

    # # Saving best-practices: if you use defaults names for the model, you can reload it using from_pretrained()
    # if args.do_train and (args.local_rank == -1 or torch.distributed.get_rank() == 0):
    #     # Create output directory if needed
    #     if not os.path.exists(args.output_dir) and args.local_rank in [-1, 0]:
    #         os.makedirs(args.output_dir)
    #     #
    #     logger.info("Saving model checkpoint to %s", args.output_dir)
    #     # Save a trained model, configuration and tokenizer using `save_pretrained()`.
    #     # They can then be reloaded using `from_pretrained()`
    #     model_to_save = model.module if hasattr(model,
    #                                             'module') else model  # Take care of distributed/parallel training
    #     model_to_save.save_pretrained(args.output_dir)
    #     tokenizer.save_pretrained(args.output_dir)
    #
    #     # Good practice: save your training arguments together with the trained model
    #     torch.save(args, os.path.join(args.output_dir, 'training_args.bin'))
    #
    #     # Load a trained model and vocabulary that you have fine-tuned
    #     model = model_class.from_pretrained(args.output_dir)
    #     tokenizer = tokenizer_class.from_pretrained(args.output_dir, do_lower_case=args.do_lower_case)
    #     model.to(args.device)

    # Evaluation
    results = {}
    if args.do_eval and args.local_rank in [-1, 0]:
        tokenizer = tokenizer_class.from_pretrained(
            args.tokenizer_name if args.tokenizer_name else args.model_name_or_path,
            do_lower_case=args.do_lower_case)
        checkpoints = [args.output_dir]
        if args.eval_all_checkpoints:
            checkpoints = list(
                os.path.dirname(c) for c in sorted(glob.glob(args.output_dir + '/**/' + WEIGHTS_NAME, recursive=True)))
            logging.getLogger("pytorch_transformers.modeling_utils").setLevel(logging.WARN)  # Reduce logging
        logger.info("Evaluate the following checkpoints: %s", checkpoints)
        for checkpoint in checkpoints:
            global_step = checkpoint.split('-')[-1] if len(checkpoints) > 1 else ""
            model = model_class.from_pretrained(checkpoint)
            model.to(args.device)
            result = evaluate(global_step, args, model, tokenizer, prefix=global_step)
            result = dict((k + '_{}'.format(global_step), v) for k, v in result.items())
            results.update(result)

    # Preidction
    results = {}
    if args.do_test and args.local_rank in [-1, 0]:
        if args.TEST_STEP != -1:
            global_step = args.TEST_STEP
            resumefile = args.output_dir + '/checkpoint-' + str(global_step) + '/'
            model = model_class.from_pretrained(resumefile)

            model.to(args.device)
            result = predict(args, model, tokenizer, prefix=global_step)
            result = dict((k + '_{}'.format(global_step), v) for k, v in result.items())
            results.update(result)
        else:
            tokenizer = tokenizer_class.from_pretrained(
                args.tokenizer_name if args.tokenizer_name else args.model_name_or_path,
                do_lower_case=args.do_lower_case)
            checkpoints = [args.output_dir]
            if args.eval_all_checkpoints:
                checkpoints = list(
                    os.path.dirname(c) for c in sorted(glob.glob(args.output_dir + '/**/' + WEIGHTS_NAME, recursive=True)))
                logging.getLogger("pytorch_transformers.modeling_utils").setLevel(logging.WARN)  # Reduce logging
            logger.info("Evaluate the following checkpoints: %s", checkpoints)
            for checkpoint in checkpoints:
                global_step = checkpoint.split('-')[-1] if len(checkpoints) > 1 else ""
                model = model_class.from_pretrained(checkpoint)
                model.to(args.device)
                result = predict(args, model, tokenizer, prefix=global_step)
                result = dict((k + '_{}'.format(global_step), v) for k, v in result.items())
                results.update(result)

    return results


def load_default_settings():
    print('---------- load default settings ----------')

    """
        python main.py --model_type=bert \
                --model_name_or_path=./model/bert/torch/cased_L-12_H-768_A-12 \
                --config_name=./model/bert/torch/cased_L-12_H-768_A-12/bert_config.json \
                --task_name=covid19 \
                --do_train \
                --do_eval \
                --do_lower_case \
                --data_dir=./data/Corpus \
                --max_seq_length=256 \
                --per_gpu_train_batch_size=16 \
                --per_gpu_eval_batch_size=16 \
                --learning_rate=2e-5 \
                --num_train_epochs=2.0 \
                --logging_steps=100 \
                --save_steps=100 \
                --output_dir=./output/covid19/ \
                --overwrite_output_dir
    """
    parser = argparse.ArgumentParser()

    # # Required parameters
    parser.add_argument("--data_dir", default=None, type=str, required=False,
                        help="The input data dir. Should contain the .tsv files (or other data files) for the task.")
    parser.add_argument("--model_type", default=None, type=str, required=False,
                        help="Model type selected in the list: " + ", ".join(MODEL_CLASSES.keys()))
    parser.add_argument("--model_name_or_path", default=None, type=str, required=False,
                        help="Path to pre-trained model or shortcut name selected in the list: " + ", ".join(
                            ALL_MODELS))
    parser.add_argument("--task_name", default=None, type=str, required=False,
                        help="The name of the task to train selected in the list: " + ", ".join(processors.keys()))
    parser.add_argument("--output_dir", default=None, type=str, required=False,
                        help="The output directory where the model predictions and checkpoints will be written.")

    # # Other parameters
    parser.add_argument("--config_name", default="bert_config.json", type=str,
                        help="Pretrained config name or path if not the same as model_name")
    parser.add_argument("--tokenizer_name", default="", type=str,
                        help="Pretrained tokenizer name or path if not the same as model_name")
    parser.add_argument("--cache_dir", default="", type=str,
                        help="Where do you want to store the pre-trained models downloaded from s3")
    parser.add_argument("--mesh_dir", default="", type=str,
                        help="MeSH vocabulory dir")
    parser.add_argument("--max_seq_length", default=128, type=int,
                        help="The maximum total input sequence length after tokenization. Sequences longer "
                             "than this will be truncated, sequences shorter will be padded.")
    parser.add_argument("--do_train", action='store_true',
                        help="Whether to run training.")
    parser.add_argument("--do_eval", action='store_true',
                        help="Whether to run eval on the dev set.")
    parser.add_argument("--do_lower_case", action='store_true',
                        help="Set this flag if you are using an uncased model.")
    parser.add_argument("--per_gpu_train_batch_size", default=8, type=int,
                        help="Batch size per GPU/CPU for training.")
    parser.add_argument("--per_gpu_eval_batch_size", default=8, type=int,
                        help="Batch size per GPU/CPU for evaluation.")
    parser.add_argument('--gradient_accumulation_steps', type=int, default=1,
                        help="Number of updates steps to accumulate before performing a backward/update pass.")
    parser.add_argument("--learning_rate", default=5e-5, type=float,
                        help="The initial learning rate for Adam.")
    parser.add_argument("--weight_decay", default=0.0, type=float,
                        help="Weight deay if we apply some.")
    parser.add_argument("--adam_epsilon", default=1e-8, type=float,
                        help="Epsilon for Adam optimizer.")
    parser.add_argument("--max_grad_norm", default=1.0, type=float,
                        help="Max gradient norm.")
    parser.add_argument("--num_train_epochs", default=3.0, type=float,
                        help="Total number of training epochs to perform.")
    parser.add_argument("--max_steps", default=-1, type=int,
                        help="If > 0: set total number of training steps to perform. Override num_train_epochs.")
    parser.add_argument("--warmup_steps", default=0, type=int,
                        help="Linear warmup over warmup_steps.")
    parser.add_argument("--warmup_proportion", default=0.1, type=float,
                        help="Proportion of training to perform linear learning rate warmup for,E.g., 0.1 = 10% of "
                             "training.")
    parser.add_argument('--logging_steps', type=int, default=200,
                        help="Log every X updates steps.")
    parser.add_argument('--save_steps', type=int, default=200,
                        help="Save checkpoint every X updates steps.")
    parser.add_argument("--eval_all_checkpoints", action='store_true',
                        help="Evaluate all checkpoints starting with the same prefix as model_name ending and ending "
                             "with step number")
    parser.add_argument("--no_cuda", action='store_true',
                        help="Avoid using CUDA when available")
    parser.add_argument('--overwrite_output_dir', action='store_true',
                        help="Overwrite the content of the output directory")
    parser.add_argument('--overwrite_cache', action='store_true',
                        help="Overwrite the cached training and evaluation sets")
    parser.add_argument('--seed', type=int, default=42,
                        help="random seed for initialization")
    parser.add_argument('--fp16', action='store_true',
                        help="Whether to use 16-bit (mixed) precision (through NVIDIA apex) instead of 32-bit")
    parser.add_argument('--fp16_opt_level', type=str, default='O1',
                        help="For fp16: Apex AMP optimization level selected in ['O0', 'O1', 'O2', and 'O3']."
                             "See details at https://nvidia.github.io/apex/amp.html")
    parser.add_argument("--local_rank", type=int, default=-1,
                        help="For distributed training: local_rank")
    parser.add_argument('--server_ip', type=str, default='', help="For distant debugging.")
    parser.add_argument('--server_port', type=str, default='', help="For distant debugging.")

    parser.add_argument('--overwrite_batch_size', type=int, default=-1, help="Overwrite batch size.")
    parser.add_argument('--overwrite_loss_mode', type=str, default='', help="Overwrite loss mode.")
    parser.add_argument('--overwrite_focal_alpha', type=float, default=-1, help="Overwrite loss focal alpha.")

    args = parser.parse_args()
    return args

def load_additinoal_settings(args):
    # additional parameters

    # general
    args.FORCE_CPU = False
    args.local_rank = 0

    # train para
    args.RESUME_STEP = -1
    args.DEBUG_TRAIN_INFO = False

    # eval para
    args.DEBUG_EVAL_DUMP = False

    # test para
    args.TEST_STEP = -1

    # loss&output option
    args.CLASSIFIER_THRES = 0.5
    args.LOSS_PARA = {'mode': 'bce',
                      'use_weight': True}
    args.MESH_GRAPH_PATH = ''

    # class experiments
    args.TOPN_MODE = 'None'
    args.TOPN_MAPPING = ''
    args.CLASS_WEIGHT = ''

    return args

def set_logger(logger, LOG_FILE, clean=False):
    logging.basicConfig(filename=LOG_FILE,
                        format='%(asctime)s - %(levelname)s - %(name)s - %(funcName)s - %(message)s',
                        datefmt='%m/%d/%Y %H:%M:%S')

    logger = logging.getLogger()
    logger.setLevel(logging.DEBUG)

    if clean:
        if os.path.isfile(LOG_FILE):
            with open(LOG_FILE, 'w') as f:
                pass

    ch = logging.StreamHandler()

    logger.addHandler(ch)

    return logger


logger = None

if __name__ == "__main__":
    #     global logger
    logger = set_logger(logger, 'log.txt', True)

    args = load_default_settings()

    args = load_additinoal_settings(args)

    main(args)
