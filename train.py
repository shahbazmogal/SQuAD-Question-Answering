"""Train a model on SQuAD.

Author:
    Chris Chute (chute@stanford.edu)
"""

import numpy as np
import random
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.optim.lr_scheduler as sched
import torch.utils.data as data
import util
from torchsummary import summary

from args import get_train_args
from collections import OrderedDict
from json import dumps
from models import BiDAF, PreTrainedBERT
from tensorboardX import SummaryWriter
from tqdm import tqdm
from ujson import load as json_load 
from util import collate_fn, SQuAD, convert_char_idx_to_token_idx, get_BERT_input
from transformers import BertTokenizerFast
import os

def main(args):
    # Set up logging and devices
    args.save_dir = util.get_save_dir(args.save_dir, args.name, training=True)
    log = util.get_logger(args.save_dir, args.name)
    tbx = SummaryWriter(args.save_dir)
    device, args.gpu_ids = util.get_available_devices()
    log.info(f'Args: {dumps(vars(args), indent=4, sort_keys=True)}')
    args.batch_size *= max(1, len(args.gpu_ids))

    # Set random seed
    log.info(f'Using random seed {args.seed}...')
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)

    # Get model
    log.info('Building model...')

    # TODO: Make this an arg, there's a copy in util also, also try 64 instead of 48
    question_max_token_length = 48
    BERT_max_sequence_length = 512
    model = PreTrainedBERT(device)
    for param in model.context_BERT.parameters():
        param.requires_grad = False

    for param in model.question_BERT.parameters():
        param.requires_grad = False

    model = nn.DataParallel(model, args.gpu_ids)
    if args.load_path:
        log.info(f'Loading checkpoint from {args.load_path}...')
        model, step = util.load_model(model, args.load_path, args.gpu_ids)
    else:
        step = 0
    model = model.to(device)
    model.train()
    ema = util.EMA(model, args.ema_decay)

    # Get saver
    saver = util.CheckpointSaver(args.save_dir,
                                 max_checkpoints=args.max_checkpoints,
                                 metric_name=args.metric_name,
                                 maximize_metric=args.maximize_metric,
                                 log=log)

    # Get optimizer and scheduler
    optimizer = optim.Adadelta(model.parameters(), args.lr,
                               weight_decay=args.l2_wd)
    scheduler = sched.LambdaLR(optimizer, lambda s: 1.)  # Constant LR

    # Get data loader
    log.info('Building dataset...')
    train_dataset = SQuAD(args.train_record_file, args.use_squad_v2)
    print("Training DataSet Loaded")
    train_loader = data.DataLoader(train_dataset,
                                   batch_size=args.batch_size,
                                   shuffle=True,
                                   num_workers=args.num_workers)
                                #    collate_fn=collate_fn)
    dev_dataset = SQuAD(args.dev_record_file, args.use_squad_v2)
    dev_loader = data.DataLoader(dev_dataset,
                                 batch_size=args.batch_size,
                                 shuffle=False,
                                 num_workers=args.num_workers)
                                #  collate_fn=collate_fn)

    # Train
    log.info('Training...')
    steps_till_eval = args.eval_steps
    epoch = step // len(train_dataset)
    question_tokenizer = BertTokenizerFast.from_pretrained('bert-base-uncased', do_lower_case=True)
    context_tokenizer = BertTokenizerFast.from_pretrained('bert-base-uncased', do_lower_case=True)
    
    debug_count = 0

    
    while epoch != args.num_epochs:
        epoch += 1
        log.info(f'Starting epoch {epoch}...')
        with torch.enable_grad(), \
                tqdm(total=len(train_loader.dataset)) as progress_bar:
            for contexts, questions, answer_starts, answer_ends, ids in train_loader:
                batch_size = args.batch_size
                optimizer.zero_grad()
                debug_question_number = 2
                print(questions[debug_question_number])
                questions_encoded_dict, questions_input_ids, questions_attn_mask, questions_token_type_ids = get_BERT_input(list(questions), question_tokenizer, question_max_token_length, device)
                contexts_encoded_dict, contexts_input_ids, contexts_attn_mask, contexts_token_type_ids = get_BERT_input(list(contexts), context_tokenizer, BERT_max_sequence_length, device)
                debug_tokens = contexts_encoded_dict.tokens(debug_question_number)
                log_p1, log_p2 = model(questions_input_ids, questions_attn_mask, questions_token_type_ids, contexts_input_ids, contexts_attn_mask, contexts_token_type_ids)
                answer_start_token_idx, answer_end_token_idx = convert_char_idx_to_token_idx(contexts_encoded_dict, answer_starts, answer_ends)
                answer_start_token_idx, answer_end_token_idx = answer_start_token_idx.to(device), answer_end_token_idx.to(device)

                # Get F1 and EM scores
                debug_p1, debug_p2 = log_p1.exp(), log_p2.exp()
                debug_pred_start_token_idxs, debug_pred_end_token_idxs = util.discretize(debug_p1, debug_p2, args.max_ans_len, args.use_squad_v2)
                print("Actual Answer:", debug_tokens[answer_start_token_idx[debug_question_number]:answer_end_token_idx[debug_question_number] + 1])
                print("Predicted Answer:", debug_tokens[debug_pred_start_token_idxs[debug_question_number]:debug_pred_end_token_idxs[debug_question_number]])
                debug_count += 1
                # if debug_count > 200:
                #     exit()
                loss = F.nll_loss(log_p1, answer_start_token_idx) + F.nll_loss(log_p2, answer_end_token_idx)
                loss_val = loss.item()

                model.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)
                optimizer.step()
                scheduler.step(step // batch_size)
                ema(model, step // batch_size)

                # Log info
                print("Logging info")
                step += batch_size
                progress_bar.update(batch_size)
                progress_bar.set_postfix(epoch=epoch,
                                         NLL=loss_val)
                tbx.add_scalar('train/NLL', loss_val, step)
                tbx.add_scalar('train/LR',
                               optimizer.param_groups[0]['lr'],
                               step)

                steps_till_eval -= batch_size
                if steps_till_eval <= 0:
                    steps_till_eval = args.eval_steps

                    # Evaluate and save checkpoint
                    log.info(f'Evaluating at step {step}...')
                    ema.assign(model)
                    results, pred_dict = evaluate(model, question_tokenizer, context_tokenizer, dev_loader, device,
                                                  args.dev_eval_file,
                                                  args.max_ans_len,
                                                  args.use_squad_v2)
                    saver.save(step, model, results[args.metric_name], device)
                    ema.resume(model)

                    # Log to console
                    results_str = ', '.join(f'{k}: {v:05.2f}' for k, v in results.items())
                    log.info(f'Dev {results_str}')

                    # Log to TensorBoard
                    log.info('Visualizing in TensorBoard...')
                    for k, v in results.items():
                        tbx.add_scalar(f'dev/{k}', v, step)
                    util.visualize(tbx,
                                   pred_dict=pred_dict,
                                   eval_path=args.dev_eval_file,
                                   step=step,
                                   split='dev',
                                   num_visuals=args.num_visuals)


def evaluate(model, question_tokenizer, context_tokenizer, data_loader, device, eval_file, max_len, use_squad_v2):
    args = get_train_args()
    question_max_token_length = 48
    BERT_max_sequence_length = 512
    nll_meter = util.AverageMeter()

    model.eval()
    pred_dict = {}
    with open(eval_file, 'r') as fh:
        gold_dict = json_load(fh)
    with torch.no_grad(), \
            tqdm(total=len(data_loader.dataset)) as progress_bar:
        for contexts, questions, answer_starts, answer_ends, ids in data_loader:
            batch_size = args.batch_size
            questions_encoded_dict, questions_input_ids, questions_attn_mask, questions_token_type_ids = get_BERT_input(list(questions), question_tokenizer, question_max_token_length, device)
            contexts_encoded_dict, contexts_input_ids, contexts_attn_mask, contexts_token_type_ids = get_BERT_input(list(contexts), context_tokenizer, BERT_max_sequence_length, device)
            log_p1, log_p2 = model(questions_input_ids, questions_attn_mask, questions_token_type_ids, contexts_input_ids, contexts_attn_mask, contexts_token_type_ids)
            answer_start_token_idx, answer_end_token_idx = convert_char_idx_to_token_idx(contexts_encoded_dict, answer_starts, answer_ends)
            answer_start_token_idx, answer_end_token_idx = answer_start_token_idx.to(device), answer_end_token_idx.to(device)
            loss = F.nll_loss(log_p1, answer_start_token_idx) + F.nll_loss(log_p2, answer_end_token_idx)
            nll_meter.update(loss.item(), batch_size)

            # Get F1 and EM scores
            p1, p2 = log_p1.exp(), log_p2.exp()
            # print(p1, p2)
            start_token_idxs, end_token_idxs = util.discretize(p1, p2, max_len, use_squad_v2)
            start_word_idx, end_word_idx= [], []
            print("Number of sentences in batch:", start_token_idxs.shape[0])
            for i in range(0, start_token_idxs.shape[0]):
                print("Question:", questions[i])
                # print("Number of tokens in context", len(contexts_encoded_dict.tokens(i))) # Always 512
                print("Number of characters in context", len(contexts[i]))
                print("Start Token Idx actual answer:", answer_start_token_idx[i], " at char idx", contexts_encoded_dict.token_to_chars(i, answer_start_token_idx[i])[0])
                print("End Token Idx actual answer:", answer_end_token_idx[i], " at char idx", contexts_encoded_dict.token_to_chars(i, answer_end_token_idx[i])[1])
                print("Actual answer:", contexts[i][contexts_encoded_dict.token_to_chars(i, answer_start_token_idx[i])[0]: 
                contexts_encoded_dict.token_to_chars(i, answer_end_token_idx[i])[1]])
                print(i, "Pred Token:", start_token_idxs[i], ", Word:", contexts_encoded_dict.token_to_word(i, start_token_idxs[i]))
                print(i, "Pred Token:", end_token_idxs[i], ", Word:", contexts_encoded_dict.token_to_word(i, end_token_idxs[i]))
                context_words = contexts[i].split()
                print(i, "Predicted Answer:", context_words[contexts_encoded_dict.token_to_word(i, start_token_idxs[i]):contexts_encoded_dict.token_to_word(i, end_token_idxs[i]) + 1])
                print()
                print()
                # TODO: Append to start word idx
                # except:
                #     start_word_idx.append(len(contexts[i]) - 1)
                #     end_word_idx.append(len(contexts[i]) - 1)
                #     print("Passage longer than BERT Model expects")
            # print(log_p1[8])
            # print(p1[8])
            # print(starts[8])
            # Log info
            progress_bar.update(batch_size)
            progress_bar.set_postfix(NLL=nll_meter.avg)

            preds, _ = util.convert_tokens(gold_dict,
                                           ids.tolist(),
                                           start_word_idx,
                                           end_word_idx,
                                           use_squad_v2)
            pred_dict.update(preds)

    model.train()
    # print(gold_dict)
    # print(pred_dict)
    # exit() 
    results = util.eval_dicts(gold_dict, pred_dict, use_squad_v2)
    results_list = [('NLL', nll_meter.avg),
                    ('F1', results['F1']),
                    ('EM', results['EM'])]
    print(results_list)
    if use_squad_v2:
        results_list.append(('AvNA', results['AvNA']))
    results = OrderedDict(results_list)

    return results, pred_dict


if __name__ == '__main__':
    main(get_train_args())
