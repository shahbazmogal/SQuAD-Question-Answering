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

from args import get_train_args
from collections import OrderedDict
from json import dumps
from models import BiDAF, PreTrainedBERT
from tensorboardX import SummaryWriter
from tqdm import tqdm
from ujson import load as json_load 
from util import collate_fn, SQuAD, convert_char_idx_to_token_idx
from transformers import BertTokenizerFast
import os

def main(args):
    # Set up logging and devices
    # TODO: Added
    # TOKENIZERS_PARALLELISM=False
    # os.environ["TOKENIZERS_PARALLELISM"] = "false"
    args.save_dir = util.get_save_dir(args.save_dir, args.name, training=True)
    log = util.get_logger(args.save_dir, args.name)
    tbx = SummaryWriter(args.save_dir)
    device, args.gpu_ids = util.get_available_devices()
    # TODO: Make this an arg, there's a copy in util also
    BERT_max_sequence_length = 512
    log.info(f'Args: {dumps(vars(args), indent=4, sort_keys=True)}')
    args.batch_size *= max(1, len(args.gpu_ids))

    # Set random seed
    log.info(f'Using random seed {args.seed}...')
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)

    # Get embeddings
    # log.info('Loading embeddings...')
    # word_vectors = util.torch_from_json(args.word_emb_file)

    # Get model
    log.info('Building model...')
    # model = BiDAF(word_vectors=word_vectors,
    #               hidden_size=args.hidden_size,
    #               drop_prob=args.drop_prob)
    model = PreTrainedBERT(device)
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
    while epoch != args.num_epochs:
        epoch += 1
        log.info(f'Starting epoch {epoch}...')
        with torch.enable_grad(), \
                tqdm(total=len(train_loader.dataset)) as progress_bar:
            for contexts, questions, answer_starts, answer_ends, ids in train_loader:
                # Setup for forward
                # cw_idxs = cw_idxs.to(device)
                # qw_idxs = qw_idxs.to(device)
                # batch_size = cw_idxs.size(0)
                batch_size = 32
                optimizer.zero_grad()
                tokenizer = BertTokenizerFast.from_pretrained('bert-base-uncased', do_lower_case=True)
                sequence_tuples = list(zip(contexts,questions))
                encoded_dict = tokenizer.batch_encode_plus(
                                    sequence_tuples,                      # Context to encode.
                                    add_special_tokens = True, # Add '[CLS]' and '[SEP]'
                                    max_length = BERT_max_sequence_length,           # Pad & truncate all sentences.
                                    padding = 'max_length',
                                    truncation=True,
                                    return_attention_mask = True,   # Construct attn. masks.
                                    return_tensors = 'pt',     # Return pytorch tensors.
                            )
                input_ids = torch.as_tensor(encoded_dict['input_ids'])
                attention_mask = torch.as_tensor(encoded_dict['attention_mask'])
                token_type_ids = torch.as_tensor(encoded_dict['token_type_ids'])
                # Forward
                input_ids, attention_mask = input_ids.to(device), attention_mask.to(device)
                log_p1, log_p2 = model(input_ids, attention_mask, token_type_ids)
                answer_start_token_idx, answer_end_token_idx = convert_char_idx_to_token_idx(encoded_dict, answer_starts, answer_ends)
                answer_start_token_idx, answer_end_token_idx = answer_start_token_idx.to(device), answer_end_token_idx.to(device)
                # print(questions[0])
                # answer_start_char_idx, answer_end_char_idx = encoded_dict.token_to_chars(0, answer_start_token_idx[0])[0], encoded_dict.token_to_chars(0, answer_end_token_idx[0])[1]
                # print(contexts[0][answer_start_char_idx:answer_end_char_idx])
                # print(contexts[0].split()[encoded_dict.token_to_word(0, answer_start_token_idx[0]):encoded_dict.token_to_word(0, answer_end_token_idx[0])])
                # Avoid NLL_Loss error when value > N_class, ie, longer paragraph
                # answer_start[answer_start > BERT_max_sequence_length - 1], answer_end[answer_end > BERT_max_sequence_length - 1] = BERT_max_sequence_length - 1, BERT_max_sequence_length - 1
                # print(y1)
                loss = F.nll_loss(log_p1, answer_start_token_idx) + F.nll_loss(log_p2, answer_end_token_idx)
                # print("Calculated loss")
                loss_val = loss.item()
                # print("Copied loss")

                # Backward
                # Added below myself, confirm
                model.zero_grad()
                loss.backward()
                # print("Finished backprop")
                nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)
                optimizer.step()
                scheduler.step(step // batch_size)
                # print("Calculated scheduler step")
                ema(model, step // batch_size)
                # print("Calculated ema step")

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
                    results, pred_dict = evaluate(model, tokenizer, dev_loader, device,
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


def evaluate(model, tokenizer, data_loader, device, eval_file, max_len, use_squad_v2):
    BERT_max_sequence_length = 512
    nll_meter = util.AverageMeter()

    model.eval()
    pred_dict = {}
    with open(eval_file, 'r') as fh:
        gold_dict = json_load(fh)
    with torch.no_grad(), \
            tqdm(total=len(data_loader.dataset)) as progress_bar:
        for contexts, questions, answer_starts, answer_ends, ids in data_loader:
            # for (context, question, answer_start, answer_end, id) in zip(contexts, questions, answer_starts, answer_ends, ids):
            #     print(question)
            #     print(context[answer_start:answer_end])
            # Setup for forward
            # cw_idxs = cw_idxs.to(device)
            # qw_idxs = qw_idxs.to(device)
            # batch_size = cw_idxs.size(0)
            # print(contexts[0])
            # print(questions[0])
            # print(answer_starts[0])
            # print(answer_ends[0])
            batch_size = 32
            # tokenizer = BertTokenizerFast.from_pretrained('bert-base-uncased', do_lower_case=True)
            sequence_tuples = list(zip(contexts,questions))
            encoded_dict = tokenizer.batch_encode_plus(
                                sequence_tuples,                      # Context to encode.
                                add_special_tokens = True, # Add '[CLS]' and '[SEP]'
                                max_length = BERT_max_sequence_length,           # Pad & truncate all sentences.
                                padding = 'max_length',
                                truncation=True,
                                return_attention_mask = True,   # Construct attn. masks.
                                return_tensors = 'pt',     # Return pytorch tensors.
                        )
            input_ids = torch.as_tensor(encoded_dict['input_ids'])
            attention_mask = torch.as_tensor(encoded_dict['attention_mask'])
            token_type_ids = torch.as_tensor(encoded_dict['token_type_ids'])
            # Forward
            input_ids, attention_mask = input_ids.to(device), attention_mask.to(device)
            log_p1, log_p2 = model(input_ids, attention_mask, token_type_ids)
            # print(answer_starts)
            answer_start_token_idx, answer_end_token_idx = convert_char_idx_to_token_idx(encoded_dict, answer_starts, answer_ends)
            answer_start_token_idx, answer_end_token_idx = answer_start_token_idx.to(device), answer_end_token_idx.to(device)
            # Avoid NLL_Loss error when value > N_class, ie, longer paragraph
            # answer_start[answer_start > BERT_max_sequence_length - 1], answer_end[answer_end > BERT_max_sequence_length - 1] = BERT_max_sequence_length - 1, BERT_max_sequence_length - 1
            # print(y1)
            loss = F.nll_loss(log_p1, answer_start_token_idx) + F.nll_loss(log_p2, answer_end_token_idx)
            nll_meter.update(loss.item(), batch_size)

            # Get F1 and EM scores
            p1, p2 = log_p1.exp(), log_p2.exp()
            # print(p1, p2)
            start_token_idxs, end_token_idxs = util.discretize(p1, p2, max_len, use_squad_v2)
            start_word_idx, end_word_idx= [], []
            print("Number of sentences in batch:", start_token_idxs.shape[0])
            for i in range(0, start_token_idxs.shape[0]):
                start_word_idx.append(encoded_dict.token_to_word(i, start_token_idxs[i]))
                end_word_idx.append(encoded_dict.token_to_word(i, end_token_idxs[i]))
                print("Answer", contexts[i][encoded_dict.token_to_chars(i, answer_start_token_idx[i])[0]: 
                encoded_dict.token_to_chars(i, answer_end_token_idx[i])[1]])
                print(i, "Token:", start_token_idxs[i], ", Word:", encoded_dict.token_to_word(i, start_token_idxs[i]))
                print(i, "Token:", end_token_idxs[i], ", Word:", encoded_dict.token_to_word(i, end_token_idxs[i]))
                break
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
