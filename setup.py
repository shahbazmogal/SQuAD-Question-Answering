# """Download and pre-process SQuAD and GloVe.

# Usage:
#     > source activate squad
#     > python setup.py

# Pre-processing code adapted from:
#     > https://github.com/HKUST-KnowComp/R-Net/blob/master/prepro.py

# Author:
#     Chris Chute (chute@stanford.edu)
# """

import numpy as np
import os
import spacy
import ujson as json
from ujson import load as json_load
import urllib.request

from args import get_setup_args
from codecs import open
from collections import Counter
from subprocess import run
from tqdm import tqdm
from zipfile import ZipFile


def download_url(url, output_path, show_progress=True):
    class DownloadProgressBar(tqdm):
        def update_to(self, b=1, bsize=1, tsize=None):
            if tsize is not None:
                self.total = tsize
            self.update(b * bsize - self.n)

    if show_progress:
        # Download with a progress bar
        with DownloadProgressBar(unit='B', unit_scale=True,
                                 miniters=1, desc=url.split('/')[-1]) as t:
            urllib.request.urlretrieve(url,
                                       filename=output_path,
                                       reporthook=t.update_to)
    else:
        # Simple download with no progress bar
        urllib.request.urlretrieve(url, output_path)

def download(args):
    downloads = [
        # Can add other downloads here (e.g., other word vectors)
        ('GloVe word vectors', args.glove_url),
    ]

    for name, url in downloads:
        output_path = url_to_data_path(url)
        if not os.path.exists(output_path):
            print(f'Downloading {name}...')
            download_url(url, output_path)

        if os.path.exists(output_path) and output_path.endswith('.zip'):
            extracted_path = output_path.replace('.zip', '')
            if not os.path.exists(extracted_path):
                print(f'Unzipping {name}...')
                with ZipFile(output_path, 'r') as zip_fh:
                    zip_fh.extractall(extracted_path)

    print('Downloading spacy language model...')
    run(['python', '-m', 'spacy', 'download', 'en'])

def is_answerable(example):
    return example['answer_end'] > 0

def save(filename, obj, message=None):
    if message is not None:
        print(f"Saving {message}...")
        with open(filename, "w") as fh:
            json.dump(obj, fh)

# preprocessing
def preprocess(data, out_file):
    answers = []
    contexts = []
    questions = []
    answer_starts = []
    answer_ends = []
    ids = []
    string_ids_temp = []
    total = 0
    for article in data:
        for paragraph in article['paragraphs']:
            #looking at each context
            context = paragraph['context']
            for qas in paragraph['qas']:
                total += 1
                question = qas['question']
                id = total
                string_id = qas['id']
                if qas['answers']!= []:
                    for answer in qas['answers']:
                        ids.append(id)
                        string_ids_temp.append(string_id)
                        questions.append(question)
                        contexts.append(context)
                        answers.append(answer['text'])
                        answer_starts.append(answer["answer_start"])
                        answer_ends.append(answer["answer_start"]+len(answer['text']))
                #if it is impossible
                else:
                    ids.append(id)
                    string_ids_temp.append(string_id)
                    questions.append(question)
                    contexts.append(context)
                    answers.append("")
                    answer_starts.append(-1)
                    answer_ends.append(-1)
            # print("`Answers", len(answers))
            # print("Questions", len(questions))
            # print("Answer Starts", len(answer_starts))
            # print("Answer Ends", len(answers))
            # print("-------------")
            # print()
            # print()
            # print()
            # print()`
    np.savez(out_file,
        contexts = np.array(contexts),
        questions = np.array(questions),
        answer_starts=np.array(answer_starts),
        answer_ends=np.array(answer_ends),
        ids=np.array(ids))
    return contexts, questions, answers, answer_starts, answer_ends

if __name__ == '__main__':
    # # Get command-line args
    args_ = get_setup_args()
    # For debugging
    print("Started creating train file")
    with open('data/train-v2.0.json', 'r') as fh:
	    train_json = json_load(fh)
	    train_json = train_json["data"]
    contexts, questions, answers, answer_starts, answer_ends = preprocess(train_json, args_.train_record_file)

    # count = 0
    # for (context, question, answer_start, answer_end) in zip(contexts, questions, answer_starts, answer_ends):
        # if count < 10:
            # print(question)
            # print(context[answer_start:answer_end])
            # count += 1

    count = 0
    print("Started creating dev file")
    with open('data/dev-v2.0.json', 'r') as fh:
	    dev_json = json_load(fh)
	    dev_json = dev_json["data"]
    contexts, questions, answers, answer_starts, answer_ends = preprocess(dev_json, args_.dev_record_file)
    # for (context, question, answer_start, answer_end) in zip(contexts, questions, answer_starts, answer_ends):
    #     if count < 10:
    #         print(question)
    #         print(context[answer_start:answer_end])
    #         count += 1

    print("Started creating test file")
    with open('data/test-v2.0.json', 'r') as fh:
	    test_json = json_load(fh)
	    test_json = test_json["data"]
    contexts, questions, answers, answer_starts, answer_ends = preprocess(test_json, args_.test_record_file)