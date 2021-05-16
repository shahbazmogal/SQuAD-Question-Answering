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
                ids.append(id)
                string_ids_temp.append(string_id)
                questions.append(question)
                contexts.append(context)
                if qas['answers']!= []:
                    for answer in qas['answers']:
                        answers.append(answer['text'])
                        answer_starts.append(answer["answer_start"])
                        answer_ends.append(answer["answer_start"]+len(answer['text']))
                #if it is impossible
                else:
                    answers.append("")
                    answer_starts.append(-1)
                    answer_ends.append(-1)
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
    with open('data/train-v2.0.json', 'r') as fh:
	    train_json = json_load(fh)
	    train_json = train_json["data"]
    contexts, questions, answers, answer_starts, answer_ends = preprocess(train_json, args_.train_record_file)
