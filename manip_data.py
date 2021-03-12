import json
import util
import random
import pprint
from pathlib import Path
from tqdm import tqdm
from transformers import MarianTokenizer, MarianMTModel
from typing import List


def read_squad_small(path, n_splits=5, seed=1):
    random.seed(seed)
    path = Path(path)

    with open(path, 'rb') as f:
        squad_dict = json.load(f)

    n_groups = len(squad_dict['data'])
    n_groups_per_split = n_groups // n_splits # per group!
    print(f'Picking {n_groups_per_split} out of {n_groups} passages.')

    small_squad_dict = {'data': [random.sample(squad_dict['data'], n_groups_per_split)]}
    # for group in ((squad_dict['data'])):
    #     chosen_passages = random.sample(group['paragraphs'], n_samples_per_split)
    #     small_squad_dict['data'].append(chosen_passages) # appends list of sampled passages
    #     print('chosen passages:\n', chosen_passages)
    return {'data': random.sample(squad_dict['data'], n_groups_per_split), 'version': '1.1'}

    # Approach 2:
    # # 1. create qa:context dict
    # qa_d = {}
    # for group in squad_dict['data']:
    #     for passage in group['paragraphs']:
    #         for qa in passage['qas']:
    #             qa_d[qa] = passage['context']

    # # 2. sample from qa:context dict & create context:qas dict
    # chosen_qas = dict(random.sample(qa_d.items(), n_samples_per_split))
    # small_squad_dict = {'data': []}
    # for qa,context in chosen_qas:
    #     if context not in small_squad_dict['data']:
    #         small_squad_dict['data']['context']
    #         # TODO

def split_set(data_dir, datasets, n_splits=5):

    # reference from train.get_dataset
    datasets = datasets.split(',')
    for dataset in datasets:
        fp = f'{data_dir}/{dataset}'
        dataset_dict_curr = read_squad_small(fp, n_splits=n_splits)
        print(f'Saving {fp}_small')
        with open(f'{fp}_small', 'w+') as writefp:
            json.dump(dataset_dict_curr, writefp)

def translate_set():
    src = 'en'  # source language
    trg = 'fr'  # target language
    sample_text = "I like my cat, Tommy"
    mname = f'Helsinki-NLP/opus-mt-{src}-{trg}'

    tokenizer = MarianTokenizer.from_pretrained(mname)
    model = MarianMTModel.from_pretrained(mname)
    translated = model.generate(**tokenizer.prepare_seq2seq_batch([sample_text], return_tensors="pt"))
    tgt_text = [tokenizer.decode(t, skip_special_tokens=True) for t in translated]
    print(tgt_text)
    # model = MarianMTModel.from_pretrained(mname)
    # tok = MarianTokenizer.from_pretrained(mname)
    # input_ids = tokenizer("Studies have been shown that owning a dog is good for you", return_tensors="pt").input_ids  # Batch size 1
    # decoder_input_ids = tokenizer("<pad> Studien haben gezeigt dass es hilfreich ist einen Hund zu besitzen", return_tensors="pt", add_special_tokens=False).input_ids  # Batch size 1

    # outputs = model(input_ids=input_ids, decoder_input_ids=decoder_input_ids)
    # last_hidden_states = outputs.decoder_hidden_states
    # print(last_hidden_states)

def main():
    translate_set()
    # split_set('datasets/indomain_train', 'squad,nat_questions,newsqa')

main()
