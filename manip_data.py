import json
import util
import random
import pprint
from pathlib import Path
from tqdm import tqdm
from transformers import MarianTokenizer, MarianMTModel
from typing import List

def translate(sample_texts):
    print(sample_texts)
    src = 'en'  # source language
    trg = 'fr'  # target language
    forward_mname = f'Helsinki-NLP/opus-mt-{src}-{trg}'
    backward_mname = f'Helsinki-NLP/opus-mt-{trg}-{src}'

    forward_tokenizer = MarianTokenizer.from_pretrained(forward_mname)
    foward_model = MarianMTModel.from_pretrained(forward_mname)
    backward_tokenizer = MarianTokenizer.from_pretrained(backward_mname)
    backward_model = MarianMTModel.from_pretrained(backward_mname)

    translated = foward_model.generate(**forward_tokenizer.prepare_seq2seq_batch(sample_texts, return_tensors="pt"))
    tgt_text = [forward_tokenizer.decode(t, skip_special_tokens=True) for t in translated]
    back_translated = backward_model.generate(**backward_tokenizer.prepare_seq2seq_batch(tgt_text, return_tensors="pt"))
    output = [backward_tokenizer.decode(t, skip_special_tokens=True) for t in back_translated]
    print(output)
    return output

def augment_squad(path):
    path = Path(path)

    with open(path, 'rb') as f:
        squad_dict = json.load(f)

    new_squad_data = squad_dict
    i = 1000
    title_batch = []
    context_batch = []
    question_batch = []
    print(squad_dict['data'])
    for group in (squad_dict['data']):
        if i % 100 == 0:
            print(group)
            print(len(title_batch), len(context_batch), len(question_batch))
        backtranslated = {}
        # backtranslated['title'] = translate(group['title'])
        # print(group.keys())
        if 'title' in group:
            title_batch += [group['title']]
        backtranslated['paragraphs'] = []
        if 'paragraphs' in group:
            for paragraph in group['paragraphs']:
                bt_para = {'context': '', 'qas': {'question': '', 'id': '', 'answers':[]}}
                # bt_para['context'] = translate(paragraph['context'])
                context_batch += [paragraph['context']]
                if 'qas' in paragraph:
                    for qa in paragraph['qas']:
                        question_batch += [qa['question']]
                        # bt_para['qas']['question'] = translate(qa['question'])
                        #  bt_para['qas']['id'] = i
                        # i += 1
                # backtranslated['paragraphs'] += [bt_para]
            # print(backtranslated)
        else:
            break
        i += 1
        # print(i)
        new_squad_data['data'].append(backtranslated)
    translate(title_batch)
    print("title done!")
    translate(context_batch)
    print("contexts done!")
    translate(question_batch)
    print("all done!")
    return new_squad_data

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

def translate_set(data_dir, datasets):
    datasets = datasets.split(',')
    for dataset in datasets:
        fp = f'{data_dir}/{dataset}'
        dataset_dict_curr = augment_squad(fp)
        print(f'Saving {fp}_augmented')
        fp = f'{data_dir}/{dataset}'
        with open(f'{fp}_augmented', 'w+') as writefp:
            json.dump(dataset_dict_curr, writefp)
    # model = MarianMTModel.from_pretrained(mname)
    # tok = MarianTokenizer.from_pretrained(mname)
    # input_ids = tokenizer("Studies have been shown that owning a dog is good for you", return_tensors="pt").input_ids  # Batch size 1
    # decoder_input_ids = tokenizer("<pad> Studien haben gezeigt dass es hilfreich ist einen Hund zu besitzen", return_tensors="pt", add_special_tokens=False).input_ids  # Batch size 1

    # outputs = model(input_ids=input_ids, decoder_input_ids=decoder_input_ids)
    # last_hidden_states = outputs.decoder_hidden_states
    # print(last_hidden_states)

def main():
    translate_set('datasets/oodomain_train', 'duorc,race,relation_extraction')  
    # split_set('datasets/indomain_train', 'squad,nat_questions,newsqa')

main()
