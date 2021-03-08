import json
import util
import random
import pprint
from pathlib import Path
from tqdm import tqdm

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

def main():
    split_set('datasets/indomain_train', 'squad,nat_questions,newsqa')

main()
