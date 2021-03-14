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

    SPECIAL_TOKENS = [' <*> ',' <**> ',' <***> ']
    # forward_tokenizer = MarianTokenizer.from_pretrained(forward_mname)
    # foward_model = MarianMTModel.from_pretrained(forward_mname)
    # backward_tokenizer = MarianTokenizer.from_pretrained(backward_mname)
    # backward_model = MarianMTModel.from_pretrained(backward_mname)
    # translated = foward_model.generate(**forward_tokenizer.prepare_seq2seq_batch(sample_texts, return_tensors="pt"))
    # tgt_text = [forward_tokenizer.decode(t, skip_special_tokens=True) for t in translated]
    # back_translated = backward_model.generate(**backward_tokenizer.prepare_seq2seq_batch(tgt_text, return_tensors="pt"))
    # output = [backward_tokenizer.decode(t, skip_special_tokens=True) for t in back_translated]
    bt_map = {}
    for i in range(len(sample_texts)):
        bt_map[sample_texts[i]] = sample_texts[i]
    return bt_map

def read_squad(path):
    path = Path(path)
    with open(path, 'rb') as f:
        squad_dict = json.load(f)
    data_dict = {'question': [], 'context': [], 'id': [], 'answer': []}
    for group in squad_dict['data']:
        for passage in group['paragraphs']: # error here, list indices must be integers, not str: passage is list
            context = passage['context']
            for qa in passage['qas']:
                question = qa['question']
                if len(qa['answers']) == 0:
                    data_dict['question'].append(question)
                    data_dict['context'].append(context)
                    data_dict['id'].append(qa['id'])
                else:
                    for answer in  qa['answers']:
                        data_dict['question'].append(question)
                        data_dict['context'].append(context)
                        data_dict['id'].append(qa['id'])
                        data_dict['answer'].append(answer)
    id_map = ddict(list)
    for idx, qid in enumerate(data_dict['id']):
        id_map[qid].append(idx)

    data_dict_collapsed = {'question': [], 'context': [], 'id': []}
    if data_dict['answer']:
        data_dict_collapsed['answer'] = []
    for qid in id_map:
        ex_ids = id_map[qid]
        data_dict_collapsed['question'].append(data_dict['question'][ex_ids[0]])
        data_dict_collapsed['context'].append(data_dict['context'][ex_ids[0]])
        data_dict_collapsed['id'].append(qid)
        if data_dict['answer']:
            all_answers = [data_dict['answer'][idx] for idx in ex_ids]
            data_dict_collapsed['answer'].append({'answer_start': [answer['answer_start'] for answer in all_answers],
                                                  'text': [answer['text'] for answer in all_answers]})
    return data_dict_collapsed

def parse_batch(squad_dict):
    title_batch = []        # list of strings (titles)
    context_batch = []
    question_batch = []
    ids_batch = []          # list of lists (passages) of lists (qas) of ids
    answer_batch = []
    answer_start_batch = [] # list of lists (passages) of lists (qas) of lists (answer starts) of ints (answer start index)
    i = 0
    for group in (squad_dict['data']):
        if i % 100 == 0:
            print(group)
            print(len(title_batch), len(context_batch), len(question_batch))
       
        # print(group.keys())
        if 'title' in group:
            title_batch += [group['title']]
        if 'paragraphs' in group and len(group['paragraphs']) > 0:
            for paragraph in group['paragraphs']:
                bt_para = {'context': '', 'qas': {'question': '', 'id': '', 'answers':[]}}
                # bt_para['context'] = translate(paragraph['context'])
                context_batch += [paragraph['context']]
                if 'qas' in paragraph:
                    for qa in paragraph['qas']:
                        question_batch += [qa['question']]
                        for answer in qa['answers']:
                            answer_batch += [answer['text']]
                        # bt_para['qas']['question'] = translate(qa['question'])
                        #  bt_para['qas']['id'] = i
                        # i += 1
                # backtranslated['paragraphs'] += [bt_para]
            # print(backtranslated)
        else:
            break
        i += 1
        # print(i)
    # for group in (squad_dict['data']):
    #     title_batch += [group['title']]
    #     context_batch_item = [] # list of contexts strings separated by <sep>
    #     # such that context_batch[i] returns list of passages assoc w/ titles[i]
        
    #     question_batch_item = []
    #     ids_batch_item = []      # each item in list corresponds to one context
    #     answer_batch_item = []
    #     answer_start_batch_item = []
    #     for p_i, passage in enumerate(group['paragraphs']):
    #         context_batch_item.append(passage['context'])
    #         ids_batch_qa_item = [] # each item in list correponds to one qa object
    #         answer_start_batch_qa_item = []
    #         for q_i, qa in enumerate(passage['qas']): # id assoc w/ each qa object
    #             question_batch_item.append(qa['question'])
                
    #             ids_batch_qa_item.append(qa['id'])
                
    #             # want answer_batch to be a string: 
    #             # answer text sep  <***>  if they are dif answers, all belong to same question (qa object)
    #             # separated by  <**>  if they are dif qa objects, same context
    #             # separated by  <*>  if they are dif contexts
    #             answer_start_batch_ans_item = [] # each item in list corresponds to one answer in the current qa object
    #             for a_i,answer in enumerate(qa['answers']):
    #                 answer_batch_item.append(answer['text'])
    #                 answer_start_batch_ans_item.append(answer['answer_start'])
    #                 # if a_i != len(qa['answers'])-1: # don't append separator tokens at the end if this is the last item
    #                 #     answer_batch_item.append(' <***> ')

    #             ids_batch_item.append(ids_batch_qa_item)
    #             answer_start_batch_qa_item.append(answer_start_batch_ans_item)
    #             # if q_i != len(passage['qas'])-1:    # don't append separator tokens at the end if this is the last item
    #             #     question_batch_item.append(' <**> ')
    #             #     answer_batch_item.append(' <**> ')
            
    #         ids_batch.append(ids_batch_item)
    #         answer_start_batch_item.append(answer_start_batch_qa_item)
    #         # if p_i != len(group['paragraphs'])-1:   # don't append separator tokens at the end if this is the last item
    #         #     context_batch_item.append(' <*> ')
    #         #     question_batch_item.append(' <*> ')
    #         #     answer_batch_item.append(' <*> ')
            
    #     context_batch.append(''.join(context_batch_item))
    #     question_batch.append(''.join(question_batch_item))
    #     answer_batch.append(''.join(answer_batch_item))
    #     answer_start_batch.append(answer_start_batch_item)

    # print("\ntitle_batch:\n",title_batch, "\ncontext_batch:\n",context_batch, "\nquestion_batch:\n",question_batch, "\nids_batch:\n",ids_batch, "\nanswer_batch:\n",answer_batch,"\nanswer_start_batch:\n",answer_start_batch)
    # print(len(context_batch))
    return title_batch, context_batch, question_batch, ids_batch, answer_batch, answer_start_batch

def augment_squad(path):
    path = Path(path)

    with open(path, 'rb') as f:
        squad_dict = json.load(f)

    title_batch, context_batch, question_batch, ids_batch, answer_batch, answer_start_batch = parse_batch(squad_dict)
    
    title_map = translate(title_batch)
    print("title done translating!")
    context_map = translate(context_batch)
    print("contexts done translating!")
    question_map = translate(question_batch)
    print("questions done translating!")
    answer_map = translate(answer_batch)
    print("all done translating!")

    # reconstruct_backtranslated_data
    new_squad_data = squad_dict
    q_id = 0
    for group in squad_dict['data']:
        bt_group = {}
        main_group = {}
        bt_group['title'] = group['title']
        bt_passages = []
        for passage in group['paragraphs']: # error here, list indices must be integers, not str: passage is list
            context = passage['context']
            paragraph_dict = {}
            paragraph_dict['context'] = context_map[context]
            bt_qas = []
            for qa in passage['qas']:
                qa_dict = {}
                question = qa['question']
                qa_dict['question'] = question_map[question]
                qa_dict['id'] = q_id
                i += 1
                answers = []
                for answer in  qa['answers']:
                    answer_dict = {}
                    answer_dict['answer_start'] = answer['answer_start']
                    answer_dict['text'] = answer_map[answer['text']]
                    answers += [answer_dict]
                bt_qas += [qa_dict]
            paragraph_dict['qas'] = bt_qas
            bt_passages += [paragraph_dict]
        bt_group['paragraphs'] += [bt_passages]
        new_squad_data['data'].append(bt_group)
    # for g_i in range(len(titles)):
    #     bt_group = {}
    #     bt_group['title'] = titles[g_i]
    #     bt_group['paragraphs'] = []

    #     contexts_by_cxt = contexts[g_i].split(' <*> ')
    #     questions_by_cxt = questions[g_i].split(' <*> ')
    #     answers_by_cxt = answers[g_i].split(' <*> ')
    #     print("answer_start_batch[g_i]",len(answer_start_batch[g_i]))

    #     for p_i in range(len(contexts_by_cxt)):
    #         bt_passage = {}
    #         bt_passage['context'] = contexts_by_cxt[p_i]
    #         bt_passage['qas'] = []

    #         questions_by_qas = questions_by_cxt[p_i].split(' <**> ')
    #         answers_by_qas = answers_by_cxt[p_i].split(' <**> ')
    #         print("answer_start_batch[g_i][p_i]",len(answer_start_batch[g_i][p_i]))

    #         for q_i in range(len(questions_by_qas)):
    #             bt_qa = {}
    #             bt_qa['question'] = questions_by_qas[q_i]
    #             bt_qa['id'] = ids_batch[g_i][p_i][q_i]
    #             bt_qa['answers'] = []

    #             answers_by_ans = answers_by_qas[q_i].split(' <***> ')
    #             print("answers_by_qas[q_i]",answers_by_qas[q_i])
    #             print("answers_by_ans",answers_by_ans)
    #             print("answer_start_batch[g_i][p_i][q_i]",len(answer_start_batch[g_i][p_i][q_i]))
    #             for a_i in range(len(answers_by_ans)):
    #                 print('a_i',a_i)
    #                 bt_ans = {}
    #                 bt_ans['answer_start'] = answer_start_batch[g_i][p_i][q_i][a_i]
    #                 bt_ans['text'] = answers_by_ans[a_i]

    #                 bt_qa['answers'].append(bt_ans)
    #             bt_passage['qas'].append(bt_qa)
    #         bt_group['paragraphs'].append(bt_passage)
    return new_squad_data

###############################################################################

    # path = Path(path)

    # with open(path, 'rb') as f:
    #     squad_dict = json.load(f)

    # new_squad_data = squad_dict
    # i = 1000
    # title_batch = []
    # context_batch = []
    # question_batch = []
    # answer_batch = {}
    # answer_batch_starts = {}
    # raw_answers = []
    # # print(squad_dict['data'])
    # for group in (squad_dict['data']):
    #     if i % 100 == 0:
    #         print(group)
    #         print(len(title_batch), len(context_batch), len(question_batch))
       
    #     # print(group.keys())
    #     if 'title' in group:
    #         title_batch += [group['title']]
    #     if 'paragraphs' in group and len(group['paragraphs']) > 0:
    #         for paragraph in group['paragraphs']:
    #             bt_para = {'context': '', 'qas': {'question': '', 'id': '', 'answers':[]}}
    #             # bt_para['context'] = translate(paragraph['context'])
    #             context_batch += [paragraph['context']]
    #             if 'qas' in paragraph:
    #                 for qa in paragraph['qas']:
    #                     question_batch += [qa['question']]
    #                     for answer in qa['answers']:
    #                         if qa['question'] not in answer_batch_starts:
    #                             answer_batch_starts[qa['question']] = []
    #                         if qa['question'] not in answer_batch:
    #                             answer_batch[qa['question']] = []
    #                         answer_batch_starts[qa['question']] += [answer['answer_start']]
    #                         answer_batch[qa['question']] += [answer['text']]
    #                         raw_answers += [answer['text']]
    #                     # bt_para['qas']['question'] = translate(qa['question'])
    #                     #  bt_para['qas']['id'] = i
    #                     # i += 1
    #             # backtranslated['paragraphs'] += [bt_para]
    #         # print(backtranslated)
    #     else:
    #         break
    #     i += 1
    #     # print(i)
    # titles = translate(title_batch)
    # print("title done!")
    # contexts = translate(context_batch)
    # print("contexts done!")
    # questions = translate(question_batch)
    # question_to_index = {}
    # a_count = 0
    # for key in answer_batch.keys():
    #     question_to_index[key] = a_count
    #     a_count += len(answer_batch[key])
    # print(context_batch)
    # print(list(answer_batch.values()))
    # answers = translate(raw_answers)
    # print("all done!")
    # counter = 0
    # for i in range(len(titles)):
    #     backtranslated = {}
    #     backtranslated['paragraphs'] = []
    #     backtranslated['title'] = titles[i]
    #     bt_para = {'context': '', 'qas': {'question': '', 'id': '', 'answers':[]}}
    #     bt_para['context'] = contexts[i]
    #     bt_para['qas']['question'] = questions[i]
    #     bt_para['qas']['answer'] = []
    #     for question in answer_batch.keys():
    #         for j in range(counter, question_to_index[question]):
    #             bt_para['qas']['answer'] += [{'answer_start': answer_batch_starts[question][j - counter], 'text': answers[j]}]
    #             counter += 1
    #     backtranslated['paragraphs'] += [bt_para]
    #     new_squad_data['data'].append(backtranslated)
    # return new_squad_data

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
    # translate_set('datasets/oodomain_train', 'smol')  
    translate_set('datasets/oodomain_train', 'duorc,race,relation_extraction')  
    # split_set('datasets/indomain_train', 'squad,nat_questions,newsqa')

main()
