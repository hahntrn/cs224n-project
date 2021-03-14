import argparse
import json
import os
from collections import OrderedDict
import torch
import csv
import util
import numpy as np
from transformers import DistilBertTokenizerFast
from transformers import DistilBertForQuestionAnswering
from transformers import AdamW
from tensorboardX import SummaryWriter
from sentence_transformers import SentenceTransformer # pip3 install -U sentence-transformers
from sentence_transformers import util as sent_util


from torch.utils.data import DataLoader
from torch.utils.data.sampler import RandomSampler, SequentialSampler
from args import get_train_test_args

from tqdm import tqdm

from heapq import nlargest, nsmallest
import random
import re


def prepare_eval_data(dataset_dict, tokenizer):
    tokenized_examples = tokenizer(dataset_dict['question'],
                                   dataset_dict['context'],
                                   truncation="only_second",
                                   stride=128,
                                   max_length=384,
                                   return_overflowing_tokens=True,
                                   return_offsets_mapping=True,
                                   padding='max_length')
    # Since one example might give us several features if it has a long context, we need a map from a feature to
    # its corresponding example. This key gives us just that.
    sample_mapping = tokenized_examples.pop("overflow_to_sample_mapping")

    # For evaluation, we will need to convert our predictions to substrings of the context, so we keep the
    # corresponding example_id and we will store the offset mappings.
    tokenized_examples["id"] = []
    for i in tqdm(range(len(tokenized_examples["input_ids"]))):
        # Grab the sequence corresponding to that example (to know what is the context and what is the question).
        sequence_ids = tokenized_examples.sequence_ids(i)
        # One example can give several spans, this is the index of the example containing this span of text.
        sample_index = sample_mapping[i]
        tokenized_examples["id"].append(dataset_dict["id"][sample_index])
        # Set to None the offset_mapping that are not part of the context so it's easy to determine if a token
        # position is part of the context or not.
        tokenized_examples["offset_mapping"][i] = [
            (o if sequence_ids[k] == 1 else None)
            for k, o in enumerate(tokenized_examples["offset_mapping"][i])
        ]

    return tokenized_examples

# pass indomain and oodomain dataset dicts in here
def prepare_train_data(args, dataset_dict, tokenizer, augment_dataset_dicts=None,
        sent_model=None):
    tokenized_examples = tokenizer(dataset_dict['question'],
                                   dataset_dict['context'],
                                   truncation="only_second",
                                   stride=128,
                                   max_length=384,
                                   return_overflowing_tokens=True,
                                   return_offsets_mapping=True,
                                   padding='max_length')
    sample_mapping = tokenized_examples["overflow_to_sample_mapping"]
    offset_mapping = tokenized_examples["offset_mapping"]
    RE_SENTENCE_DELIMS = r"\. |\? |\! |\n"

    ### FINETUNE if augment dataset is prodived
    # if augment flag is True:
    # for each context in dataset_dict['context']:
        # find the top 50% closest contexts from augment_dataset_dict['context']
        # randomly sample one of these contexts and add it to the original context

    if augment_dataset_dicts is not None:
        print(f"Augmenting contexts in prepare_training_data...mask={args.mask}, sep_sentences={args.sep_sentences}")

        if sent_model is not None:
            print("Calculating sentence embedding and cosine similarities...")

            sent_embedding = sent_model.encode(dataset_dict['context'], convert_to_tensor=True, show_progress_bar=False) # ind context embeddings
            aug_embeddings_classes = [sent_model.encode(aug_dict['context'], convert_to_tensor=True, show_progress_bar=False)
                                    for aug_dict in augment_dataset_dicts]
            cosine_sim_classes = [sent_util.pytorch_cos_sim(sent_embedding, aug_emb)
                                    for aug_emb in aug_embeddings_classes]

            # print("sent_embedding",sent_embedding) #D
            # print("aug_embeddings_classes",aug_embeddings_classes)#D
            # print("cosine_sim_classes",cosine_sim_classes)#D

            print("Appending demonstrations...")
            for context_i, ind_context in enumerate(tqdm(dataset_dict['context'])):
                for class_i in range(len(augment_dataset_dicts)):
                    augment_dataset_dict = augment_dataset_dicts[class_i]
                    selected_context = augment_dataset_dict['context'][torch.argmax(cosine_sim_classes[class_i][context_i])]
                    if args.mask:
                        word_to_mask = random.choice(selected_context.split())
                        selected_context = selected_context.replace(word_to_mask, tokenizer.mask_token)
                    if args.sep_sentences:
                        selected_context = re.sub(RE_SENTENCE_DELIMS, tokenizer.sep_token, selected_context)
                    if args.single_sentence_demons:
                        sentences = re.split(RE_SENTENCE_DELIMS, selected_context)
                        ind_context_embedding = sent_embedding[context_i]
                        aug_context_sentences = sent_model.encode(sentences, convert_to_tensor=True, show_progress_bar=False)
                        cosine_sim_sentences = sent_util.pytorch_cos_sim(sent_embedding, aug_context_sentences)[0]
                        # print('max index in tensor: ', torch.argmax(cosine_sim_sentences))
                        # print('len of cosine_sim_sentences: ', len(cosine_sim_sentences))
                        # print(cosine_sim_sentences)

                        top_k = 1

                        # Sort the results in decreasing order and get the first top_k
                        top_results = np.argpartition(-cosine_sim_sentences, range(top_k))[0:top_k]
                        selected_sentence = sentences[top_results[0]]

                        # selected_sentence = sentences[torch.argmax(cosine_sim_sentences)]
                        # print(f"--selected_sentence: {selected_sentence}")
                        # print(f"--Most relevant to: {selected_context}")
                        selected_context = selected_sentence # should call this demonstration

                    dataset_dict['context'][context_i] += ' ' + tokenizer.sep_token + ' ' + selected_context
                    # print("selected_context",selected_context)

                # best_demonstrations = [
                #                 augment_dataset_dict['context'][
                #                     torch.argmax(cosine_sim_classes[class_i][context_i])]
                #                 for class_i in len(augment_dataset_dicts)]
                # print("best_demonstrations",best_demonstrations)#D

        else:
            # list of <num_class> lists each containing <num_contexts_in_class> context strings
            aug_freq_lists = [util.get_freq_list(augment_dataset_dict) for augment_dataset_dict in augment_dataset_dicts]

            # loop over each in-domain context
            for context_i, ind_context in enumerate(tqdm(dataset_dict['context'])):
                # for each class of out-of-domain dataset
                for class_i, augment_dataset_dict in enumerate(augment_dataset_dicts):
                    # compute similarity scores for each context in this class
                    sim_scores = []
                    for aug_context_i, aug_context in enumerate(augment_dataset_dict['context']):
                        sim_scores.append((util.get_dict_similarity(aug_freq_lists[class_i][aug_context_i], util.get_freq_dict(aug_context)),aug_context_i))

                    # append the a random context in the top 50% most similar to the in-domain example's context
                    num_contexts_in_class = len(augment_dataset_dict['context'])
                    choice_i = random.randint(0, num_contexts_in_class // 2 - 1) # range inclusive
                    choices = nlargest(num_contexts_in_class // 2, sim_scores)
                    chosen_aug_context_score, chosen_aug_context_i = choices[choice_i]
                    selected_context = augment_dataset_dict['context'][chosen_aug_context_i]
                    if args.mask:
                        for i in range(len(selected_context.split()) // 12):
                            words = selected_context.split(" ")
                            word_to_mask = random.randint(0, len(words)-1)
                            new_context = ""
                            for i in range(len(words)):
                                if i == word_to_mask:
                                    new_context += tokenizer.mask_token
                                else:
                                    new_context += words[i]
                                new_context += " "
                            selected_context = new_context
                        print(f"Masked context: {selected_context}")
                    if args.sep_sentences:
                        selected_context = re.sub(RE_SENTENCE_DELIMS, tokenizer.sep_token, selected_context)
                        print(f"Separated context: {selected_context}")

                    print("Appending demonstration:", dataset_dict['context'][context_i] + ' ' + tokenizer.sep_token + ' ' + selected_context[:-1:])
                    dataset_dict['context'][context_i] = dataset_dict['context'][context_i] + ' ' + tokenizer.sep_token + ' ' + selected_context[:-1:]
        print("Done augmenting contexts!")
    ### END FINETUNE

    # Let's label those examples!
    tokenized_examples["start_positions"] = []
    tokenized_examples["end_positions"] = []
    tokenized_examples['id'] = []
    inaccurate = 0
    for i, offsets in enumerate(tqdm(offset_mapping)):
        # We will label impossible answers with the index of the CLS token.
        input_ids = tokenized_examples["input_ids"][i]
        cls_index = input_ids.index(tokenizer.cls_token_id)

        # Grab the sequence corresponding to that example (to know what is the context and what is the question).
        sequence_ids = tokenized_examples.sequence_ids(i)

        # One example can give several spans, this is the index of the example containing this span of text.
        sample_index = sample_mapping[i]
        answer = dataset_dict['answer'][sample_index]
        # Start/end character index of the answer in the text.
        start_char = answer['answer_start'][0]
        end_char = start_char + len(answer['text'][0])
        tokenized_examples['id'].append(dataset_dict['id'][sample_index])
        # Start token index of the current span in the text.
        token_start_index = 0
        while sequence_ids[token_start_index] != 1:
            token_start_index += 1

        # End token index of the current span in the text.
        token_end_index = len(input_ids) - 1
        while sequence_ids[token_end_index] != 1:
            token_end_index -= 1

        # Detect if the answer is out of the span (in which case this feature is labeled with the CLS index).
        if not (offsets[token_start_index][0] <= start_char and offsets[token_end_index][1] >= end_char):
            tokenized_examples["start_positions"].append(cls_index)
            tokenized_examples["end_positions"].append(cls_index)
        else:
            # Otherwise move the token_start_index and token_end_index to the two ends of the answer.
            # Note: we could go after the last offset if the answer is the last word (edge case).
            while token_start_index < len(offsets) and offsets[token_start_index][0] <= start_char:
                token_start_index += 1
            tokenized_examples["start_positions"].append(token_start_index - 1)
            while offsets[token_end_index][1] >= end_char:
                token_end_index -= 1
            tokenized_examples["end_positions"].append(token_end_index + 1)
            # assertion to check if this checks out
            context = dataset_dict['context'][sample_index]
            offset_st = offsets[tokenized_examples['start_positions'][-1]][0]
            offset_en = offsets[tokenized_examples['end_positions'][-1]][1]
            if context[offset_st : offset_en] != answer['text'][0]:
                inaccurate += 1

    total = len(tokenized_examples['id'])
    print(f"Preprocessing not completely accurate for {inaccurate}/{total} instances")
    return tokenized_examples



# pass in indomain and oodomain dataset dicts
def read_and_process(args, tokenizer, dataset_dict, dir_name, dataset_name, split,
        augment_dataset_dicts=None, sent_model=None):
    #TODO: cache this if possible
    cache_path = f'{dir_name}/{dataset_name}_encodings.pt'
    if os.path.exists(cache_path) and not args.recompute_features:
        tokenized_examples = util.load_pickle(cache_path)
    else:
        if split=='train':
            # if augment flag is true/augment dataset not none:
            # tokenized_examples = prepare_train_data(args, dataset_dict, augment_dataset_dict, tokenizer)
            tokenized_examples = prepare_train_data(args, dataset_dict, tokenizer,
                    augment_dataset_dicts=augment_dataset_dicts,
                    sent_model=sent_model)
        else:
            tokenized_examples = prepare_eval_data(dataset_dict, tokenizer)
        util.save_pickle(tokenized_examples, cache_path)
    return tokenized_examples



#TODO: use a logger, use tensorboard
class Trainer():
    def __init__(self, args, log):
        self.lr = args.lr
        self.num_epochs = args.num_epochs
        self.device = args.device
        self.eval_every = args.eval_every
        self.path = os.path.join(args.save_dir, 'checkpoint')
        self.num_visuals = args.num_visuals
        self.save_dir = args.save_dir
        self.log = log
        self.visualize_predictions = args.visualize_predictions
        if not os.path.exists(self.path):
            os.makedirs(self.path)

    def save(self, model):
        model.save_pretrained(self.path)

    def evaluate(self, model, data_loader, data_dict, return_preds=False, split='validation'):
        device = self.device

        model.eval()
        pred_dict = {}
        all_start_logits = []
        all_end_logits = []
        with torch.no_grad(), \
                tqdm(total=len(data_loader.dataset)) as progress_bar:
            for batch in data_loader:
                # Setup for forward
                input_ids = batch['input_ids'].to(device)
                attention_mask = batch['attention_mask'].to(device)
                batch_size = len(input_ids)
                outputs = model(input_ids, attention_mask=attention_mask)
                # Forward
                start_logits, end_logits = outputs.start_logits, outputs.end_logits
                # TODO: compute loss

                all_start_logits.append(start_logits)
                all_end_logits.append(end_logits)
                progress_bar.update(batch_size)

        # Get F1 and EM scores
        start_logits = torch.cat(all_start_logits).cpu().numpy()
        end_logits = torch.cat(all_end_logits).cpu().numpy()
        preds = util.postprocess_qa_predictions(data_dict,
                                                 data_loader.dataset.encodings,
                                                 (start_logits, end_logits))
        if split == 'validation':
            results = util.eval_dicts(data_dict, preds)
            results_list = [('F1', results['F1']),
                            ('EM', results['EM'])]
        else:
            results_list = [('F1', -1.0),
                            ('EM', -1.0)]
        results = OrderedDict(results_list)
        if return_preds:
            return preds, results
        return results

    # def finetuning
        # should finetune our model
        # at each in-domain training step, sample one example from ind, concatenate them w/ current
        # example x_in = ("paragraph", "question abt it") <---- dont worry abt rn its too hard
        # example y_in = ("answer to question") <--- same
        # each train step -- find the top 50% samples in ood terms of similarity w/ the current ind x
            # for each class:
                # find top 50% samples in ood term
                # we can concat 1 random out of the top 50% to x
            # continue training appropriately


        # concatenate them to x
        # continue training accordingly
        # answer prompt by filling in answer

    def finetune(self, model, train_dataloader, eval_dataloader, val_dict):
        device = self.device
        model.to(device)
        optim = AdamW(model.parameters(), lr=self.lr)
        global_idx = 0
        best_scores = {'F1': -1.0, 'EM': -1.0}
        tbx = SummaryWriter(self.save_dir)

        for epoch_num in range(self.num_epochs):
            self.log.info(f'Epoch: {epoch_num}')
            with torch.enable_grad(), tqdm(total=len(train_dataloader.dataset)) as progress_bar:
                for batch in train_dataloader:
                    optim.zero_grad()
                    model.train()
                    input_ids = batch['input_ids'].to(device)
                    attention_mask = batch['attention_mask'].to(device)
                    start_positions = batch['start_positions'].to(device)
                    end_positions = batch['end_positions'].to(device)
                    outputs = model(input_ids, attention_mask=attention_mask,
                                    start_positions=start_positions,
                                    end_positions=end_positions)
                    loss = outputs[0]
                    loss.backward()
                    optim.step()
                    progress_bar.update(len(input_ids))
                    progress_bar.set_postfix(epoch=epoch_num, NLL=loss.item())
                    tbx.add_scalar('train/NLL', loss.item(), global_idx)
                    if (global_idx % self.eval_every) == 0:
                        self.log.info(f'Evaluating at step {global_idx}...')
                        preds, curr_score = self.evaluate(model, eval_dataloader, val_dict, return_preds=True)
                        results_str = ', '.join(f'{k}: {v:05.2f}' for k, v in curr_score.items())
                        self.log.info('Visualizing in TensorBoard...')
                        for k, v in curr_score.items():
                            tbx.add_scalar(f'val/{k}', v, global_idx)
                        self.log.info(f'Eval {results_str}')
                        if self.visualize_predictions:
                            util.visualize(tbx,
                                           pred_dict=preds,
                                           gold_dict=val_dict,
                                           step=global_idx,
                                           split='val',
                                           num_visuals=self.num_visuals)
                        if curr_score['F1'] >= best_scores['F1']:
                            best_scores = curr_score
                            self.save(model)
                    global_idx += 1
        return best_scores


    def train(self, model, train_dataloader, eval_dataloader, val_dict):
        device = self.device
        model.to(device)
        optim = AdamW(model.parameters(), lr=self.lr)
        global_idx = 0
        best_scores = {'F1': -1.0, 'EM': -1.0}
        tbx = SummaryWriter(self.save_dir)

        for epoch_num in range(self.num_epochs):
            self.log.info(f'Epoch: {epoch_num}')
            with torch.enable_grad(), tqdm(total=len(train_dataloader.dataset)) as progress_bar:
                for batch in train_dataloader:
                    optim.zero_grad()
                    model.train()
                    input_ids = batch['input_ids'].to(device)
                    attention_mask = batch['attention_mask'].to(device)
                    start_positions = batch['start_positions'].to(device)
                    end_positions = batch['end_positions'].to(device)
                    outputs = model(input_ids, attention_mask=attention_mask,
                                    start_positions=start_positions,
                                    end_positions=end_positions)
                    loss = outputs[0]
                    loss.backward()
                    optim.step()
                    progress_bar.update(len(input_ids))
                    progress_bar.set_postfix(epoch=epoch_num, NLL=loss.item())
                    tbx.add_scalar('train/NLL', loss.item(), global_idx)
                    if (global_idx % self.eval_every) == 0:
                        self.log.info(f'Evaluating at step {global_idx}...')
                        preds, curr_score = self.evaluate(model, eval_dataloader, val_dict, return_preds=True)
                        results_str = ', '.join(f'{k}: {v:05.2f}' for k, v in curr_score.items())
                        self.log.info('Visualizing in TensorBoard...')
                        for k, v in curr_score.items():
                            tbx.add_scalar(f'val/{k}', v, global_idx)
                        self.log.info(f'Eval {results_str}')
                        if self.visualize_predictions:
                            util.visualize(tbx,
                                           pred_dict=preds,
                                           gold_dict=val_dict,
                                           step=global_idx,
                                           split='val',
                                           num_visuals=self.num_visuals)
                        if curr_score['F1'] >= best_scores['F1']:
                            best_scores = curr_score
                            self.save(model)
                    global_idx += 1
        return best_scores

def get_dataset(args, datasets, data_dir, tokenizer, split_name, augment_size=0,
        augment_datasets=None, augment_data_dir=None, sent_model=None):
    datasets = datasets.split(',')
    dataset_dict = None
    dataset_name=''
    for dataset in datasets:
        dataset_name += f'_{dataset}'
        dataset_dict_curr = util.read_squad(f'{data_dir}/{dataset}') # error here
        dataset_dict = util.merge(dataset_dict, dataset_dict_curr)

    augment_dataset_dicts = None
    if augment_datasets is not None:
        augment_dataset_dicts = []
        for aug_dataset in augment_datasets.split(','):
            # dataset_name += f'_{aug_dataset}'
            augment_dataset_dict_curr = util.read_squad(f'{augment_data_dir}/{aug_dataset}')
            # print("augment_dataset_dict_curr in ", augment_dataset_dict_curr)
            augment_dataset_dicts += [augment_dataset_dict_curr]
        # print("augment_dataset_dicts in get_dataset", augment_dataset_dicts) #D

    data_encodings = read_and_process(args, tokenizer, dataset_dict, data_dir, dataset_name, split_name,
                            augment_dataset_dicts=augment_dataset_dicts,
                            sent_model=sent_model) # pass in both indomain and oodomain dataset dicts
    return util.QADataset(data_encodings, train=(split_name=='train')), dataset_dict

def main():
    # define parser and arguments
    args = get_train_test_args()
    util.set_seed(args.seed)
    # if --load-checkpoint flag is True, load pretrained model from --load-dir
    # pretrained = os.path.join(args.load_dir, 'checkpoint') if args.load_checkpoint else 'distilbert-base-uncased'
    pretrained = 'distilbert-base-uncased'
    model = DistilBertForQuestionAnswering.from_pretrained(pretrained)
    # args.device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    # model.to(args.device)

    tokenizer = DistilBertTokenizerFast.from_pretrained('distilbert-base-uncased')
    special_tokens_dict = {'additional_special_tokens': ['[MASK]', '[SEP]']} # tokenizer.mask_token and mask_token_id? see .cls_token
    num_added_toks = tokenizer.add_special_tokens(special_tokens_dict)
    model.resize_token_embeddings(len(tokenizer))
    sent_model = SentenceTransformer('distilbert-base-uncased') if args.use_sent_transformer else None

    # if using small ind train datasets, change the dataset names
    if args.train_small:
        print("Using small train datasets!")
        args.train_datasets = 'squad_small,nat_questions_small,newsqa_small'

    if args.do_train:
        if not os.path.exists(args.save_dir):
            os.makedirs(args.save_dir)
        args.save_dir = util.get_save_dir(args.save_dir, args.run_name)
        log = util.get_logger(args.save_dir, 'log_train')
        log.info(f'Args: {json.dumps(vars(args), indent=4, sort_keys=True)}')
        log.info("Preparing Training Data...")
        args.device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
        trainer = Trainer(args, log)
        train_dataset, _ = get_dataset(args, args.train_datasets, args.train_dir, tokenizer, 'train') # type QADataset
        log.info("Preparing Validation Data...")
        val_dataset, val_dict = get_dataset(args, args.val_datasets, args.val_dir, tokenizer, 'val')
        train_loader = DataLoader(train_dataset,
                                batch_size=args.batch_size,
                                sampler=RandomSampler(train_dataset))
        (train_loader)
        val_loader = DataLoader(val_dataset,
                                batch_size=args.batch_size,
                                sampler=SequentialSampler(val_dataset))
        best_scores = trainer.train(model, train_loader, val_loader, val_dict)

    if args.do_finetune_ood_vanilla:
        if not os.path.exists(args.save_dir):
            os.makedirs(args.save_dir)
        args.save_dir = util.get_save_dir(args.save_dir, args.run_name)
        log = util.get_logger(args.save_dir, 'log_train')
        log.info(f'Args: {json.dumps(vars(args), indent=4, sort_keys=True)}')
        log.info("Preparing Training Data...")
        args.device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
        trainer = Trainer(args, log)

        # load saved model tuned on in-domain train set
        checkpoint_path = os.path.join(args.load_dir, 'checkpoint')
        model = DistilBertForQuestionAnswering.from_pretrained(checkpoint_path)
        model.to(args.device)

        # TODO: sample |dev| examples from ood train to augment

        val_dataset, val_dict = get_dataset(args, args.train_datasets, args.val_dir, tokenizer, 'val')
        # sample len(val_dataset) examples from augment_dataset train

        # TODO augment size wrong and unused?
        train_dataset, _ = get_dataset(args, args.finetune_datasets, args.finetune_dir, tokenizer, 'train') # type QADataset
        log.info("Preparing Validation Data...")
        train_loader = DataLoader(train_dataset,
                                batch_size=args.batch_size,
                                sampler=RandomSampler(train_dataset))
        val_loader = DataLoader(val_dataset,
                                batch_size=args.batch_size,
                                sampler=SequentialSampler(val_dataset))


        best_scores = trainer.train(model, train_loader, val_loader, val_dict)

    if args.do_train_demons:
        if not os.path.exists(args.save_dir):
            os.makedirs(args.save_dir)
        args.save_dir = util.get_save_dir(args.save_dir, args.run_name)
        log = util.get_logger(args.save_dir, 'log_train')
        log.info(f'Args: {json.dumps(vars(args), indent=4, sort_keys=True)}')
        log.info("Preparing Training Data...")
        args.device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
        trainer = Trainer(args, log)
        val_dataset, val_dict = get_dataset(args, args.val_datasets, args.val_dir, tokenizer, 'val')
        train_dataset, _ = get_dataset(args, args.train_datasets, args.train_dir, tokenizer, 'train',
                                augment_size=len(val_dataset),
                                augment_datasets=args.finetune_datasets,
                                augment_data_dir=args.finetune_dir,
                                sent_model=sent_model) # type QADataset)

        log.info("Preparing Validation Data...")
        train_loader = DataLoader(train_dataset,
                                batch_size=args.batch_size,
                                sampler=RandomSampler(train_dataset))
        (train_loader)
        val_loader = DataLoader(val_dataset,
                                batch_size=args.batch_size,
                                sampler=SequentialSampler(val_dataset))
        best_scores = trainer.train(model, train_loader, val_loader, val_dict)

    if args.do_finetune_demons:
        if not os.path.exists(args.save_dir):
            os.makedirs(args.save_dir)
        args.save_dir = util.get_save_dir(args.save_dir, args.run_name)
        log = util.get_logger(args.save_dir, 'log_train')
        log.info(f'Args: {json.dumps(vars(args), indent=4, sort_keys=True)}')
        log.info("Preparing Training Data...")
        args.device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
        trainer = Trainer(args, log)
        val_dataset, val_dict = get_dataset(args, args.val_datasets, args.val_dir, tokenizer, 'val')

        if args.augment:
            args.finetune_datasets = 'duorc_augmented,race_augmented,relation_extraction_augmented'
        train_dataset, _ = get_dataset(args, args.finetune_datasets, args.finetune_dir, tokenizer, 'train',
                                augment_size=len(val_dataset),
                                augment_datasets=args.train_datasets,
                                augment_data_dir=args.train_dir,
                                sent_model=sent_model) # type QADataset)

        log.info("Preparing Validation Data...")
        train_loader = DataLoader(train_dataset,
                                batch_size=args.batch_size,
                                sampler=RandomSampler(train_dataset))
        (train_loader)
        val_loader = DataLoader(val_dataset,
                                batch_size=args.batch_size,
                                sampler=SequentialSampler(val_dataset))

        # load saved model tuned on in-domain train set
        checkpoint_path = os.path.join(args.load_dir, 'checkpoint')
        model = DistilBertForQuestionAnswering.from_pretrained(checkpoint_path)
        model.to(args.device)

        best_scores = trainer.train(model, train_loader, val_loader, val_dict)


    if args.do_finetune:
        if not os.path.exists(args.save_dir):
            os.makedirs(args.save_dir)
        args.save_dir = util.get_save_dir(args.save_dir, args.run_name)
        log = util.get_logger(args.save_dir, 'log_train')
        log.info(f'Args: {json.dumps(vars(args), indent=4, sort_keys=True)}')
        log.info("Preparing Training Data...")
        args.device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
        trainer = Trainer(args, log)

        # TODO: sample |dev| examples from ood train to augment

        val_dataset, val_dict = get_dataset(args, args.finetune_datasets, args.finetune_dir, tokenizer, 'val') # use ood train as validation set
        # val_dataset, val_dict = get_dataset(args, args.train_datasets, args.val_dir, tokenizer, 'val')
        # sample len(val_dataset) examples from augment_dataset train

        # TODO augment size wrong and unused?
        train_dataset, _ = get_dataset(args, args.train_datasets, args.train_dir, tokenizer, 'train',
                                augment_size=len(val_dataset),
                                augment_datasets=args.finetune_datasets,
                                augment_data_dir=args.finetune_dir,
                                sent_model=None) # type QADataset



        log.info("Preparing Validation Data...")
        train_loader = DataLoader(train_dataset,
                                batch_size=args.batch_size,
                                sampler=RandomSampler(train_dataset))
        val_loader = DataLoader(val_dataset,
                                batch_size=args.batch_size,
                                sampler=SequentialSampler(val_dataset))
        best_scores = trainer.train(model, train_loader, val_loader, val_dict)

    if args.do_finetune_sentence_bert:
        if not os.path.exists(args.save_dir):
            os.makedirs(args.save_dir)
        args.save_dir = util.get_save_dir(args.save_dir, args.run_name)
        log = util.get_logger(args.save_dir, 'log_train')
        log.info(f'Args: {json.dumps(vars(args), indent=4, sort_keys=True)}')
        log.info("Preparing Training Data...")
        args.device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
        trainer = Trainer(args, log)

        # TODO: sample |dev| examples from ood train to augment

        val_dataset, val_dict = get_dataset(args, args.finetune_datasets, args.finetune_dir, tokenizer, 'val') # use ood train as validation set
        # val_dataset, val_dict = get_dataset(args, args.train_datasets, args.val_dir, tokenizer, 'val')
        # sample len(val_dataset) examples from augment_dataset train

        # TODO augment size wrong and unused?
        train_dataset, _ = get_dataset(args, args.train_datasets, args.train_dir, tokenizer, 'train',
                                augment_size=len(val_dataset),
                                augment_datasets=args.finetune_datasets,
                                augment_data_dir=args.finetune_dir,
                                sent_model=sent_model) # type QADataset



        log.info("Preparing Validation Data...")
        train_loader = DataLoader(train_dataset,
                                batch_size=args.batch_size,
                                sampler=RandomSampler(train_dataset))
        val_loader = DataLoader(val_dataset,
                                batch_size=args.batch_size,
                                sampler=SequentialSampler(val_dataset))
        best_scores = trainer.train(model, train_loader, val_loader, val_dict)

    if args.do_finetune_load_checkpoint:
        if not os.path.exists(args.save_dir):
            os.makedirs(args.save_dir)
        args.save_dir = util.get_save_dir(args.save_dir, args.run_name)
        log = util.get_logger(args.save_dir, 'log_train')
        log.info(f'Args: {json.dumps(vars(args), indent=4, sort_keys=True)}')
        log.info("Preparing Training Data...")
        args.device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
        trainer = Trainer(args, log)

        # load saved model tuned on in-domain train set
        checkpoint_path = os.path.join(args.load_dir, 'checkpoint')
        model = DistilBertForQuestionAnswering.from_pretrained(checkpoint_path)
        model.to(args.device)

        # TODO: sample |dev| examples from ood train to augment

        val_dataset, val_dict = get_dataset(args, args.train_datasets, args.val_dir, tokenizer, 'val')
        # sample len(val_dataset) examples from augment_dataset train

        # TODO augment size wrong and unused?
        train_dataset, _ = get_dataset(args, args.train_datasets, args.train_dir, tokenizer, 'train',
            augment_datasets=args.finetune_datasets, augment_data_dir=args.finetune_dir) # type QADataset
        log.info("Preparing Validation Data...")
        train_loader = DataLoader(train_dataset,
                                batch_size=args.batch_size,
                                sampler=RandomSampler(train_dataset))
        val_loader = DataLoader(val_dataset,
                                batch_size=args.batch_size,
                                sampler=SequentialSampler(val_dataset))
        best_scores = trainer.train(model, train_loader, val_loader, val_dict)


    if args.do_augment_ood:
        # create directory to save checkpoints
        if not os.path.exists(args.save_dir):
            os.makedirs(args.save_dir)
        args.save_dir = util.get_save_dir(args.save_dir, args.run_name)
        log = util.get_logger(args.save_dir, 'log_train')
        log.info(f'Args: {json.dumps(vars(args), indent=4, sort_keys=True)}')
        log.info("Preparing Training Data...")
        args.device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
        trainer = Trainer(args, log)

        # TODO: sample |dev| examples from ood train to augment
        # sample |dev examples from augment_dataset train
        # try run train indomain with --val-dir datasets/oodomain_val to use oodomain tune??
        # which val sets do we need to keep pristine and which can we use for metalearing/training hyperpameters?
        val_dataset, val_dict = get_dataset(args, args.train_datasets, args.val_dir, tokenizer, 'val')

        train_dataset, _ = get_dataset(args, args.finetune_datasets, args.finetune_dir, tokenizer, 'train',
                                augment_datasets=args.train_datasets,
                                augment_data_dir=args.train_dir,
                                sent_model=None) # type QADataset

        log.info("Preparing Validation Data...")
        train_loader = DataLoader(train_dataset,
                                batch_size=args.batch_size,
                                sampler=RandomSampler(train_dataset))
        val_loader = DataLoader(val_dataset,
                                batch_size=args.batch_size,
                                sampler=SequentialSampler(val_dataset))

        # load saved model tuned on in-domain train set
        checkpoint_path = os.path.join(args.save_dir, 'checkpoint')
        model = DistilBertForQuestionAnswering.from_pretrained(checkpoint_path)
        model.to(args.device)

        best_scores = trainer.train(model, train_loader, val_loader, val_dict)

    if args.do_eval:
        args.device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
        split_name = 'test' if 'test' in args.eval_dir else 'validation'
        log = util.get_logger(args.save_dir, f'log_{split_name}')
        trainer = Trainer(args, log)
        checkpoint_path = os.path.join(args.save_dir, 'checkpoint')
        model = DistilBertForQuestionAnswering.from_pretrained(checkpoint_path)
        model.to(args.device)
        eval_dataset, eval_dict = get_dataset(args, args.eval_datasets, args.eval_dir, tokenizer, split_name)
        eval_loader = DataLoader(eval_dataset,
                                 batch_size=args.batch_size,
                                 sampler=SequentialSampler(eval_dataset))
        eval_preds, eval_scores = trainer.evaluate(model, eval_loader,
                                                   eval_dict, return_preds=True,
                                                   split=split_name)
        results_str = ', '.join(f'{k}: {v:05.2f}' for k, v in eval_scores.items())
        log.info(f'Eval {results_str}')
        # Write submission file
        sub_path = os.path.join(args.save_dir, split_name + '_' + args.sub_file)
        log.info(f'Writing submission file to {sub_path}...')
        with open(sub_path, 'w', newline='', encoding='utf-8') as csv_fh:
            csv_writer = csv.writer(csv_fh, delimiter=',')
            csv_writer.writerow(['Id', 'Predicted'])
            for uuid in sorted(eval_preds):
                csv_writer.writerow([uuid, eval_preds[uuid]])


if __name__ == '__main__':
    main()
