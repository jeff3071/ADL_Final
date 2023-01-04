# Importing stock ml libraries
import warnings
warnings.simplefilter('ignore')
import numpy as np
import pandas as pd
from tqdm import tqdm
from sklearn import metrics
import transformers
import torch
from torch.utils.data import Dataset, DataLoader, RandomSampler, SequentialSampler
from transformers import AutoTokenizer, DistilBertModel, AutoModelForMaskedLM
from transformers import AutoTokenizer, AutoModelForSequenceClassification,RobertaForSequenceClassification,BloomForSequenceClassification
import logging
from torch import cuda
from metric import hamming_score,mapk,apk
from dataset import MultiLabelDataset
from util import to_taget_output, same_seeds
from argparse import ArgumentParser, Namespace
from pathlib import Path

def main():
    same_seeds(123)
    MAX_LEN = 384
    TRAIN_BATCH_SIZE = 8
    VALID_BATCH_SIZE = 8
    LEARNING_RATE = 1e-05
    EPOCH = 5
    output_file = args.model_output_path
    model_name = args.model_name

    logger = logging.getLogger()
    logger.setLevel(logging.DEBUG)
    formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s: - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S')

    file = open(f'{output_file}log.txt')
    fh = logging.FileHandler(f'{output_file}log.txt')
    fh.setLevel(logging.DEBUG)
    fh.setFormatter(formatter)
    ch = logging.StreamHandler()
    ch.setLevel(logging.DEBUG)
    ch.setFormatter(formatter)
    logger.addHandler(ch)
    logger.addHandler(fh)


    device = 'cuda' if cuda.is_available() else 'cpu'

    train_data = pd.read_csv('train_data.csv').reset_index(drop=True)
    val_seen_data = pd.read_csv('val_seen_data.csv').reset_index(drop=True)
    test_seen_data = pd.read_csv('test_seen_data.csv').reset_index(drop=True)
    val_unseen_data = pd.read_csv('val_unseen_data.csv').reset_index(drop=True)
    test_unseen_data = pd.read_csv('test_unseen_data.csv').reset_index(drop=True)

    tokenizer = AutoTokenizer.from_pretrained(model_name , truncation=True, model_max_length =MAX_LEN)

    logger.info("TRAIN Dataset: {}".format(train_data.shape))
    logger.info("VAL seen Dataset: {}".format(val_seen_data.shape))
    logger.info("TEST seen Dataset: {}".format(test_seen_data.shape))
    logger.info("VAL unseen Dataset: {}".format(val_unseen_data.shape))
    logger.info("TEST unseen Dataset: {}".format(test_unseen_data.shape))

    training_set = MultiLabelDataset(train_data, tokenizer, MAX_LEN)
    val_seen_set = MultiLabelDataset(val_seen_data, tokenizer, MAX_LEN)
    val_unseen_set = MultiLabelDataset(val_unseen_data, tokenizer, MAX_LEN)


    train_params = {'batch_size': TRAIN_BATCH_SIZE,
                    'shuffle': True,
                    'num_workers': 0
                    }

    test_params = {'batch_size': VALID_BATCH_SIZE,
                    'shuffle': True,
                    'num_workers': 0
                    }

    training_loader = DataLoader(training_set, **train_params)
    val_seen_loader = DataLoader(val_seen_set, **test_params)
    val_unseen_loader = DataLoader(val_unseen_set, **test_params)

    model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=91, problem_type="multi_label_classification",ignore_mismatched_sizes=True)

    model.to(device)

    optimizer = torch.optim.AdamW(params =  model.parameters(), lr=LEARNING_RATE)
    # optimizer = Ranger21(model.parameters(), lr = LEARNING_RATE, num_epochs = EPOCH, num_batches_per_epoch = len(training_loader))
    best_score = 0

    for epoch in range(EPOCH):
        now_score = 0
        model.train()
        for _,data in tqdm(enumerate(training_loader)):
            ids = data['ids'].to(device, dtype = torch.long)
            mask = data['mask'].to(device, dtype = torch.long)
            token_type_ids = data['token_type_ids'].to(device, dtype = torch.long)
            targets = data['targets'].to(device, dtype = torch.float)
            #outputs = model(ids, mask, token_type_ids, labels = targets)

            outputs = model(input_ids = ids, attention_mask = mask, labels = targets)

            optimizer.zero_grad()
            loss = outputs.loss
            if _%2500==0 or _==len(training_loader)-1:
                logger.info(f'Epoch: {epoch+1}, Loss:  {loss.item()}')
            
            loss.backward()
            optimizer.step()

        model.eval()
        fin_targets=[]
        fin_outputs=[]

        with torch.no_grad():
            for _, data in tqdm(enumerate(val_seen_loader, 0)):
                ids = data['ids'].to(device, dtype = torch.long)
                mask = data['mask'].to(device, dtype = torch.long)
                token_type_ids = data['token_type_ids'].to(device, dtype = torch.long)
                targets = data['targets'].to(device, dtype = torch.float)

                outputs = model(input_ids = ids, attention_mask = mask, labels = targets)
                fin_targets.extend(targets.cpu().detach().numpy().tolist())
                fin_outputs.extend(torch.sigmoid((outputs.logits)).cpu().detach().numpy().tolist())

        kaggle_output, kaggle_target = [], []
        kaggle_output = to_taget_output(fin_outputs)
        kaggle_target = to_taget_output(fin_targets)
        score = mapk(kaggle_target, kaggle_output)
        logger.info(f"mapk seen Score = {score}")

        now_score += score

        fin_targets=[]
        fin_outputs=[]

        with torch.no_grad():
            for _, data in tqdm(enumerate(val_unseen_loader, 0)):
                ids = data['ids'].to(device, dtype = torch.long)
                mask = data['mask'].to(device, dtype = torch.long)
                token_type_ids = data['token_type_ids'].to(device, dtype = torch.long)
                targets = data['targets'].to(device, dtype = torch.float)

                outputs = model(input_ids = ids, attention_mask = mask, labels = targets)
                fin_targets.extend(targets.cpu().detach().numpy().tolist())
                fin_outputs.extend(torch.sigmoid((outputs.logits)).cpu().detach().numpy().tolist())


        kaggle_output, kaggle_target = [], []
        kaggle_output= to_taget_output(fin_outputs)
        kaggle_target= to_taget_output(fin_targets)
        score = mapk(kaggle_target, kaggle_output)
        logger.info(f"mapk unseen Score = {score}")

        now_score += score
        now_score /= 2
        if now_score > best_score:
            best_score = now_score
            model.save_pretrained(output_file)
            tokenizer.save_pretrained(output_file)
            logger.info('Saved')

def parse_args() -> Namespace:
    parser = ArgumentParser()

    parser.add_argument("--model_name", type=Path)
    parser.add_argument("--model_output_path", type=Path)
    args = parser.parse_args()
    return args


if __name__ == "__main__":
    args = parse_args()
    main(args)
