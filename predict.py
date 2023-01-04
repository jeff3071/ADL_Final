from metric import hamming_score,mapk,apk
from dataset import MultiLabelDataset
from util import to_taget_output
import json
import pickle
from argparse import ArgumentParser, Namespace
from pathlib import Path
from typing import Dict
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
from transformers import AutoTokenizer, AutoModelForSequenceClassification,RobertaForSequenceClassification
import logging
logging.basicConfig(level=logging.ERROR)
from torch import cuda

def main(args):
  device = 'cuda' if cuda.is_available() else 'cpu'
  type = args.type

  submit_file_name = f'{type}_submission.csv'
  test_data = pd.read_csv(f'test_{type}_data.csv').reset_index(drop=True)
  model_paths = [args.model_path]

  ensemble_output = []

  for model_path in model_paths:
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model = AutoModelForSequenceClassification.from_pretrained(model_path)
    model = model.to(device)

    test_params = {
      'batch_size': 8,
      'shuffle': False,
      'num_workers': 0
    }

    test_set = MultiLabelDataset(test_data, tokenizer, 128)
    test_loader = DataLoader(test_set, **test_params)
    print("TEST Dataset: {}".format(test_data.shape))

    model.eval()
    fin_outputs=[]
    users = []

    with torch.no_grad():
      for _, data in tqdm(enumerate(test_loader, 0)):
        ids = data['ids'].to(device, dtype = torch.long)
        mask = data['mask'].to(device, dtype = torch.long)
        token_type_ids = data['token_type_ids'].to(device, dtype = torch.long)
        targets = data['targets'].to(device, dtype = torch.float)
        users.extend(data['user'])
        outputs = model(ids, mask)
        fin_outputs.extend(torch.sigmoid((outputs.logits)).cpu().detach().numpy().tolist())

    if not ensemble_output:
      ensemble_output = fin_outputs.copy()
    else:
      for i in range(len(fin_outputs)):
        for j in range(len(fin_outputs[0])):
          ensemble_output[i][j] += fin_outputs[i][j]

  kaggle_output = to_taget_output(ensemble_output)

  print('users count:', len(users), 'ans count:', len(kaggle_output))
  with open(submit_file_name, "w", encoding="utf-8") as file:
    file.write("user_id,subgroup\n")
    if len(kaggle_output) == len(users):
      print("write predict result to csv")
      for i in range(len(users)):
        subgroup = ' '.join(str(tag) if tag else [] for tag in kaggle_output[i])
        file.write(f'{users[i]},{subgroup}\n')
    else:
      print('user\'s len != output len')

  file.close()
  print('done')

def parse_args() -> Namespace:
    parser = ArgumentParser()

    parser.add_argument("--type", type=str, default="unseen")
    parser.add_argument("--model_path", type=Path)
    parser.add_argument("--output_path", type=Path)
    parser.add_argument(
        "--device", type=torch.device, help="cpu, cuda, cuda:0, cuda:1", default="cpu"
    )
    args = parser.parse_args()
    return args


if __name__ == "__main__":
    args = parse_args()
    main(args)
