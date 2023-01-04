import torch
import numpy as np
import pandas as pd

def to_taget_output(l):
  res = []
  for single_output in l:
    s = []
    output = single_output.copy()
    # output_sort.sort(reverse=True)
    sorted_s = sorted(range(len(output)), key = lambda k : -output[k])
    output_len = 0
    for i in output:
      if i != 0:
        output_len+=1
    sorted_output = sorted_s[:output_len]
    sorted_output = [i+1 for i in sorted_output]
    res.append(sorted_output)
  return res

def same_seeds(seed):
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)  
    np.random.seed(seed)  
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def get_id2text(document_csv_path):
  doc_id_to_text = {}
  doc_df = pd.read_csv(document_csv_path)
  id_text_pair = zip(doc_df["subgroup_id"], doc_df["subgroup_name"])
  for i, pair in enumerate(id_text_pair, start=1):
      doc_id, doc_text = pair
      doc_id_to_text[str(doc_id)] = doc_text
      
      print("Progress: %d/%d\r" % (i, len(doc_df)), end='')
  doc_id_to_text['0'] = "NULL"
  return doc_id_to_text