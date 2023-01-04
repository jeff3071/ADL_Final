from torch.utils.data import Dataset, DataLoader, RandomSampler, SequentialSampler
import torch

class MultiLabelDataset(Dataset):

    def __init__(self, dataframe, tokenizer, max_len):
        self.tokenizer = tokenizer
        self.data = dataframe
        self.text = dataframe.text
        self.targets = self.data.labels
        self.max_len = max_len
        self.user_id = dataframe.user_id

    def __len__(self):
        return len(self.text)

    def __getitem__(self, index):
        text = self.text[index]
        user = self.user_id[index]
        inputs = self.tokenizer.encode_plus(
            str(text),
            None,
            add_special_tokens=True,
            max_length=self.max_len,
            pad_to_max_length=True,
            return_token_type_ids=True
        )
        ids = inputs['input_ids']
        mask = inputs['attention_mask']
        token_type_ids = inputs["token_type_ids"]

        return {
            'ids': torch.tensor(ids, dtype=torch.long),
            'mask': torch.tensor(mask, dtype=torch.long),
            'token_type_ids': torch.tensor(token_type_ids, dtype=torch.long),
            'targets': torch.tensor(eval(self.targets[index]), dtype=torch.float),
            'user': user
        }

