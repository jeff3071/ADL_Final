## ADL Final

利用Roberta、Distilbert決定User的Group(multi-label classification)

### Preprocess

```bash
python preprocess.py [--data_dir] #default: ./data/
```

### Training

```bash
# unseen
python train.py --model_name hfl/chinese-roberta-wwm-ext --model_output_path ./models/roberta_data_clean

# seen
python train.py --model_name distilbert-base-multilingual-cased --model_output_path ./models/distilbert_data_clean
```

### Testing

```bash
bash predict.sh \
  [--output_path]
```

### Reproduce my result

```bash
bash download.sh
python preprocess.py [--data_dir]
bash predict.sh
```