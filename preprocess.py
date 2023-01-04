import pandas as pd
from argparse import ArgumentParser, Namespace
from pathlib import Path

def generate_train_data(dataframe, data_name):
  def tolist(s):
    target = [0]*91
    if str(s) == 'nan':
      # target[-1] = 1
      return target

    s_list = str(s).split(' ')
    

    for i in s_list:
      target[int(i)-1] = 1
    return target

  dataframe['labels'] = dataframe['subgroup'].map(tolist)

  def createtextlist(row):
    row["occupation_titles"] = str(row["occupation_titles"])
    row["recreation_names"] = str(row["recreation_names"])
    row["interests"] = str(row["interests"])
    interest_type_list = []
    interest_detail_list = []
    if row["interests"] != 'nan':
      for interest in row["interests"].split(","):
        interest= interest.split("_")
        interest_type, interest_detail = interest[0], interest[1]
        if interest_type not in interest_type_list:
          interest_type_list.append(interest_type)
        interest_detail_list.append(interest_detail)
    

    res = (str(row["occupation_titles"]) + ','.join(interest_detail_list)+ ','.join(interest_type_list)).replace("nan", "")
    return res

    # return str(row['gender']) + '\n' + str(row["occupation_titles"]) + '\n' + str(row["interests"]) + '\n' + str(row['recreation_names'])


  dataframe['text'] = dataframe.apply(lambda row: createtextlist(row), axis=1)
  dataframe = dataframe[['user_id','text','labels']]
  dataframe.to_csv(f'{data_name}', index=False)


def parse_args() -> Namespace:
    parser = ArgumentParser()
    parser.add_argument(
        "--data_dir",
        type=Path,
        help="Directory to the dataset.",
        default="./data/",
    )

    args = parser.parse_args()
    return args

if __name__ == "__main__":
  args = parse_args()
  data_dir = args.data_dir
  users = pd.read_csv(data_dir / 'users.csv')
  train = pd.read_csv(data_dir / 'train.csv')

  train_group = pd.read_csv(data_dir /'train_group.csv')
  train_data_group = pd.merge(users, train_group, on='user_id')
  generate_train_data(train_data_group, 'train_data.csv')

  val_seen_group = pd.read_csv(data_dir / "val_seen_group.csv")
  print('val_seen_group: ', len(val_seen_group))
  val_data_group = pd.merge(users, val_seen_group, on='user_id')
  generate_train_data(val_data_group, 'val_seen_data.csv')

  test_seen_group = pd.read_csv(data_dir / "test_seen_group.csv")
  print('test_seen_group: ', len(test_seen_group))
  test_data_group = pd.merge(users, test_seen_group, on='user_id')
  generate_train_data(test_data_group,'test_seen_data.csv')

  val_unseen_group = pd.read_csv(data_dir / "val_unseen_group.csv")
  print('val_unseen_group: ', len(val_unseen_group))
  val_unseendata_group = pd.merge(users, val_unseen_group, on='user_id')
  generate_train_data(val_unseendata_group,'val_unseen_data.csv')

  test_unseen_group = pd.read_csv(data_dir / "test_unseen_group.csv")
  print('test_unseen_group:', len(test_unseen_group))
  test_unseendata_group = pd.merge(users, test_unseen_group, on='user_id')
  generate_train_data(test_unseendata_group,'test_unseen_data.csv')