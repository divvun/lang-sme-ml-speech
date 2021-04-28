from utils.files import get_files
from pathlib import Path
from typing import Union


def ljspeech(path: Union[str, Path], dt=False):
    csv_file = get_files(path, extension='.csv')
  
    if dt:
        for f in csv_file:
            if str(f).endswith('tmp_tts_train.csv'):
                csv_file = [f]
    else:
        for f in csv_file:
            if str(f).endswith('/cleaned_train_tts.csv'):
                csv_file = [f]
       
    text_dict = {}
    print(f'Using {csv_file} train file ...')
    # exit()
    with open(csv_file[0], encoding='utf-8') as f :
        for line in f :
            split = line.split('|')
            # print(split)
            text_dict[split[0]] = split[-1]
    # print(len(text_dict))
    # exit()
    return text_dict