import pickle
import pandas as pd
from pandas.core.accessor import register_dataframe_accessor
from transformers import Wav2Vec2CTCTokenizer
from transformers import Wav2Vec2FeatureExtractor
from transformers import Wav2Vec2Processor
# from pandas.io.parsers import read_csv
from datasets import Dataset
import random
import re
import torchaudio
import json
import librosa
import numpy as np
import pyarrow.parquet as pq
import librosa
# import wavio
import argparse
import pickle
import subprocess

def read_dt_data():
    txt_path = 'data/speech-sme-asr/dt_set.txt'
    data = pd.read_csv(txt_path, delimiter='\n', header=None, names=['path', 'sentence'])
    
    has_colon = data['path'].str.contains('|')
    data[['path', 'sentence']] = data.loc[has_colon, 'path'].str.split('|', expand=True)

    # return(data)
    # return data.to_pickle('./'+str(txt_path.split('/')[-1].split('.')[0] + '.pkl'))
    data = Dataset.from_pandas(data)
    # print(data)
    return data

def read_txt(txt_path):
    data = pd.read_csv(txt_path, delimiter='\n', header=None, names=['path', 'sentence'])
    
    has_colon = data['path'].str.contains('|')
    data[['path', 'sentence']] = data.loc[has_colon, 'path'].str.split('|', expand=True)

    # return(data)
    # return data.to_pickle('./'+str(txt_path.split('/')[-1].split('.')[0] + '.pkl'))
    data = Dataset.from_pandas(data)
    # print(data)
    return data

CHARS_TO_IGNORE = r'[\,\?\.\!\-\;\:\"\“\%\‘\”\�\$\©\~\)\(\§\'\d]'
def remove_special_characters(batch):
    batch["sentence"] = re.sub(CHARS_TO_IGNORE, '', batch["sentence"]).lower() + " "
    return batch

def extract_all_chars(batch):
  all_text = " ".join(batch["sentence"])
  vocab = list(set(all_text))
  return {"vocab": [vocab], "all_text": [all_text]}

def build_vocab_dict(train_dataset, test_dataset):
    # extract all chars
    vocab_train = train_dataset.map(extract_all_chars, batched=True, batch_size=-1, keep_in_memory=True, remove_columns=train_dataset.column_names)
    vocab_test = test_dataset.map(extract_all_chars, batched=True, batch_size=-1, keep_in_memory=True, remove_columns=test_dataset.column_names)

    vocab_list = list(set(vocab_train["vocab"][0]) | set(vocab_test["vocab"][0]))
    vocab_dict = {v: k for k, v in enumerate(vocab_list)} 
    # print(vocab_dict)
    vocab_dict["|"] = vocab_dict[" "]
    del vocab_dict[" "]
    vocab_dict["[UNK]"] = len(vocab_dict)
    vocab_dict["[PAD]"] = len(vocab_dict)
    
    return vocab_dict

def write_vocab_dict_to_disk(vocab_dict, vocab_path="vocab.json"):
    with open(vocab_path, 'w') as vocab_file:
        json.dump(vocab_dict, vocab_file)

# from load_dataset import train, test

# processor.save_pretrained('./model_output')

def speech_file_to_array_fn(batch):
    speech_array, sampling_rate = torchaudio.load('./data/' + batch["path"])
    batch["speech"] = speech_array[0].numpy()
    batch["sampling_rate"] = sampling_rate
    batch["target_text"] = batch["sentence"]
    return batch

def processor_init():
    tokenizer = Wav2Vec2CTCTokenizer("./vocab.json", unk_token="[UNK]", pad_token="[PAD]", word_delimiter_token="|")
    feature_extractor = Wav2Vec2FeatureExtractor(feature_size=1, sampling_rate=16000, padding_value=0.0, do_normalize=True, return_attention_mask=True)
    processor = Wav2Vec2Processor(feature_extractor=feature_extractor, tokenizer=tokenizer)
    processor.save_pretrained('asr_output/new_processor/')
    return processor

def prepare(reg=True, from_scratch=False):
    # load data
    test = read_txt('./data/speech-sme-asr/test_asr.txt')
    train = read_txt('./data/speech-sme-asr/train_asr.txt')

    # remove special characters
    train = train.map(remove_special_characters)
    test = test.map(remove_special_characters)

    # build vocab dict
    
    if from_scratch:
        vocab_dict = build_vocab_dict(train, test)
        write_vocab_dict_to_disk(vocab_dict)
     
        processor = processor_init()
    if reg:
        # processor = processor_init()
        processor = Wav2Vec2Processor.from_pretrained('./asr_output/pretrained_processor')
            
    def prepare_dataset(batch):
    
        # check that all files have the correct sampling rate
        assert (
            len(set(batch["sampling_rate"])) == 1
        ), f"Make sure all inputs have the same sampling rate of {processor.feature_extractor.sampling_rate}."

        batch["input_values"] = processor(batch["speech"], sampling_rate=batch["sampling_rate"][0]).input_values

        with processor.as_target_processor():
            batch["labels"] = processor(batch["target_text"]).input_ids
        return batch
   
    # speech file to array
    train = train.map(speech_file_to_array_fn, remove_columns=train.column_names)
    test = test.map(speech_file_to_array_fn, remove_columns=test.column_names)

    print("Preparing train dataset")
    train = train.map(prepare_dataset, remove_columns=train.column_names, batch_size=1, num_proc=1, batched=True)
    print("Preparing test dataset")
    test = test.map(prepare_dataset, remove_columns=test.column_names, batch_size=1, num_proc=1, batched=True)
    print("Done")
    
    pickle.dump(train,open('./data/speech-sme-asr/train_asr.pkl', 'wb'))

    pickle.dump(test, open('./data/speech-sme-asr/test_asr.pkl', 'wb') )

    return train, test

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Preprocessing for ASR')
    # parser.add_argument('--reg', '-r', default=True, help="regular run")
    parser.add_argument('--dt', '-dt', default=False, help="if preproccesing for dual transformation")


    parser.add_argument('--from_scratch', '-fs', default=False, help="if preprocession for new training. Usually shouldn't be called")
    args = parser.parse_args()
    if args.from_scratch:
        prepare(reg=False, from_scratch=True)
    if args.dt:
        read_dt_data()
    else:
        print('Running regular data loading...')
        prepare(reg=True, from_scratch=False)

    
