import pandas as pd
from datasets import Dataset
import random
import re
import torchaudio
import json
import librosa
import numpy as np
from transformers import Wav2Vec2Processor, Wav2Vec2CTCTokenizer, Wav2Vec2FeatureExtractor

def create_processor(path: str = "./vocab.json"):
    tokenizer = Wav2Vec2CTCTokenizer(path, unk_token="[UNK]", pad_token="[PAD]", word_delimiter_token="|")
    feature_extractor = Wav2Vec2FeatureExtractor(feature_size=1, sampling_rate=16000, padding_value=0.0, do_normalize=True, return_attention_mask=True)
    processor = Wav2Vec2Processor(feature_extractor=feature_extractor, tokenizer=tokenizer)
    return processor

def read_txt(txt_path):
    data = pd.read_csv(txt_path, delimiter='\n', header=None, names=['path', 'sentence'])

    has_colon = data['path'].str.contains('|')
    data[['path', 'sentence']] = data.loc[has_colon, 'path'].str.split('|', expand=True)

    data = Dataset.from_pandas(data)

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

def speech_file_to_array_fn(batch):
    speech_array, sampling_rate = torchaudio.load('data/speech-sme-main/' + batch["path"])
    batch["speech"] = speech_array[0].numpy()
    batch["sampling_rate"] = sampling_rate
    batch["target_text"] = batch["sentence"]
    return batch

def dt_speech_file_to_array_fn(batch):
    speech_array, sampling_rate = torchaudio.load(batch["path"])
    batch["speech"] = speech_array[0].numpy()
    batch["sampling_rate"] = sampling_rate
    batch["target_text"] = batch["sentence"]
    return batch

def resample(batch):
    batch["speech"] = librosa.resample(np.asarray(batch["speech"]), 48_000, 16_000)
    batch["sampling_rate"] = 16_000
    return batch

def prepare_dt_data(data):
    data_df = pd.DataFrame(data, columns=['path', 'sentence'])
    data = Dataset.from_pandas(data_df)
    # print(data)
    data = data.map(remove_special_characters)
    data = data.map(dt_speech_file_to_array_fn, remove_columns=data.column_names)
    data = data.map(resample, num_proc=4)
    processor = create_processor()

    def prepare_dataset(batch):
        # check that all files have the correct sampling rate
        assert (
            len(set(batch["sampling_rate"])) == 1
        ), f"Make sure all inputs have the same sampling rate of {processor.feature_extractor.sampling_rate}."

        batch["input_values"] = processor(batch["speech"], sampling_rate=batch["sampling_rate"][0]).input_values

        with processor.as_target_processor():
            batch["labels"] = processor(batch["target_text"]).input_ids
        return batch
        
    data = data.map(prepare_dataset, remove_columns=data.column_names, batch_size=4, num_proc=4, batched=True)

    return data

def prepare_test_dt():
    # load data
    test_raw = read_txt('data/speech-sme-main/test.txt')
 
    # remove special characters
    test = test_raw.map(remove_special_characters)
    test = test.map(speech_file_to_array_fn, remove_columns=test.column_names)

    # resample
    test = test.map(resample, num_proc=4)
    
    return test_raw, test

def prepare_data():
    # load data
    test = read_txt('data/speech-sme-main/test.txt')
    train = read_txt('data/speech-sme-main/train.txt')
 
    # remove special characters
    train = train.map(remove_special_characters)
    test = test.map(remove_special_characters)
    

    return train, test

def prepare_asr_data():
    train, test = prepare_data()

    # build vocab dict
    # vocab_dict = build_vocab_dict(train, test)
    # write_vocab_dict_to_disk(vocab_dict)

    # speech file to array
    train = train.map(speech_file_to_array_fn, remove_columns=train.column_names)
    test = test.map(speech_file_to_array_fn, remove_columns=test.column_names)

    # resample
    train = train.map(resample, num_proc=4)
    test = test.map(resample, num_proc=4)

    # rand_int = random.randint(0, len(train))

    # print("Target text:", train[rand_int]["target_text"])
    # print("Input array shape:", np.asarray(train[rand_int]["speech"]).shape)
    # print("Sampling rate:", train[rand_int]["sampling_rate"])

    processor = create_processor()
    def prepare_dataset(batch):
        # check that all files have the correct sampling rate
        assert (
            len(set(batch["sampling_rate"])) == 1
        ), f"Make sure all inputs have the same sampling rate of {processor.feature_extractor.sampling_rate}."

        batch["input_values"] = processor(batch["speech"], sampling_rate=batch["sampling_rate"][0]).input_values

        with processor.as_target_processor():
            batch["labels"] = processor(batch["target_text"]).input_ids
        return batch

    # print("Prepare train dataset")
    train = train.map(prepare_dataset, remove_columns=train.column_names, batch_size=4, num_proc=4, batched=True)
    # print("Prepare test dataset")
    test = test.map(prepare_dataset, remove_columns=test.column_names, batch_size=4, num_proc=4, batched=True)
    # print("Done")

    return train, test
