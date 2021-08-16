# from load_dataset import read_txt
# from transformers import Wav2Vec2ForCTC, Wav2Vec2Processor, Trainer, TrainingArguments
import torch
# from extract_features import processor, tokenizer
# from datasets import load_metric
import torchaudio
import random
import pandas as pd
from trainer_asr import DataCollatorCTCWithPadding
# from trainer import trainer
import numpy as np
import re
import torch
import torchaudio
from datasets import load_dataset, load_metric, Dataset
from transformers import Wav2Vec2ForCTC, Wav2Vec2Processor
import pickle

def main():
    def read_txt(txt_path):
        data = pd.read_csv(txt_path, delimiter='\n', header=None, names=['path', 'sentence'])
        
        has_colon = data['path'].str.contains('|')
        data[['path', 'sentence']] = data.loc[has_colon, 'path'].str.split('|', expand=True)

        data = Dataset.from_pandas(data)
        return(data)

    # test = pd.read_parquet('./data/speech-sme-asr/test_asr.parquet')
    # test = Dataset.from_pandas(test)
    # with open('./data/speech-sme-asr/test_asr.pkl', 'rb') as f:
    #     test = pickle.load(f)
        # print(test.shape)
    #     # exit()
    test = read_txt('./data/speech-sme-asr/test_asr.txt')
    processor = Wav2Vec2Processor.from_pretrained('./asr_output/pretrained_processor')
    model = Wav2Vec2ForCTC.from_pretrained("checkpoints/sme_speech_tts.asr_forward/checkpoint-27364").to('cpu')
    # print(model)
    # exit()
    # resampler = torchaudio.transforms.Resample(48_000, 16_000)
    CHARS_TO_IGNORE = r'[\,\?\.\!\-\;\:\"\“\%\‘\”\�\$\©\~\)\(\§\'\d]'
    def remove_special_characters(batch):
        batch["sentence"] = re.sub(CHARS_TO_IGNORE, '', batch["sentence"]).lower() + " "
        return batch
    test = test.map(remove_special_characters)

    def speech_file_to_array_fn(batch):
        speech_array, sampling_rate = torchaudio.load('./data/'+ batch["path"])
        batch["speech"] = speech_array.squeeze().numpy()
        return batch

    test = test.map(speech_file_to_array_fn)
    input_dict = processor(test['speech'][:11],sampling_rate=16000, return_tensors="pt", padding=True)
    with torch.no_grad():
        logits = model(input_dict.input_values.to('cpu')).logits


    predicted_ids = torch.argmax(logits, dim=-1)

    print("Prediction:", processor.batch_decode(predicted_ids))
    print("Reference:", test["sentence"][:11])
    # exit()
    wer = load_metric("wer")

    # resampler = torchaudio.transforms.Resample(48_000, 16_000)

    
    # # Preprocessing the datasets.
    # # We need to read the audio files as arrays
    def evaluate(batch):
        inputs = processor(batch["speech"], sampling_rate=16_000, return_tensors="pt", padding=True)

        with torch.no_grad():
            logits = model(inputs.input_values.to('cpu'), attention_mask=inputs.attention_mask.to('cpu')).logits

        pred_ids = torch.argmax(logits, dim=-1)
        batch["pred_strings"] = processor.batch_decode(pred_ids)
        return batch

    result = test.map(evaluate, batched=True, batch_size=8) # batch_size=8 -> requires ~14.5GB GPU memory

    print("WER: {:2f}".format(100 * wer.compute(predictions=result["pred_strings"], references=result["sentence"])))

if __name__ =='__main__':
    main()

