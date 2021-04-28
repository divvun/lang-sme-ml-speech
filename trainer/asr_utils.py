# from load_dataset import read_txt
# from transformers import Wav2Vec2ForCTC, Wav2Vec2Processor, Trainer, TrainingArguments
from os import getcwd
import torch
# from extract_features import processor, tokenizer
# from datasets import load_metric
import torchaudio
import random
import pandas as pd
# from trainer import DataCollatorCTCWithPadding
# from trainer import trainer
import numpy as np
import re
import torch
import torchaudio
from datasets import load_metric, Dataset
from transformers import Wav2Vec2ForCTC, Wav2Vec2Processor

def evaluate_asr():
    def read_txt(txt_path):
        data = pd.read_csv(txt_path, delimiter='\n', header=None, names=['path', 'sentence'])
        
        has_colon = data['path'].str.contains('|')
        data[['path', 'sentence']] = data.loc[has_colon, 'path'].str.split('|', expand=True)

        data = Dataset.from_pandas(data)
        return(data)

    test = read_txt('./data/speech-sme-asr/test_asr.txt')
    processor = Wav2Vec2Processor.from_pretrained("asr_output/pretrained_processor")
    # print(processor.__dict__)
    # print(processor.tokenizer)

    # exit()
    model = Wav2Vec2ForCTC.from_pretrained("asr_output/checkpoint-27363").to("cpu")
    # print(model)
    # exit()
    # resampler = torchaudio.transforms.Resample(new_freq=16_000)

    def speech_file_to_array_fn(batch):
        speech_array, sampling_rate = torchaudio.load('./data/'+ batch["path"])
        batch["speech"] = speech_array[0].numpy()
        return batch

    test_dataset = test.map(speech_file_to_array_fn)
    input_dict = processor(test_dataset['speech'][:11],sampling_rate=16000, return_tensors="pt", padding=True)
    with torch.no_grad():
        logits = model(input_dict.input_values.to("cpu")).logits


    predicted_ids = torch.argmax(logits, dim=-1)

    print("Prediction:", processor.batch_decode(predicted_ids))
    print("Reference:", test_dataset["sentence"][:11])

    wer = load_metric("wer")

    resampler = torchaudio.transforms.Resample(48_000, 16_000)

    def evaluate_batch(batch):
        inputs = processor(batch["speech"], sampling_rate=16_000, return_tensors="pt", padding=True)

        with torch.no_grad():
            logits = model(inputs.input_values.to("cpu"), attention_mask=inputs.attention_mask.to("cpu")).logits

        pred_ids = torch.argmax(logits, dim=-1)
        batch["pred_strings"] = processor.batch_decode(pred_ids)
        return batch

    result = test_dataset.map(evaluate_batch, batched=True, batch_size=8) # batch_size=8 -> requires ~14.5GB GPU memory

    msg = "WER: {:2f}".format(100 * wer.compute(predictions=result["pred_strings"], references=result["sentence"]))
    print(msg)
    return msg


if __name__ =='__main__':
    evaluate_asr()

