from typing import Dict, List, Optional, Union
from utils import checkpoints
import torch
from torch import nn
from typing import Any, Dict, List, Optional, Union

import torch.nn.functional as F
from torch.utils.data.dataloader import DataLoader
# from torch.cuda.amp import autocast

# import warnings
from transformers import Trainer, TrainingArguments,Wav2Vec2ForCTC
from dataclasses import dataclass, field
from datasets import load_metric,Dataset

from transformers.models.wav2vec2.processing_wav2vec2 import Wav2Vec2Processor

# from models.sme_asr import create_model
# from utils.load_asr_dataset import create_processor
# from utils.load_asr_dataset import prepare_asr_data
import numpy as np
import pandas as pd
import re 
import subprocess
import torchaudio
import librosa

from process_for_asr import read_dt_data

def create_model(path):
    model = Wav2Vec2ForCTC.from_pretrained(path)
    return model

# model = Wav2Vec2ForCTC()
processor = Wav2Vec2Processor.from_pretrained('./asr_output/pretrained_processor')

def load_dt_data(data):
    data_df = pd.DataFrame(data, columns=['path', 'sentence'])
    data = Dataset.from_pandas(data_df)

    CHARS_TO_IGNORE = r'[\,\?\.\!\-\;\:\"\“\%\‘\”\�\$\©\~\)\(\§\'\d]'
    def remove_special_characters(batch):
        batch["sentence"] = re.sub(CHARS_TO_IGNORE, '', batch["sentence"]).lower() + " "
        return batch
    def dt_speech_file_to_array_fn(batch):
        speech_array, sampling_rate = torchaudio.load(batch["path"])
        batch["speech"] = speech_array[0].numpy()
        batch["sampling_rate"] = sampling_rate
        batch["target_text"] = batch["sentence"]
        return batch
    def resample(batch):
        batch["speech"] = librosa.resample(np.asarray(batch["speech"]), 22_050, 16_000)
        batch["sampling_rate"] = 16_000
        return batch

    # print(data)
    data = data.map(remove_special_characters)
    data = data.map(dt_speech_file_to_array_fn, remove_columns=data.column_names)
    data = data.map(resample, num_proc=4)
    # processor = create_processor()

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



def asr_predict_for_dt(model):
    asr_test = read_dt_data()

    asr_pr = processor
    paths = asr_test['path']
    # model = Wav2Vec2ForCTC.from_pretrained('./asr_output/checkpoint-27363')
    model.to('cuda')

    resampler = torchaudio.transforms.Resample(new_freq=16_000)
    def speech_file_to_array_fn(batch):
        # print(batch['path'])
        speech_array, sampling_rate = torchaudio.load('./data/'+ batch["path"])
        batch["speech"] = resampler(speech_array).squeeze().numpy()
        return batch

    asr_test = asr_test.map(speech_file_to_array_fn)
    input_dict = asr_pr(asr_test['speech'], sampling_rate=16000, return_tensors="pt", padding=True).to('cuda')
    with torch.no_grad():
        logits = model(input_dict.input_values.to('cuda')).logits

    predicted_ids = torch.argmax(logits, dim=-1)
    preds = asr_pr.batch_decode(predicted_ids)
    # print("Prediction:", asr_pr.batch_decode(predicted_ids))
    # print("Reference:", asr_test["sentence"])
    lines = []
    for path, pred in zip(paths, preds):
        # print(pred)
        lines.append(path.split('/')[-1].split('.')[0] + '|' + pred+'\n')
    # print(lines)
    with open('./data/speech-sme-tts/tmp_tts_train.csv', 'w') as f:
        print("writing dt training file...")
        f.writelines(lines)
    # model.to('cuda')

class CustomTrainer(Trainer):
    def training_step(self, model: nn.Module, inputs: Dict[str, Union[torch.Tensor, Any]]) -> torch.Tensor:
        """
        Perform a training step on a batch of inputs.
        Subclass and override to inject custom behavior.
        Args:
            model (:obj:`nn.Module`):
                The model to train.
            inputs (:obj:`Dict[str, Union[torch.Tensor, Any]]`):
                The inputs and targets of the model.
                The dictionary will be unpacked before being fed to the model. Most models expect the targets under the
                argument :obj:`labels`. Check your model's documentation for all accepted arguments.
        Return:
            :obj:`torch.Tensor`: The tensor with training loss on this batch.
        """

        model.train()
        inputs = self._prepare_inputs(inputs)

        # if self.use_amp:
        #     with autocast():
        #         loss = self.compute_loss(model, inputs)
        # else:
        loss = self.compute_loss(model, inputs)

        if self.args.n_gpu > 1:
            if model.module.config.ctc_loss_reduction == "mean":
                loss = loss.mean()
            elif model.module.config.ctc_loss_reduction == "sum":
                loss = loss.sum() / (inputs["labels"] >= 0).sum()
            else:
                raise ValueError(f"{model.config.ctc_loss_reduction} is not valid. Choose one of ['mean', 'sum']")

        if self.args.gradient_accumulation_steps > 1:
            loss = loss / self.args.gradient_accumulation_steps

        if self.use_amp:
            self.scaler.scale(loss).backward()
        # elif self.use_apex:
        #     with amp.scale_loss(loss, self.optimizer) as scaled_loss:
        #         scaled_loss.backward()
        elif self.deepspeed:
            self.deepspeed.backward(loss)
        else:
            loss.backward()

        return loss.detach()

@dataclass
class DataCollatorCTCWithPadding:
    """
    Data collator that will dynamically pad the inputs received.
    Args:
        processor (:class:`~transformers.Wav2Vec2Processor`)
            The processor used for proccessing the data.
        padding (:obj:`bool`, :obj:`str` or :class:`~transformers.tokenization_utils_base.PaddingStrategy`, `optional`, defaults to :obj:`True`):
            Select a strategy to pad the returned sequences (according to the model's padding side and padding index)
            among:
            * :obj:`True` or :obj:`'longest'`: Pad to the longest sequence in the batch (or no padding if only a single
              sequence if provided).
            * :obj:`'max_length'`: Pad to a maximum length specified with the argument :obj:`max_length` or to the
              maximum acceptable input length for the model if that argument is not provided.
            * :obj:`False` or :obj:`'do_not_pad'` (default): No padding (i.e., can output a batch with sequences of
              different lengths).
        max_length (:obj:`int`, `optional`):
            Maximum length of the ``input_values`` of the returned list and optionally padding length (see above).
        max_length_labels (:obj:`int`, `optional`):
            Maximum length of the ``labels`` returned list and optionally padding length (see above).
        pad_to_multiple_of (:obj:`int`, `optional`):
            If set will pad the sequence to a multiple of the provided value.
            This is especially useful to enable the use of Tensor Cores on NVIDIA hardware with compute capability >=
            7.5 (Volta).
    """

    processor: processor
    padding: Union[bool, str] = True
    max_length: Optional[int] = None
    max_length_labels: Optional[int] = None
    pad_to_multiple_of: Optional[int] = None
    pad_to_multiple_of_labels: Optional[int] = None

    def __call__(self, features: List[Dict[str, Union[List[int], torch.Tensor]]]) -> Dict[str, torch.Tensor]:
        # split inputs and labels since they have to be of different lenghts and need
        # different padding methods
        input_features = [{"input_values": feature["input_values"]} for feature in features]
        label_features = [{"input_ids": feature["labels"]} for feature in features]

        batch = self.processor.pad(
            input_features,
            padding=self.padding,
            max_length=self.max_length,
            pad_to_multiple_of=self.pad_to_multiple_of,
            return_tensors="pt",
        )
        with self.processor.as_target_processor():
            labels_batch = self.processor.pad(
                label_features,
                padding=self.padding,
                max_length=self.max_length_labels,
                pad_to_multiple_of=self.pad_to_multiple_of_labels,
                return_tensors="pt",
            )

        # replace padding with -100 to ignore loss correctly
        labels = labels_batch["input_ids"].masked_fill(labels_batch.attention_mask.ne(1), -100)

        batch["labels"] = labels

        return batch


data_collator = DataCollatorCTCWithPadding(processor=processor, padding=True)
wer_metric = load_metric("wer")

def compute_metrics(pred):
    pred_logits = pred.predictions
    pred_logits = pred_logits.cpu()
    pred_ids = np.argmax(pred_logits, axis=-1)

    pred.label_ids[pred.label_ids == -100] = processor.tokenizer.pad_token_id

    pred_str = processor.batch_decode(pred_ids)
    # we do not want to group tokens when computing the metrics
    label_str = processor.batch_decode(pred.label_ids, group_tokens=False)

    wer = wer_metric.compute(predictions=pred_str, references=label_str)

    return {"wer": wer}

training_args = TrainingArguments(
        output_dir="model_output/",
        # group_by_length=True,
        per_device_train_batch_size=8,
        gradient_accumulation_steps=2,
        evaluation_strategy="steps",
        num_train_epochs=30,
        # fp16=True,
        save_steps=100,
        eval_steps=500,
        logging_steps=500,
        learning_rate=1e-4,
        weight_decay=0.005,
        # learning_rate=3e-4,
        warmup_steps=1000,
        save_total_limit=2,
    )
def init_trainer(train, test):
    trainer = CustomTrainer(
        model=create_model('./checkpoints/sme_speech_tts.asr_forward/checkpoint-27363'),
        data_collator=data_collator,
        args=training_args,
        compute_metrics=compute_metrics,
        train_dataset=train,
        eval_dataset=test,
        tokenizer=processor.feature_extractor,
    )
    return trainer



class ForwardSession:

    def __init__(self,
                 path,
                 index: int,
                 r: int,
                 lr: int,
                 max_step: int,
                 bs: int,
                 train_set: DataLoader,
                 val_set: DataLoader) -> None:

        self.path = path
        self.index = index
        self.r = r
        self.lr = lr
        self.max_step = max_step
        self.bs = bs
        self.train_set = train_set
        self.val_set = val_set
        self.val_sample = next(iter(val_set))

class TTSSession:

    def __init__(self,
                 index: int,
                 r: int,
                 lr: int,
                 max_step: int,
                 bs: int,
                 train_set: DataLoader,
                 val_set: DataLoader) -> None:

        self.index = index
        self.r = r
        self.lr = lr
        self.max_step = max_step
        self.bs = bs
        self.train_set = train_set
        self.val_set = val_set
        self.val_sample = next(iter(val_set))


class ASRSession:

    def __init__(self,
                 processor,
                 index: int,
                 r: int,
                 lr: int,
                 max_step: int,
                 bs: int,
                 train_set: DataLoader,
                 test_set: DataLoader) -> None:
        self.processor = processor
        self.index = index
        self.r = r
        self.lr = lr
        self.max_step = max_step
        self.bs = bs
        self.train_set = train_set
        self.test_set = test_set
        # self.val_sample = next(iter(val_set))


class VocSession:

    def __init__(self,
                 index: int,
                 lr: int,
                 max_step: int,
                 bs: int,
                 train_set: DataLoader,
                 val_set: list,
                 val_set_samples: list) -> None:
        self.index = index
        self.lr = lr
        self.max_step = max_step
        self.bs = bs
        self.train_set = train_set
        self.val_set = val_set
        self.val_set_samples = val_set_samples


class Averager:

    def __init__(self):
        self.count = 0
        self.val = 0.

    def add(self, val):
        self.val += float(val)
        self.count += 1

    def reset(self):
        self.val = 0.
        self.count = 0

    def get(self):
        return self.val / self.count


class MaskedL1(torch.nn.Module):

    def forward(self, x, target, lens):
        target.requires_grad = False
        max_len = target.size(2)
        mask = pad_mask(lens, max_len)
        try:
            mask = mask.unsqueeze(1).expand_as(x)
        except:
            # print('x', x.shape)
            # print(' m', mask.shape)
            # x = x.view(x.size(0), x.size(1))
            # mask = mask.unsqueeze(1).expand_as(x)
            pass
        loss = F.l1_loss(
            x * mask, target * mask, reduction='sum')
        return loss / mask.sum()


# Adapted from https://gist.github.com/jihunchoi/f1434a77df9db1bb337417854b398df1
def pad_mask(lens, max_len):
    batch_size = lens.size(0)
    seq_range = torch.arange(0, max_len).long()
    seq_range = seq_range.unsqueeze(0)
    seq_range = seq_range.expand(batch_size, max_len)
    if lens.is_cuda:
        seq_range = seq_range.cuda()
    lens = lens.unsqueeze(1)
    lens = lens.expand_as(seq_range)
    mask = seq_range < lens
    return mask.float()
