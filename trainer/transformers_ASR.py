# import torch

# from transformers import Wav2Vec2ForCTC
# from dataclasses import dataclass, field
# from typing import Any, Dict, List, Optional, Union
# from datasets import load_metric
# from transformers.models.wav2vec2.processing_wav2vec2 import Wav2Vec2Processor
# # from load_dataset import prepare
# from models.sme_asr import create_model
# from transformers import TrainingArguments
# from transformers import Trainer
# import numpy as np
# from utils.load_asr_dataset import create_processor


# @dataclass
# class DataCollatorCTCWithPadding:
#     """
#     Data collator that will dynamically pad the inputs received.
#     Args:
#         processor (:class:`~transformers.Wav2Vec2Processor`)
#             The processor used for proccessing the data.
#         padding (:obj:`bool`, :obj:`str` or :class:`~transformers.tokenization_utils_base.PaddingStrategy`, `optional`, defaults to :obj:`True`):
#             Select a strategy to pad the returned sequences (according to the model's padding side and padding index)
#             among:
#             * :obj:`True` or :obj:`'longest'`: Pad to the longest sequence in the batch (or no padding if only a single
#               sequence if provided).
#             * :obj:`'max_length'`: Pad to a maximum length specified with the argument :obj:`max_length` or to the
#               maximum acceptable input length for the model if that argument is not provided.
#             * :obj:`False` or :obj:`'do_not_pad'` (default): No padding (i.e., can output a batch with sequences of
#               different lengths).
#         max_length (:obj:`int`, `optional`):
#             Maximum length of the ``input_values`` of the returned list and optionally padding length (see above).
#         max_length_labels (:obj:`int`, `optional`):
#             Maximum length of the ``labels`` returned list and optionally padding length (see above).
#         pad_to_multiple_of (:obj:`int`, `optional`):
#             If set will pad the sequence to a multiple of the provided value.
#             This is especially useful to enable the use of Tensor Cores on NVIDIA hardware with compute capability >=
#             7.5 (Volta).
#     """

#     processor: Wav2Vec2Processor
#     padding: Union[bool, str] = True
#     max_length: Optional[int] = None
#     max_length_labels: Optional[int] = None
#     pad_to_multiple_of: Optional[int] = None
#     pad_to_multiple_of_labels: Optional[int] = None

#     def __call__(self, features: List[Dict[str, Union[List[int], torch.Tensor]]]) -> Dict[str, torch.Tensor]:
#         # split inputs and labels since they have to be of different lenghts and need
#         # different padding methods
#         input_features = [{"input_values": feature["input_values"]} for feature in features]
#         label_features = [{"input_ids": feature["labels"]} for feature in features]

#         batch = self.processor.pad(
#             input_features,
#             padding=self.padding,
#             max_length=self.max_length,
#             pad_to_multiple_of=self.pad_to_multiple_of,
#             return_tensors="pt",
#         )
#         with self.processor.as_target_processor():
#             labels_batch = self.processor.pad(
#                 label_features,
#                 padding=self.padding,
#                 max_length=self.max_length_labels,
#                 pad_to_multiple_of=self.pad_to_multiple_of_labels,
#                 return_tensors="pt",
#             )

#         # replace padding with -100 to ignore loss correctly
#         labels = labels_batch["input_ids"].masked_fill(labels_batch.attention_mask.ne(1), -100)

#         batch["labels"] = labels

#         return batch

# def get_trainer():
#     data_collator = DataCollatorCTCWithPadding(processor=processor, padding=True)
#     wer_metric = load_metric("wer")

#     def compute_metrics(pred):
#         pred_logits = pred.predictions
#         pred_ids = np.argmax(pred_logits, axis=-1)

#         pred.label_ids[pred.label_ids == -100] = processor.tokenizer.pad_token_id

#         pred_str = processor.batch_decode(pred_ids)
#         # we do not want to group tokens when computing the metrics
#         label_str = processor.batch_decode(pred.label_ids, group_tokens=False)

#         wer = wer_metric.compute(predictions=pred_str, references=label_str)

#         return {"wer": wer}

#     model = Wav2Vec2ForCTC.from_pretrained(
#         "facebook/wav2vec2-large-xlsr-53",
#         attention_dropout=0.1,
#         hidden_dropout=0.1,
#         feat_proj_dropout=0.0,
#         mask_time_prob=0.05,
#         layerdrop=0.1,
#         gradient_checkpointing=True,
#         ctc_loss_reduction="mean",
#         pad_token_id=processor.tokenizer.pad_token_id,
#         vocab_size=len(processor.tokenizer)
#     )

#     model.freeze_feature_extractor()

#     training_args = TrainingArguments(
#         output_dir="model_output/",
#         # output_dir="./wav2vec2-large-xlsr-turkish-demo",
#         group_by_length=True,
#         per_device_train_batch_size=16,
#         gradient_accumulation_steps=2,
#         evaluation_strategy="steps",
#         num_train_epochs=30,
#         fp16=True,
#         save_steps=100,
#         eval_steps=500,
#         logging_steps=500,
#         learning_rate=1e-4,
#         weight_decay=0.005,
#         # learning_rate=3e-4,
#         warmup_steps=1000,
#         save_total_limit=2,
#     )

#     train, test = prepare()

#     trainer = Trainer(
#         model=model,
#         data_collator=data_collator,
#         args=training_args,
#         compute_metrics=compute_metrics,
#         train_dataset=train,
#         eval_dataset=test,
#         tokenizer=processor.feature_extractor,
#     )

    
#     class CustomTrainer(Trainer):
#         pass

#     trainer = Trainer(
#             model=model,
#             data_collator=data_collator,
#             args=training_args,
#             compute_metrics=compute_metrics,
#             train_dataset=train,
#             eval_dataset=test,
#             tokenizer=processor.feature_extractor,
#         )


# if __name__ == "__main__":
#     get_trainer()