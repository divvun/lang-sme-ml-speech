from utils.load_asr_dataset import create_processor
from transformers import Wav2Vec2ForCTC,Wav2Vec2Processor

def create_model():
    
    processor = Wav2Vec2Processor.from_pretrained('./asr_output/pretrained_processor')
    model = Wav2Vec2ForCTC.from_pretrained(
        "./asr_output/checkpoint-27363",
        attention_dropout=0.1,
        hidden_dropout=0.1,
        feat_proj_dropout=0.0,
        mask_time_prob=0.05,
        layerdrop=0.1,
        gradient_checkpointing=True,
        ctc_loss_reduction="mean",
        pad_token_id=processor.tokenizer.pad_token_id,
        vocab_size=len(processor.tokenizer)
    )
    model.freeze_feature_extractor()
    return model
