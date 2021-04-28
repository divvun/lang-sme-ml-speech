import argparse
import itertools

from pathlib import Path
import re
import os
import torch
from torch import optim
from torch.utils.data.dataloader import DataLoader
from trainer.forward_trainer import get_last_checkpoint
from models.forward_tacotron import ForwardTacotron
from models.tacotron import Tacotron
from trainer.forward_trainer import ForwardTrainer
from utils import hparams as hp
from utils import checkpoints
from utils.checkpoints import restore_checkpoint
from utils.dataset import get_tts_datasets
from utils.display import *
from utils.paths import Paths
from utils.text.symbols import phonemes
from transformers import AdamW, Wav2Vec2ForCTC,Wav2Vec2Processor
from trainer.common import create_model


def create_gta_features(model: Tacotron,
                        train_set: DataLoader,
                        val_set: DataLoader,
                        save_path: Path):
    model.eval()
    device = next(model.parameters()).device  # use same device as model parameters
    iters = len(train_set) + len(val_set)
    dataset = itertools.chain(train_set, val_set)
    for i, (x, mels, ids, x_lens, mel_lens, dur) in enumerate(dataset, 1):
        x, m, dur, x_lens, mel_lens = x.to(device), mels.to(device), dur.to(device), \
                                             x_lens.to(device), mel_lens.to(device) #, pitch.to(device)

        with torch.no_grad():
            _, gta, _, _ = model(x, mels, dur, mel_lens ) #, pitch)
        gta = gta.cpu().numpy()
        for j, item_id in enumerate(ids):
            mel = gta[j][:, :mel_lens[j]]
            np.save(str(save_path/f'{item_id}.npy'), mel, allow_pickle=False)
        bar = progbar(i, iters)
        msg = f'{bar} {i}/{iters} Batches '
        stream(msg)


if __name__ == '__main__':
    # Parse Arguments
    parser = argparse.ArgumentParser(description='Train Tacotron TTS')
    parser.add_argument('--force_gta', '-g', action='store_true', help='Force the model to create GTA features')
    parser.add_argument('--force_cpu', '-c', action='store_true', help='Forces CPU-only training, even when in CUDA capable environment')
    parser.add_argument('--hp_file', metavar='FILE', default='hparams.py', help='The file to use for the hyperparameters')
    args = parser.parse_args()

    hp.configure(args.hp_file)  # Load hparams from file

    paths = Paths(hp.data_path, hp.voc_model_id, hp.tts_model_id)
    assert len(os.listdir(paths.alg)) > 0, f'Could not find alignment files in {paths.alg}, please predict ' \
                                           f'alignments first with python train_tacotron.py --force_align!'

    force_gta = args.force_gta

    if not args.force_cpu and torch.cuda.is_available():
        device = torch.device('cuda')
        for session in hp.forward_schedule:
            _, _, batch_size = session
            if batch_size % torch.cuda.device_count() != 0:
                raise ValueError('`batch_size` must be evenly divisible by n_gpus!')
    else:
        device = torch.device('cpu')
    print('Using device:', device)

    # Instantiate Forward TTS Model
    print('\nInitialising Forward TTS Model...\n')
    model = ForwardTacotron(embed_dims=hp.forward_embed_dims,
                            num_chars=len(phonemes),
                            durpred_rnn_dims=hp.forward_durpred_rnn_dims,
                            durpred_conv_dims=hp.forward_durpred_conv_dims,
                            durpred_dropout=hp.forward_durpred_dropout,
                            # pitch_rnn_dims=hp.forward_pitch_rnn_dims,
                            # pitch_conv_dims=hp.forward_pitch_conv_dims,
                            # pitch_dropout=hp.forward_pitch_dropout,
                            # pitch_emb_dims=hp.forward_pitch_emb_dims,
                            # pitch_proj_dropout=hp.forward_pitch_proj_dropout,
                            rnn_dim=hp.forward_rnn_dims,
                            postnet_k=hp.forward_postnet_K,
                            postnet_dims=hp.forward_postnet_dims,
                            prenet_k=hp.forward_prenet_K,
                            prenet_dims=hp.forward_prenet_dims,
                            highways=hp.forward_num_highways,
                            dropout=hp.forward_dropout,
                            n_mels=hp.num_mels).to(device)

    model_parameters = filter(lambda p: p.requires_grad, model.parameters())
    params = sum([np.prod(p.size()) for p in model_parameters])
    print(f'num params {params}')

    optimizer = optim.Adam(model.parameters())
    restore_checkpoint('forward', paths, model, optimizer, create_if_missing=True)
    print('\nASR model...')
    print(f'Creating ASR model and optimizer...')


    modelasr_folder = './checkpoints/sme_speech_tts.asr_forward/'
    modelasr_name = 'checkpoint-'


    model_to_load, step = get_last_checkpoint(modelasr_folder, modelasr_name)
    asr_model = create_model(model_to_load)
    # print(model_to_load)
    if step == 27363:
  
        print("Didn't find trained ASR models from DualTransformation training. Starting from the finetuned checkpoint...")
    else:
        print(f"Resuming ASR from step {step}...")
  
    # no_decay = ['bias', 'LayerNorm.weight']

    # optimizer_grouped_parameters = [
            # {'params': [p for n, p in asr_model.named_parameters() if not any(nd in n for nd in no_decay)], 'weight_decay': 0.01},
            # {'params': [p for n, p in asr_model.named_parameters() if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}
        # ]

    # asr_model.to(device)
    optimizer_asr = AdamW(asr_model.parameters(), lr=1e-5)
    
            
    try:
        optimizer_asr.load_state_dict(torch.load(model_to_load + '/optimizer.pt',  map_location='cpu'))
        # print(device)
        print('Loaded optimizer for ASR...')
        # optimizer_asr.to(device)
    except:
        print('--empty opt--')
        # except:
        #     print('didn;t find pretrained ASR optimizer')
    

    if force_gta:
        print('Creating Ground Truth Aligned Dataset...\n')
        train_set, val_set = get_tts_datasets(paths.data, 8, r=1, model_type='forward')
        create_gta_features(model, train_set, val_set, paths.gta)
        print('\n\nYou can now train WaveRNN on GTA features - use python train_wavernn.py --gta\n')
    else:
        trainer = ForwardTrainer(paths)
        # print('initial', torch.cuda.memory_allocated(device=device))

        trainer.train(model, asr_model, optimizer, optimizer_asr)
