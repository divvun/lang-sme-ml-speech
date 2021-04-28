import argparse
import itertools
from pathlib import Path

import torch
from torch import optim
from torch.utils.data.dataloader import DataLoader
from typing import Tuple

from models.tacotron import Tacotron
from trainer.taco_trainer import TacoTrainer
from utils import hparams as hp
from utils.checkpoints import restore_checkpoint
from utils.dataset import get_tts_datasets, filter_max_len
from utils.display import *
from utils.dsp import np_now
from utils.duration_extraction import extract_durations_per_count, extract_durations_with_dijkstra
from utils.files import pickle_binary, unpickle_binary
from utils.metrics import attention_score
from utils.paths import Paths
from utils.text import phonemes


def normalize_pitch(phoneme_pitches):
    nonzeros = np.concatenate([v[np.where(v != 0.0)[0]]
                               for item_id, v in phoneme_pitches])
    mean, std = np.mean(nonzeros), np.std(nonzeros)
    for item_id, v in phoneme_pitches:
        zero_idxs = np.where(v == 0.0)[0]
        v -= mean
        v /= std
        v[zero_idxs] = 0.0
    return mean, std


# adapted from https://github.com/NVIDIA/DeepLearningExamples/blob/
# 0b27e359a5869cd23294c1707c92f989c0bf201e/PyTorch/SpeechSynthesis/FastPitch/extract_mels.py
def extract_pitch(save_path: Path) -> Tuple[float, float]:
    train_data = unpickle_binary('data/train_dataset.pkl')
    val_data = unpickle_binary('data/val_dataset.pkl')
    all_data = filter_max_len(train_data + val_data)
    phoneme_pitches = []
    for prog_idx, (item_id, mel_len) in enumerate(all_data, 1):
        dur = np.load(paths.alg / f'{item_id}.npy')
        assert np.sum(dur) == mel_len
    
        pitch = np.load(paths.raw_pitch / f'{item_id}.npy')
    
        durs_cum = np.cumsum(np.pad(dur, (1, 0)))
        pitch_char = np.zeros((dur.shape[0],), dtype=np.float)
        for idx, a, b in zip(range(mel_len), durs_cum[:-1], durs_cum[1:]):
            values = pitch[a:b][np.where(pitch[a:b] != 0.0)[0]]
            values = values[np.where(values < hp.pitch_max_freq)[0]]
            pitch_char[idx] = np.mean(values) if len(values) > 0 else 0.0
        phoneme_pitches.append((item_id, pitch_char))
        bar = progbar(prog_idx, len(all_data))
        msg = f'{bar} {prog_idx}/{len(all_data)} Files '
        stream(msg)
       

    mean, var = normalize_pitch(phoneme_pitches)
    for item_id, phoneme_pitch in phoneme_pitches:
        np.save(str(save_path / f'{item_id}.npy'), phoneme_pitch, allow_pickle=False)

    print(f'\nPitch mean: {mean} var: {var}')
    return mean, var


def create_gta_features(model: Tacotron,
                        train_set: DataLoader,
                        val_set: DataLoader,
                        save_path: Path):
    model.eval()
    device = next(model.parameters()).device  # use same device as model parameters
    iters = len(train_set) + len(val_set)
    dataset = itertools.chain(train_set, val_set)
    for i, (x, mels, ids, x_lens, mel_lens) in enumerate(dataset, 1):
        x, mels = x.to(device), mels.to(device)
        with torch.no_grad():
            _, gta, _ = model(x, mels)
        gta = gta.cpu().numpy()
        for j, item_id in enumerate(ids):
            mel = gta[j][:, :mel_lens[j]]
            np.save(str(save_path/f'{item_id}.npy'), mel, allow_pickle=False)
        bar = progbar(i, iters)
        msg = f'{bar} {i}/{iters} Batches '
        stream(msg)


def create_align_features(model: Tacotron,
                          train_set: DataLoader,
                          val_set: DataLoader,
                          save_path_alg: Path,
                        #   save_path_pitch: Path
                        ):
    assert model.r == 1, f'Reduction factor of tacotron must be 1 for creating alignment features! ' \
                         f'Reduction factor was: {model.r}'
    model.eval()
    device = next(model.parameters()).device  # use same device as model parameters
    if val_set is not None:
        iters = len(val_set) + len(train_set)
        dataset = itertools.chain(train_set, val_set)
    else:
        # print('here')
        iters = len(train_set)
        # print(iters)
        dataset = itertools.chain(train_set)
   
    att_score_dict = {}

    if hp.extract_durations_with_dijkstra:
        print('Extracting durations using dijkstra...')
       
        dur_extraction_func = extract_durations_with_dijkstra
        
    else:
        print('Extracting durations using attention peak counts...')
        dur_extraction_func = extract_durations_per_count
    # for i in dataset:
        # print(i)
    for i, (x, mels, ids, x_lens, mel_lens) in enumerate(dataset, 1):
        x, mels = x.to(device), mels.to(device)
        # print(x)
        # print(mels)
        with torch.no_grad():
            _, _, att_batch = model(x, mels)
        align_score, sharp_score = attention_score(att_batch, mel_lens, r=1)
        att_batch = np_now(att_batch)
        seq, att, mel_len, item_id = x[0], att_batch[0], mel_lens[0], ids[0]
        align_score, sharp_score = float(align_score[0]), float(sharp_score[0])
        att_score_dict[item_id] = (align_score, sharp_score)
        durs = dur_extraction_func(seq, att, mel_len)
        if np.sum(durs) != mel_len:
            print(f'WARNINNG: Sum of durations did not match mel length for item {item_id}!')
        np.save(str(save_path_alg / f'{item_id}.npy'), durs, allow_pickle=False)
        bar = progbar(i, iters)
        msg = f'{bar} {i}/{iters} Batches '
        stream(msg)
    pickle_binary(att_score_dict, paths.data / 'att_score_dict.pkl')
    # print('Extracting Pitch Values...')
    # extract_pitch(save_path_pitch)


if __name__ == '__main__':
    # Parse Arguments
    parser = argparse.ArgumentParser(description='Train Tacotron TTS')
    parser.add_argument('--force_train', '-f', action='store_true', help='Forces the model to train past total steps')
    parser.add_argument('--force_gta', '-g', action='store_true', help='Force the model to create GTA features')
    parser.add_argument('--force_align', '-a', action='store_true', help='Force the model to create attention alignment features')
    parser.add_argument('--force_cpu', '-c', action='store_true', help='Forces CPU-only training, even when in CUDA capable environment')
    parser.add_argument('--extract_pitch', '-p', action='store_true', help='Extracts phoneme-pitch values only')
    parser.add_argument('--hp_file', metavar='FILE', default='hparams.py', help='The file to use for the hyperparameters')
    parser.add_argument('--fa_dt', action='store_true', help='if for DT')

    args = parser.parse_args()

    hp.configure(args.hp_file)  # Load hparams from file
    paths = Paths(hp.data_path, hp.voc_model_id, hp.tts_model_id)

    if args.extract_pitch:
        print('Extracting Pitch Values...')
        mean, var = extract_pitch(paths.phon_pitch)
        print('\n\nYou can now train ForwardTacotron - use python train_forward.py\n')
        exit()

    if not args.force_cpu and torch.cuda.is_available():
        device = torch.device('cuda')
        for session in hp.tts_schedule:
            _, _, _, batch_size = session
            if batch_size % torch.cuda.device_count() != 0:
                raise ValueError('`batch_size` must be evenly divisible by n_gpus!')
    else:
        device = torch.device('cpu')
    print('Using device:', device)

    # Instantiate Tacotron Model
    print('\nInitialising Tacotron Model...\n')
    model = Tacotron(embed_dims=hp.tts_embed_dims,
                     num_chars=len(phonemes),
                     encoder_dims=hp.tts_encoder_dims,
                     decoder_dims=hp.tts_decoder_dims,
                     n_mels=hp.num_mels,
                     fft_bins=hp.num_mels,
                     postnet_dims=hp.tts_postnet_dims,
                     encoder_K=hp.tts_encoder_K,
                     lstm_dims=hp.tts_lstm_dims,
                     postnet_K=hp.tts_postnet_K,
                     num_highways=hp.tts_num_highways,
                     dropout=hp.tts_dropout,
                     stop_threshold=hp.tts_stop_threshold).to(device)

    model_parameters = filter(lambda p: p.requires_grad, model.parameters())
    params = sum([np.prod(p.size()) for p in model_parameters])
    print(f'Num Params: {params}')
    optimizer = optim.Adam(model.parameters())
    restore_checkpoint('tts', paths, model, optimizer, create_if_missing=True, device=device)

    if args.force_gta:
        print('Creating Ground Truth Aligned Dataset...\n')
        train_set, val_set = get_tts_datasets(paths.data, 8, model.r)
        create_gta_features(model, train_set, val_set, paths.gta)
        print('\n\nYou can now train WaveRNN on GTA features - use python train_wavernn.py --gta\n')
    elif args.force_align:
        print('Creating Attention Alignments and Pitch Values...')
        train_set, val_set = get_tts_datasets(paths.data, 1, model.r)
        create_align_features(model, train_set, val_set, paths.alg) # paths.phon_pitch)
        print('\n\nYou can now train ForwardTacotron - use python train_forward.py\n')
    elif args.fa_dt:

        print('Creating Attention Alignments for DT...')
        train_set_dt = get_tts_datasets(paths.data, 1, model.r, model_type='forward_pre_dt')
        create_align_features(model, train_set_dt, None, paths.alg_dt) # paths.phon_pitch)
        print('\n\nYou can now train ForwardTacotron - use python train_forward.py\n')

    else:
        trainer = TacoTrainer(paths)
        trainer.train(model, optimizer)
        print('Creating Attention Alignments and Pitch Values...')
        train_set, val_set = get_tts_datasets(paths.data, 8, model.r)
        create_align_features(model, train_set, val_set, paths.alg) # paths.phon_pitch)
        print('\n\nYou can now train ForwardTacotron - use python train_forward.py\n')









