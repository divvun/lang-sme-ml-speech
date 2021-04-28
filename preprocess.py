import argparse
from multiprocessing import Pool, cpu_count
from pathlib import Path
from random import Random
# import pyworld as pw
from typing import Tuple, Dict

from utils.display import *
from utils.dsp import *
from utils.files import get_files, pickle_binary
from utils.paths import Paths
from utils.text import clean_text
from utils.text.recipes import ljspeech


# Helper functions for argument types
def valid_n_workers(num):
    n = int(num)
    if n < 1:
        raise argparse.ArgumentTypeError('%r must be an integer greater than 0' % num)
    return n


class Preprocessor:

    def __init__(self, paths: Paths, text_dict: Dict[str, str]):
        self.paths = paths
        self.text_dict = text_dict

    def __call__(self, path: Path) -> Tuple[str, int, str]:
        wav_id = path.stem
        m, x = self._convert_file(path)
        np.save(self.paths.mel/f'{wav_id}.npy', m, allow_pickle=False)
        np.save(self.paths.quant/f'{wav_id}.npy', x, allow_pickle=False)
        # np.save(self.paths.raw_pitch/f'{wav_id}.npy', raw_pitch, allow_pickle=False)
        text = self.text_dict[wav_id]
        text = clean_text(text)
        return wav_id, m.shape[-1], text

    def _convert_file(self, path: Path) -> Tuple[np.array, np.array]:
        y = load_wav(path)
        if hp.trim_long_silences:
           y = trim_long_silences(y)
        if hp.trim_start_end_silence:
           y = trim_silence(y)
        peak = np.abs(y).max()
        if hp.peak_norm or peak > 1.0:
            y /= peak
        mel = melspectrogram(y)
        # pitch, _ = pw.dio(y.astype(np.float64), hp.sample_rate,
        #                   frame_period=hp.hop_length / hp.sample_rate * 1000)
        if hp.voc_mode == 'RAW':
            quant = encode_mu_law(y, mu=2**hp.bits) if hp.mu_law else float_2_label(y, bits=hp.bits)
        elif hp.voc_mode == 'MOL':
            quant = float_2_label(y, bits=16)
        else:
            raise ValueError(f'Unexpected voc mode {hp.voc_mode}, should be either RAW or MOL.')
        return mel.astype(np.float32), quant.astype(np.int64)


parser = argparse.ArgumentParser(description='Preprocessing for WaveRNN and Tacotron')
parser.add_argument('--path', '-p', help='directly point to dataset path (overrides hparams.wav_path')
parser.add_argument('--extension', '-e', metavar='EXT', default='.wav', help='file extension to search for in dataset folder')
parser.add_argument('--num_workers', '-w', metavar='N', type=valid_n_workers, default=cpu_count()-1, help='The number of worker threads to use for preprocessing')
parser.add_argument('--hp_file', metavar='FILE', default='hparams.py', help='The file to use for the hyperparameters')
parser.add_argument('--dt', '-d', default=False, help='if preprocess tmp file')
args = parser.parse_args()

hp.configure(args.hp_file)  # Load hparams from file
if args.path is None:
    args.path = hp.wav_path

extension = args.extension
path = args.path


if __name__ == '__main__':

    wav_files = get_files(path, extension)
    # print(path, extension)
    wav_ids = {w.stem for w in wav_files}
    paths = Paths(hp.data_path, hp.voc_model_id, hp.tts_model_id)
    # print(f'\n{len(wav_files)} {extension[1:]} files found in "{path}"')
    assert len(wav_files) > 0, f'Found no wav files in {path}, exiting.'
    # print(wav_files)
    # exit()
    if args.dt:
        print('running preproccesing for dt...')
        text_dict = ljspeech(path, dt=True)
    else:
        text_dict = ljspeech(path)
    # print(len(text_dict))
    # print(len(wav_ids))
    # for item_id, text in text_dict.items():
    #     # print(item_id)

    #     if item_id in wav_ids:
    #         # print(item_id)

    text_dict = {item_id: text for item_id, text in text_dict.items() if item_id in wav_ids}
    # exit()
    wav_files = [w for w in wav_files if w.stem in text_dict]
    print(f'Using {len(wav_files)} wav files that are indexed in metafile.\n')

    n_workers = max(1, args.num_workers)

    simple_table([
        ('Sample Rate', hp.sample_rate),
        ('Bit Depth', hp.bits),
        ('Mu Law', hp.mu_law),
        ('Hop Length', hp.hop_length),
        ('CPU Usage', f'{n_workers}/{cpu_count()}'),
        ('Num Validation', hp.n_val)
    ])

    pool = Pool(processes=n_workers)
    dataset = []
    cleaned_texts = []
    preprocessor = Preprocessor(paths, text_dict)

    for i, (item_id, length, cleaned_text) in enumerate(pool.imap_unordered(preprocessor, wav_files), 1):
        if item_id in text_dict:
            dataset += [(item_id, length)]
            cleaned_texts += [(item_id, cleaned_text)]
        bar = progbar(i, len(wav_files))
        message = f'{bar} {i}/{len(wav_files)} '
        stream(message)

    text_dict = {id: text for id, text in cleaned_texts}
    print(len(text_dict))
    dataset.sort()
    random = Random(hp.seed)
    random.shuffle(dataset)
    # print(len(train_dataset))
    if not args.dt:
        train_dataset = dataset[hp.n_val:]
        val_dataset = dataset[:hp.n_val]
        # sort val dataset longest to shortest
        val_dataset.sort(key=lambda d: -d[1])
        print(f'First val sample: {val_dataset[0][0]}')

        pickle_binary(val_dataset, paths.data/'val_dataset.pkl')

        pickle_binary(text_dict, paths.data/'text_dict.pkl')
        pickle_binary(train_dataset, paths.data/'train_dataset.pkl')
    else:
        pickle_binary(text_dict, paths.data/'text_dict_dt.pkl')
        pickle_binary(dataset, paths.data/'train_dataset_dt.pkl')

    print('\n\nCompleted. Ready to run "python train_tacotron.py" or "python train_wavernn.py". \n')
