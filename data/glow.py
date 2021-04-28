import os
import glob
import tqdm
import argparse
from scipy.io.wavfile import write
import torch


def inference_mel(input_mel_path):
    # vocoder = torch.load('waveglow_10000_st_ch', map_location='cpu')
    # vocoder.eval()
   
    mel = torch.load(input_mel_path)
    mel = torch.autograd.Variable(mel.cuda())
    mel = torch.unsqueeze(mel, 0)
    mel = mel.half() if is_fp16 else mel
    with torch.no_grad():
        audio = waveglow.infer(mel, sigma=sigma)
        if denoiser_strength > 0:
            audio = denoiser(audio, denoiser_strength)
        audio = audio * MAX_WAV_VALUE
    audio = audio.squeeze()
    audio = audio.cpu().numpy()
    audio = audio.astype('int16')
    audio_path = os.path.join(
        output_dir, "{}_synthesis.wav".format(file_name))
    write(audio_path, sampling_rate, audio)
    print(audio_path)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-i', '--input_mel_path', type=str, required=True,
                        help=".mel file to convert to wav")
    args = parser.parse_args()

    inference_mel(args.input_mel_path)