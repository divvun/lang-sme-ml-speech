# MODIFIED from:
# *****************************************************************************
#  Copyright (c) 2018, NVIDIA CORPORATION.  All rights reserved.
#
#  Redistribution and use in source and binary forms, with or without
#  modification, are permitted provided that the following conditions are met:
#      * Redistributions of source code must retain the above copyright
#        notice, this list of conditions and the following disclaimer.
#      * Redistributions in binary form must reproduce the above copyright
#        notice, this list of conditions and the following disclaimer in the
#        documentation and/or other materials provided with the distribution.
#      * Neither the name of the NVIDIA CORPORATION nor the
#        names of its contributors may be used to endorse or promote products
#        derived from this software without specific prior written permission.
#
#  THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
#  AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
#  IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
#  ARE DISCLAIMED. IN NO EVENT SHALL NVIDIA CORPORATION BE LIABLE FOR ANY
#  DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES
#  (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;
#  LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND
#  ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
#  (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
#  SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
#
# *****************************************************************************
import os
from scipy.io.wavfile import write
import torch
# from mel2samp import files_to_list, MAX_WAV_VALUE
from .denoiser import Denoiser
import tqdm
import glob


def waveglow_generate(mel_file, sentence, tts_k, waveglow_path='waveglow/waveglow_14000_st', sigma=0.6, output_dir='./audios', sampling_rate=22050):
    # mel_files = files_to_list(mel_files)
    waveglow = torch.load(waveglow_path, map_location='cpu')['model']
    # print('Loaded WaveGlow model!')
    for k, m in waveglow.named_modules():
        m._non_persistent_buffers_set = set()  # pytorch 1.6.0 compatability
    waveglow = waveglow.remove_weightnorm(waveglow)
    device = torch.device('cpu')
    waveglow.to(device).eval()
  
    # mel = torch.load(mel_file)
    # print("Loaded mel...",  mel_file)
      
   
    with torch.no_grad():
        audio = waveglow.infer(mel_file, sigma=sigma)
        audio = audio * 32768.0
    audio = audio.squeeze()
    audio = audio.cpu().numpy()
    audio = audio.astype('int16')
    audio_path = os.path.join(output_dir, "{}_synthesis.wav".format(str(sentence[:7]) + '_' + str(tts_k)))
    print('Generating wav ...')
    write(audio_path, sampling_rate, audio)
    print('Saved at: ', audio_path)
    return audio_path


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('-f', "--mel_file", required=True)
    parser.add_argument('-w', '--waveglow_path',default='waveglow/waveglow_14000_st',
                        help='Path to waveglow decoder checkpoint with model')
    parser.add_argument('-o', "--output_dir",default='./audio', required=True)
    parser.add_argument("-s", "--sigma", default=.6, type=float)
    parser.add_argument("--sampling_rate", default=22050, type=int)
    # parser.add_argument("--is_fp16", action="store_true")
    # parser.add_argument("-d", "--denoiser_strength", default=0.0, type=float,
                        # help='Removes model bias. Start with 0.1 and adjust')

    args = parser.parse_args()

    waveglow_generate(args.mel_file, args.waveglow_path, args.sigma, args.output_dir,
         args.sampling_rate)


