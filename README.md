
# How to run

## Data preparation:
1. You need to clone the data repo TWICE (we need to keep two copies of data as we will do different things with it)! The data repo is private, you need to have access. The fisrt data folder (first clone) should be renamed to `data/speech-sme-tts` and the second one - `data/speech-sme-asr`. You also need sme-freecorpus.txt to be in home dir. 

2. `cd data/` and run `python preprocess_asr_tts.py`. This will take some time. It will write the training files, split them and resample data for TTS and ASR tasks. 

3. `cd ..` and run `python preprocess.py`, then `python train_tacotron.py --force_align` and `python process_for_asr.py` - these will finish data prep for tts and asr.
 
4. If everything worked out fine with the previous steps, you can now start the common training of TTS and ASR with `python train_forward.py`. Note, the common training was not testet because of cuda OOM, but each function inside it was tested. 

5. Preptrained models (more of them) necessary for this run are [here](https://drive.google.com/drive/folders/18nTVbsUlkbN4duvcbIeSS_gNsmG5bOiZ?usp=sharing). Place the folder (don't rename) `checkpoint-27363` that you dowloaded in `asr_output/` AND in  `checkpoints/sme_speech_tts.asr_forward/` (make a new dir in `checkpoints/`).


Further instructions on inference TBD.
This repository has reused code from ForwardTacotron (majority), Tacotron, WaveGlow and Huggingface (links and references to be added).







