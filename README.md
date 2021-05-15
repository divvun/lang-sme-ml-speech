
# How to run

## Data preparation:
1. You need to clone the data repo 
```sh
cd data/
git clone git@github.com:giellalt/speech-sme.git speech-sme-tts
cp -r speech-sme-tts speech-sme-asr
```

We need two copies of data! The data repo is private, you need to have access. The fisrt data folder (first clone) should be renamed to `data/speech-sme-tts` and the second one - `data/speech-sme-asr`. You also need `sme-freecorpus.txt` to be in home dir. 

2. You still should be in `data/` . Run `python preprocess_asr_tts.py`. This will take some time. It will write the training files, split them and resample data for TTS and ASR tasks. 


3. `cd ..` and run `python preprocess.py`, then `python train_tacotron.py --force_align` and `python process_for_asr.py` (requires a lot of RAM) - these will finish data prep for tts and asr. If you cannot run `python process_for_asr.py` you can download pickled dataset from  [here](https://drive.google.com/drive/folders/18nTVbsUlkbN4duvcbIeSS_gNsmG5bOiZ?usp=sharing).
 
4. Preptrained models are [here](https://drive.google.com/drive/folders/18nTVbsUlkbN4duvcbIeSS_gNsmG5bOiZ?usp=sharing). Place the folder (don't rename) `checkpoint-27363` that you dowloaded in `asr_output/` AND in  `checkpoints/sme_speech_tts.asr_forward/` (make a new dir in `checkpoints/`). Place the files from `tacotron` in `checkpoints/sme_speech_tts.tacotron/`, and from `forward_tacotron` in `checkpoints/sme_speech_tts.forward/`.
Put `waveglow_14000_st` in `waveglow/` folder.

## Training

If everything worked out fine with the previous steps, you can now start the common training of TTS and ASR with `python train_forward.py`. Note, the common training was not testet because of cuda OOM, but each function inside it was tested. 


Further instructions on inference TBD.
This repository has reused code from ForwardTacotron (majority), Tacotron, WaveGlow and Huggingface (links and references to be added).

## Inference

When you run `bash run_forward_inf` this will generate audio in `audio` folder from sentences.txt.
Run `predict.py` to inference with ASR model.


# Supercomputer run

1. Log in as instructed [here](https://documentation.sigma2.no/getting_started/getting_started.html). 
2. Go to `~/cluster/projects/nn9866k/`
4. mkdir [your project folder]
5. Run `module load PyTorch/1.4.0-fosscuda-2019b-Python-3.7.4 `
6. Run `module unload PyTorch/1.4.0-fosscuda-2019b-Python-3.7.4 `
7. You can now put your code and data in [your project folder] -- e.g. `git clone` or upload (like Transmit) or scp (`scp -r [your things] user@saga.sigma2.no:/the/path/to/the/shared/place`)
8. Make virtual env `python3 -m venv env` and ACTIVATE!
9. `pip install [your requirements.txt]`.
10. Do some edits if you need (if you need to test your code). Nothing that requires cuda would work here. Only text cleaning and similar tasks. 
11. Deactivate env.
12. Create a file like `run_training.sh` - more [here](https://documentation.sigma2.no/getting_started/tutorials/gpu.html)
13. `sbatch [your shell script]` will queue your task and run. You will see the running output in a file {job_id}.out, but please note, it will take a while before you see the first print statement. They arrive in batches (e.g. only after epoch is finished, you will see the prints). To see if the training is going, you can monitor .csv file with gpu usage stats. 
