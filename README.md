# TTS and ASR for the North Sámi language

This project presents a common training procedure for TTS and ASR models suitable for a low-resource setup. During this common training, we sequentially run supervised and unsupervised training, the models produce new unpaired data and 'learn from each other'. 

This repository containes reused code from [ForwardTacotron](https://github.com/as-ideas/ForwardTacotron) (majority of the structure), [Tacotron2](https://github.com/NVIDIA/tacotron2), [WaveGlow](https://github.com/NVIDIA/waveglow/tree/5bc2a53e20b3b533362f974cfa1ea0267ae1c2b1) and Huggingface [tutorial](https://huggingface.co/blog/fine-tune-xlsr-wav2vec2) .

## How to run
The code runs with Python 3.8.8. We are working on compatibility between the verions (currently incompatible).

## Data preparation:   
Install `requirements.txt`.   
1. You need to clone the data repo 
```sh
cd data/
git clone git@github.com:giellalt/speech-sme.git speech-sme-tts
cp -r speech-sme-tts speech-sme-asr
```

We need two copies of data! The data repo is private, you need to have access. The fisrt data folder (first clone) should be renamed to `data/speech-sme-tts` and the second one - `data/speech-sme-asr`. You also need `sme-freecorpus.txt` to be in home dir. 

2. You still should be in `data/` . Run `python preprocess_asr_tts.py`. This will take some time. It will write the training files, split them and resample data for TTS and ASR tasks. 


3. `cd ..` and run `python preprocess.py`, then `python train_tacotron.py --force_align` and `python process_for_asr.py` (requires a lot of RAM) - these will finish data prep for tts and asr. If you cannot run `python process_for_asr.py` you can download pickled dataset from  [here](https://drive.google.com/drive/folders/18nTVbsUlkbN4duvcbIeSS_gNsmG5bOiZ?usp=sharing).
 
4. Preptrained models are [here](https://drive.google.com/drive/folders/18nTVbsUlkbN4duvcbIeSS_gNsmG5bOiZ?usp=sharing). Place the folder (don't rename) `checkpoint-27363` that you dowloaded in `asr_output/` AND in  `checkpoints/sme_speech_tts.asr_forward/` (make a new dir in `checkpoints/`). Place the files from `tacotron` in `checkpoints/sme_speech_tts.tacotron/`. If you want to run inference, you need to put files from `forward_tacotron` in `checkpoints/sme_speech_tts.forward/`.
Put `waveglow_14000_st` in `waveglow/` folder.
`sme-freecorpus.txt` should be in home dir. 

## Training

If everything worked out fine with the previous steps, you can now start the common training of TTS and ASR with `python train_forward.py`. This repo is setup for inference, so if you want to train the models, you need to do a bit extra work. You need `forward_tacotron/forward_step430K_weights.pyt` &`forward_tacotron/forward_step_430K_optim.pyt`. Change the paths in `utils/paths.py` respectively.

Alternatively, with your own data, you need to repeat Data preparation steps with the tacotron model that you trained, for asr you would need to run `python process_for_asr.py --from_scratch`. This will create and save a new processor and vocab. 

You would need to train ASR and TTS models without `dual_transformation` for around 500 steps for ASR and at least 300K for TTS. 

## Inference

When you run `python gen_forward --alpha .95 waveglow` or `python gen_forward --alpha .95 griffinlim` this will generate audio in `audio` folder from sentences.txt. The vocoder would be waveglow (recommended) or griffinlim respectively. `--alpha` value (float) is responsible for teh speed of the audio. 

 Run `predict.py` to inference with ASR model. This will both run WER calculation over the whole test set and will print out the predictions for the first 10 sentences in the dataset. 


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

# Running with Nix

A nix-shell file exists (`shell.nix`) that can be used with the nix package manager to take care of installing all Python and system requirements in one command.

1. Install nix (the package manager, not the OS) following the instructions on https://nixos.org/download.html. This should not be more than running the `curl` command.
2. Inside the top level directory of this repository, run `nix-shell`. This might take a while downloading dependencies and ultimately drop you in a nix-shell.
3. Proeeed with the "Data Preparation" step above and continue on to training the model.