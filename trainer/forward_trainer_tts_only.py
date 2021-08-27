# from preprocess import Preprocessor

import itertools
from json import decoder
import json
import warnings
from utils.metrics import attention_score
from utils.duration_extraction import extract_durations_per_count, extract_durations_with_dijkstra
from utils.paths import Paths
import subprocess
from sys import path
import time
from utils.files import pickle_binary, unpickle_binary
from utils.text import clean_text, text_to_sequence
from transformers import Wav2Vec2Processor, AdamW, EvalPrediction
from torch.utils.data.dataloader import DataLoader
from utils import checkpoints
import torchaudio
from process_for_asr import prepare
from transformers.utils.dummy_pt_objects import Wav2Vec2ForCTC
from typing import Tuple
from models.tacotron import Tacotron

import torch
from torch.autograd import grad

from torch.optim.optimizer import Optimizer
from torch.utils.data.dataset import Dataset
from torch.utils.tensorboard import SummaryWriter

from models.forward_tacotron import ForwardTacotron
from trainer.common import Averager, ForwardSession, MaskedL1, ASRSession, init_trainer, asr_predict_for_dt, load_dt_data
from trainer.asr_utils import evaluate_asr
from utils import hparams as hp
from utils.checkpoints import restore_checkpoint, save_checkpoint
from utils.dataset import get_tts_datasets
from utils.decorators import ignore_exception
from utils.display import progbar, stream, simple_table, plot_mel
from utils.dsp import float_2_label, reconstruct_waveform, np_now, save_wav
from utils.paths import Paths
import os
import re
import numpy as np


def get_unpaired_txt(stop_point):
    with open('sme-freecorpus.txt', 'r', newline='\n') as text_file: 
        sents_list = []
        for line_terminated in text_file: 
            if len(line_terminated) > 115:
                line_terminated = line_terminated[:100]
            sents_list.append(line_terminated)
           
            if len(sents_list) > stop_point:
                break
       
    return sents_list



class ForwardTrainer:

    def __init__(self, paths: Paths) -> None:
        self.paths = paths
        self.writer = SummaryWriter(log_dir=paths.forward_log, comment='v1')
        self.l1_loss = MaskedL1()

    def train(self, model_tts: ForwardTacotron, optimizer_tts: Optimizer) -> None:
        
        for i, session_params in enumerate(hp.forward_schedule, 1):
            lr, max_step, bs = session_params
            if model_tts.get_step() < max_step:
                path=self.paths.data
                tts_train_set, tts_val_set = get_tts_datasets(path=self.paths.data, batch_size=bs, r=1, model_type='forward')
            
                tts_session = ForwardSession(path, index=i, r=1, lr=lr, max_step=max_step, bs=bs, train_set=tts_train_set, val_set=tts_val_set, )
                self.train_session(model_tts, optimizer_tts, tts_session)

    def train_session(self, model_tts: ForwardTacotron, optimizer_tts: Optimizer, tts_session: ForwardSession) -> None:
       
        current_step = model_tts.get_step()
        tts_training_steps = tts_session.max_step - current_step
    
        
        
        total_iters = len(tts_session.train_set)
        epochs = tts_training_steps // total_iters + 1
        simple_table([('TTS Steps', str(tts_training_steps // 1000) + 'k Steps'),
                      ('Batch Size TTS', tts_session.bs),
                      ('Learning Rate', tts_session.lr)])

        for g in optimizer_tts.param_groups:
            g['lr'] = tts_session.lr

        m_loss_avg = Averager()
        dur_loss_avg = Averager()
        duration_avg = Averager()
      
        device = next(model_tts.parameters()).device  # use same device as model parameters
        warnings.filterwarnings('ignore', category=UserWarning)
        for e in range(1, epochs + 1):
            
            #tts train loop for epoch
            for i, (x, m, ids, x_lens, mel_lens, dur) in enumerate(tts_session.train_set, 1):
                start = time.time()
                model_tts.train()
                x, m, dur, x_lens, mel_lens = x.to(device), m.to(device), dur.to(device),\
                                                     x_lens.to(device), mel_lens.to(device)

                m1_hat, m2_hat, dur_hat = model_tts(x, m, dur, mel_lens)

                m1_loss = self.l1_loss(m1_hat, m, mel_lens)
                m2_loss = self.l1_loss(m2_hat, m, mel_lens)

                dur_loss = self.l1_loss(dur_hat.unsqueeze(1), dur.unsqueeze(1), x_lens)

                tts_s_loss = m1_loss + m2_loss + 0.1 * dur_loss 
                optimizer_tts.zero_grad()
                tts_s_loss.backward()
                torch.nn.utils.clip_grad_norm_(model_tts.parameters(), hp.tts_clip_grad_norm)
                optimizer_tts.step()
                m_loss_avg.add(m1_loss.item() + m2_loss.item())
                dur_loss_avg.add(dur_loss.item())
                step = model_tts.get_step()
                k = step // 1000

                duration_avg.add(time.time() - start)
                # pitch_loss_avg.add(pitch_loss.item())

                speed = 1. / duration_avg.get()
                msg_tts = f'| TTS MODEL (supervised training ): '\
                      f'| Epoch: {e}/{epochs} ({i}/{total_iters}) | Mel Loss: {m_loss_avg.get():#.4} ' \
                      f'| Dur Loss: {dur_loss_avg.get():#.4} ' \
                      f'| {speed:#.2} steps/s | Step: {k}k | '

                if step % hp.forward_checkpoint_every == 0:
                    ckpt_name = f'forward_step{k}K'
                    save_checkpoint('forward', self.paths, model_tts, optimizer_tts,
                                    name=ckpt_name, is_silent=True)

                if step % hp.forward_plot_every == 0:

                    self.generate_plots(model_tts, tts_session)

                self.writer.add_scalar('Mel_Loss/train', m1_loss + m2_loss, model_tts.get_step())
                self.writer.add_scalar('Duration_Loss/train', dur_loss, model_tts.get_step())
                self.writer.add_scalar('Params/batch_size', tts_session.bs, model_tts.get_step())
                self.writer.add_scalar('Params/learning_rate', tts_session.lr, model_tts.get_step())


                stream(msg_tts)
     
           
            m_val_loss, dur_val_loss = self.evaluate(model_tts, tts_session.val_set)
            eval_tts_msg = f'| TTS MODEL (supervised eval ): '\
                        f'| Epoch: {e}/{epochs} | Val Loss: {m_val_loss:#.4} ' \
                        f'| Dur Val Loss: {dur_val_loss:#.4} ' \
                        
            stream(eval_tts_msg)
            # tts_eval_loss = m_val_loss + dur_val_loss
          

    def evaluate(self, model: ForwardTacotron, val_set: Dataset) -> Tuple[float, float,float]:
        model.eval()
        m_val_loss = 0
        dur_val_loss = 0
        # pitch_val_loss = 0
        device = next(model.parameters()).device
        for i, (x, m, ids, x_lens, mel_lens, dur) in enumerate(val_set, 1):
            x, m, dur, x_lens, mel_lens = x.to(device), m.to(device), dur.to(device), \
                                                 x_lens.to(device), mel_lens.to(device)
            with torch.no_grad():
                m1_hat, m2_hat, dur_hat = model(x, m, dur, mel_lens)
                m1_loss = self.l1_loss(m1_hat, m, mel_lens)
                m2_loss = self.l1_loss(m2_hat, m, mel_lens)
                dur_loss = self.l1_loss(dur_hat.unsqueeze(1), dur.unsqueeze(1), x_lens)
                m_val_loss += m1_loss.item() + m2_loss.item()
                dur_val_loss += dur_loss.item()
        m_val_loss /= len(val_set)
        dur_val_loss /= len(val_set)
        return m_val_loss, dur_val_loss #, pitch_val_loss

    @ignore_exception
    def generate_plots(self, model: ForwardTacotron, session: ForwardSession) -> None:
        model.eval()
        device = next(model.parameters()).device
        x, m, ids, x_lens, mel_lens, dur = session.val_sample
        x, m, dur, mel_lens = x.to(device), m.to(device), dur.to(device), mel_lens.to(device) 

        m1_hat, m2_hat, dur_hat = model(x, m, dur, mel_lens)
        m1_hat = np_now(m1_hat)[0, :600, :]
        m2_hat = np_now(m2_hat)[0, :600, :]
        m = np_now(m)[0, :600, :]

        m1_hat_fig = plot_mel(m1_hat)
        m2_hat_fig = plot_mel(m2_hat)
        m_fig = plot_mel(m)
       
        self.writer.add_figure('Ground_Truth_Aligned/target', m_fig, model.step)
        self.writer.add_figure('Ground_Truth_Aligned/linear', m1_hat_fig, model.step)
        self.writer.add_figure('Ground_Truth_Aligned/postnet', m2_hat_fig, model.step)

        m2_hat_wav = reconstruct_waveform(m2_hat)
        target_wav = reconstruct_waveform(m)

        self.writer.add_audio(
            tag='Ground_Truth_Aligned/target_wav', snd_tensor=target_wav,
            global_step=model.step, sample_rate=hp.sample_rate)
        self.writer.add_audio(
            tag='Ground_Truth_Aligned/postnet_wav', snd_tensor=m2_hat_wav,
            global_step=model.step, sample_rate=hp.sample_rate)

        m1_hat, m2_hat, dur_hat = model.generate(x[0, :x_lens[0]].tolist())
        m1_hat_fig = plot_mel(m1_hat)
        m2_hat_fig = plot_mel(m2_hat)

        self.writer.add_figure('Generated/target', m_fig, model.step)
        self.writer.add_figure('Generated/linear', m1_hat_fig, model.step)
        self.writer.add_figure('Generated/postnet', m2_hat_fig, model.step)

        m2_hat_wav = reconstruct_waveform(m2_hat)
        self.writer.add_audio(
            tag='Generated/target_wav', snd_tensor=target_wav,
            global_step=model.step, sample_rate=hp.sample_rate)
        self.writer.add_audio(
            tag='Generated/postnet_wav', snd_tensor=m2_hat_wav,
            global_step=model.step, sample_rate=hp.sample_rate)
