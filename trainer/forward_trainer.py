# from preprocess import Preprocessor

import itertools
from json import decoder
import json
import warnings
from utils.metrics import attention_score
from utils.duration_extraction import extract_durations_per_count, extract_durations_with_dijkstra
# from pathlib import Path
from utils.paths import Paths
# from .trainer_pt_utils import save_state
import subprocess
from sys import path
import time
from utils.files import pickle_binary, unpickle_binary
from utils.text import clean_text, text_to_sequence
from transformers import Wav2Vec2Processor, AdamW
from torch.utils.data.dataloader import DataLoader
# from transformers import trainer
# from transformers.trainer_utils import get_last_checkpoint
from utils import checkpoints
import torchaudio
# from transformers import trainer
# from utils.load_asr_dataset import prepare_asr_data, prepare_dt_data, prepare_test_dt
from process_for_asr import prepare
from transformers.utils.dummy_pt_objects import Wav2Vec2ForCTC
# from trainer.dual_transformation import dual_transform
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

def get_last_checkpoint(folder, modelasr_name):
    content = os.listdir(folder)
    # print(content)
    checkpoints = []
    
    for path in content:
        
        if str(path.split('.')[0]).startswith(modelasr_name):
            checkpoints.append(path)

    if len(checkpoints) == 0:
        return
    _re_checkpoint = re.compile(r'(\d+)')
    max_checkpoint = max(checkpoints, key=lambda x: int(_re_checkpoint.search(x).groups()[0]))
    step_num = int(_re_checkpoint.search(max_checkpoint).groups()[0])

    return os.path.join(folder, max_checkpoint), step_num



class ForwardTrainer:

    def __init__(self, paths: Paths) -> None:
        self.paths = paths
        self.writer = SummaryWriter(log_dir=paths.forward_log, comment='v1')
        self.l1_loss = MaskedL1()

    def train(self, model_tts: ForwardTacotron, model_asr: Wav2Vec2ForCTC, optimizer_tts: Optimizer, optimizer_asr: Optimizer) -> None:
        print("Loading ASR training data...")
        asr_train_set  = unpickle_binary('./data/speech-sme-asr/train_asr.pkl')
        asr_test_set = unpickle_binary('./data/speech-sme-asr/test_asr.pkl')
        # exit()
        asr_trainer = init_trainer(asr_train_set, asr_test_set)
      
        for i, session_params in enumerate(hp.forward_schedule, 1):
            lr, max_step, bs = session_params
            if model_tts.get_step() < max_step:
                path=self.paths.data
                # print(path)
                tts_train_set, tts_val_set = get_tts_datasets(path=self.paths.data, batch_size=bs, r=1, model_type='forward')
                
                asr_train_set = asr_trainer.get_train_dataloader()
                asr_test_set = asr_trainer.get_test_dataloader(asr_test_set)
                asr_pr = Wav2Vec2Processor.from_pretrained('./asr_output/pretrained_processor')
             
                tts_session = ForwardSession(path, index=i, r=1, lr=lr, max_step=max_step, bs=bs, train_set=tts_train_set, val_set=tts_val_set, )
                asr_session = ASRSession(asr_pr,index=i, r=1, lr=lr, max_step=max_step, bs=4, train_set=asr_train_set, test_set=asr_test_set)
                self.train_session(model_tts, model_asr, optimizer_tts, tts_session, asr_session, asr_trainer, optimizer_asr)

    def train_session(self, model_tts: ForwardTacotron, model_asr: Wav2Vec2ForCTC, optimizer_tts: Optimizer, tts_session: ForwardSession, asr_session: ASRSession, asr_trainer, optimizer_asr) -> None:
        # print(tts_session.path)
        # exit()
        asr_trainer_state = {
                'logs': []}
        current_step = model_tts.get_step()
        tts_training_steps = tts_session.max_step - current_step
        try:
            _, asr_current_step = get_last_checkpoint('./checkpoints/sme_speech_tts.asr_forward/', 'model_at')
            asr_training_steps = tts_session.max_step - asr_current_step
        except:
            asr_current_step = 0
            asr_training_steps = tts_training_steps
        
        
        total_iters = len(tts_session.train_set)
        epochs = tts_training_steps // total_iters + 1
        simple_table([('TTS Steps', str(tts_training_steps // 1000) + 'k Steps'),
                      ('ASR Steps', str(asr_training_steps //1000) + 'k Steps'),
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
                # tts_s_loss.backward()
                torch.nn.utils.clip_grad_norm_(model_tts.parameters(), hp.tts_clip_grad_norm)
                # optimizer_tts.step()
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
                # print(msg_tts)
            # print(torch.cuda.memory_allocated(device=device))
            # model_tts = model_tts.to('cpu')

            for step, inputs in enumerate(asr_session.train_set):
             
                optimizer_asr.zero_grad()
       
                model_asr.to(device)
                for k, v in inputs.items():
                    if isinstance(v, torch.Tensor):
                        inputs[k] = v.to(device)
                model_asr.train()
                outputs = model_asr(**inputs)
                asr_s_loss = outputs["loss"] if isinstance(outputs, dict) else outputs[0]
                # asr_s_loss = asr_s_loss.mean()
                
                msg_asr =  f'| ASR MODEL (supervised training) : '\
                            f'| Epoch: {e}/{epochs} ({step}/{len(asr_session.train_set)}) | Loss ASR: {asr_s_loss:#.4} '\
                            f' ||||||||||||||||||||||'
          
                stream(msg_asr)
            # # model_asr.to('cuda')
           
            m_val_loss, dur_val_loss = self.evaluate(model_tts, tts_session.val_set)
            eval_tts_msg = f'| TTS MODEL (supervised eval ): '\
                        f'| Epoch: {e}/{epochs} | Val Loss: {m_val_loss:#.4} ' \
                        f'| Dur Val Loss: {dur_val_loss:#.4} ' \
                        
            stream(eval_tts_msg)
            tts_eval_loss = m_val_loss + dur_val_loss
            #     print(eval_tts_msg)

            # ASR eval supervised 
            print('\nEvaluating ASR model ...')
            # model_asr.to('cpu')
            eval_output = asr_trainer.prediction_loop(asr_session.test_set, "Prediction", metric_key_prefix='eval')

            asr_eval_loss = eval_output.metrics['eval_loss']
            eval_wer = eval_output.metrics['eval_wer']
            
            msg_asr_eval =   f'| ASR MODEL (supervised eval) : Epoch {e}/{epochs} | Loss ASR: {asr_eval_loss:#.4} | WER: {eval_wer} |||||||||||||||||||||||||||||||||||||||||||||||||||||'
            stream(msg_asr_eval)
            
            # dual transformation loop 
            # tts_s_loss = 3
            # asr_s_loss = 1
            tts_u_loss, asr_u_loss = self.dual_transform(model_tts, model_asr, optimizer_tts, optimizer_asr, asr_session.test_set, m_loss_avg, dur_loss_avg, device, asr_current_step, e, epochs, duration_avg, total_iters, tts_s_loss, asr_s_loss, tts_session.lr, tts_session.path)
            step += 1
            asr_path = f'checkpoint-{step}'
            modelasr_folder = './checkpoints/sme_speech_tts.asr_forward/'
            new_check = modelasr_folder + asr_path
            os.makedirs(new_check, exist_ok=True)

            # asr_path, asr_step = get_last_checkpoint(modelasr_folder, modelasr_name)
            
            save_checkpoint('forward', self.paths, model_tts, optimizer_tts, is_silent=True)
         
            # asr_u_loss = 2

            if "logs" not in asr_trainer_state:
                asr_trainer_state['logs'] = []
            asr_trainer_state['logs'].append({'step' : step,
                    'epoch' : e,
                    'asr_s_loss': int(asr_s_loss),
                    'asr_u_loss': int(asr_u_loss),
                    'tts_s_loss': int(tts_s_loss),
                    'tts_u_loss': int(tts_u_loss),
                    'tts_eval_loss':int(tts_eval_loss),
                    'asr_eval_loss': int(asr_eval_loss), 
                    'eval_wer': eval_wer
                    })
            
            with open(f'{modelasr_folder+ asr_path}/dt_trainer_state.json', 'w') as f:
                json.dump(asr_trainer_state, f)

            model_asr.save_pretrained(f'{new_check}')
          
            
            torch.save(optimizer_asr.state_dict(), f'{new_check}/optimizer.pt')

            print("Exiting due to cuda OOM!")
            exit(11)

    def dual_transform(self, model_tts, model_asr, optimizer_tts, optimizer_asr, asr_test_set, m_loss_avg, dur_loss_avg, device, asr_current_step, e, epochs, duration_avg, total_iters, tts_s_loss, asr_s_loss, tts_lr, tts_dt_path):
            print('\n\nStarting DualTransformation loop...\n')
            # exit()
            tmp_dir = './checkpoints/sme_speech_tts.asr_forward/dual_transform_tmp'
            # generate tmp ASR training data
            asr_train_data = []
            input_set = get_unpaired_txt(99)
            # print(input_set)
            text = [clean_text(v) for v in input_set]
            inputs = [text_to_sequence(t) for t in text]
            
            # generate unpaired data for ASR from TTS
            for i, x in enumerate(inputs, 1):
                _, m, dur = model_tts.generate(x, alpha=1.)
                wav = reconstruct_waveform(m, n_iter=32)
                wav_path = os.path.join(tmp_dir, f'{i}.wav')
                save_wav(wav, wav_path)
                asr_train_data.append((wav_path, text[i-1]))

            # print(asr_train_data)
            dt_asr_data = load_dt_data(asr_train_data)
            # reinit trainer with only tmp train data
            asr_trainer_dt = init_trainer(dt_asr_data, None)
            dt_train = asr_trainer_dt.get_train_dataloader()
            
            # unsuper train loop for ASR
            for step, inputs in enumerate(dt_train, 1):
                # model_asr.cpu()
                model_asr.train()
                model_asr.to(device)
                for k, v in inputs.items():
                    if isinstance(v, torch.Tensor):
                        inputs[k] = v.to(device)
                # model_asr.train()
                outputs = model_asr(**inputs)
                asr_u_loss = outputs["loss"] if isinstance(outputs, dict) else outputs[0]
                # asr_u_loss.detach()
                # asr_u_loss = asr_s_loss.mean()
                
                # model_name = step + asr_current_step
                msg_asr =   f'| ASR MODEL (unsupervised training) : '\
                        f'| Epoch: {e}/{epochs} ({step}/{len(dt_train)}) | Loss ASR: {asr_u_loss:#.4} '\
                        f' ||||||||||||||||||||||||||||||||||||||||||||||||'
                stream(msg_asr)

            # for f in os.listdir(tmp_dir):
            #     file_path = os.path.join(tmp_dir, f)
            #     if f.endswith('.wav'):
            #         os.unlink(file_path)

            # generate tmp TTS data from ASR
            # model_asr.to(device)
            asr_predict_for_dt(model_asr)
            
            subprocess.check_output('python preprocess.py -p "./data/speech-sme-tts" -d=True', shell=True, stderr=subprocess.STDOUT)
            print('Finished preprocessing for tmp data!')
      
            tmp_tts_train = get_tts_datasets(tts_dt_path, batch_size=2, r=1, model_type='forward_dt')
            print("Loaded tmp dataset!")
            # unsuper TTS training
            
            for i, (x, m, ids, x_lens, mel_lens, dur) in enumerate(tmp_tts_train, 1):
                start = time.time()
                model_tts.to(device)
                model_tts.train()
                # optimizer_tts.zero_grad()
                x, m, dur, x_lens, mel_lens = x.to(device), m.to(device), dur.to(device),\
                                                     x_lens.to(device), mel_lens.to(device)

                m1_hat, m2_hat, dur_hat = model_tts(x, m, dur, mel_lens)

                m1_loss = self.l1_loss(m1_hat, m, mel_lens)
                m2_loss = self.l1_loss(m2_hat, m, mel_lens)

                dur_loss = self.l1_loss(dur_hat.unsqueeze(1), dur.unsqueeze(1), x_lens)

                tts_u_loss = m1_loss + m2_loss + 0.1 * dur_loss 
                # optimizer_tts.zero_grad()
                # tts_u_loss.backward()
                torch.nn.utils.clip_grad_norm_(model_tts.parameters(), hp.tts_clip_grad_norm)
                # optimizer_tts.step()
                m_loss_avg.add(m1_loss.item() + m2_loss.item())
                dur_loss_avg.add(dur_loss.item())
                step = model_tts.get_step()
                k = step // 1000

                duration_avg.add(time.time() - start)
                # pitch_loss_avg.add(pitch_loss.item())

                speed = 1. / duration_avg.get()
                msg_tts = f'| TTS MODEL (unsupervised training ): '\
                      f'| Epoch: {e}/{epochs} ({i}/{total_iters}) | Mel Loss: {m_loss_avg.get():#.4} ' \
                      f'| Dur Loss: {dur_loss_avg.get():#.4} ' \
                      f'| {speed:#.2} steps/s | Step: {k}k | '

                stream(msg_tts)
            # m_val_loss, dur_val_loss = self.evaluate(model_tts, tts_session.val_set)
            #TODO: combine L and update
            # asr_s_loss = torch.tensor(asr_s_loss).to(device)
            combined_loss = 0.5 * (tts_s_loss + asr_s_loss) + (tts_u_loss + asr_u_loss)
            # backwards
            combined_loss.to(device)
            print(combined_loss)
            combined_loss.backward()
            optimizer_tts.step()

            for state in optimizer_asr.state.values():
                for k, v in state.items():
                    if torch.is_tensor(v):
                        state[k] = v.to(device)

            optimizer_asr.step()

            m_loss_avg.reset()
            duration_avg.reset()
            # pitch_loss_avg.reset()
            dt_msg = f'\n\nFinished DT loop in epoch {e}!\n'
            stream(dt_msg)
            print(' ')
            return tts_u_loss, asr_u_loss

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
                # pitch_val_loss += self.l1_loss(pitch_hat, pitch.unsqueeze(1), x_lens)
                m_val_loss += m1_loss.item() + m2_loss.item()
                dur_val_loss += dur_loss.item()
        m_val_loss /= len(val_set)
        dur_val_loss /= len(val_set)
        # pitch_val_loss /= len(val_set)
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
        # pitch_fig = plot_pitch(np_now(pitch[0]))
        # pitch_gta_fig = plot_pitch(np_now(pitch_hat.squeeze()[0]))

        # self.writer.add_figure('Pitch/target', pitch_fig, model.step)
        # self.writer.add_figure('Pitch/ground_truth_aligned', pitch_gta_fig, model.step)
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

        # pitch_gen_fig = plot_pitch(np_now(pitch_hat.squeeze()))

        # self.writer.add_figure('Pitch/generated', pitch_gen_fig, model.step)
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
