# import pandas as pd
# from datasets import Dataset
# import random
# import re
# import torchaudio
# import json
# import librosa
# import numpy as np
# import pyarrow.parquet as pq
# import librosa
# # import wavio
# import argparse
import subprocess

def pre_prepocess():
    print('Writing paired data to a file...')
    subprocess.check_output('python extract_filenames.py', shell=True, stderr=subprocess.STDOUT)
    # subprocess.check_output('python extract_filenames_tts.py', shell=True, stderr=subprocess.STDOUT)
    # exit()
    print('Splitting...')
    # subprocess.check_output('python split_files_tts.py', shell=True, stderr=subprocess.STDOUT)
    subprocess.check_output('python split_files.py', shell=True, stderr=subprocess.STDOUT)

    print('Resapling for TTS ...')
    
    subprocess.check_output('python downsample_wav.py', shell=True, stderr=subprocess.STDOUT)

    print('Cleaning paths for TTS...')
    subprocess.check_output('python clean_for_tts.py', shell=True, stderr=subprocess.STDOUT)




if __name__ == '__main__':
    
    pre_prepocess()
