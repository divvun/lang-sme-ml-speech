import os
import glob
# import docx2txt


def list_files(wav_root_path, filetype = 'wav'):
    glob_path = "{}/**/*.{}".format(wav_root_path, filetype)
    # print(glob_path)
    filelist = glob.glob(glob_path, recursive=True)
    filelist.sort()
    return filelist

def get_wav_file_desc(wav_file_path, desc_folder_path):
    wav_filename = '.'.join(os.path.basename(wav_file_path).split('.')[:-1])
    # print(desc_folder_path)
    desc = None

    # if os.path.isfile(os.path.join(desc_folder_path, f'{wav_filename}.word22')):
        # desc = read_desc_word22(os.path.join(desc_folder_path, f'{wav_filename}.word22'))
    if os.path.isfile(os.path.join(desc_folder_path, f'{wav_filename}.txt')):
        # print('here')
        desc = read_desc_txt(os.path.join(desc_folder_path, f'{wav_filename}.txt'))
    # if os.path.isfile(os.path.join(desc_folder_path, f'{wav_filename}.docx')):
    #     desc = convert_docx(os.path.join(desc_folder_path, f'{wav_filename}.docx'))

    return desc

def read_desc_word22(path):
    words = []
    with open(path, 'r') as file:
        contents = file.readlines()
        for i, line in enumerate(contents):
            if i == 0:
                continue
            fields = line.split(' ')
            words.append(fields[-1].rstrip())

    return ' '.join(words)

def read_desc_txt(path):
    words = []
    with open(path, 'r') as file:
        contents = file.readlines()
        for line in contents:
            # print(line)
            words.append(line.rstrip())

    return ' '.join(words)

# def convert_docx(path):
#     words = []
#     text = docx2txt.process(path)
#     contents = text.split('\n')
#     for line in contents:
#         words.append(line.rstrip())
#     return ' '.join(words)
    # print(text)
    # extract text and write images in /tmp/img_dir
    # text = docx2txt.process("file.docx", "/tmp/img_dir")

def extract_descriptions(path):
    # target_folders_asr = [('speech-sme-asr/wav-16-16', "speech-sme-asr/text"), ('speech-sme-asr/wav-16-22', "speech-sme-asr/text")]
    target_folders_tts = [('speech-sme-tts/wav-16-16',"speech-sme-asr/text"), ('speech-sme-tts/wav-16-22', "speech-sme-tts/text"),]
    # descriptions_asr = []
    descriptions_tts = []

    # for t in target_folders_asr:
    #     for wav_file_path in list_files(t if type(t) is not tuple else t[0]):
    #         desc = get_wav_file_desc(wav_file_path, t if type(t) is not tuple else t[1])
    #         # print(desc)
    #         if desc is not None:
    #             descriptions_asr.append((wav_file_path, desc))

    for t in target_folders_tts:
        for wav_file_path in list_files(t if type(t) is not tuple else t[0]):
            desc = get_wav_file_desc(wav_file_path, t if type(t) is not tuple else t[1])
            if desc is not None:
                descriptions_tts.append((wav_file_path, desc))

    # descriptions in one file, not separate files in a folder
    # acapela_wav_files_m = list_files('speech-sme-asr/akseptanstest_acapela/sme_male')
    # acapela_wav_files_f = list_files('speech-sme-asr/akseptanstest_acapela/sme_female')
    # with open('speech-sme-asr/akseptanstest_acapela/sme_test_sentences_mod.txt', 'r') as f:
        # acapela_descriptions = [line.rstrip() for line in f.readlines()]
        
    # descriptions_asr += zip(acapela_wav_files_m, acapela_descriptions)
    # descriptions_asr += zip(acapela_wav_files_f, acapela_descriptions)
    # descriptions_tts += zip(acapela_wav_files_f, acapela_descriptions)
    # print(descriptions[:3])
    # mod_desr = [(d[0].split('.')[0].split('/')[-1], d[1]) for d in descriptions]
    # print(mod_desr[:11])
    # descriptions = mod_desr
    # print(descriptions[:3])
    return descriptions_tts

def save_pipe_divided_file(descriptions, path):
    with open(path, 'w') as file:
        for i, line in enumerate(descriptions):
            # print('line', line)
            file.write('|'.join(line))
            if i != len(descriptions) - 1:
                file.write('\n')

if __name__ == '__main__':
    print("Extracting descriptions...")
    descriptions_tts = extract_descriptions('./data/speech-sme-tts')
    print("Writing file...")
    # save_pipe_divided_file(descriptions_asr, 'speech-sme-asr/desc_asr.psv')
    save_pipe_divided_file(descriptions_tts, 'speech-sme-tts/desc_tts.psv')

    print("Done!")
 