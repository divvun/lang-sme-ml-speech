
def clean_paths(path):
    with open(path, 'r') as f:
        filepaths_and_text = [('./' + str(line).strip()).strip().split('|')  for line in f]
        new_paths = []
        for p, t in filepaths_and_text:
            new_paths.append(p.split('/')[-1].split('.')[0] + '|' + t)
        # paths = filepaths_and_text[0]
    # if not path.split('/')[-1].split('.')[0] == 'dt_set':
    new_name = path.split('/')[0] + '/cleaned_' + path.split('/')[-1].split('.')[0] + '.csv'
    with open(new_name, 'w') as f:
        f.write('\n'.join(new_paths))

if __name__ == '__main__':
    print('Overwtiting paths ...')
    # clean_paths('speech-sme-asr/dt_set.txt')
    clean_paths('speech-sme-tts/train_tts.txt')

    print('Done!')

