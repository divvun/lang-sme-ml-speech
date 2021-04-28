

def split_data(path, tts=False):
    test_lines = []
    train_lines = []
    # test_dt_lines = []
    with open(path, 'r') as f:
        lines = f.readlines()
        for i, line in enumerate(lines):
            if tts:
                train_lines.append(line)
                # print(train_lines)
                # return train_lines
            else:
                if i < 300:
                    test_lines.append(line)
                # if 300 < i < 600:
                #     test_dt_lines.append(line)
                if 300 <= i < 4000:
                    train_lines.append(line)
    return test_lines, train_lines


def write_to_file(path, out_path_train, out_path_test, tts=False):
    
    if tts:
        test, train = split_data(path, tts=True)
        # print(len(train))
        with open(out_path_train, 'w') as f:
            f.write(''.join(train))
        # with open(out_path_test, 'w') as f:
        #     f.write(''.join(test))
    else:
        test, train = split_data(path)

        with open(out_path_train, 'w') as f:
            f.write(''.join(train))
        with open(out_path_test, 'w') as f:
            f.write(''.join(test))
    # with open(out_path_test_dt, 'w') as f:
    #     f.write(''.join(test_dt))
    
def split_for_dt():
    with open('speech-sme-tts/train_tts.txt', 'r') as f:
        lines = f.readlines()
        dt_set = []
        for i, line in enumerate(lines):
            if i < 100:
                dt_set.append(line)
    with open('speech-sme-asr/dt_set.txt', 'w') as f:
        f.write(''.join(dt_set))




if __name__ == '__main__':
    print('Splitting data...')
    print("Writing files...")
    write_to_file('speech-sme-asr/desc_asr.psv', 'speech-sme-asr/train_asr.txt', 'speech-sme-asr/test_asr.txt')
    write_to_file('speech-sme-tts/desc_tts.psv', 'speech-sme-tts/train_tts.txt', 'speech-sme-tts/test_tts_dt.txt', tts=True)
    split_for_dt()
    print("Done!")
        