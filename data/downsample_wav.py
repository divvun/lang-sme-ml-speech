import librosa
import soundfile as sf


def downsample(path, split='|', rate=16000):

    with open(path, encoding='utf-8') as f:
        filepaths_and_text = [('./' + str(line).strip()).strip().split(split)  for line in f]

    for e, i in enumerate(filepaths_and_text):

        wavs, _ = i[0], i[1]
    
        if librosa.get_samplerate(wavs) != rate:
            # print("found this sr", librosa.get_samplerate(wavs))
            # print(wavs, rate)
            
            y, _ = librosa.load(wavs, sr=rate)
            sf.write(wavs, y, rate)
            if e % 100 == 0:
                print(f"Done with {e} audio files")
        else: 
            continue

if __name__ == '__main__':
    
    downsample('speech-sme-tts/train_tts.txt', rate=22050)
    # downsample('speech-sme-asr/test_asr.txt')
    # downsample('speech-sme-asr/train_asr.txt')
    # downsample('speech-sme-asr/dt_set.txt')

    # downsample('test_asr_dt.txt')