""" from https://github.com/keithito/tacotron """

'''
Defines the set of symbols used in text input to the model.

The default is a set of ASCII characters that works well for English or text that has been run through Unidecode. For other data, you can modify _characters. See TRAINING_DATA.md for details. '''
from utils.text import cmudict

_pad = '_'
_punctuation = '!\'().:;? '
_special = '-'

LETTERS = 'AaÁáBbCcČčDdĐđEeFfGgHhIiJjKkLlMmNnŊŋOoPpRrSsŠšTtŦŧUuVvZzŽžÑñøØÆæøØÅåÄä'
# Prepend "@" to ARPAbet symbols to ensure uniqueness (some are the same as uppercase letters):
# _arpabet = [s for s in LETTERS]

# Phonemes
# _vowels = 'iyɨʉɯuɪʏʊeøɘəɵɤoɛœɜɞʌɔæɐaɶɑɒᵻ'
# _non_pulmonic_consonants = 'ʘɓǀɗǃʄǂɠǁʛ'
# _pulmonic_consonants = 'pbtdʈɖcɟkɡqɢʔɴŋɲɳnɱmʙrʀⱱɾɽɸβfvθðszʃʒʂʐçʝxɣχʁħʕhɦɬɮʋɹɻjɰlɭʎʟ'
# _suprasegmentals = 'ˈˌːˑ'
# _other_symbols = 'ʍwɥʜʢʡɕʑɺɧ'
# _diacrilics = 'ɚ˞ɫ'
phonemes = sorted(list(
   _pad + _punctuation + _special + LETTERS))

phonemes_set = set(phonemes)