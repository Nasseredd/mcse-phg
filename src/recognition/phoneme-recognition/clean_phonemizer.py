import os
import sys
import glob
import torch
import argparse
import torchaudio
from tqdm import tqdm
from transformers import Wav2Vec2ForCTC, Wav2Vec2Processor
import subprocess

# Phonemize the text using eSpeak
def phonemize(text):
    with open(text, 'r') as file:
        transcription = file.read()
    result = subprocess.run(['espeak', '-q', '--ipa', transcription], stdout=subprocess.PIPE)
    phonemes = result.stdout.decode('utf-8').strip()
    return phonemes

def load_configs(args):
    """
    Load configuration settings from a JSON file.

    Parameters
    ----------

    Returns
    -------
    dict
        Configuration settings.
    """

    corpora_folder = '/srv/storage/talc3@storage4.nancy.grid5000.fr/multispeech/calcul/users/nmonir/corpora/'
    config = {
        'gender': args.gender,
        'noise_type': args.noise_type,
        'text_folder': os.path.join(corpora_folder, args.experiment_name, 'text', args.gender),
        'export_folder': os.path.join(corpora_folder, args.experiment_name, 'clean-transcription/', args.gender)
    }

    return config

def parse_arguments():
    """
    Parse command line arguments.
    
    Returns
    -------
    Namespace
        The arguments parsed from the command line.
    """
    parser = argparse.ArgumentParser(description="Arg parser")
    parser.add_argument('-e', '--experiment_name', type=str, required=True, help='Experiment Name')
    parser.add_argument('-g', '--gender', type=str, required=True, 
                        choices=['male','female'],
                        help='Gender')
    parser.add_argument('-n', '--noise_type', type=str, required=True, 
                        choices=['babble-noise','speech-shaped-noise','white-noise'],
                        help='Noise type')
    return parser.parse_args()

def main():

    args = parse_arguments()
    configs = load_configs(args)
    file_paths = glob.glob(os.path.join(configs['text_folder'], '*'))
    
    for audio_file in tqdm(file_paths):
        phonemes = phonemize(audio_file)
        filename = audio_file.split('/')[-1].replace('.txt','')
        export_file = os.path.join(configs['export_folder'], f'{filename}.txt')
        with open(export_file, 'w') as file:
            file.write(phonemes)

if __name__ == '__main__':
    main()