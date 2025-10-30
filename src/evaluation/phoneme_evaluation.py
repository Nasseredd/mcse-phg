import os 
import sys
import json 
import glob
import argparse
import librosa
import numpy as np
import pandas as pd
from tqdm import tqdm
from siw_snr import fwSNRseg
from mir_eval.separation import bss_eval_images

# Constants
COLORS = {
    "blue": "\033[34m",
    "red": "\033[31m",
    "default": "\033[0m"
}

def load_phoneme_categories():
    phoneme_categories_file = '/home/nasmonir/PhD/se-up-evaluations/configs/phoneme_segmentation/phoneme-classes.json'
    with open(phoneme_categories_file, 'r') as f:
        phoneme_categories = list(json.load(f).keys())
    return phoneme_categories

def update_evaluation_file(phcat_attributes, configs):

    evaluation_file_columns = configs['evaluation_file_columns']
    phcat_attributes_line = ','.join([str(phcat_attributes[col]) for col in evaluation_file_columns]) + '\n' # We use evaluation_file_columns to make sure it keeps the same order

    # Update the evaluation file with the phcat attributes
    with open(f"{configs['evaluation_file']}", 'a') as file:
        file.write(phcat_attributes_line)

def create_phcat_attributes(phcat_metrics, microphone, phoneme_category, configs):

    # Package results into a dictionary
    phcat_attributes = {
            'MODEL': configs['model'],
            'SCENARIO': configs['scenario'],
            'MICROPHONE': microphone,
            'GENDER': configs['gender'],
            'INSTANCE_ID': configs['speaker'],
            'CATEGORY': phoneme_category,
            }

    # Update with phoneme category metrics
    for metric_name, metric_value in phcat_metrics.items():
        phcat_attributes[metric_name] = round(metric_value, 4)

    return phcat_attributes

def compute_phcat_metrics(speech, noise, mixture, estimated_speech, sample_rate=16000):
    """
    
    """
    reference      = np.vstack([speech, noise])
    residual_noise = mixture - estimated_speech
    estimation_in  = np.vstack([mixture, mixture])
    estimation_out = np.vstack([estimated_speech, residual_noise])

    sdr_in, _, sir_in, sar_in, _    = bss_eval_images(reference_sources=reference, estimated_sources=estimation_in, compute_permutation=False)
    sdr_out, _, sir_out, sar_out, _ = bss_eval_images(reference_sources=reference, estimated_sources=estimation_out, compute_permutation=False)

    fwSIR_out = fwSNRseg(speech.flatten(), estimated_speech.flatten(), sample_rate)

    metrics = {
        'SDR_in': sdr_in[0],
        'SDR_out': sdr_out[0],
        'DELTA_SDR': sdr_out[0] - sdr_in[0],
        'SIR_in': sir_in[0],
        'SIR_out': sir_out[0],
        'DELTA_SIR': sir_out[0] - sir_in[0],
        'SAR_in': sar_in[0],
        'SAR_out': sar_out[0],
        'DELTA_SAR': sar_out[0] - sar_in[0],
        'FW-SIR_out': fwSIR_out
    }
    return metrics

def select_phcat_from_audios(
        speech_phcat, noise_phcat, mixture_phcat, estimated_speech_phcat, 
        speech, noise, mixture, estimated_speech, 
        start, end):
    
    start_sample = int(start * 16000)
    end_sample = int(end * 16000)

    speech_phcat = np.concatenate([speech_phcat, speech[:, start_sample:end_sample+1]], axis=1)
    noise_phcat = np.concatenate([noise_phcat, noise[:, start_sample:end_sample+1]], axis=1)
    mixture_phcat = np.concatenate([mixture_phcat, mixture[:, start_sample:end_sample+1]], axis=1)
    estimated_speech_phcat = np.concatenate([estimated_speech_phcat, estimated_speech[:, start_sample:end_sample+1]], axis=1)

    return speech_phcat, noise_phcat, mixture_phcat, estimated_speech_phcat

def retrieve_phonemes(phoneme_seg_file):
    '''Takes phoneme segmentation json file of a speech audio, and outputs a dictionary of phonemes and their corresponding timeframes.'''
    # Load phoneme segmentations 
    with open(phoneme_seg_file, 'r') as f:
        phoneme_segs = json.load(f)['segmentation']
    
    phoneme_segs = [(s[0], s[1], s[2], s[3]) for s in phoneme_segs]

    return phoneme_segs

def load_audio_and_estimated_speech(file_paths):
    # Load audio
    speech = librosa.load(file_paths['speech'], sr=None, mono=False)[0][[0]]
    noise = librosa.load(file_paths['noise'], sr=None, mono=False)[0][[0]]
    mixture = librosa.load(file_paths['mixture'], sr=None, mono=False)[0][[0]]
    estimated_speech = librosa.load(file_paths['estimated_speech_ch0'], sr=None, mono=False)[0]

    return speech.reshape(1,-1), noise.reshape(1,-1), mixture.reshape(1,-1), estimated_speech.reshape(1,-1)

def load_paths(clean_speech_file, configs):
    
    instance_id = os.path.basename(os.path.splitext(clean_speech_file)[0])
    file_paths = {
        'instance_id'           : instance_id,
        'phoneme_segmentations' : os.path.join(configs['phoneme_segmentation_folder'], f'{instance_id}.json'),
        'speech'                : os.path.join(configs['mix_folder'], instance_id, 'reverberated_speech.wav'),
        'noise'                 : os.path.join(configs['mix_folder'], instance_id, 'reverberated_scaled_noise.wav'),
        'mixture'               : os.path.join(configs['mix_folder'], instance_id, 'mixture.wav'),
        'estimated_speech_ch0'  : os.path.join(configs['enhancement_folder'], instance_id, 'estimated_speech_ch0.wav'),
        'estimated_speech_ch2'  : os.path.join(configs['enhancement_folder'], instance_id, 'estimated_speech_ch2.wav'),
    }

    return file_paths

def initialize_phoneme_cat_audios():
    # Initialize
    speech_phcat           = np.empty((1,1))
    noise_phcat            = np.empty((1,1))
    mixture_phcat          = np.empty((1,1))
    estimated_speech_phcat = np.empty((1,1))

    return speech_phcat, noise_phcat, mixture_phcat, estimated_speech_phcat

def compute_and_export_all_phoneme_cat_metrics(clean_speech_files, configs):
    
    phoneme_categories = configs['phoneme_categories'] # Load phoneme categories
    
    for phoneme_category in phoneme_categories:
        speech_phcat, noise_phcat, mixture_phcat, estimated_speech_phcat = initialize_phoneme_cat_audios()
    
        for clean_speech_file in tqdm(clean_speech_files, desc=phoneme_category):
            file_paths = load_paths(clean_speech_file, configs)
            speech, noise, mixture, estimated_speech = load_audio_and_estimated_speech(file_paths)
            phonemes_timeframes = retrieve_phonemes(file_paths['phoneme_segmentations'])

            # Loop through phonemes
            for start, end, _, current_phoneme_category in phonemes_timeframes:
                if current_phoneme_category == phoneme_category:

                    # Select samples corresponding to the current phoneme category from audios 
                    speech_phcat, noise_phcat, mixture_phcat, estimated_speech_phcat = select_phcat_from_audios(
                        speech_phcat, noise_phcat, mixture_phcat, estimated_speech_phcat, 
                        speech, noise, mixture, estimated_speech, 
                        start, end
                    )

        # Compute metrics of the current phoneme category
        metrics_phcat = compute_phcat_metrics(
            speech=speech_phcat, 
            noise=noise_phcat, 
            mixture=mixture_phcat, 
            estimated_speech=estimated_speech_phcat
        )

        # 
        phcat_attributes = create_phcat_attributes(
            phcat_metrics=metrics_phcat, 
            microphone='Front-Left', 
            phoneme_category=phoneme_category, 
            configs=configs
            )
            
        update_evaluation_file(phcat_attributes, configs)

def create_evaluation_file(args):
     
    corpora = '/srv/storage/talc3@storage4.nancy.grid5000.fr/multispeech/calcul/users/nmonir/corpora/'
    evaluation_file_columns = ['MODEL', 'SCENARIO', 'MICROPHONE', 'GENDER', 'INSTANCE_ID', 'CATEGORY', 'SDR_in', 'SDR_out', 'DELTA_SDR', 'SIR_in', 'SIR_out', 'DELTA_SIR', 'SAR_in', 'SAR_out', 'DELTA_SAR', 'FW-SIR_out']
    header = ','.join(evaluation_file_columns) + '\n'
    
    # Check if the file exists
    evaluation_file = os.path.join(corpora, args.experiment_name, 'evaluation', args.gender, args.noise_type, args.model, args.scenario, 'phoneme_level_evaluations.csv')
    with open(evaluation_file, 'w') as file:
            file.write(header)

def load_clean_speech_files(speaker, config):
    clean_speech_files = glob.glob(os.path.join(config['clean_folder'], f"{speaker}*.wav"))
    return list(clean_speech_files)

def load_configs(speaker, args):

    corpora = '/srv/storage/talc3@storage4.nancy.grid5000.fr/multispeech/calcul/users/nmonir/corpora/'
    configs = {
        'model': args.model,
        'gender': args.gender,
        'scenario': args.scenario,
        'noise_type': args.noise_type,
        'speaker': speaker,
        'phoneme_categories': load_phoneme_categories(),
        'clean_folder': os.path.join(corpora, args.experiment_name, 'clean', args.gender),
        'mix_folder': os.path.join(corpora, args.experiment_name, 'mix', args.gender, args.noise_type, args.scenario),
        'enhancement_folder': os.path.join(corpora, args.experiment_name, 'enhancement', args.gender, args.noise_type, args.model, args.scenario),
        'phoneme_segmentation_folder': os.path.join(corpora, args.experiment_name, 'phoneme_segmentation', args.gender),
        'evaluation_file': os.path.join(corpora, args.experiment_name, 'evaluation', args.gender, args.noise_type, args.model, args.scenario, 'phoneme_level_evaluations.csv'),
        'metrics_list': ['SDR','SAR','SIR','PESQ','STOI','HASPI','FW-SIR'],
        'evaluation_file_columns': ['MODEL', 'SCENARIO', 'MICROPHONE', 'INSTANCE_ID', 'CATEGORY', 'SDR_in', 'SDR_out', 'DELTA_SDR', 'SIR_in', 'SIR_out', 'DELTA_SIR', 'SAR_in', 'SAR_out', 'DELTA_SAR', 'FW-SIR_out']
    }
    return configs

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
    parser.add_argument('-m', '--model', type=str, required=True, 
                        choices=['tango','mvdr','fasnet'],
                        help='Model')
    parser.add_argument('-n', '--noise_type', type=str, required=True, 
                        choices=['babble-noise','speech-shaped-noise','white-noise'],
                        help='Noise type')
    parser.add_argument('-s', "--scenario", type=str, required=True, 
                        choices=['0dBS0N45', '0dBS0N90', 'p5dBS0N45', 'p5dBS0N90', 'm5dBS0N45', 'm5dBS0N90'],
                        help="Scenario tag")
    return parser.parse_args()

def main():
    """
    
    """
    args = parse_arguments()

    corpora_folder = '/srv/storage/talc3@storage4.nancy.grid5000.fr/multispeech/calcul/users/nmonir/corpora/'
    clean_files = glob.glob(os.path.join(corpora_folder, args.experiment_name, 'clean', args.gender, '*.wav'))
    speakers = list(set([clean_file.split('/')[-1].split('-')[0] for clean_file in clean_files]))
    
    print(speakers)

    create_evaluation_file(args)

    for speaker in speakers:
        print(f'SPEAKER {speaker}')
        configs = load_configs(speaker, args)
        clean_speech_files = load_clean_speech_files(speaker, configs)
        compute_and_export_all_phoneme_cat_metrics(clean_speech_files, configs)


if __name__ == '__main__':
    main()