import os 
import sys
import glob
import argparse
import librosa
import numpy as np
import pandas as pd
from tqdm import tqdm
from pesq import pesq
from pystoi import stoi
from siw_snr import fwSNRseg
from clarity.utils.audiogram import Audiogram
from mir_eval.separation import bss_eval_images
from clarity.evaluator.haspi.haspi import haspi_v2

# Constants
COLORS = {
    "blue": "\033[34m",
    "red": "\033[31m",
    "default": "\033[0m"
}

def load_clean_speech_files(config):
    clean_speech_files = glob.glob(os.path.join(config['clean_folder'], "*.wav"))
    return list(clean_speech_files)

def load_paths(clean_speech_file, configs):
    instance_id = os.path.basename(os.path.splitext(clean_speech_file)[0])
    file_paths = {
        'instance_id'           : instance_id,
        'speech'                : os.path.join(configs['mix_folder'], instance_id, 'reverberated_speech.wav'),
        'noise'                 : os.path.join(configs['mix_folder'], instance_id, 'reverberated_scaled_noise.wav'),
        'mixture'               : os.path.join(configs['mix_folder'], instance_id, 'mixture.wav'),
        'estimated_speech_ch0'  : os.path.join(configs['enhancement_folder'], instance_id, 'estimated_speech_ch0.wav'),
        'estimated_speech_ch2'  : os.path.join(configs['enhancement_folder'], instance_id, 'estimated_speech_ch2.wav'),
    }

    return file_paths

def load_audios(speech_path, noise_path, mixture_path, estimated_speech_ch0_path, estimated_speech_ch2_path):
    # Load audio
    speech = librosa.load(speech_path, sr=None, mono=False)[0][[0, 2]]
    noise = librosa.load(noise_path, sr=None, mono=False)[0][[0, 2]]
    mixture = librosa.load(mixture_path, sr=None, mono=False)[0][[0, 2]]

    # Load estimated speech
    estimated_speech_ch0, sr = librosa.load(estimated_speech_ch0_path, sr=None, mono=False)
    estimated_speech_ch2, _ = librosa.load(estimated_speech_ch2_path, sr=None, mono=False)

    # Combine estimated speech channels
    estimated_speech = np.vstack((estimated_speech_ch0, estimated_speech_ch2))

    return speech, noise, mixture, estimated_speech

def load_audios_single_channel(speech_path, noise_path, mixture_path, estimated_speech_ch0_path):
    # Load audio
    speech  = librosa.load(speech_path, sr=None, mono=False)[0][[0]]
    noise   = librosa.load(noise_path, sr=None, mono=False)[0][[0]]
    mixture = librosa.load(mixture_path, sr=None, mono=False)[0][[0]]
    estimated_speech, _ = librosa.load(estimated_speech_ch0_path, sr=None, mono=False)

    return speech.reshape(1, -1), noise.reshape(1, -1), mixture.reshape(1, -1), estimated_speech.reshape(1, -1)

def compute_pesq(speech, mixture, estimated_speech, sample_rate):
    
    pesq_in = []
    pesq_out = []
    delta_pesq = []
    
    for channel in range(speech.shape[0]):
        pesq_in.append(pesq(sample_rate, speech[channel, :], mixture[channel, :], 'wb'))
        pesq_out.append(pesq(sample_rate, speech[channel, :], estimated_speech[channel, :], 'wb'))
        delta_pesq.append(pesq_out[channel] - pesq_in[channel])

    return np.array(pesq_in), np.array(pesq_out), np.array(delta_pesq)

def compute_stoi(speech, mixture, estimated_speech, sample_rate):
    
    stoi_in = []
    stoi_out = []
    delta_stoi = []
    
    for channel in range(speech.shape[0]):
        stoi_in.append(stoi(speech[channel, :], mixture[channel, :], sample_rate, extended=False))
        stoi_out.append(stoi(speech[channel, :], estimated_speech[channel, :], sample_rate, extended=False))
        delta_stoi.append(stoi_out[channel] - stoi_in[channel])

    return np.array(stoi_in), np.array(stoi_out), np.array(delta_stoi)

def haspi(reference_signal, processed_signal, sample_rate):
    # Define a dummy audiogram for normal hearing (no hearing loss)
    # The audiogram is expected to be a numpy array with hearing loss at 6 audiometric frequencies: [250, 500, 1000, 2000, 4000, 6000] Hz
    hearing_loss = np.zeros(6)
    frequencies = np.array([250, 500, 1000, 2000, 4000, 6000])

    # Create an Audiogram object
    audiogram = Audiogram(levels=hearing_loss, frequencies=frequencies)

    # Initialize lists to store results for each channel
    haspi_scores = []

    # Compute HASPI for each channel
    for channel in range(reference_signal.shape[0]):
        haspi_score, _ = haspi_v2(
            reference=reference_signal[channel],
            reference_sample_rate=sample_rate,
            processed=processed_signal[channel],
            processed_sample_rate=sample_rate,
            audiogram=audiogram,
            level1=65.0,
            f_lp=320.0,
            itype=0
        )
        haspi_scores.append(haspi_score)

    return np.array(haspi_scores)

def compute_haspi(speech, mixture, estimated_speech, sample_rate):
    
    haspi_in = []
    haspi_out = []
    delta_haspi = []
    
    for channel in range(speech.shape[0]):
        haspi_in.extend(haspi(speech[channel, :].reshape(1,-1), mixture[channel, :].reshape(1,-1), sample_rate))
        haspi_out.extend(haspi(speech[channel, :].reshape(1,-1), estimated_speech[channel, :].reshape(1,-1), sample_rate))
        delta_haspi.append(haspi_out[channel] - haspi_in[channel])

    return np.array(haspi_in), np.array(haspi_out), np.array(delta_haspi)

def compute_metrics(speech, noise, mixture, estimated_speech, sample_rate, metrics_list, mic2micname):
    
    #mic2micname = {0: 'Front-Left', 1: 'Front-Right'}
    metrics = {micname:{} for micname in mic2micname.values()}

    if 'SIR' in metrics_list or 'SIR' in metrics_list or 'SIR' in metrics_list:
        reference      = np.vstack([speech, noise])
        residual_noise = mixture - estimated_speech
        estimation_in  = np.vstack([mixture, mixture])
        estimation_out = np.vstack([estimated_speech, residual_noise])
        
        sdr_in, _, sir_in, sar_in, _    = bss_eval_images(reference_sources=reference, estimated_sources=estimation_in, compute_permutation=False)
        sdr_out, _, sir_out, sar_out, _ = bss_eval_images(reference_sources=reference, estimated_sources=estimation_out, compute_permutation=False)

        delta_sdr = sdr_out - sdr_in
        delta_sir = sir_out - sir_in
        delta_sar = sar_out - sar_in

        for mic, micname in mic2micname.items():            
            metrics[micname].update({
                "MICROPHONE": micname,
                "SDR_in": round(sdr_in[mic], 2), 
                "SDR_out": round(sdr_out[mic], 2), 
                "DELTA_SDR": round(delta_sdr[mic], 2), 
                "SIR_in": round(sir_in[mic], 2), 
                "SIR_out": round(sir_out[mic], 2), 
                "DELTA_SIR": round(delta_sir[mic], 2), 
                "SAR_in": round(sar_in[mic], 2), 
                "SAR_out": round(sar_out[mic], 2), 
                "DELTA_SAR": round(delta_sar[mic], 2),
            })

    # Calculate PESQ
    if 'PESQ' in metrics_list:
        pesq_in, pesq_out, delta_pesq = compute_pesq(speech, mixture, estimated_speech, sample_rate)
        for mic, micname in mic2micname.items():
            metrics[micname].update({
                "MICROPHONE": micname, 
                "PESQ_in": round(pesq_in[mic], 2),
                "PESQ_out": round(pesq_out[mic], 2),
                "DELTA_PESQ": round(delta_pesq[mic], 2),
            })

    # Calculate STOI
    if 'STOI' in metrics_list:
        stoi_in, stoi_out, delta_stoi = compute_stoi(speech, mixture, estimated_speech, sample_rate)
        for mic, micname in mic2micname.items():
            metrics[micname].update({
                "MICROPHONE": micname, 
                "STOI_in": round(stoi_in[mic], 2),
                "STOI_out": round(stoi_out[mic], 2),
                "DELTA_STOI": round(delta_stoi[mic], 2),
            })
        
    # Calculate HASPI 
    if 'HASPI' in metrics_list:
        haspi_in, haspi_out, delta_haspi = compute_haspi(speech, mixture, estimated_speech, sample_rate)
        for mic, micname in mic2micname.items():
            metrics[micname].update({
                "MICROPHONE": micname,
                "HASPI_in": round(haspi_in[mic], 2),
                "HASPI_out": round(haspi_out[mic], 2),
                "DELTA_HASPI": round(delta_haspi[mic], 2),
            })
        
    # Calculate FW-SIR 
    if 'FW-SIR' in metrics_list:
        for mic, micname in mic2micname.items():
            speech_mic = speech[mic, :].flatten()
            estimated_speech_mic = estimated_speech[mic, :].flatten()
            fwSIR_out = fwSNRseg(speech_mic, estimated_speech_mic, sample_rate)
            metrics[micname].update({
                'FW-SIR_out': round(fwSIR_out, 2)
            })
        
    return metrics

def create_instance_attributes(instance_metrics, instance_id, configs):

    # Package results into a dictionary
    instance_attributes = {
            'MODEL': configs['model'],
            'SCENARIO': configs['scenario'],
            'MICROPHONE': 'Front-Left',
            'INSTANCE_ID': instance_id,
            'CATEGORY': 'utterance',
            }
    
    # Choose microphone index 
    # if microphone == 'Front-Left': mic_idx = 0
    # else: mic_idx = 1

    # Update with instance metrics
    for metric_name, metric_value in instance_metrics.items():
        instance_attributes[metric_name] = metric_value

    return instance_attributes

def compute_and_export_instance_metrics(clean_speech_file, configs):

    # Get file paths
    file_paths = load_paths(clean_speech_file, configs)

    # Load audios
    speech, noise, mixture, estimated_speech = load_audios_single_channel(
        speech_path=file_paths['speech'], 
        noise_path=file_paths['noise'], 
        mixture_path=file_paths['mixture'], 
        estimated_speech_ch0_path=file_paths['estimated_speech_ch0']
    )

    # Compute metrics and display results
    sample_rate = 16000
    metrics_list = configs['metrics_list']
    mic2micname = {0: 'Front-Left'}
    instance_metrics = compute_metrics(speech, noise, mixture, estimated_speech, sample_rate, metrics_list, mic2micname)
    
    # Save (only Front-Left)
    instance_attributes = create_instance_attributes(
        instance_metrics=instance_metrics['Front-Left'], 
        instance_id=file_paths['instance_id'], 
        configs=configs
    )
    update_evaluation_file(instance_attributes, configs)

def compute_and_export_all_instances_metrics(clean_speech_files, configs):

    model = configs['model']
    scenario = configs['scenario']
    noise_type = configs['noise_type']
    print(f'[INFO] Model: {model} --- Scenario: {scenario} --- Noise Type: {noise_type}') 
    
    # Compute metrics of all files
    for clean_speech_file in tqdm(clean_speech_files):
        compute_and_export_instance_metrics(clean_speech_file, configs)

def create_evaluation_file(configs):
     
    evaluation_file_columns = configs['evaluation_file_columns']
    header = ','.join(evaluation_file_columns) + '\n'

    
    # Check if the file exists
    evaluation_file = configs['evaluation_file']
    
    
    # Ensure the directory exists
    directory = os.path.dirname(evaluation_file)
    if not os.path.exists(directory):
        #os.makedirs(directory)
        print(directory)
        print("ERROR: the directory of the evaluation file does not exist!")
    
    with open(evaluation_file, 'w') as file:
            file.write(header)

def update_evaluation_file(instance_attributes, configs):

    evaluation_file_columns = configs['evaluation_file_columns']
    instance_attributes_line = ','.join([str(instance_attributes[col]) for col in evaluation_file_columns]) + '\n' # We use evaluation_file_columns to make sure it keeps the same order
    print(instance_attributes_line)

    # Update the evaluation file with the instance attributes
    with open(f"{configs['evaluation_file']}", 'a') as file:
        file.write(instance_attributes_line)

def load_configs(args):

    corpora_folder = '/srv/storage/talc3@storage4.nancy.grid5000.fr/multispeech/calcul/users/nmonir/corpora/'
    configs = {
        'model': args.model,
        'gender': args.gender,
        'scenario': args.scenario,
        'noise_type': args.noise_type,
        'metrics_list': ['SDR','SAR','SIR','PESQ','STOI','HASPI','FW-SIR'], #RESTORE
        'evaluation_file_columns': ['MODEL', 'SCENARIO', 'MICROPHONE', 'INSTANCE_ID', 'CATEGORY', 'SDR_in', 'SDR_out', 'DELTA_SDR', 'SIR_in', 'SIR_out', 'DELTA_SIR', 'SAR_in', 'SAR_out', 'DELTA_SAR', 'PESQ_in', 'PESQ_out', 'DELTA_PESQ', 'STOI_in', 'STOI_out', 'DELTA_STOI', 'HASPI_in', 'HASPI_out', 'DELTA_HASPI', 'FW-SIR_out'], #RESTORE
        'clean_folder': os.path.join(corpora_folder, args.experiment_name, 'clean', args.gender),
        'mix_folder': os.path.join(corpora_folder, args.experiment_name, 'mix', args.gender, args.noise_type, args.scenario),
        'enhancement_folder': os.path.join(corpora_folder, args.experiment_name, 'enhancement', args.gender, args.noise_type, args.model, args.scenario),
        'evaluation_file': os.path.join(corpora_folder, args.experiment_name, 'evaluation', args.gender, args.noise_type, args.model, args.scenario, 'utterance_level_evaluations.csv'),
        # 'metrics_list': ['FW-SIR'], # TMP
        # 'evaluation_file': os.path.join(corpora_folder, args.experiment_name, 'evaluation', args.gender, args.noise_type, args.model, args.scenario, 'utterance_level_evaluations_fwsir.csv'), # TMP
        # 'evaluation_file_columns': ['MODEL', 'SCENARIO', 'MICROPHONE', 'INSTANCE_ID', 'CATEGORY', 'FW-SIR_out'] # TMP
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
    configs = load_configs(args)
    instance_ids = load_clean_speech_files(configs)
    create_evaluation_file(configs)
    
    # Compute All Instances Metrics
    compute_and_export_all_instances_metrics(instance_ids, configs)

if __name__ == '__main__':
    main()