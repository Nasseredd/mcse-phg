#!/usr/bin/env python
"""
This module generates estimated speech using the FASNET Beamformer.
"""

# Standard Library Imports
import os
import sys
import json
import glob
import argparse

# Third-party Library Imports
import librosa
import soundfile as sf
from tqdm import tqdm

# Local Imports
from fasnet_espnet_single import fasnet_loading, fasnet_inference

# Constants
COLORS = {
    "blue": "\033[34m",
    "red": "\033[31m",
    "default": "\033[0m"
}

def parse_arguments():
    """
    Parse command line arguments.

    Returns
    -------
    Namespace
        The arguments parsed from the command line.
    """
    parser = argparse.ArgumentParser(description='Process and enhance audio files using FASNET Beamforming.')
    parser.add_argument('-e', '--experiment_name', type=str, required=True, help='Experiment Name')
    parser.add_argument('-g', '--gender', type=str, required=True, 
                        choices=['male','female'],
                        help='Gender')
    parser.add_argument('-d', '--device', type=str, default='cpu', 
                        help='Specifies the compute device for the FASNET Beamformer. Options are "cpu" or "cuda" for GPU. Default is "cpu".')
    parser.add_argument('-n', '--noise_type', type=str, required=True, 
                        choices=['babble-noise','speech-shaped-noise','white-noise'],
                        help='Noise type')
    parser.add_argument('-s', "--scenario", type=str, required=True, 
                        choices=['0dBS0N45', '0dBS0N90', 'p5dBS0N45', 'p5dBS0N90', 'm5dBS0N45', 'm5dBS0N90'],
                        help="Scenario tag")
    return parser.parse_args()

def load_configs(args):
    """
    Load ...

    Parameters
    ----------

    Returns
    -------
    """
    corpora_folder = '/srv/storage/talc3@storage4.nancy.grid5000.fr/multispeech/calcul/users/nmonir/corpora/'
    configs = {
        'scenario': args.scenario,
        'device': args.device,
        'noise_type': args.noise_type,
        'gender': args.gender,
        'mix_folder': os.path.join(corpora_folder, args.experiment_name, 'mix', args.gender, args.noise_type, args.scenario),
        'enhancement_folder': os.path.join(corpora_folder, args.experiment_name, 'enhancement', args.gender, args.noise_type, 'fasnet', args.scenario),
    }
    return configs

def create_and_save_estimated_speech_file(mixture_file, enhancements_folder, fasnet_beamformer, reference_channel, scenario):
    """
    Enhance speech from a mixture file using the FASNET Beamformer and save the result.
    """
    mixture_id = os.path.basename(os.path.dirname(mixture_file))
    output_folder = os.path.join(enhancements_folder, mixture_id)
    enhanced_mixture_file = os.path.join(output_folder, f"estimated_speech_ch{reference_channel}.wav")

    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    if os.path.exists(enhanced_mixture_file):
        print(COLORS["red"] + f"File {enhanced_mixture_file} already exists. Skipping..." + COLORS["default"])
        return
    
    mixture, sample_rate = librosa.load(mixture_file, sr=None, mono=False)
    enhanced_mixture = fasnet_inference(mixture=mixture, samplerate=sample_rate, enh_model_mc=fasnet_beamformer)
    sf.write(enhanced_mixture_file, enhanced_mixture, sample_rate)

def process_scenario(scenario, mixtures_folder, enhancements_folder, beamformers, ref_channels):
    """
    Process all mixture files for a specific scenario using FASNET Beamformer.
    """
    print(COLORS["blue"] + f"[START] Enhancing mixtures for scenario {scenario}..." + COLORS["default"])
    mixture_files = glob.glob(os.path.join(mixtures_folder, "*/mixture.wav"))

    for mixture_file in tqdm(mixture_files, desc="Enhancing"):
        for beamformer, channel in zip(beamformers, ref_channels):
            create_and_save_estimated_speech_file(mixture_file, enhancements_folder, beamformer, channel, scenario)

    print(COLORS["blue"] + f"[END] Enhancements for scenario {scenario} completed." + COLORS["default"])

def main():
    """
    Main function to execute the module workflow.
    """
    args = parse_arguments()
    configs = load_configs(args)
    ref_channels = [0, 2]
    fasnet_beamformers = [fasnet_loading(device=configs['device'], ref_channel=ch) for ch in ref_channels]

    process_scenario(
        scenario=configs['scenario'], 
        mixtures_folder=configs['mix_folder'], 
        enhancements_folder=configs['enhancement_folder'], 
        beamformers=fasnet_beamformers, 
        ref_channels=ref_channels
    )

if __name__ == '__main__':
    main()
