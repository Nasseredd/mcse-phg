#!/usr/bin/env python
"""
Process audio to generate mixtures with speech and various types of noise
at specified signal-to-noise ratios (SNR), utilizing room impulse responses (RIR).
"""

import os
import sys
import json
import glob
import shutil
import librosa
import argparse
import numpy as np
import soundfile as sf
from tqdm import tqdm
from typing import Dict

# Constants for ANSI color codes for terminal output
COLORS = {
    "blue": "\033[34m",
    "red": "\033[31m",
    "magenta": "\033[35m",
    "cyan": "\033[36m",
    "green": "\033[32m",
    "yellow": "\033[33m",
    "grey": "\033[90m",
    "default": "\033[0m"
}

def output_filenames(scenario_dir: str, speech_id: str) -> Dict[str, str]:
    """
    Generate output filenames for the audio processing results.

    Parameters
    ----------
    scenario_dir : str
        Path to the scenario-specific directory where the outputs will be saved.
    speech_id : str
        Identifier of the speech file being processed (used to create a subfolder).

    Returns
    -------
    dict[str, str]
        Dictionary mapping output file types to their corresponding absolute paths:
        - "speech_reverberated": path to the reverberated speech file (.wav)
        - "noise_reverberated_scaled": path to the reverberated and scaled noise file (.wav)
        - "noise_scaled": path to the scaled noise file (.wav)
        - "mixture": path to the mixture file (.wav)
    """
    output_dir = os.path.join(scenario_dir, speech_id)
    os.makedirs(output_dir, exist_ok=True)

    filenames = {
        "speech_reverberated": os.path.join(output_dir, "reverberated_speech.wav"),
        "noise_reverberated_scaled": os.path.join(output_dir, "reverberated_scaled_noise.wav"),
        "noise_scaled": os.path.join(output_dir, "scaled_noise.wav"),
        "mixture": os.path.join(output_dir, "mixture.wav"),
    }

    return filenames

def extract_snr_from_scenario(scenario):
    """
    Extract the desired SNR from a scenario string.

    Parameters
    ----------
    scenario : str
        Scenario code, e.g., '0dBS0N45', 'p5dBS0N45', 'm5dBS0N45'.

    Returns
    -------
    int
        The desired SNR extracted from the scenario. It returns 0 for '0dB',
        5 for 'p5dB', and -5 for 'm5dB'.
    """
    # Check for presence of SNR markers and extract corresponding values
    if '0dB' in scenario:
        return 0
    elif 'p5dB' in scenario:
        return 5
    elif 'm5dB' in scenario:
        return -5
    else:
        raise ValueError("Invalid scenario format. Expected SNR part not found.")

def extract_noise_angle(scenario):
    """
    Extract noise angle string from a scenario string.

    Parameters
    ----------
    scenario : str
        The scenario string from which the last two characters will be extracted.

    Returns
    -------
    str
        The noise angle of the input string.
    """
    # Validate that the input is sufficiently long
    if len(scenario) < 2:
        raise ValueError("The input string must be at least 2 characters long.")
    
    # Return the noise angle
    return scenario[-2:]

def load_config(args):
    """
    Load configuration settings from a JSON file.

    Parameters
    ----------
    scenario : str
        scenario tag.

    Returns
    -------
    dict
        Configuration settings including paths and SNR settings.
    """
    gender            = args.gender
    scenario          = args.scenario
    noise_type        = args.noise_type
    corpus_dir       = args.corpus_dir
    noise_angle       = extract_noise_angle(scenario)
    desired_snr       = extract_snr_from_scenario(scenario)
    config = {
        'silence-threshold': 0.01,
        'desired-snr': desired_snr,
        'clean_dir': os.path.join(corpus_dir, 'clean', gender),
        'noise_file': os.path.join(corpus_dir, 'noise', noise_type, f'{noise_type}.wav'),
        'speech_rir': os.path.join(corpus_dir, 'room_impulse_response', 'rir_0.npz'),
        'noise_rir': os.path.join(corpus_dir, 'room_impulse_response', f'rir_p{noise_angle}.npz'),
        'scenario_dir': os.path.join(corpus_dir, 'mix', gender, noise_type, scenario),
    }
    
    return config

def adjust_noise_length(speech, noise):
    """
    Adjust the length of the noise signal to match the length of the speech signal.

    Parameters
    ----------
    speech : ndarray
        The speech audio signal.
    noise : ndarray
        The noise audio signal.

    Returns
    -------
    ndarray
        The noise signal adjusted to the length of the speech signal.
    """
    repeat_count   = np.ceil(speech.shape[0] / noise.shape[0])
    adjusted_noise = np.tile(noise, int(repeat_count))[:speech.shape[0]]
    return adjusted_noise

def compute_gain(speech_samples, noise_samples, snr_dB):
    """
    Compute the gain factor needed to adjust the noise level to achieve the desired SNR.

    Parameters
    ----------
    speech_samples : ndarray
        Samples of the speech signal.
    noise_samples : ndarray
        Samples of the noise signal.
    snr_dB : float
        Desired signal-to-noise ratio in decibels.

    Returns
    -------
    float
        The gain factor by which the noise signal should be multiplied.
    """
    speech_power = np.mean(speech_samples**2)
    noise_power  = np.mean(noise_samples**2)
    gain_factor  = np.sqrt(speech_power / (noise_power * 10**(snr_dB / 10)))
    return gain_factor

def load_and_assemble_rir(rir_file_path):
    """
    Load room impulse responses (RIR) from an NPZ file and assemble them into a single array.

    Parameters
    ----------
    rir_file_path : str
        Path to the NPZ file containing the RIR data.

    Returns
    -------
    ndarray
        A 2D NumPy array where each column represents a different RIR channel.

    Notes
    -----
    The NPZ file is expected to contain several arrays corresponding to different RIR channels.
    This function specifically looks for the channels 'phl_left_front', 'phl_left_rear',
    'phl_right_front', and 'phl_right_rear'. It is crucial that these keys exist within the NPZ file.
    """
    # Load the RIR data from the NPZ file
    rir_npz = np.load(rir_file_path)

    # Assemble specific RIR channels into a single array
    try:
        rir = np.stack((
            rir_npz['phl_left_front'],
            rir_npz['phl_left_rear'],
            rir_npz['phl_right_front'],
            rir_npz['phl_right_rear']
        ), axis=-1).T
    except KeyError as e:
        raise KeyError(f"Missing one or more RIR channels in the NPZ file: {e}")

    return rir

def apply_room_impulse_response(signal, rir):
    """
    Apply room impulse response to a signal.

    Parameters
    ----------
    signal : ndarray
        The input audio signal.
    rir : 
        The room impulse response.

    Returns
    -------
    ndarray
        The signal after convolution with the room impulse response.
    """
    channels = [np.convolve(signal, ir_channel, mode="full") for ir_channel in rir]
    reverberated = np.vstack(channels).T
    return reverberated

def prepare_experiment_directory(args):
    """
    Verify if the experiment directory exists and prompt the user before erasing it.

    Parameters
    ----------
    directory : str
        Path to the experiment-level directory (e.g., mix/.../0dBS0N45/).

    Notes
    -----
    - If the directory exists, the user is asked whether to erase it.
    - Typing 'y' or 'yes' will delete and recreate the directory.
    - Any other response will stop the program safely.
    """
    directory = os.path.join(args.corpus_dir, 'mix', args.gender, args.noise_type, args.scenario)

    if os.path.exists(directory):
        response = input(f"The experiment directory '{directory}' already exists.\n"
                         f"Would you like to erase it and continue? [y/n]: ").strip().lower()
        if response in ("y", "yes"):
            shutil.rmtree(directory)
            os.makedirs(directory)
            print(COLORS['green'] + f"[OK] Existing directory '{directory}' was deleted and recreated." + COLORS['default'])
        else:
            print(COLORS['red'] + "[INFO] Mixture generation aborted — the experiment directory already exists." + COLORS['default'])
            sys.exit(0)
    else:
        os.makedirs(directory)
        print(COLORS['green'] + f"[OK] Created experiment directory '{directory}'." + COLORS['default'])

def verifications(config):

    # Verify existance of speech signals at args.corpus_dir/clean 
    if not os.path.exists(config['clean_dir']): 
        sys.exit(COLORS['red'] + f"[KO] Clean directory '{config['clean_dir']}' does not exist." + COLORS['default'])
    else:
        print(COLORS['green'] + f"[OK] Located clean speech directory '{config['clean_dir']}'." + COLORS['default'])

    # Verify existance of noise signals at args.corpus_dir/noise/args.noise_type
    if not os.path.exists(config['noise_file']): 
        sys.exit(COLORS['red'] + f"[KO] Noise file '{config['noise_file']}' does not exist." + COLORS['default'])
    else:
        print(COLORS['green'] + f"[OK] Located noise file '{config['noise_file']}'." + COLORS['default'])

    # Verify existance of RIRs at args.corpus_dir/room_impulse_response
    if not os.path.exists(config['speech_rir']): 
        sys.exit(COLORS['red'] + f"[KO] RIR directory '{config['speech_rir']}' does not exist." + COLORS['default'])
    else:
        print(COLORS['green'] + f"[OK] Located speech RIR file '{config['speech_rir']}'." + COLORS['default'])
    if not os.path.exists(config['noise_rir']): 
        sys.exit(COLORS['red'] + f"[KO] RIR directory '{config['noise_rir']}' does not exist." + COLORS['default'])
    else:
        print(COLORS['green'] + f"[OK] Located noise RIR file '{config['noise_rir']}'.\n" + COLORS['default'])

def main():
    """
    Main function to process audio and generate mixtures.
    """
    args = parse_arguments()
    config = load_config(args)

    # Verifications
    verifications(config)
    prepare_experiment_directory(args)

    # Starting mixtures generation
    print(COLORS['blue'] + "\nStarting mixtures generation..." + COLORS['default'])

    # Load and assemble speech and noise room impulse responses 
    speech_rir = load_and_assemble_rir(config['speech_rir'])
    noise_rir = load_and_assemble_rir(config['noise_rir'])
    
    # Load the noise 
    sample_rate = 16000  # Fixed sample rate
    noise_file   = config['noise_file']
    noise, _  = librosa.load(noise_file, sr=sample_rate, mono=True)
    
    # Load speech files 
    speech_files = glob.glob(os.path.join(config['clean_dir'], '*.wav'))

    # Generate mixtures
    for speech_file in tqdm(speech_files, desc="Processing Speech Files"):
        speech, _ = librosa.load(speech_file, sr=sample_rate, mono=True)

        # Adjust noise length and compute gain
        noise = adjust_noise_length(speech, noise)
        gain_factor = compute_gain(speech, noise, config['desired-snr'])

        # Apply gain and RIRs
        noise *= gain_factor
        speech_reverberated = apply_room_impulse_response(speech, speech_rir)
        noise_reverberated  = apply_room_impulse_response(noise, noise_rir)

        # Create mixture
        mixture = speech_reverberated + noise_reverberated

        # Output filenames and save files
        filenames = output_filenames(config['scenario_dir'], os.path.splitext(os.path.basename(speech_file))[0])
        sf.write(filenames['speech_reverberated'], speech_reverberated, sample_rate)
        sf.write(filenames['noise_reverberated_scaled'], noise_reverberated, sample_rate)
        sf.write(filenames['mixture'], mixture, sample_rate)

    print(COLORS['green'] + "[OK] Mixtures generation completed." + COLORS['default'])

def parse_arguments():
    """
    Parse command line arguments using argparse.

    Returns
    -------
    argparse.Namespace
        The parsed arguments with values for each option.
    """
    parser = argparse.ArgumentParser(description="Process audio to generate mixtures with specified SNR and RIR.")
    parser.add_argument("--corpus_dir", type=str, required=True, help="Directory where you store all the corpora you could use")
    parser.add_argument("--scenario", type=str, required=True, help="Scenario tag",
                        choices=['0dBS0N45', '0dBS0N90', 'p5dBS0N45', 'p5dBS0N90', 'm5dBS0N45', 'm5dBS0N90'])
    parser.add_argument("--gender", type=str, required=True, help="Gender of the speaker.",
                        choices=['male', 'female'])
    parser.add_argument("--noise_type", type=str, required=True, help="Type of noise to use in the mixtures.",
                        choices=['white-noise', 'babble-noise', 'speech-shaped-noise'])
    return parser.parse_args()

if __name__ == "__main__":
    main()