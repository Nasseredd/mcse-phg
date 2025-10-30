#!/usr/bin/env python
"""
This module includes an illustration of fasnet enhancement using a pre-trained version : 'lichenda/chime4_fasnet_dprnn_tac'
"""

# import standard libraries
import sys 

# import third-party libraries
import glob 
import librosa
import numpy as np
import soundfile as sf
from espnet2.bin.enh_inference import SeparateSpeech
from espnet_model_zoo.downloader import ModelDownloader

# Colors
blue_color = "\033[34m"
default_color = "\033[0m"

# Load FasNet Pretrained Model
def fasnet_loading(device: str, ref_channel: int) -> SeparateSpeech:
    """
    Load the FasNet Beamformer model.

    Parameters
    ----------
    device : str
        The device to use for model loading ('cpu' or 'cuda').
    ref_channel: int
        reference microphone channel

    Returns
    -------
    enh_model_mc : espnet2.enh.SeparateSpeech
        The loaded FasNet Beamformer model.
    """
    # Load the Speech Enhancement model
    print(blue_color + "\n[INFO] Loading FASNET Beamformer ... ")
    tag = "lichenda/chime4_fasnet_dprnn_tac"
    d = ModelDownloader()
    cfg = d.download_and_unpack(tag)
    enh_model_mc = SeparateSpeech(
        train_config=cfg["train_config"],
        model_file=cfg["model_file"],
        # for segment-wise process on long speech
        normalize_segment_scale=False,
        show_progressbar=True,
        ref_channel=ref_channel,
        normalize_output_wav=True,
        device=device
    )

    print("[INFO] FasNet Beamformer loaded\n" + default_color)
    return enh_model_mc

# FasNet Inference
def fasnet_inference(mixture: np.ndarray, samplerate: int, enh_model_mc: SeparateSpeech) -> np.ndarray:
    """
    Perform speech enhancement using the FasNet beamforming model.

    Parameters
    ----------
    mixture : numpy.ndarray
        The input audio mixture of shape (n_samples, n_channels).
    samplerate : int
        The sampling rate of the input audio.
    enh_model_mc : SeparateSpeech
        The loaded FasNet beamforming model.

    Returns
    -------
    numpy.ndarray
        Enhanced audio signal of shape (n_samples,).
    """
    mixture = mixture.T
    wave = enh_model_mc(mixture[None, ...], samplerate)
    return wave[0].T

# Example
if __name__ == "__main__":
    # Specifications 
    device                = "cpu"
    ref_channel_ch0         = int(sys.argv[1])
    ref_channel_ch2         = int(sys.argv[2])
    mixture_file          = sys.argv[3]
    
    # Load mixture
    mixture, sr = librosa.load(mixture_file, sr=None, mono=False) # (samples, channels)
    
    # Load FasNet
    fasnet_beamformer_ch0 = fasnet_loading(device=device, ref_channel=ref_channel_ch0)
    fasnet_beamformer_ch2 = fasnet_loading(device=device, ref_channel=ref_channel_ch2)
    
    # Perform FasNet inference on the mixture
    estimated_speech_ch0 = fasnet_inference(mixture=mixture, samplerate=sr, enh_model_mc=fasnet_beamformer_ch0) # (samples, 1-channel)
    estimated_speech_ch2 = fasnet_inference(mixture=mixture, samplerate=sr, enh_model_mc=fasnet_beamformer_ch2) # (samples, 1-channel)
    
    # Export the enhanced mixture to a .wav file
    sf.write('enhanced_ch0.wav', estimated_speech_ch0, sr)
    sf.write('enhanced_ch2.wav', estimated_speech_ch2, sr)