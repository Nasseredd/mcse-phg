#!/usr/bin/env python
"""
This module includes an illustration of mvdr enhancement using a pre-trained version : espnet/Wangyou_Zhang_chime4_enh_train_enh_beamformer_mvdr_raw
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

# Load MVDR Pretrained Model
def mvdr_loading(device: str, ref_channel: int) -> SeparateSpeech:
    """
    Load the MVDR Beamformer model.

    Parameters
    ----------
    device : str
        The device to use for model loading ('cpu' or 'cuda').
    ref_channel: int
        reference microphone channel

    Returns
    -------
    enh_model_mc : espnet2.enh.SeparateSpeech
        The loaded MVDR Beamformer model.
    """
    # Load the Speech Enhancement model
    print(blue_color + "\n[INFO] Loading MVDR Beamformer ... ")
    tag = "espnet/Wangyou_Zhang_chime4_enh_train_enh_beamformer_mvdr_raw"
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

    print("[INFO] MVDR Beamformer loaded\n" + default_color)
    return enh_model_mc

# MVDR Inference
def mvdr_inference(mixture: np.ndarray, samplerate: int, enh_model_mc: SeparateSpeech) -> np.ndarray:
    """
    Perform speech enhancement using the MVDR beamforming model.

    Parameters
    ----------
    mixture : numpy.ndarray
        The input audio mixture of shape (n_samples, n_channels).
    samplerate : int
        The sampling rate of the input audio.
    enh_model_mc : SeparateSpeech
        The loaded MVDR beamforming model.

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
    ref_channel_A         = int(sys.argv[1]) # front-left 0
    ref_channel_B         = int(sys.argv[2]) # front-right 2 
    mixture_file          = sys.argv[3]
    enhanced_mixture_path = sys.argv[4]
    
    # Load mixture
    mixture, sr = librosa.load(mixture_file, sr=None, mono=False) # (samples, channels)
    
    # Load MVDR
    #mvdr_beamformer_A = mvdr_loading(device=device, ref_channel=ref_channel_A)
    #mvdr_beamformer_B = mvdr_loading(device=device, ref_channel=ref_channel_B)
    
    # Perform MVDR inference on the mixture
    #estimated_speech_A = mvdr_inference(mixture=mixture, samplerate=sr, enh_model_mc=mvdr_beamformer_A) # (samples, 1-channel)
    #estimated_speech_B = mvdr_inference(mixture=mixture, samplerate=sr, enh_model_mc=mvdr_beamformer_A) # (samples, 1-channel)
    
    # Export the enhanced mixture to a .wav file
    # estimated_speech = np.concatenate((estimated_speech_A, estimated_speech_B), axis=1)
    #sf.write(enhanced_mixture_path, estimated_speech, sr)