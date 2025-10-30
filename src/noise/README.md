# Noise

### White noise

This script generates White Noise, i.e., a random signal with equal intensity at all frequencies, having a constant power spectral density.
It can be used for evaluating noise robustness in speech enhancement, denoising, or auditory masking experiments.

To execute the script from the command line, use the following command:

```shell
python3 src/data_preparation/noise/white_noise.py \
  --duration <duration_in_seconds> \
  --sample_rate <sampling_rate_in_hz> \
  --output_dir <path/to/output/folder>

```

### Speech-shaped noise

This script generates Speech-Shaped Noise (SSN) from a clean speech signal by preserving its spectral envelope and randomizing its phase. It is typically used to create noise signals that share the same long-term spectrum as human speech, useful for speech enhancement, intelligibility, or masking experiments. 

To execute the script from the command line, use the following command:

```shell
python3 src/data_preparation/noise/speech_shaped_noise.py \
  --speech_path <path/to/speech_signal.wav> \
  --output_dir <path/to/output/folder>
```

In this study, we used speech recordings from 5 male and 5 female speakers to generate the speech-shaped noise. To create the input signal, we simply concatenated the speech segments of all 10 speakers sequentially into a single waveform.