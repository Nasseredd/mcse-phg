# Evaluating Multichannel Speech Enhancement Algorithms at the Phoneme Scale Across Genders

**Authors:**  
Nasser-Eddine Monir, Paul Magron, Romain Serizel  
Université de Lorraine, Inria, CNRS, Loria, Multispeech Team  

**Paper:**  
[Evaluating Multichannel Speech Enhancement Algorithms at the Phoneme Scale Across Genders](https://doi.org/10.48550/arXiv.2506.18691)  
*arXiv:2506.18691 [cs.SD]*  

---

### Overview

This repository contains the code and data preparation scripts used in our paper *“Evaluating Multichannel Speech Enhancement Algorithms at the Phoneme Scale Across Genders”*.  
The study investigates how **gender and phonetic content** affect the performance of **multichannel speech enhancement (SE) algorithms**.  

While most SE evaluations are done at the *utterance* level, this work provides a **phoneme-scale analysis** that reveals:
- Subtle yet systematic **gender-specific spectral differences**,  
- Strong **phoneme-dependent variations** in enhancement quality,  
- Better interference reduction and fewer artifacts for **female speech**, especially for plosives, fricatives, and vowels.

---

### Mixture Generation

This script processes clean speech and noise recordings to generate reverberant mixtures at predefined signal-to-noise ratios (SNRs), using room impulse responses (RIRs). It supports multiple noise types (e.g., speech-shaped noise) and can handle male or female speakers.

The output includes reverberated speech, scaled noise, and the final speech–noise mixture, all saved as `.wav` files.

Before running the mixture generation script, ensure that the following folders and files exist inside your corpus directory (<corpus_dir>):

```markdown
<corpus_dir>/
├── clean/
│   ├── male/
│   └── female/
├── noise/
│   ├── white-noise/
│   │   └── white-noise.wav
│   └── speech-shaped-noise/
│       └── speech-shaped-noise.wav
└── room_impulse_response/
    ├── rir_0.npz
    ├── rir_p45.npz
    └── rir_p90.npz
```

To execute the script from the command line, use the following command:

```shell
python3 src/data_preparation/mixing/generate_mixtures.py \
  --corpus_dir <path/to/corpus_directory> \
  --scenario <scenario_tag> \
  --gender <male_or_female> \
  --noise_type <noise_type>
```

**Arguments:**

- `<path/to/corpus_directory>` — Path to the root corpus directory containing all required subdirectories.
- `<scenario_tag>` — Acoustic scenario specifying SNR and noise position (e.g., 0dBS0N45, p5dBS0N90, m5dBS0N45).
- `<male_or_female>` — Speaker gender: male or female.
- `<noise_type>` — Type of noise to be mixed: white-noise, babble-noise, or speech-shaped-noise.

**Scenario Naming Convention** 

Each scenario tag encodes two key acoustic parameters:
- the signal-to-noise ratio (SNR) between speech and noise, and
- the azimuth angles of the speech (S) and noise (N) sources relative to the listener (0° = in front).

Example: 0dBS0N45
- 0dB $\rightarrow$ The target SNR is 0 decibels, meaning the speech and noise have equal power.
- S0 $\rightarrow$ The speech source is positioned at 0°, i.e., directly in front of the listener (frontal position).
- N45 $\rightarrow$ The noise source is positioned at 45° on the right-hand side of the listener.

### White noise

This script generates white noise, i.e., a random signal with equal intensity at all frequencies, having a constant power spectral density.
It can be used for evaluating noise robustness in speech enhancement, denoising, or auditory masking experiments.

To execute the script from the command line, use the following command:

```shell
python3 src/data_preparation/noise/white_noise.py \
  --duration <duration_in_seconds> \
  --sample_rate <sampling_rate_in_hz> \
  --output_dir <path/to/output/folder>

```

### Speech-shaped noise

This script generates speech-shaped noise (SSN) from a clean speech signal by preserving its spectral envelope and randomizing its phase. It is typically used to create noise signals that share the same long-term spectrum as human speech, useful for speech enhancement, intelligibility, or masking experiments. 

To execute the script from the command line, use the following command:

```shell
python3 src/data_preparation/noise/speech_shaped_noise.py \
  --speech_path <path/to/speech_signal.wav> \
  --output_dir <path/to/output/folder>
```

In this study, we used speech recordings from 5 male and 5 female speakers to generate the speech-shaped noise. To create the input signal, we simply concatenated the speech segments of all 10 speakers sequentially into a single waveform.

















































