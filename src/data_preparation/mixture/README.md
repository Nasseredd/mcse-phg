# Mixture Generation

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