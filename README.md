# SE-UP-Evaluations

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


















































<!-- 

Speech Enhancement Evaluations at the Utterance and Phoneme category levels

```
<project-resources>: /srv/storage/talc3@storage4.nancy.grid5000.fr/multispeech/calcul/users/nmonir/<project-resources>
```


1. **Setup and Configs**
* Configs
    * Add (or modify) a config file in "src/setup/configs" to setup the project
    * Phoneme categories 
* Setup the project 
    ```shell 
    python3 start_setup.py -c configs/config.json -n <project-name>
    ```
2. Data Collection
    Purpose: Upload data in <project-resources>
    - 2.1. Import clean speech: audios in `<project-resources>/clean` and transcriptions in `<project-resources>/transcriptions`
    - 2.2. Import Babble Noise and 5-10 Speech audios to create the speech shaped noise. Note that the speech audios must be approx. half male and half female.
    - 2.3. Import (or Compute) Room Impulse Responses
3. Data Preparation
    - 3.1. Create speech shaped noise from 5-10 downloaded speech files and export to `<project-resources>/noise/speech-shaped-noise`
        ```shell
            python3 src/data_preparation/noise/speech_shaped_noise.py
        ```
    - 3.2. Create white noise and export to `<project-resources>/noise/white-noise`.
        ```shell
        python3 src/data_preparation/noise/white_noise.py
        ```
    - 3.3. Create mixtures (and reverberated speech and noise) from speech and noise audios.
        * For a single scenario : 
            ```shell
                python3 src/data_preparation/mixture/mixtures.py --experiment_name <EXPERIMENT_NAME> --scenario <SCENARIO> --noise_type <NOISE_TYPE>
            ```
        * For multiple scenarios : 
            ```shell
                source scripts/mixtures/mixture-jobs.sh grvingt psamsea
            ```
--TODO
    3.4. Phoneme Segmentation using MFA
    3.5. Convert the TextGrid file to a JSON file to achieve a better-organized structure for phoneme segmentations.

4. Inference 
    * Inference using MVDR 
        ```shell
        conda activate espnet-se-venv
        ```
        ```shell
        python3 src/models/mvdr-espnet/mvdr_espnet_multiple.py -e <experiment-name> -s <scenario-tag> -n <noise-type>
        ```
        Example 
        ```shell
        python3 src/models/mvdr-espnet/mvdr_espnet_multiple.py -e psamsea -s 0dBS0N45 -n speech-shaped-noise
        ```
    * Inference using FaSNet
        ```shell
        conda activate espnet-se-venv
        ```
        ```shell
        python3 src/models/fasnet-espnet/fasnet_espnet_multiple.py -e <experiment-name> -s <scenario-tag> -n <noise-type>
        ```
        Example 
        ```shell
        python3 src/models/fasnet-espnet/fasnet_espnet_multiple.py -e psamsea -s 0dBS0N45 -n speech-shaped-noise
        ```
    * Inference using Tango
        ```shell
        conda activate tango
        ```
        ```shell
        python3 src/models/tango/tango_multiple.py -e <experiment-name> -s <scenario-tag> -n <noise-type>
        ```
        Example 
        ```shell
        python3 src/models/tango/tango_multiple.py -e psamsea -s 0dBS0N45 -n speech-shaped-noise
        ```
    * All Model's Experiments
        ```shell
        source scripts/models/run_model.sh <EXP_NAME> <SCENARIO> <MODEL> <NOISE_TYPE>
        ```
    * Jobs
        ```shell
        cd logs/models/tango
        source scripts/models/model_jobs.sh  <CLUSTER> <EXP_NAME> <MODEL> <NOISE_TYPE>
        ```
        ```shell
        source scripts/models/model_jobs.sh grvingt psamsea mvdr speech-shaped-noise
        ```

5. Evaluation
    * Utterance level evaluation 
        * Evaluation
            ```shell
            python3 src/evaluation/utterance_evaluation.py -e <EXPERIMENT_NAME> -n <NOISE_TYPE> -s <SCENARIO> -m <MODEL>
            ```

        * Jobs
            ```shell
            source scripts/evaluation/evaluation_job.sh <CLUSTER> <EXPERIMENT_NAME>
            ```
    * Phoneme level evaluation
        * Evaluation
            ```shell
            python3 src/evaluation/utterance_evaluation.py -e <EXPERIMENT_NAME> -n <NOISE_TYPE> -s <SCENARIO> -m <MODEL>
            ```

        * Jobs
            ```shell
            source scripts/evaluation/evaluation_job.sh <CLUSTER> <EXPERIMENT_NAME>
            ``` -->