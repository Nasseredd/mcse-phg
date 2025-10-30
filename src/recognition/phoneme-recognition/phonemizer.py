import os
import glob
import torch
import argparse
import torchaudio
from tqdm import tqdm
from transformers import Wav2Vec2ForCTC, Wav2Vec2Processor
import subprocess

# Ensure espeak is in the PATH
os.environ["PATH"] += os.pathsep + os.path.expanduser("~/espeak-ng/bin")

def load_model():
    # Load the pre-trained Wav2Vec2 ASR model and processor
    processor = Wav2Vec2Processor.from_pretrained("facebook/wav2vec2-large-960h")
    model = Wav2Vec2ForCTC.from_pretrained("facebook/wav2vec2-large-960h")
    model.eval()
    print()
    return model, processor

def transcribe(audio_file, model, processor):
    # Load the speech signal
    waveform, sample_rate = torchaudio.load(audio_file)
    waveform = waveform.mean(dim=0)  # Convert to mono if stereo

    # Resample if necessary
    if sample_rate != 16000:
        resampler = torchaudio.transforms.Resample(orig_freq=sample_rate, new_freq=16000)
        waveform = resampler(waveform)
        sample_rate = 16000

    # Extract features
    inputs = processor(waveform, sampling_rate=sample_rate, return_tensors="pt", padding=True)
    with torch.no_grad():
        logits = model(inputs.input_values).logits

    # Get predicted IDs and decode to text
    predicted_ids = torch.argmax(logits, dim=-1)
    transcription = processor.batch_decode(predicted_ids)[0]
    return transcription

# Phonemize the text using eSpeak
def audio_to_phonemes(audio_file, model, processor):
    transcription = transcribe(audio_file, model, processor)
    result = subprocess.run(['espeak', '-q', '--ipa', transcription], stdout=subprocess.PIPE)
    phonemes = result.stdout.decode('utf-8').strip()
    return phonemes

def load_configs(args):
    """
    Load configuration settings from a JSON file.

    Parameters
    ----------
    scenario : str
        scenario tag.

    Returns
    -------
    dict
        Configuration settings.
    """

    corpora_folder = '/srv/storage/talc3@storage4.nancy.grid5000.fr/multispeech/calcul/users/nmonir/corpora/'
    config = {
        'gender': args.gender,
        'noise_type': args.noise_type,
        'model': args.model,
        'scenario': args.scenario,
        'clean_folder': os.path.join(corpora_folder, args.experiment_name, 'clean', args.gender),
        'enhancement_folder': os.path.join(corpora_folder, args.experiment_name, 'enhancement', args.gender, args.noise_type, args.model, args.scenario),
        'export_folder': os.path.join(corpora_folder, args.experiment_name, 'enhancement-transcription/phonemizer', args.gender, args.noise_type, args.model, args.scenario)
    }

    return config

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

def load_files(configs):
    file_paths = glob.glob(os.path.join(configs['enhancement_folder'], f'*/estimated_speech_ch0.wav'))
    return file_paths

def main():

    args = parse_arguments()
    configs = load_configs(args); print("Configs Loaded")
    file_paths = load_files(configs); print("Filepaths Loaded")
    model, processor = load_model(); print("Model Loaded")

    for audio_file in tqdm(file_paths):
        phonemes = audio_to_phonemes(audio_file, model, processor)
        filename = audio_file.split('/')[-2]
        export_file = os.path.join(configs['export_folder'], f'{filename}.txt')
        with open(export_file, 'w') as file:
            file.write(phonemes)

if __name__ == '__main__':
    main()