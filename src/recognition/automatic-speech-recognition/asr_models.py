import os
import sys
import glob
import torch
import librosa
import argparse
import torchaudio
from tqdm import tqdm
from transformers import Wav2Vec2ForCTC, Wav2Vec2Processor


class TranscribeWav2Vec:
    def __init__(self, model_name="facebook/wav2vec2-large-960h"):
        self.processor = Wav2Vec2Processor.from_pretrained(model_name)
        self.model = Wav2Vec2ForCTC.from_pretrained(model_name)
        print('[INFO] Wav2Vec (facebook/wav2vec2-large-960h) Loaded.')

    def transcribe(self, audio_file):
        # Load the audio file
        waveform, sample_rate = torchaudio.load(audio_file)

        # Resample the audio if necessary
        if sample_rate != 16000:
            resampler = torchaudio.transforms.Resample(sample_rate, 16000)
            waveform = resampler(waveform)

        # Process the audio
        inputs = self.processor(waveform.squeeze(), sampling_rate=16000, return_tensors="pt", padding=True)

        # Perform inference
        with torch.no_grad():
            logits = self.model(inputs.input_values).logits

        # Decode the predicted ids to text
        predicted_ids = torch.argmax(logits, dim=-1)
        transcription = self.processor.batch_decode(predicted_ids)

        return transcription[0]

class TranscribeCRDNN:
    def __init__(self):
        from speechbrain.inference.ASR import EncoderDecoderASR
        self.model = EncoderDecoderASR.from_hparams(
            source="speechbrain/asr-crdnn-transformerlm-librispeech", 
            savedir="pretrained_models/asr-crdnn-transformerlm-librispeech"
        )

    def trancribe(self, audio_file):
        transcription = self.model.transcribe_file(audio_file)
        return transcription

class TranscribeWhisper:
    def __init__(self):
        import whisper
        self.model = whisper.load_model("base.en")
        print("[INFO] Whisper ASR Model has been loaded!")

    def transcribe(self, audio_file):
        result = self.model.transcribe(audio_file)
        return result["text"].upper()

class TranscribeConformerCTC:
    def __init__(self):
        import nemo.collections.asr as nemo_asr
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = nemo_asr.models.EncDecCTCModelBPE.from_pretrained(model_name="stt_en_conformer_ctc_large")
        self.model.to(self.device)

    def transcribe(self, audio_file):
        transcription = self.model.transcribe([audio_file])
        return transcription[0].upper()

class TranscribeConformerTransducer:
    def __init__(self):
        import nemo.collections.asr as nemo_asr
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = nemo_asr.models.EncDecRNNTBPEModel.from_pretrained(model_name="stt_en_conformer_transducer_large") 
        self.model.to(self.device)

    def transcribe(self, audio_file):
        transcription = self.model.transcribe([audio_file])
        return transcription[0][0].upper()
    

def transcribe(gender, noise_type, model, scenario, asr_model, asr_model_name):
    
    folder = '/srv/storage/talc3@storage4.nancy.grid5000.fr/multispeech/calcul/users/nmonir/corpora/segender'
    filenames = [os.path.basename(file) for file in glob.glob(os.path.join(folder, 'enhancement', gender, noise_type, model, scenario, '*'))]
    
    for filename in tqdm(filenames):
        # Enhancement file
        filepath = os.path.join(folder, 'enhancement', gender, noise_type, model, scenario, filename, "estimated_speech_ch0.wav")
        
        # Output Folder 
        output_folder = os.path.join(folder, 'enhancement-transcription/asr', gender, noise_type, model, scenario, filename)
        if not os.path.exists(output_folder):
            os.makedirs(output_folder)

        # Transcription (output) file
        output_file = os.path.join(output_folder, f'transcription_{asr_model_name}.txt')

        # If transcription file does not exist
        # if os.path.exists(output_file):
        #     print(f"[INFO] {gender}, {noise_type}, {model}, {scenario}, {filename}")
        # else:
        transcription = asr_model.transcribe(filepath)
        with open(output_file, 'w') as file:
            file.write(transcription)
            print(output_file)
            

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
        Configuration settings.
    """

    corpora_folder = '/srv/storage/talc3@storage4.nancy.grid5000.fr/multispeech/calcul/users/nmonir/corpora/'
    config = {
        'gender': args.gender,
        'noise_type': args.noise_type,
        'model': args.model,
        'scenario': args.scenario,
        'asr_model': dictionary_of_models(args.asr_model_name),
        'asr_model_name': args.asr_model_name,
        'clean_folder': os.path.join(corpora_folder, args.experiment_name, 'clean', args.gender),
        'noise_file': os.path.join(corpora_folder, args.experiment_name, 'noise', args.noise_type, f'{args.noise_type}.wav'),
        'scenario_folder': os.path.join(corpora_folder, args.experiment_name, 'mix', args.gender, args.noise_type, args.scenario),
    }
    
    return config

def dictionary_of_models(asr_model_name):
    
    if asr_model_name == 'wav2vec':
        return TranscribeWav2Vec()
    
    elif asr_model_name == 'wav2vec-lv60':
        return TranscribeWav2Vec(model_name="facebook/wav2vec2-large-960h-lv60-self")
    
    elif asr_model_name == 'whisper':
        return TranscribeWhisper()
    
    elif asr_model_name == 'conformer-ctc':
        return TranscribeConformerCTC()
    
    elif asr_model_name == 'conformer-transducer':
        return TranscribeConformerTransducer()
    
    else:
        print('[ERROR] The ASR Model is unknown!')
        return

def parse_arguments():
    """
    Parse command line arguments using argparse.

    Returns
    -------
    argparse.Namespace
        The parsed arguments with values for each option.
    """
    parser = argparse.ArgumentParser(description="Process audio to generate mixtures with specified SNR and RIR.")
    parser.add_argument("--experiment_name", type=str, required=True, help="Experiment Name")
    parser.add_argument("--model", type=str, required=True, 
                        choices=['fasnet', 'tango'],
                        help="Model")
    parser.add_argument("--scenario", type=str, required=True, 
                        choices=['0dBS0N45', '0dBS0N90', 'p5dBS0N45', 'p5dBS0N90', 'm5dBS0N45', 'm5dBS0N90'],
                        help="Scenario tag")
    parser.add_argument("--gender", type=str, required=True, 
                        choices=['male', 'female'],
                        help="Gender")
    parser.add_argument("--noise_type", type=str, required=True, 
                        choices=['white-noise', 'babble-noise', 'speech-shaped-noise'],
                        help="Type of noise to use in the mixtures.")
    parser.add_argument("--asr_model_name", type=str, required=True, 
                        choices=['wav2vec','wav2vec-lv60','whisper','conformer-ctc','conformer-transducer'],
                        help="Model.")
    return parser.parse_args()

def main(configs):

    transcribe(
        gender=configs['gender'], 
        noise_type=configs['noise_type'], 
        model=configs['model'], 
        scenario=configs['scenario'], 
        asr_model=configs['asr_model'], 
        asr_model_name=configs['asr_model_name']
    )

if __name__ == '__main__':

    args = parse_arguments()
    configs = load_config(args)
    main(configs)