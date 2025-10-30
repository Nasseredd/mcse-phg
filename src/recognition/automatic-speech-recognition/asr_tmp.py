import nemo.collections.asr as nemo_asr
import torch

# Check if a GPU is available and use it if possible
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")


# Initialize the ASR model
# asr_model = nemo_asr.models.EncDecCTCModelBPE.from_pretrained(model_name="stt_en_conformer_ctc_large")
asr_model = nemo_asr.models.EncDecRNNTBPEModel.from_pretrained(model_name="stt_en_conformer_transducer_large")
asr_model.to(device)

# Define the path to your audio file
audio_file = "/srv/storage/talc3@storage4.nancy.grid5000.fr/multispeech/calcul/users/nmonir/corpora/segender/enhancement/female/speech-shaped-noise/fasnet/0dBS0N45/121-121726-0000/estimated_speech_ch0.wav"

# Transcribe the audio file
transcription = asr_model.transcribe([audio_file])
print(transcription[0])



"""
import soundfile as sf
from espnet2.bin.asr_inference import Speech2Text
from espnet_model_zoo.downloader import ModelDownloader

# Initialize the model downloader
d = ModelDownloader()

# Download and unpack the model from ESPnet model zoo
model_path = d.download_and_unpack("espnet/simple")

# Initialize the ASR model
model = Speech2Text.from_pretrained(model_path)

# Define the path to your audio file
audio_file = "/srv/storage/talc3@storage4.nancy.grid5000.fr/multispeech/calcul/users/nmonir/corpora/segender/enhancement/female/speech-shaped-noise/fasnet/0dBS0N45/121-121726-0000/estimated_speech_ch0.wav"
audio, rate = sf.read(audio_file)

# Transcribe the audio file
transcription = model(audio)
print(f"Transcription: {' '.join([x['text'] for x in transcription])}")

"""