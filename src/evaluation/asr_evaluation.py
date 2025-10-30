import os
import sys
import glob
import argparse
from tqdm import tqdm 
from jiwer import wer

def compute_wer(filename, configs):
    
    asr_model_name = configs['asr_model_name']
    groundtruth_file = os.path.join(configs['corpus'], 'text', configs['gender'], f"{filename}.txt")
    transcription_file = os.path.join(configs['corpus'], 'enhancement-transcription/asr', configs['gender'], configs['noise_type'], configs['model'], configs['scenario'], filename, f'transcription_{asr_model_name}.txt')

    with open(groundtruth_file, 'r') as file:
        ground_truth = file.readline().strip().upper()

    with open(transcription_file, 'r') as file:
        predicted_transcription = file.readline().strip().upper()
    
    error_rate = wer(ground_truth, predicted_transcription)
    
    return {
        'MODEL': configs['model'],
        'GENDER': configs['gender'], 
        'NOISE_TYPE': configs['noise_type'],
        'SCENARIO': configs['scenario'],
        'FILENAME': filename,
        'ASR_MODEL': configs['asr_model_name'],
        'WER': error_rate,
        # 'GROUNDTRUTH': ground_truth,
        # 'TRANSCRIPTION': predicted_transcription
    }

def load_configs(args):

    corpora_folder = '/srv/storage/talc3@storage4.nancy.grid5000.fr/multispeech/calcul/users/nmonir/corpora/'
    configs = {
        'model': args.model,
        'gender': args.gender,
        'scenario': args.scenario,
        'noise_type': args.noise_type,
        'asr_model_name': args.asr_model_name,
        'filenames': [os.path.basename(file) for file in glob.glob(os.path.join(corpora_folder, args.experiment_name, 'enhancement', args.gender, args.noise_type, args.model, args.scenario, '*'))],
        'wer_evaluations_file': os.path.join(corpora_folder, args.experiment_name, 'evaluation', args.gender, args.noise_type, args.model, args.scenario, f'wer_evaluations_{args.asr_model_name}.csv'),
        'corpus': os.path.join(corpora_folder, args.experiment_name)
    }
    return configs

def parse_arguments():
    """
    Parse command line arguments.

    Returns
    -------
    Namespace
        The arguments parsed from the command line.
    """
    parser = argparse.ArgumentParser(description="Process audio to generate mixtures with specified SNR and RIR.")
    parser.add_argument("-e", "--experiment_name", type=str, required=True, help="Experiment Name")
    parser.add_argument("-m", "--model", type=str, required=True, 
                        choices=['fasnet', 'tango'],
                        help="Model")
    parser.add_argument("-s", "--scenario", type=str, required=True, 
                        choices=['0dBS0N45', '0dBS0N90', 'p5dBS0N45', 'p5dBS0N90', 'm5dBS0N45', 'm5dBS0N90'],
                        help="Scenario tag")
    parser.add_argument("-g", "--gender", type=str, required=True, 
                        choices=['male', 'female'],
                        help="Gender")
    parser.add_argument("-n", "--noise_type", type=str, required=True, 
                        choices=['white-noise', 'babble-noise', 'speech-shaped-noise'],
                        help="Type of noise to use in the mixtures.")
    parser.add_argument("-a", "--asr_model_name", type=str, required=True, 
                        choices=['wav2vec','wav2vec-lv60','whisper','conformer-ctc','conformer-transducer'],
                        help="Model.")
    return parser.parse_args()

def create_evaluations_file(configs):

    head = ','.join(['MODEL', 'GENDER', 'NOISE_TYPE', 'SCENARIO', 'FILENAME', 'ASR_MODEL', 'WER']) + '\n'
    with open(configs['wer_evaluations_file'], 'w') as file:
        file.write(head)        

def export_line(evaluation_dict, configs):        

    evaluation_line = ','.join([str(evaluation_dict[key]) for key in evaluation_dict.keys()]) + '\n'
    with open(configs['wer_evaluations_file'], 'a') as file:
            file.write(evaluation_line)

def main():

    args = parse_arguments()
    configs =  load_configs(args)
    create_evaluations_file(configs)

    for filename in tqdm(configs['filenames']):        
        evaluation_dict = compute_wer(filename, configs)
        export_line(evaluation_dict, configs)
            
if __name__ == '__main__':

    main()
    