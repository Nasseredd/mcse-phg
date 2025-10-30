import os 
import glob
import argparse
from tqdm import tqdm

def levenshtein_distance(s1, s2):
    m, n = len(s1), len(s2)
    
    # Create a distance matrix and initialize it
    dp = [[0] * (n + 1) for _ in range(m + 1)]
    
    for i in range(m + 1):
        dp[i][0] = i
    for j in range(n + 1):
        dp[0][j] = j

    # Populate the distance matrix
    for i in range(1, m + 1):
        for j in range(1, n + 1):
            cost = 0 if s1[i - 1] == s2[j - 1] else 1
            dp[i][j] = min(dp[i - 1][j] + 1,      # Deletion
                           dp[i][j - 1] + 1,      # Insertion
                           dp[i - 1][j - 1] + cost)  # Substitution

    return dp[m][n]

def load_configs(args):

    corpora_folder = '/srv/storage/talc3@storage4.nancy.grid5000.fr/multispeech/calcul/users/nmonir/corpora/'
    configs = {
        'model': args.model,
        'gender': args.gender,
        'scenario': args.scenario,
        'noise_type': args.noise_type,
        'filenames': [os.path.basename(file) for file in glob.glob(os.path.join(corpora_folder, args.experiment_name, 'enhancement', args.gender, args.noise_type, args.model, args.scenario, '*'))],
        'levenshtein_evaluations_file': os.path.join(corpora_folder, args.experiment_name, 'evaluation', args.gender, args.noise_type, args.model, args.scenario, f'levenshtein_evaluations.csv'),
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
    return parser.parse_args()

def create_evaluations_file(configs):

    head = ','.join(['MODEL', 'GENDER', 'NOISE_TYPE', 'SCENARIO', 'FILENAME', 'LEVENSHTEIN_DISTANCE']) + '\n'
    with open(configs['levenshtein_evaluations_file'], 'w') as file:
        file.write(head)  

def export_line(evaluation_dict, configs):        

    evaluation_line = ','.join([str(evaluation_dict[key]) for key in evaluation_dict.keys()]) + '\n'
    with open(configs['levenshtein_evaluations_file'], 'a') as file:
            file.write(evaluation_line)

def compute_levenshtein_distance(filename, configs):
    
    # Groundtruth
    groundtruth_path = os.path.join(configs['corpus'], 'clean-transcription', configs['gender'], f'{filename}.txt')
    with open(groundtruth_path, 'r') as file:
        groundtruth = file.read()

    # Enhancement transcription 
    transcription_file = os.path.join(configs['corpus'], 'enhancement-transcription/phonemizer', configs['gender'], configs['noise_type'], configs['model'], configs['scenario'], f'{filename}.txt')
    with open(transcription_file, 'r') as file:
        transcription = file.read()

    return {
        'MODEL': configs['model'],
        'GENDER': configs['gender'], 
        'NOISE_TYPE': configs['noise_type'],
        'SCENARIO': configs['scenario'],
        'FILENAME': filename,
        'LEVENSHTEIN_DISTANCE': levenshtein_distance(groundtruth, transcription),
    }

def main():

    args = parse_arguments()
    configs = load_configs(args)
    create_evaluations_file(configs)
    
    for filename in tqdm(configs['filenames']):
        evaluation_dict = compute_levenshtein_distance(filename, configs)
        export_line(evaluation_dict, configs)

if __name__ == '__main__':
    main()
