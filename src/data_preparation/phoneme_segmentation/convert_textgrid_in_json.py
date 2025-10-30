import os
import sys
import glob
import re
import json
import argparse
from tqdm import tqdm 
from textgrid import TextGrid

# Constants for ANSI color codes for terminal output
COLORS = {
    "blue": "\033[34m",
    "red": "\033[31m",
    "magenta": "\033[35m",
    "cyan": "\033[36m",
    "green": "\033[32m",
    "yellow": "\033[33m",
    "grey": "\033[90m",
    "default": "\033[0m"
}

def phoneme2class(phoneme, p2c):
    """
    Convert a phoneme to its class based on a mapping dictionary.

    Parameters
    ----------
    phoneme : str
        The phoneme to classify.
    p2c : dict
        Phoneme to class mapping dictionary.

    Returns
    -------
    str
        The class of the phoneme, 'silence' for specific undefined phonemes,
        or 'NaN' if the phoneme is not recognized.
    """
    if phoneme in ['', 'spn']:
        return "silence"
    return p2c.get(phoneme, "NaN")

def get_phones_intervals(textgrid_file, p2c):
    with open(textgrid_file, "r") as file:
        text = file.read()

    # Select phones (item [2])
    phones = text.split('item [2]:', 1)[1]

    # Select the duration (first x_max)
    match = re.search(r'xmax\s*=\s*([\d.]+)', phones)
    if match:
        duration = float(match.group(1))
    else:
        print(COLORS['red'], "[ERROR] No 'xmax =' pattern found in the text.", COLORS['default'])
        duration = None

    # Select 
    match = re.search(r'intervals: size\s*=\s*([\d.]+)', phones)
    if match:
        n_segmentations = int(match.group(1))
    else:
        print(COLORS['red'], "[ERROR] No 'intervals: size =' pattern found in the text.", COLORS['default'])
        n_segmentations = None

    # Select all phone intervals 
    pattern = re.compile(r'intervals \[\d+\]:\s*xmin = ([\d.]+)\s*xmax = ([\d.]+)\s*text = "([^"]*)"')
    matches = pattern.findall(phones)
    segmentation = [[float(xmin), float(xmax), phone, phoneme2class(phone, p2c)] for xmin, xmax, phone in matches]
    
    return {
        "duration": duration, 
        "n_segmentations": n_segmentations, 
        "segmentation": segmentation
    }

def process_textgrid_files(textgrid_folder, p2c):
    """
    Process all TextGrid files in a folder.

    Parameters
    ----------
    textgrid_folder : str
        Path to the folder containing TextGrid files.
    p2c : dict
        Phoneme to class mapping dictionary.

    Returns
    -------
    None
    """
    textgrid_files = glob.glob(f'{textgrid_folder}/*.TextGrid')
    for textgrid_file in tqdm(textgrid_files):
        
        # Extract phone intervals
        phone_data = get_phones_intervals(textgrid_file, p2c)

        # Output the JSON file
        json_file = textgrid_file.replace('.TextGrid', '.json')
        with open(json_file, "w") as file:
            json.dump(phone_data, file, indent=4)

def parse_arguments():
    """
    Parse command line arguments.

    Returns
    -------
    Namespace
        Contains all command line arguments.
    """
    parser = argparse.ArgumentParser(description='Convert TextGrid files to JSON format with phoneme classification.')
    parser.add_argument('-f', '--phoneme_classes_file', type=str, help='Path to the JSON file containing phoneme to class mappings.')
    parser.add_argument('-t', '--phoneme_segmentation_dir', type=str, help='Folder containing TextGrid files to process.')
    parser.add_argument('-g', '--gender', type=str, required=True, help="Gender of the speaker.", choices=['male', 'female'])
    return parser.parse_args()

def main():
    args = parse_arguments()
    
    with open(args.phoneme_classes_file, 'r') as file:
        class2phonemes = json.load(file)
    
    p2c = {p: c for c, phonemes in class2phonemes.items() for p in phonemes}
    
    textgrid_folder = os.path.join(args.phoneme_segmentation_dir, args.gender)

    process_textgrid_files(textgrid_folder, p2c)

if __name__ == '__main__':
    main()