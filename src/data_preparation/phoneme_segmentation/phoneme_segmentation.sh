$INPUT_DIR=$1
$OUTPUT_DIR=$2

# Activate the MFA aligner conda environment
conda activate mfa-aligner

# Download english dictionary
mfa model download dictionary english_mfa

# Download english acoustic model
mfa model download acoustic english_mfa

# Run the phoneme segmentation script
mfa align $INPUT_DIR english_mfa english_mfa $OUTPUT_DIR