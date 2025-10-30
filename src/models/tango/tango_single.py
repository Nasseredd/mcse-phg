import os
import sys
import glob
import argparse
import numpy as np
import os.path as osp
from tqdm import tqdm 
import soundfile as sf
from tango.utils import file_io
from tango.filter.filter import GevdFilter
from tango.filter.tango import TangoFilter
from tango.data.waveforms import RoomWaveforms
from tango.dnn.models import load_one_model

MIC_DISTRIB = [2, 2]

def load_models(sn_path, mn_path, n_nodes=None):
    """Load weights and bias of models needed to run Tango with a 2 nodes configuration.

    Args:
        sn_path (str): Path to the pytorch file that contains data for single-node model.
        mn_path (str): Path to the pytorch file that contains data for multi-node model.
        n_nodes (int): Number of nodes to select during loading  by default the full model
            is loaded (all nodes).

    Returns:
        Tuple[tango.models.CFNN, tango.models.CFNN]:
            * Convolutional neural network to estimate time-frequency masks
            for single-node step (1st step)
            * Convolutional neural network to estimate time-frequency masks for multi-node steps.
    """
    config_data = {
        "archi": "CFNN",
        "n_frames_in": 7,
        "params": {
            "cnn_filters": [32, 64, 64],
            "conv_kernels": [3, 3, 3],
            "conv_strides": [1, 1, 1],
            "pool_kernels": [[1, 4], [1, 4], [1, 4]],
            "pool_strides": [None, None, None],
            "ff_units": [256, 257],
            "conv_padding": [[0, 1], [0, 1], [0, 1]],
        },
    }
    sn_model = load_one_model(model_path=sn_path, config_params=config_data)
    mn_model = load_one_model(model_path=mn_path, config_params=config_data, n_nodes=n_nodes)
    return [sn_model, mn_model]

def get_parsed_arg():
    """Get parsed arguments from cli.

    Returns:
        namespace populated with cli arguments.
    """
    parser = argparse.ArgumentParser(
        description=main.__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )

    model_dir = '/home/nasmonir/PhD/ansd-se-benchmark/tango/tango/tests/test_rsc/dnn'
    parser.add_argument(
        "-m",
        "--model-paths",
        type=str,
        nargs=2,
        default=(
            osp.join(model_dir, "BPbr_model.pt"),
            osp.join(model_dir, "X1kz_model.pt"),
        ),
        help=(
            "Paths to pytorch files that contain weights and bias of the model, "
            "the first path for single-node model and the second one for multi-node model."
        ),
    )

    parser.add_argument(
        "--mu",
        type=float,
        default=1.0,
        help=(
            "Trade-off parameter between the noise reduction and the speech distortion, "
            "(default=1.0). Lower values produce output with lower speech distortion "
            "whereas greater values give an advantage to noise reduction."
        ),
    )

    parser.add_argument(
        "--rank",
        type=int,
        default=1,
        help=(
            "Rank of the approximation used to compute the beamformer, must be positive "
            "and lower or equal to 3 (for signals from the PHL: 2 nodes with 2 mics each). "
            "A value greater or equal to 3 is equivalent to full rank. Default value is 1."
        ),
    )

    parser.add_argument(
        "--steps",
        type=int,
        default=2,
        help=("Number of fitering steps to applied, default is 2."),
    )

    parser.add_argument(
        "--single-out",
        action="store_true",
        help=(
            "If enabled, write signals as single channel wav files, "
            "write stereo wav files otherwise."
        ),
    )

    return parser.parse_args()

def main(model_paths, mu, rank, steps):
    """Run Tango algorithm on signals."""

    # Create Tango filter object
    tango_filter = TangoFilter(
        multi_filter=GevdFilter(mu=mu, rank=rank),
        mic_distrib=MIC_DISTRIB,
        steps=steps,
        use_dist_masks=True,
        ref_mic_ind=0,
    )

    # Load models
    (sn_model, mn_model) = load_models(sn_path=model_paths[0], mn_path=model_paths[1], n_nodes=2)
    
    return tango_filter, sn_model, mn_model

def single_inference(mixture_wav_path, output_dir, tango_filter, sn_model, mn_model):

    # Load input wave files
    PERMUTATIONS = [0, 2, 1, 3]
    waveforms = RoomWaveforms.load_multichannel(
        mixture_wav_path, permutations=PERMUTATIONS, mic_distrib=MIC_DISTRIB
    )

    # Process Tango algorithm
    waveforms.normalise()
    output_tensors = tango_filter.process_one_room(
        waveforms.to_tensors(), models=[sn_model, mn_model]
    )
    waveforms.output = output_tensors.cpu().numpy()
    waveforms.normalise()

    # Write ouput signals
    signal_types = ["mixture", "output"]
    signals_to_write = set(signal_types).intersection(waveforms.defined_signals)

    for sgn_type in signals_to_write:
        if sgn_type == "output":
            file_io.write_stereo_wav_files(
                signals=np.expand_dims(getattr(waveforms, sgn_type), axis=0),
                output_dir=output_dir,
                file_names=(sgn_type,),
                normalise=False,
                sampling_rate=waveforms.sampling_rate,
            )

# if __name__ == "__main__":
#     # Settings
#     args = get_parsed_arg()
#     data_dir = '/home/nasmonir/PhD/ansd-se-benchmark/tango/tango-mytutorial/data/'
#     mixture_path = data_dir + 'mixture-4ch.wav'
#     output_dir = data_dir
#     # Load the model
#     tango_filter, sn_model, mn_model = main(model_paths=args.model_paths, mu=args.mu, rank=args.rank, steps=args.steps,)           
#     # Tango Inference
#     single_inference(mixture_path, output_dir, tango_filter, sn_model, mn_model)