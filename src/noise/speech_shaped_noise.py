import os
import argparse
import numpy as np
import soundfile as sf


def gen_ssn_from_sig(x: np.ndarray) -> np.ndarray:
    """
    Generate speech-shaped noise (SSN) from a given signal.

    Parameters
    ----------
    x : np.ndarray
        Input speech signal.

    Returns
    -------
    np.ndarray
        Speech-shaped noise with the same spectral envelope as the input.
    """
    X = np.fft.rfft(x)
    random_phase = np.exp(2j * np.pi * np.random.random(X.shape[-1]))
    noise_mag = np.abs(X) * random_phase
    noise = np.fft.irfft(noise_mag, x.shape[-1])
    return np.real(noise)


def main(speech_path: str, output_dir: str):
    """Load an input wav file, generate SSN, and save the result."""
    if not os.path.exists(speech_path):
        raise FileNotFoundError(f"Input file not found: {speech_path}")
    if not os.path.exists(output_dir):
        os.makedirs(output_dir, exist_ok=True)

    x, sr = sf.read(speech_path)
    noise = gen_ssn_from_sig(x)

    output_path = os.path.join(output_dir, "speech-shaped-noise.wav")
    sf.write(output_path, noise, sr)
    print(f"[OK] Saved speech-shaped noise to: {output_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate speech-shaped noise (SSN) from a speech signal.")
    parser.add_argument("--speech_path", required=True, help="Path to the speech input WAV file")
    parser.add_argument("--output_dir", required=True, help="Directory to save the generated SSN file")
    args = parser.parse_args()

    main(args.speech_path, args.output_dir)