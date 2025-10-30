import os
import argparse
import numpy as np
import soundfile as sf


def generate_white_noise(duration: float, sample_rate: int) -> np.ndarray:
    """
    Generate a white noise signal of specified duration and sampling rate.

    Parameters
    ----------
    duration : float
        Duration of the white noise signal in seconds.
    sample_rate : int
        Sampling rate of the signal in Hz.

    Returns
    -------
    np.ndarray
        White noise signal normalized to have zero mean and unit variance.
    """
    num_samples = int(duration * sample_rate)
    white_noise = np.random.normal(0, 1, num_samples)
    white_noise = (white_noise - np.mean(white_noise)) / np.std(white_noise)
    return white_noise


def main(duration: float, sample_rate: int, output_dir: str):
    """Generate white noise and save it as a WAV file."""
    if not os.path.exists(output_dir):
        os.makedirs(output_dir, exist_ok=True)

    noise = generate_white_noise(duration, sample_rate)
    output_path = os.path.join(output_dir, "white-noise.wav")
    sf.write(output_path, noise, sample_rate)
    print(f"[OK] Saved white noise to: {output_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate a white noise signal and save it as a WAV file.")
    parser.add_argument("--duration", type=float, required=True, help="Duration of the white noise in seconds.")
    parser.add_argument("--sample_rate", type=int, required=True, help="Sampling rate of the signal in Hz.")
    parser.add_argument("--output_dir", required=True, help="Directory to save the generated white noise file.")
    args = parser.parse_args()

    main(args.duration, args.sample_rate, args.output_dir)
