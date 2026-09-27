"""FFT convolution reverb with a user-supplied impulse response (IR).

Both WAV inputs use the standard 48 kHz / 16-bit PCM / mono|stereo
validation; the source clip must be 1-10 s long and the impulse response
1-2 s long and not all silence. Each output channel is

    out = (1 - wet) * dry_zero_padded + wet * (source * ir)

where ``*`` is the linear convolution of length N + M - 1, computed once
per channel pair with a single radix-2 FFT length. Channel routing:

* mono IR applied to every source channel (mono or stereo source);
* stereo IR with a stereo source: left convolves left, right right;
* stereo IR with a mono source: the mono dry signal is copied into both
  output channels and each IR channel gives its own wet channel, so the
  stereo image of the response is preserved.

If the mix exceeds the int16 range, a single non-positive gain is applied
to every channel of the whole clip before quantization; nothing is
clipped. wet = 0 skips the convolution and emits the source verbatim,
zero-padded with the reverb tail length.
"""
from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

from .loudness import (
    MAX_GAIN_BACKOFF_STEPS,
    SAMPLE_RATE,
    AudioValidationError,
    _read_wav,
    _write_wav,
)

MIN_IR_DURATION_S = 1.0
MAX_IR_DURATION_S = 2.0


@dataclass
class ConvolutionReverbResult:
    wav_bytes: bytes
    output_frames: int
    post_gain_db: float


def _read_ir_wav(wav_bytes: bytes) -> np.ndarray:
    """Decode the IR with the standard WAV checks plus IR-specific ones.

    Returns float64 samples in [-1, 1] with shape (n_frames, n_channels);
    the IR must be 1-2 s long and carry at least one non-zero sample.
    """
    samples = _read_wav(wav_bytes)
    frames = samples.shape[0]
    duration = frames / SAMPLE_RATE
    if not MIN_IR_DURATION_S <= duration <= MAX_IR_DURATION_S:
        raise AudioValidationError(
            f"Impulse response duration must be 1-2 s (got {duration:.3f} s)"
        )
    if not np.any(samples):
        raise AudioValidationError("Impulse response must not be all silence")
    return samples


def _fft_convolve(
    source: np.ndarray, impulse: np.ndarray, fft_len: int
) -> np.ndarray:
    """Linear convolution of two 1-D signals via one radix-2 FFT pair."""
    out_len = source.shape[0] + impulse.shape[0] - 1
    spectrum = np.fft.rfft(source, fft_len) * np.fft.rfft(impulse, fft_len)
    return np.fft.irfft(spectrum, fft_len)[:out_len]


def _quantize_unclipped(mixed: np.ndarray) -> tuple[np.ndarray, float]:
    """Quantize float samples to 16-bit PCM without ever clipping.

    If every rounded code already fits int16, the gain stays 1. Otherwise a
    single uniform gain (the same factor for every channel) is backed off
    just far enough for the observed worst code, leaving a 2-LSB margin for
    rounding, and the whole clip is re-quantized. Returns the int16 PCM and
    the linear gain actually applied.
    """
    gain = 1.0
    pcm: np.ndarray | None = None
    for _ in range(MAX_GAIN_BACKOFF_STEPS):
        codes = np.round(mixed * gain * 32768.0)
        hi = float(np.max(codes, initial=0.0))
        lo = float(np.min(codes, initial=0.0))
        if hi <= 32767.0 and lo >= -32768.0:
            pcm = codes.astype("<i2")
            break
        factors = []
        if hi > 32767.0:
            factors.append((32767.0 - 2.0) / hi)
        if lo < -32768.0:
            factors.append((32768.0 - 2.0) / -lo)
        gain *= min(factors)

    if pcm is None:
        # Defensive: the margin above always converges in one step, but if
        # rounding ever surprises us, attenuate once more rather than clip.
        gain *= 0.5
        pcm = np.round(mixed * gain * 32768.0).astype("<i2")
    return pcm, gain


def reverb_wav(
    wav_bytes: bytes, ir_wav_bytes: bytes, wet: float
) -> ConvolutionReverbResult:
    source = _read_wav(wav_bytes)
    impulse = _read_ir_wav(ir_wav_bytes)

    n_frames, src_channels = source.shape
    ir_frames, ir_channels = impulse.shape
    output_frames = n_frames + ir_frames - 1
    out_channels = max(src_channels, ir_channels)

    # Dry signal, zero-padded out to the full convolution length; a mono
    # source feeding a stereo IR is copied into both output channels.
    dry = np.zeros((output_frames, out_channels), dtype=np.float64)
    if src_channels == out_channels:
        dry[:n_frames] = source
    else:
        dry[:n_frames, 0] = source[:, 0]
        dry[:n_frames, 1] = source[:, 0]

    if wet == 0.0:
        # No wet path needed; the padded dry signal is the final mix.
        mixed = dry
    else:
        fft_len = 1 << (output_frames - 1).bit_length()
        wet_signal = np.zeros((output_frames, out_channels), dtype=np.float64)
        if ir_channels == 1:
            # One mono response convolved with every source channel.
            ir_mono = impulse[:, 0]
            for channel in range(src_channels):
                wet_signal[:, channel] = _fft_convolve(
                    source[:, channel], ir_mono, fft_len
                )
        elif src_channels == 2:
            # Stereo response on a stereo source: channel-by-channel.
            for channel in range(2):
                wet_signal[:, channel] = _fft_convolve(
                    source[:, channel], impulse[:, channel], fft_len
                )
        else:
            # Stereo response on a mono source: each IR channel builds its
            # own wet channel; the dry signal is the copy handled above.
            for channel in range(2):
                wet_signal[:, channel] = _fft_convolve(
                    source[:, 0], impulse[:, channel], fft_len
                )
        mixed = (1.0 - wet) * dry + wet * wet_signal

    pcm_out, gain = _quantize_unclipped(mixed)
    post_gain_db = 0.0 if gain == 1.0 else 20.0 * math.log10(gain)

    return ConvolutionReverbResult(
        wav_bytes=_write_wav(pcm_out, out_channels),
        output_frames=output_frames,
        post_gain_db=post_gain_db,
    )
