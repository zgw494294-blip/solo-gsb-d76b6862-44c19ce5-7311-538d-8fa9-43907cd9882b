import base64
import io
import json
import wave

import numpy as np
import pytest
from fastapi.testclient import TestClient

from app.main import app

client = TestClient(app)

RATE = 48000
SRC_N = 2 * RATE      # 2 s source
IR_N = RATE + 24000   # 1.5 s impulse response


def wav_b64(pcm: np.ndarray, channels: int = 1, rate: int = RATE) -> str:
    buf = io.BytesIO()
    with wave.open(buf, "wb") as wf:
        wf.setnchannels(channels)
        wf.setsampwidth(2)
        wf.setframerate(rate)
        wf.writeframes(pcm.astype("<i2").tobytes())
    return base64.b64encode(buf.getvalue()).decode()


def decode_wav(b64: str):
    raw = base64.b64decode(b64)
    with wave.open(io.BytesIO(raw), "rb") as wf:
        params = (wf.getnchannels(), wf.getsampwidth(), wf.getframerate())
        pcm = np.frombuffer(wf.readframes(wf.getnframes()), dtype="<i2")
        frames = wf.getnframes()
    return params, frames, pcm


def post_reverb(src, ir, wet=0.5, channels=1, ir_channels=1):
    payload = {
        "wav_base64": wav_b64(src, channels),
        "ir_wav_base64": wav_b64(ir, ir_channels),
        "wet": wet,
    }
    return client.post("/v1/convolution-reverb", json=payload)


def sine(freq, amp, n, phase=0.0):
    t = np.arange(n) / RATE
    return np.round(amp * np.sin(2 * np.pi * freq * t + phase) * 32768).astype("<i2")


def make_ir(n=IR_N, seed=0, amp=0.02):
    # Exponentially decaying noise, a typical synthetic impulse response.
    # Kept quiet so the wet path stays in range without the post gain.
    rng = np.random.default_rng(seed)
    t = np.arange(n) / RATE
    env = np.exp(-3.0 * t)
    return np.round(amp * env * rng.standard_normal(n) * 32768).astype("<i2")


def test_wet_zero_is_zero_padded_dry_source():
    src = sine(440, 0.2, SRC_N)
    ir = make_ir()
    resp = post_reverb(src, ir, wet=0.0)
    assert resp.status_code == 200
    data = resp.json()
    assert data["output_frames"] == SRC_N + IR_N - 1
    assert data["post_gain_db"] == 0.0
    params, frames, out = decode_wav(data["wav_base64"])
    assert params == (1, 2, RATE)
    assert frames == SRC_N + IR_N - 1
    assert out.size == SRC_N + IR_N - 1
    # First N frames are the source bit-for-bit; the reverb tail is silence.
    assert np.array_equal(out[:SRC_N], src)
    assert np.all(out[SRC_N:] == 0)


def test_output_matches_reference_fft_convolution():
    src = sine(440, 0.1, SRC_N)
    ir = make_ir()
    wet = 0.6
    resp = post_reverb(src, ir, wet=wet)
    assert resp.status_code == 200
    data = resp.json()
    assert data["post_gain_db"] == 0.0
    _, _, out = decode_wav(data["wav_base64"])

    s = src.astype(np.float64) / 32768.0
    h = ir.astype(np.float64) / 32768.0
    full = np.fft.irfft(np.fft.rfft(s, 1 << (SRC_N + IR_N - 2).bit_length())
                       * np.fft.rfft(h, 1 << (SRC_N + IR_N - 2).bit_length()))
    conv = full[: SRC_N + IR_N - 1]
    dry = np.zeros_like(conv)
    dry[:SRC_N] = s
    expected = np.round(((1 - wet) * dry + wet * conv) * 32768).astype("<i2")
    assert np.max(np.abs(out.astype(np.int64) - expected.astype(np.int64))) <= 1


def test_identity_impulse_wet_one_keeps_source_in_place():
    src = sine(997, 0.15, SRC_N)
    ir = np.zeros(RATE, dtype="<i2")  # exactly 1 s
    ir[0] = 32767
    resp = post_reverb(src, ir, wet=1.0)
    assert resp.status_code == 200
    data = resp.json()
    assert data["output_frames"] == SRC_N + RATE - 1
    assert data["post_gain_db"] == 0.0
    _, _, out = decode_wav(data["wav_base64"])
    # Identity IR: source unchanged, one fewer tail frame (tail all zero).
    assert np.array_equal(out[:SRC_N], src)
    assert np.all(out[SRC_N:] == 0)


def test_tail_carries_reverb_after_source_end():
    src = np.zeros(SRC_N, dtype="<i2")
    src[SRC_N - 1000] = 20000  # single impulse 1000 frames before the end
    ir = make_ir(seed=3)
    resp = post_reverb(src, ir, wet=1.0)
    assert resp.status_code == 200
    _, _, out = decode_wav(resp.json()["wav_base64"])
    # Frames beyond the source length must contain the decaying IR tail.
    tail = out[SRC_N:]
    assert tail.size == IR_N - 1
    assert np.count_nonzero(tail) > RATE // 4
    # Decaying envelope: tail energy near the end is far below its start.
    energy_early = np.mean(tail[: RATE // 4].astype(np.float64) ** 2)
    energy_late = np.mean(tail[-RATE // 4 :].astype(np.float64) ** 2)
    assert energy_late < energy_early * 0.01


def test_overflow_applies_negative_gain_without_clipping():
    # Loud source + loud wet path can exceed full scale.
    src = sine(440, 0.9, SRC_N)
    ir = make_ir(seed=7, amp=0.8)
    ir[:100] = np.maximum(ir[:100], 20000)  # strong early reflection
    resp = post_reverb(src, ir, wet=0.8)
    assert resp.status_code == 200
    data = resp.json()
    assert data["post_gain_db"] < 0.0
    _, _, out = decode_wav(data["wav_base64"])
    assert int(np.max(out)) <= 32767
    assert int(np.min(out)) >= -32768
    # No clipped run against either rail.
    assert np.count_nonzero(out == 32767) < 10
    assert np.count_nonzero(out == -32768) < 10


def test_silent_source_is_allowed():
    src = np.zeros(SRC_N, dtype="<i2")
    ir = make_ir()
    resp = post_reverb(src, ir, wet=0.5)
    assert resp.status_code == 200
    _, _, out = decode_wav(resp.json()["wav_base64"])
    assert np.all(out == 0)


def test_mono_ir_applied_to_both_stereo_channels():
    left = sine(440, 0.1, SRC_N)
    right = sine(660, 0.08, SRC_N)
    src = np.stack([left, right], axis=1)
    ir = make_ir(seed=11)
    resp = post_reverb(src, ir, wet=0.7, channels=2, ir_channels=1)
    assert resp.status_code == 200
    data = resp.json()
    params, frames, out = decode_wav(data["wav_base64"])
    assert params == (2, 2, RATE)
    assert frames == SRC_N + IR_N - 1
    out2 = out.reshape(-1, 2)

    s_l = left.astype(np.float64) / 32768.0
    s_r = right.astype(np.float64) / 32768.0
    h = ir.astype(np.float64) / 32768.0
    fft_len = 1 << (frames - 1).bit_length()
    conv_l = np.fft.irfft(np.fft.rfft(s_l, fft_len) * np.fft.rfft(h, fft_len), fft_len)[:frames]
    conv_r = np.fft.irfft(np.fft.rfft(s_r, fft_len) * np.fft.rfft(h, fft_len), fft_len)[:frames]
    dry_l = np.zeros(frames)
    dry_r = np.zeros(frames)
    dry_l[:SRC_N] = s_l
    dry_r[:SRC_N] = s_r
    exp_l = np.round((0.3 * dry_l + 0.7 * conv_l) * 32768)
    exp_r = np.round((0.3 * dry_r + 0.7 * conv_r) * 32768)
    assert np.max(np.abs(out2[:, 0].astype(np.int64) - exp_l.astype(np.int64))) <= 1
    assert np.max(np.abs(out2[:, 1].astype(np.int64) - exp_r.astype(np.int64))) <= 1


def test_stereo_ir_with_mono_source_copies_dry_per_channel():
    src = sine(440, 0.1, SRC_N)
    ir_l = make_ir(seed=1)
    ir_r = make_ir(seed=2)
    ir = np.stack([ir_l, ir_r], axis=1)
    wet = 0.5
    resp = post_reverb(src, ir, wet=wet, channels=1, ir_channels=2)
    assert resp.status_code == 200
    params, frames, out = decode_wav(resp.json()["wav_base64"])
    assert params == (2, 2, RATE)
    out2 = out.reshape(-1, 2)
    # Dry is copied into both output channels: first N frames of both
    # share the same dry component, and the two tails differ.
    s = src.astype(np.float64) / 32768.0
    h_l = ir_l.astype(np.float64) / 32768.0
    h_r = ir_r.astype(np.float64) / 32768.0
    fft_len = 1 << (frames - 1).bit_length()
    conv_l = np.fft.irfft(np.fft.rfft(s, fft_len) * np.fft.rfft(h_l, fft_len), fft_len)[:frames]
    conv_r = np.fft.irfft(np.fft.rfft(s, fft_len) * np.fft.rfft(h_r, fft_len), fft_len)[:frames]
    dry = np.zeros(frames)
    dry[:SRC_N] = s
    exp_l = np.round((0.5 * dry + 0.5 * conv_l) * 32768).astype("<i2")
    exp_r = np.round((0.5 * dry + 0.5 * conv_r) * 32768).astype("<i2")
    assert np.max(np.abs(out2[:, 0].astype(np.int64) - exp_l.astype(np.int64))) <= 1
    assert np.max(np.abs(out2[:, 1].astype(np.int64) - exp_r.astype(np.int64))) <= 1
    # The two channels are genuinely different (stereo response preserved).
    assert not np.array_equal(out2[:, 0], out2[:, 1])


def test_stereo_ir_with_stereo_source_convolves_per_channel():
    left = sine(440, 0.08, SRC_N)
    right = sine(880, 0.08, SRC_N)
    src = np.stack([left, right], axis=1)
    ir_l = make_ir(seed=4)
    ir_r = make_ir(seed=5)
    ir = np.stack([ir_l, ir_r], axis=1)
    resp = post_reverb(src, ir, wet=0.9, channels=2, ir_channels=2)
    assert resp.status_code == 200
    params, frames, out = decode_wav(resp.json()["wav_base64"])
    assert params == (2, 2, RATE)
    out2 = out.reshape(-1, 2)
    assert not np.array_equal(out2[:, 0], out2[:, 1])
    # Channel separation: left output contains the 440 Hz band energy,
    # right output the 880 Hz band.
    def band_rms(x, freq):
        n = x.size
        spec = np.abs(np.fft.rfft(x.astype(np.float64)))
        k = round(freq * n / RATE)
        return np.sqrt(np.sum(spec[max(0, k - 2) : k + 3] ** 2))

    l440, l880 = band_rms(out2[:, 0], 440), band_rms(out2[:, 0], 880)
    r440, r880 = band_rms(out2[:, 1], 440), band_rms(out2[:, 1], 880)
    assert l440 > l880
    assert r880 > r440


def test_invalid_base64_returns_422():
    payload = {
        "wav_base64": "!!!not-base64!!!",
        "ir_wav_base64": base64.b64encode(b"x").decode(),
        "wet": 0.5,
    }
    assert client.post("/v1/convolution-reverb", json=payload).status_code == 422


def test_garbage_wav_returns_422():
    src = wav_b64(sine(440, 0.1, SRC_N))
    payload = {
        "wav_base64": src,
        "ir_wav_base64": base64.b64encode(b"not a wav").decode(),
        "wet": 0.5,
    }
    assert client.post("/v1/convolution-reverb", json=payload).status_code == 422


def test_source_wrong_rate_returns_422():
    src = wav_b64(sine(440, 0.1, SRC_N), rate=44100)
    ir = wav_b64(make_ir())
    payload = {"wav_base64": src, "ir_wav_base64": ir, "wet": 0.5}
    assert client.post("/v1/convolution-reverb", json=payload).status_code == 422


def test_source_too_short_returns_422():
    src = wav_b64(sine(440, 0.1, RATE // 2))  # 0.5 s
    ir = wav_b64(make_ir())
    payload = {"wav_base64": src, "ir_wav_base64": ir, "wet": 0.5}
    assert client.post("/v1/convolution-reverb", json=payload).status_code == 422


@pytest.mark.parametrize("ir_len_frames", [RATE - 1, 2 * RATE + 1])
def test_ir_duration_out_of_range_returns_422(ir_len_frames):
    src = wav_b64(sine(440, 0.1, SRC_N))
    ir = wav_b64(sine(1000, 0.1, ir_len_frames))
    payload = {"wav_base64": src, "ir_wav_base64": ir, "wet": 0.5}
    assert client.post("/v1/convolution-reverb", json=payload).status_code == 422


@pytest.mark.parametrize("ir_len_frames", [RATE, 2 * RATE])
def test_ir_duration_bounds_accepted(ir_len_frames):
    src = wav_b64(sine(440, 0.05, SRC_N))
    ir = np.zeros(ir_len_frames, dtype="<i2")
    ir[0] = 100
    payload = {"wav_base64": src, "ir_wav_base64": wav_b64(ir), "wet": 0.5}
    assert client.post("/v1/convolution-reverb", json=payload).status_code == 200


def test_silent_ir_returns_422():
    src = wav_b64(sine(440, 0.1, SRC_N))
    ir = wav_b64(np.zeros(IR_N, dtype="<i2"))
    payload = {"wav_base64": src, "ir_wav_base64": ir, "wet": 0.5}
    assert client.post("/v1/convolution-reverb", json=payload).status_code == 422


def test_stereo_silent_ir_returns_422():
    src = wav_b64(sine(440, 0.1, SRC_N))
    ir = np.zeros((IR_N, 2), dtype="<i2")
    payload = {"wav_base64": src, "ir_wav_base64": wav_b64(ir, 2), "wet": 0.5}
    assert client.post("/v1/convolution-reverb", json=payload).status_code == 422


def test_ir_wrong_bit_depth_returns_422():
    # 8-bit PCM WAV
    buf = io.BytesIO()
    with wave.open(buf, "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(1)
        wf.setframerate(RATE)
        wf.writeframes(np.full(RATE, 128, dtype=np.uint8).tobytes())
    src = wav_b64(sine(440, 0.1, SRC_N))
    payload = {
        "wav_base64": src,
        "ir_wav_base64": base64.b64encode(buf.getvalue()).decode(),
        "wet": 0.5,
    }
    assert client.post("/v1/convolution-reverb", json=payload).status_code == 422


@pytest.mark.parametrize("wet", [-0.01, 1.01])
def test_wet_out_of_range_returns_422(wet):
    src = wav_b64(sine(440, 0.1, SRC_N))
    ir = wav_b64(make_ir())
    payload = {"wav_base64": src, "ir_wav_base64": ir, "wet": wet}
    assert client.post("/v1/convolution-reverb", json=payload).status_code == 422


@pytest.mark.parametrize("wet", [0.0, 1.0])
def test_wet_bounds_accepted(wet):
    src = wav_b64(sine(440, 0.05, SRC_N))
    ir = wav_b64(make_ir())
    payload = {"wav_base64": src, "ir_wav_base64": ir, "wet": wet}
    assert client.post("/v1/convolution-reverb", json=payload).status_code == 200


@pytest.mark.parametrize("bad", [float("nan"), float("inf"), float("-inf")])
def test_non_finite_wet_returns_422(bad):
    src = wav_b64(sine(440, 0.1, SRC_N))
    ir = wav_b64(make_ir())
    payload = {"wav_base64": src, "ir_wav_base64": ir, "wet": bad}
    resp = client.post(
        "/v1/convolution-reverb",
        content=json.dumps(payload),
        headers={"Content-Type": "application/json"},
    )
    assert resp.status_code == 422


@pytest.mark.parametrize("field", ["wav_base64", "ir_wav_base64", "wet"])
def test_missing_field_returns_422(field):
    payload = {
        "wav_base64": wav_b64(sine(440, 0.1, SRC_N)),
        "ir_wav_base64": wav_b64(make_ir()),
        "wet": 0.5,
    }
    del payload[field]
    assert client.post("/v1/convolution-reverb", json=payload).status_code == 422
