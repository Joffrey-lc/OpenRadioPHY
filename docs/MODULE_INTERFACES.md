# Fixed PHY Module Interfaces

Each physical-layer operation has a fixed interface. The default modules use
NumPy; replacements must keep the same shape, dtype, ordering, and profile.

## Tensor convention

| Symbol | Meaning |
| --- | --- |
| `Bp` | protocol bits before PHY padding |
| `B` | padded PHY bits |
| `Q` | data constellation symbols |
| `N` | IDFT/DFT size (`fft_len`) |
| `F` | OFDM symbols in one PHY frame |
| `C` | cyclic-prefix length (`cp_len`) |
| `Z` | preamble samples |

Rules:

- Bits are `numpy.uint8` vectors containing only zero or one.
- Internal complex tensors are `numpy.complex128`; file/channel boundaries may
  use `complex64`.
- A resource/time grid always has shape `[N, F]`: rows are subcarriers or time
  samples, columns are OFDM symbols.
- A CP grid has shape `[N+C, F]`.
- DFT/IDFT operate on axis 0 and never apply `fftshift`.
- Parallel/serial conversion always uses Fortran order (`order="F"`).

For the default QPSK frame with a 1165-byte protocol frame:

```text
N=256, C=64, F=23
resource_grid     [256, 23]
symbols_with_cp   [320, 23]
payload_stream    [7360]
waveform          [8383] = 1023 preamble + 7360 payload samples
```

## Transmitter modules

| File | Callable | Fixed interface |
| --- | --- | --- |
| `coding.py` | `encode_bits` | `uint8[Bp] -> uint8[B]` |
| `modulation.py` | `modulate_bits` | `uint8[B] -> complex128[Q]` |
| `resource_mapping.py` | `map_to_resource_grid` | `complex128[Q] -> complex128[N,F]` |
| `idft.py` | `idft` | `complex128[N,F] -> complex128[N,F]` |
| `add_cp.py` | `add_cp` | `complex128[N,F] -> complex128[N+C,F]` |
| `parallel_to_serial.py` | `serialize_symbols` | `complex128[N+C,F] -> complex128[(N+C)F]` |
| `frame_assembly.py` | `assemble_frame` | `preamble[Z], payload[S] -> waveform[Z+S]` |
| `channel.py` | `apply_channel` | `complex128[S] -> complex64[S']` |

The baseline `coding.py` performs padding, scrambling, and interleaving. A
future FEC or learned encoder belongs at this boundary.

## Receiver modules

| File | Callable | Fixed interface |
| --- | --- | --- |
| `frame_sync.py` | `synchronize_frame` | `capture[S] -> payload_with_margin` |
| `timing_sync.py` | `align_payload` | `payload_with_margin -> complex128[(N+C)F]` |
| `frequency_sync.py` | `correct_frequency_offset` | `complex128[(N+C)F] -> same shape` |
| `serial_to_parallel.py` | `deserialize_symbols` | `complex128[(N+C)F] -> complex128[N+C,F]` |
| `remove_cp.py` | `remove_cp` | `complex128[N+C,F] -> complex128[N,F]` |
| `dft.py` | `dft` | `complex128[N,F] -> complex128[N,F]` |
| `channel_estimation.py` | `estimate_channel` | `Y[N,F] -> H[N,F]` |
| `equalization.py` | `equalize` | `Y[N,F], H[N,F] -> Xeq[N,F]` |
| `resource_demapping.py` | `extract_data_symbols` | `Xeq[N,F] -> complex128[Q]` |
| `demodulation.py` | `demodulate_symbols` | `complex128[Q] -> uint8[B]` |
| `decoding.py` | `decode_bits` | `uint8[B] -> uint8[B]` |

## Replacing a module

`main.py` explicitly constructs `TX_MODULES` and `RX_MODULES`. The
`OfdmTransmitter` and `OfdmReceiver` classes call those functions in order and
return `TxFrameTrace`/`RxFrameTrace`, which expose every intermediate tensor.

A framework-backed module only needs an adapter at its boundary:

```python
def neural_idft_adapter(resource_grid):
    # complex128 [N,F]
    model_input = to_model_tensor(resource_grid)
    model_output = trained_idft(model_input)
    return to_numpy_complex128(model_output)  # must remain [N,F]

modules = TxModules(idft=neural_idft_adapter)
transmitter = OfdmTransmitter(modules)
```

The adapter handles framework conversion; neighboring modules remain NumPy.

## Configuration ownership

- `d2d_local/ofdm_conf.py`: all public OFDM dimensions, pilot, modulation,
  interleaver, preamble, equalizer, and detection parameters.
- `d2d_local/config.py`: protocol, channel, and combined simulation settings;
  it re-exports `OfdmConfig` for compatibility.
- The OTA manifest carries the exact `OfdmConfig` used for a measured capture.
