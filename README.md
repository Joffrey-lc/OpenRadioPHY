# OpenRadioPHY

**A modular Python physical-layer platform for reproducible simulation and point-to-point over-the-air communication experiments.**

OpenRadioPHY is a Python-first OFDM research platform. It provides a readable image-to-image physical-layer chain whose intermediate signals can be inspected, plotted, and replaced one block at a time.

The long-term goal is a modular PHY playground where analytical and learnable blocks share the same interfaces. Modulation, pilot design, channel estimation, equalization, coding, and waveform processing can then be studied without rewriting the rest of the link. Keeping the implementation close to NumPy also makes it straightforward to connect with PyTorch- or JAX-based experiments.

The current release supports complete link simulation and decoding of measured point-to-point OTA IQ. Both paths use the same Python receiver.

Current version: **v0.1.0**

![OpenRadioPHY system overview](https://mymarkdown-pic.oss-cn-chengdu.aliyuncs.com/img220/20260707211817961.png)

<p align="center"><img src="https://mymarkdown-pic.oss-cn-chengdu.aliyuncs.com/img220/20260707191315733.png" alt="OpenRadioPHY processing flow" width="760"></p>

## Project Goals

- Keep the complete PHY chain readable and runnable in Python.
- Give every PHY block a fixed array shape, dtype, and configuration contract.
- Use one receiver interface for simulated waveforms and measured OTA IQ.
- Preserve intermediate tensors for plotting, debugging, and dataset generation.
- Allow individual analytical blocks to be replaced by learned models later.
- Make experiments repeatable through explicit configurations, random seeds, checksums, and JSON summaries.

## Signal Chain

```text
image/file
    -> PMAT framing and CRC
    -> coding
    -> modulation
    -> resource mapping
    -> IDFT
    -> cyclic prefix
    -> frame assembly
    -> channel
    -> frame, timing, and frequency synchronization
    -> cyclic-prefix removal
    -> DFT
    -> channel estimation
    -> equalization
    -> resource demapping
    -> demodulation
    -> decoding
    -> CRC check and image/file reconstruction
```

For measured data, the receiver starts from a complex-IQ capture and its manifest, then follows the same synchronization-to-reconstruction path used by the simulator.

## Project Status

| Implemented | In progress | Planned |
| :--- | :--- | :--- |
| File-per-module NumPy PHY with fixed interfaces<br>PMAT metadata, chunking, frame CRC, and file CRC<br>BPSK, QPSK, 16QAM, and 64QAM<br>256-point CP-OFDM with guard/DC carriers<br>Zadoff-Chu preamble and matched-filter synchronization<br>CP fine timing and frequency-offset correction<br>Scattered pilots, channel interpolation, equalization, and CPE correction<br>AWGN, delay, CFO, and static-multipath channels<br>Common receiver for simulation and measured IQ<br>Manifest and IQ integrity validation<br>Correlation, constellation, PAPR, and JSON diagnostics<br>CRC-independent raw-RGB image preview<br>Bundled BPSK, QPSK, 16QAM, and 64QAM point-to-point OTA IQ with regression tests | Expanding OTA datasets across distances and channel conditions<br>Improving short-capture, low-SNR, and difficult-multipath robustness<br>Adding BER, EVM, and soft-information diagnostics | FEC and coded/uncoded comparisons<br>Additional pilot layouts, estimators, and equalizers<br>Learnable encoder, equalizer, detector, and waveform blocks<br>PyTorch/JAX adapters and AI dataset export<br>Interactive experiment dashboard<br>Standard-oriented PHY profiles where useful |

## Modular PHY

`main.py` is the main program. It constructs the transmitter and receiver from small modules in physical-layer order:

```text
TX: coding.py -> modulation.py -> resource_mapping.py -> idft.py
    -> add_cp.py -> parallel_to_serial.py -> frame_assembly.py

RX: frame_sync.py -> timing_sync.py -> frequency_sync.py
    -> serial_to_parallel.py -> remove_cp.py -> dft.py
    -> channel_estimation.py -> equalization.py
    -> resource_demapping.py -> demodulation.py -> decoding.py
```

The main frequency-domain and time-domain grids use shape `[N, F]`, where `N` is the number of subcarriers and `F` is the number of OFDM symbols. The precise shape and dtype of every boundary are listed in [`docs/MODULE_INTERFACES.md`](docs/MODULE_INTERFACES.md). `TxFrameTrace` and `RxFrameTrace` expose the intermediate arrays produced by a run.

OFDM parameters are defined in `d2d_local/ofdm_conf.py`. Protocol and channel settings are defined in `d2d_local/config.py`.

## Quick Start

### Image simulation

Run the default QPSK link with the included NJU image:

```powershell
python main.py
# equivalent to:
python main.py simulate
```

Run 16QAM through the deterministic multipath profile:

```powershell
python main.py simulate samples/NJU.jpg `
  --bits-per-symbol 4 `
  --channel-config configs/multipath_channel.json `
  --output-dir outputs/simulate_16qam_multipath
```

The modulation choices are `1`, `2`, `4`, and `6` bits per symbol for BPSK, QPSK, 16QAM, and 64QAM.

After decoding, `main.py` opens one result window containing only the constellation and recovered image. Simulation uses raw RGB payloads by default, so the received image is still reconstructed when CRC fails and the image panel is marked `CRC FAILED - PREVIEW ONLY`. Synchronization and PAPR diagnostics are still saved under the output directory. Use `--payload-mode file` for exact file-byte transfer or `--no-show` for automated and headless runs.

### Point-to-point OTA IQ decoding

Decode a measured IQ capture described by a manifest:

```powershell
python main.py replay `
  --manifest path/to/manifest.json `
  --output-dir outputs/ota_decode
```

The manifest supplies the IQ format, sample count, checksum, OFDM profile, protocol settings, and expected payload information. The input IQ file uses little-endian `complex64` (`.fc32`) samples.

After installing a validated sample bundle under `samples/ota/`, select it directly:

```powershell
python main.py replay --capture bpsk
python main.py replay --capture qpsk
python main.py replay --capture 16qam
python main.py replay --capture 64qam
```

## Outputs

Each run writes:

```text
outputs/<run>/
  recovered/<original-name>
  diagnostics/correlation.png
  diagnostics/constellation.png
  diagnostics/papr_ccdf.png      # simulation only
  summary.json
```

CRC is reported as an integrity flag and does not suppress image output. In raw-RGB mode, missing or corrupted payload bytes appear directly as damaged pixels; the result window marks the image as `CRC PASSED` or `CRC FAILED - PREVIEW ONLY`. Diagnostic plots and `summary.json` are retained for every run.

## Python Interface

```python
from d2d_local import SimulationConfig, decode_capture, load_ota_capture, simulate_capture

simulated = simulate_capture("samples/NJU.jpg", SimulationConfig(payload_mode="raw_rgb"))
simulation_result = decode_capture(simulated, "outputs/api_simulation")

measured = load_ota_capture("path/to/manifest.json")
ota_result = decode_capture(measured, "outputs/api_ota")
```

Both input paths produce a `CaptureBundle` containing IQ samples, `OfdmConfig`, `ProtocolConfig`, provenance, and optional expected-payload metadata.

## Repository Layout

```text
main.py                    main simulation and measured-IQ entry point
d2d_local/                 PHY, protocol, channel, and diagnostic modules
d2d_local/ofdm_conf.py     OFDM parameters and carrier layout
d2d_local/phy_chain.py     transmitter/receiver composition and traces
docs/MODULE_INTERFACES.md  fixed module contracts
configs/                   deterministic channel profiles
samples/NJU.jpg            default transmission and test image
samples/ota/               measured-IQ bundle locations
tests/                     unit and end-to-end tests
```

## Current Limitations

- The baseline coding blocks perform padding, scrambling, and interleaving but do not yet include FEC.
- Demodulation currently uses hard decisions.
- A corrupted compressed file may no longer be readable as an image; raw-RGB mode is provided for visual error comparison.
- PMAT is a research frame format, not a Wi-Fi, LTE, or 5G NR profile.

## Roadmap

1. Expand the reproducible point-to-point OTA dataset across additional channel conditions.
2. Add FEC, soft information, BER/EVM reporting, and more channel estimators.
3. Add framework adapters for learned coding, equalization, detection, and waveform modules.
4. Build dataset and evaluation tools for AI-for-PHY experiments.
5. Extend the link to additional OTA conditions, multi-node experiments, and selected standard-oriented profiles.

## Related Work

- [*DBU-OFDM: A Trainable Deep Block-Unitary OFDM Waveform for Integrated Sensing and Communication*](https://arxiv.org/abs/2604.10296) is related to the broader goal of preserving the OFDM structure while introducing trainable blocks. OpenRadioPHY does not implement DBU-OFDM.

- *Contract-First Design for AI-Native Wireless Physical Layers* discusses the motivation and design principles behind OpenRadioPHY’s modular physical-layer architecture.

## License

OpenRadioPHY is released under the Apache License 2.0. See `LICENSE` and `NOTICE`.
