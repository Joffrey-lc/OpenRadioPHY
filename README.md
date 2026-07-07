# D2D_pyMat

**D2D_pyMat: A Python PHY Prototyping Platform for AI-Native Wireless Experiments**

D2D_pyMat is a USRP + Python OFDM link for modular physical-layer experimentation. It is built for rapid AI-for-PHY validation over real radio hardware, with a readable signal chain that can be inspected, modified, and extended without rebuilding a full wireless stack.

The long-term goal is a modular PHY playground where conventional blocks and learnable blocks can be swapped without changing the whole system. Researchers should be able to replace modulation, pilot layouts, channel estimation, equalization, PAPR control, coding, or neural receiver modules, and then validate the result on captured IQ or over real USRP hardware. The Python implementation is intentional: it keeps the PHY close to NumPy/PyTorch/JAX-style workflows and makes experiment code easier to connect to AI training and evaluation pipelines.

Current version: **v0.1.0**

<img src="https://camo.githubusercontent.com/ed412e2043fd14b4885f1610b5bd299fba63cb41334ad980536e8496790afe76/68747470733a2f2f6d796d61726b646f776e2d7069632e6f73732d636e2d6368656e6764752e616c6979756e63732e636f6d2f696d673232302f32303236303730373139313331353733332e706e67" alt="image-20260707191315323" style="zoom:15%;" />

## Roadmap / Project Status

| Implemented | In Progress / Experimental | Planned |
| :--- | :--- | :--- |
| Single-USRP TX/RX loopback<br>PMAT image/file framing<br>BPSK, QPSK, 16QAM, and 64QAM<br>Custom CP-OFDM modem<br>Zadoff-Chu preamble synchronization<br>Scattered pilot insertion<br>Segmented pilot-aided channel interpolation<br>CRC strict/debug modes<br>Raw-RGB debug payload mode<br>Constellation, correlation, PAPR, and JSON diagnostics | PAPR clipping and reporting workflow<br>Saved-IQ offline replay and debugging flow<br>Per-run reproducibility metadata<br>Robustness tuning for short captures and low-SNR runs | FEC and coded/uncoded comparisons<br>Neural receiver and neural equalizer hooks<br>Trainable waveform modules<br>Two-USRP and multi-node D2D topology<br>Dataset capture tools for AI-for-PHY experiments<br>Real-time visual dashboard<br>Standard-compliant profiles where needed |

## Highlights

- **Python-first PHY chain**: core framing, OFDM processing, demodulation, parsing, and diagnostics are implemented in Python.
- **Modular OFDM components**: modulation, pilots, preamble, synchronization, equalization, and payload handling are separated into replaceable modules.
- **Real-hardware validation path**: the main application runs a single-USRP loopback through GNU Radio/UHD and records raw IQ for repeatable analysis.
- **AI-oriented extension points**: the code is structured so conventional PHY blocks can later be replaced by learned waveform, equalization, detection, or decoding modules.
- **Diagnostics for PHY debugging**: constellation plots, preamble correlation, PAPR reports, CRC statistics, and summary JSON files are generated for each run.

## At a Glance

| Component | Main entry points | Purpose |
| :--- | :--- | :--- |
| USRP application | `python -m D2D_pyMat.apps.trx_usrp` | Transmit an image/file, capture IQ, decode frames, and write diagnostics |
| Protocol layer | `D2D_pyMat.protocol` | PMAT framing, metadata, chunking, CRC, and payload reconstruction |
| OFDM layer | `D2D_pyMat.ofdm` | CP-OFDM TX/RX, pilots, preamble detection, timing, equalization, and demapping |
| Modulation | `D2D_pyMat.qam` | BPSK, QPSK, 16QAM, and 64QAM mapping and hard decisions |
| Runtime | `D2D_pyMat.runtime` | GNU Radio custom blocks, UHD flowgraph, and streaming decode helpers |
| Diagnostics | `D2D_pyMat.viz` | Correlation, constellation, and PAPR visualization |

## Quick Navigation

- Project scope: [What It Is / What It Is Not](#what-it-is--what-it-is-not)
- Current PHY chain: [Current Link](#current-link)
- Hardware setup: [Physical Topology](#physical-topology), [Before the First USRP Run](#before-the-first-usrp-run)
- Running the link: [Requirements and Quick Start](#requirements-and-quick-start)
- Output files: [Payload, CRC, Diagnostics, and PAPR](#payload-crc-diagnostics-and-papr)
- Maintenance: [Versioning](#versioning), [Roadmap / Project Status](#roadmap--project-status)

## What It Is / What It Is Not

### What D2D_pyMat is

- A readable physical-layer prototyping platform for academic experiments.
- A Python implementation of an end-to-end image/file link over USRP hardware.
- A place to test synchronization, pilot design, equalization, modulation, PAPR handling, CRC behavior, and future AI-for-PHY blocks.
- A bridge between offline simulation and real-radio validation.

### What D2D_pyMat is not

- It is not a Wi-Fi, LTE, or 5G NR compliant stack.
- It is not a production communication stack.
- It is not designed for interoperability, certification, MAC-layer studies, or full network behavior.
- It does not currently include FEC, retransmission, multi-user scheduling, or a standard-compliant packet format.

This repository uses a research OFDM frame format. It borrows familiar OFDM design ideas such as preamble-based synchronization, pilot-aided equalization, guard carriers, and CP-OFDM, and its frame organization is inspired by 802.11a-style OFDM practice. It also follows the broader structure-preserving OFDM design philosophy discussed in the DBU-OFDM paper, where OFDM structure is preserved while leaving room for trainable blocks. However, this project does **not** implement 802.11a and does **not** implement DBU-OFDM. Use it for PHY prototyping, not for Wi-Fi interoperability experiments.

## Current Link

The implemented link is:

```text
image/file input
  -> PMAT protocol framing with metadata and CRC
  -> bit scrambling and block interleaving
  -> BPSK/QPSK/16QAM/64QAM mapping
  -> custom CP-OFDM framing with Zadoff-Chu preamble and scattered pilots
  -> GNU Radio/UHD streaming TX
  -> single-USRP loopback channel
  -> IQ capture to complex64 file
  -> preamble detection and CP fine timing
  -> segmented pilot-aided channel estimation and CPE correction
  -> frame parsing, CRC validation, and payload reconstruction
  -> constellation, synchronization, PAPR, and summary diagnostics
```

## Physical Topology

The default setup uses one USRP as both transmitter and receiver. The TX and RX ports can be connected through a controlled RF path or placed in a short-range over-the-air setup.

```text
Host PC running Python/GNU Radio/UHD
        |
        | Ethernet
        |
      USRP
   TX/RX port  ---------------- RF path ----------------  RX2 port
             coax + attenuator / splitter / antennas
```

Recommended starting topology:

- One USRP reachable at `addr=192.168.10.2`.
- TX antenna/port: `TX/RX`.
- RX antenna/port: `RX2`.
- Internal clock and time sources for a single-radio loopback: `--clock-source internal` and `--time-source internal`.
- Conservative TX amplitude and gain settings first; raise RX gain until the captured peak is comfortably above the noise floor without clipping.

<img src="https://camo.githubusercontent.com/39a865c71852c21e2c77a704035cee7a7772c59e31d86b3be026513dd5608dee/68747470733a2f2f6d796d61726b646f776e2d7069632e6f73732d636e2d6368656e6764752e616c6979756e63732e636f6d2f696d673232302f32303236303730373139333134373232362e706e67" alt="image-20260707193147163" style="zoom:25%;" />

## Repository Layout

```text
D2D_pyMat/
  apps/trx_usrp.py          Single-USRP image/file link application
  config.py                 Protocol and OFDM configuration objects
  protocol.py               PMAT framing, metadata, parsing, and CRC
  ofdm.py                   OFDM TX/RX, pilots, synchronization, equalization
  qam.py                    BPSK/QPSK/QAM mapping and hard decisions
  preamble.py               Zadoff-Chu preamble utilities
  interleaver.py            Scrambler and block interleaver
  viz.py                    Diagnostic plots
  runtime/gr_blocks.py      GNU Radio custom source block
  runtime/gr_flowgraphs.py  UHD/GNU Radio TRX flowgraph
  runtime/streaming.py      IQ stream decoding helpers
README.md
VERSION
```

`VERSION` and `D2D_pyMat.__version__` carry the release identifier. Runtime artifacts are written to `trx_outputs/` by default and are ignored by Git.

## Requirements and Quick Start

Use a Python environment with:

- Python 3.10 or compatible
- NumPy
- Pillow
- GNU Radio with UHD support
- A working UHD installation that can discover and configure the USRP

The command examples below use a Conda environment named `usrp` on Windows PowerShell. Run from the repository root so relative input and output paths resolve correctly.

### Before the First USRP Run

- Confirm the USRP can be discovered by UHD before running the Python application.
- Start with a controlled RF path. For a cabled loopback, use appropriate attenuation before connecting TX to RX.
- Use internal clock/time sources for one-USRP loopback. Use external references only when the hardware setup requires them.
- Start with QPSK strict/file mode and no PAPR clipping. Increase modulation order only after the QPSK baseline is clean.
- Keep the first run conservative: moderate TX gain, moderate RX gain, and enough `--duration` to cover a full payload cycle.

### Recommended First Run

```powershell
D:\anaconda\envs\usrp\python.exe -m D2D_pyMat.apps.trx_usrp cat.jpg `
  --addr addr=192.168.10.2 `
  --freq 3.6e9 `
  --sample-rate 2e6 `
  --tx-gain 20 `
  --rx-gain 20 `
  --amplitude 0.2 `
  --clock-source internal `
  --time-source internal `
  --tx-antenna TX/RX `
  --rx-antenna RX2 `
  --payload-size 1024 `
  --metadata-size 64 `
  --bits-per-symbol 2 `
  --repeats 1 `
  --continuous `
  --duration 6 `
  --tx-warmup-ms 500 `
  --rx-settle-ms 500 `
  --threshold-factor 0.5 `
  --tx-source-mode streaming `
  --crc-mode strict `
  --payload-mode file `
  --output-name cat_qpsk_strict.jpg `
  --constellation-frames 8 `
  --papr-report
```

Check the startup line printed by the app:

```text
payload_bytes=... frames=... one_cycle_seconds=... warmup_seconds=... capture_seconds=...
```

`--duration` is the requested usable receive window after TX warmup. The app captures `warmup_seconds + duration` seconds, then discards the configured RX settle interval before decoding. For reliable strict-mode recovery, choose:

```text
duration > one_cycle_seconds + margin
```

For a conservative first run, use QPSK (`--bits-per-symbol 2`) and omit `--papr-clip`.

## Payload, CRC, Diagnostics, and PAPR

### Payload and CRC Modes

`--crc-mode strict` is for validated file transfer. With `--payload-mode auto`, strict mode sends the original file bytes and writes output only when the frame CRCs and file CRC pass.

`--crc-mode debug` is for low-SNR visualization. With `--payload-mode auto`, debug mode sends raw RGB pixels and writes a best-effort PNG even when CRC validation fails. Missing chunks are filled with zeros, so the output can still show visible channel impairment.

Explicit payload modes are also available:

- `--payload-mode file`: transmit the input file bytes.
- `--payload-mode raw-rgb`: transmit expanded RGB pixels for visual debugging.
- `--payload-mode auto`: choose `file` for strict mode and `raw-rgb` for debug mode.

### Output Files

By default, outputs are written under `trx_outputs/`:

- `trx_pymat_<modulation>.fc32`: captured complex64 IQ samples.
- `diagnostics/trx_correlation_<modulation>.png`: preamble matched-filter metric.
- `diagnostics/trx_constellation_<modulation>.png`: equalized RX constellation.
- `diagnostics/trx_papr_ccdf_<modulation>.png`: TX PAPR CCDF when `--papr-report` is enabled.
- `trx_summary_<modulation>.json`: reproducibility summary with version, PHY settings, frame counts, CRC statistics, diagnostics paths, and timing information.
- Recovered output image/file when CRC and payload-mode conditions allow it.

The constellation plot length is controlled by:

```powershell
--constellation-frames 8
```

This parameter selects how many detected PHY frames are included in the plot. The number of OFDM symbols represented depends on modulation and frame size, and is recorded in the summary JSON.

### PAPR Controls

PAPR reporting is optional:

```powershell
--papr-report
```

This computes TX-side peak, RMS, PAPR in dB, and a CCDF plot before the USRP flowgraph starts.

Soft clipping is optional:

```powershell
--papr-clip 0.9
```

The limiter is disabled when `--papr-clip` is omitted. Use clipping only after a clean no-clipping baseline is established, because clipping can reduce PAPR while increasing EVM.

## Versioning

Current version: **v0.1.0**

The project uses semantic versioning:

- Patch versions: bug fixes and diagnostics that do not change the command-line contract.
- Minor versions: new PHY options, diagnostics, or compatible protocol extensions.
- Major versions: changes that break saved captures, protocol framing, or command-line compatibility.

The package exposes `D2D_pyMat.__version__`, the CLI supports `--version`, and every summary JSON records the version used for the run.

## Current Limitations

- The link does not yet include FEC or retransmission.
- Strict mode requires all chunks and the file CRC to pass.
- Equalization is pilot-aided and designed for the current single-USRP loopback profile.
- USRP gain, antenna path, and ADC utilization still need to be tuned for each RF setup.
- Multi-radio synchronization and multi-node networking are not included in v0.1.0.
- The current frame format is research-oriented and non-standard.

## Related Work

- [DBU-OFDM: A Trainable Deep Block-Unitary OFDM Waveform for Integrated Sensing and Communication](https://arxiv.org/abs/2604.10296) motivates structure-preserving, AI-enhanced OFDM design. D2D_pyMat does not implement DBU-OFDM, but it is aligned with the goal of keeping OFDM structure while leaving room for trainable modules.
- [OpenISAC](https://github.com/zhouzhiwen2000/OpenISAC) is an open-source real-time OFDM experimentation platform for ISAC research and is a useful reference for organizing over-the-air PHY projects.
