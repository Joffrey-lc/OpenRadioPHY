# D2D_pyMat

**D2D_pyMat v0.1.0** is a Python-based OFDM image link for single-USRP loopback experiments. It provides an end-to-end reference chain from image framing to USRP transmission, IQ capture, PHY decoding, CRC validation, and diagnostic visualization.

The current release focuses on a reproducible single-radio TRX workflow that is useful for physical-layer prototyping, link debugging, and classroom or lab demonstrations.

<img src="https://mymarkdown-pic.oss-cn-chengdu.aliyuncs.com/img220/20260707191315733.png" alt="image-20260707191315323" style="zoom:18%;" />

## Current Link

The implemented link is:

```text
image/file input
  -> PMAT protocol framing with metadata and CRC
  -> bit scrambling and block interleaving
  -> BPSK/QPSK/16QAM/64QAM mapping
  -> CP-OFDM framing with Zadoff-Chu preamble and scattered pilots
  -> GNU Radio/UHD streaming TX
  -> single-USRP loopback channel
  -> IQ capture to complex64 file
  -> preamble detection and CP fine timing
  -> pilot-aided channel estimation and CPE correction
  -> frame parsing, CRC validation, and payload reconstruction
  -> constellation, synchronization, PAPR, and summary diagnostics
```

**Figure placeholder: end-to-end D2D_pyMat signal-processing chain.**

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
- Internal clock and time sources for a single-radio loopback.（Or config `--clock-source internal `  and `--time-source internal `）
- Conservative TX amplitude and gain settings first; raise RX gain until the captured peak is comfortably above the noise floor without clipping.

<img src="https://mymarkdown-pic.oss-cn-chengdu.aliyuncs.com/img220/20260707193147226.png" alt="image-20260707193147163" style="zoom:25%;" />

## Repository Layout

```text
D2D_pyMat/
  apps/trx_usrp.py          Single-USRP image link application
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

Runtime artifacts are written to `trx_outputs/` by default and are ignored by Git.

## Requirements

Use a Python environment with:

- Python 3.10 or compatible
- NumPy
- Pillow
- GNU Radio with UHD support
- A working UHD installation that can discover and configure the USRP

The development environment used for the current command examples is a Conda environment named `usrp`.

## Quick Start

Run from the repository root so relative input and output paths resolve correctly:

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
  --duration 3 `
  --tx-warmup-ms 500 `
  --rx-settle-ms 500 `
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

For a conservative first run, use QPSK (`--bits-per-symbol 2`) and no PAPR clipping.

## Payload and CRC Modes

`--crc-mode strict` is for validated file transfer. With `--payload-mode auto`, strict mode sends the original file bytes and writes output only when the frame CRCs and file CRC pass.

`--crc-mode debug` is for low-SNR visualization. With `--payload-mode auto`, debug mode sends raw RGB pixels and writes a best-effort PNG even when CRC validation fails. Missing chunks are filled with zeros, so the image can still show visible channel impairment.

Explicit payload modes are also available:

- `--payload-mode file`: transmit the input file bytes.
- `--payload-mode raw-rgb`: transmit expanded RGB pixels for visual debugging.
- `--payload-mode auto`: choose `file` for strict mode and `raw-rgb` for debug mode.

## Diagnostics

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

This parameter selects how many detected PHY frames are included in the plot. The number of OFDM symbols represented depends on the modulation and frame size, and is recorded in the summary JSON.

## PAPR Controls

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

## Roadmap

Planned development directions:

- Add FEC options and compare uncoded/coded image recovery.
- Add per-frame EVM, per-subcarrier EVM, and worst-frame diagnostics.
- Add richer pilot-layout experiments and channel-estimation modes.
- Add automated offline regression tests from saved IQ captures.
- Add a small configuration file format for reproducible experiment presets.
- Extend from single-USRP loopback to two-USRP and multi-node D2D experiments.
- Add optional real-time visualization for constellation, synchronization, and packet statistics.
- Package the project for easier installation and environment recreation.