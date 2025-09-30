# Myo Armband BLE — Real-Time EMG (8 ch) + BLE Gateway

This repo streams **8-channel EMG** from a Myo armband (or compatible BLE source), merges multiple BLE streams, and renders real-time plots. An **Arduino/Feather gateway** forwards packets over UART to the host PC; Python visualizes the EMG.

## Features
- Real-time plotting of **8 channels** with PyQtGraph (2×4 grid)
- Tolerant parser for lines like: `eK,<timestamp_us>,v0,...,v7`
- Round-robin **merge of up to 4 BLE streams** (K ∈ {0,1,2,3})
- Adjustable window length, sample rate, and buffer sizes
- Minimal Arduino sketch (`Myoband.ino`) to forward BLE packets

## Repository layout
- `myo_emg_rt_pyqtgraph.py` — Python GUI/merger/serial reader
- `Myoband.ino` — Arduino/Feather sketch bridging BLE→UART
- `docs/protocol.md` — Exact serial format + timing rules
- `docs/wiring.md` — Hardware notes/pins and level-shifting (if any)

## Quick start (Python)
```bash
# 1) Python 3.9+ venv recommended
pip install -r requirements.txt

# 2) Set serial port in the script header:
#    PORT="COM5" on Windows, "/dev/ttyACM0" or "/dev/ttyUSB0" on Linux, "/dev/tty.usbmodem*" on macOS

# 3) Run
python myo_emg_rt_pyqtgraph.py
