# Serial Protocol (BLE EMG → UART → Host)

**Line format (ASCII, one frame per line):**
eK,<timestamp_us>,v0,v1,v2,v3,v4,v5,v6,v7
- `e` : literal prefix
- `K` : stream index `0..3` (allows up to 4 parallel BLE sources)
- `timestamp_us` : source‐side timestamp in microseconds (int)
- `v0..v7` : 8 signed EMG samples (ints, typically −128..127)

**Examples**
e0,12345678,-4,7,2,-1,0,1,-2,5
e1,12345720,3,1,0,-1,2,3,1,-2


**Merging**
- The Python app expects a round-robin sequence: `e0,e1,e2,e3,e0,...`
- If one stream lags, the app will resync gently (skip toward the longest queue).

**Baud**
- 115200 default (see Python `BAUD`).
