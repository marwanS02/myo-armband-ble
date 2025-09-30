# Wiring Notes

- **Gateway board**: Feather nRF52 / Teensy / MKR WiFi (as used in your setup)
- **Host link**: USB serial @ 115200 baud
- **Myo armband**: connects via BLE to the gateway (sketch handles pairing/data)
- **Power**: USB for gateway; ensure common GND if using external modules

> If you split BLE into multiple peripherals, each peripheral maps to a stream index K in `eK, ...`.
