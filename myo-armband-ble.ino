/*
  Myo Armband BLE → UART Gateway
  ------------------------------
  Author: Mohamad Marwan Sidani

  Purpose
  - Connect to a Myo armband (or compatible EMG BLE peripheral)
  - Read 8-channel EMG samples (optionally from multiple BLE sources)
  - Print ASCII frames over UART for the host app to parse.

  Serial Protocol (one line per sample)
  eK,<timestamp_us>,v0,v1,v2,v3,v4,v5,v6,v7

  Where:
    K ∈ {0,1,2,3}  : stream index (use 0 if single device)
    timestamp_us   : source timestamp in microseconds
    v0..v7         : 8 signed EMG samples (e.g., -128..127)

  Baud: 115200 (match Python)

  Notes
  - Keep lines '\n' terminated.
  - Avoid extra prints to Serial to keep timing stable.
  - If you run multiple peripherals, round-robin the indices K.

  (Implementation-specific BLE pairing/characteristics go below.)
*/

#include <bluefruit.h>

// --- Myo base UUIDs (little-endian form used by Bluefruit) ---
const uint8_t UUID_MYO_CONTROL[16]   = {0x42,0x48,0x12,0x4A,0x7F,0x2C,0x48,0x47,0xB9,0xDE,0x04,0xA9,0x01,0x00,0x06,0xD5}; // d5060001-...
const uint8_t UUID_MYO_CMD_CHAR[16]  = {0x42,0x48,0x12,0x4A,0x7F,0x2C,0x48,0x47,0xB9,0xDE,0x04,0xA9,0x01,0x04,0x06,0xD5}; // d5060401-...

const uint8_t UUID_MYO_EMG_SVC[16]   = {0x42,0x48,0x12,0x4A,0x7F,0x2C,0x48,0x47,0xB9,0xDE,0x04,0xA9,0x05,0x00,0x06,0xD5}; // d5060005-...
const uint8_t UUID_MYO_EMG_0[16]     = {0x42,0x48,0x12,0x4A,0x7F,0x2C,0x48,0x47,0xB9,0xDE,0x04,0xA9,0x05,0x01,0x06,0xD5}; // d5060105-...
const uint8_t UUID_MYO_EMG_1[16]     = {0x42,0x48,0x12,0x4A,0x7F,0x2C,0x48,0x47,0xB9,0xDE,0x04,0xA9,0x05,0x02,0x06,0xD5}; // d5060205-...
const uint8_t UUID_MYO_EMG_2[16]     = {0x42,0x48,0x12,0x4A,0x7F,0x2C,0x48,0x47,0xB9,0xDE,0x04,0xA9,0x05,0x03,0x06,0xD5}; // d5060305-...
const uint8_t UUID_MYO_EMG_3[16]     = {0x42,0x48,0x12,0x4A,0x7F,0x2C,0x48,0x47,0xB9,0xDE,0x04,0xA9,0x05,0x04,0x06,0xD5}; // d5060405-...

// OPTIONAL: lock to your MAC (df:da:1a:1d:a6:35) — set true to use filter
const bool USE_MAC_FILTER = false;
const uint8_t MYO_MAC[6]  = {0xDF,0xDA,0x1A,0x1D,0xA6,0x35};

BLEClientService        svcControl(UUID_MYO_CONTROL);
BLEClientCharacteristic chrCommand(UUID_MYO_CMD_CHAR);

BLEClientService        svcEMG(UUID_MYO_EMG_SVC);
BLEClientCharacteristic chrEMG0(UUID_MYO_EMG_0);
BLEClientCharacteristic chrEMG1(UUID_MYO_EMG_1);
BLEClientCharacteristic chrEMG2(UUID_MYO_EMG_2);
BLEClientCharacteristic chrEMG3(UUID_MYO_EMG_3);

uint16_t conn_handle = BLE_CONN_HANDLE_INVALID;

// ---------- Helpers ----------
bool mac_match(const ble_gap_addr_t& addr) {
  for (int i = 0; i < 6; ++i) if (addr.addr[i] != MYO_MAC[i]) return false;
  return true;
}




// Vibrate: 1=short, 2=med, 3=long
void vibrate(uint8_t type = 1) {
  uint8_t cmd[3] = { 0x03, 0x01, type };
  chrCommand.write(cmd, sizeof(cmd));
}

// Set mode: EMG=raw(0x02), IMU=off(0x00), Classifier=off(0x00)
// Command format is: 0x01 <len=3> <emg> <imu> <classifier>
bool set_mode_emg_raw_only() {
  uint8_t payload[5] = { 0x01, 0x03, 0x02, 0x00, 0x00 };
  return chrCommand.write(payload, sizeof(payload));
}

// Prevent sleep (so it keeps streaming): 0x09 <len=1> <mode=0>
bool sleep_mode_never() {
  uint8_t cmd[3] = { 0x09, 0x01, 0x00 };
  return chrCommand.write(cmd, sizeof(cmd));
}

// Unlock hold (no double-tap needed): 0x0A <len=1> <hold=0x01>
bool unlock_hold() {
  uint8_t cmd[3] = { 0x0A, 0x01, 0x01 };
  return chrCommand.write(cmd, sizeof(cmd));
}

// Identify which EMG characteristic fired (0..3)
int emg_src_index(BLEClientCharacteristic* chr) {
  if (chr == &chrEMG0) return 0;
  if (chr == &chrEMG1) return 1;
  if (chr == &chrEMG2) return 2;
  if (chr == &chrEMG3) return 3;
  return -1;
}

void emg_notify_cb(BLEClientCharacteristic* chr, uint8_t* data, uint16_t len) {
  if (!data || len < 16) return;

  int src = emg_src_index(chr);
  if (src < 0) return;

  unsigned long t = micros();

  // Each packet usually has 2 samples × 8 channels (int8)
  // First sample
  Serial.print('e'); Serial.print(src); Serial.print(',');
  Serial.print(t); Serial.print(',');
  for (int i = 0; i < 8; ++i) {
    int8_t v = (int8_t)data[i];
    Serial.print(v);
    Serial.print(i < 7 ? ',' : '\n');
  }

  // Second sample
  Serial.print('e'); Serial.print(src); Serial.print(',');
  Serial.print(t); Serial.print(',');
  for (int i = 0; i < 8; ++i) {
    int8_t v = (int8_t)data[8 + i];
    Serial.print(v);
    Serial.print(i < 7 ? ',' : '\n');
  }
}


// ---------- Scan / Connect flow ----------
void scan_cb(ble_gap_evt_adv_report_t* report) {
  if (USE_MAC_FILTER && !mac_match(report->peer_addr)) {
    Bluefruit.Scanner.resume();
    return;
  }
  Bluefruit.Central.connect(report);
}

void connect_cb(uint16_t handle) {
  conn_handle = handle;
  Serial.println(F("Connected. Discovering Myo services..."));

  // Control service + Command characteristic
  if (!svcControl.discover(handle)) { Serial.println(F("No Control svc.")); Bluefruit.disconnect(handle); return; }
  if (!chrCommand.discover())       { Serial.println(F("No Command chr.")); Bluefruit.disconnect(handle); return; }

  // EMG service + 4 notify characteristics
  if (!svcEMG.discover(handle))     { Serial.println(F("No EMG svc."));     Bluefruit.disconnect(handle); return; }
  if (!chrEMG0.discover())          { Serial.println(F("No EMG0 chr."));    Bluefruit.disconnect(handle); return; }
  if (!chrEMG1.discover())          { Serial.println(F("No EMG1 chr."));    Bluefruit.disconnect(handle); return; }
  if (!chrEMG2.discover())          { Serial.println(F("No EMG2 chr."));    Bluefruit.disconnect(handle); return; }
  if (!chrEMG3.discover())          { Serial.println(F("No EMG3 chr."));    Bluefruit.disconnect(handle); return; }

  // Configure: raw EMG only, never sleep, unlock
  if (!set_mode_emg_raw_only()) { Serial.println(F("Failed set_mode.")); Bluefruit.disconnect(handle); return; }
  if (!sleep_mode_never())      { Serial.println(F("Failed sleep_off.")); Bluefruit.disconnect(handle); return; }
  if (!unlock_hold())           { Serial.println(F("Failed unlock."));    Bluefruit.disconnect(handle); return; }

  // Subscribe to all four EMG streams (Notify)
  chrEMG0.setNotifyCallback(emg_notify_cb);
  chrEMG1.setNotifyCallback(emg_notify_cb);
  chrEMG2.setNotifyCallback(emg_notify_cb);
  chrEMG3.setNotifyCallback(emg_notify_cb);

  bool ok = true;
  ok &= chrEMG0.enableNotify();
  ok &= chrEMG1.enableNotify();
  ok &= chrEMG2.enableNotify();
  ok &= chrEMG3.enableNotify();

  if (!ok) { Serial.println(F("Failed to enable EMG notifications.")); Bluefruit.disconnect(handle); return; }

  vibrate(1);
  Serial.println(F("Ready. Streaming EMG as CSV (e,ch0..ch7) @ ~200 Hz."));
}

void disconnect_cb(uint16_t, uint8_t) {
  conn_handle = BLE_CONN_HANDLE_INVALID;
  Serial.println(F("Disconnected. Rescanning..."));
  Bluefruit.Scanner.start(0);
}

// ---------- Setup / Loop ----------
void setup() {
  Serial.begin(115200);
  while (!Serial) { delay(5); }

  Bluefruit.begin(0, 1); // central only
  Bluefruit.setName("Feather-Myo-EMG");
  Bluefruit.Central.setConnectCallback(connect_cb);
  Bluefruit.Central.setDisconnectCallback(disconnect_cb);

  // Begin client objects
  svcControl.begin();  chrCommand.begin();
  svcEMG.begin();
  chrEMG0.begin();  chrEMG1.begin();  chrEMG2.begin();  chrEMG3.begin();

  // Scanner
  Bluefruit.Scanner.setRxCallback(scan_cb);
  Bluefruit.Scanner.restartOnDisconnect(true);
  Bluefruit.Scanner.useActiveScan(true);
  Bluefruit.Scanner.filterUuid(UUID_MYO_CONTROL);   // Myo advertises control UUID
  Bluefruit.Scanner.setInterval(160, 80);
  Bluefruit.Scanner.start(0);

  Serial.println(F("Scanning for Myo..."));
}

void loop() { }
