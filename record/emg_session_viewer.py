#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
EMG Session Inspector (bulletproof CSV decoding)
------------------------------------------------
- Pick a session from emg_sessions/<Participant>/<YYYYmmdd_HHMMSS>/.
- Load semicolon CSVs (samples.csv, events.csv) with robust decoding:
  tries utf-8 → cp1252 → latin-1, then utf-8(errors='replace').
- Plot 8 EMG channels (2×4 grid), overlay:
    • Class intervals (from samples.active_label_id), toggle per class
    • Vertical marker lines (from events.csv), toggle per event type
"""

import os, sys, csv
from collections import defaultdict, namedtuple

import numpy as np
from PyQt5 import QtCore, QtGui, QtWidgets
import pyqtgraph as pg

SESSIONS_ROOT = "emg_sessions"

LABELS = {
    0: "rest",
    1: "mild_extension",
    2: "full_extension",
    3: "mild_flexion",
    4: "full_flexion",
    5: "mild_radial_flexion",
    6: "full_radial_flexion",
    7: "mild_ulnar_flexion",
    8: "full_ulnar_flexion",
}

CLASS_COLORS = {
    0:  QtGui.QColor("#7f8c8d"),
    1:  QtGui.QColor("#82e0aa"),
    2:  QtGui.QColor("#27ae60"),
    3:  QtGui.QColor("#74b9ff"),
    4:  QtGui.QColor("#0984e3"),
    5:  QtGui.QColor("#f9ca24"),
    6:  QtGui.QColor("#f39c12"),
    7:  QtGui.QColor("#e056fd"),
    8:  QtGui.QColor("#8e44ad"),
}

MARKER_COLORS = {
    "GET_READY":          QtGui.QColor("#e67e22"),
    "MOVE_START":         QtGui.QColor("#2ecc71"),
    "MOVE_END":           QtGui.QColor("#2ecc71"),
    "MOVE_ABORT":         QtGui.QColor("#c0392b"),
    "PAUSE":              QtGui.QColor("#7f8c8d"),
    "RESUME":             QtGui.QColor("#7f8c8d"),
    "FAILED":             QtGui.QColor("#c0392b"),
    "NOISY":              QtGui.QColor("#d35400"),
    "AUTOPAUSE_NO_DATA":  QtGui.QColor("#e84393"),
    "SESSION_START":      QtGui.QColor("#2980b9"),
    "SESSION_END":        QtGui.QColor("#2980b9"),
}

PlotLayer = namedtuple("PlotLayer", ["graphics_items", "visible"])

# ---------- robust CSV loader ----------
def load_csv_table(path, delimiter=';'):
    """
    Returns (header:list[str] or None, rows:list[list[str]]).
    Tries utf-8 → cp1252 → latin-1; last resort utf-8 with replacement.
    """
    for enc in ("utf-8", "cp1252", "latin-1"):
        try:
            with open(path, "r", newline="", encoding=enc, errors="strict") as fp:
                reader = csv.reader(fp, delimiter=delimiter)
                header = next(reader, None)
                rows = [row for row in reader]
            return header, rows
        except UnicodeDecodeError:
            continue
    # final fallback: never crash
    with open(path, "r", newline="", encoding="utf-8", errors="replace") as fp:
        reader = csv.reader(fp, delimiter=delimiter)
        header = next(reader, None)
        rows = [row for row in reader]
    return header, rows

class SessionInspector(QtWidgets.QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("EMG Session Inspector")
        pg.setConfigOptions(antialias=True)

        # ==== Left panel ====
        left = QtWidgets.QWidget()
        left_layout = QtWidgets.QVBoxLayout(left)

        chooser = QtWidgets.QFormLayout()
        self.participant_combo = QtWidgets.QComboBox()
        self.session_combo = QtWidgets.QComboBox()
        chooser.addRow("Participant:", self.participant_combo)
        chooser.addRow("Session:", self.session_combo)
        left_layout.addLayout(chooser)

        btns = QtWidgets.QHBoxLayout()
        self.btn_refresh = QtWidgets.QPushButton("Refresh")
        self.btn_load = QtWidgets.QPushButton("Load")
        btns.addWidget(self.btn_refresh); btns.addWidget(self.btn_load)
        left_layout.addLayout(btns)

        left_layout.addWidget(self._section_label("Class intervals"))
        self.class_checks = {}
        grid_classes = QtWidgets.QGridLayout()
        for k in range(0, 9):
            cb = QtWidgets.QCheckBox(f"{k}: {LABELS[k].replace('_',' ')}")
            cb.setChecked(k != 0)  # hide 'rest' by default
            self.class_checks[k] = cb
            r, c = divmod(k, 2)
            grid_classes.addWidget(cb, r, c)
        left_layout.addLayout(grid_classes)

        left_layout.addWidget(self._section_label("Markers"))
        self.marker_checks = {}
        markers = list(MARKER_COLORS.keys())
        grid_markers = QtWidgets.QGridLayout()
        for i, m in enumerate(markers):
            cb = QtWidgets.QCheckBox(m); cb.setChecked(True)
            self.marker_checks[m] = cb
            r, c = divmod(i, 2)
            grid_markers.addWidget(cb, r, c)
        left_layout.addLayout(grid_markers)

        left_layout.addStretch()

        # ==== Right: plots ====
        right = QtWidgets.QWidget()
        grid = QtWidgets.QGridLayout(right)
        self.plots, self.curves = [], []
        for ch in range(8):
            pw = pg.PlotWidget()
            pw.showGrid(x=True, y=True, alpha=0.25)
            pw.setLabel("left", f"ch{ch}")
            pw.setLabel("bottom", "time (s)")
            curve = pw.plot([], [])
            if ch > 0:
                pw.setXLink(self.plots[0])
            r, c = divmod(ch, 4)
            grid.addWidget(pw, r, c)
            self.plots.append(pw); self.curves.append(curve)

        splitter = QtWidgets.QSplitter()
        splitter.addWidget(left); splitter.addWidget(right)
        splitter.setStretchFactor(1, 1)
        self.setCentralWidget(splitter)
        self.resize(1400, 800)

        # Data containers
        self.session_path = None
        self.time_s = None
        self.y = None
        self.labels = None
        self.events = []
        self.interval_layers = {k: PlotLayer([], True) for k in range(9)}
        self.marker_layers = {m: PlotLayer([], True) for m in MARKER_COLORS}

        # Signals
        self.btn_refresh.clicked.connect(self._refresh_participants)
        self.participant_combo.currentIndexChanged.connect(self._refresh_sessions)
        self.btn_load.clicked.connect(self._load_selected)
        for k, cb in self.class_checks.items():
            cb.stateChanged.connect(lambda _, kk=k: self._toggle_class_layer(kk))
        for m, cb in self.marker_checks.items():
            cb.stateChanged.connect(lambda _, mm=m: self._toggle_marker_layer(mm))

        self._refresh_participants()

    # ---------- UI helpers ----------
    def _section_label(self, text):
        lab = QtWidgets.QLabel(text); f = lab.font(); f.setBold(True); lab.setFont(f); return lab

    def _refresh_participants(self):
        self.participant_combo.blockSignals(True)
        self.participant_combo.clear()
        if os.path.isdir(SESSIONS_ROOT):
            parts = sorted([d for d in os.listdir(SESSIONS_ROOT)
                            if os.path.isdir(os.path.join(SESSIONS_ROOT, d))])
            self.participant_combo.addItems(parts)
        self.participant_combo.blockSignals(False)
        self._refresh_sessions()

    def _refresh_sessions(self):
        self.session_combo.clear()
        part = self.participant_combo.currentText().strip()
        if not part:
            return
        pdir = os.path.join(SESSIONS_ROOT, part)
        sessions = []
        if os.path.isdir(pdir):
            for d in sorted(os.listdir(pdir)):
                sdir = os.path.join(pdir, d)
                if os.path.isdir(sdir) and os.path.isfile(os.path.join(sdir, "samples.csv")):
                    sessions.append(d)
        self.session_combo.addItems(sessions)

    def _load_selected(self):
        part = self.participant_combo.currentText().strip()
        sess = self.session_combo.currentText().strip()
        if not part or not sess:
            QtWidgets.QMessageBox.warning(self, "Missing", "Pick a participant and a session.")
            return
        self.session_path = os.path.join(SESSIONS_ROOT, part, sess)
        if self._load_data():
            self._plot_all()

    # ---------- Data loading ----------
    def _load_data(self):
        samples_path = os.path.join(self.session_path, "samples.csv")
        events_path  = os.path.join(self.session_path, "events.csv")
        if not (os.path.isfile(samples_path) and os.path.isfile(events_path)):
            QtWidgets.QMessageBox.warning(self, "Missing files", "samples.csv or events.csv not found.")
            return False

        # samples.csv
        _, srows = load_csv_table(samples_path, delimiter=';')
        ts_us, labels = [], []
        chs = [[] for _ in range(8)]
        for row in srows:
            try:
                ts_us.append(int(row[0]))       # timestamp_us
                labels.append(int(row[3]))      # active_label_id
                for k in range(8):
                    chs[k].append(float(row[6+k]))  # ch0..ch7
            except Exception:
                continue
        if not ts_us:
            QtWidgets.QMessageBox.warning(self, "Empty", "No sample rows found.")
            return False

        ts0 = ts_us[0]
        time_s = (np.array(ts_us, dtype=np.int64) - ts0) / 1e6
        y = np.vstack([np.array(chs[k], dtype=np.float32) for k in range(8)])
        labels_arr = np.array(labels, dtype=np.int16)

        # events.csv
        _, erows = load_csv_table(events_path, delimiter=';')
        events = []
        for row in erows:
            try:
                t_s = (int(row[0]) - ts0) / 1e6
                events.append({"t": float(t_s), "event": row[2], "detail": row[3]})
            except Exception:
                continue

        self.time_s, self.y, self.labels, self.events = time_s, y, labels_arr, events
        return True

    # ---------- Plot building ----------
    def _clear_layers(self):
        for k in range(8):
            pw = self.plots[k]
            for it in list(pw.items()):
                if it is self.curves[k]:
                    continue
                try: pw.removeItem(it)
                except Exception: pass
        self.interval_layers = {k: PlotLayer([], self.class_checks[k].isChecked()) for k in range(9)}
        self.marker_layers = {m: PlotLayer([], self.marker_checks[m].isChecked()) for m in MARKER_COLORS}

    def _plot_all(self):
        self._clear_layers()

        x = self.time_s
        for ch in range(8):
            self.curves[ch].setData(x, self.y[ch, :])
            lo, hi = np.percentile(self.y[ch, :], [1, 99])
            pad = max(10.0, 0.1 * (hi - lo))
            self.plots[ch].setYRange(lo - pad, hi + pad, padding=0)

        intervals_by_label = self._build_intervals(self.labels, self.time_s)
        for label_id, intervals in intervals_by_label.items():
            self._add_class_intervals(label_id, intervals)

        markers_by_type = defaultdict(list)
        for ev in self.events:
            markers_by_type[ev["event"]].append(ev["t"])
        for mtype, times in markers_by_type.items():
            self._add_markers(mtype, times)

        for k in range(9): self._apply_class_visibility(k)
        for m in MARKER_COLORS.keys(): self._apply_marker_visibility(m)

        tmin, tmax = float(self.time_s[0]), float(self.time_s[-1])
        for pw in self.plots: pw.setXRange(tmin, tmax, padding=0)

    def _build_intervals(self, label_ids, times):
        intervals = defaultdict(list)
        if label_ids.size == 0:
            return intervals
        current = int(label_ids[0]); start = 0
        for i in range(1, len(label_ids)):
            if int(label_ids[i]) != current:
                t0 = float(times[start]); t1 = float(times[i-1])
                if t1 > t0: intervals[current].append((t0, t1))
                current = int(label_ids[i]); start = i
        t0 = float(times[start]); t1 = float(times[-1])
        if t1 > t0: intervals[current].append((t0, t1))
        return intervals

    def _add_class_intervals(self, label_id, intervals):
        color = CLASS_COLORS.get(label_id, QtGui.QColor("gray"))
        c = QtGui.QColor(color); c.setAlpha(60)
        items = []
        for (t0, t1) in intervals:
            for pw in self.plots:
                reg = pg.LinearRegionItem(values=(t0, t1), brush=c, movable=False)
                reg.setZValue(-10)
                pw.addItem(reg)
                items.append(reg)
        self.interval_layers[label_id] = PlotLayer(items, True)

    def _add_markers(self, marker_type, times):
        color = MARKER_COLORS.get(marker_type, QtGui.QColor("white"))
        pen = pg.mkPen(color=color, width=2, style=QtCore.Qt.SolidLine)
        items = []
        for t in times:
            for pw in self.plots:
                line = pg.InfiniteLine(pos=t, angle=90, pen=pen, movable=False)
                line.setZValue(10)
                pw.addItem(line)
                items.append(line)
        self.marker_layers[marker_type] = PlotLayer(items, True)

    # ---------- Toggles ----------
    def _toggle_class_layer(self, label_id):
        self.interval_layers[label_id] = self.interval_layers.get(label_id, PlotLayer([], True))
        self.interval_layers[label_id] = self.interval_layers[label_id]._replace(
            visible=self.class_checks[label_id].isChecked()
        )
        self._apply_class_visibility(label_id)

    def _apply_class_visibility(self, label_id):
        layer = self.interval_layers.get(label_id)
        if not layer: return
        vis = self.class_checks[label_id].isChecked()
        for it in layer.graphics_items:
            it.setVisible(vis)

    def _toggle_marker_layer(self, marker_type):
        self.marker_layers[marker_type] = self.marker_layers.get(marker_type, PlotLayer([], True))
        self.marker_layers[marker_type] = self.marker_layers[marker_type]._replace(
            visible=self.marker_checks[marker_type].isChecked()
        )
        self._apply_marker_visibility(marker_type)

    def _apply_marker_visibility(self, marker_type):
        layer = self.marker_layers.get(marker_type)
        if not layer: return
        vis = self.marker_checks[marker_type].isChecked()
        for it in layer.graphics_items:
            it.setVisible(vis)


def main():
    app = QtWidgets.QApplication(sys.argv)
    win = SessionInspector()
    win.show()
    sys.exit(app.exec_())

if __name__ == "__main__":
    main()
