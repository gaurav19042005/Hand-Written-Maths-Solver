"""
app.py
Tkinter GUI: draw a math equation on a canvas, click Solve, see the answer.

Controls
────────
  Draw        – left-click drag
  Clear       – Clear button  (or press 'c')
  Solve       – Solve button  (or press Enter)
  Quit        – Quit button   (or press Escape)
"""

import tkinter as tk
from tkinter import font as tkfont
import numpy as np
import cv2
import os
import sys

# ── project modules ───────────────────────────────────────────────────────────
from segment import get_segments
from predict import predict_sequence
from solver  import solve_from_symbols

# ── constants ─────────────────────────────────────────────────────────────────
CANVAS_W   = 700
CANVAS_H   = 160
PEN_WIDTH  = 12
BG_COLOR   = "#1a1a2e"      # dark background (white strokes show clearly)
FG_COLOR   = "#ffffff"
BTN_SOLVE  = "#4ecca3"
BTN_CLEAR  = "#e94560"
FONT_MAIN  = ("Consolas", 28, "bold")
FONT_SMALL = ("Consolas", 13)
TMP_IMG    = "_tmp_eq.png"


class MathSolverApp:
    def __init__(self, root: tk.Tk):
        self.root = root
        root.title("Handwritten Math Equation Solver")
        root.configure(bg="#0f0f23")
        root.resizable(False, False)

        self._build_ui()
        self._bind_keys()

    # ── UI Construction ───────────────────────────────────────────────────────

    def _build_ui(self):
        pad = dict(padx=12, pady=6)

        # ── Title
        tk.Label(self.root, text="✏  Handwritten Math Solver",
                 font=("Consolas", 16, "bold"),
                 bg="#0f0f23", fg="#4ecca3").pack(pady=(14, 4))

        tk.Label(self.root,
                 text="Draw your equation below (e.g. 25 + 37 =)",
                 font=FONT_SMALL, bg="#0f0f23", fg="#aaaaaa").pack()

        # ── Drawing canvas
        canvas_frame = tk.Frame(self.root, bg="#0f0f23",
                                highlightbackground="#4ecca3",
                                highlightthickness=2)
        canvas_frame.pack(padx=20, pady=10)

        self.canvas = tk.Canvas(canvas_frame, width=CANVAS_W, height=CANVAS_H,
                                bg=BG_COLOR, cursor="crosshair",
                                highlightthickness=0)
        self.canvas.pack()

        self.canvas.bind("<ButtonPress-1>",   self._on_press)
        self.canvas.bind("<B1-Motion>",       self._on_drag)
        self.canvas.bind("<ButtonRelease-1>", self._on_release)

        # ── Buttons
        btn_frame = tk.Frame(self.root, bg="#0f0f23")
        btn_frame.pack(pady=6)

        self._btn(btn_frame, "⚡ Solve  [Enter]", BTN_SOLVE,
                  self.solve).pack(side=tk.LEFT, **pad)
        self._btn(btn_frame, "🗑 Clear  [C]", BTN_CLEAR,
                  self.clear).pack(side=tk.LEFT, **pad)
        self._btn(btn_frame, "✕ Quit  [Esc]", "#555566",
                  self.root.quit).pack(side=tk.LEFT, **pad)

        # ── Result display
        result_frame = tk.Frame(self.root, bg="#0f0f23")
        result_frame.pack(fill=tk.X, padx=20, pady=(4, 4))

        tk.Label(result_frame, text="Expression :",
                 font=FONT_SMALL, bg="#0f0f23", fg="#888888").grid(
                 row=0, column=0, sticky="w")
        self.lbl_expr = tk.Label(result_frame, text="—",
                                  font=FONT_SMALL,
                                  bg="#0f0f23", fg="#e2e2e2")
        self.lbl_expr.grid(row=0, column=1, sticky="w", padx=8)

        tk.Label(result_frame, text="Answer     :",
                 font=FONT_MAIN, bg="#0f0f23", fg="#888888").grid(
                 row=1, column=0, sticky="w")
        self.lbl_answer = tk.Label(result_frame, text="—",
                                    font=FONT_MAIN,
                                    bg="#0f0f23", fg=BTN_SOLVE)
        self.lbl_answer.grid(row=1, column=1, sticky="w", padx=8)

        # ── Status bar
        self.lbl_status = tk.Label(self.root, text="Ready",
                                    font=("Consolas", 11),
                                    bg="#0f0f23", fg="#666677")
        self.lbl_status.pack(pady=(0, 10))

        # Internal drawing state
        self._prev_x = None
        self._prev_y = None
        self._has_drawing = False

    def _btn(self, parent, text, color, cmd):
        return tk.Button(parent, text=text, command=cmd,
                         font=("Consolas", 12, "bold"),
                         bg=color, fg="#0f0f23",
                         activebackground=color, activeforeground="#0f0f23",
                         relief=tk.FLAT, padx=14, pady=6, cursor="hand2")

    def _bind_keys(self):
        self.root.bind("<Return>",  lambda e: self.solve())
        self.root.bind("<KP_Enter>",lambda e: self.solve())
        self.root.bind("c",        lambda e: self.clear())
        self.root.bind("C",        lambda e: self.clear())
        self.root.bind("<Escape>", lambda e: self.root.quit())

    # ── Drawing events ────────────────────────────────────────────────────────

    def _on_press(self, event):
        self._prev_x = event.x
        self._prev_y = event.y

    def _on_drag(self, event):
        if self._prev_x is not None:
            self.canvas.create_line(
                self._prev_x, self._prev_y, event.x, event.y,
                fill=FG_COLOR, width=PEN_WIDTH,
                capstyle=tk.ROUND, smooth=True)
        self._prev_x = event.x
        self._prev_y = event.y
        self._has_drawing = True

    def _on_release(self, event):
        self._prev_x = None
        self._prev_y = None

    # ── Actions ───────────────────────────────────────────────────────────────

    def clear(self):
        self.canvas.delete("all")
        self.lbl_expr.config(text="—")
        self.lbl_answer.config(text="—", fg=BTN_SOLVE)
        self.lbl_status.config(text="Cleared")
        self._has_drawing = False

    def solve(self):
        if not self._has_drawing:
            self.lbl_status.config(text="Draw an equation first!")
            return

        self.lbl_status.config(text="Processing…")
        self.root.update()

        try:
            # 1) Save canvas as image
            self._save_canvas_image(TMP_IMG)

            # 2) Segment
            segs = get_segments(TMP_IMG)
            if not segs:
                self._show_error("No symbols detected — try drawing larger")
                return

            # 3) Predict
            predictions = predict_sequence(segs)
            symbols = [sym for sym, _ in predictions]

            # 4) Solve
            result = solve_from_symbols(symbols)

            # 5) Display
            self.lbl_expr.config(text=result["expression"] or "—")
            if result["error"]:
                self.lbl_answer.config(text=result["answer"], fg=BTN_CLEAR)
            else:
                self.lbl_answer.config(
                    text=f'{result["expression"]} = {result["answer"]}',
                    fg=BTN_SOLVE)

            conf_avg = np.mean([c for _, c in predictions]) * 100
            self.lbl_status.config(
                text=f'Symbols: {" ".join(symbols)}   |   avg confidence: {conf_avg:.1f}%')

        except FileNotFoundError as e:
            self._show_error(f"Model not found: {e}\nRun  python train_model.py  first.")
        except Exception as e:
            self._show_error(str(e))

    def _show_error(self, msg: str):
        self.lbl_answer.config(text="Error", fg=BTN_CLEAR)
        self.lbl_status.config(text=msg)

    def _save_canvas_image(self, path: str):
        """Render the canvas to a PNG file via PostScript → OpenCV."""
        ps = self.canvas.postscript(colormode='gray')

        # Try Ghostscript path first (best quality)
        try:
            import subprocess, tempfile
            with tempfile.NamedTemporaryFile(suffix=".ps",
                                             delete=False) as f:
                f.write(ps.encode())
                ps_path = f.name
            subprocess.run(
                ["gs", "-dNOPAUSE", "-dBATCH", "-dSAFER",
                 "-sDEVICE=pngmono",
                 f"-sOutputFile={path}", ps_path],
                check=True, capture_output=True)
            os.unlink(ps_path)
        except Exception:
            # Fallback: use PIL via Pillow if available
            try:
                from PIL import Image, ImageGrab
                # Grab the canvas widget directly
                x = self.canvas.winfo_rootx()
                y = self.canvas.winfo_rooty()
                w = x + CANVAS_W
                h = y + CANVAS_H
                img = ImageGrab.grab(bbox=(x, y, w, h)).convert("L")
                img.save(path)
            except Exception:
                # Last resort: numpy black canvas with drawn pixels
                # (won't work without PS/PIL, but keeps app alive)
                blank = np.zeros((CANVAS_H, CANVAS_W), dtype=np.uint8)
                cv2.imwrite(path, blank)


# ── Entry point ───────────────────────────────────────────────────────────────

if __name__ == "__main__":
    # Check model exists
    if not os.path.exists("model.h5"):
        print("ERROR: model.h5 not found.")
        print("Please run:  python train_model.py")
        sys.exit(1)

    root = tk.Tk()
    app  = MathSolverApp(root)
    root.mainloop()

    # Clean up temp file
    if os.path.exists("_tmp_eq.png"):
        os.remove("_tmp_eq.png")
