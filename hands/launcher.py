# launcher.py
# GUI 실행기 (Tkinter)
# - 11_5.py 등 실행 대상 스크립트를 GUI에서 지정(Browse...)
# - 옵션 편집, 효과 파일 탐색, 프리셋 저장/불러오기(JSON)
# - 실시간 로그(stdout/stderr) 표시, 종료시 자식 프로세스 안전 종료

import os, sys, json, signal, subprocess, threading, queue, tkinter as tk
from tkinter import ttk, filedialog, messagebox

# Wayland 환경에서 OpenCV(Qt) 경고 회피
os.environ.setdefault("QT_QPA_PLATFORM", "xcb")

# 기본값
DEFAULTS = {
    "script": os.path.abspath("11_5.py"),  # 기본 실행 대상
    "camera": 6,
    "width": 1280,
    "height": 720,
    "fps": 30,
    "mirror": True,
    "model": "hand_landmarker.task",
    "effect": "A.mp4",
    "effect_width": 220,
    "effect_fps": 24.0,
    "shot_speed": 800.0,
    "shot_life": 1.2,
    "cooldown": 0.25,
    "max_hands": 2,
    "y_offset": 0,
    "target_rect": "300,280,180,90",
    "target_sensor": False,
    "impact_life": 0.0,
    "EXT_ON": 0.62,
    "EXT_OFF": 0.55,
    "THUMB_MIN": 0.40,
    "HOLD_GRACE_S": 0.20,
    "EDGE_FEATHER_PX": 4,
    "SPILL_REDUCE": 0.15,
}

PYTHON_EXE = sys.executable  # 현재 venv 파이썬

class Launcher(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("Finger-Gun Demo Launcher")
        self.geometry("920x720")
        self.minsize(920, 720)

        self.proc = None
        self.log_queue = queue.Queue()
        self.reader_threads = []

        self._build_ui()
        self._load_defaults_to_form()
        self.after(80, self._poll_log_queue)
        self.protocol("WM_DELETE_WINDOW", self.on_close)

    def _build_ui(self):
        frm = ttk.Frame(self, padding=10)
        frm.pack(fill="both", expand=True)
        left = ttk.Frame(frm); left.pack(side="left", fill="y")
        right = ttk.Frame(frm); right.pack(side="right", fill="both", expand=True)

        row = 0
        def add_row(label, widget):
            nonlocal row
            ttk.Label(left, text=label, width=18).grid(row=row, column=0, sticky="w", pady=2)
            widget.grid(row=row, column=1, sticky="we", pady=2)
            row += 1
        left.grid_columnconfigure(1, weight=1)

        # 실행 스크립트
        self.var_script = tk.StringVar()
        script_row = ttk.Frame(left)
        script_ent = ttk.Entry(script_row, textvariable=self.var_script, width=34)
        script_btn = ttk.Button(script_row, text="Browse...", command=self.browse_script)
        script_ent.pack(side="left", fill="x", expand=True)
        script_btn.pack(side="right")
        add_row("script (py)", script_row)

        # 카메라/크기/FPS
        self.var_camera = tk.IntVar();   add_row("camera", ttk.Entry(left, textvariable=self.var_camera, width=16))
        self.var_width  = tk.IntVar();   add_row("width",  ttk.Entry(left, textvariable=self.var_width,  width=16))
        self.var_height = tk.IntVar();   add_row("height", ttk.Entry(left, textvariable=self.var_height, width=16))
        self.var_fps    = tk.IntVar();   add_row("fps",    ttk.Entry(left, textvariable=self.var_fps,    width=16))

        self.var_mirror = tk.BooleanVar()
        add_row("mirror", ttk.Checkbutton(left, variable=self.var_mirror, text="enable"))

        # 모델/효과
        self.var_model = tk.StringVar(); add_row("model",  ttk.Entry(left, textvariable=self.var_model, width=28))

        self.var_effect = tk.StringVar()
        fx_row = ttk.Frame(left)
        fx_ent = ttk.Entry(fx_row, textvariable=self.var_effect, width=24)
        fx_btn = ttk.Button(fx_row, text="Browse...", command=self.browse_effect)
        fx_ent.pack(side="left", fill="x", expand=True)
        fx_btn.pack(side="right")
        add_row("effect (mp4)", fx_row)

        self.var_effect_w   = tk.IntVar();    add_row("effect_width", ttk.Entry(left, textvariable=self.var_effect_w, width=16))
        self.var_effect_fps = tk.DoubleVar(); add_row("effect_fps",   ttk.Entry(left, textvariable=self.var_effect_fps, width=16))

        # 발사 파라미터
        self.var_shot_speed = tk.DoubleVar(); add_row("shot_speed", ttk.Entry(left, textvariable=self.var_shot_speed, width=16))
        self.var_shot_life  = tk.DoubleVar(); add_row("shot_life",  ttk.Entry(left, textvariable=self.var_shot_life,  width=16))
        self.var_cooldown   = tk.DoubleVar(); add_row("cooldown",   ttk.Entry(left, textvariable=self.var_cooldown,   width=16))

        self.var_max_hands  = tk.IntVar();    add_row("max_hands", ttk.Entry(left, textvariable=self.var_max_hands, width=16))
        self.var_y_offset   = tk.IntVar();    add_row("y_offset",  ttk.Entry(left, textvariable=self.var_y_offset,  width=16))

        # 히트박스
        self.var_target_rect   = tk.StringVar(); add_row("target_rect", ttk.Entry(left, textvariable=self.var_target_rect, width=24))
        self.var_target_sensor = tk.BooleanVar(); add_row("target_sensor", ttk.Checkbutton(left, variable=self.var_target_sensor, text="enable"))

        self.var_impact_life = tk.DoubleVar(); add_row("impact_life", ttk.Entry(left, textvariable=self.var_impact_life, width=16))

        # 고급 제스처 / 크로마키
        self.var_EXT_ON        = tk.DoubleVar(); add_row("EXT_ON",        ttk.Entry(left, textvariable=self.var_EXT_ON,        width=16))
        self.var_EXT_OFF       = tk.DoubleVar(); add_row("EXT_OFF",       ttk.Entry(left, textvariable=self.var_EXT_OFF,       width=16))
        self.var_THUMB_MIN     = tk.DoubleVar(); add_row("THUMB_MIN",     ttk.Entry(left, textvariable=self.var_THUMB_MIN,     width=16))
        self.var_HOLD_GRACE_S  = tk.DoubleVar(); add_row("HOLD_GRACE_S",  ttk.Entry(left, textvariable=self.var_HOLD_GRACE_S,  width=16))
        self.var_EDGE_FEATHER  = tk.IntVar();    add_row("EDGE_FEATHER_PX", ttk.Entry(left, textvariable=self.var_EDGE_FEATHER, width=16))
        self.var_SPILL_REDUCE  = tk.DoubleVar(); add_row("SPILL_REDUCE",  ttk.Entry(left, textvariable=self.var_SPILL_REDUCE,  width=16))

        # 버튼들
        btns = ttk.Frame(left); btns.grid(row=row, column=0, columnspan=2, pady=(8, 2), sticky="we"); row += 1
        ttk.Button(btns, text="Run", command=self.run_proc).pack(side="left", padx=2)
        ttk.Button(btns, text="Stop", command=self.stop_proc).pack(side="left", padx=2)
        ttk.Button(btns, text="Save Preset", command=self.save_preset).pack(side="left", padx=8)
        ttk.Button(btns, text="Load Preset", command=self.load_preset).pack(side="left", padx=2)

        # 안내문
        help_txt = (
            "실행 창 단축키(11_5.py):  A: 히트박스 적용/숨김  E: 표시  S: 좌표 출력  R: 임팩트 리셋\n"
            "                         +/- (]/[, ↑/↓, 키패드 +/-): 임팩트 크기  Q/Esc: 종료"
        )
        ttk.Label(left, text=help_txt, justify="left", wraplength=360).grid(row=row, column=0, columnspan=2, sticky="we", pady=6)

        # 오른쪽: 커맨드 미리보기 + 로그
        cmd_box = ttk.LabelFrame(right, text="Command Preview")
        cmd_box.pack(fill="x")
        self.cmd_preview = tk.Text(cmd_box, height=3)
        self.cmd_preview.pack(fill="x")
        self.cmd_preview.configure(font=("Consolas", 10))

        log_label = ttk.Label(right, text="Log")
        log_label.pack(anchor="w")
        self.txt = tk.Text(right, height=30)
        self.txt.pack(fill="both", expand=True)
        self.txt.configure(font=("Consolas", 10))
        self.txt.tag_configure("ERR", foreground="#ff6060")
        self.txt.tag_configure("OUT", foreground="#d0d0d0")

    def _load_defaults_to_form(self):
        d = DEFAULTS
        self.var_script.set(d["script"])
        self.var_camera.set(d["camera"])
        self.var_width.set(d["width"])
        self.var_height.set(d["height"])
        self.var_fps.set(d["fps"])
        self.var_mirror.set(d["mirror"])
        self.var_model.set(d["model"])
        self.var_effect.set(d["effect"])
        self.var_effect_w.set(d["effect_width"])
        self.var_effect_fps.set(d["effect_fps"])
        self.var_shot_speed.set(d["shot_speed"])
        self.var_shot_life.set(d["shot_life"])
        self.var_cooldown.set(d["cooldown"])
        self.var_max_hands.set(d["max_hands"])
        self.var_y_offset.set(d["y_offset"])
        self.var_target_rect.set(d["target_rect"])
        self.var_target_sensor.set(d["target_sensor"])
        self.var_impact_life.set(d["impact_life"])
        self.var_EXT_ON.set(d["EXT_ON"])
        self.var_EXT_OFF.set(d["EXT_OFF"])
        self.var_THUMB_MIN.set(d["THUMB_MIN"])
        self.var_HOLD_GRACE_S.set(d["HOLD_GRACE_S"])
        self.var_EDGE_FEATHER.set(d["EDGE_FEATHER_PX"])
        self.var_SPILL_REDUCE.set(d["SPILL_REDUCE"])

    def browse_script(self):
        path = filedialog.askopenfilename(
            title="Select script (11_5.py)",
            filetypes=[("Python", "*.py"), ("All files", "*.*")]
        )
        if path:
            self.var_script.set(path)

    def browse_effect(self):
        path = filedialog.askopenfilename(
            title="Select effect video (mp4)",
            filetypes=[("Video", "*.mp4 *.mov *.mkv *.avi"), ("All files", "*.*")]
        )
        if path:
            self.var_effect.set(path)

    def _collect_args(self):
        args = [
            PYTHON_EXE, self.var_script.get(),
            "--camera", str(self.var_camera.get()),
            "--width",  str(self.var_width.get()),
            "--height", str(self.var_height.get()),
            "--fps",    str(self.var_fps.get()),
            "--model",  self.var_model.get(),
            "--effect", self.var_effect.get(),
            "--effect_width", str(self.var_effect_w.get()),
            "--effect_fps",   str(self.var_effect_fps.get()),
            "--shot_speed",   str(self.var_shot_speed.get()),
            "--shot_life",    str(self.var_shot_life.get()),
            "--cooldown",     str(self.var_cooldown.get()),
            "--max_hands",    str(self.var_max_hands.get()),
            "--y_offset",     str(self.var_y_offset.get()),
            "--target_rect",  self.var_target_rect.get(),
            "--impact_life",  str(self.var_impact_life.get()),
            "--EXT_ON",       str(self.var_EXT_ON.get()),
            "--EXT_OFF",      str(self.var_EXT_OFF.get()),
            "--THUMB_MIN",    str(self.var_THUMB_MIN.get()),
            "--HOLD_GRACE_S", str(self.var_HOLD_GRACE_S.get()),
            "--EDGE_FEATHER_PX", str(self.var_EDGE_FEATHER.get()),
            "--SPILL_REDUCE", str(self.var_SPILL_REDUCE.get()),
        ]
        if self.var_mirror.get():
            args.append("--mirror")
        if self.var_target_sensor.get():
            args.append("--target_sensor")
        return args

    def run_proc(self):
        script = self.var_script.get()
        if not os.path.exists(script):
            messagebox.showerror("Error", f"Script not found:\n{script}")
            return
        if self.proc and self.proc.poll() is None:
            messagebox.showinfo("Running", "Already running.")
            return

        args = self._collect_args()
        self._set_cmd_preview(args)
        self._append_log("OUT", f"$ {' '.join(args)}\n")
        try:
            env = os.environ.copy()
            env.setdefault("QT_QPA_PLATFORM", "xcb")
            self.proc = subprocess.Popen(
                args,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                env=env
            )
            # 로그 리더
            self.reader_threads = [
                threading.Thread(target=self._read_stream, args=(self.proc.stdout, "OUT"), daemon=True),
                threading.Thread(target=self._read_stream, args=(self.proc.stderr, "ERR"), daemon=True),
            ]
            for t in self.reader_threads: t.start()
        except Exception as e:
            messagebox.showerror("Launch failed", str(e))

    def stop_proc(self):
        if self.proc and self.proc.poll() is None:
            self._append_log("OUT", "[launcher] stopping...\n")
            try:
                if os.name == "nt":
                    self.proc.send_signal(signal.CTRL_BREAK_EVENT)
                else:
                    self.proc.terminate()
            except Exception as e:
                self._append_log("ERR", f"[launcher] terminate error: {e}\n")

    def _read_stream(self, stream, tag):
        for line in iter(stream.readline, ''):
            self.log_queue.put((tag, line))
        stream.close()

    def _poll_log_queue(self):
        try:
            while True:
                tag, line = self.log_queue.get_nowait()
                self._append_log(tag, line)
        except queue.Empty:
            pass
        if self.proc and self.proc.poll() is not None:
            rc = self.proc.returncode
            self._append_log("OUT", f"[launcher] process exited (code={rc})\n")
            self.proc = None
        self.after(80, self._poll_log_queue)

    def _append_log(self, tag, text):
        self.txt.insert("end", text, tag)
        self.txt.see("end")

    def _set_cmd_preview(self, args):
        self.cmd_preview.delete("1.0", "end")
        self.cmd_preview.insert("end", " ".join(args))

    def save_preset(self):
        data = {
            "script": self.var_script.get(),
            "camera": self.var_camera.get(),
            "width": self.var_width.get(),
            "height": self.var_height.get(),
            "fps": self.var_fps.get(),
            "mirror": self.var_mirror.get(),
            "model": self.var_model.get(),
            "effect": self.var_effect.get(),
            "effect_width": self.var_effect_w.get(),
            "effect_fps": self.var_effect_fps.get(),
            "shot_speed": self.var_shot_speed.get(),
            "shot_life": self.var_shot_life.get(),
            "cooldown": self.var_cooldown.get(),
            "max_hands": self.var_max_hands.get(),
            "y_offset": self.var_y_offset.get(),
            "target_rect": self.var_target_rect.get(),
            "target_sensor": self.var_target_sensor.get(),
            "impact_life": self.var_impact_life.get(),
            "EXT_ON": self.var_EXT_ON.get(),
            "EXT_OFF": self.var_EXT_OFF.get(),
            "THUMB_MIN": self.var_THUMB_MIN.get(),
            "HOLD_GRACE_S": self.var_HOLD_GRACE_S.get(),
            "EDGE_FEATHER_PX": self.var_EDGE_FEATHER.get(),
            "SPILL_REDUCE": self.var_SPILL_REDUCE.get(),
        }
        path = filedialog.asksaveasfilename(
            title="Save preset as JSON",
            defaultextension=".json",
            filetypes=[("JSON", "*.json")]
        )
        if path:
            with open(path, "w", encoding="utf-8") as f:
                json.dump(data, f, indent=2, ensure_ascii=False)
            self._append_log("OUT", f"[preset] saved: {path}\n")

    def load_preset(self):
        path = filedialog.askopenfilename(
            title="Load preset JSON",
            filetypes=[("JSON", "*.json"), ("All files", "*.*")]
        )
        if not path: return
        try:
            with open(path, "r", encoding="utf-8") as f:
                d = json.load(f)
            # 존재하는 키만 반영
            def set_if(k, var):
                if k in d: var.set(d[k])
            set_if("script", self.var_script)
            set_if("camera", self.var_camera)
            set_if("width", self.var_width)
            set_if("height", self.var_height)
            set_if("fps", self.var_fps)
            set_if("mirror", self.var_mirror)
            set_if("model", self.var_model)
            set_if("effect", self.var_effect)
            set_if("effect_width", self.var_effect_w)
            set_if("effect_fps", self.var_effect_fps)
            set_if("shot_speed", self.var_shot_speed)
            set_if("shot_life", self.var_shot_life)
            set_if("cooldown", self.var_cooldown)
            set_if("max_hands", self.var_max_hands)
            set_if("y_offset", self.var_y_offset)
            set_if("target_rect", self.var_target_rect)
            set_if("target_sensor", self.var_target_sensor)
            set_if("impact_life", self.var_impact_life)
            set_if("EXT_ON", self.var_EXT_ON)
            set_if("EXT_OFF", self.var_EXT_OFF)
            set_if("THUMB_MIN", self.var_THUMB_MIN)
            set_if("HOLD_GRACE_S", self.var_HOLD_GRACE_S)
            set_if("EDGE_FEATHER_PX", self.var_EDGE_FEATHER)
            set_if("SPILL_REDUCE", self.var_SPILL_REDUCE)
            self._append_log("OUT", f"[preset] loaded: {path}\n")
        except Exception as e:
            messagebox.showerror("Load failed", str(e))

    def on_close(self):
        if self.proc and self.proc.poll() is None:
            try:
                if os.name == "nt":
                    self.proc.send_signal(signal.CTRL_BREAK_EVENT)
                else:
                    self.proc.terminate()
            except Exception:
                pass
        self.destroy()

if __name__ == "__main__":
    try:
        Launcher().mainloop()
    except tk.TclError as e:
        print("[ERROR] Tkinter init failed. On Ubuntu: sudo apt install -y python3-tk", file=sys.stderr)
        raise
