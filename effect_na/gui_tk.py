# -*- coding: utf-8 -*-
import sys
import tkinter as tk
from tkinter import ttk

from PySide6.QtWidgets import QApplication, QLabel
from PySide6.QtCore import Qt

from energy_skill import run_energy_skill   # ✅ 기존 함수 그대로 사용
from PySide6.QtCore import QTimer

class VideoWindow:
    def __init__(self, tk_parent, w=1280, h=720, cam_index=0):
        self.tk_parent = tk_parent

        self.qt_app = QApplication.instance()
        if not self.qt_app:
            self.qt_app = QApplication(sys.argv)

        self.label = QLabel()
        self.label.setWindowFlags(Qt.FramelessWindowHint | Qt.WindowStaysOnTopHint)
        self.label.setStyleSheet("background-color:black;")
        self.label.resize(w, h)
        self.center_video(w, h)
        self.label.show()

        # ✅ run_energy_skill 그대로 호출
        self.update_fn = run_energy_skill(self.label, cam_index)

        # ✅ PySide6 QTimer로 60fps 갱신
        self.timer = QTimer()
        self.timer.timeout.connect(self.update_frame)
        self.timer.start(16)

    def center_video(self, w, h):
        self.tk_parent.update_idletasks()
        x = self.tk_parent.winfo_rootx() + (self.tk_parent.winfo_width() - w) // 2
        y = self.tk_parent.winfo_rooty() + (self.tk_parent.winfo_height() - h) // 2
        self.label.setGeometry(x, y, w, h)

    def update_frame(self):
        self.update_fn()
        self.center_video(self.label.width(), self.label.height())
        QApplication.processEvents()


# ======================================
# ✅ Tkinter GUI (씬 전환 구조 그대로)
# ======================================
class App(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("Energy Skill Tk GUI (with PySide6 video)")
        self.geometry("1280x800")

        self.title_scene = TitleScene(self)
        self.loading_scene = LoadingScene(self)
        self.main_scene = MainScene(self)

        for scene in (self.title_scene, self.loading_scene, self.main_scene):
            scene.place(relwidth=1, relheight=1)

        self.show_scene(self.title_scene)

    def show_scene(self, scene):
        scene.tkraise()

        # PySide6 영상을 킨 뒤 씬을 전환하면
        # 영상만 계속 띄워져 있어도 GUI는 정상 동작


# ✅ Scene 1
class TitleScene(tk.Frame):
    def __init__(self, master):
        super().__init__(master, bg="#1b1b2f")

        title = tk.Label(self, text="ENERGY SKILL",
                         fg="white", bg="#1b1b2f",
                         font=("Arial", 48, "bold"))
        title.pack(pady=200)

        start_btn = tk.Button(self, text="START",
                              font=("Arial", 24, "bold"),
                              bg="#8a2be2", fg="white",
                              command=self.go_loading)
        start_btn.pack()

    def go_loading(self):
        self.master.show_scene(self.master.loading_scene)
        self.master.loading_scene.start_loading()


# ✅ Scene 2
class LoadingScene(tk.Frame):
    def __init__(self, master):
        super().__init__(master, bg="#0f0f17")

        label = tk.Label(self, text="Loading...",
                         font=("Arial", 30), bg="#0f0f17", fg="white")
        label.pack(pady=300)

        self.progress = ttk.Progressbar(self, orient="horizontal",
                                        length=450, mode="determinate")
        self.progress.pack()

    def start_loading(self):
        self.loading()

    def loading(self):
        v = self.progress["value"] + 2
        self.progress["value"] = v
        if v >= 100:
            self.master.show_scene(self.master.main_scene)
        else:
            self.after(30, self.loading)


# ✅ Scene 3 (버튼 눌렀을 때 PySide6 영상 실행)
class MainScene(tk.Frame):
    def __init__(self, master):
        super().__init__(master, bg="#1e1e2e")

        label = tk.Label(self, text="Press to Start Camera",
                         font=("Arial", 26), fg="white", bg="#1e1e2e")
        label.pack(pady=80)

        btn = tk.Button(self,
                        text="Start Energy Skill",
                        font=("Arial", 22, "bold"),
                        bg="#8a2be2", fg="white",
                        command=self.start_video)
        btn.pack(pady=30)

        self.video = None

    def start_video(self):
        if self.video is None:
            # ✅ 여기서 PySide6 영상창 실행!
            self.video = VideoWindow(
                self.master,
                w=1180, h=660,
                cam_index=4
            )


# ======================================
# ✅ 실행
# ======================================
if __name__ == "__main__":
    app = App()
    app.mainloop()
