# -*- coding: utf-8 -*-
import sys
import tkinter as tk
from tkinter import ttk

from gyeongjun_eff import run_energy_skill_tk   
from hand_human_lightsaber import run_lightsaber_tk  

class App(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("Energy Skill Tk GUI")
        self.geometry("1280x800")

        # ✅ 설정 값 기본
        self.cam_index = 0
        self.mirror_on = False

        # ✅ 씬 생성
        self.title_scene = TitleScene(self)
        self.loading_scene = LoadingScene(self)
        self.main_scene = MainScene(self)
        self.setting_scene = SettingScene(self)

        # 전체 배치
        for scene in (self.title_scene, self.loading_scene, self.main_scene, self.setting_scene):
            scene.place(relwidth=1, relheight=1)

        self.show_scene(self.title_scene)

    def show_scene(self, scene):
        scene.tkraise()


# ✅ Scene 1: Title
class TitleScene(tk.Frame):
    def __init__(self, master):
        super().__init__(master, bg="#1b1b2f")

        tk.Label(self, text="ENERGY SKILL",
                 fg="white", bg="#1b1b2f",
                 font=("Arial", 48, "bold")).pack(pady=200)

        tk.Button(self, text="START", font=("Arial", 24, "bold"),
                  bg="#8a2be2", fg="white",
                  width=12, height=2,
                  command=self.go_loading).pack()

    def go_loading(self):
        self.master.show_scene(self.master.loading_scene)
        self.master.loading_scene.start_loading()


# ✅ Scene 2: Loading
class LoadingScene(tk.Frame):
    def __init__(self, master):
        super().__init__(master, bg="#0f0f17")

        tk.Label(self, text="Loading...", font=("Arial", 30),
                 bg="#0f0f17", fg="white").pack(pady=300)

        self.progress = ttk.Progressbar(self, orient="horizontal",
                                        length=450, mode="determinate")
        self.progress.pack()

    def start_loading(self):
        self.progress["value"] = 0
        self.loading()

    def loading(self):
        v = self.progress["value"] + 2
        self.progress["value"] = v
        if v >= 100:
            self.master.show_scene(self.master.main_scene)
        else:
            self.after(30, self.loading)


# ✅ Scene 3: Main
class MainScene(tk.Frame):
    def __init__(self, master):
        super().__init__(master, bg="#1e1e2e")

        tk.Label(self, text="Main Screen",
                 font=("Arial", 32), fg="white", bg="#1e1e2e").pack(pady=20)

        self.video_label = tk.Label(self, bg="black")
        self.video_label.place(relx=0.5, rely=0.40, anchor="center",
                               width=1060, height=600)

        self.settings_btn = tk.Button(
            self, text="Settings", font=("Arial", 14, "bold"),
            bg="#444", fg="white",
            command=self.open_settings
        )
        self.after(100, self.place_settings_button)

        self.energy_skill_btn = tk.Button(
            self, text="Start Energy Skill",
            font=("Arial", 20, "bold"),
            bg="#8a2be2", fg="white",
            command=self.start_skill
        )
        self.energy_skill_btn.place(relx=0.3, rely=0.80)

        self.lightsaber_btn = tk.Button(
            self, text="Start Lightsaber",
            font=("Arial", 20, "bold"),
            bg="#8a2be2", fg="white",
            command=self.start_lightsaber
        )
        self.lightsaber_btn.place(relx=0.6, rely=0.80)

        self.stop_btn = tk.Button(
            self, text="Stop",
            font=("Arial", 18),
            bg="#c0392b", fg="white",
            command=self.stop_skill
        )
        self.stop_btn.place(relx=0.9, rely=0.80)

        # ✅ 상태
        self.update_fn = None
        self.running = False

    def place_settings_button(self):
        vx = self.video_label.winfo_x()
        vy = self.video_label.winfo_y()
        vw = self.video_label.winfo_width()
        self.settings_btn.place(x=vx + vw - 100, y=vy + 10)

    def start_skill(self):
        # ✅ 먼저 확실하게 초기화
        self.running = False
        self.update_fn = None

        # ✅ 예외 안전하게 VideoCapture 생성
        try:
            self.update_fn = run_energy_skill_tk(
                self.video_label,
                cam_index=self.master.cam_index,
                mirror=self.master.mirror_on
            )
        except Exception as e:
            print("Camera start error:", e)
            self.video_label.config(image="", bg="black")
            self.update_fn = None
            return

        self.running = True
        self.update_video()

    def start_lightsaber(self):
        # ✅ 먼저 확실하게 초기화
        self.running = False
        self.update_fn = None

        # ✅ 예외 안전하게 VideoCapture 생성
        try:
            self.update_fn = run_lightsaber_tk(
                self.video_label,
                cam_index=self.master.cam_index,
                mirror=self.master.mirror_on
            )
        except Exception as e:
            print("Camera start error:", e)
            self.video_label.config(image="", bg="black")
            self.update_fn = None
            return

        self.running = True
        self.update_video()


    def update_video(self):
        # after 루프가 중복되거나 고아 프로세스 되는 걸 방지
        if self.running and self.update_fn:
            try:
                self.update_fn()
            except Exception as e:
                print("Runtime error:", e)
                self.stop_skill()   # ✅ 오류 발생하면 강제로 정지
                return

        if self.running:
            self.after(16, self.update_video)

    def stop_skill(self):
        self.running = False
        self.update_fn = None
        self.video_label.config(image="", bg="black")
        self.video_label.imgtk = None

    def open_settings(self):
        self.stop_skill()  # ✅ 설정 들어갈 때도 확실히 정지
        self.master.show_scene(self.master.setting_scene)

        self.master.show_scene(self.master.setting_scene)

# ✅ Scene 4: Settings
class SettingScene(tk.Frame):
    def __init__(self, master):
        super().__init__(master, bg="#111")

        tk.Label(self, text="Settings",
                 font=("Arial", 40, "bold"),
                 bg="#111", fg="white").pack(pady=40)

        # ✅ Camera Index 선택: ComboBox
        tk.Label(self, text="Camera Index",
                 bg="#111", fg="white", font=("Arial", 18)).pack(pady=10)

        self.cam_box = ttk.Combobox(self, font=("Arial", 18), width=10, state="readonly")
        self.cam_box['values'] = [0, 1, 2, 3, 4]  # 선택지
        self.cam_box.current(0)
        self.cam_box.pack()

        # ✅ Mirror 체크
        self.mirror_var = tk.BooleanVar()
        tk.Checkbutton(self, text="Mirror Mode", variable=self.mirror_var,
                       font=("Arial", 18), bg="#111", fg="white",
                       selectcolor="#222").pack(pady=20)

        # ✅ Save 버튼
        tk.Button(self, text="Save",
                  font=("Arial", 20, "bold"),
                  bg="#8a2be2", fg="white",
                  command=self.save).pack(pady=20)

        # ✅ Back 버튼
        tk.Button(self, text="Back",
                  font=("Arial", 18),
                  bg="#444", fg="white",
                  command=self.back).pack(pady=10)

        # ✅ 저장 상태 표시용
        self.status_label = tk.Label(self, text="", bg="#111", fg="#aaa", font=("Arial", 14))
        self.status_label.pack(pady=10)

    def save(self):
        try:
            idx = int(self.cam_box.get())
            self.master.cam_index = idx
        except:
            self.status_label.config(text="Invalid camera index")
            return

        self.master.mirror_on = self.mirror_var.get()
        self.status_label.config(text="Settings saved.")
        self.master.show_scene(self.master.main_scene)


    def back(self):
        self.master.show_scene(self.master.main_scene)


# ✅ 실행
if __name__ == "__main__":
    app = App()
    app.mainloop()
