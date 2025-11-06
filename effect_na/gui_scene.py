from PySide6.QtWidgets import (
    QApplication, QWidget, QLabel, QPushButton, QVBoxLayout, QStackedWidget, QProgressBar
)
from PySide6.QtCore import Qt, QTimer
from PySide6.QtGui import QFont, QColor, QPalette

from energy_skill import run_energy_skill


class TitleScene(QWidget):
    def __init__(self, switch_fn):
        super().__init__()
        self.switch_fn = switch_fn

        self.setStyleSheet("background-color: #1b1b2f;")

        title = QLabel("ENERGY SKILL")
        title.setFont(QFont("Arial", 48, QFont.Bold))
        title.setStyleSheet("color: white;")
        title.setAlignment(Qt.AlignCenter)

        btn = QPushButton("START")
        btn.setFixedHeight(60)
        btn.setStyleSheet("""
            QPushButton {
                background-color: #8a2be2;
                color: white;
                font-size: 24px;
                border-radius: 12px;
            }
            QPushButton:hover { background-color: #b06af5; }
            QPushButton:pressed { background-color: #6b1fb8; }
        """)
        btn.clicked.connect(self.start)

        layout = QVBoxLayout()
        layout.addStretch()
        layout.addWidget(title)
        layout.addWidget(btn, alignment=Qt.AlignCenter)
        layout.addStretch()
        self.setLayout(layout)

    def start(self):
        self.switch_fn(1)   # 로딩 씬으로 이동


class LoadingScene(QWidget):
    def __init__(self, switch_fn):
        super().__init__()
        self.switch_fn = switch_fn

        self.setStyleSheet("background-color: #0f0f17; color:white;")
        label = QLabel("Loading...")
        label.setFont(QFont("Arial", 30))
        label.setAlignment(Qt.AlignCenter)

        self.progress = QProgressBar()
        self.progress.setRange(0, 100)
        self.progress.setValue(0)
        self.progress.setStyleSheet("QProgressBar {height: 25px;}")

        layout = QVBoxLayout()
        layout.addStretch()
        layout.addWidget(label)
        layout.addWidget(self.progress)
        layout.addStretch()
        self.setLayout(layout)

        # 로딩 애니메이션 타이머
        self.timer = QTimer()
        self.timer.timeout.connect(self.update_progress)
        self.timer.start(40)

    def update_progress(self):
        val = self.progress.value() + 2
        self.progress.setValue(val)
        if val >= 100:
            self.timer.stop()
            self.switch_fn(2)     # 실행 화면으로 이동


class MainScene(QWidget):
    def __init__(self):
        super().__init__()
        self.setStyleSheet("background-color: #1e1e2e;")

        self.video = QLabel("Camera Feed")
        self.video.setAlignment(Qt.AlignCenter)
        self.video.setStyleSheet("""
            QLabel {
                background-color: #000000;
                color: white;
                border: 3px solid #8a2be2;
                border-radius: 10px;
                font-size: 22px;
            }
        """)
        self.video.setFixedHeight(600)

        btn = QPushButton("Start Energy Skill")
        btn.setFixedHeight(60)

        layout = QVBoxLayout()
        layout.setContentsMargins(50, 40, 50, 40)
        layout.addWidget(self.video)
        layout.addWidget(btn, alignment=Qt.AlignCenter)
        self.setLayout(layout)

        btn.clicked.connect(self.start_skill)
        self.timer = QTimer()

    def start_skill(self):
        update_fn = run_energy_skill(self.video, cam_index=4)
        self.timer.timeout.connect(update_fn)
        self.timer.start(16)


class SceneManager(QStackedWidget):
    def __init__(self):
        super().__init__()

        # ---
        # 씬 등록
        # ---
        self.title = TitleScene(self.switch_scene)
        self.loading = LoadingScene(self.switch_scene)
        self.main = MainScene()

        self.addWidget(self.title)     # index 0
        self.addWidget(self.loading)   # index 1
        self.addWidget(self.main)      # index 2

        self.setCurrentIndex(0)        # 처음엔 제목 씬

    def switch_scene(self, idx):
        self.setCurrentIndex(idx)


if __name__ == "__main__":
    app = QApplication([])
    w = SceneManager()
    w.resize(1280, 800)
    w.show()
    app.exec()
