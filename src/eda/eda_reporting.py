"""EDA 단계가 함께 사용하는 표·그림 저장과 기본 시각화 설정."""

import time
import warnings
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager


def log(message):
    print(f"[EDA {time.strftime('%H:%M:%S')}] {message}", flush=True)


def configure_plots():
    fonts = {font.name for font in font_manager.fontManager.ttflist}
    for font in [
        "AppleGothic",
        "Malgun Gothic",
        "NanumGothic",
        "Noto Sans CJK KR",
        "Noto Sans CJK JP",
    ]:
        if font in fonts:
            plt.rcParams["font.family"] = [font, "DejaVu Sans"]
            break
    else:
        warnings.warn("한글 폰트가 없어 그림의 한글이 깨질 수 있습니다.")
    plt.rcParams.update(
        {
            "axes.unicode_minus": False,
            "figure.dpi": 120,
            "font.size": 11,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.titlepad": 12,
        }
    )


class Report:
    def __init__(self, output_dir):
        self.root = Path(output_dir)
        self.tables = self.root / "tables"
        self.figures = self.root / "figures"
        self.tables.mkdir(parents=True, exist_ok=True)
        self.figures.mkdir(parents=True, exist_ok=True)
        self.figures_written = 0

    def save_table(self, frame, filename):
        frame.to_csv(
            self.tables / filename,
            index=False,
            encoding="utf-8-sig",
            date_format="%Y-%m-%d",
        )
        log(f"표 저장: {filename} ({len(frame):,}행)")

    def show_figure(self, fig, name, caption):
        fig.tight_layout()
        fig.savefig(
            self.figures / f"{name}.png",
            dpi=160,
            bbox_inches="tight",
            facecolor="white",
        )
        plt.close(fig)
        self.figures_written += 1
        log(f"그림 저장: {name}.png")
        self.say(caption)

    @staticmethod
    def say(text):
        print(text.replace("**", ""), flush=True)

    @staticmethod
    def display(frame):
        print(frame.to_string(index=False), flush=True)
