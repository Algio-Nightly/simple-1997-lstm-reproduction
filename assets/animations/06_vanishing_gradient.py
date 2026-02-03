"""
Vanishing Gradient Comparison Animation

Shows why RNNs fail (gradients vanish) vs LSTM (gradients preserved via CEC).
This is the OPENING animation for the thesis presentation.

Scene: VanishingGradient
Duration: ~15 seconds

Run: cd assets/animations && manimgl 06_vanishing_gradient.py VanishingGradient -w
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from manimlib import *
from colors import GRADIENT_RED, CEC_BLUE, TEXT_WHITE, ACCENT_YELLOW, WEIGHT_GRAY


class VanishingGradient(InteractiveScene):
    
    def construct(self):
        title = Text(
            "The Vanishing Gradient Problem",
            color=TEXT_WHITE,
            font_size=48
        )
        title.to_edge(UP, buff=0.5)
        
        self.play(Write(title), run_time=1.5)
        self.wait(0.5)
        
        rnn_label = Text("Vanilla RNN", color=WEIGHT_GRAY, font_size=28)
        rnn_label.move_to(LEFT * 3.5 + UP * 1.5)
        
        lstm_label = Text("LSTM (CEC)", color=CEC_BLUE, font_size=28)
        lstm_label.move_to(RIGHT * 3.5 + UP * 1.5)
        
        divider = Line(
            UP * 2 + DOWN * 0.5,
            DOWN * 2.5,
            color=WEIGHT_GRAY,
            stroke_width=1
        )
        
        self.play(
            FadeIn(rnn_label),
            FadeIn(lstm_label),
            ShowCreation(divider),
            run_time=1
        )
        self.wait(0.5)
        
        timesteps = ["t-4", "t-3", "t-2", "t-1", "t"]
        rnn_magnitudes = [0.9, 0.6, 0.3, 0.1, 0.02]
        lstm_magnitudes = [1.0, 1.0, 1.0, 1.0, 1.0]
        
        max_bar_height = 1.5
        bar_width = 0.4
        spacing = 1.0
        
        rnn_bars = VGroup()
        rnn_time_labels = VGroup()
        rnn_start_x = -5.5
        bar_y = -0.5
        
        for i, (ts, mag) in enumerate(zip(timesteps, rnn_magnitudes)):
            bar = Rectangle(
                width=bar_width,
                height=max_bar_height * mag,
                color=GRADIENT_RED,
                fill_color=GRADIENT_RED,
                fill_opacity=0.8,
                stroke_width=1
            )
            bar.move_to([rnn_start_x + i * spacing, bar_y + max_bar_height * mag / 2, 0])
            
            time_label = Tex(ts, color=TEXT_WHITE, font_size=18)
            time_label.move_to([rnn_start_x + i * spacing, bar_y - 0.3, 0])
            
            rnn_bars.add(bar)
            rnn_time_labels.add(time_label)
        
        lstm_bars = VGroup()
        lstm_time_labels = VGroup()
        lstm_start_x = 1.5
        
        for i, (ts, mag) in enumerate(zip(timesteps, lstm_magnitudes)):
            bar = Rectangle(
                width=bar_width,
                height=max_bar_height * mag,
                color=CEC_BLUE,
                fill_color=CEC_BLUE,
                fill_opacity=0.8,
                stroke_width=1
            )
            bar.move_to([lstm_start_x + i * spacing, bar_y + max_bar_height * mag / 2, 0])
            
            time_label = Tex(ts, color=TEXT_WHITE, font_size=18)
            time_label.move_to([lstm_start_x + i * spacing, bar_y - 0.3, 0])
            
            lstm_bars.add(bar)
            lstm_time_labels.add(time_label)
        
        initial_rnn_bars = VGroup()
        for i in range(len(timesteps)):
            bar = Rectangle(
                width=bar_width,
                height=max_bar_height,
                color=GRADIENT_RED,
                fill_color=GRADIENT_RED,
                fill_opacity=0.8,
                stroke_width=1
            )
            bar.move_to([rnn_start_x + i * spacing, bar_y + max_bar_height / 2, 0])
            initial_rnn_bars.add(bar)
        
        self.play(
            *[FadeIn(bar) for bar in initial_rnn_bars],
            *[FadeIn(bar) for bar in lstm_bars],
            *[FadeIn(label) for label in rnn_time_labels],
            *[FadeIn(label) for label in lstm_time_labels],
            run_time=1.5
        )
        self.wait(0.5)
        
        self.play(
            *[Transform(initial_rnn_bars[i], rnn_bars[i]) for i in range(len(timesteps))],
            run_time=2
        )
        self.wait(0.5)
        
        rnn_status = Text("Gradient VANISHES", color=GRADIENT_RED, font_size=20)
        rnn_status.move_to(LEFT * 3.5 + DOWN * 2)
        
        lstm_status = Text("Gradient PRESERVED", color=CEC_BLUE, font_size=20)
        lstm_status.move_to(RIGHT * 3.5 + DOWN * 2)
        
        self.play(
            FadeIn(rnn_status),
            FadeIn(lstm_status),
            run_time=1
        )
        self.wait(1)
        
        takeaway = Text(
            "LSTM's CEC preserves gradients — vanilla RNN gradients vanish",
            color=TEXT_WHITE,
            font_size=24
        )
        takeaway.to_edge(DOWN, buff=0.3)
        
        self.play(FadeIn(takeaway, shift=UP * 0.3), run_time=1)
        self.wait(2)
        
        if os.getenv("MANIM_DEV"):
            self.embed()
