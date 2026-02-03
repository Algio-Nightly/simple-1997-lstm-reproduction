"""
Long Time Lag Animation - Experiment 2 (Section 5.2)

Visualizes the Long Time Lag task where LSTM must store a value
at the start and reproduce it after a long delay filled with distractors.

Scene: LongTimeLag
Duration: ~20 seconds
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from manimlib import *
from colors import (
    CEC_BLUE, INPUT_GATE_GREEN, GRADIENT_RED, WEIGHT_GRAY, TEXT_WHITE, ACCENT_YELLOW
)


class LongTimeLag(InteractiveScene):
    
    def construct(self):
        title = Text("Long Time Lag (Exp 5.2)", color=TEXT_WHITE, font_size=40)
        title.to_edge(UP, buff=0.4)
        self.play(Write(title), run_time=1)
        
        subtitle = Text("Store value at start, recall after long delay", color=WEIGHT_GRAY, font_size=20)
        subtitle.next_to(title, DOWN, buff=0.15)
        self.play(FadeIn(subtitle), run_time=0.5)
        
        target_box = RoundedRectangle(
            width=1.2, height=0.8, corner_radius=0.1,
            color=ACCENT_YELLOW, fill_color=ACCENT_YELLOW,
            fill_opacity=0.3, stroke_width=2
        )
        target_box.move_to(LEFT * 4.5 + DOWN * 0.5)
        
        target_label = Text("TARGET", color=ACCENT_YELLOW, font_size=14)
        target_label.next_to(target_box, UP, buff=0.1)
        
        target_value = Text("3.5", color=TEXT_WHITE, font_size=24, weight=BOLD)
        target_value.move_to(target_box.get_center())
        
        self.play(
            FadeIn(target_box),
            FadeIn(target_label),
            FadeIn(target_value),
            run_time=0.8
        )
        
        store_arrow = Arrow(
            target_box.get_bottom(),
            target_box.get_bottom() + DOWN * 0.8,
            color=INPUT_GATE_GREEN,
            stroke_width=2
        )
        store_label = Text("Store", color=INPUT_GATE_GREEN, font_size=12)
        store_label.next_to(store_arrow, RIGHT, buff=0.1)
        
        self.play(
            ShowCreation(store_arrow),
            FadeIn(store_label),
            run_time=0.5
        )
        
        distractors = VGroup()
        num_distractors = 12
        start_x = -3.0
        spacing = 0.5
        
        for i in range(num_distractors):
            d_box = RoundedRectangle(
                width=0.4, height=0.4, corner_radius=0.05,
                color=WEIGHT_GRAY, fill_color=WEIGHT_GRAY,
                fill_opacity=0.2, stroke_width=1
            )
            d_box.move_to([start_x + i * spacing, -1.5, 0])
            
            d_label = Text("d", color=WEIGHT_GRAY, font_size=10)
            d_label.move_to(d_box.get_center())
            
            distractors.add(VGroup(d_box, d_label))
        
        dots = Text("...", color=WEIGHT_GRAY, font_size=20)
        dots.move_to([start_x + num_distractors * spacing - 0.3, -1.5, 0])
        
        self.play(
            LaggedStart(*[FadeIn(d) for d in distractors], lag_ratio=0.1),
            run_time=1.5
        )
        self.play(FadeIn(dots), run_time=0.3)
        
        eod_box = RoundedRectangle(
            width=0.6, height=0.4, corner_radius=0.05,
            color=GRADIENT_RED, fill_color=GRADIENT_RED,
            fill_opacity=0.2, stroke_width=1
        )
        eod_box.move_to(RIGHT * 3.5 + DOWN * 1.5)
        
        eod_label = Text("EOD", color=GRADIENT_RED, font_size=10)
        eod_label.move_to(eod_box.get_center())
        
        self.play(FadeIn(eod_box), FadeIn(eod_label), run_time=0.5)
        
        recall_arrow = Arrow(
            eod_box.get_top(),
            eod_box.get_top() + UP * 0.8,
            color=INPUT_GATE_GREEN,
            stroke_width=2
        )
        recall_label = Text("Recall", color=INPUT_GATE_GREEN, font_size=12)
        recall_label.next_to(recall_arrow, RIGHT, buff=0.1)
        
        output_box = RoundedRectangle(
            width=1.2, height=0.8, corner_radius=0.1,
            color=INPUT_GATE_GREEN, fill_color=INPUT_GATE_GREEN,
            fill_opacity=0.3, stroke_width=2
        )
        output_box.move_to(RIGHT * 3.5 + UP * 0.5)
        
        output_label = Text("Output", color=INPUT_GATE_GREEN, font_size=14)
        output_label.next_to(output_box, UP, buff=0.1)
        
        output_value = Text("3.5", color=TEXT_WHITE, font_size=24, weight=BOLD)
        output_value.move_to(output_box.get_center())
        
        self.play(
            ShowCreation(recall_arrow),
            FadeIn(recall_label),
            run_time=0.5
        )
        
        self.play(
            FadeIn(output_box),
            FadeIn(output_label),
            FadeIn(output_value),
            run_time=0.8
        )
        
        self.wait(0.5)
        
        length_indicator = Text("100-1000 timesteps", color=WEIGHT_GRAY, font_size=14)
        length_indicator.next_to(distractors, DOWN, buff=0.4)
        
        self.play(FadeIn(length_indicator), run_time=0.5)
        
        self.wait(0.5)
        
        takeaway = Text(
            "LSTM stores information across 100-1000 timesteps",
            color=TEXT_WHITE,
            font_size=18
        )
        takeaway.to_edge(DOWN, buff=0.3)
        
        self.play(
            FadeOut(length_indicator),
            run_time=0.3
        )
        self.play(FadeIn(takeaway, shift=UP * 0.2), run_time=0.8)
        
        self.wait(2)
        
        if os.getenv("MANIM_DEV"):
            self.embed()
