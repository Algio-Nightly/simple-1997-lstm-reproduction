"""
Two-Sequence Problem Animation - Experiment 3 (Section 5.3)

Visualizes the Two-Sequence task with two parallel input streams,
signal elements followed by a long noise tail.

Scene: TwoSequence
Duration: ~20 seconds
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from manimlib import *
from colors import (
    CEC_BLUE, INPUT_GATE_GREEN, OUTPUT_GATE_ORANGE, WEIGHT_GRAY, TEXT_WHITE, ACCENT_YELLOW
)


class TwoSequence(InteractiveScene):
    
    def construct(self):
        title = Text("Two-Sequence Problem (Exp 5.3)", color=TEXT_WHITE, font_size=40)
        title.to_edge(UP, buff=0.4)
        self.play(Write(title), run_time=1)
        
        subtitle = Text("Classify based on signals before noise tail", color=WEIGHT_GRAY, font_size=20)
        subtitle.next_to(title, DOWN, buff=0.15)
        self.play(FadeIn(subtitle), run_time=0.5)
        
        stream1_y = 0.5
        stream2_y = -0.8
        
        signal1_box = RoundedRectangle(
            width=0.8, height=0.5, corner_radius=0.08,
            color=ACCENT_YELLOW, fill_color=ACCENT_YELLOW,
            fill_opacity=0.3, stroke_width=2
        )
        signal1_box.move_to(LEFT * 4 + stream1_y * UP)
        
        signal1_label = Text("S₁", color=ACCENT_YELLOW, font_size=16, weight=BOLD)
        signal1_label.move_to(signal1_box.get_center())
        
        signal2_box = RoundedRectangle(
            width=0.8, height=0.5, corner_radius=0.08,
            color=ACCENT_YELLOW, fill_color=ACCENT_YELLOW,
            fill_opacity=0.3, stroke_width=2
        )
        signal2_box.move_to(LEFT * 4 + stream2_y * UP)
        
        signal2_label = Text("S₂", color=ACCENT_YELLOW, font_size=16, weight=BOLD)
        signal2_label.move_to(signal2_box.get_center())
        
        stream1_label = Text("Stream 1", color=TEXT_WHITE, font_size=12)
        stream1_label.next_to(signal1_box, LEFT, buff=0.2)
        
        stream2_label = Text("Stream 2", color=TEXT_WHITE, font_size=12)
        stream2_label.next_to(signal2_box, LEFT, buff=0.2)
        
        self.play(
            FadeIn(signal1_box), FadeIn(signal1_label),
            FadeIn(signal2_box), FadeIn(signal2_label),
            FadeIn(stream1_label), FadeIn(stream2_label),
            run_time=0.8
        )
        
        noise_segments1 = VGroup()
        noise_segments2 = VGroup()
        
        num_segments = 8
        start_x = -2.8
        spacing = 0.6
        
        for i in range(num_segments):
            n1 = RoundedRectangle(
                width=0.5, height=0.3, corner_radius=0.05,
                color=WEIGHT_GRAY, fill_color=WEIGHT_GRAY,
                fill_opacity=0.15, stroke_width=1
            )
            n1.move_to([start_x + i * spacing, stream1_y, 0])
            
            wave1 = Text("~", color=WEIGHT_GRAY, font_size=14)
            wave1.move_to(n1.get_center())
            
            noise_segments1.add(VGroup(n1, wave1))
            
            n2 = RoundedRectangle(
                width=0.5, height=0.3, corner_radius=0.05,
                color=WEIGHT_GRAY, fill_color=WEIGHT_GRAY,
                fill_opacity=0.15, stroke_width=1
            )
            n2.move_to([start_x + i * spacing, stream2_y, 0])
            
            wave2 = Text("~", color=WEIGHT_GRAY, font_size=14)
            wave2.move_to(n2.get_center())
            
            noise_segments2.add(VGroup(n2, wave2))
        
        dots1 = Text("...", color=WEIGHT_GRAY, font_size=16)
        dots1.move_to([start_x + num_segments * spacing - 0.2, stream1_y, 0])
        
        dots2 = Text("...", color=WEIGHT_GRAY, font_size=16)
        dots2.move_to([start_x + num_segments * spacing - 0.2, stream2_y, 0])
        
        self.play(
            LaggedStart(
                *[FadeIn(n) for n in noise_segments1],
                lag_ratio=0.08
            ),
            LaggedStart(
                *[FadeIn(n) for n in noise_segments2],
                lag_ratio=0.08
            ),
            run_time=1.5
        )
        
        self.play(
            FadeIn(dots1), FadeIn(dots2),
            run_time=0.3
        )
        
        arrow1 = Arrow(
            noise_segments1[-1].get_right() + RIGHT * 0.3,
            noise_segments1[-1].get_right() + RIGHT * 1.2,
            color=WEIGHT_GRAY,
            stroke_width=1.5
        )
        
        arrow2 = Arrow(
            noise_segments2[-1].get_right() + RIGHT * 0.3,
            noise_segments2[-1].get_right() + RIGHT * 1.2,
            color=WEIGHT_GRAY,
            stroke_width=1.5
        )
        
        self.play(
            ShowCreation(arrow1),
            ShowCreation(arrow2),
            run_time=0.5
        )
        
        class_box = RoundedRectangle(
            width=1.5, height=0.8, corner_radius=0.1,
            color=OUTPUT_GATE_ORANGE, fill_color=OUTPUT_GATE_ORANGE,
            fill_opacity=0.3, stroke_width=2
        )
        class_box.move_to(RIGHT * 3.5)
        
        class_label = Text("Class", color=OUTPUT_GATE_ORANGE, font_size=14)
        class_label.next_to(class_box, UP, buff=0.1)
        
        class_value = Text("A / B", color=TEXT_WHITE, font_size=20, weight=BOLD)
        class_value.move_to(class_box.get_center())
        
        self.play(
            FadeIn(class_box),
            FadeIn(class_label),
            FadeIn(class_value),
            run_time=0.8
        )
        
        self.wait(0.5)
        
        noise_label = Text("Noise tail", color=WEIGHT_GRAY, font_size=14)
        noise_label.next_to(noise_segments1, DOWN, buff=0.4)
        
        self.play(FadeIn(noise_label), run_time=0.5)
        
        self.wait(0.5)
        
        takeaway = Text(
            "LSTM classifies based on values seen before noise tail",
            color=TEXT_WHITE,
            font_size=18
        )
        takeaway.to_edge(DOWN, buff=0.3)
        
        self.play(
            FadeOut(noise_label),
            run_time=0.3
        )
        self.play(FadeIn(takeaway, shift=UP * 0.2), run_time=0.8)
        
        self.wait(2)
        
        if os.getenv("MANIM_DEV"):
            self.embed()
