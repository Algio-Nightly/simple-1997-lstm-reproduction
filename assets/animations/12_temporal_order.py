"""
Temporal Order Problem Animation - Experiment 6 (Section 5.6)

Visualizes the Temporal Order task where LSTM must classify
sequences based on the order of special symbols X and Y.

Scene: TemporalOrder
Duration: ~20 seconds
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from manimlib import *
from colors import (
    CEC_BLUE, INPUT_GATE_GREEN, OUTPUT_GATE_ORANGE, WEIGHT_GRAY, TEXT_WHITE, ACCENT_YELLOW
)


class TemporalOrder(InteractiveScene):
    
    def construct(self):
        title = Text("Temporal Order Problem (Exp 5.6)", color=TEXT_WHITE, font_size=40)
        title.to_edge(UP, buff=0.4)
        self.play(Write(title), run_time=1)
        
        subtitle = Text("Classify based on order of key symbols", color=WEIGHT_GRAY, font_size=20)
        subtitle.next_to(title, DOWN, buff=0.15)
        self.play(FadeIn(subtitle), run_time=0.5)
        
        sequence_y = -0.5
        start_x = -4.5
        
        b_box = RoundedRectangle(
            width=0.5, height=0.5, corner_radius=0.08,
            color=WEIGHT_GRAY, fill_color=WEIGHT_GRAY,
            fill_opacity=0.2, stroke_width=1
        )
        b_box.move_to([start_x, sequence_y, 0])
        
        b_label = Text("B", color=TEXT_WHITE, font_size=14)
        b_label.move_to(b_box.get_center())
        
        self.play(FadeIn(b_box), FadeIn(b_label), run_time=0.3)
        
        distractors1 = VGroup()
        for i in range(2):
            d = RoundedRectangle(
                width=0.4, height=0.4, corner_radius=0.05,
                color=WEIGHT_GRAY, fill_color=WEIGHT_GRAY,
                fill_opacity=0.15, stroke_width=1
            )
            d.move_to([start_x + 1.0 + i * 0.6, sequence_y, 0])
            
            dl = Text("d", color=WEIGHT_GRAY, font_size=10)
            dl.move_to(d.get_center())
            
            distractors1.add(VGroup(d, dl))
        
        self.play(*[FadeIn(d) for d in distractors1], run_time=0.4)
        
        x_box = RoundedRectangle(
            width=0.5, height=0.5, corner_radius=0.08,
            color=ACCENT_YELLOW, fill_color=ACCENT_YELLOW,
            fill_opacity=0.4, stroke_width=2
        )
        x_box.move_to([start_x + 2.7, sequence_y, 0])
        
        x_label = Text("X", color=TEXT_WHITE, font_size=16, weight=BOLD)
        x_label.move_to(x_box.get_center())
        
        x_order = Text("1st", color=INPUT_GATE_GREEN, font_size=10)
        x_order.next_to(x_box, UP, buff=0.08)
        
        self.play(
            FadeIn(x_box), FadeIn(x_label), FadeIn(x_order),
            run_time=0.5
        )
        
        distractors2 = VGroup()
        for i in range(2):
            d = RoundedRectangle(
                width=0.4, height=0.4, corner_radius=0.05,
                color=WEIGHT_GRAY, fill_color=WEIGHT_GRAY,
                fill_opacity=0.15, stroke_width=1
            )
            d.move_to([start_x + 3.7 + i * 0.6, sequence_y, 0])
            
            dl = Text("d", color=WEIGHT_GRAY, font_size=10)
            dl.move_to(d.get_center())
            
            distractors2.add(VGroup(d, dl))
        
        self.play(*[FadeIn(d) for d in distractors2], run_time=0.4)
        
        y_box = RoundedRectangle(
            width=0.5, height=0.5, corner_radius=0.08,
            color=ACCENT_YELLOW, fill_color=ACCENT_YELLOW,
            fill_opacity=0.4, stroke_width=2
        )
        y_box.move_to([start_x + 5.4, sequence_y, 0])
        
        y_label = Text("Y", color=TEXT_WHITE, font_size=16, weight=BOLD)
        y_label.move_to(y_box.get_center())
        
        y_order = Text("2nd", color=INPUT_GATE_GREEN, font_size=10)
        y_order.next_to(y_box, UP, buff=0.08)
        
        self.play(
            FadeIn(y_box), FadeIn(y_label), FadeIn(y_order),
            run_time=0.5
        )
        
        distractors3 = VGroup()
        for i in range(2):
            d = RoundedRectangle(
                width=0.4, height=0.4, corner_radius=0.05,
                color=WEIGHT_GRAY, fill_color=WEIGHT_GRAY,
                fill_opacity=0.15, stroke_width=1
            )
            d.move_to([start_x + 6.4 + i * 0.6, sequence_y, 0])
            
            dl = Text("d", color=WEIGHT_GRAY, font_size=10)
            dl.move_to(d.get_center())
            
            distractors3.add(VGroup(d, dl))
        
        self.play(*[FadeIn(d) for d in distractors3], run_time=0.4)
        
        e_box = RoundedRectangle(
            width=0.5, height=0.5, corner_radius=0.08,
            color=WEIGHT_GRAY, fill_color=WEIGHT_GRAY,
            fill_opacity=0.2, stroke_width=1
        )
        e_box.move_to([start_x + 8.3, sequence_y, 0])
        
        e_label = Text("E", color=TEXT_WHITE, font_size=14)
        e_label.move_to(e_box.get_center())
        
        self.play(FadeIn(e_box), FadeIn(e_label), run_time=0.3)
        
        arrow = Arrow(
            [start_x + 8.6, sequence_y, 0],
            [start_x + 9.3, sequence_y, 0],
            color=WEIGHT_GRAY,
            stroke_width=1.5
        )
        
        self.play(ShowCreation(arrow), run_time=0.3)
        
        class_box = RoundedRectangle(
            width=1.2, height=0.7, corner_radius=0.1,
            color=OUTPUT_GATE_ORANGE, fill_color=OUTPUT_GATE_ORANGE,
            fill_opacity=0.3, stroke_width=2
        )
        class_box.move_to([start_x + 10.5, sequence_y, 0])
        
        class_label = Text("Class", color=OUTPUT_GATE_ORANGE, font_size=12)
        class_label.next_to(class_box, UP, buff=0.08)
        
        class_value = Text("XY", color=TEXT_WHITE, font_size=20, weight=BOLD)
        class_value.move_to(class_box.get_center())
        
        self.play(
            FadeIn(class_box),
            FadeIn(class_label),
            FadeIn(class_value),
            run_time=0.6
        )
        
        self.wait(0.5)
        
        classes_text = Text(
            "Classes: XX, XY, YX, YY",
            color=WEIGHT_GRAY,
            font_size=14
        )
        classes_text.next_to(sequence_y * UP, DOWN, buff=0.8)
        
        self.play(FadeIn(classes_text), run_time=0.5)
        
        self.wait(0.5)
        
        takeaway = Text(
            "LSTM classifies based on temporal order of key symbols",
            color=TEXT_WHITE,
            font_size=18
        )
        takeaway.to_edge(DOWN, buff=0.3)
        
        self.play(
            FadeOut(classes_text),
            run_time=0.3
        )
        self.play(FadeIn(takeaway, shift=UP * 0.2), run_time=0.8)
        
        self.wait(2)
        
        if os.getenv("MANIM_DEV"):
            self.embed()
