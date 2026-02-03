"""
Multiplication Problem Animation

Visualizes the Multiplication Problem experiment (Section 5.5 of the 1997 paper).
Shows a sequence with two marked positions and the target formula for multiplication.

Scene: MultiplicationProblem
Duration: ~20 seconds

Run: cd assets/animations && manimgl 11_multiplication.py MultiplicationProblem -w
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from manimlib import *
from colors import (
    CEC_BLUE, ACCENT_YELLOW, TEXT_WHITE, WEIGHT_GRAY, INPUT_GATE_GREEN
)


class MultiplicationProblem(InteractiveScene):
    
    def construct(self):
        title = Text("Multiplication Problem (Experiment 5.5)", color=TEXT_WHITE, font_size=44)
        title.to_edge(UP, buff=0.4)
        self.play(Write(title), run_time=1.5)
        self.wait(0.5)
        
        subtitle = Text(
            "Task: Multiply two marked values from a long sequence",
            color=WEIGHT_GRAY, font_size=20
        )
        subtitle.next_to(title, DOWN, buff=0.2)
        self.play(FadeIn(subtitle), run_time=0.5)
        
        seq_length = 10
        box_size = 0.5
        spacing = 0.65
        start_x = -3.0
        seq_y = 0.5
        
        value_boxes = VGroup()
        marker_boxes = VGroup()
        value_labels = VGroup()
        marker_labels = VGroup()
        
        values = [0.3, 0.7, 0.2, 0.8, 0.5, 0.1, 0.9, 0.4, 0.6, 0.2]
        markers = [0, 0, 0, 1, 0, 0, 0, 1, 0, 0]
        
        for i in range(seq_length):
            value_box = Square(
                side_length=box_size,
                color=CEC_BLUE,
                fill_color=CEC_BLUE,
                fill_opacity=0.2,
                stroke_width=2
            )
            value_box.move_to([start_x + i * spacing, seq_y, 0])
            value_boxes.add(value_box)
            
            val_label = Text(f"{values[i]:.1f}", color=TEXT_WHITE, font_size=14)
            val_label.move_to(value_box.get_center())
            value_labels.add(val_label)
            
            marker_box = Square(
                side_length=box_size,
                color=WEIGHT_GRAY,
                fill_color=WEIGHT_GRAY,
                fill_opacity=0.1,
                stroke_width=1
            )
            marker_box.move_to([start_x + i * spacing, seq_y - 0.7, 0])
            marker_boxes.add(marker_box)
            
            marker_val = markers[i]
            m_label = Text(str(marker_val), color=WEIGHT_GRAY, font_size=14)
            m_label.move_to(marker_box.get_center())
            marker_labels.add(m_label)
        
        value_row_label = Text("Values:", color=TEXT_WHITE, font_size=14)
        value_row_label.next_to(value_boxes[0], LEFT, buff=0.3)
        
        marker_row_label = Text("Markers:", color=TEXT_WHITE, font_size=14)
        marker_row_label.next_to(marker_boxes[0], LEFT, buff=0.3)
        
        self.play(
            *[FadeIn(box) for box in value_boxes],
            *[FadeIn(label) for label in value_labels],
            FadeIn(value_row_label),
            run_time=1
        )
        
        self.play(
            *[FadeIn(box) for box in marker_boxes],
            *[FadeIn(label) for label in marker_labels],
            FadeIn(marker_row_label),
            run_time=1
        )
        
        self.wait(0.5)
        
        marked_indices = [3, 7]
        
        for idx in marked_indices:
            highlight = SurroundingRectangle(
                VGroup(value_boxes[idx], marker_boxes[idx]),
                color=ACCENT_YELLOW,
                buff=0.08,
                stroke_width=3
            )
            self.play(ShowCreation(highlight), run_time=0.5)
            
            marker_labels[idx].set_color(ACCENT_YELLOW)
            marker_boxes[idx].set_fill(ACCENT_YELLOW, opacity=0.3)
        
        self.wait(0.5)
        
        x1_val = values[marked_indices[0]]
        x2_val = values[marked_indices[1]]
        
        x1_label = Text(f"X1 = {x1_val}", color=ACCENT_YELLOW, font_size=20)
        x1_label.next_to(value_boxes[marked_indices[0]], UP, buff=0.4)
        
        x2_label = Text(f"X2 = {x2_val}", color=ACCENT_YELLOW, font_size=20)
        x2_label.next_to(value_boxes[marked_indices[1]], UP, buff=0.4)
        
        self.play(FadeIn(x1_label), FadeIn(x2_label), run_time=0.8)
        self.wait(0.5)
        
        target_value = x1_val * x2_val
        
        formula = Tex(
            R"\text{Target} = X_1 \times X_2",
            font_size=32
        )
        formula.next_to(VGroup(value_boxes, marker_boxes), DOWN, buff=0.8)
        
        result = Text(
            f"= {x1_val} × {x2_val} = {target_value:.2f}",
            color=INPUT_GATE_GREEN, font_size=20
        )
        result.next_to(formula, DOWN, buff=0.2)
        
        self.play(FadeIn(formula), run_time=1)
        self.play(FadeIn(result), run_time=0.8)
        self.wait(1)
        
        takeaway = Text(
            "LSTM remembers marked values to compute product",
            color=TEXT_WHITE, font_size=22
        )
        takeaway.to_edge(DOWN, buff=0.4)
        
        self.play(FadeIn(takeaway, shift=UP * 0.2), run_time=1)
        self.wait(2)
        
        if os.getenv("MANIM_DEV"):
            self.embed()
