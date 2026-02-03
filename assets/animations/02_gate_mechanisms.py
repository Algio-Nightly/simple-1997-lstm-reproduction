"""
Gate Mechanisms Animation

Visualizes the input gate and output gate of the 1997 LSTM.
Shows sigmoid gating, the "scissors" (h_prev.detach()), and 
explains why there's no forget gate in the original paper.

Scene: GateMechanisms
Duration: ~30 seconds

Run: cd assets/animations && manimgl 02_gate_mechanisms.py GateMechanisms -w
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from manimlib import *
from colors import (
    CEC_BLUE, INPUT_GATE_GREEN, OUTPUT_GATE_ORANGE, 
    GRADIENT_RED, WEIGHT_GRAY, TEXT_WHITE, ACCENT_YELLOW
)


class GateMechanisms(InteractiveScene):
    
    def construct(self):
        title = Text("Gate Mechanisms (1997 LSTM)", color=TEXT_WHITE, font_size=44)
        title.to_edge(UP, buff=0.4)
        self.play(Write(title), run_time=1.5)
        self.wait(0.5)
        
        cell_box = RoundedRectangle(
            width=5.5, height=3.0, corner_radius=0.2,
            color=CEC_BLUE, fill_opacity=0.1, stroke_width=2
        )
        cell_box.move_to(UP * 0.3)
        cell_label = Text("LSTM Cell", color=CEC_BLUE, font_size=16)
        cell_label.next_to(cell_box, UP, buff=0.1)
        
        self.play(ShowCreation(cell_box), FadeIn(cell_label), run_time=1)
        
        input_gate = RoundedRectangle(
            width=1.2, height=0.7, corner_radius=0.1,
            color=INPUT_GATE_GREEN, fill_color=INPUT_GATE_GREEN,
            fill_opacity=0.3, stroke_width=2
        )
        input_gate.move_to(cell_box.get_center() + LEFT * 1.5 + UP * 0.6)
        ig_label = Text("Input\nGate", color=INPUT_GATE_GREEN, font_size=12)
        ig_label.move_to(input_gate.get_center())
        
        output_gate = RoundedRectangle(
            width=1.2, height=0.7, corner_radius=0.1,
            color=OUTPUT_GATE_ORANGE, fill_color=OUTPUT_GATE_ORANGE,
            fill_opacity=0.3, stroke_width=2
        )
        output_gate.move_to(cell_box.get_center() + RIGHT * 1.5 + UP * 0.6)
        og_label = Text("Output\nGate", color=OUTPUT_GATE_ORANGE, font_size=12)
        og_label.move_to(output_gate.get_center())
        
        cec_box = RoundedRectangle(
            width=1.4, height=0.7, corner_radius=0.1,
            color=CEC_BLUE, fill_color=CEC_BLUE,
            fill_opacity=0.4, stroke_width=2
        )
        cec_box.move_to(cell_box.get_center() + DOWN * 0.6)
        cec_label = Text("CEC (s_c)", color=TEXT_WHITE, font_size=12)
        cec_label.move_to(cec_box.get_center())
        
        loop_arc = CurvedArrow(
            cec_box.get_right() + UP * 0.25,
            cec_box.get_right() + DOWN * 0.25,
            angle=-TAU * 0.4,
            color=CEC_BLUE,
            stroke_width=2.5
        )
        
        loop_label = Text("1.0", color=CEC_BLUE, font_size=11)
        loop_label.next_to(loop_arc, RIGHT, buff=0.1)
        
        self.play(
            FadeIn(input_gate), FadeIn(ig_label),
            FadeIn(output_gate), FadeIn(og_label),
            FadeIn(cec_box), FadeIn(cec_label),
            ShowCreation(loop_arc), FadeIn(loop_label),
            run_time=1.5
        )
        
        self.wait(0.5)
        
        sigmoid_eq = Tex(R"\sigma(x) \in [0,1]", font_size=28)
        sigmoid_eq.next_to(cell_box, LEFT, buff=0.4)
        sigmoid_eq.shift(UP * 0.8)
        
        self.play(FadeIn(sigmoid_eq), run_time=0.8)
        
        gate_action = Text("Gates multiply signals", color=ACCENT_YELLOW, font_size=16)
        gate_action.next_to(cell_box, RIGHT, buff=0.4)
        
        self.play(FadeIn(gate_action), run_time=0.8)
        self.wait(1)
        
        scissors_title = Text('"The Scissors"', color=GRADIENT_RED, font_size=22)
        scissors_subtitle = Text("Truncated Backprop", color=GRADIENT_RED, font_size=18)
        scissors_group = VGroup(scissors_title, scissors_subtitle).arrange(DOWN, buff=0.1)
        scissors_group.next_to(cell_box, DOWN, buff=0.5)
        
        self.play(FadeIn(scissors_group, shift=UP * 0.2), run_time=0.8)
        
        h_prev_arrow = Arrow(
            cell_box.get_left() + LEFT * 1.2,
            cell_box.get_left() + LEFT * 0.2,
            color=WEIGHT_GRAY, stroke_width=2
        )
        h_prev_label = Text("h_prev", color=WEIGHT_GRAY, font_size=12)
        h_prev_label.next_to(h_prev_arrow, UP, buff=0.05)
        
        scissors_mark = Text("X", color=GRADIENT_RED, font_size=20, weight=BOLD)
        scissors_mark.move_to(h_prev_arrow.get_center())
        
        detach_label = Text(".detach()", color=GRADIENT_RED, font_size=11)
        detach_label.next_to(scissors_mark, DOWN, buff=0.08)
        
        self.play(
            ShowCreation(h_prev_arrow), FadeIn(h_prev_label),
            run_time=0.8
        )
        self.play(
            FadeIn(scissors_mark), FadeIn(detach_label),
            run_time=0.8
        )
        
        complexity_label = Text("O(1) per timestep", color=INPUT_GATE_GREEN, font_size=14)
        complexity_label.next_to(scissors_group, DOWN, buff=0.3)
        
        self.play(FadeIn(complexity_label), run_time=0.5)
        self.wait(1.5)
        
        self.play(
            FadeOut(scissors_group), FadeOut(complexity_label),
            FadeOut(sigmoid_eq), FadeOut(gate_action),
            FadeOut(cell_box), FadeOut(cell_label),
            FadeOut(input_gate), FadeOut(ig_label),
            FadeOut(output_gate), FadeOut(og_label),
            FadeOut(cec_box), FadeOut(cec_label),
            FadeOut(loop_arc), FadeOut(loop_label),
            FadeOut(h_prev_arrow), FadeOut(h_prev_label),
            FadeOut(scissors_mark), FadeOut(detach_label),
            run_time=0.8
        )
        
        endcard_box = RoundedRectangle(
            width=9, height=2.0, corner_radius=0.15,
            color=ACCENT_YELLOW, fill_color=ACCENT_YELLOW,
            fill_opacity=0.15, stroke_width=2
        )
        endcard_box.move_to(DOWN * 0.5)
        
        endcard_title = Text("Why no forget gate?", color=ACCENT_YELLOW, font_size=28)
        endcard_title.move_to(endcard_box.get_center() + UP * 0.5)
        
        endcard_text = Text(
            "1997 LSTM: forgetting is indirect (overwrite via input gate).\n"
            "Explicit forget gate added later (Gers et al., 2000).",
            color=TEXT_WHITE, font_size=18
        )
        endcard_text.move_to(endcard_box.get_center() + DOWN * 0.3)
        
        self.play(
            FadeIn(endcard_box),
            FadeIn(endcard_title),
            run_time=0.8
        )
        self.play(FadeIn(endcard_text), run_time=0.8)
        
        self.wait(3)
        
        if os.getenv("MANIM_DEV"):
            self.embed()
