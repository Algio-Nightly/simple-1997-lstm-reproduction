"""
BPTT Complexity Comparison Animation

Compares Full BPTT O(T²) vs Truncated BPTT O(T) complexity.
Shows how the "scissors" (gradient truncation through gates) reduces
computational complexity from quadratic to linear.

Key insight: In LSTM, the CEC is INTERNAL to each cell (s_c self-loop),
while the inter-cell connections (h) are what get truncated.

Scene: BPTTComplexity
Duration: ~30 seconds

Run: cd assets/animations && manimgl 07_bptt_complexity.py BPTTComplexity -w
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from manimlib import *
from colors import (
    CEC_BLUE, GRADIENT_RED, WEIGHT_GRAY, TEXT_WHITE, 
    ACCENT_YELLOW, INPUT_GATE_GREEN
)


class BPTTComplexity(InteractiveScene):
    
    def construct(self):
        title = Text(
            "BPTT Complexity: Full vs Truncated",
            color=TEXT_WHITE,
            font_size=44
        )
        title.to_edge(UP, buff=0.4)
        
        self.play(Write(title), run_time=1.5)
        self.wait(0.5)
        
        part1_label = Text("Full BPTT (Vanilla RNN)", color=ACCENT_YELLOW, font_size=28)
        part1_label.next_to(title, DOWN, buff=0.3)
        
        self.play(FadeIn(part1_label), run_time=0.5)
        
        num_cells = 6
        cell_width = 0.9
        cell_height = 0.6
        spacing = 1.3
        start_x = -3.2
        cell_y = -0.3
        
        cells = VGroup()
        cell_labels = VGroup()
        
        for i in range(num_cells):
            cell = RoundedRectangle(
                width=cell_width,
                height=cell_height,
                corner_radius=0.1,
                color=CEC_BLUE,
                fill_color=CEC_BLUE,
                fill_opacity=0.3,
                stroke_width=2
            )
            cell.move_to([start_x + i * spacing, cell_y, 0])
            cells.add(cell)
            
            label = Text("RNN\nCell", color=TEXT_WHITE, font_size=12)
            label.move_to(cell.get_center())
            cell_labels.add(label)
        
        h_arrows = VGroup()
        for i in range(num_cells - 1):
            arrow = Arrow(
                cells[i].get_right(),
                cells[i + 1].get_left(),
                color=WEIGHT_GRAY,
                stroke_width=2,
                buff=0.05,
                max_tip_length_to_length_ratio=0.15
            )
            h_arrows.add(arrow)
        
        init_label = Text("h(0)", color=TEXT_WHITE, font_size=16)
        init_label.next_to(cells[0], LEFT, buff=0.3)
        init_arrow = Arrow(
            init_label.get_right(),
            cells[0].get_left(),
            color=WEIGHT_GRAY,
            stroke_width=2,
            buff=0.08
        )
        
        final_label = Text("h(t)", color=TEXT_WHITE, font_size=16)
        final_label.next_to(cells[-1], RIGHT, buff=0.3)
        final_arrow = Arrow(
            cells[-1].get_right(),
            final_label.get_left(),
            color=WEIGHT_GRAY,
            stroke_width=2,
            buff=0.08
        )
        
        error_nodes = VGroup()
        error_labels = VGroup()
        error_y = cell_y + 1.0
        
        for i in range(num_cells):
            error = RoundedRectangle(
                width=0.6,
                height=0.35,
                corner_radius=0.08,
                color=GRADIENT_RED,
                fill_color=GRADIENT_RED,
                fill_opacity=0.2,
                stroke_width=2
            )
            error.move_to([start_x + i * spacing, error_y, 0])
            error_nodes.add(error)
            
            label = Text("Error", color=GRADIENT_RED, font_size=10)
            label.move_to(error.get_center())
            error_labels.add(label)
        
        v_arrows = VGroup()
        for i in range(num_cells):
            arrow = Arrow(
                cells[i].get_top(),
                error_nodes[i].get_bottom(),
                color=WEIGHT_GRAY,
                stroke_width=1.5,
                buff=0.05,
                max_tip_length_to_length_ratio=0.2
            )
            v_arrows.add(arrow)
        
        inputs_label = Text("Inputs x(t)", color=TEXT_WHITE, font_size=14)
        inputs_label.next_to(VGroup(*cells), DOWN, buff=0.6)
        
        self.play(
            *[FadeIn(cell) for cell in cells],
            *[FadeIn(label) for label in cell_labels],
            run_time=1
        )
        
        self.play(
            *[ShowCreation(arrow) for arrow in h_arrows],
            FadeIn(init_label), ShowCreation(init_arrow),
            FadeIn(final_label), ShowCreation(final_arrow),
            run_time=1
        )
        
        self.play(
            *[FadeIn(node) for node in error_nodes],
            *[FadeIn(label) for label in error_labels],
            *[ShowCreation(arrow) for arrow in v_arrows],
            FadeIn(inputs_label),
            run_time=1
        )
        
        self.wait(0.5)
        
        backprop_arrows = VGroup()
        
        for i in range(num_cells):
            for j in range(i + 1):
                start = error_nodes[i].get_bottom() + DOWN * 0.05
                end = cells[j].get_top() + UP * 0.05
                
                if i != j:
                    start = error_nodes[i].get_bottom() + LEFT * 0.08
                    end = cells[j].get_top() + RIGHT * 0.08
                
                arrow = Arrow(
                    start, end,
                    color=GRADIENT_RED,
                    stroke_width=1.5 if i == j else 1.0,
                    buff=0.02,
                    max_tip_length_to_length_ratio=0.1
                )
                backprop_arrows.add(arrow)
        
        self.play(
            LaggedStart(
                *[ShowCreation(arrow) for arrow in backprop_arrows],
                lag_ratio=0.02
            ),
            run_time=2
        )
        
        self.wait(0.5)
        
        complexity_full = Tex(R"O(T^2)", color=GRADIENT_RED, font_size=36)
        complexity_full.next_to(error_nodes, RIGHT, buff=0.6)
        complexity_full.shift(DOWN * 0.2)
        
        complexity_box = SurroundingRectangle(
            complexity_full,
            color=GRADIENT_RED,
            buff=0.12,
            stroke_width=2
        )
        
        self.play(
            FadeIn(complexity_full),
            ShowCreation(complexity_box),
            run_time=0.8
        )
        
        self.wait(1)
        
        scissors_text = Text(
            '"The Scissors" — h_prev.detach() truncates gate gradients',
            color=ACCENT_YELLOW,
            font_size=18
        )
        scissors_text.to_edge(DOWN, buff=0.25)
        
        self.play(
            FadeIn(scissors_text, shift=UP * 0.2),
            run_time=0.8
        )
        
        self.wait(0.5)
        
        diagonal_arrows = VGroup(*[
            arrow for idx, arrow in enumerate(backprop_arrows)
            if not self._is_vertical_arrow(idx, num_cells)
        ])
        
        part2_label = Text("Truncated BPTT (1997 LSTM)", color=INPUT_GATE_GREEN, font_size=28)
        part2_label.move_to(part1_label.get_center())
        
        self.play(
            FadeOut(diagonal_arrows, scale=0.5),
            FadeOut(complexity_full),
            FadeOut(complexity_box),
            ReplacementTransform(part1_label, part2_label),
            run_time=1.5
        )
        
        new_cell_labels = VGroup()
        for i, cell in enumerate(cells):
            new_label = Text("LSTM\nCell", color=TEXT_WHITE, font_size=11)
            new_label.move_to(cell.get_center())
            new_cell_labels.add(new_label)
        
        self.play(
            *[ReplacementTransform(old, new) for old, new in zip(cell_labels, new_cell_labels)],
            run_time=0.8
        )
        
        cec_loops = VGroup()
        for i, cell in enumerate(cells):
            loop_arc = Arc(
                start_angle=-PI * 0.4,
                angle=-PI * 1.2,
                radius=0.18,
                color=CEC_BLUE,
                stroke_width=2.5
            )
            loop_arc.next_to(cell, RIGHT, buff=-0.15)
            loop_arc.shift(UP * 0.02)
            
            loop_tip = Triangle(
                color=CEC_BLUE,
                fill_color=CEC_BLUE,
                fill_opacity=1
            )
            loop_tip.scale(0.05)
            loop_tip.rotate(PI * 0.6)
            loop_tip.move_to(loop_arc.get_end())
            
            cec_loops.add(VGroup(loop_arc, loop_tip))
        
        cec_label = Text("CEC (s_c): 1.0 self-loop inside each cell", color=CEC_BLUE, font_size=14)
        cec_label.next_to(inputs_label, DOWN, buff=0.4)
        
        self.play(
            *[ShowCreation(loop) for loop in cec_loops],
            FadeIn(cec_label),
            run_time=1.5
        )
        
        for arrow in h_arrows:
            arrow.set_color(WEIGHT_GRAY)
            arrow.set_stroke(opacity=0.3)
        
        h_cut_label = Text("h path: truncated (detached)", color=WEIGHT_GRAY, font_size=12)
        h_cut_label.next_to(h_arrows[2], DOWN, buff=0.15)
        
        self.play(
            *[arrow.animate.set_stroke(opacity=0.3) for arrow in h_arrows],
            FadeIn(h_cut_label),
            run_time=0.8
        )
        
        complexity_trunc = Tex(R"O(T)", color=INPUT_GATE_GREEN, font_size=36)
        complexity_trunc.next_to(error_nodes, RIGHT, buff=0.6)
        complexity_trunc.shift(DOWN * 0.2)
        
        complexity_box_trunc = SurroundingRectangle(
            complexity_trunc,
            color=INPUT_GATE_GREEN,
            buff=0.12,
            stroke_width=2
        )
        
        self.play(
            FadeIn(complexity_trunc),
            ShowCreation(complexity_box_trunc),
            run_time=0.8
        )
        
        self.wait(1)
        
        self.play(
            FadeOut(scissors_text),
            run_time=0.3
        )
        
        takeaway = Text(
            "Scissors cut h-path gradients → O(T). CEC inside cells preserves error flow.",
            color=TEXT_WHITE,
            font_size=18
        )
        takeaway.to_edge(DOWN, buff=0.3)
        
        self.play(
            FadeIn(takeaway, shift=UP * 0.2),
            run_time=1
        )
        
        self.wait(2.5)
        
        if os.getenv("MANIM_DEV"):
            self.embed()
    
    def _is_vertical_arrow(self, idx: int, num_cells: int) -> bool:
        current_idx = 0
        for error_i in range(num_cells):
            for cell_j in range(error_i + 1):
                if current_idx == idx:
                    return error_i == cell_j
                current_idx += 1
        return False
