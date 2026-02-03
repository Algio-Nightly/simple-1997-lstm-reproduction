"""
RNN Unrolling Animation

Shows transformation from folded RNN (single cell with recurrent loop)
to unrolled sequence (multiple timesteps with shared weights).

Scene: RNNUnrolling
Duration: ~15 seconds

Run: cd assets/animations && manimgl 04_rnn_unrolling.py RNNUnrolling -w
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from manimlib import *
from colors import CEC_BLUE, WEIGHT_GRAY, TEXT_WHITE, ACCENT_YELLOW


class RNNUnrolling(InteractiveScene):
    
    def construct(self):
        title = Text(
            "RNN Unrolling Through Time",
            color=TEXT_WHITE,
            font_size=48
        )
        title.to_edge(UP, buff=0.5)
        
        self.play(Write(title), run_time=1.5)
        self.wait(0.5)
        
        folded_cell = RoundedRectangle(
            width=1.5,
            height=1.0,
            corner_radius=0.2,
            color=CEC_BLUE,
            fill_color=CEC_BLUE,
            fill_opacity=0.3,
            stroke_width=3
        )
        folded_cell.shift(UP * 0.5)
        
        cell_label = Text("RNN", color=TEXT_WHITE, font_size=24)
        cell_label.move_to(folded_cell.get_center())
        
        loop_arrow = CurvedArrow(
            folded_cell.get_right() + UP * 0.3,
            folded_cell.get_right() + DOWN * 0.3,
            angle=-TAU * 0.6,
            color=WEIGHT_GRAY,
            stroke_width=3
        )
        loop_arrow.shift(RIGHT * 0.3)
        
        w_label = Text("W", color=WEIGHT_GRAY, font_size=20)
        w_label.next_to(loop_arrow, RIGHT, buff=0.1)
        
        folded_group = VGroup(folded_cell, cell_label, loop_arrow, w_label)
        
        self.play(
            FadeIn(folded_cell),
            FadeIn(cell_label),
            ShowCreation(loop_arrow),
            FadeIn(w_label),
            run_time=1.5
        )
        self.wait(1)
        
        folded_label = Text("Folded View", color=ACCENT_YELLOW, font_size=24)
        folded_label.next_to(folded_group, DOWN, buff=0.5)
        self.play(FadeIn(folded_label), run_time=0.5)
        self.wait(0.5)
        
        self.play(FadeOut(folded_label), run_time=0.3)
        
        timesteps = ["t-2", "t-1", "t", "t+1", "t+2"]
        cell_width = 1.0
        cell_height = 0.8
        spacing = 1.5
        
        unrolled_cells = VGroup()
        cell_labels = VGroup()
        time_labels = VGroup()
        
        start_x = -3.0
        
        for i, ts in enumerate(timesteps):
            cell = RoundedRectangle(
                width=cell_width,
                height=cell_height,
                corner_radius=0.15,
                color=CEC_BLUE,
                fill_color=CEC_BLUE,
                fill_opacity=0.3,
                stroke_width=2
            )
            cell.move_to([start_x + i * spacing, 0.5, 0])
            
            label = Text("RNN", color=TEXT_WHITE, font_size=16)
            label.move_to(cell.get_center())
            
            time_label = Tex(ts, color=TEXT_WHITE, font_size=20)
            time_label.next_to(cell, DOWN, buff=0.2)
            
            unrolled_cells.add(cell)
            cell_labels.add(label)
            time_labels.add(time_label)
        
        connection_arrows = VGroup()
        w_labels = VGroup()
        
        for i in range(len(timesteps) - 1):
            arrow = Arrow(
                unrolled_cells[i].get_right(),
                unrolled_cells[i + 1].get_left(),
                color=WEIGHT_GRAY,
                stroke_width=2,
                buff=0.1
            )
            connection_arrows.add(arrow)
            
            w_text = Text("W", color=WEIGHT_GRAY, font_size=14)
            w_text.next_to(arrow, UP, buff=0.05)
            w_labels.add(w_text)
        
        unrolled_group = VGroup(unrolled_cells, cell_labels, time_labels, connection_arrows, w_labels)
        
        self.play(
            ReplacementTransform(folded_cell.copy(), unrolled_cells),
            FadeOut(folded_group),
            run_time=2
        )
        
        self.play(
            *[FadeIn(label) for label in cell_labels],
            *[FadeIn(label) for label in time_labels],
            run_time=1
        )
        
        self.play(
            *[ShowCreation(arrow) for arrow in connection_arrows],
            *[FadeIn(label) for label in w_labels],
            run_time=1.5
        )
        self.wait(0.5)
        
        shared_box = SurroundingRectangle(
            w_labels,
            color=ACCENT_YELLOW,
            buff=0.2,
            stroke_width=2
        )
        shared_label = Text("Same weights!", color=ACCENT_YELLOW, font_size=20)
        shared_label.next_to(shared_box, UP, buff=0.2)
        
        self.play(
            ShowCreation(shared_box),
            FadeIn(shared_label),
            run_time=1
        )
        self.wait(1)
        
        self.play(
            FadeOut(shared_box),
            FadeOut(shared_label),
            run_time=0.5
        )
        
        takeaway = Text(
            "Same weights applied at every timestep — enables BPTT",
            color=TEXT_WHITE,
            font_size=24
        )
        takeaway.to_edge(DOWN, buff=0.5)
        
        self.play(FadeIn(takeaway, shift=UP * 0.3), run_time=1)
        self.wait(2)
        
        if os.getenv("MANIM_DEV"):
            self.embed()
