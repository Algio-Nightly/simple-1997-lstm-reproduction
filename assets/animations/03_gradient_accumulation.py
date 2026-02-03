"""
Gradient Accumulation Animation

Visualizes how gradients accumulate in the LSTM's hidden state across time.
Shows individual components (y_in, y_out, g, h, s_c) as colored orbs flowing
into a growing gradient accumulator bar.

Scene: GradientAccumulation
Duration: ~25 seconds

Run: cd assets/animations && manimgl 03_gradient_accumulation.py GradientAccumulation -w
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from manimlib import *
from colors import (
    CEC_BLUE, INPUT_GATE_GREEN, OUTPUT_GATE_ORANGE,
    CELL_INPUT_TEAL, GRADIENT_RED, TEXT_WHITE, ACCENT_YELLOW
)


class GradientAccumulation(InteractiveScene):
    """
    Animation showing gradient accumulation across LSTM cells during backprop.
    """

    def construct(self):
        # ====================================================================
        # Title
        # ====================================================================
        title = Text(
            "Gradient Accumulation in LSTM",
            color=TEXT_WHITE,
            font_size=44
        )
        title.to_edge(UP, buff=0.5)

        self.play(Write(title), run_time=1.5)
        self.wait(0.5)

        # ====================================================================
        # Create 3 LSTM cells horizontally
        # Cell positions: t-2 (left), t-1 (center), t (right)
        # ====================================================================
        cell_width = 1.8
        cell_height = 2.5
        spacing = 2.5
        start_x = -2.5
        cell_y = 0.3

        cells = []
        time_labels = []

        timesteps = ["t-2", "t-1", "t"]

        for i, ts in enumerate(timesteps):
            # Create cell box
            cell = RoundedRectangle(
                width=cell_width,
                height=cell_height,
                corner_radius=0.15,
                color=CEC_BLUE,
                fill_color=CEC_BLUE,
                fill_opacity=0.2,
                stroke_width=2
            )
            x_pos = start_x + i * spacing
            cell.move_to([x_pos, cell_y, 0])
            cells.append(cell)

            # Time label above cell
            time_label = Tex(ts, color=TEXT_WHITE, font_size=24)
            time_label.next_to(cell, UP, buff=0.2)
            time_labels.append(time_label)

            self.play(FadeIn(cell), Write(time_label), run_time=0.8)

        self.wait(0.5)

        # ====================================================================
        # Component labels inside each cell
        # ====================================================================
        components = [
            ("y_{in}", INPUT_GATE_GREEN),
            ("y_{out}", OUTPUT_GATE_ORANGE),
            ("g", CELL_INPUT_TEAL),
            ("h", CEC_BLUE),
            ("s_c", CEC_BLUE)
        ]

        component_labels = [[] for _ in range(3)]

        for cell_idx, cell in enumerate(cells):
            for comp_idx, (comp_name, color) in enumerate(components):
                label = Tex(
                    comp_name,
                    color=color,
                    font_size=20
                )
                # Position vertically within cell
                y_offset = 0.8 - comp_idx * 0.4
                label.move_to([cell.get_center()[0], cell_y + y_offset, 0])
                component_labels[cell_idx].append(label)
                self.play(FadeIn(label), run_time=0.3)

        self.wait(0.5)

        # ====================================================================
        # Create gradient accumulator bar at bottom
        # ====================================================================
        accumulator = Rectangle(
            width=8.0,
            height=0.5,
            color=GRADIENT_RED,
            fill_color=GRADIENT_RED,
            fill_opacity=0.3,
            stroke_width=3
        )
        accumulator.move_to([0, -2.2, 0])

        acc_label = Text("Gradient Accumulator", color=GRADIENT_RED, font_size=24)
        acc_label.next_to(accumulator, DOWN, buff=0.2)

        self.play(
            FadeIn(accumulator),
            FadeIn(acc_label),
            run_time=1
        )
        self.wait(0.5)

        # ====================================================================
        # Animate backprop flow: right to left (t -> t-1 -> t-2)
        # For each cell, animate orbs from each component to accumulator
        # ====================================================================
        accumulator_growth = 0

        for cell_idx in range(2, -1, -1):  # t, t-1, t-2
            cell = cells[cell_idx]

            for comp_idx, (comp_name, color) in enumerate(components):
                label = component_labels[cell_idx][comp_idx]

                # Create orb at component position
                orb = Circle(
                    radius=0.15,
                    color=color,
                    fill_color=color,
                    fill_opacity=0.9,
                    stroke_width=2
                )
                orb.move_to(label.get_center())

                # Target position on accumulator
                target_y = -2.2
                target_x = -3.5 + accumulator_growth * 0.3

                self.play(FadeIn(orb), run_time=0.3)

                # Animate orb flowing down to accumulator
                self.play(
                    orb.animate.move_to([target_x, target_y, 0]),
                    run_time=0.8
                )

                # Grow accumulator
                accumulator_growth += 1
                new_width = 8.0 + accumulator_growth * 0.15
                new_acc = Rectangle(
                    width=new_width,
                    height=0.5,
                    color=GRADIENT_RED,
                    fill_color=GRADIENT_RED,
                    fill_opacity=0.3,
                    stroke_width=3
                )
                new_acc.move_to([0, -2.2, 0])

                self.play(
                    Transform(accumulator, new_acc),
                    FadeOut(orb),
                    run_time=0.3
                )

        self.wait(0.5)

        # ====================================================================
        # Takeaway message
        # ====================================================================
        takeaway = Text(
            "Gradient accumulates without vanishing through the CEC",
            color=TEXT_WHITE,
            font_size=24
        )
        takeaway.to_edge(DOWN, buff=0.3)

        self.play(FadeIn(takeaway, shift=UP * 0.3), run_time=1)
        self.wait(2)

        # Optional: Interactive mode for development
        if os.getenv("MANIM_DEV"):
            self.embed()
