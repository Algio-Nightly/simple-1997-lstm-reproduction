"""
Embedded Reber Grammar Animation - Experiment 1 (Section 5.1)

Visualizes the Embedded Reber Grammar task where LSTM must predict
the next valid symbol in a nested finite state grammar.

Scene: ReberGrammar
Duration: ~25 seconds
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from manimlib import *
from colors import (
    CEC_BLUE, INPUT_GATE_GREEN, OUTPUT_GATE_ORANGE,
    GRADIENT_RED, WEIGHT_GRAY, TEXT_WHITE, ACCENT_YELLOW
)


class ReberGrammar(InteractiveScene):
    
    def construct(self):
        title = Text("Embedded Reber Grammar (Exp 5.1)", color=TEXT_WHITE, font_size=40)
        title.to_edge(UP, buff=0.4)
        self.play(Write(title), run_time=1)
        
        subtitle = Text("Predict the next valid symbol in a nested grammar", color=WEIGHT_GRAY, font_size=20)
        subtitle.next_to(title, DOWN, buff=0.15)
        self.play(FadeIn(subtitle), run_time=0.5)
        
        explanation = Text(
            "Reber Grammar: A finite state machine with deterministic paths",
            color=TEXT_WHITE,
            font_size=14
        )
        explanation.next_to(subtitle, DOWN, buff=0.3)
        explanation2 = Text(
            "Two entry choices lead to different valid symbol sequences",
            color=TEXT_WHITE,
            font_size=14
        )
        explanation2.next_to(explanation, DOWN, buff=0.15)
        self.play(FadeIn(explanation), FadeIn(explanation2), run_time=0.8)
        
        self.wait(0.3)
        
        node_radius = 0.25
        node_color = CEC_BLUE
        
        states = {}
        state_labels = {}
        
        positions = {
            'S': (-4.5, 0, 0),
            '1': (-2.5, 1, 0),
            '2': (-2.5, -1, 0),
            '3': (0, 1, 0),
            '4': (0, -1, 0),
            '5': (2.5, 1, 0),
            '6': (2.5, -1, 0),
            'E': (4.5, 0, 0),
        }
        
        for name, pos in positions.items():
            circle = Circle(radius=node_radius, color=node_color, stroke_width=2)
            circle.set_fill(node_color, opacity=0.2)
            circle.move_to(pos)
            states[name] = circle
            
            if name in ['S', 'E']:
                label = Text(name, color=TEXT_WHITE, font_size=16)
            else:
                label = Text(name, color=TEXT_WHITE, font_size=14)
            label.move_to(circle.get_center())
            state_labels[name] = label
        
        transitions = [
            ('S', '1', 'B', (0, 0.3, 0), INPUT_GATE_GREEN),      
            ('S', '2', 'B', (0, -0.3, 0), OUTPUT_GATE_ORANGE),   
            ('1', '3', 'T', (0, 0.2, 0), INPUT_GATE_GREEN),
            ('2', '4', 'P', (0, -0.2, 0), OUTPUT_GATE_ORANGE),
            ('3', '3', 'S', (0, 0.5, 0), INPUT_GATE_GREEN),      
            ('4', '4', 'V', (0, -0.5, 0), OUTPUT_GATE_ORANGE),   
            ('3', '5', 'X', (0, 0.2, 0), INPUT_GATE_GREEN),
            ('4', '6', 'V', (0, -0.2, 0), OUTPUT_GATE_ORANGE),
            ('5', '3', 'S', (0.1, 0.5, 0), INPUT_GATE_GREEN),
            ('6', '4', 'P', (0.1, -0.5, 0), OUTPUT_GATE_ORANGE),
            ('5', 'E', 'T', (0, 0.3, 0), INPUT_GATE_GREEN),
            ('6', 'E', 'P', (0, -0.3, 0), OUTPUT_GATE_ORANGE),
        ]
        
        arrows = VGroup()
        trans_labels = VGroup()
        
        for (start, end, symbol, offset, path_color) in transitions:
            if start == end:
                loop = Arc(
                    start_angle=PI * 0.3,
                    angle=PI * 1.4,
                    radius=0.35,
                    color=path_color,
                    stroke_width=2.5
                )
                loop_pos = positions[start] + np.array(offset)
                loop.move_to(loop_pos)
                arrows.add(loop)
                
                lbl = Text(symbol, color=ACCENT_YELLOW, font_size=12)
                lbl.next_to(loop, UP if offset[1] > 0 else DOWN, buff=0.08)
                trans_labels.add(lbl)
            else:
                start_pos = positions[start] + np.array([0, 0, 0])
                end_pos = positions[end] + np.array([0, 0, 0])
                direction = end_pos - start_pos
                direction = direction / np.linalg.norm(direction)
                
                arrow = Arrow(
                    start_pos + direction * node_radius,
                    end_pos - direction * node_radius,
                    color=path_color,
                    stroke_width=2,
                    buff=0.02,
                    max_tip_length_to_length_ratio=0.15
                )
                arrows.add(arrow)
                
                mid = (start_pos + end_pos) / 2 + np.array(offset) * 0.3
                lbl = Text(symbol, color=ACCENT_YELLOW, font_size=12)
                lbl.move_to(mid)
                trans_labels.add(lbl)
        
        self.play(
            *[FadeIn(s) for s in states.values()],
            *[FadeIn(l) for l in state_labels.values()],
            run_time=1
        )
        
        self.play(
            *[ShowCreation(a) for a in arrows],
            *[FadeIn(l) for l in trans_labels],
            run_time=1.5
        )
        
        self.wait(0.5)
        
        # Example sequence 1 (top path)
        example_seq_1 = ['B', 'T', 'S', 'X', 'S', 'X', 'T', 'E']
        path_states_1 = ['S', '1', '3', '3', '5', '3', '5', 'E']
        
        seq_label_1 = Text(
            "Example 1 (Top): B → T → S → X → S → X → T → E",
            color=INPUT_GATE_GREEN,
            font_size=16
        )
        seq_label_1.to_edge(DOWN, buff=1.0)
        
        self.play(FadeIn(seq_label_1), run_time=0.5)
        
        highlight = Circle(radius=node_radius + 0.08, color=INPUT_GATE_GREEN, stroke_width=3)
        highlight.move_to(positions['S'])
        self.play(ShowCreation(highlight), run_time=0.3)
        
        for i, state in enumerate(path_states_1[1:], 1):
            self.play(
                highlight.animate.move_to(positions[state]),
                run_time=0.3
            )
        
        self.wait(0.3)
        
        # Example sequence 2 (bottom path)
        example_seq_2 = ['B', 'P', 'V', 'V', 'P', 'V', 'P', 'E']
        path_states_2 = ['S', '2', '4', '4', '6', '4', '6', 'E']
        
        seq_label_2 = Text(
            "Example 2 (Bottom): B → P → V → V → P → V → P → E",
            color=OUTPUT_GATE_ORANGE,
            font_size=16
        )
        seq_label_2.next_to(seq_label_1, DOWN, buff=0.3)
        
        self.play(FadeIn(seq_label_2), run_time=0.5)
        
        highlight.set_color(OUTPUT_GATE_ORANGE)
        highlight.move_to(positions['S'])
        
        for i, state in enumerate(path_states_2[1:], 1):
            self.play(
                highlight.animate.move_to(positions[state]),
                run_time=0.3
            )
        
        self.wait(0.5)
        
        key_insight = Text(
            "Key: First symbol (B) choice determines entire valid sequence",
            color=GRADIENT_RED,
            font_size=14
        )
        key_insight.next_to(seq_label_2, DOWN, buff=0.4)
        
        self.play(FadeIn(key_insight, shift=UP * 0.2), run_time=0.8)
        
        self.wait(0.5)
        
        self.play(
            FadeOut(seq_label_1),
            FadeOut(seq_label_2),
            FadeOut(key_insight),
            FadeOut(highlight),
            run_time=0.5
        )
        
        takeaway = Text(
            "LSTM must remember which path was taken and enforce",
            color=TEXT_WHITE,
            font_size=16
        )
        takeaway.to_edge(DOWN, buff=0.6)
        takeaway2 = Text(
            "the corresponding valid endings—a test of long-term memory",
            color=TEXT_WHITE,
            font_size=16
        )
        takeaway2.next_to(takeaway, DOWN, buff=0.15)
        
        self.play(FadeIn(takeaway, shift=UP * 0.2), FadeIn(takeaway2, shift=UP * 0.2), run_time=0.8)
        
        self.wait(2)
        
        if os.getenv("MANIM_DEV"):
            self.embed()
