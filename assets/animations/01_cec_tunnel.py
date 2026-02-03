"""
CEC / "The 1.0 Tunnel" Animation

Visualizes the Constant Error Carousel (CEC) concept from the 1997 LSTM paper.
The key insight is that the cell state update has a coefficient of 1.0 for the
previous state, creating a "tunnel" through which gradients flow without vanishing.

Scene: CECTunnel
Duration: ~12 seconds

Run: cd assets/animations && manimgl 01_cec_tunnel.py CECTunnel -w -i -l --fps 24
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from manimlib import *
from colors import (
    CEC_BLUE, INPUT_GATE_GREEN, CELL_INPUT_TEAL,
    GRADIENT_RED, TEXT_WHITE, ACCENT_YELLOW
)


class CECTunnel(InteractiveScene):
    """
    Animation showing the Constant Error Carousel (CEC) concept.
    
    The CEC is the core innovation of the 1997 LSTM that solves the
    vanishing gradient problem through a fixed 1.0 self-connection.
    """
    
    def construct(self):
        # ====================================================================
        # Title
        # ====================================================================
        title = Text(
            "The Constant Error Carousel (CEC)",
            color=TEXT_WHITE,
            font_size=48
        )
        title.to_edge(UP, buff=0.5)
        
        self.play(Write(title), run_time=1.5)
        self.wait(0.5)
        
        # ====================================================================
        # Main Equation: Cell State Update
        # ====================================================================
        equation = Tex(
            R"s_c(t) = s_c(t-1) + y_{in}(t) \cdot g(net_c(t))",
            font_size=42,
            t2c={
                "s_c": CEC_BLUE,
                "y_{in}": INPUT_GATE_GREEN,
                "g": CELL_INPUT_TEAL,
            }
        )
        equation.shift(UP * 0.5)
        
        self.play(Write(equation), run_time=2)
        self.wait(1)
        
        # ====================================================================
        # Highlight the 1.0 coefficient (implicit in the + sign)
        # ====================================================================
        # Create annotation pointing to the s_c(t-1) term
        highlight_box = SurroundingRectangle(
            equation[0][5:13],  # s_c(t-1)
            color=ACCENT_YELLOW,
            buff=0.1
        )
        
        coefficient_label = Tex(
            R"\text{Coefficient} = 1.0",
            font_size=32,
            color=ACCENT_YELLOW
        )
        coefficient_label.next_to(highlight_box, DOWN, buff=0.3)
        
        self.play(
            ShowCreation(highlight_box),
            FadeIn(coefficient_label, shift=UP * 0.2),
            run_time=1.5
        )
        self.wait(1)
        
        # ====================================================================
        # Fade out coefficient annotation before showing tunnel
        # ====================================================================
        self.play(
            FadeOut(highlight_box),
            FadeOut(coefficient_label),
            run_time=0.5
        )
        
        # ====================================================================
        # Visual: The "1.0 Tunnel" representation
        # ====================================================================
        tunnel = Rectangle(
            width=4.0,
            height=0.6,
            color=CEC_BLUE,
            fill_color=CEC_BLUE,
            fill_opacity=0.3,
            stroke_width=3
        )
        tunnel.shift(DOWN * 1.0)
        
        tunnel_label = Text("The 1.0 Tunnel", color=CEC_BLUE, font_size=28)
        tunnel_label.next_to(tunnel, UP, buff=0.2)
        
        # Time labels
        t_minus_1 = Tex(R"t-1", color=TEXT_WHITE, font_size=28)
        t_minus_1.next_to(tunnel, LEFT, buff=0.3)
        
        t_label = Tex(R"t", color=TEXT_WHITE, font_size=28)
        t_label.next_to(tunnel, RIGHT, buff=0.3)
        
        self.play(
            FadeIn(tunnel),
            FadeIn(tunnel_label),
            Write(t_minus_1),
            Write(t_label),
            run_time=1.5
        )
        self.wait(0.5)
        
        # ====================================================================
        # Gradient Flow Animation
        # ====================================================================
        # Create gradient arrow flowing through the tunnel (backward direction)
        gradient_arrow = Arrow(
            tunnel.get_right() + RIGHT * 0.3,
            tunnel.get_left() + LEFT * 0.3,
            color=GRADIENT_RED,
            stroke_width=4,
            buff=0
        )
        
        gradient_label = Tex(
            R"\nabla",
            color=GRADIENT_RED,
            font_size=36
        )
        gradient_label.next_to(gradient_arrow, UP, buff=0.1)
        
        self.play(
            ShowCreation(gradient_arrow),
            FadeIn(gradient_label),
            run_time=1.5
        )
        self.wait(0.5)
        
        # ====================================================================
        # Derivative Equation: Gradient Preservation
        # ====================================================================
        derivative_eq = Tex(
            R"\frac{\partial s_c(t)}{\partial s_c(t-1)} = 1.0^k = 1.0",
            font_size=32,
            t2c={
                "s_c": CEC_BLUE,
                "1.0": ACCENT_YELLOW,
            }
        )
        derivative_eq.next_to(tunnel, DOWN, buff=0.5)
        
        self.play(Write(derivative_eq), run_time=1.5)
        self.wait(1)
        
        # ====================================================================
        # Takeaway message
        # ====================================================================
        takeaway = Text(
            "Gradient flows through the 1.0 tunnel without vanishing",
            color=TEXT_WHITE,
            font_size=24
        )
        takeaway.next_to(derivative_eq, DOWN, buff=0.3)
        
        self.play(
            FadeIn(takeaway, shift=UP * 0.3),
            run_time=1
        )
        self.wait(2)
        
        # Optional: Interactive mode for development
        if os.getenv("MANIM_DEV"):
            self.embed()
