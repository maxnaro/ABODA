"""Mutable annotation state shared with the OpenCV mouse callback.

OpenCV mouse callbacks only accept a single opaque userdata parameter,
so a shared mutable object is the standard pattern. The module-level
`state` instance is passed explicitly to functions that need it.
"""

from annotator.constants import DEFAULT_RADIUS_PX, DEFAULT_THRESHOLD_SECONDS


class AnnotationState:
    """Per-session and per-video annotation state."""

    def __init__(self, radius=DEFAULT_RADIUS_PX, threshold=DEFAULT_THRESHOLD_SECONDS):
        self.radius = radius
        self.threshold = threshold
        self.mouse_pos = None
        self.reset_video()

    def reset_video(self):
        """Clear per-video fields for the next annotation."""
        self.paused = False
        self.drawing = False
        self.roi_start = None
        self.roi_end = None
        self.roi_committed = None
        self.abandon_frame = None
        self.has_abandonment = False
        self.marked_negative = False
        self.show_hud = True
        self.current_frame = 0
