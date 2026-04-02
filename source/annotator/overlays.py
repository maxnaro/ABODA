"""Drawing functions for radius circle, ROI bounding box, and HUD overlays.

All functions take `state` as an explicit parameter to avoid global coupling.
"""

import cv2

from annotator.constants import (
    BOX_COLOR,
    BOX_THICKNESS,
    HUD_ALPHA,
    HUD_HEIGHT,
    RADIUS_ALPHA,
    RADIUS_COLOR,
)


def draw_radius_circle(frame, state):
    """Draw a translucent ownership-radius circle centred on the ROI
    (if committed) or the mouse cursor."""
    if state.roi_committed:
        x, y, w, h = state.roi_committed
        center = (x + w // 2, y + h // 2)
    elif state.mouse_pos is not None:
        center = state.mouse_pos
    else:
        return
    overlay = frame.copy()
    cv2.circle(overlay, center, state.radius, RADIUS_COLOR, 2, cv2.LINE_AA)
    cv2.circle(overlay, center, state.radius, RADIUS_COLOR, -1, cv2.LINE_AA)
    cv2.addWeighted(overlay, RADIUS_ALPHA, frame, 1 - RADIUS_ALPHA, 0, frame)
    cv2.circle(frame, center, state.radius, RADIUS_COLOR, 1, cv2.LINE_AA)


def draw_roi_preview(frame, state):
    """Draw the in-progress drag rectangle or the committed bounding box."""
    if state.drawing and state.roi_start and state.roi_end:
        cv2.rectangle(frame, state.roi_start, state.roi_end, BOX_COLOR, BOX_THICKNESS)
    elif state.roi_committed:
        x, y, w, h = state.roi_committed
        cv2.rectangle(frame, (x, y), (x + w, y + h), BOX_COLOR, BOX_THICKNESS)
        label = f"ROI: {x},{y} {w}x{h}"
        cv2.putText(
            frame, label, (x, y - 8),
            cv2.FONT_HERSHEY_SIMPLEX, 0.5, BOX_COLOR, 1, cv2.LINE_AA,
        )


def draw_hud(frame, state, video_name, frame_idx, total_frames, fps, video_num, video_total):
    """Render a semi-transparent 5-row HUD bar at the bottom of the frame."""
    h, w = frame.shape[:2]
    overlay = frame.copy()
    y_top = h - HUD_HEIGHT
    cv2.rectangle(overlay, (0, y_top), (w, h), (0, 0, 0), -1)
    cv2.addWeighted(overlay, HUD_ALPHA, frame, 1 - HUD_ALPHA, 0, frame)

    white = (255, 255, 255)
    grey = (160, 160, 160)
    green = (0, 220, 100)
    yellow = (0, 220, 255)
    font = cv2.FONT_HERSHEY_SIMPLEX
    lh = 22

    time_sec = frame_idx / fps if fps > 0 else 0
    mins, secs = divmod(int(time_sec), 60)
    info = (
        f"[{video_num}/{video_total}] {video_name}  |  "
        f"Frame {frame_idx}/{total_frames}  |  {mins:02d}:{secs:02d}"
    )
    cv2.putText(frame, info, (10, y_top + lh), font, 0.50, white, 1, cv2.LINE_AA)

    status_parts = []
    if state.marked_negative:
        status_parts.append(("NEGATIVE SAMPLE", yellow))
    elif state.has_abandonment and state.abandon_frame is not None:
        status_parts.append((f"Abandon frame: {state.abandon_frame}", green))
    if state.roi_committed:
        x, y_, ww, hh = state.roi_committed
        status_parts.append((f"ROI: [{x},{y_},{ww},{hh}]", green))
    if not status_parts:
        status_parts.append(("No annotation yet", grey))

    x_cursor = 10
    for text, color in status_parts:
        cv2.putText(
            frame, text, (x_cursor, y_top + lh * 2), font, 0.45, color, 1, cv2.LINE_AA
        )
        x_cursor += cv2.getTextSize(text, font, 0.45, 1)[0][0] + 20

    mode = "PAUSED" if state.paused else "PLAYING"
    params = (
        f"{mode}  |  Radius: {state.radius}px  |  "
        f"Threshold: {state.threshold}s"
    )
    cv2.putText(
        frame, params, (10, y_top + lh * 3), font, 0.45,
        yellow if state.paused else green, 1, cv2.LINE_AA,
    )

    keys = (
        "SPACE:Play/Pause  a/d:+/-5s  T:+threshold  "
        "Drag:ROI  F:Negative  ENTER:Save+Next  Q:Quit"
    )
    cv2.putText(frame, keys, (10, y_top + lh * 4 + 4), font, 0.38, grey, 1, cv2.LINE_AA)

    adj_keys = "+/-:Radius  [/]:Threshold  H:Toggle HUD"
    cv2.putText(frame, adj_keys, (10, y_top + lh * 5 + 4), font, 0.38, grey, 1, cv2.LINE_AA)
