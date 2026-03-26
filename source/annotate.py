"""
Abandoned Luggage Annotation Tool

Rapidly annotate video datasets for abandoned-luggage detection.
Loads video files from a directory or a single file, lets the user tag the
exact abandonment frame + bounding box, and saves results to ground_truth.json.

Usage:
    python annotate.py [VIDEO_PATH_OR_DIR] [--output ground_truth.json]
"""

import argparse
import json
import sys
from pathlib import Path

import cv2
import numpy as np

SKIP_SECONDS = 5
DEFAULT_THRESHOLD_SECONDS = 15
DEFAULT_RADIUS_PX = 200
WINDOW_NAME = "Abandoned Luggage Annotator"
HUD_HEIGHT = 140
HUD_ALPHA = 0.70
BOX_COLOR = (0, 255, 0)
BOX_THICKNESS = 2
RADIUS_COLOR = (255, 180, 0)
RADIUS_ALPHA = 0.25
TRACKBAR_NAME = "Frame"


class AnnotationState:
    def __init__(self, radius=DEFAULT_RADIUS_PX, threshold=DEFAULT_THRESHOLD_SECONDS):
        self.radius = radius
        self.threshold = threshold
        self.mouse_pos = None
        self.reset_video()

    def reset_video(self):
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


state = AnnotationState()


def mouse_callback(event, x, y, flags, _param):
    if event == cv2.EVENT_MOUSEMOVE:
        state.mouse_pos = (x, y)

    if not state.paused:
        return

    if event == cv2.EVENT_LBUTTONDOWN:
        state.drawing = True
        state.roi_start = (x, y)
        state.roi_end = (x, y)
        state.roi_committed = None

    elif event == cv2.EVENT_MOUSEMOVE and state.drawing:
        state.roi_end = (x, y)

    elif event == cv2.EVENT_LBUTTONUP and state.drawing:
        state.drawing = False
        state.roi_end = (x, y)
        x0 = min(state.roi_start[0], x)
        y0 = min(state.roi_start[1], y)
        w = abs(x - state.roi_start[0])
        h = abs(y - state.roi_start[1])
        if w > 3 and h > 3:
            state.roi_committed = (x0, y0, w, h)
        else:
            state.roi_committed = None


def on_trackbar(_pos):
    pass


def draw_radius_circle(frame):
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


def draw_roi_preview(frame):
    if state.drawing and state.roi_start and state.roi_end:
        cv2.rectangle(frame, state.roi_start, state.roi_end, BOX_COLOR, BOX_THICKNESS)
    elif state.roi_committed:
        x, y, w, h = state.roi_committed
        cv2.rectangle(frame, (x, y), (x + w, y + h), BOX_COLOR, BOX_THICKNESS)
        label = f"ROI: {x},{y} {w}x{h}"
        cv2.putText(
            frame,
            label,
            (x, y - 8),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            BOX_COLOR,
            1,
            cv2.LINE_AA,
        )


def draw_hud(frame, video_name, frame_idx, total_frames, fps, video_num, video_total):
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
        f"{mode}  |  Radius: {state.radius}px  |  " f"Threshold: {state.threshold}s"
    )
    cv2.putText(
        frame,
        params,
        (10, y_top + lh * 3),
        font,
        0.45,
        yellow if state.paused else green,
        1,
        cv2.LINE_AA,
    )

    keys = (
        "SPACE:Play/Pause  a/d:+/-5s  T:+threshold  "
        "Drag:ROI  F:Negative  ENTER:Save+Next  Q:Quit"
    )
    cv2.putText(frame, keys, (10, y_top + lh * 4 + 4), font, 0.38, grey, 1, cv2.LINE_AA)

    adj_keys = "+/-:Radius  [/]:Threshold  H:Toggle HUD"
    cv2.putText(
        frame, adj_keys, (10, y_top + lh * 5 + 4), font, 0.38, grey, 1, cv2.LINE_AA
    )


def annotate_video(video_path, video_num, video_total):
    state.reset_video()
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        print(f"[WARN] Cannot open {video_path}, skipping.")
        return None, False

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    frame_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    cv2.namedWindow(WINDOW_NAME, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(WINDOW_NAME, min(frame_w, 1280), min(frame_h, 800))
    cv2.setMouseCallback(WINDOW_NAME, mouse_callback, {"frame_h": frame_h})
    cv2.createTrackbar(
        TRACKBAR_NAME, WINDOW_NAME, 0, max(total_frames - 1, 1), on_trackbar
    )

    state.current_frame = 0
    last_display = None
    quit_all = False

    while True:
        trackbar_pos = cv2.getTrackbarPos(TRACKBAR_NAME, WINDOW_NAME)
        if trackbar_pos < 0:
            break

        if state.paused and trackbar_pos != state.current_frame:
            state.current_frame = trackbar_pos
            cap.set(cv2.CAP_PROP_POS_FRAMES, state.current_frame)
            last_display = None

        if not state.paused:
            ret, frame = cap.read()
            if not ret:
                state.paused = True
                state.current_frame = max(total_frames - 1, 0)
                cap.set(cv2.CAP_PROP_POS_FRAMES, state.current_frame)
                continue
            state.current_frame = int(cap.get(cv2.CAP_PROP_POS_FRAMES)) - 1
            cv2.setTrackbarPos(TRACKBAR_NAME, WINDOW_NAME, state.current_frame)
            last_display = frame.copy()
        else:
            if last_display is None:
                cap.set(cv2.CAP_PROP_POS_FRAMES, state.current_frame)
                ret, frame = cap.read()
                if ret:
                    last_display = frame.copy()
            if last_display is None:
                cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                ret, frame = cap.read()
                last_display = (
                    frame.copy() if ret else np.zeros((frame_h, frame_w, 3), np.uint8)
                )

        display = last_display.copy()
        draw_radius_circle(display)
        draw_roi_preview(display)
        if state.show_hud:
            draw_hud(
                display,
                video_path.name,
                state.current_frame,
                total_frames,
                fps,
                video_num,
                video_total,
            )
        cv2.imshow(WINDOW_NAME, display)

        wait_ms = 1 if not state.paused else 30
        key = cv2.waitKey(wait_ms) & 0xFF

        if key == ord(" "):
            state.paused = not state.paused
            if not state.paused:
                cap.set(cv2.CAP_PROP_POS_FRAMES, state.current_frame)

        elif key == ord("q") or key == ord("Q"):
            quit_all = True
            break

        elif key == 13:
            if state.roi_committed and not state.marked_negative:
                state.has_abandonment = True
                state.abandon_frame = state.current_frame
            break

        elif key == ord("f") or key == ord("F"):
            state.marked_negative = True
            state.has_abandonment = False
            state.roi_committed = None
            state.abandon_frame = None

        elif key == ord("t") or key == ord("T"):
            jump = int(state.threshold * fps)
            state.current_frame = min(state.current_frame + jump, total_frames - 1)
            cap.set(cv2.CAP_PROP_POS_FRAMES, state.current_frame)
            cv2.setTrackbarPos(TRACKBAR_NAME, WINDOW_NAME, state.current_frame)
            state.paused = True
            last_display = None

        elif key == 81 or key == 2 or key == ord("a"):
            jump = int(SKIP_SECONDS * fps)
            state.current_frame = max(state.current_frame - jump, 0)
            cap.set(cv2.CAP_PROP_POS_FRAMES, state.current_frame)
            cv2.setTrackbarPos(TRACKBAR_NAME, WINDOW_NAME, state.current_frame)
            state.paused = True
            last_display = None

        elif key == 83 or key == 3 or key == ord("d"):
            jump = int(SKIP_SECONDS * fps)
            state.current_frame = min(state.current_frame + jump, total_frames - 1)
            cap.set(cv2.CAP_PROP_POS_FRAMES, state.current_frame)
            cv2.setTrackbarPos(TRACKBAR_NAME, WINDOW_NAME, state.current_frame)
            state.paused = True
            last_display = None

        elif key == ord("+") or key == ord("="):
            state.radius = min(state.radius + 10, 500)

        elif key == ord("-"):
            state.radius = max(state.radius - 10, 10)

        elif key == ord("]"):
            state.threshold = min(state.threshold + 5, 300)

        elif key == ord("["):
            state.threshold = max(state.threshold - 5, 5)

        elif key == ord("h") or key == ord("H"):
            state.show_hud = not state.show_hud

    cap.release()
    cv2.destroyWindow(WINDOW_NAME)

    if state.marked_negative:
        record = {
            "has_abandonment": False,
            "true_abandon_frame": None,
            "bag_roi": [],
            "radius_px": state.radius,
            "threshold_s": state.threshold,
        }
    else:
        record = {
            "has_abandonment": state.has_abandonment,
            "true_abandon_frame": state.abandon_frame,
            "bag_roi": list(state.roi_committed) if state.roi_committed else [],
            "radius_px": state.radius,
            "threshold_s": state.threshold,
        }

    return record, quit_all


def load_annotations(path):
    if path.exists():
        with open(path, "r") as f:
            return json.load(f)
    return {}


def save_annotations(path, data):
    with open(path, "w") as f:
        json.dump(data, f, indent=2)
    print(f"[INFO] Saved annotations to {path}")


def main():
    parser = argparse.ArgumentParser(description="Abandoned Luggage Annotation Tool")
    parser.add_argument(
        "input_path",
        nargs="?",
        default=".",
        help="Path to a single .mp4/.avi video file, or a directory containing video files",
    )
    parser.add_argument(
        "--output",
        "-o",
        default="ground_truth.json",
        help="Output JSON file (default: ground_truth.json in video directory)",
    )
    parser.add_argument(
        "--radius",
        "-r",
        type=int,
        default=DEFAULT_RADIUS_PX,
        help=f"Initial ownership radius in pixels (default: {DEFAULT_RADIUS_PX})",
    )
    parser.add_argument(
        "--threshold",
        "-t",
        type=int,
        default=DEFAULT_THRESHOLD_SECONDS,
        help=f"Initial abandonment threshold in seconds (default: {DEFAULT_THRESHOLD_SECONDS})",
    )
    args = parser.parse_args()

    input_path = Path(args.input_path).resolve()

    if input_path.is_file():
        if input_path.suffix.lower() not in (".mp4", ".avi"):
            print(f"[ERROR] {input_path} is not a supported video file (.mp4 / .avi).")
            sys.exit(1)
        videos = [input_path]
        video_dir = input_path.parent
    elif input_path.is_dir():
        video_dir = input_path
        videos = sorted(
            p for p in video_dir.iterdir() if p.suffix.lower() in (".mp4", ".avi")
        )
        if not videos:
            print(f"[ERROR] No .mp4 or .avi files found in {video_dir}")
            sys.exit(1)
    else:
        print(f"[ERROR] {input_path} does not exist.")
        sys.exit(1)

    output_path = Path(args.output)
    if not output_path.is_absolute():
        output_path = video_dir / output_path

    global state
    state = AnnotationState(radius=args.radius, threshold=args.threshold)

    annotations = load_annotations(output_path)

    remaining = [v for v in videos if v.name not in annotations]
    if not remaining:
        print(
            "[INFO] All videos already annotated. Delete entries from "
            f"{output_path} to re-annotate."
        )
        sys.exit(0)

    print(
        f"[INFO] {len(remaining)} video(s) to annotate "
        f"({len(videos) - len(remaining)} already done)."
    )

    for i, vpath in enumerate(remaining, start=1):
        print(f"\n── Loading [{i}/{len(remaining)}] {vpath.name} ──")
        record, quit_all = annotate_video(vpath, i, len(remaining))

        if record is not None:
            annotations[vpath.name] = record
            save_annotations(output_path, annotations)

        if quit_all:
            print("[INFO] Quit requested. Exiting.")
            break

    cv2.destroyAllWindows()
    print("[DONE] Annotation session finished.")


if __name__ == "__main__":
    main()
