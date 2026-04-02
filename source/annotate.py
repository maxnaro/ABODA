"""Abandoned Luggage Annotation Tool

Rapidly annotate video datasets for abandoned-luggage detection.
Supports multiple abandonment events per video. Loads video files from a
directory or a single file and saves results to ground_truth.json.

Usage:
    python annotate.py [VIDEO_PATH_OR_DIR] [--output ground_truth.json]
"""

import argparse
import sys
from pathlib import Path

import cv2
import numpy as np

from annotator.constants import (
    DEFAULT_RADIUS_PX,
    DEFAULT_THRESHOLD_SECONDS,
    SKIP_SECONDS,
    TRACKBAR_NAME,
    WINDOW_NAME,
)
from annotator.overlays import (
    draw_committed_annotations,
    draw_hud,
    draw_radius_circle,
    draw_roi_preview,
)
from annotator.persistence import load_annotations, save_annotations
from annotator.state import AnnotationState

state = AnnotationState()


def mouse_callback(event, x, y, flags, _param):
    """Handle cursor tracking and ROI drawing (drawing only while paused)."""
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


def annotate_video(video_path, video_num, video_total):
    """Run the interactive annotation loop for a single video.

    Supports multiple annotations per video via the N key. Each press
    commits the current ROI + abandon frame and clears for the next.
    Enter saves all committed annotations and moves to the next video.

    Returns (record, quit_all) where record is a list of annotation dicts
    and quit_all indicates the user wants to exit the session.
    """
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
        draw_committed_annotations(display, state)
        draw_radius_circle(display, state)
        draw_roi_preview(display, state)
        if state.show_hud:
            draw_hud(
                display,
                state,
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

        elif key == ord("n") or key == ord("N"):
            # Commit current annotation, stamp abandon frame, clear for next
            if state.paused and state.roi_committed and not state.marked_negative:
                state.abandon_frame = state.current_frame
                state.commit_annotation()

        elif key == 13:
            # Commit any in-progress annotation, then save all and move on
            if state.paused and state.roi_committed and not state.marked_negative:
                state.abandon_frame = state.current_frame
                state.commit_annotation()
            break

        elif key == ord("f") or key == ord("F"):
            state.marked_negative = True
            state.has_abandonment = False
            state.roi_committed = None
            state.abandon_frame = None
            state.annotations.clear()

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
        record = [
            {
                "has_abandonment": False,
                "true_abandon_frame": None,
                "bag_roi": [],
                "radius_px": state.radius,
                "threshold_s": state.threshold,
            }
        ]
    else:
        record = (
            state.annotations
            if state.annotations
            else [
                {
                    "has_abandonment": False,
                    "true_abandon_frame": None,
                    "bag_roi": [],
                    "radius_px": state.radius,
                    "threshold_s": state.threshold,
                }
            ]
        )

    return record, quit_all


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
