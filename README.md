# ABODA

ABandoned Objects DAtaset (ABODA) is a public dataset for abandoned object detection. ABODA comprises 11 sequences labeled with various real-application scenarios that are challenging for abandoned-object detection. The situations include crowded scenes, marked changes in lighting condition, night-time detection, as well as indoor and outdoor environments.

## Annotation Tool

An interactive video annotation tool for labeling ground-truth abandonment events. Each video can contain multiple abandonment events, each defined by a spatial bounding box and a temporal anchor frame.

### Requirements

```
pip install -r source/requirements.txt
```

Dependencies: `numpy`, `opencv-python`

### Usage

Annotate all videos in a directory:

```
python source/annotate.py path/to/videos/
```

Annotate a single video:

```
python source/annotate.py path/to/video.avi
```

Options:

```
--output, -o     Output JSON file (default: ground_truth.json in video directory)
--radius, -r     Initial ownership radius in pixels (default: 200)
--threshold, -t  Initial abandonment threshold in seconds (default: 15)
```

### Workflow

1. The video starts playing. Press **Space** to pause at any point.
2. While paused, click and drag to draw a bounding box around the abandoned bag.
3. Use the trackbar or **a**/**d** keys to navigate to the frame where the abandonment should be recorded.
4. Press **N** to commit the annotation and begin annotating the next event in the same video.
5. Repeat steps 2-4 for each abandonment event.
6. Press **Enter** to save all annotations for this video and move to the next one.
7. If the video contains no abandonment, press **F** to mark it as a negative sample.

### Controls

| Key          | Action                                         |
| ------------ | ---------------------------------------------- |
| Space        | Play / Pause                                   |
| a            | Skip back 5 seconds and pause                  |
| d            | Skip forward 5 seconds and pause               |
| T            | Jump forward by threshold seconds and pause    |
| Click + Drag | Draw ROI bounding box (paused only)            |
| N            | Commit current annotation, start next          |
| Enter        | Save all annotations, next video               |
| F            | Mark video as negative sample (no abandonment) |
| +/=          | Increase radius by 10px (max 500)              |
| -            | Decrease radius by 10px (min 10)               |
| ]            | Increase threshold by 5s (max 300)             |
| [            | Decrease threshold by 5s (min 5)               |
| H            | Toggle HUD overlay                             |
| Q            | Quit session                                   |

### HUD

The bottom overlay displays:

- Video name, frame number, and timestamp
- Number of committed annotations and current ROI coordinates
- Play/pause state, current radius, and threshold values
- Keybinding reference

Previously committed annotations appear as numbered yellow boxes on the video frame.

## Output Format

Annotations are saved to `ground_truth.json`. Each video maps to a list of annotation objects:

```json
{
  "video1.avi": [
    {
      "has_abandonment": true,
      "true_abandon_frame": 2016,
      "bag_roi": [133, 323, 54, 46],
      "radius_px": 200,
      "threshold_s": 15
    }
  ],
  "video7.avi": [
    {
      "has_abandonment": true,
      "true_abandon_frame": 449,
      "bag_roi": [332, 39, 46, 34],
      "radius_px": 200,
      "threshold_s": 15
    },
    {
      "has_abandonment": true,
      "true_abandon_frame": 4389,
      "bag_roi": [136, 81, 46, 69],
      "radius_px": 200,
      "threshold_s": 15
    }
  ]
}
```

### Fields

| Field                | Type               | Description                                                 |
| -------------------- | ------------------ | ----------------------------------------------------------- |
| `has_abandonment`    | bool               | Whether an abandonment event occurs                         |
| `true_abandon_frame` | int or null        | Frame number of the abandonment event                       |
| `bag_roi`            | [x, y, w, h] or [] | Bounding box of the abandoned bag at the abandon frame      |
| `radius_px`          | int                | Pixel distance threshold for owner proximity                |
| `threshold_s`        | int                | Seconds a bag must be unattended to be considered abandoned |

Negative samples (no abandonment) have `has_abandonment: false`, `true_abandon_frame: null`, and an empty `bag_roi`.

## Project Structure

```
source/
  annotate.py          Entry point: CLI, mouse handling, video loop
  annotator/
    constants.py       Shared constants (radii, colors, thresholds)
    state.py           AnnotationState class
    overlays.py        Radius circle, ROI box, HUD, committed annotations
    persistence.py     JSON load/save
```
