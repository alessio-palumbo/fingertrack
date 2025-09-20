# FingerTrack

FingerTrack is a Python application that detects hand/finger states using a webcam and outputs them to configurable consumers.

It supports:

- Stable finger state detection with smoothing.
- Optional live OpenCV window display.
- Modular consumer system (stdout, HTTP, gRPC, etc.).
- Easy integration with other applications.

## Features

- Detects which fingers are up per hand.
- Debouncing/smoothing over configurable history size.
- Output can be sent to multiple consumers:
  - Stdout
  - OpenCV window
  - HTTP API or custom consumers
- Optional continuous window refresh without affecting other consumers.
- Graceful shutdown on Ctrl+C or external SIGTERM.

## Quickstart (Local)

```bash
git clone <repo-url>
cd fingertrack
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
python app.py --show-window
```

- --show-window enables the live OpenCV feed.
- The finger states are printed to stdout by default.

Example output

```json
{
  "hands": [
    {
      "label": "left",
      "fingers": [1, 0, 0, 0, 0],
      "gesture": null
    },
    {
      "label": "right",
      "fingers": [1, 1, 1, 1, 1],
      "gesture": "swipe_left"
    }
  ]
}
```

## Command-Line Arguments

Fingertrack supports a few runtime options to tweak gesture detection and hand processing behavior:

**--frame-skip N**

- Description: Only process every Nth frame from the camera feed.
- Default: 1 (process every frame)
- Pros: Reduces CPU usage on slower machines.
- Cons: Skipping too many frames may make gestures less responsive.

**--buffer-size N**

- Description: Number of historical positions tracked per hand for smoothing gestures.
- Default: 5
- Pros: Larger values improve stability and reduce false positives.
- Cons: Too large may delay gesture recognition slightly.

**--gesture-threshold T**

- Description: Minimum normalized movement required to recognize a swipe.
- Default: 0.1
- Pros: Increasing reduces accidental gestures caused by small jitters.
- Cons: Setting too high may make gestures harder to trigger.

**--consumer [stdout|http]**

- Description: Select which output consumer to use.
- Default: `stdout`
- Options:
  - `stdout`: Print gesture events as JSON to the terminal.
  - `http`: Send gesture events to an HTTP endpoint.

**--url URL**

- Description: URL for the HTTP consumer when `--consumer http` is selected.
- Required if `--consumer http` is used.

**--show-window**

- Description: Display the camera feed in an OpenCV window.
- Default: Disabled
- Pros: Useful for debugging and visual verification of hand/gesture detection.
- Cons: Consumes additional CPU/GPU and may be distracting in production.

### Usage Tip:

- On a fast machine, you can reduce frame-skip to 1 for maximum responsiveness.
- For smoother gesture detection in a noisy environment, increase buffer-size slightly.
- Adjust threshold based on how sensitive you want swipes to be; balance responsiveness vs accidental triggers.

## Docker

You can run FingerTrack either with a live webcam (Linux only) or with a video file (all platforms).

### Linux (with webcam)

If you’re on Linux and have a webcam available at /dev/video0:

```bash
docker run --rm --device=/dev/video0 -it alessi0/fingertrack

```

This will start the app and stream from your webcam.

### macOS / Windows (with video file)

Docker Desktop on macOS/Windows does not expose /dev/video0.
Instead, you can run the app using a sample video:

```bash
docker run --rm -it -v $(pwd):/data -w /data alessi0/fingertrack python app.py --video myclip.mp4

```

Replace sample.mp4 with the path to a local video file.
(You may need to mount your file into the container, e.g. -v $(pwd):/data -w /data.)

## 🖥️ Native Builds (macOS / Windows)

For users who want live webcam support without Docker:

### macOS

Prebuilt binaries available for:

- amd64 (Intel Macs)
- arm64 (Apple Silicon)

1. Download the appropriate binary from the GitHub Releases
2. Make it executable and run:

```bash
chmod +x fingertrack
./fingertrack
```

- OpenCV will access your Mac camera natively via AVFoundation.

### Windows

- Prebuilt binary (.exe) available for amd64 Windows systems.
- Download the executable from GitHub Releases and run it directly.
- Webcam access is handled via DirectShow.

### Notes

- These native builds are fully self-contained — no Python or dependencies required.
- Use Docker only if you are on Linux or an ARM device and prefer a containerized environment.
