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
