#!/usr/bin/env python3

import time
from picamera2 import Picamera2, Preview

def main():
    # Create Picamera2 instance
    picam2 = Picamera2()

    # Configure the camera for a preview stream (adjust resolution as needed)
    preview_config = picam2.create_preview_configuration(main={"size": (640, 480)})
    picam2.configure(preview_config)

    # Start the preview (QTGL uses the GPU; use Preview.QT if you prefer software rendering)
    picam2.start_preview(Preview.QTGL)
    picam2.start()

    print("Preview is running. Press Ctrl+C to stop.")
    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        pass
    finally:
        picam2.stop_preview()
        picam2.close()
        print("Preview stopped.")

if __name__ == "__main__":
    main()