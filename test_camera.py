"""
test_camera.py — Run this to find working camera index and format.
Usage: python test_camera.py
"""
import cv2

print("=" * 50)
print("  Camera Detection Test")
print("=" * 50)

found = []
for i in range(6):
    cap = cv2.VideoCapture(i, cv2.CAP_DSHOW)
    if cap.isOpened():
        ret, frame = cap.read()
        if ret and frame is not None:
            h, w = frame.shape[:2]
            channels = frame.shape[2] if len(frame.shape) == 3 else 1
            print(f"  Camera {i}: WORKING  — {w}x{h}, {channels} channels")
            found.append(i)
        else:
            print(f"  Camera {i}: Opens but can't read frames")
        cap.release()
    else:
        cap.release()
        cap2 = cv2.VideoCapture(i)
        if cap2.isOpened():
            ret, frame = cap2.read()
            if ret and frame is not None:
                h, w = frame.shape[:2]
                print(f"  Camera {i}: WORKING (default backend) — {w}x{h}")
                found.append(i)
            else:
                print(f"  Camera {i}: Opens but can't read (default backend)")
            cap2.release()
        else:
            cap2.release()
            print(f"  Camera {i}: Not found")

print("=" * 50)
if found:
    print(f"  Working camera index(es): {found}")
    print(f"  Use index: {found[0]}")
else:
    print("  No working camera found!")
    print("  Make sure Camo is open on both iPhone and PC.")
print("=" * 50)
