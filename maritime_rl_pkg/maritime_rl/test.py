import cv2
print(f"OpenCV version: {cv2.__version__}")
print(f"Has VideoWriter_fourcc: {hasattr(cv2, 'VideoWriter_fourcc')}")
print(f"Available codecs: {dir(cv2)}")