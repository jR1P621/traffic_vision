# cviz.py
# May 2021
# Some video processing utilities that are nice to have.


import os
import cv2


def vid_dimz(src, width=None):
	""" Get the original height x width dimensions of a video.
	Or get the new dimensions after resizing to a specified width. """
	cap = cv2.VideoCapture(src)
	if width is None:
	    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
	    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
	else:
	    (h, w) = resize(cap.read()[1], width=width).shape[:2]
	cap.release()
	return w, h


def resize(frame, width=None, height=None):
	""" Resize a video frame to a new width, preserving the original aspect ratio. """
	if width is None and height is None:
	    return frame
	(h, w) = frame.shape[:2]
	if width is None:
	    ratio = height / float(h)
	    dim = (int(w * ratio), height)
	else:
	    ratio = width / float(w)
	    dim = (width, int(h * ratio))
	return cv2.resize(frame, dim)


def vid_writer(output, w, h, fps):
	""" Set up the output video writer. """
	fourcc = cv2.VideoWriter_fourcc(*"MJPG")
	writer = cv2.VideoWriter(output, fourcc, fps, (w, h), True)
	return writer


def valid_vidtyp(in_vid):
	""" Check if the file extension is a valid video file type. """
	exts = set(['.avi', '.mp4', '.mjpeg', '.h264',])
	ext = os.path.splitext(os.path.basename(in_vid))[1]
	return ext in exts


def frame_cnt(src, manual=False):
	""" Get the number of frames in a video. """
	cap = cv2.VideoCapture(src)
	frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
	# If the capture property fails, count the frames manually, O(n) time.
	if frames <= 0 or manual:
		frames = 0
		while True:
			check, frame = cap.read()
			if not check or frame is None:
				break
			frames += 1
	cap.release()
	return frames


def vid_fps(src):
	""" Get the frames per second of the video. """
	cap = cv2.VideoCapture(src)
	fps = int(cap.get(cv2.CAP_PROP_FPS))
	cap.release()
	# Just set 20fps for weird video files that return a large number from the cap property.
	if fps > 50:
		return 20
	return fps


def set_pos(vs, start_frame):
	""" Set the index of the frame to be read next in a video stream.
	Useful for setting the first frame for a block of frames. """
	vs.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
	# If the video property fails, set it manually, O(n) time.
	while vs.get(cv2.CAP_PROP_POS_FRAMES) < start_frame:
		check, frame = vs.read()
		if not check or frame is None:
		    break
	return vs


def avi_conv(src):
	""" Convert a video file type to .avi """
	path, ext = os.path.splitext(src)
	cap = cv2.VideoCapture(src)
	w, h = vid_dimz(src)
	writer = vid_writer(f"{path}.avi", w, h, int(cap.get(cv2.CAP_PROP_FPS)))
	# O(n) time.
	while True:
		check, frame = cap.read()
		if not check or frame is None:
		    break
		writer.write(frame)
	cap.release()
	writer.release()
	return f"{path}.avi"



##
