from typing import List, Tuple
from tools.utils import CONFIG
from tools.video.videostream import Video_Thread
import tools.video.cviz as cviz
from TrafficDetector import TrafficDetector
import os
import time
from queue import Empty, Queue
from threading import Thread
import numpy as np
from math import sin, cos, tan, radians, dist
from numpy.core.shape_base import hstack


class Camera:
    def __init__(self,
                 vid_path: str,
                 output_path: str,
                 azimuth: float,
                 fov: float,
                 offset: List[float] = [0, 0, 0],
                 time_offset: float = 0.0,
                 fps: float = 30,
                 dist_coeff: List[float] = [0.0, 0.0, 0.0],
                 omega: float = 0,
                 name: str = None) -> None:
        if name:
            self.name = name
        else:
            self.name = os.path.basename(vid_path)
        self.azimuth: float = azimuth
        self.fov: float = fov
        self.offset: List[float] = offset
        self.videos: List[str] = self._get_video_files(vid_path)
        self.current_vid: int = 0
        self.fps: float = fps
        self.base_timestamp: float = self._get_starting_timestamp(
            self.current_vid)
        self.current_timestamp: float = self.base_timestamp
        self.time_offset: float = time_offset
        self.dist_coeff: List[float] = dist_coeff
        self.omega: float = omega
        self.buf: Queue = Queue(maxsize=CONFIG['frame buffer size'])

        self.res_X, self.res_Y = cviz.vid_dimz(self.videos[self.current_vid])
        self.detector: TrafficDetector = TrafficDetector()
        self.vid_writer = cviz.vid_writer(output_path, self.res_X, self.res_Y,
                                          10)
        self.v_stream: Video_Thread = None
        self._seek_to_start()
        self.buf_thread = Thread(
            target=self.fill_buffer,
            args=(),
            daemon=True,
        )
        self.stop: bool = False
        self.buf_thread.start()
        self.current_frame: np.ndarray = None
        self.transforms: List[np.ndarray] = self._get_transform_matrices()

    def fill_buffer(self) -> None:
        '''
        Fills the Camera object's frame buffer with frames taken from its v_stream
        object.  If the v_stream buffer is empty, shutdown the v_stream and open a
        new v_stream for the next video file.  Then continue.
        '''
        while not self.stop:
            frame = self.v_stream.read()
            if frame is None:
                self.v_stream.release()
                self.current_vid += 1
                if self.current_vid == len(self.videos) or (cviz.vid_dimz(
                        self.videos[self.current_vid])) != (self.res_X,
                                                            self.res_Y):
                    break
                else:
                    self.v_stream = Video_Thread(
                        self.videos[self.current_vid], )
                    self.base_timestamp = self._get_starting_timestamp(
                        self.current_vid)
                    self.v_stream.start()
            else:
                timestamp = self.base_timestamp + self.v_stream.get_current_timestamp(
                    self.fps)
                self.buf.put((frame, timestamp))

    def get_frame(self, time: float) -> Tuple[np.ndarray, float]:
        '''
        Gets the closest frame to the passed in timestamp.
        '''
        while self.buf.qsize() > 0 and dist(
            [time], [self.current_timestamp + self.time_offset]) >= dist(
                [time], [self.buf.queue[0][1] + self.time_offset]):
            (frame, timestamp) = self.buf.get()
            self.current_timestamp = timestamp
            self.current_frame = frame
        if self.buf.qsize() > 0:
            return self.current_frame, self.current_timestamp + self.time_offset
        return None, None

    def get_detections(self,
                       frame: np.ndarray) -> Tuple[List[object], np.ndarray]:
        '''
        Gets objects detected in the frame.  Returns object data and modified
        frame with drawn bounding boxes.
        '''
        objects, frame = self.detector.traffic_detections(frame)
        return objects, frame

    def shutdown(self) -> None:
        '''
        Stops all camera threads
        '''
        self.stop = True
        self.v_stream.release()
        self.vid_writer.release()

    def write_frame(self, frame: np.ndarray) -> None:
        '''
        Writes frame to output file.
        '''
        self.vid_writer.write(frame)

    def _get_video_files(self, path: str) -> List[str]:
        '''
        Gets paths of all video files in directory.
        '''
        vids = []
        try:
            for f in os.listdir(path):
                vids.append(f'{path}' + '/' + f)
        except:
            pass
        return vids

    def _get_starting_timestamp(self, video_num: int) -> float:
        '''
        Gets the starting timestamp of the specified video file.
        '''
        filename = os.path.basename(self.videos[video_num])
        return time.mktime(time.strptime(filename[:19], '%Y-%m-%d_%H-%M-%S'))

    def _seek_to_start(self) -> None:
        '''
        Iterates through video files then frames until the configured starting
        timestamp is reached.
        '''
        while self._get_starting_timestamp(self.current_vid +
                                           1) <= CONFIG['from']:
            self.current_vid += 1
        self.v_stream = Video_Thread(self.videos[self.current_vid], )
        self.base_timestamp = self._get_starting_timestamp(self.current_vid)
        self.current_timestamp = self.base_timestamp
        self.v_stream.start()
        while self.base_timestamp + self.time_offset + self.v_stream.get_current_timestamp(
                self.fps) < CONFIG['from']:
            self.v_stream.read()

    def _get_transform_matrices(self) -> List[np.ndarray]:
        '''
        Creates transform matrices for projecting LiDAR points.
        '''
        S = self.res_X / (2 * tan(radians(self.fov / 2)))
        # Translate
        tT = np.array(
            [[1, 0, 0, -self.offset[0]],\
             [0, 1, 0, -self.offset[1]],\
             [0, 0, 1, -self.offset[2]],\
             [0, 0, 0, 1]])
        # Rotate
        rTz = np.array(
            [[cos(radians(self.azimuth)), -sin(radians(self.azimuth)), 0, 0],\
             [sin(radians(self.azimuth)), cos(radians(self.azimuth)), 0, 0],\
             [0, 0, 1, 0],\
             [0, 0, 0, 1]])
        rTx = np.array(
            [[1, 0, 0, 0],\
             [0, cos(radians(self.omega)), -sin(radians(self.omega)), 0],\
             [0, sin(radians(self.omega)), cos(radians(self.omega)), 0],\
             [0, 0, 0, 1]])
        # Top down to side
        cTa = np.array(
            [[1, 0, 0, 0],\
             [0, 0, 1, 0],\
             [0, 1, 0, 0],\
             [0, 0, 0, 1]])
        # Scale (Normalize to [-1,1]) and flip y axis
        sT = np.array(
            [[1/CONFIG['max distance'], 0, 0, 0],\
             [0, -1/CONFIG['max distance'], 0, 0],\
             [0, 0, 1/CONFIG['max distance'], 0],\
             [0, 0, 0, 1]])
        # Scale (Normalize x,y to fov and resolution)
        sTxy = np.array(
            [[S, 0, 0, 0],\
             [0, S, 0, 0],\
             [0, 0, 0, 0],\
             [0, 0, 0, 1]])
        # Translate to pixels on video
        tTxy = np.array(
            [[1, 0, 0, self.res_X/2],\
             [0, 1, 0, self.res_Y/2],\
             [0, 0, 1, 0],\
             [0, 0, 0, 1]])

        return [rTx @ rTz @ tT, sT @ cTa, tTxy @ sTxy]

    def transform_points(self, points: np.ndarray) -> np.ndarray:
        '''
        Projects LiDAR points onto the camera frame.
        Returns points with cartesian coords that align to frame pixels.
        '''
        # Get array in format ready for transforms:
        # [[x0, x1, ..., xn]
        #  [y0, y1, ..., yn]
        #  [z0, z1, ..., zn]
        #  [1 , 1 , ..., 1 ]]
        newp = np.hstack([points[:, 2:5], np.ones((points.shape[0], 1), int)])
        newp = np.transpose(newp)

        # Apply offset, azimuth, and omega
        newp = self.transforms[0] @ newp

        # Remove points outside camera's view
        fov1 = np.arctan(abs(newp[0] / newp[1]))
        fov2 = np.arctan(abs(newp[0] / newp[2]))
        valid_points = (fov1 <= radians((self.fov / 2) + 10)) * (
            fov2 <= radians((self.fov / 2) + 90 + 10)) * (newp[1] > 0)
        newp = newp[:, valid_points]

        # Change from bird's eye to side view and normalize
        newp = self.transforms[1] @ newp

        # Compensate for camera distortion
        newp[0] = newp[0] / newp[2]
        newp[1] = newp[1] / newp[2]
        r2 = newp[0]**2 + newp[1]**2
        newp[0] = newp[0] * (1 + self.dist_coeff[0] * r2 + self.dist_coeff[1] *
                             r2**2 + self.dist_coeff[2] * r2**3)
        newp[1] = newp[1] * (1 + self.dist_coeff[0] * r2 + self.dist_coeff[1] *
                             r2**2 + self.dist_coeff[2] * r2**3)

        # Normalize to camera pixel values
        newp = self.transforms[2] @ newp

        # Put back in original point cloud format
        newp = np.transpose(newp)
        new_points = hstack([
            points[valid_points, 0:2], newp[:, 0:3], points[valid_points, 5:9]
        ])

        return new_points