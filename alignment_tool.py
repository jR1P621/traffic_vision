'''
This tool can help with camera/lidar alignment and syncronization by
allowing manual adjustments to camera settings.

THIS IS NOT A ROBUST TOOL!!!!
It is not meant for eventual end users.
It will crash.
There are magic numbers all over the place.
'''

from tools.lidar.lidar import *
from scapy.all import *
from graphics import Point, Rectangle, Text, GraphWin, Entry
from time import sleep
import os
import cv2
import numpy as np
from Camera import Camera
from tools.lidar.VelodyneManager import VelodyneManager
from tools.utils import CONFIG, read_config, save_config

CAMERAS = read_config('cameras.json')
CUR_CAM = 0  # the index of the camera you're modifying
SCALING = 0.75  # display scaling: camera resolution * SCALING
DETECTIONS = True  # show detection objects

CAMERA_INPUT_PATH = 'test_data/video'
CAMERA_OUTPUT_PATH = 'output_videos'
LIDAR_INPUT_PATH = 'test_data/lidar/Day2.pcap'


class Button:
    def __init__(self, text, pos, size, just) -> None:
        if just == 0:
            self.p1 = Point(pos[0], pos[1])
            self.p2 = Point(pos[0] + size[0], pos[1] + size[1])
        elif just == 1:
            self.p1 = Point(pos[0] - (size[0] // 2), pos[1])
            self.p2 = Point(pos[0] + (size[0] // 2), pos[1] + size[1])
        else:
            self.p1 = Point(pos[0] - size[0], pos[1])
            self.p2 = Point(pos[0], pos[1] + size[1])
        self.pt = Point(self.p1.x + (size[0] // 2), self.p1.y + (size[1] // 2))

        self.rect = Rectangle(self.p1, self.p2)
        self.text = Text(self.pt, text)

    def draw(self, window):
        self.rect.draw(window)
        self.text.draw(window)

    def is_clicked(self, pt: Point):
        return self.p1.getX() < pt.getX() < self.p2.getX() and \
            self.p1.getY() < pt.getY() < self.p2.getY()


camera: Camera = Camera(CAMERA_INPUT_PATH,
                        os.path.join(CAMERA_OUTPUT_PATH,
                                     f'output{CUR_CAM}.mp4'),
                        azimuth=CAMERAS[CUR_CAM]['azimuth'],
                        fov=CAMERAS[CUR_CAM]['fov'],
                        offset=CAMERAS[CUR_CAM]['offset'],
                        fps=CAMERAS[CUR_CAM]['fps'],
                        time_offset=CAMERAS[CUR_CAM]['time_offset'],
                        dist_coeff=CAMERAS[CUR_CAM]['dist_coeff'],
                        omega=CAMERAS[CUR_CAM]['omega'],
                        name=CAMERAS[CUR_CAM]['name'])

v_man = VelodyneManager(
    CONFIG['type'],
    LIDAR_INPUT_PATH,
    None,
)
points = v_man.get_frame()

print('reading PCAP', end='')
while v_man.frames.empty() and v_man.running:
    sleep(.2)
    print('.', end='')
print()

# Setup GUI.  Lots of magic numbers here!
gui = GraphWin('Alignment Tool', 500, len(CAMERAS[CUR_CAM]) * 55 + 200)
btns = {
    'next':
    Button('Next', (5, gui.getHeight() - 35), (75, 30), 0),
    'render':
    Button('Render', (gui.getWidth() // 2, gui.getHeight() - 35), (75, 30), 1),
    'quit':
    Button('Quit', (gui.getWidth() - 5, gui.getHeight() - 35), (75, 30), 2),
    'save':
    Button('Save', (gui.getWidth() - 5, gui.getHeight() - 70), (75, 30), 2),
}
boxs = {}
bx_pt = Point(gui.width // 2, 10)
for key, index in CAMERAS[CUR_CAM].items():
    txt = Text(bx_pt, key)
    txt.draw(gui)
    bx_pt.y += 20
    boxs[key] = Entry(bx_pt, 40)
    boxs[key].draw(gui)
    bx_pt.y += 35
for b in btns.values():
    b.draw(gui)
next_frame = 1
next_count = Entry(Point(btns['next'].pt.getX(), btns['next'].pt.getY() - 35),
                   5)
next_count.setText(str(next_frame))
next_count.draw(gui)

running = True
oof = False
while running:
    points, timestampL = v_man.get_frame()
    next_frame -= 1
    while next_frame <= 0:
        frame, _ = camera.get_frame(timestampL)
        if frame is None or points is None:
            oof = True
            Text(Point(gui.getWidth() // 2,
                       gui.getHeight() - 50), "Out of Frames").draw(gui)
        if not oof:
            frame_out = frame.copy()
            if DETECTIONS:
                objects, frame_out = camera.get_detections(frame_out)
            for key, value in boxs.items():
                exec(f'boxs["{key}"].setText(str(camera.{key}))')
            camera.transforms = camera._get_transform_matrices()
            cam_points = camera.transform_points(points)
            for p in cam_points:
                hls = (int(p[5] / CONFIG['max distance'] * 180), 100, 255)
                bgr = cv2.cvtColor(np.uint8([[hls]]), cv2.COLOR_HLS2BGR)[0][0]
                color = (int(bgr[0]), int(bgr[1]), int(bgr[2]))
                cv2.circle(img=frame_out,
                           center=(int(p[2]), int(p[3])),
                           radius=int(1 + CONFIG['max distance'] / (p[5])),
                           color=color,
                           thickness=-1)
            birds_eye_scale = 5
            birds_eye = np.zeros(shape=[
                CONFIG['max distance'] * 2 * birds_eye_scale,
                CONFIG['max distance'] * 2 * birds_eye_scale, 3
            ],
                                 dtype=np.uint8)
            for p in points:
                hls = (int(p[5] / CONFIG['max distance'] * 180), 100, 255)
                bgr = cv2.cvtColor(np.uint8([[hls]]), cv2.COLOR_HLS2BGR)[0][0]
                color = (int(bgr[0]), int(bgr[1]), int(bgr[2]))
                cv2.circle(
                    img=birds_eye,
                    center=(int(p[2] * birds_eye_scale +
                                CONFIG['max distance'] * birds_eye_scale),
                            int(p[3] * -birds_eye_scale +
                                CONFIG['max distance'] * birds_eye_scale)),
                    radius=0,
                    color=color,
                    thickness=-1)
            cv2.imshow(
                'Display',
                cv2.resize(frame_out, (int(
                    camera.res_X * SCALING), int(camera.res_Y * SCALING))))
            cv2.imshow('LiDAR', birds_eye)
            cv2.waitKey(1)

        while True:
            click = gui.getMouse()
            if btns['next'].is_clicked(click):
                next_frame = eval(next_count.getText())
                break
            elif btns['quit'].is_clicked(click):
                running = False
                next_frame = 1
                cv2.destroyAllWindows()
                break
            elif btns['render'].is_clicked(click):
                break
            elif btns['save'].is_clicked(click):
                save_config(CAMERAS, 'cameras.json')
                break
        for key, value in boxs.items():
            try:
                if isinstance(eval(f'camera.{key}'), str):
                    exec(f'camera.{key} = value.getText()')
                    CAMERAS[CUR_CAM][key] = value.getText()
                else:
                    exec(f'camera.{key} = eval(value.getText())')
                    CAMERAS[CUR_CAM][key] = eval(value.getText())
            except Exception as e:
                exec(f'camera.{key} = value.getText()')
