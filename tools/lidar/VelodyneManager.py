# VelodyneManager class by Arash Javanmard
# https://github.com/ArashJavan/veloparser

# Modified by Jon Rippe for UAA Traffic Engineering Project
# Capstone Fall 2021

# Uses data extracted by Lidar class to generate individual
# 360 degree frames.  Saves each frame as a point cloud.

import os
from pathlib import Path
from queue import Queue
import dpkt
import datetime
import numpy as np
import time
from threading import Thread

from tools.lidar.gps import utc_to_weekseconds
from tools.lidar.lidar import *
from tools.utils import read_config

CONFIG = read_config()


class VelodyneManager():
    def __init__(self, type, pcap_path, out_root):
        self.pcap_path = Path(pcap_path)
        self.lidar_type = type
        self.lidar = None
        self.out_root = out_root
        self.frames = Queue(maxsize=CONFIG['frame buffer size'])
        self.thread = Thread(
            target=self.run,
            args=(),
            daemon=True,
        )
        self.running = True

        self.pos_X = None
        self.pos_Y = None
        self.pos_Z = None
        self.intensities = None
        self.latitudes = None
        self.timestamps = None
        self.distances = None
        self.indicies = None
        self.longitudes = None

        self.frame_nr = 0
        self.cur_azimuth = None
        self.last_azimuth = None
        self.datetime = None

        self.gps_fp = None

        if type.lower() == "velodynevlp16":
            self.lidar = VelodyneVLP16()

        self.start()

    def get_pcap_length(self):
        # open pcap file
        try:
            fpcap = open(self.pcap_path, 'rb')
            lidar_reader = dpkt.pcap.Reader(fpcap)
        except Exception as ex:
            print(str(ex))
            return 0

        counter = 0
        # itearte through each data packet and timestamps
        for _, _ in enumerate(lidar_reader):
            counter += 1
            break
        fpcap.close()
        return counter

    def start(self):
        self.thread.start()

    def run(self):
        """
        Exteractis point clouds from pcap file
        :return:
        """
        pcap_len = self.get_pcap_length()
        if pcap_len <= 0:
            return

        # open pcap file
        try:
            fpcap = open(self.pcap_path, 'rb')
            self.lidar_reader = dpkt.pcap.Reader(fpcap)
        except Exception as ex:
            print(str(ex))
            return

        # create output folder hierarchy
        if (CONFIG['text'] or CONFIG['ply']) and not self.create_folders():
            return

        # iterate through each data packet and timestamps
        # pbar = tqdm(total=pcap_len)
        reader_enum = list(self.lidar_reader)
        reader_enum_arr = np.array(reader_enum)
        reader_ts = reader_enum_arr[:, 0].astype(np.float)
        start_idx = len(reader_ts[reader_ts < CONFIG['from']]) - 1
        del reader_enum_arr
        del reader_ts
        for idx in range(start_idx, len(reader_enum)):
            (ts, buf) = reader_enum[idx]
            # Set starting index
            if ts < CONFIG['from']:
                # print(f'seeking: {idx} - {ts}')
                continue
            if 0 < CONFIG['to'] < ts:
                break
            if not self.running:
                break

            self.datetime = datetime.datetime.utcfromtimestamp(ts)

            eth = dpkt.ethernet.Ethernet(buf)
            data = eth.data.data.data

            # handle Position-Frame (GPS-Data)
            if CONFIG['gps']:
                if eth.data.data.sport == CONFIG['gps-port']:
                    self.process_gps_frame(data, ts, idx)

            # Handle Data-Frame (Point clouds)
            if eth.data.data.sport == CONFIG['data-port']:
                self.process_data_frame(data, ts, idx)

            # pbar.update(1)
            idx += 1

        if self.gps_fp is not None:
            self.gps_fp.close()

    def process_data_frame(self, data, timestamp, index):

        cur_X, cur_Y, cur_Z, cur_intensities, cur_latitudes, cur_timestamps, cur_distances = self.lidar.process_data_frame(
            data, index)

        # number of sequences
        n_seq = int(len(cur_X) / self.lidar.count_lasers)

        cur_indicies = np.tile(np.arange(self.lidar.count_lasers), n_seq)
        cur_longitudes = np.tile(self.lidar.omega, n_seq)

        # initilaise states
        if index == 0 or self.pos_X is None:
            self.pos_X = cur_X
            self.pos_Y = cur_Y
            self.pos_Z = cur_Z
            self.intensities = cur_distances
            self.latitudes = cur_latitudes
            self.timestamps = cur_timestamps
            self.distances = cur_distances
            self.indicies = cur_indicies
            self.longitudes = cur_longitudes

        if self.cur_azimuth is None:
            self.cur_azimuth = cur_latitudes
            self.last_azimuth = cur_latitudes

        # update current azimuth before checking for roll over
        self.cur_azimuth = cur_latitudes

        # check if a frame is finished
        idx_rollovr = self.is_roll_over()

        # handle rollover (full 360° frame)
        if idx_rollovr is not None:

            if idx_rollovr > 0:
                self.pos_X = np.hstack((self.pos_X, cur_X[0:idx_rollovr + 1]))
                self.pos_Y = np.hstack((self.pos_Y, cur_Y[0:idx_rollovr + 1]))
                self.pos_Z = np.hstack((self.pos_Z, cur_Z[0:idx_rollovr + 1]))
                self.intensities = np.hstack(
                    (self.intensities, cur_intensities[0:idx_rollovr + 1]))
                self.latitudes = np.hstack(
                    (self.latitudes, cur_latitudes[0:idx_rollovr + 1]))
                self.timestamps = np.hstack(
                    (self.timestamps, cur_timestamps[0:idx_rollovr + 1]))
                self.distances = np.hstack(
                    (self.distances, cur_distances[0:idx_rollovr + 1]))
                self.indicies = np.hstack(
                    (self.indicies, cur_indicies[0:idx_rollovr + 1]))
                self.longitudes = np.hstack(
                    (self.longitudes, cur_longitudes[0:idx_rollovr + 1]))

            min, sec, micro = self.time_from_lidar(self.timestamps[0])
            self.datetime = self.datetime.replace(minute=min,
                                                  second=sec,
                                                  microsecond=int(micro))
            gpsweek, gpsdays, gpsseconds, gpsmicrosec = utc_to_weekseconds(
                self.datetime, 0)

            # Remove noise points, points further than max distance, & outside z-range
            noise_points = []
            for i in range(self.timestamps.shape[0]):
                if (self.distances[i] == 0
                        or self.distances[i] > CONFIG['max distance'] > 0
                    ) or not (CONFIG['z-range'][0] <= self.pos_Z[i] <=
                              CONFIG['z-range'][1]):
                    noise_points.append(i)
            self.pos_X = np.delete(self.pos_X, noise_points)
            self.pos_Y = np.delete(self.pos_Y, noise_points)
            self.pos_Z = np.delete(self.pos_Z, noise_points)
            self.intensities = np.delete(self.intensities, noise_points)
            self.latitudes = np.delete(self.latitudes, noise_points)
            self.timestamps = np.delete(self.timestamps, noise_points)
            self.distances = np.delete(self.distances, noise_points)
            self.indicies = np.delete(self.indicies, noise_points)
            self.longitudes = np.delete(self.longitudes, noise_points)

            # Remove random points down to max point count
            if self.timestamps.shape[0] > CONFIG['max points'] > 0:
                random_indices = np.random.choice(self.timestamps.shape[0],
                                                  size=CONFIG['max points'],
                                                  replace=False)
                self.pos_X = self.pos_X[random_indices]
                self.pos_Y = self.pos_Y[random_indices]
                self.pos_Z = self.pos_Z[random_indices]
                self.intensities = self.intensities[random_indices]
                self.latitudes = self.latitudes[random_indices]
                self.timestamps = self.timestamps[random_indices]
                self.distances = self.distances[random_indices]
                self.indicies = self.indicies[random_indices]
                self.longitudes = self.longitudes[random_indices]

            # Add to frame queue
            if not (CONFIG['text'] or CONFIG['ply']):
                IDs = np.array([*range(self.timestamps.shape[0])])
                frame = np.stack((IDs, self.timestamps, self.pos_X, self.pos_Y,
                                  self.pos_Z, self.distances, self.intensities,
                                  self.latitudes, self.longitudes),
                                 axis=1)
                self.frames.put((frame, timestamp), block=True)

            # Write to file(s)
            if CONFIG['text']:
                fpath = "{}/{}_frame_{}.{:06d}.txt".format(
                    self.txt_path, self.frame_nr, gpsseconds, gpsmicrosec)
                write_pcl_txt(fpath, self.timestamps, self.pos_X, self.pos_Y,
                              self.pos_Z, self.indicies, self.intensities,
                              self.latitudes, self.longitudes, self.distances)

            if CONFIG['ply']:
                fpath = "{}/{}_frame_{}.{:06d}.pcd".format(
                    self.pcl_path, self.frame_nr, gpsseconds, gpsmicrosec)
                write_pcd(fpath, self.pos_X, self.pos_Y, self.pos_Z,
                          self.intensities)

            # reset states
            if idx_rollovr > 0:
                self.pos_X = cur_X[idx_rollovr + 1:]
                self.pos_Y = cur_Y[idx_rollovr + 1:]
                self.pos_Z = cur_Z[idx_rollovr + 1:]
                self.intensities = cur_intensities[idx_rollovr + 1:]
                self.latitudes = cur_latitudes[idx_rollovr + 1:]
                self.timestamps = cur_timestamps[idx_rollovr + 1:]
                self.distances = cur_distances[idx_rollovr + 1:]
                self.indicies = cur_indicies[idx_rollovr + 1:]
                self.longitudes = cur_longitudes[idx_rollovr + 1:]
            else:
                self.pos_X = cur_X
                self.pos_Y = cur_Y
                self.pos_Z = cur_Z
                self.intensities = cur_intensities
                self.latitudes = cur_latitudes
                self.timestamps = cur_timestamps
                self.distances = cur_distances
                self.indicies = cur_indicies
                self.longitudes = cur_longitudes

            self.frame_nr += 1

            # reset roll over check
            self.cur_azimuth = None
            return

        self.pos_X = np.hstack((self.pos_X, cur_X))
        self.pos_Y = np.hstack((self.pos_Y, cur_Y))
        self.pos_Z = np.hstack((self.pos_Z, cur_Z))
        self.intensities = np.hstack((self.intensities, cur_intensities))
        self.latitudes = np.hstack((self.latitudes, cur_latitudes))
        self.timestamps = np.hstack((self.timestamps, cur_timestamps))
        self.distances = np.hstack((self.distances, cur_distances))
        self.indicies = np.hstack((self.indicies, cur_indicies))
        self.longitudes = np.hstack((self.longitudes, cur_longitudes))

        self.last_azimuth = cur_latitudes

    def process_gps_frame(self, data, timestamp, index):
        gps_msg = self.lidar.process_position_frame(data, index)

        # open gps file to write in
        if self.gps_fp is None:
            try:
                gps_path = Path("{}/{}.txt".format(self.out_path, "data_gps"))
                self.gps_fp = open(gps_path, 'w')
                # write point cloud as a text-file
                header = "UTC-Time, Week, Seconds [sow], Status, Latitude [Deg], Longitudes [Deg], Velocity [m/s]\n"
                self.gps_fp.write(header)
            except Exception as ex:
                print(ex)

        txt = "{}, {}, {}, {}, {} {}, {} {}, {}\n".format(
            gps_msg.datetime, gps_msg.weeks, gps_msg.seconds, gps_msg.status,
            gps_msg.lat, gps_msg.lat_ori, gps_msg.long, gps_msg.lat_ori,
            gps_msg.velocity)
        self.gps_fp.write(txt)

    def is_roll_over(self):
        """
        Check if one frame is completed, therefore 360° rotation of the lidar
        :return:
        """
        diff_cur = self.cur_azimuth[0:-1] - self.cur_azimuth[1:]
        diff_cur_last = self.cur_azimuth - self.last_azimuth

        res_cur = np.where(diff_cur > 0.)[0]
        res_cur_last = np.where(diff_cur_last < 0.)[0]
        if res_cur.size > 0:
            index = res_cur[0]
            return index
        elif res_cur_last.size > 0:
            index = res_cur_last[0]
            return index
        else:
            return None

    def time_from_lidar(self, timestamp):
        """
        convert the timestamp [top of the hour in microsec] of a firing into
        minutes, seconds and microseconds
        :param timestamp:
        :return:
        """
        try:
            micro = timestamp % (1000 * 1000)
            min_float = (timestamp / (1000 * 1000 * 60)) % 60
            min = int(min_float)
            sec = int((timestamp / (1000 * 1000)) % 60)
            min = int(min)
        except Exception as ex:
            print(ex)
        return min, sec, micro

    def create_folders(self):
        self.out_path = Path("{}/{}".format(self.out_root,
                                            self.lidar_type.lower()))

        # creating output dir
        try:
            os.makedirs(self.out_path.absolute())
        except Exception as ex:
            print(str(ex))
            return False

        # create point cloud dirs
        self.pcl_path = Path("{}/{}".format(self.out_path, "data_pcl"))
        try:
            os.makedirs(self.pcl_path.absolute())
        except Exception as ex:
            print(str(ex))
            return False

        # if text-files are desired, create text-file dir
        if CONFIG['text']:
            self.txt_path = Path("{}/{}".format(self.out_path, "data_ascii"))
            try:
                os.makedirs(self.txt_path.absolute())
            except Exception as ex:
                print(str(ex))
                return False
        return True

    def get_frame(self):
        if not self.more_frames():
            return None, None
        frame = self.frames.get()
        if frame is None:
            return None, None
        return frame[0], frame[1]

    def more_frames(self, t=0):
        while self.frames.qsize() == 0 and self.running and t < 5:
            time.sleep(0.1)
            t += 1
        return self.frames.qsize() > 0

    def stop(self):
        self.running = False


def write_pcl_txt(path,
                  timestamps,
                  X,
                  Y,
                  Z,
                  laser_id,
                  intensities=None,
                  latitudes=None,
                  longitudes=None,
                  distances=None):
    header = "time,X,Y,Z,id,intensity,latitude,longitudes,distance\n"
    try:
        fp = open(path, 'w')
        fp.write(header)
    except Exception as ex:
        print(str(ex))
        return

    M = np.vstack((timestamps, X, Y, Z, laser_id))

    if intensities is not None:
        M = np.vstack((M, intensities))
    if latitudes is not None:
        M = np.vstack((M, latitudes))
    if longitudes is not None:
        M = np.vstack((M, longitudes))
    if distances is not None:
        M = np.vstack((M, distances))

    np.savetxt(fp,
               M.T,
               fmt=('%d', '%.6f', '%.6f', '%.6f', '%d', '%d', '%.3f', '%.3f',
                    '%.3f'),
               delimiter=',')
    fp.close()


def write_pcd(path, X, Y, Z, intensities=None):
    template = """VERSION {}\nFIELDS {}\nSIZE {}\nTYPE {}\nCOUNT {}\nWIDTH {}\nHEIGHT {}\nVIEWPOINT {}\nPOINTS {}\nDATA {}\n"""

    X = X.astype(np.float32).reshape(1, -1)
    Y = Y.astype(np.float32).reshape(1, -1)
    Z = Z.astype(np.float32).reshape(1, -1)

    if intensities is not None:
        I = intensities.astype(np.float32).reshape(1, -1)
        M = np.hstack((X.T, Y.T, Z.T, I.T))
        pc_data = M.view(
            np.dtype([('x', np.float32), ('y', np.float32), ('z', np.float32),
                      ('i', np.float32)]))
        tmpl = template.format("0.7", "x y z intensity", "4 4 4 4", "F F F F",
                               "1 1 1 1", pc_data.size, "1", "0 0 0 1 0 0 0",
                               pc_data.size, "binary")
    else:
        M = np.hstack((X.T, Y.T, Z.T))
        pc_data = M.view(
            np.dtype([('x', np.float32), ('y', np.float32),
                      ('z', np.float32)]))
        tmpl = template.format("0.7", "x y z", "4 4 4", "F F F", "1 1 1",
                               pc_data.size, "1", "0 0 0 1 0 0 0",
                               pc_data.size, "binary")

    fp = open(path, 'wb')
    fp.write(tmpl.encode())
    fp.write(pc_data.tostring())
    fp.close()
