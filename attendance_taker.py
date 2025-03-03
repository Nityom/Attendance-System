import dlib
import numpy as np
import cv2
import os
import pandas as pd
import time
import logging
import sqlite3
import datetime
from collections import defaultdict
import threading


# Dlib  / Use frontal face detector of Dlib
detector = dlib.get_frontal_face_detector()

# Dlib landmark / Get face landmarks
predictor = dlib.shape_predictor('data/data_dlib/shape_predictor_68_face_landmarks.dat')

# Dlib Resnet Use Dlib resnet50 model to get 128D face descriptor
face_reco_model = dlib.face_recognition_model_v1("data/data_dlib/dlib_face_recognition_resnet_model_v1.dat")

# Create a connection to the database
conn = sqlite3.connect("attendance.db")
cursor = conn.cursor()

# Create a table for the current date
current_date = datetime.datetime.now().strftime("%Y_%m_%d")  # Replace hyphens with underscores
table_name = "attendance" 
create_table_sql = f"""CREATE TABLE IF NOT EXISTS {table_name} (
    name TEXT, 
    time TEXT, 
    date DATE, 
    snapshots INTEGER DEFAULT 0, 
    status TEXT, 
    UNIQUE(name, date)
)"""
cursor.execute(create_table_sql)


# Commit changes and close the connection
conn.commit()
conn.close()


class Face_Recognizer:
    def __init__(self):
        self.font = cv2.FONT_ITALIC

        # FPS
        self.frame_time = 0
        self.frame_start_time = 0
        self.fps = 0
        self.fps_show = 0
        self.start_time = time.time()

        # cnt for frame
        self.frame_cnt = 0

        #  Save the features of faces in the database
        self.face_features_known_list = []
        # / Save the name of faces in the database
        self.face_name_known_list = []

        #  List to save centroid positions of ROI in frame N-1 and N
        self.last_frame_face_centroid_list = []
        self.current_frame_face_centroid_list = []

        # List to save names of objects in frame N-1 and N
        self.last_frame_face_name_list = []
        self.current_frame_face_name_list = []

        #  cnt for faces in frame N-1 and N
        self.last_frame_face_cnt = 0
        self.current_frame_face_cnt = 0

        # Save the e-distance for faceX when recognizing
        self.current_frame_face_X_e_distance_list = []

        # Save the positions and names of current faces captured
        self.current_frame_face_position_list = []
        #  Save the features of people in current frame
        self.current_frame_face_feature_list = []

        # e distance between centroid of ROI in last and current frame
        self.last_current_frame_centroid_e_distance = 0

        #  Reclassify after 'reclassify_interval' frames
        self.reclassify_interval_cnt = 0
        self.reclassify_interval = 10

        # Add new attributes for snapshot-based attendance
        self.attendance_counter = defaultdict(int)
        self.session_started = False
        self.session_duration = 120  # 2 minutes for testing (change to 3600 for 1 hour)
        self.snapshot_interval = 10  # 10 seconds for testing (change to 300 for 5 minutes)
        self.required_snapshots = 6  # Minimum snapshots for attendance
        self.total_snapshots = 12  # Total snapshots in session
        self.last_snapshot_time = 0
        self.session_start_time = None
        
        # Flag to indicate if a snapshot was taken in the current interval
        self.snapshot_taken = False
        
        # Current snapshot number
        self.snapshot_number = 0

    #  "features_all.csv"  / Get known faces from "features_all.csv"
    def get_face_database(self):
        if os.path.exists("data/features_all.csv"):
            path_features_known_csv = "data/features_all.csv"
            csv_rd = pd.read_csv(path_features_known_csv, header=None)
            for i in range(csv_rd.shape[0]):
                features_someone_arr = []
                self.face_name_known_list.append(csv_rd.iloc[i][0])
                for j in range(1, 129):
                    if csv_rd.iloc[i][j] == '':
                        features_someone_arr.append('0')
                    else:
                        features_someone_arr.append(csv_rd.iloc[i][j])
                self.face_features_known_list.append(features_someone_arr)
            logging.info("Faces in Database： %d", len(self.face_features_known_list))
            return 1
        else:
            logging.warning("'features_all.csv' not found!")
            logging.warning("Please run 'get_faces_from_camera.py' "
                            "and 'features_extraction_to_csv.py' before 'face_reco_from_camera.py'")
            return 0

    def update_fps(self):
        now = time.time()
        # Refresh fps per second
        if str(self.start_time).split(".")[0] != str(now).split(".")[0]:
            self.fps_show = self.fps
        self.start_time = now
        self.frame_time = now - self.frame_start_time
        self.fps = 1.0 / self.frame_time
        self.frame_start_time = now

    @staticmethod
    # / Compute the e-distance between two 128D features
    def return_euclidean_distance(feature_1, feature_2):
        feature_1 = np.array(feature_1)
        feature_2 = np.array(feature_2)
        dist = np.sqrt(np.sum(np.square(feature_1 - feature_2)))
        return dist

    # / Use centroid tracker to link face_x in current frame with person_x in last frame
    def centroid_tracker(self):
        for i in range(len(self.current_frame_face_centroid_list)):
            e_distance_current_frame_person_x_list = []
            #  For object 1 in current_frame, compute e-distance with object 1/2/3/4/... in last frame
            for j in range(len(self.last_frame_face_centroid_list)):
                self.last_current_frame_centroid_e_distance = self.return_euclidean_distance(
                    self.current_frame_face_centroid_list[i], self.last_frame_face_centroid_list[j])

                e_distance_current_frame_person_x_list.append(
                    self.last_current_frame_centroid_e_distance)

            last_frame_num = e_distance_current_frame_person_x_list.index(
                min(e_distance_current_frame_person_x_list))
            self.current_frame_face_name_list[i] = self.last_frame_face_name_list[last_frame_num]

    #  cv2 window / putText on cv2 window
    def draw_note(self, img_rd):
        # Keep existing display code
        cv2.putText(img_rd, "Face Recognizer with Deep Learning", (20, 40), self.font, 1, (255, 255, 255), 1, cv2.LINE_AA)
        cv2.putText(img_rd, "Frame:  " + str(self.frame_cnt), (20, 100), self.font, 0.8, (0, 255, 0), 1, cv2.LINE_AA)
        cv2.putText(img_rd, "FPS:    " + str(self.fps.__round__(2)), (20, 130), self.font, 0.8, (0, 255, 0), 1, cv2.LINE_AA)
        cv2.putText(img_rd, "Faces:  " + str(self.current_frame_face_cnt), (20, 160), self.font, 0.8, (0, 255, 0), 1, cv2.LINE_AA)
        
        # Add snapshot timing information
        if self.session_started:
            time_to_next = max(0, self.snapshot_interval - (time.time() - self.last_snapshot_time))
            cv2.putText(img_rd, f"Next Snapshot: {time_to_next:.1f}s", (20, 190), self.font, 0.8, (0, 255, 0), 1, cv2.LINE_AA)
            session_time = time.time() - self.session_start_time
            cv2.putText(img_rd, f"Session Time: {session_time:.0f}s / {self.session_duration}s", (20, 220), self.font, 0.8, (0, 255, 0), 1, cv2.LINE_AA)
            cv2.putText(img_rd, f"Snapshot: {self.snapshot_number}/{self.total_snapshots}", (20, 250), self.font, 0.8, (0, 255, 0), 1, cv2.LINE_AA)
            
            # Display snapshot count for recognized faces
            y_pos = 280
            for name, count in self.attendance_counter.items():
                cv2.putText(img_rd, f"{name}: {count}/{self.total_snapshots} snapshots", (20, y_pos), self.font, 0.8, (0, 255, 0), 1, cv2.LINE_AA)
                y_pos += 30
        
        cv2.putText(img_rd, "Q: Quit", (20, 450), self.font, 0.8, (255, 255, 255), 1, cv2.LINE_AA)

        for i in range(len(self.current_frame_face_name_list)):
            img_rd = cv2.putText(img_rd, "Face_" + str(i + 1), tuple(
                [int(self.current_frame_face_centroid_list[i][0]), int(self.current_frame_face_centroid_list[i][1])]),
                                 self.font,
                                 0.8, (255, 190, 0),
                                 1,
                                 cv2.LINE_AA)

    def start_session(self):
        """Initialize a new attendance session"""
        self.session_started = True
        self.session_start_time = time.time()
        self.attendance_counter.clear()
        self.last_snapshot_time = time.time()
        self.snapshot_taken = False
        self.snapshot_number = 0
        logging.info(f"New session started - Duration: {self.session_duration} seconds")

    def end_session(self):
        """End the current session and calculate final attendance"""
        self.session_started = False
        logging.info("Session ended - Calculating final attendance")
        
        conn = sqlite3.connect("attendance.db")
        cursor = conn.cursor()
        current_date = datetime.datetime.now().strftime('%Y-%m-%d')
        
        for name, count in self.attendance_counter.items():
            # Skip "unknown" faces
            if name == "unknown":
                continue
                
            status = "Present" if count >= self.required_snapshots else "Absent"
            logging.info(f"{name}: {count}/{self.total_snapshots} snapshots - {status}")
            
            cursor.execute("""
                INSERT OR REPLACE INTO attendance (name, time, date, snapshots, status)
                VALUES (?, ?, ?, ?, ?)
            """, (name, datetime.datetime.now().strftime('%H:%M:%S'), current_date, count, status))
        
        conn.commit()
        conn.close()

    def take_snapshot(self):
        """Take a snapshot and record attendance for all recognized faces"""
        recognized_faces = False
        
        # Increment snapshot number
        self.snapshot_number += 1
        
        # Record attendance for all recognized faces in the current frame
        for name in self.current_frame_face_name_list:
            if name != "unknown":
                recognized_faces = True
                self.attendance_counter[name] = self.attendance_counter.get(name, 0) + 1
                logging.info(f"Snapshot {self.snapshot_number}/{self.total_snapshots}: Recorded for {name}")
        
        # If no recognized faces, log it
        if not recognized_faces:
            logging.info(f"Snapshot {self.snapshot_number}/{self.total_snapshots}: No recognized faces, skipping")
        
        self.last_snapshot_time = time.time()
        self.snapshot_taken = True
        
        # If we've reached the total number of snapshots, end the session
        if self.snapshot_number >= self.total_snapshots:
            logging.info("Reached maximum number of snapshots, ending session")
            self.end_session()

    #  Face detection and recognition wit OT from input video stream
    def process(self, stream):
        # 1. Get faces known from "features.all.csv"
        if self.get_face_database():
            while stream.isOpened():
                current_time = time.time()
                
                # Start session if not started
                if not self.session_started:
                    self.start_session()
                
                # Check if session has ended
                if self.session_started and current_time - self.session_start_time >= self.session_duration:
                    self.end_session()
                    break

                # Check if it's time for a new snapshot
                if current_time - self.last_snapshot_time >= self.snapshot_interval:
                    self.snapshot_taken = False  # Reset for new interval

                self.frame_cnt += 1
                logging.debug("Frame " + str(self.frame_cnt) + " starts")
                flag, img_rd = stream.read()
                if not flag:
                    break
                    
                kk = cv2.waitKey(1)

                # 2.  Detect faces for frame X
                faces = detector(img_rd, 0)

                # 3.  Update cnt for faces in frames
                self.last_frame_face_cnt = self.current_frame_face_cnt
                self.current_frame_face_cnt = len(faces)

                # 4.  Update the face name list in last frame
                self.last_frame_face_name_list = self.current_frame_face_name_list[:]

                # 5.  update frame centroid list
                self.last_frame_face_centroid_list = self.current_frame_face_centroid_list
                self.current_frame_face_centroid_list = []

                # 6.1  if cnt not changes
                if (self.current_frame_face_cnt == self.last_frame_face_cnt) and (
                        self.reclassify_interval_cnt != self.reclassify_interval):
                    logging.debug("scene 1:   No face cnt changes in this frame!!!")

                    self.current_frame_face_position_list = []

                    if "unknown" in self.current_frame_face_name_list:
                        self.reclassify_interval_cnt += 1

                    if self.current_frame_face_cnt != 0:
                        for k, d in enumerate(faces):
                            self.current_frame_face_position_list.append(tuple(
                                [faces[k].left(), int(faces[k].bottom() + (faces[k].bottom() - faces[k].top()) / 4)]))
                            self.current_frame_face_centroid_list.append(
                                [int(faces[k].left() + faces[k].right()) / 2,
                                 int(faces[k].top() + faces[k].bottom()) / 2])

                            img_rd = cv2.rectangle(img_rd,
                                                   tuple([d.left(), d.top()]),
                                                   tuple([d.right(), d.bottom()]),
                                                   (255, 255, 255), 2)

                    #  Multi-faces in current frame, use centroid-tracker to track
                    if self.current_frame_face_cnt != 1:
                        self.centroid_tracker()

                    for i in range(self.current_frame_face_cnt):
                        # 6.2 Write names under ROI
                        img_rd = cv2.putText(img_rd, self.current_frame_face_name_list[i],
                                             self.current_frame_face_position_list[i], self.font, 0.8, (0, 255, 255), 1,
                                             cv2.LINE_AA)
                    
                # 6.2  If cnt of faces changes, 0->1 or 1->0 or ...
                else:
                    logging.debug("scene 2: / Faces cnt changes in this frame")
                    self.current_frame_face_position_list = []
                    self.current_frame_face_X_e_distance_list = []
                    self.current_frame_face_feature_list = []
                    self.reclassify_interval_cnt = 0

                    # 6.2.1  Face cnt decreases: 1->0, 2->1, ...
                    if self.current_frame_face_cnt == 0:
                        logging.debug("  / No faces in this frame!!!")
                        # clear list of names and features
                        self.current_frame_face_name_list = []
                    # 6.2.2 / Face cnt increase: 0->1, 0->2, ..., 1->2, ...
                    else:
                        logging.debug("  scene 2.2  Get faces in this frame and do face recognition")
                        self.current_frame_face_name_list = []
                        for i in range(len(faces)):
                            shape = predictor(img_rd, faces[i])
                            self.current_frame_face_feature_list.append(
                                face_reco_model.compute_face_descriptor(img_rd, shape))
                            self.current_frame_face_name_list.append("unknown")

                        # 6.2.2.1 Traversal all the faces in the database
                        for k in range(len(faces)):
                            logging.debug("  For face %d in current frame:", k + 1)
                            self.current_frame_face_centroid_list.append(
                                [int(faces[k].left() + faces[k].right()) / 2,
                                 int(faces[k].top() + faces[k].bottom()) / 2])

                            self.current_frame_face_X_e_distance_list = []

                            # 6.2.2.2  Positions of faces captured
                            self.current_frame_face_position_list.append(tuple(
                                [faces[k].left(), int(faces[k].bottom() + (faces[k].bottom() - faces[k].top()) / 4)]))

                            # 6.2.2.3 
                            # For every faces detected, compare the faces in the database
                            for i in range(len(self.face_features_known_list)):
                                # 
                                if str(self.face_features_known_list[i][0]) != '0.0':
                                    e_distance_tmp = self.return_euclidean_distance(
                                        self.current_frame_face_feature_list[k],
                                        self.face_features_known_list[i])
                                    logging.debug("      with person %d, the e-distance: %f", i + 1, e_distance_tmp)
                                    self.current_frame_face_X_e_distance_list.append(e_distance_tmp)
                                else:
                                    #  person_X
                                    self.current_frame_face_X_e_distance_list.append(999999999)

                            # 6.2.2.4 / Find the one with minimum e distance
                            similar_person_num = self.current_frame_face_X_e_distance_list.index(
                                min(self.current_frame_face_X_e_distance_list))

                            if min(self.current_frame_face_X_e_distance_list) < 0.4:
                                name = self.face_name_known_list[similar_person_num]
                                self.current_frame_face_name_list[k] = name
                                logging.debug("Face recognition result: %s", name)
                            else:
                                logging.debug("  Face recognition result: Unknown person")

                # Check if it's time for a snapshot and one hasn't been taken yet in this interval
                if current_time - self.last_snapshot_time >= self.snapshot_interval and not self.snapshot_taken:
                    self.take_snapshot()
                
                # Draw information on frame
                self.draw_note(img_rd)

                # 8.  'q'  / Press 'q' to exit
                if kk == ord('q'):
                    if self.session_started:
                        self.end_session()
                    break

                self.update_fps()
                cv2.namedWindow("camera", 1)
                cv2.imshow("camera", img_rd)

                logging.debug("Frame ends\n\n")


    def run(self):
        # cap = cv2.VideoCapture("video.mp4")  # Get video stream from video file
        cap = cv2.VideoCapture(0)              # Get video stream from camera
        self.process(cap)

        cap.release()
        cv2.destroyAllWindows()
    


def main():
    # logging.basicConfig(level=logging.DEBUG) # Set log level to 'logging.DEBUG' to print debug info of every frame
    logging.basicConfig(level=logging.INFO)
    Face_Recognizer_con = Face_Recognizer()
    Face_Recognizer_con.run()


if __name__ == '__main__':
    main()