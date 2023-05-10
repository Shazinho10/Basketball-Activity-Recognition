import numpy as np
import argparse
from trackers.byte_tracker import BYTETracker
# from memory import Memory
# from helper import Helper as hlp
import time
import datetime
# from objects import Objects as obj


def make_parser():
    parser = argparse.ArgumentParser("ByteTrack Demo!")
    # tracking args
    parser.add_argument("--track_thresh", type=float, default=0.3, help="tracking confidence threshold")
    parser.add_argument("--track_buffer", type=int, default=30, help="the frames for keep lost tracks")
    parser.add_argument("--match_thresh", type=int, default=0.9, help="matching threshold for tracking")
    parser.add_argument('--min-box-area', type=float, default=1, help='filter out tiny boxes')
    parser.add_argument("--mot20", dest="mot20", default=False, action="store_true", help="test mot20.")
    return parser

class Adlytic_YoloX_CPP_ByteTrack():
    def __init__(self, in_img_size=(640, 640), out_img_size=(720, 1280)):
        args = make_parser().parse_args()
        # self.stream_id      = stream_id
        # self.data = self.memory.get_stream_data(stream_id)  
        self.byte_tracker   = BYTETracker(args)
        self.in_img_size    = in_img_size
        self.out_img_size   = out_img_size


    def to_tlbr(self, tlwh):
        tlbr = np.empty(4)
        xmin = float(tlwh[0])
        ymin = float(tlwh[1])
        tlbr[0] = int(round(xmin, 0))
        tlbr[1] = int(round(ymin, 0))
        tlbr[2] = int(round(xmin + float(tlwh[2]) - 1., 0))
        tlbr[3] = int(round(ymin + float(tlwh[3]) - 1., 0))
        return tlbr


    def process_detections(self, detections):
        processed_detections = []
        for detection in detections:
            tlwh = detection[:4]
            class_id= detection[4]
            score = detection[5]

            # print("DET", class_id, score)

            # tlbr = self.to_tlbr(tlwh)
            tlbr = tlwh
            tlbr = np.append(tlbr, score)
            tlbr = np.append(tlbr, class_id)
            processed_detections.append(tlbr)

        processed_detections = np.array(processed_detections)
        return processed_detections


    def track(self, detections):
        #t1=time.time()
        processed_detections = self.process_detections(detections=detections)
        if len(processed_detections):
            track_bbs_ids = self.byte_tracker.update(processed_detections, self.in_img_size, self.out_img_size)
        else:
            track_bbs_ids=[]

        return track_bbs_ids
        #print("total time:", time.time()- t1)

    # def write_person_objects(self, tracked_people):
    #     person_objects = dict()
    #     for track in tracked_people:
    #         identity = track.track_id
    #         time_now = str(datetime.datetime.now()).replace(' ', 'T') + 'Z'
    #         person_info = obj.create_person_object(id=identity, T=time_now, MK=-1, EMP=0)
    #         person_objects[identity] = person_info

    #     self.data[gnrl.PERSON_OBJECTS] = person_objects

    def apply(self, detections):
        # detections = self.data[detections]
        tracked_people  = self.track(detections)
        #print("Tracker Time: ", time.time() - t0)
        # print("TRACKED PEOPLE", tracked_people)
        # self.data[gnrl.TRACKED_OBJECTS] = tracked_people
        # print(detections)
        # print(tracked_people)
        # bboxes, class_ids =  hlp.tpeople_to_dict(tracked_people)
        # print(bboxes, class_ids)
        # self.data[gnrl.TRACKED_PEOPLE] = bboxes
        # self.data[gnrl.CLASS_IDS] = class_ids

        # self.write_person_objects(tracked_people)
        return tracked_people
      