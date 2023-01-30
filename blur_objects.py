# Modified from https://github.com/luxonis/depthai-experiments/tree/master/gen2-blur-faces

import blobconverter
import cv2
import depthai as dai
import numpy as np

# These can be used to select which objects to detect/track
labelMap = {
    0: "person",           1: "bicycle",     2: "car",            3: "motorbike",      4: "aeroplane",    5: "bus",            6: "train",
    7: "truck",            8: "boat",        9: "traffic light", 10: "fire hydrant",  11: "stop sign",   12: "parking meter", 13: "bench",
    14: "bird",           15: "cat",        16: "dog",           17: "horse",         18: "sheep",       19: "cow",           20: "elephant",
    21: "bear",           22: "zebra",      23: "giraffe",       24:  "backpack",     25: "umbrella",    26: "handbag",       27: "tie",
    28: "suitcase",       29: "frisbee",    30: "skis",          31: "snowboard",     32: "sports ball", 33: "kite",          34: "baseball bat",
    35: "baseball glove", 36: "skateboard", 37: "surfboard",     38: "tennis racket", 39: "bottle",      40: "wine glass",    41: "cup",
    42: "fork",           43: "knife",      44: "spoon",         45: "bowl",          46: "banana",      47: "apple",         48: "sandwich",
    49: "orange",         50: "broccoli",   51: "carrot",        52: "hot dog",       53: "pizza",       54: "donut",         55: "cake",
    56: "chair",          57: "sofa",       58: "pottedplant",   59: "bed",           60: "diningtable", 61: "toilet",        62: "tvmonitor",
    63: "laptop",         64: "mouse",      65: "remote",        66: "keyboard",      67: "cell phone",  68: "microwave",     69:  "oven",
    70: "toaster",        71: "sink",       72: "refrigerator",  73: "book",          74: "clock",       75: "vase",          76:  "scissors",
    77: "teddy bear",     78: "hair drier", 79: "toothbrush"
}

class HostSync:
    def __init__(self):
        self.arrays = {}
    def add_msg(self, name, msg):
        if not name in self.arrays:
            self.arrays[name] = []
        self.arrays[name].append(msg)
    def get_msgs(self, seq):
        ret = {}
        for name, arr in self.arrays.items():
            for i, msg in enumerate(arr):
                if msg.getSequenceNum() == seq:
                    ret[name] = msg
                    self.arrays[name] = arr[i:]
                    break
        return ret

def create_pipeline():
    print("Creating pipeline...")
    pipeline = dai.Pipeline()

    # ColorCamera
    print("Creating Color Camera...")
    cam = pipeline.create(dai.node.ColorCamera)
    cam.setPreviewSize(416,416)
    cam.setResolution(dai.ColorCameraProperties.SensorResolution.THE_1080_P)
    cam.setInterleaved(False)
    cam.setColorOrder(dai.ColorCameraProperties.ColorOrder.BGR)
    cam.setFps(40)
    cam.setVideoSize(1080,1080)

    cam_xout = pipeline.create(dai.node.XLinkOut)
    cam_xout.setStreamName("frame")
    cam.video.link(cam_xout.input)

    # NeuralNetwork
    print("Creating Object Detection Neural Network...")
    object_det_nn = pipeline.create(dai.node.YoloDetectionNetwork)
    object_det_nn.setBlobPath(blobconverter.from_zoo(
            name="yolo-v3-tiny-tf", #face-detection-retail-0004
            shaves=6,
        ))
    object_det_nn.setConfidenceThreshold(0.5)
    object_det_nn.input.setBlocking(False)
    object_det_nn.setNumClasses(80)
    object_det_nn.setCoordinateSize(4)
    object_det_nn.setAnchors([10, 14, 23, 27, 37, 58, 81, 82, 135, 169, 344, 319])
    object_det_nn.setAnchorMasks({"side26": [1, 2, 3], "side13": [3, 4, 5]})
    object_det_nn.setIouThreshold(0.5)
    object_det_nn.setNumInferenceThreads(2)

    # Link Object ImageManip -> Object detection NN node
    cam.preview.link(object_det_nn.input)

    objectTracker = pipeline.create(dai.node.ObjectTracker)
    objectTracker.setDetectionLabelsToTrack([62, 63, 67])  # track phone, tv, laptop
    # possible tracking types: ZERO_TERM_COLOR_HISTOGRAM, ZERO_TERM_IMAGELESS, SHORT_TERM_IMAGELESS, SHORT_TERM_KCF
    objectTracker.setTrackerType(dai.TrackerType.SHORT_TERM_IMAGELESS)
    # take the smallest ID when new object is tracked, possible options: SMALLEST_ID, UNIQUE_ID
    objectTracker.setTrackerIdAssignmentPolicy(dai.TrackerIdAssignmentPolicy.SMALLEST_ID)
    objectTracker.setMaxObjectsToTrack(20)

    # Linking
    object_det_nn.passthrough.link(objectTracker.inputDetectionFrame)
    object_det_nn.passthrough.link(objectTracker.inputTrackerFrame)
    object_det_nn.out.link(objectTracker.inputDetections)
    # Send object detections to the host (for bounding boxes)

    pass_xout = pipeline.create(dai.node.XLinkOut)
    pass_xout.setStreamName("pass_out")
    objectTracker.passthroughTrackerFrame.link(pass_xout.input)

    tracklets_xout = pipeline.create(dai.node.XLinkOut)
    tracklets_xout.setStreamName("tracklets")
    objectTracker.out.link(tracklets_xout.input)
    print("Pipeline created.")
    return pipeline

with dai.Device(create_pipeline()) as device:
    frame_q = device.getOutputQueue("frame")
    tracklets_q = device.getOutputQueue("tracklets")
    pass_q = device.getOutputQueue("pass_out")
    sync=HostSync()
    while True:
        sync.add_msg("color", frame_q.get())

        # Using tracklets instead of ImgDetections in case NN inaccuratelly detected object, so blur
        # will still happen on all tracklets (even LOST ones)
        nn_in = tracklets_q.tryGet()
        if nn_in is not None:
            seq = pass_q.get().getSequenceNum()
            msgs = sync.get_msgs(seq)

            if not 'color' in msgs: continue
            frame = msgs["color"].getCvFrame()

            for t in nn_in.tracklets:
                # Expand the bounding box a bit so it fits the object nicely (also convering hair/chin/beard)
                t.roi.x -= t.roi.width / 10
                t.roi.width = t.roi.width * 1.2
                t.roi.y -= t.roi.height / 7
                t.roi.height = t.roi.height * 1.2

                roi = t.roi.denormalize(frame.shape[1], frame.shape[0])
                bbox = [int(roi.topLeft().x), int(roi.topLeft().y), int(roi.bottomRight().x), int(roi.bottomRight().y)]
                #print(bbox)

                object = frame[bbox[1]:bbox[3], bbox[0]:bbox[2]]
                fh, fw, fc = object.shape
                frame_h, frame_w, frame_c = frame.shape

                # Create blur mask around the object
                mask = np.zeros((frame_h, frame_w), np.uint8)
                #polygon = cv2.ellipse2Poly((bbox[0] + int(fw /2), bbox[1] + int(fh/2)), (int(fw /2), int(fh/2)), 0,0,360,delta=1)
                #cv2.fillConvexPoly(mask, polygon, 255)
                # mask out the bounding box
                mask[bbox[1]:bbox[3], bbox[0]:bbox[2]] = 255

                frame_copy = frame.copy()
                frame_copy = cv2.blur(frame_copy, (80, 80))
                object_extracted = cv2.bitwise_and(frame_copy, frame_copy, mask=mask)
                background_mask = cv2.bitwise_not(mask)
                background = cv2.bitwise_and(frame, frame, mask=background_mask)
                # Blur the object
                frame = cv2.add(background, object_extracted)

            cv2.imshow("Frame", cv2.resize(frame, (900,900)))

        if cv2.waitKey(1) == ord('q'):
            break


