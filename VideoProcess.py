import cv2
import math

class VideoProcess():
    def __init__(self, videoFilePath):
        self.isPoseCached = []
        self.cachedPose = []

        self.cap = cv2.VideoCapture(videoPath)
        self.framerate = self.cap.get(cv2.CAP_PROP_FPS)

    def getPoseByTime(self, t: float):
        frame = t / self.framerate
        frameFloor = math.floor(frame)
        frameCeil = math.ceil(frame)
        if frameFloor == frame: # frameÇÇøÇÂÇ§Ç«éwíËÇµÇΩèÍçá
            return self.getPoseByFrame(frameFloor)
        else: # frameä‘ÇÃéûçèÇÃèÍçáÇÕÅAê¸å`ï‚ä‘Ç∑ÇÈ
            poseFloor = self.getPoseByFrame(frameFloor)
            poseCeil = self.getPoseByFrame(frameCeil)
            return self.avgPoses(poseFloor, poseCeil, frame - frameFloor)

    def avgPoses(self, pose1, pose2, ratio2: float):
        ratio1 = 1 - ratio2
        return pose1 * ratio1 + pose2 * ratio2

    def getPoseByFrame(self, frameNum: int):
        if self.isPoseCached[frameNum]:
            return self.cachedPose[frameNum]
        else:
            self.cachedPose[frameNum] = self. calcPose[frameNum]
            self.isPoseCached[frameNum] = True
            return self.cachedPose[frameNum]

    def calcPose(self, frameNum: int):
        return 1