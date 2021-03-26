# System information:
# - Linux Mint 18.1 Cinnamon 64-bit
# - Python 2.7 with OpenCV 3.2.0

import numpy as np
import cv2
from cv2 import aruco
import glob
import time

class ArucoCalibrator():
    def __init__(self, row = 4, column = 3, arucoDict = aruco.DICT_5X5_1000, squareLength=0.055, markerLength=0.04125):
        # ChAruco board variables
        CHARUCOBOARD_ROWCOUNT = row
        CHARUCOBOARD_COLCOUNT = column 
        self.ARUCO_DICT = aruco.Dictionary_get(arucoDict)
        self.EXPECTED_RESPONSE = CHARUCOBOARD_ROWCOUNT * CHARUCOBOARD_COLCOUNT / 2
        self.SQUARE_LENGTH = squareLength

        # Create constants to be passed into OpenCV and Aruco methods
        self.CHARUCO_BOARD = aruco.CharucoBoard_create(
                squaresX=CHARUCOBOARD_COLCOUNT,
                squaresY=CHARUCOBOARD_ROWCOUNT,
                squareLength=squareLength,
                markerLength=markerLength,
                dictionary=self.ARUCO_DICT)

        ## To print out the calibration image
        #imboard = CHARUCO_BOARD.draw((2598, 2598))
        #cv2.imwrite("chessboard4x3.tiff", imboard)


    def getCharucoData(self, img):
        # Grayscale the image
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        # Find aruco markers in the query image
        corners, ids, _ = aruco.detectMarkers(
                image=gray,
                dictionary=self.ARUCO_DICT)

        if len(corners) > 0:
            # Get charuco corners and ids from detected aruco markers
            #response, charuco_corners, charuco_ids = aruco.interpolateCornersCharuco(
            return aruco.interpolateCornersCharuco(
                    markerCorners=corners,
                    markerIds=ids,
                    image=gray,
                    board=self.CHARUCO_BOARD)

            ## If a Charuco board was found, let's collect image/corner points
            ## Requiring the exact number of AR markers
            #if response == minNumberOfData:
                ## Outline the aruco markers found in our query image
                #img = aruco.drawDetectedMarkers(
                #        image=img, 
                #        corners=corners)
                ## Draw the Charuco board we've detected to show our calibrator the board was properly detected
                #img = aruco.drawDetectedCornersCharuco(
                #        image=img,
                #        charucoCorners=charuco_corners,
                #        charucoIds=charuco_ids)
                ## Reproportion the image, maxing width or height at 1000
                #proportion = max(img.shape) / 1000.0
                #img = cv2.resize(img, (int(img.shape[1]/proportion), int(img.shape[0]/proportion)))
                ## Pause to display each image, waiting for key press
                #cv2.imshow('Charuco board', img)
                #cv2.waitKey(0)
        else:
            return 0, None, None

    def calcCameraIntrinsicFromCharucoData(self, corners_all, ids_all, image_size):
        # Make sure at least one image was found
        if len(corners_all) < 1:
            # Calibration failed because there were no images, warn the user
            print("Calibration was unsuccessful. No images of charucoboards were found.")
            # Exit for failure
            return False, None, None, None, None
        if len(corners_all) == ids_all:
            print("lengths of the two arrays must be the same.")
            # Exit for failure
            return False, None, None, None, None

        print('Caliculating with {} frames'.format(len(corners_all)))
        start = time.time()
        # Now that we've seen all of our images, perform the camera calibration
        # based on the set of points we've discovered
        calibration, cameraMatrix, distCoeffs, nView_rvecs, nView_tvecs = aruco.calibrateCameraCharuco(
                charucoCorners=corners_all,
                charucoIds=ids_all,
                board=self.CHARUCO_BOARD,
                imageSize=image_size,
                cameraMatrix=None,
                distCoeffs=None)
        t = time.time() - start
        print('Caliculatiion took {} seconds'.format(t))
        return calibration, cameraMatrix, distCoeffs, nView_rvecs, nView_tvecs

    ### Public ###
    def getCameraIntrinsicFromVideo(self, videoPath, skip_frame = 4, max_frames_for_calc = 32):
        corners_all = []
        ids_all = []
        image_size = None
        cap = cv2.VideoCapture(videoPath)
        while(cap.isOpened()):
            ret, img = cap.read()
            if not ret:
                break

            responce, charuco_corners, charuco_ids = self.getCharucoData(img)
            if responce >= self.EXPECTED_RESPONSE:
                corners_all.append(charuco_corners)
                ids_all.append(charuco_ids)
                # If our image size is unknown, set it now
                if not image_size:
                    image_size = img.shape[1::-1] # HxWxC -> WxH
                
                if len(corners_all) >= max_frames_for_calc:
                    break
            
                frame_Num = int(cap.get(cv2.CAP_PROP_POS_FRAMES)) # Get the current frame position
                cap.set(cv2.CAP_PROP_POS_FRAMES, frame_Num + skip_frame) # skip frames for evaluation using different images

        ## Destroy any open CV windows
        #cv2.destroyAllWindows()

        return self.calcCameraIntrinsicFromCharucoData(corners_all, ids_all, image_size)

    ### Public ###
    def getCameraIntrinsicFromImages(self, imageNameList):
        corners_all = []
        ids_all = []
        image_size = None
        for imageName in imageNameList:
            img = cv2.imread(imageName)

            responce, charuco_corners, charuco_ids = self.getCharucoData(img)
            if responce >= self.EXPECTED_RESPONSE:
                corners_all.append(charuco_corners)
                ids_all.append(charuco_ids)
                # If our image size is unknown, set it now
                if not image_size:
                    image_size = img.shape[1::-1] # HxWxC -> WxH
        ## Destroy any open CV windows
        #cv2.destroyAllWindows()

        return self.calcCameraIntrinsicFromCharucoData(corners_all, ids_all, image_size)

    def getObjPntOfAruco(self, id):
        x = id % 2 #* self.SQUARE_LENGTH
        y = (id - x) / 2 #* self.SQUARE_LENGTH
        return [x, y, 0]

    ### Public ###
    def getRmatAndTvecFromImgWithCharuco(self, img, cameraMatrix, distCoeffs):
        retval, rvecs, tvecs = self.getRvecAndTvecFromImgWithCharuco(img, cameraMatrix, distCoeffs)
        if retval:
            r_mat, jacob = cv2.Rodrigues(rvecs)
            return retval, r_mat.get(), tvecs.get()
        else:
            return retval, None, None

    def getRvecAndTvecFromImgWithCharuco(self, img, cameraMatrix, distCoeffs, minResponse = 3):
        response, charuco_corners, charuco_ids = self.getCharucoData(img)
        # Requiring at least 4 squares
        if response > minResponse:
            objPnt = []
            imgPnt = []
            for corner, id in zip(charuco_corners, charuco_ids):
                objPnt.append(self.getObjPntOfAruco(id[0]))
                imgPnt.append(corner[0])

            objPntUMat = cv2.UMat(np.array(objPnt))
            imgPntUMat = cv2.UMat(np.array(imgPnt))
            
            if not isinstance(cameraMatrix, cv2.UMat):
                cameraMatrix = cv2.UMat(np.array(cameraMatrix))
            if not isinstance(distCoeffs, cv2.UMat):
                distCoeffs = cv2.UMat(np.array(distCoeffs))

            return cv2.solvePnP(objPntUMat, imgPntUMat, cameraMatrix, distCoeffs) #retval, rvecs, tvecs = 
        else:
            return False, None, None

    def drawWorldBox(self, img, rvecs, tvecs, cameraMatrix, distCoeffs, scale = 1):
        img = self.drawBox(img, rvecs, tvecs, cameraMatrix, distCoeffs,  0, 0, 0, (0,0,0),   scale)
        img = self.drawBox(img, rvecs, tvecs, cameraMatrix, distCoeffs,  2, 0, 0, (255,0,0), scale)
        img = self.drawBox(img, rvecs, tvecs, cameraMatrix, distCoeffs,  4, 0, 0, (255,0,0), scale)
        img = self.drawBox(img, rvecs, tvecs, cameraMatrix, distCoeffs,  6, 0, 0, (255,0,0), scale)
        img = self.drawBox(img, rvecs, tvecs, cameraMatrix, distCoeffs,  8, 0, 0, (255,0,0), scale)
        img = self.drawBox(img, rvecs, tvecs, cameraMatrix, distCoeffs, 10, 0, 0, (255,255,255), scale)
        img = self.drawBox(img, rvecs, tvecs, cameraMatrix, distCoeffs, 12, 0, 0, (255,0,0), scale)
        img = self.drawBox(img, rvecs, tvecs, cameraMatrix, distCoeffs, 14, 0, 0, (255,0,0), scale)
        img = self.drawBox(img, rvecs, tvecs, cameraMatrix, distCoeffs, 16, 0, 0, (255,0,0), scale)
        img = self.drawBox(img, rvecs, tvecs, cameraMatrix, distCoeffs, 18, 0, 0, (255,0,0), scale)
        img = self.drawBox(img, rvecs, tvecs, cameraMatrix, distCoeffs, 20, 0, 0, (255,255,255), scale)
        img = self.drawBox(img, rvecs, tvecs, cameraMatrix, distCoeffs, 0,  2, 0, (0,255,0), scale)
        img = self.drawBox(img, rvecs, tvecs, cameraMatrix, distCoeffs, 0,  4, 0, (0,255,0), scale)
        img = self.drawBox(img, rvecs, tvecs, cameraMatrix, distCoeffs, 0,  6, 0, (0,255,0), scale)
        img = self.drawBox(img, rvecs, tvecs, cameraMatrix, distCoeffs, 0,  8, 0, (0,255,0), scale)
        img = self.drawBox(img, rvecs, tvecs, cameraMatrix, distCoeffs, 0, 10, 0, (255,255,255), scale)
        img = self.drawBox(img, rvecs, tvecs, cameraMatrix, distCoeffs, 0, 12, 0, (0,255,0), scale)
        img = self.drawBox(img, rvecs, tvecs, cameraMatrix, distCoeffs, 0, 14, 0, (0,255,0), scale)
        img = self.drawBox(img, rvecs, tvecs, cameraMatrix, distCoeffs, 0, 16, 0, (0,255,0), scale)
        img = self.drawBox(img, rvecs, tvecs, cameraMatrix, distCoeffs, 0, 18, 0, (0,255,0), scale)
        img = self.drawBox(img, rvecs, tvecs, cameraMatrix, distCoeffs, 0, 20, 0, (255,255,255), scale)
        img = self.drawBox(img, rvecs, tvecs, cameraMatrix, distCoeffs, 0, 0,  2, (0,0,255), scale)
        img = self.drawBox(img, rvecs, tvecs, cameraMatrix, distCoeffs, 0, 0,  4, (0,0,255), scale)
        img = self.drawBox(img, rvecs, tvecs, cameraMatrix, distCoeffs, 0, 0,  6, (0,0,255), scale)
        img = self.drawBox(img, rvecs, tvecs, cameraMatrix, distCoeffs, 0, 0,  8, (0,0,255), scale)
        img = self.drawBox(img, rvecs, tvecs, cameraMatrix, distCoeffs, 0, 0, 10, (255,255,255), scale)
        img = self.drawBox(img, rvecs, tvecs, cameraMatrix, distCoeffs, 0, 0, 12, (0,0,255), scale)
        img = self.drawBox(img, rvecs, tvecs, cameraMatrix, distCoeffs, 0, 0, 14, (0,0,255), scale)
        img = self.drawBox(img, rvecs, tvecs, cameraMatrix, distCoeffs, 0, 0, 16, (0,0,255), scale)
        img = self.drawBox(img, rvecs, tvecs, cameraMatrix, distCoeffs, 0, 0, 18, (0,0,255), scale)
        img = self.drawBox(img, rvecs, tvecs, cameraMatrix, distCoeffs, 0, 0, 20, (255,255,255), scale)

        # Reproportion the image, maxing width or height at 1000
        proportion = max(img.shape) / 1000.0
        img = cv2.resize(img, (int(img.shape[1]/proportion), int(img.shape[0]/proportion)))

        return img

    def drawBox(self, img, rvecs, tvecs, cameraMatrix, distCoeffs, ox = 0, oy = 0, oz = 0, color = (0,0,0), scale = 1, length = 1):
        objPts = np.float32([[ox,oy,oz], [ox,oy+length,oz], [ox+length,oy+length,oz], [ox+length,oy,oz],
                           [ox,oy,oz+length],[ox,oy+length,oz+length],[ox+length,oy+length,oz+length],[ox+length,oy,oz+length] ])
        objPts = objPts * scale
        imgPts, jac = cv2.projectPoints(objPts, rvecs, tvecs, cameraMatrix, distCoeffs)
        imgpts = np.int32(imgPts.get()).reshape(-1,2)

        # draw ground floor in green
        img = cv2.drawContours(img, [imgpts[:4]],-1,color,3)

        # draw pillars in blue color
        for i,j in zip(range(4),range(4,8)):
            img = cv2.line(img, tuple(imgpts[i]), tuple(imgpts[j]),color,3)

        # draw top layer in red color
        img = cv2.drawContours(img, [imgpts[4:]],-1,color,3)
        return img
