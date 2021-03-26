from CameraCalibration import ArucoCalibrator
from detectron2_util import Detectron2util
import cv2
import numpy as np

def runAThroughTest():
    calib = ArucoCalibrator()
    image_paths_test =[
        "testdata/IMG_20210208_135527.jpg",
        "testdata/IMG_20210208_135532.jpg",
        "testdata/IMG_20210208_135538.jpg",
        "testdata/IMG_20210209_114242.jpg",
        "testdata/IMG_20210209_114246.jpg",
        "testdata/IMG_20210209_114249.jpg",
        "testdata/IMG_20210209_114255.jpg",
        "testdata/IMG_20210209_114300.jpg",
        "testdata/IMG_20210209_114305.jpg",
        "testdata/IMG_20210209_114311.jpg",
        "testdata/IMG_20210209_114318.jpg",
        "testdata/IMG_20210209_114323.jpg"
        ]
    calibration, cameraMatrix, distCoeffs, nView_rvecs, nView_tvecs = calib.getCameraIntrinsicFromImages(image_paths_test)

    print(cameraMatrix)
    print(distCoeffs)

    for i, image_path in enumerate(image_paths_test):
        img = cv2.imread(image_path)
        retval, rvecs, tvecs = calib.getRvecAndTvecFromImgWithCharuco(img, cameraMatrix, distCoeffs)
        if retval:
            retimg = calib.drawWorldBox(img, rvecs, tvecs, cameraMatrix, distCoeffs)
            cv2.imshow('img',retimg)
            cv2.waitKey(0) & 0xff
            cv2.imwrite('temp/test4'+str(i)+'.png', retimg)

def runAVideoTest():
    calib = ArucoCalibrator()
    video_path = "testdata/VID_20210209_183650.mp4"
    calibration, cameraMatrix, distCoeffs, nView_rvecs, nView_tvecs = calib.getCameraIntrinsicFromVideo(video_path)

    # open
    cap = cv2.VideoCapture(video_path)
    frame_rate = int(cap.get(cv2.CAP_PROP_FPS)) # フレームレート(fps)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    # writer
    #fmt = cv2.VideoWriter_fourcc('m', 'p', '4', 'v') # ファイル形式(ここではmp4)
    #writer = cv2.VideoWriter('./temp/outtest.mp4', fmt, frame_rate, size) # ライター作成
    proportion = max(width, height) / 1000.0
    size = (int(width/proportion), int(height/proportion))
    writer = cv2.VideoWriter(
        'temp/outtest4.avi',
        cv2.VideoWriter_fourcc(*'MJPG'),
        frame_rate,
        size)

    while(cap.isOpened()):
        ret, img = cap.read()
        if not ret:
            break

        retval, rvecs, tvecs = calib.getRvecAndTvecFromImgWithCharuco(img, cameraMatrix, distCoeffs)
        if retval:
            img = calib.drawWorldBox(img, rvecs, tvecs, cameraMatrix, distCoeffs)
            #img = cv2.resize(img, (640, 480))
            writer.write(img.astype('uint8'))

            frame_Num = int(cap.get(cv2.CAP_PROP_POS_FRAMES)) # Get the current frame position
            #cap.set(cv2.CAP_PROP_POS_FRAMES, frame_Num + 4) # skip frames for evaluation using different images

    ## Destroy any open CV windows
    cv2.destroyAllWindows()
    writer.release() # ファイルを閉じる

def runH36MTest():
    calib = ArucoCalibrator()
    camera_names_test = ['54138969', '55011271', '58860488', '60457274']
    image_paths_test =[
        "C:\\Users\\User\\Downloads\\learnable-triangulation-pytorch-master\\data\\human36m\\processed\\S9\\Posing-1\\imageSequence\\54138969\\img_001771.jpg",
        "C:\\Users\\User\\Downloads\\learnable-triangulation-pytorch-master\\data\\human36m\\processed\\S9\\Posing-1\\imageSequence\\55011271\\img_001771.jpg",
        "C:\\Users\\User\\Downloads\\learnable-triangulation-pytorch-master\\data\\human36m\\processed\\S9\\Posing-1\\imageSequence\\58860488\\img_001771.jpg",
        "C:\\Users\\User\\Downloads\\learnable-triangulation-pytorch-master\\data\\human36m\\processed\\S9\\Posing-1\\imageSequence\\60457274\\img_001771.jpg"
        ]
    R_test = [
        [[-0.9153617,   0.40180838,  0.02574755],
            [ 0.05154812,  0.18037356, -0.9822465 ],
            [-0.39931902, -0.89778364, -0.18581952]],
        [[ 0.92816836,  0.37215385,  0.00224838],
            [ 0.08166409, -0.1977723,  -0.9768404 ],
            [-0.36309022,  0.9068559,  -0.2139576 ]],
        [[-0.91415495, -0.40277803, -0.04572295],
            [-0.04562341,  0.2143085,  -0.97569996],
            [ 0.4027893 , -0.8898549,  -0.21428728]],
        [[ 0.91415626, -0.40060705,  0.06190599],
            [-0.05641001, -0.2769532,  -0.9592262 ],
            [ 0.40141782,  0.8733905,  -0.27577674]]
        ]
    t_test = [
        [[-346.0509 ],
            [ 546.98083],
            [5474.481  ]],
        [[ 251.4252 ],
            [ 420.94223],
            [5588.196  ]],
        [[ 480.4826 ],
            [ 253.83238],
            [5704.2075 ]],
        [[  51.883537],
            [ 378.4209  ],
            [4406.1494  ]]
        ]
    K_test = [
        [[1.1450494e+03, 0.0000000e+00, 5.1254150e+02],
            [0.0000000e+00, 1.1437811e+03, 5.1545148e+02],
            [0.0000000e+00, 0.0000000e+00, 1.0000000e+00]],
        [[1.1496757e+03, 0.0000000e+00, 5.0884863e+02],
            [0.0000000e+00, 1.1475917e+03, 5.0806491e+02],
            [0.0000000e+00, 0.0000000e+00, 1.0000000e+00]],
        [[1.1491407e+03, 0.0000000e+00, 5.1981586e+02],
            [0.0000000e+00, 1.1487990e+03, 5.0140265e+02],
            [0.0000000e+00, 0.0000000e+00, 1.0000000e+00]],
        [[1.1455114e+03, 0.0000000e+00, 5.1496820e+02],
            [0.0000000e+00, 1.1447739e+03, 5.0188202e+02],
            [0.0000000e+00, 0.0000000e+00, 1.0000000e+00]]
        ]
    dist_test = [
        [-0.20709892,  0.24777518, -0.00142447, -0.0009757,  -0.00307515],
        [-0.19421363,  0.24040854, -0.00274089, -0.00161903,  0.00681998],
        [-0.20833819,  0.255488,   -0.00076,     0.00148439, -0.0024605 ],
        [-0.19838409,  0.21832368, -0.00181336, -0.00058721, -0.00894781]
        ]           

    for i, image_path in enumerate(image_paths_test):
        img = cv2.imread(image_path)

        R_umat = cv2.UMat(np.array(R_test[i]))
        rvecs, _ = cv2.Rodrigues(R_umat)
        tvecs = cv2.UMat(np.array(t_test[i]))
        cameraMatrix = np.array(K_test[i])
        distCoeffs = np.array(dist_test[i])

        img = calib.drawWorldBox(img, rvecs, tvecs, cameraMatrix, distCoeffs, 1000)

        cv2.imshow('img',img)
        cv2.waitKey(0) & 0xff
        cv2.imwrite('temp/test5'+str(i)+'.png', img)

def Dtectron2Test():
    det2u = Detectron2util()
    image_path = "testdata/2.jpg"
    img = cv2.imread(image_path)
    img = det2u.visualizeResult(img)
    cv2.imwrite('temp/detectron2Test.png', img)

runAThroughTest()
#runAVideoTest()
#Dtectron2Test()
