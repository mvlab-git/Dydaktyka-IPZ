import numpy as np, cv2 as cv

# Współrzędne markerów w układzie względem środka ich macierzy
TARGET_COORDS_XY = [
    (-50, 50), #id0
    (50, 50), #id1
    (-50, -50), #id2
    (50, -50)  #id3
]
CAMERA_TO_MARKERS_VECTOR = (0,1600)  # Współrzędne macierzy markerów względem kamery [X,Y]

TARGET_COORDS_XY = np.array(TARGET_COORDS_XY) + CAMERA_TO_MARKERS_VECTOR


# Ścieżka zapisu macierzy transformacji
PATH_MATRICES = 'Camera Calibration/CameraCalibration.npz'

class ArucoDetector():
    def __init__(self):
        # Aruco detection
        self.cArucoDet = cv.aruco.ArucoDetector(cv.aruco.getPredefinedDictionary(cv.aruco.DICT_6X6_50), cv.aruco.DetectorParameters())

    def getArucoCorners(self, aImg: np.ndarray) -> tuple[list[list[int]],np.ndarray]:

        aPreview = aImg.copy()

        corners, ids, _ = self.cArucoDet.detectMarkers(cv.cvtColor(aImg.copy(), cv.COLOR_BGR2GRAY))
        if ids is None: ids = []

        if len(ids)==4:       
                    
            lArucoCenters = [[int(np.mean(cor[0,:,0])),int(np.mean(cor[0,:,1]))] for cor in corners]
            lArucoCenters, ids = [c for _, c in sorted(zip(ids, lArucoCenters))], sorted(ids)

            
            for c,id in zip(lArucoCenters, ids):
                cv.circle(aPreview, (c[0], c[1]), 2, (0, 255, 0), -1)
                cv.putText(aPreview, str(id), (c[0], c[1]), cv.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv.LINE_AA)
            cv.polylines(aPreview, [np.array(lArucoCenters).astype(np.int32)], True, (0, 255, 0), 2)

            cv.line(aPreview, lArucoCenters[0], lArucoCenters[2], (0, 255, 0), 2)
            cv.line(aPreview, lArucoCenters[1], lArucoCenters[3], (0, 255, 0), 2)

        
        else:
            raise Exception(f"Not enough markers found ({len(ids)})!")

        return lArucoCenters, aPreview

class Transformer():

    def __init__(self):
        self.MatrixImageToReal = None
        self.MatrixRealToImage = None

    def saveTransformMatrices(self, sPath: str):
        np.savez(sPath, MatrixImageToReal=self.MatrixImageToReal, MatrixRealToImage=self.MatrixRealToImage)

    def loadTransformMatrices(self, sPath: str):
        with np.load(sPath) as data:
            self.MatrixImageToReal = data['MatrixImageToReal']
            self.MatrixRealToImage = data['MatrixRealToImage']
    
    def getTransformMatrices(self, lInputCoords: list, lTargetCoords: list):
        lInputCoords,lTargetCoords = np.float32(lInputCoords).reshape(-1,2), np.float32(lTargetCoords).reshape(-1,2)
        self.MatrixImageToReal = cv.getPerspectiveTransform(np.float32(lInputCoords), np.float32(lTargetCoords))
        self.MatrixRealToImage = cv.getPerspectiveTransform(np.float32(lTargetCoords), np.float32(lInputCoords))
    
    def transformToReal(self, lPoints: list[list]|np.ndarray) -> np.ndarray:
        lPoints = np.float32(lPoints).reshape(-1,1,2)
        return cv.perspectiveTransform(lPoints, self.MatrixImageToReal).reshape(-1,2)
    
    def transformToImage(self, lPoints: list[list]|np.ndarray) -> np.ndarray:
        lPoints = np.float32(lPoints).reshape(-1,1,2)
        return cv.perspectiveTransform(lPoints, self.MatrixRealToImage).reshape(-1,2)

# +++ Kalibracja +++

def calibrateCamera(aInputImage: np.ndarray, lTargetCoords: list[list[int]], sPathToSave: str):
    cAruco = ArucoDetector()
    cTransformer = Transformer()  

    try:
        lArucoCenters, aPreview = cAruco.getArucoCorners(aInputImage)
        cTransformer.getTransformMatrices(lArucoCenters, lTargetCoords)
        cTransformer.saveTransformMatrices(sPathToSave)
        cv.imwrite(sPathToSave.split('.')[0]+".jpg", aPreview)

    except Exception as e: 
        print(e)
        return False
    
    return True

# +++ Podgląd +++

def showPreviewImageToReal(aInputImage: np.ndarray, lPointsImage: list[list[int|float]], sPathToMatrices: str):

    cTransformer = Transformer()
    cTransformer.loadTransformMatrices(sPathToMatrices)

    lPointsReal = cTransformer.transformToReal(lPointsImage)

    aPreview = aInputImage.copy()
    for p_im, p_r in zip(lPointsImage,lPointsReal):
        cv.circle(aPreview, (p_im[0], p_im[1]), 5, (0, 255, 0), -1)
        cv.putText(aPreview, f"{p_r[0]:.2f},{p_r[1]:.2f}", (p_im[0], p_im[1]), cv.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv.LINE_AA)

    return aPreview


# Initialize and open camera
aInputImage = cv.imread('Camera Calibration/CalibrationInput.png')

bRes = calibrateCamera(aInputImage, TARGET_COORDS_XY, PATH_MATRICES)
if not bRes: raise Exception("Unsuccessful calibration!")

aPreview = showPreviewImageToReal(aInputImage, [[450,250],[600,150],[960,540],[1500,1000]], PATH_MATRICES)

cv.imshow('Preview', aPreview)
cv.waitKey(0)
