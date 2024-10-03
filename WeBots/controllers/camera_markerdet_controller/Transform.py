import numpy as np, cv2 as cv

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
