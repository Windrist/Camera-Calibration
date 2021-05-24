import os
import sys
import cv2
import numpy as np

# Config Variables - Enter their values according to your Checkerboard
no_of_columns = 9   # Number of columns of your Checkerboard
no_of_rows = 7 # Number of rows of your Checkerboard
square_size = 20 # Size of square on the Checkerboard in mm

class detectConfiguration(object):
    def __init__(self) -> None:
        super().__init__()

        self.confirmedImagesCounter = 0 # How many images are confirmed by user so far
        self.currentCorners = None # Array of last detected corners
        self.homographies = [] # List of homographies of each captured image
        self.capturedImagePoints = {} # Dictionary of 2D points on captured image
        self.objectPoints = {} # Dictionary of 3D points on chessboard
        self.points_in_row, self.points_in_column = no_of_rows, no_of_columns # Number of rows and columns of your Checkerboard
        x, y = square_size, square_size   # Size of square on the Checkerboard in mm

        # Object Points in 3D
        self.capturedObjectPointsLR = [[i*x, j*y, 0] for i in range(self.points_in_row, 0, -1) for j in range(self.points_in_column, 0, -1)]
        self.capturedObjectPointsRL = list(reversed(self.capturedObjectPointsLR))

    def processImage(self):
        for file in self.listFiles:
            """ Captures frame from webcam & tries to detect corners on chess board """
            print('Processing: ' + file)
            frame = cv2.imread('./input/' + file)
            resize_frame = cv2.resize(frame, (1280, 720), interpolation=cv2.INTER_AREA)
            cornersDetected, corners, imageWithCorners = self.detectCorners(resize_frame) # Detect corners on chess board
            if cornersDetected: # If corners detected successfully
                self.currentCorners = corners
                while True:
                    cv2.imshow("Corners Detected", imageWithCorners)
                    if cv2.waitKey(1000):
                        break
                
            self.confirmedImagesCounter += 1

            self.capturedImagePoints[self.confirmedImagesCounter] = self.currentCorners
            if self.currentCorners[0,0,0]<self.currentCorners[-1,0,0]:
                capturedObjectPoints=self.capturedObjectPointsLR
            else:capturedObjectPoints=self.capturedObjectPointsRL
            self.objectPoints[self.confirmedImagesCounter] = capturedObjectPoints

            h = self.computeHomography(self.currentCorners, capturedObjectPoints)
            self.homographies.append(h)

    def main(self):
        self.listFiles = os.listdir('./input/')
        if len(self.listFiles) < 3:
            rem = 3 - len(self.listFiles)
            print('\33[1m' + '\33[31m' + 'Warning! The number of captured photos should be at least 3. Please take ' + str(rem) + ' more photos!' + '\33[0m')
        else:
            self.processImage()

            M = self.buildMmatrix()
            b = self.getMinimumEigenVector(M)
            v0, lamda, alpha, betta, gamma, u0, A = self.calculateIntrinsicParam(b)
            Rt = self.calculateExtrinsicParam(A)

            print('Done!\nCheck intrinsic.npy & extrinsic.npy')

    def detectCorners(self, image):
        criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        ret, corners = cv2.findChessboardCorners(gray, (no_of_columns,no_of_rows), cv2.CALIB_CB_FAST_CHECK)
        if ret:
            cv2.cornerSubPix(gray, corners, (11,11), (-1,-1), criteria)
            cv2.drawChessboardCorners(image, (no_of_columns,no_of_rows), corners, ret)
        return ret, corners, image

    def computeHomography(self, points2D, points3D):
        U = self.buildUmatrix(points2D, points3D)
        return self.getMinimumEigenVector(U)

    def buildUmatrix(self, points2D, points3D):
        rows = self.points_in_row * self.points_in_column * 2
        U = np.zeros((rows, 9))
        for i in range(len(points2D)):
            U[i*2, 0] = points3D[i][0] # Px
            U[i*2, 1] = points3D[i][1] # Py
            U[i*2, 2] = 1
            U[i*2, 3:6] = 0
            U[i*2, 6:9] = U[i*2, 0:3] * -points2D[i, 0, 0]

            U[i*2+1, 0:3] = 0
            U[i*2+1, 3:6] = U[i*2, 0:3]
            U[i*2+1, 6:9] = U[i*2, 0:3] * -points2D[i, 0, 1]
        return U

    def calculateV(self,h1,h2):
        v = np.zeros((6,1))
        v[0,0] = h1[0] * h2[0]
        v[1,0] = h1[0] * h2[1] + h1[1] * h2[0]
        v[2,0] = h1[1] * h2[1]
        v[3,0] = h1[2] * h2[0] + h1[0] * h2[2]
        v[4,0] = h1[2] * h2[1] + h1[1] * h2[2]
        v[5,0] = h1[2] * h2[2]
        return v

    def buildMmatrix(self): # Build the matrix made by homographies to calculate B
        M = np.zeros((self.confirmedImagesCounter*2, 6))
        for i in range(len(self.homographies)):
            h1 = self.homographies[i][::3]
            h2 = self.homographies[i][1::3]
            v12 = self.calculateV(h1, h2) # 6X1
            v11 = self.calculateV(h1, h1) # 6X1
            v22 = self.calculateV(h2, h2) # 6X1
            M[2*i, :] = v12.T # 1X6
            M[2*i + 1, :] = (v11 - v22).T # 1X6
        return M

    def calculateIntrinsicParam(self, b):
        (B11, B12, B22, B13, B23, B33) = b
        v0 = (B12*B13 - B11*B23) / (B11*B22 - B12**2)
        lamda = B33 - (B13**2 + v0 * (B12*B13 - B11*B23)) / B11
        alpha = np.sqrt(lamda / B11)
        betta = np.sqrt((lamda * B11) / (B11*B22 - B12**2) )
        gamma = (-B12 * betta * alpha ** 2) / lamda
        u0 = (gamma * v0 / betta) - (B13 * alpha ** 2 / lamda)
        A = np.array([[alpha, gamma, u0], [0, betta, v0], [0,0,1]])

        # Write intrinsic parameters to file
        if not os.path.exists('./output'):
            os.mkdir('./output')    # Make output folder if not exists
        np.save('./output/intrinsic.npy', A)

        return v0, lamda, alpha, betta, gamma, u0, A

    def calculateExtrinsicParam(self, A):
        h1=self.homographies[0][::3] # 1st column of 1st image homography
        h2=self.homographies[0][1::3] # 2nd column of 1st image homography
        h3=self.homographies[0][2::3] # 3rd column of 1st image homography
        A_inv = np.linalg.inv(A)
        Ah1 = np.dot(A_inv, h2)
        lamda = 1 / np.sqrt(np.dot(Ah1, Ah1))
        r1 = lamda * np.dot(A_inv, h1) # 1st column or rotation matrix
        r2 = lamda * np.dot(A_inv, h2) # 2nd column or rotation matrix
        r3 = np.cross(r1, r2) # 3rd column or rotation matrix
        t = lamda * np.dot(A_inv, h3) # Translation Vector
        Rt = np.array([r1.T, r2.T, r3.T, t.T]).T

        # Write extrinsic parameters to file
        if not os.path.exists('./output'):
                    os.mkdir('./output') # Make output folder if not exists
        np.save('./output/extrinsic.npy', Rt)

        return Rt

    def calculateAllExtrinsicParam(self, A, lamda):
        Rts = []
        A_inv = np.linalg.inv(A)
        for homography in self.homographies:
            h1=homography[::3]
            h2=homography[1::3]
            h3=homography[2::3]
            r1 = lamda * np.dot(A_inv, h1)
            r2 = lamda * np.dot(A_inv, h2)
            r3 = np.cross(r1, r2)
            t = lamda * np.dot(A_inv, h3)
            Rt = np.array([r1.T, r2.T, r3.T, t.T]).T
            Rts.append(Rt)
        return Rts

    def getMinimumEigenVector(self, U):
        """ Return eigen vector of square matrix U with the minimum eigen value """
        P = np.dot(U.T, U)
        w,v = np.linalg.eig(P)
        i = np.where(w == np.min(w))
        e1 = v[:, i[0][0]]
        return e1

if __name__ == '__main__':
    BT = detectConfiguration()
    BT.main()