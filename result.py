import cv2
import numpy as np

class calculateDistance(object):
    def __init__(self) -> None:
        super().__init__()

        self.pointd1 = []
        self.pointd2 = []
        self.interrupt = False

        self.frameId = 0

        self.intrinsics = np.load('./output/intrinsic.npy')
        self.extrinsics = np.load('./output/extrinsic.npy')

    def getMouseClicks(self, event, x, y, flags, params):
        if event == cv2.EVENT_LBUTTONDOWN:
            print('Point clicked: {}, {}'.format(x, y))
            if not self.pointd1:
                self.pointd1 = [x, y]
            else:
                self.pointd2 = [x, y]
                dx = self.pointd1[0] - self.pointd2[0]
                dy = self.pointd1[1] - self.pointd2[1]
                dist = np.sqrt(dx**2 + dy**2)
                print('\33[1m' + '\33[32m' + 'Distance in Pixels = {0:2.2f}'.format(dist) + '\33[0m')
                self.interrupt = True

    def main(self):
        cv2.namedWindow("Get Points")
        cv2.setMouseCallback('Get Points', self.getMouseClicks)
        while True:
            image = cv2.imread('./raw/origin.jpg')
            resize_image = cv2.resize(image, (1280, 720), interpolation=cv2.INTER_AREA)
            cv2.imshow("Get Points", resize_image)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break
            elif self.interrupt:
                break
        cv2.destroyAllWindows()
        while True:
            if self.pointd1 and self.pointd2 and self.interrupt:
                cv2.line(resize_image, tuple(self.pointd1),tuple(self.pointd2), (0, 0, 255), 2)
                cv2.imshow('Distance', resize_image)
                aux = self.distanceBetweenTwoPixels()
                dist = np.sqrt(aux[0]**2 + aux[1]**2)
                print('\33[1m' + '\33[34m' + 'Calculated distance in Size (mm) = {}'.format(dist) + '\33[0m')
                self.interrupt = False
            if cv2.waitKey(0):
                break

    def realDistanceCalculator(self, x, y):
        pseudo_inv_extrinsics = np.linalg.pinv(self.extrinsics)
        intrinsics_inv = np.linalg.inv(self.intrinsics)
        pixels_matrix = np.array((x, y, 1))
        ans = np.matmul(intrinsics_inv, pixels_matrix)
        ans = np.matmul(pseudo_inv_extrinsics, ans)
        ans /= ans[-1]
        return ans

    def distanceBetweenTwoPixels(self):
        p1 = self.realDistanceCalculator(self.pointd1[0], self.pointd2[1])
        p2 = self.realDistanceCalculator(self.pointd2[0], self.pointd2[1])
        aux = p2 - p1
        self.pointd1.clear()
        self.pointd2.clear()
        return aux

if __name__ == '__main__':
    BT = calculateDistance()
    BT.main()