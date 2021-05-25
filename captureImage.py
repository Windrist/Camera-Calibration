import time
import cv2

# Config Variables - Enter their values according to your Camera
camera_config = 0 # Camera Configuration (Try 0 to 9 or more)
resolution_X = 1280 # Resolution of your camera in the horizontal
resolution_Y = 720 # Resolution of your camera in the vertical
output_folder = 'input' # Change folder according to your image (Checkerboard on input, Object on raw)

if __name__ == "__main__":
    # Camera configuration
    cap = cv2.VideoCapture(camera_config)
    cap.set(3, resolution_X)
    cap.set(4, resolution_Y)
    count = 0
    
    while True:
        ret, image = cap.read()
        cv2.imshow("Capture", image)
        key =  cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
        elif key == ord('s'):
            cv2.imwrite('./' + output_folder + '/{}.jpg'.format(count), image)
            count += 1
    
    cap.release()
    cv2.destroyAllWindows()