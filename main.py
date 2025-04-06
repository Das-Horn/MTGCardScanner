from PIL import Image
import pytesseract
import cv2
import numpy as np
from lib.cropper import crop_image

def extract_text_from_image(image_path):
    image = Image.open(crop_image(image_path))
    text = pytesseract.image_to_string(image, lang='eng', config='--psm 6')
    text = text.replace('\n', ' ').replace('\x0c', ' ').replace('  ', ' ')
    return text

def find_rectangles(image_path):
    img = cv2.imread(image_path)

    down_width = 300
    down_height = 300
    down_points = (down_width, down_height)
    resized_down = cv2.resize(img, down_points, interpolation= cv2.INTER_LINEAR)

    gray = cv2.cvtColor(resized_down, cv2.COLOR_BGR2GRAY)

    ret,thresh = cv2.threshold(gray,40,255,0)
    contours,hierarchy = cv2.findContours(thresh, 1, cv2.CHAIN_APPROX_SIMPLE)
    print("Number of contours detected:", len(contours))

    for cnt in contours:
        x1,y1 = cnt[0][0]
        approx = cv2.approxPolyDP(cnt, 0.2*cv2.arcLength(cnt, True), True)
        if len(approx) == 4:
            x, y, w, h = cv2.boundingRect(cnt)
            ratio = float(w)/h
            if ratio >= 0.9 and ratio <= 1.1:
                resized_down = cv2.drawContours(resized_down, [cnt], -1, (0,255,255), 3)
                # cv2.putText(resized_down, 'Square', (x1, y1), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)
            else:
                # cv2.putText(resized_down, 'Rectangle', (x1, y1), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
                resized_down = cv2.drawContours(resized_down, [cnt], -1, (0,255,0), 3)

    cv2.imshow("Shapes", resized_down)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


def find_rectangles_realtime():
    cap = cv2.VideoCapture(0,cv2.CAP_DSHOW)
    if not cap.isOpened():
        print("Cannot open camera")
        exit()


    while True:
        # Capture frame-by-frame
        ret, img = cap.read()

        if not ret:
            print("Can't receive frame (stream end?). Exiting ...")
            break

        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        ret,thresh = cv2.threshold(gray,40,255,0)
        contours,hierarchy = cv2.findContours(thresh, 1, cv2.CHAIN_APPROX_SIMPLE)
        print("Number of contours detected:", len(contours))

        for cnt in contours:
            x1,y1 = cnt[0][0]
            approx = cv2.approxPolyDP(cnt, 0.01*cv2.arcLength(cnt, True), True)
            if len(approx) == 4:
                x, y, w, h = cv2.boundingRect(cnt)
                ratio = float(w)/h
                if ratio >= 0.9 and ratio <= 1.1:
                    img = cv2.drawContours(img, [cnt], -1, (0,255,255), 3)
                    cv2.putText(img, 'Square', (x1, y1), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)
                else:
                    cv2.putText(img, 'Rectangle', (x1, y1), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
                    img = cv2.drawContours(img, [cnt], -1, (0,255,0), 3)

        cv2.imshow("Shapes", img)
        if cv2.waitKey(1) == ord('q'):
            break


def main():
    try: 
        print("Starting to scan ")

        # find_rectangles("./test/test1.jpg")
        # find_rectangles("./test/test_closeup_1.jpg")
        # find_rectangles_realtime()
        print(extract_text_from_image('./test/test1.jpg'))
        print(extract_text_from_image('./test/test2.jpg'))
        print(extract_text_from_image('./test/test3.jpg'))
        print(extract_text_from_image('./test/test4.jpg'))
    except KeyboardInterrupt :
        print("Goodbye")
    return

if __name__ == "__main__": 
    main()