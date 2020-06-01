import numpy as np
import cv2
from PIL import Image
import pytesseract

cap = cv2.VideoCapture(0)


def order_points(pts):
    rect = np.zeros((4, 2), dtype="float32")
    s = pts.sum(axis=1)

    rect[0] = pts[np.argmin(s)]
    rect[2] = pts[np.argmax(s)]

    diff = np.diff(pts, axis=1)
    rect[1] = pts[np.argmin(diff)]
    rect[3] = pts[np.argmax(diff)]

    return rect


while True:
    ret, frame = cap.read()
    k = cv2.waitKey(10)
    if k == 27:
        break
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (3, 3), 0)
    edged = cv2.Canny(gray, 75, 200)

    print("STEP 1")

    _, cnts, _ = cv2.findContours(edged.copy(), cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)

    cnts = sorted(cnts, key=cv2.contourArea, reverse=True)[:5]
    cv2.imshow('image', edged)

    for c in cnts:
        peri = cv2.arcLength(c, True)
        approx = cv2.approxPolyDP(c, 0.02 * peri, True)
        screenCnt = []

        if len(approx) == 4:
            contourSize = cv2.contourArea(approx)
            camSize = frame.shape[0] * frame.shape[1]
            ratio = contourSize / camSize
            print("ratio :", ratio)

            if ratio > 0.026:
                screenCnt = approx
                break

    if len(screenCnt) == 0:
        cv2.imshow("WebCam", frame)
        continue
    else:
        print("STEP 2")
        cv2.drawContours(frame, [screenCnt], -1, (0, 255, 0), 2)
        cv2.imshow("WebCam", frame)
        cv2.waitKey(0)

        rect = order_points(screenCnt.reshape(4, 2))
        (topLeft, topRight, bottomRight, bottomLeft) = rect

        w1 = abs(bottomRight[0] - bottomLeft[0])
        w2 = abs(topRight[0] - topLeft[0])
        h1 = abs(topRight[1] - bottomRight[1])
        h2 = abs(topLeft[1] - bottomLeft[1])

        maxWidth = max([w1, w2])
        maxHeight = max([h1, h2])

        dst = np.float32([[0, 0], [maxWidth-1, 0], [maxWidth-1, maxHeight-1], [0, maxHeight-1]])

        M = cv2.getPerspectiveTransform(rect, dst)
        warped = cv2.warpPerspective(frame, M, (maxWidth, maxHeight))

        print("STEP 3")

        warped = cv2.cvtColor(warped, cv2.COLOR_BGR2GRAY)
        warped = cv2.adaptiveThreshold(warped, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 21, 10)

        print("STEP 4")
        break

cap.release()
cv2.destroyAllWindows()
cv2.waitKey(1)

cv2.imshow("Scanned", warped)
cv2.waitKey(0)
cv2.destroyAllWindows()
