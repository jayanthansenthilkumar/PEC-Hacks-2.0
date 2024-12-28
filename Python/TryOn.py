import cv2
import cvzone
from cvzone.PoseModule import PoseDetector
import os
cap = cv2.VideoCapture(0)
detector = PoseDetector()
fixedRatio = 262/190
shirtRatioHeightWidth = 581 / 440
shirtFolderPath = "Shirts"
buttonRightPath = "button.png"
buttonLeftPath = "button.png"
imgButtonRight = cv2.imread(buttonRightPath, cv2.IMREAD_UNCHANGED)
imgButtonLeft = cv2.imread(buttonLeftPath, cv2.IMREAD_UNCHANGED)
listShirts = os.listdir(shirtFolderPath)
imageNumber = 0
selectionSpeed = 5
counterRight = 0
counterLeft = 0
while True:
    success, img = cap.read()
    if not success:
        break
    img = detector.findPose(img)
    lmList, bboxInfo = detector.findPosition(img, bboxWithHands=False, draw=False)
    if lmList:
        lm11 = lmList[11][0:2]
        lm12 = lmList[12][0:2]
        cv2.circle(img, tuple(lm11), 10, (255, 0, 0), cv2.FILLED)
        cv2.circle(img, tuple(lm12), 10, (0, 255, 0), cv2.FILLED)
        widthOfShirt = int((lm11[0] - lm12[0]) * fixedRatio)
        if widthOfShirt > 0:
            shirtPath = os.path.join(shirtFolderPath, listShirts[imageNumber])
            imgShirt = cv2.imread(shirtPath, cv2.IMREAD_UNCHANGED)
            if imgShirt is not None:
                imgShirt = cv2.resize(imgShirt, (widthOfShirt, int(widthOfShirt * shirtRatioHeightWidth)))
                currentScale = (lm11[0] - lm12[0]) / 190
                offset = int(44 * currentScale), int(48 * currentScale)
                try:
                    overlay_x = lm12[0] - offset[0]
                    overlay_y = lm12[1] - offset[1]
                    overlay_y = max(0, overlay_y)
                    img = cvzone.overlayPNG(img, imgShirt, (lm12[0]-offset[0],lm12[1]-offset[1]))
                except Exception as e:
                    print(f"Error in overlaying PNG: {e}")
            else:
                print(f"Failed to load shirt image: {shirtPath}")
        img = cvzone.overlayPNG(img, imgButtonRight, (1074, 293))
        img = cvzone.overlayPNG(img, imgButtonLeft, (72, 293))
        if lmList[16][1] < 300:
            counterRight += 1
            cv2.ellipse(img, (139, 360), (66, 66), 0, 0, counterRight * selectionSpeed, (0, 255, 0), 20)
            if counterRight * selectionSpeed > 360:
                counterRight = 0
                if imageNumber < len(listShirts) - 1:
                    imageNumber += 1
        elif lmList[15][1] > 900:
            counterLeft += 1
            cv2.ellipse(img, (1138, 360), (66, 66), 0, 0, counterLeft * selectionSpeed, (0, 255, 0), 20)
            if counterLeft * selectionSpeed > 360:
                counterLeft = 0
                if imageNumber > 0:
                    imageNumber -= 1
        else:
            counterRight = 0
            counterLeft = 0

    cv2.imshow("Image", img)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
cap.release()
cv2.destroyAllWindows()