import cv2
import numpy as np
import math
try:
    video = cv2.VideoCapture(0)
    while(1):

        ret,frame = video.read()
        cv2.rectangle(frame, (400,400), (100,100), (0,255,255),1)
        region = frame[100:400, 100:400]
        grey = cv2.cvtColor(region, cv2.COLOR_BGR2GRAY)
        gb = cv2.GaussianBlur(grey,(55, 55), 0)
        retval, threshold = cv2.threshold(gb, 10, 255,cv2.THRESH_BINARY_INV+cv2.THRESH_OTSU)

        image, contours, _= cv2.findContours(threshold.copy(),cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)

        max_contr = max(contours, key = lambda x: cv2.contourArea(x))

        convex_hull = cv2.convexHull(max_contr)

        contr = np.zeros(region.shape,np.uint8)
        cv2.drawContours(contr, [max_contr], 0, (255, 0, 0), 0)
        cv2.drawContours(contr, [convex_hull], 0,(0, 0, 255), 0)

        convex_hull = cv2.convexHull(max_contr, returnPoints=False)

        defects = cv2.convexityDefects(max_contr,convex_hull)
        count_defects = 0
        cv2.drawContours(threshold, contours, -1, (255, 255, 255), 3)

        for i in range(defects.shape[0]):
            s,e,f,d = defects[i,0]

            start = tuple(max_contr[s][0])
            end = tuple(max_contr[e][0])
            far = tuple(max_contr[f][0])
            a = math.sqrt((end[0] - start[0])**2 + (end[1] - start[1])**2)
            b = math.sqrt((far[0] - start[0])**2 + (far[1] - start[1])**2)
            c = math.sqrt((end[0] - far[0])**2 + (end[1] - far[1])**2)
            angle = math.acos((b**2 + c**2 - a**2)/(2*b*c)) * 57
            if angle <= 90:
                count_defects += 1

        if count_defects==0:
            cv2.putText(frame,"1", (10, 400), cv2.FONT_ITALIC, 2, (255,255,255))
        if count_defects == 1:
            cv2.putText(frame,"2", (10, 400), cv2.FONT_ITALIC, 2, (255,255,255))
        elif count_defects == 2:
            cv2.putText(frame, "3", (10, 400), cv2.FONT_ITALIC, 2,(255,255,255))
        elif count_defects == 3:
            cv2.putText(frame,"4", (10, 400), cv2.FONT_ITALIC, 2,(255,255,255))
        elif count_defects == 4:
            cv2.putText(frame,"5", (10, 400), cv2.FONT_ITALIC, 2,(255,255,255))
        else:
            cv2.putText(frame,"", (10, 400),cv2.FONT_ITALIC, 2,(255,255,255))

        cv2.imshow('Original Frame', frame)
        cv2.imshow('Contour Frame', contr)
        if cv2.waitKey(1) & 0xFF == ord('q'):
                break

    video.release()
    cv2.destroyAllWindows()
except:
    print('An error occured')
