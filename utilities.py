import cv2
def visualize_result(frame, hand, foot, region, CLASSES, put_text=None):
    cv2.circle(frame, (hand[0], hand[1]), 10, (255,0,0), -1)
    cv2.circle(frame, (foot[0], foot[1]), 10, (0,0,255), -1)
    font = cv2.FONT_HERSHEY_SIMPLEX


    if put_text is not None:
        cv2.putText(frame,
                    put_text,
                    (50, 50),
                    font, 1,
                    (0, 255, 255), 
                    3,
                    cv2.LINE_4)

        cv2.putText(frame, 
                    "shot taker", 
                    (hand[0], hand[1]), 
                    font, 1,
                    (0, 255, 255), 
                    2,
                    cv2.LINE_4)



        cv2.imshow('Frame',frame)
        cv2.waitKey(0)

    else:
        cv2.putText(frame,
                    CLASSES[region],
                    (50, 50),
                    font, 1,
                    (0, 255, 255), 
                    3,
                    cv2.LINE_4)

        cv2.putText(frame, 
                    "shot taker", 
                    (hand[0], hand[1]), 
                    font, 1,
                    (0, 255, 255), 
                    2,
                    cv2.LINE_4)



        cv2.imshow('Frame',frame)
        cv2.waitKey(0)