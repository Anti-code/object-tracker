import cv2, numpy as np
from filterpy.kalman import KalmanFilter

kalman = cv2.KalmanFilter(4,2)
#vektor
kalman.measurementMatrix = np.array([[1,0,0,0],[0,1,0,0]],np.float32)
kalman.transitionMatrix = np.array([[1,0,1,0],[0,1,0,1],[0,0,1,0],[0,0,0,1]],np.float32)
def kalmn(mp):
    z=kalman.correct(mp)
    for _ in range(6):        
        z=kalman.correct(mp)
        t=kalman.predict()
    return z

def main():
    enable_reks=False
    enable_lines=False
    thresh_show= False
    morphology = False
    mark = False
    blur=True
    erosion=False
    gray_show= False
    
    cnts=[]
    mp = np.array((2,1), np.float32) # measurement
    tp = np.zeros((2,1), np.float32) # tracked / prediction
    cap=cv2.VideoCapture('fish.avi')
    
    frame1 = None
    while True:
        k = cv2.waitKey(30) &0xFF
        
        (yakala, frame) = cap.read()
        if not yakala:
                cap.release()
                main()
        frame2 = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        if blur:
            frame2 = cv2.GaussianBlur(frame2, (25, 25), 0)

        if frame1 is None:
            frame1 = frame2
            continue
        fark = cv2.absdiff(frame1,frame2)
        
        thresh = cv2.threshold(fark, 15, 255, cv2.THRESH_BINARY)[1]
        
        if gray_show:cv2.imshow('gray',frame2)
        else:cv2.destroyWindow('gray')
        if thresh_show:cv2.imshow('first threshold',thresh)
        else:cv2.destroyWindow('first threshold')
        
        #cv2.imshow('first threshold',thresh)
        erode_kernel = np.ones((10,10),np.uint8)
        dilate_kernel = np.ones((15,15),np.uint8)
        thresh = cv2.erode(thresh, erode_kernel, iterations=2)
        if erosion:cv2.imshow('erode',thresh)
        else:cv2.destroyWindow('erode')
        thresh = cv2.dilate(thresh, dilate_kernel, iterations=2)
        if morphology:cv2.imshow('morphology',thresh)
        else:cv2.destroyWindow('morphology')
        
        (_, cnts, _) = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
        for c in cnts:
            if cv2.contourArea(c) < 100:
                continue
            (x,y,w,h)=cv2.boundingRect(c)
            if mark:                            
                cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
            mp = np.array([[np.float32(x)],[np.float32(y)]])
            m,n,_,_= kalmn(mp)
             #bgr
            if enable_reks:
                #cv2.rectangle(frame, (int(m[0]), int(n[0])), (int(m[0]) + w, int(n[0]) + h), (0, 0, 255), 2)
                cv2.circle(frame,(int(m[0]+w/2), int(n[0]+h/2)),5,(0,0,255),2)
            
        if k== ord('k'): #kalman
            if enable_reks==False:enable_reks=True
            else:enable_reks=False
        if k== ord('l'):
            if enable_lines==False:enable_lines=True
            else:enable_lines=False
        if k== ord('e'): #erosion
            if erosion==False:erosion=True
            else:erosion=False
        if k== ord('d'):# dilation
            if morphology==False:morphology=True
            else:morphology=False
        if k== ord('t'): #threshold
            if thresh_show==False:thresh_show=True
            else:thresh_show=False
        if k== ord('m'): #marks
            if mark==False:mark=True
            else:mark=False
        if k== ord('g'): #gray
            if gray_show==False:gray_show=True
            else:gray_show=False
        if k== ord('b'): #gaus blur
            if blur==False:
                blur=True
            else:
                blur=False
        if k == 27:
            cv2.destroyAllWindows()
            break
        
        #frame1=frame2
        cv2.imshow("Foo", frame)
        

if __name__=="__main__":
    main()
	
