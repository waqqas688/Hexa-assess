import cv2
import numpy as np
cam = cv2.VideoCapture(0)


def init_model():
    try:
        detector = cv2.FaceDetectorYN.create(r'models\face_detection_yunet_2023mar.onnx',"",(640,480)) #Initiating model with 640 x 480 size for faster detection
    except:
        return False
    else:
        return detector 



def run_facereg():

    image_width = 640
    image_height = 480

    buffer_x = 120
    buffer_y= 200

    bound_x1= (image_width//2)-buffer_x
    bound_y1= (image_height//2)-buffer_y
    bound_x2 = (image_width//2)+buffer_x
    bound_y2= (image_height//2)+buffer_y

    model = init_model()
    while True:
        ret,frame = cam.read()
        frame = cv2.resize(frame, (image_width,image_height)) # Resizing image to meet model size requirements.
        
        _, detection = model.detect(frame)
        detection = detection if detection is not None else np.zeros([2,2]) # Replace image with zero array if no person detected.

        if not detection.any():
            print("No person detected")
        else:
            if(len(detection)==1):
                face=detection[0]
                box=list(map(int,face[:4]))
                if(box[0]<bound_x1 or box[1]<bound_y1 or box[2]+box[0]>bound_x2 or box[3]+box[1]>bound_y2 ): #Checking if face in middle of the screen using bounding box.
                    print('Position changed. Please sit in the middle of the camera')
                else:
                    cv2.rectangle(frame ,(bound_x1,bound_y1) , (bound_x2,bound_y2) , (0,0,0),2,cv2.LINE_AA)
                    cv2.rectangle(frame,box,(0,255,0),2,cv2.LINE_AA)
                    print("Success")
            else:
                print('More than 1 person detected. Please make sure you are alone in the room')
                
        cv2.imshow('Frame',frame)
        if cv2.waitKey(1) == ord('q'):
            break
        
    cam.release()
    cv2.destroyAllWindows()

if __name__=="__main__":
    run_facereg()