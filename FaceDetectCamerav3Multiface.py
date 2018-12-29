# -*- coding: utf-8 -*-
"""
Created on Tue May 22 14:44:59 2018

@author: pedzenon
"""

import cv2
from imutils.video import FPS
from google.cloud import vision
from google.cloud.vision import types
import numpy as np
from queue import Queue
from threading import Thread
import operator
import pandas as pd

# incializo la varible de comunicacion del tread con el main
sentiment = pd.DataFrame([['NoSentiment',0,0,0]],columns=['sentiment','x','y','dist'])
    
def main():

# fps.stop()
# print(fps.fps())
    cam = cv2.VideoCapture(0)
    window = []
    FPSxseg = 12 
    client = vision.ImageAnnotatorClient()
    q = Queue()   # cola de comunicacion thread - main
    q.put("Finish")  # inicializacion    
    global sentiment    
    face_img = {'NoSentiment':cv2.imread("noSentiment.png",-1), 'anger': cv2.imread("anger.png",-1) ,
                'joy': cv2.imread("joy.png",-1) , 'surprise': cv2.imread("surprise.png",-1) ,
                'sorrow':cv2.imread("sorrow.png",-1)}
    
    # fps = FPS().start()
    while True:
        ret_val, img = cam.read()
       
        img = cv2.flip(img, 1)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        
        face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
        faces = face_cascade.detectMultiScale(gray, 1.3, 5)
                
        # voy cargando aparicion de caras
        if (len(faces) >= 1):
            window.append(1)
        else:
            window.append(0)
        # Cada 2 segundos para testear si hay presencia de cara
        if(len(window) == FPSxseg*2 + 1):
            if(sum(window) > 0.75*FPSxseg):
                image = types.Image(content= np.array(cv2.imencode('.jpg', img)[1]).tobytes())
                # Start thread if not started before
                if(q.empty() == False):
                        if(q.get() == "Finish"):
                            in_thread = {'client':client,'image':image,'queue':q}
                            t = Thread(target=GoogleCall, args=(in_thread,))
                            t.start()                
            else:
                sentiment['sentiment'] = 'NoSentiment'
                
            window.clear()
    
        # por cada cara que detecta opencv matcheo la cara de google con la de opencv
        for (x_cv,y_cv,w,h) in faces:          
            x_cv = np.float64(x_cv)
            y_cv = np.float64(y_cv)
            sentiment['dist'] = 0
            sentiment['dist'] = np.power(sentiment.x - x_cv,2) + np.power(sentiment.y - y_cv,2)
            senti = sentiment.iloc[sentiment['dist'].idxmin(),].sentiment
            face_add(face_img[senti],img,np.int32(x_cv),np.int32(y_cv),w,h)            
        
           
        cv2.imshow('my webcam', img)
        if cv2.waitKey(1) == 27: 
            break  # esc to quit
        
        # fps.update()
    cv2.destroyAllWindows()

# =============================================================================
#   @fn: face_add
#    @s_img: foto del emoji
#    @l_img: foto sobre la que hay que pegar la imagen
#    @x,y: coordenadas donde esta la cara de la persona
#    @w,h: dimensiones de la cara
#   @brief: agrega a la imagen base la foto del emoji
# =============================================================================
    
def face_add(s_img,l_img,x,y,w,h):
    
    rec_dim = max([h,w])

    dim = (rec_dim,rec_dim)
    s_img = cv2.resize(s_img, dim, interpolation = cv2.INTER_AREA)
    
    x_offset= x
    y_offset= y
    
    y1, y2 = y_offset, y_offset + s_img.shape[0]
    x1, x2 = x_offset, x_offset + s_img.shape[1]
    
    alpha_s = s_img[:, :, 3] / 255.0
    alpha_l = 1.0 - alpha_s
    
    for c in range(0, 3):
        l_img[y1:y2, x1:x2, c] = (alpha_s * s_img[:, :, c] +
                                  alpha_l * l_img[y1:y2, x1:x2, c])

# =============================================================================
#  @fn: GoogleCall
#   @in_param: diccionario con client de google, imagen para proceesar, queue de comunicacion con thread
#   @brief: Hace una llamada a la API de google y carga en la variable global sentiment el sentimeiento que la API entrego
# =============================================================================

def GoogleCall(in_param):
    print('Google Call!')
    client = in_param['client']
    image = in_param['image']
    queue = in_param['queue']
    response = client.face_detection(image=image)
    faces_ = response.face_annotations
    
    global sentiment
    sentiment = sentiment_vote(faces_)  # cargo la variable global con el resultado
    queue.put("Finish")  # aviso al main que el thread termino!

# =============================================================================
#  @fn: sentiment_vote
#    @faces: json de google
#    @brief: entrega el sentimiento resultante que envio google junto con la posicion del ojo izquierdo para traquear caras
#   @out: entrego una lista con sentimiento+posicion del ojo
# =============================================================================

def sentiment_vote(faces):
    sentiment = []
    likelihood_name = (0, 0, 1, 2,3, 4)
    google_vision = {}
    for face in faces:
        aux = {}
        google_vision['anger'] = likelihood_name[face.anger_likelihood]
        google_vision['joy'] = likelihood_name[face.joy_likelihood]
        google_vision['surprise'] = likelihood_name[face.surprise_likelihood]
        google_vision['sorrow'] = likelihood_name[face.sorrow_likelihood]
        face.landmarks[0].position.x
        aux['y'] = face.landmarks[0].position.y
        aux['x'] = face.landmarks[0].position.x
        sentiment_aux = max(google_vision.items(), key=operator.itemgetter(1))[0]        
        
        if(google_vision[sentiment_aux] >= 2):  # likely or very_likely
            aux['sentiment'] = sentiment_aux
        else:
            aux['sentiment'] = 'NoSentiment'
        
        sentiment.append(aux)
    
    return pd.DataFrame(sentiment)  # devuelvo un dataframe con el sentimiento y ubicacion
    
    
    
##############################################################################
        
if __name__ == '__main__':
    main()
    