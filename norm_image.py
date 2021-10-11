import cv2
import rawpy
import imageio
import PIL
import numpy as np 
import pandas as pd

def norm_image (path1):   #Lum:  784 - 0.049 cd
    K_Gauss = cv2.getGaussianKernel(20,5)
    df= pd.read_excel(path1, index_col=0)
    path="/content/cameracalibration/"+df['File RGB'][1]
    print(path)
    raw = rawpy.imread(path)
    rgb16 = raw.postprocess(gamma=(1,1),no_auto_bright=True,no_auto_scale=True, output_bps=16)
    height, width, channels = rgb16.shape
    full_img2_1=np.zeros((height, width),np.float64)
    for n in range(13): #Total de imagnes 13
        path="/content/cameracalibration/"+df['File RGB'][n+1]
        Xm=df['Xm'][n+1]        # 
        Ym=df['Ym'][n+1]           # 
        raw = rawpy.imread(path)
        rgb16 = raw.postprocess(gamma=(1,1),no_auto_bright=True,no_auto_scale=True, output_bps=16)
        R=rgb16[:,:,0]
        G=rgb16[:,:,1]
        B=rgb16[:,:,2]
        Yimg=(0.2162*R)+(0.7152*G)+(0.0722*B)
        #Convolve function
        dst2 = cv2.sepFilter2D(Yimg,-1,K_Gauss,K_Gauss)
        mask=np.zeros((height, width),np.uint8)
        cv2.circle(mask,(Ym,Xm), 120, (1), -1)
        full_img2_1=full_img2_1+(dst2*mask)
    return full_img2_1