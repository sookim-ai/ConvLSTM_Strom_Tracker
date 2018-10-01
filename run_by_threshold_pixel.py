import os,sys

for i in range(500):
    threshold=0.001+float(i)*0.001
    a=32
    print (str(threshold) + " starting ..")
    os.system("python error_matric_pixel_level_convLSTM.py "+str(threshold)+" "+str(a) +" >> convlstm_output_5layer_pixel.txt")  



