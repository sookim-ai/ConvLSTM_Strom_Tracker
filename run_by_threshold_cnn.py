import os,sys

#IOU:0.1
for i in range(500):
    threshold=0.5+float(i)*0.001
    a=32
    print (str(threshold) + " starting ..")
    os.system("python error_matric_IOU_cnn.py "+str(threshold)+" "+str(a) +" 0.1"+ " >> convlstm_output_iou0.1_cnn.txt")  


#IOU:0.5
for i in range(500):
    threshold=0.5+float(i)*0.001
    a=32
    print (str(threshold) + " starting ..")
    os.system("python error_matric_IOU_cnn.py "+str(threshold)+" "+str(a) +" 0.5"+ " >> convlstm_output_iou0.5_cnn.txt")

