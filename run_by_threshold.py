import os,sys

#IOU:0.1
for i in range(1002):
    threshold=float(i)*0.001
    a=32
    print (str(threshold) + " starting ..")
    os.system("python error_matric_IOU_convLSTM.py "+str(threshold)+" "+str(a) +" 0.1"+ " >> convlstm_output_5layer_iou0.1.txt")  


#IOU:0.5
for i in range(1002):
    threshold=float(i)*0.001
    a=32
    print (str(threshold) + " starting ..")
    os.system("python error_matric_IOU_convLSTM.py "+str(threshold)+" "+str(a) +" 0.5"+ " >> convlstm_output_5layer_iou0.5.txt")

