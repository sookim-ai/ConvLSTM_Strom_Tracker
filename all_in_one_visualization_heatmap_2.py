import matplotlib as mpl
from matplotlib import pyplot
import matplotlib.colors as clr
import numpy as np
#import skimage.measure
import random
for test_num in [7,8,9,10,11,12,13,14]:   
 name=["u850","v850","prect"]
 image_inn=np.load('./X_test.npy') #(84, 24, 10, 128, 257, 3)
 label_inn=np.load("./Y_test.npy") #(84, 24, 10, 128, 257, 1)
 prediction_inn=np.load("./test_result_"+str(test_num)+".npy") #(84, 24, 10, 128, 257, 1)
 d1,d2,d3,d4,d5,d6=np.shape(image_inn)
 print(np.shape(image_inn))
 for ii in [0,1,7]: #28
    image_in=image_inn[ii,:,:,:,:,:]
    label_in=label_inn[ii,:,:,:,:,0] ##[24,10,128,257]
    prediction_in=prediction_inn[ii,:,:,:,:,0] #[24,10,128,257]
    for i in [1,17,10]: #24
        image=image_in[i,:,:,:,:] #[10,128,257,3]
        label=label_in[i,:,:,:] #[10,128,257]
        prediction=prediction_in[i,:,:,:] #[10,128,257]
        s1,s2,s3=np.shape(prediction) 
        for k in range(d3): #10 time
            pyplot.figure(1,figsize=(20,5))
            yy=label[k,:,:]
            pre=prediction[k,:,:]
            img_o=image[k,:,:,:] 
            h=128; w=257;
            yy=np.reshape(yy,[h,w])
            pre=np.reshape(pre,[h,w])
            print("Timestep "+str(k))
            min_val=0;max_val=1;mid_val=(min_val+max_val)/2.0;
            # make a color map of fixed colors
            cmap = clr.LinearSegmentedColormap.from_list('custom blue', ['#244162','#DCE6F1'], N=256)
            bounds=[min_val,mid_val,max_val]
            norm = mpl.colors.BoundaryNorm(bounds, cmap.N)
            print("Filenumber",ii,"Batch ",i,"time steps ",k)
            #u850,v850,prect
            for ch in range(d6):
                pyplot.subplot(5,10,d3*ch+k+1)
                pyplot.subplots_adjust(hspace = .001)
                if k==0: pyplot.title(name[ch])
                img = pyplot.imshow(img_o[:,:,ch],interpolation='nearest',cmap = cmap)
                pyplot.axis('off')
            #Ground truth heat map
            print("ground truth density map")
            pyplot.subplot(5,10,d3*d6+k+1)
            pyplot.subplots_adjust(hspace = .001)
            if k==0: pyplot.title("Ground Truth")
            img = pyplot.imshow(yy,interpolation='nearest',cmap = cmap)
            pyplot.axis('off')
            #Prediction heat map
            pyplot.subplot(5,10,d3*(d6+1)+k+1)
            if k==0: pyplot.title('Prediction')
            pyplot.subplots_adjust(hspace = .001)
            img = pyplot.imshow(pre,interpolation='nearest',cmap = cmap)
            pyplot.axis('off')
        pyplot.tight_layout()
#        pyplot.show()
        pyplot.savefig("all_"+str(test_num)+"_"+str(ii)+"_"+str(i)+".png")

