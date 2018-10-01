import matplotlib as mpl
from matplotlib import pyplot
import matplotlib.colors as clr
import numpy as np
import skimage.measure
import random
label_in=np.load("./Y_test.npy")
prediction_in=np.load("./test_result_2.npy") #(5, 24, 5, 128, 257, 1)
d1,d2,d3,d4,d5,d6=np.shape(prediction_in)
def get_coordinate_from_ground_truth(yy):
 y_index,x_index=np.where(yy==1) #(array([19, 27, 30]), array([ 77, 119,  57]))
 return y_index,x_index

def get_coordnate_from_prediction(pre,x_i):
 pre_copy=pre
 x_index=[]
 y_index=[]
 for k in range(len(x_i)):
    y,x=np.where(pre_copy==np.amax(pre_copy))
    pre_copy[y,x]=0
    x_index.append(x); y_index.append(y);
 return y_index,x_index

def obtain_rmse(x_i,y_i,x_j,y_j):
 #Calculate every pair distance
 rmse_list=[]
 for i in range(len(x_i)):
    for j in range(len(x_j)):
        rmse_list.append(pow(pow(float(x_i[i])-float(x_j[j]),2)+pow(float(y_i[i])-float(y_j[j]),2),0.5))
 rmse_list.sort()
 rmse=rmse_list[0:len(x_i)]
 return rmse 
              
rmse_all=[]
x_gt=[];y_gt=[];x_pre=[];y_pre=[];
for ll in range(d1):
    rmse=[]
    x_gt_d=[];y_gt_d=[];x_pre_d=[];y_pre_d=[];
    for i in range(24):
        label=label_in[ll,i,:,:,:,0] #(183, 24, 5, 128, 257, 6)
        prediction=prediction_in[ll,i,:,:,:,0]
        s1,s2,s3=np.shape(prediction)
        mse_t=[]
        rmse_time=[]
        for k in range(s1):
           yy=label[k,:,:]
           pre=prediction[k,:,:]; 
           h=128; w=257;
           yy=np.reshape(yy,[h,w])
           pre=np.reshape(pre,[h,w])
           print("Timestep "+str(k))
           min_val=0
           max_val=1
           mid_val=(min_val+max_val)/2.0
           # make a color map of fixed colors
           cmap = clr.LinearSegmentedColormap.from_list('custom blue', ['#244162','#DCE6F1'], N=256)
           bounds=[min_val,mid_val,max_val]
           norm = mpl.colors.BoundaryNorm(bounds, cmap.N)
           print(str(ll)+"th batch "+str(i)+ "time "+str(k)+"(1) ground truth label")
           # tell imshow about color map so that only set colors are used
           img = pyplot.imshow(yy,interpolation='nearest',cmap = cmap)
           print("Coordinate")
           y_i,x_i=get_coordinate_from_ground_truth(yy)
           y_gt_d.append(y_i); x_gt_d.append(x_i);
           for j in range(len(x_i)):
               print(str(j)+" th storm"+str(x_i[j])+" , "+str(y_i[j]))
               pyplot.plot(x_i[j], y_i[j], 'r+')
        # make a color bar
    #    pyplot.colorbar(img,cmap=cmap,
    #                norm=norm,boundaries=bounds,ticks=[min_val,mid_val,max_val])
        #pyplot.show()
        #pyplot.savefig("ground_truth_"+str(ll)+"_"+str(i)+"_"+str(k)+".png")
           print(str(ll)+"th batch "+str(i)+ "time "+str(k)+"(2) prediction label")
           img = pyplot.imshow(pre,interpolation='nearest',cmap = cmap)
           print("Coordinate")
           y_j,x_j=get_coordnate_from_prediction(pre,x_i)
           y_pre_d.append(y_j); x_pre_d.append(x_j);
           for j in range(len(x_j)):
               print(str(j)+" th storm"+str(x_j[j])+" , "+str(y_j[j]))
               pyplot.plot(x_j[j], y_j[j], 'r+')
           rmse_result=obtain_rmse
           rmse_time.append(rmse_result)
           rmse_time.append(rmse_result)
           #pyplot.show()
           #pyplot.savefig("prediction_"+str(ll)+"_"+str(i)+"_"+str(k)+".png")
        rmse.append(rmse_time) 
    np.save("rmse_"+str(ll)+".npy",rmse)  
    rmse_all.append(rmse)
x_gt.append(x_gt_d); y_gt.append(y_gt_d); x_pre.append(x_pre_d); y_pre.append(y_pre_d);
coordinate=[x_gt,y_gt,x_pre,y_pre]
np.save("coordinate.npy",coordinate)
np.save("rmse_all.npy",rmse_all)


