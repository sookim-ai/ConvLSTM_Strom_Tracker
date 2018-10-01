#python  error_matric_noplot_v1.py 0.3[threshold] 10[radius] 
import numpy as np
import sys
import random
label_in=np.load("./groundtruth_convlstm.npy")
prediction_in=np.load("./output_convlstm.npy") 
d1,d2,d3,d4,d5,d6=np.shape(prediction_in)
threshold=0.01#
a=20 # Default
threshold=float(sys.argv[1]) #0.01
a=float(sys.argv[2])#print(threshold)

#Input: ground truth image "yy" with shape of (h,w)
#output: list of cordinates for hurricane center in image 
#Explain:Because we make gaussian heat-map for ground truth label
#        Hurricane center should have always the value of "1.0" which is the maximum of gaussian distribution
def get_coordinate_from_ground_truth(yy):
 y_index,x_index=np.where(yy==1.0) #(array([19, 27, 30]), array([ 77, 119,  57]))
 return y_index,x_index





#Input: output_image "pre" from convLSTM model with shape (h,w) 
#output: coordinate lists of groups of cluster
#         G_y=[[y_index in group 1], [y_index in group2], [y_index in group3] ... ]
#         G_x=[[x_index in group 1], [x_index in group2], [x_index in group3] ... ]
def get_pixel_group_from_prediction(pre,threshold):
 pre_copy=pre
 x_index=[]
 y_index=[]
 y,x=np.where(pre_copy>threshold)
 G_y=[]; G_x=[];
 max_diameter_of_hurricane = 10 #unit pixel (0.5 degree=55.5km)
 #Grouping
 if len(y)!=0:
     g_y=[y[0]];g_x=[x[0]] 
     for i in xrange(1,len(y)):
         # Check which group is closest group from the point (x[i],y[i])
         # Include point to the group where distance between that group and point is closer than maximum distance of hurricane.
         if len(G_y)>0:
             dist=[]
             for j in range(len(G_y)):
                 dist_j=[]
                 for k in range(len(G_y[j])):
                     dist_j.append(pow(float(G_y[j][k])-float(y[i]),2)+pow(float(G_x[j][k])-float(x[i]),2))
                 dist.append(dist_j)
             if np.min(np.min(dist))<max_diameter_of_hurricane:
                 for ii in range(len(dist)):
                     if np.min(np.min(dist)) in dist[ii]:
                         G_y[ii].append(y[i])
                         G_x[ii].append(x[i])
             else:
                 G_y.append([y[i]]); G_x.append([x[i]]);
         else:
             #Initial generation of cluseter
             #If distance between current point and previous point is smaller than maximum distance of hurricane.
             #Tide together as cluster
             dist=pow(float(g_y[-1])-float(y[i]),2)+pow(float(g_x[-1])-float(x[i]),2)
             if dist<max_diameter_of_hurricane:
                 g_y.append(y[i]); g_x.append(x[i]);
             else:
                 G_y.append(g_y); G_x.append(g_x);
                 g_y=[];g_x=[];
                 g_y.append(y[i]); g_x.append(x[i]);
     G_y.append(g_y); G_x.append(g_x);
 return G_y,G_x


def cluser_matching(yy,G_y,G_x,x_i,y_i,x_j,y_j): # [image, cluster list, gt_x,gt_y,pre_x,pre_y]
 Y_y,Y_x=get_pixel_group_from_prediction(yy,threshold);
 cluster_gt_x=[];cluster_gt_y=[];cluster_pre_x=[];cluster_pre_y=[];
 for k in range(len(y_i)):
     for i in range(len(Y_y)):
         if y_i[k] in Y_y[i]:
             cluster_gt_y.append(Y_y[i])
 for k in range(len(x_i)):
     for i in range(len(Y_x)):
         if x_i[k] in Y_x[i]:
             cluster_gt_x.append(Y_x[i])
 for k in range(len(y_j)):
     for i in range(len(G_y)):
         if y_j[k] in G_y[i]:
             cluster_pre_y.append(G_y[i])
 for k in range(len(x_j)):
     for i in range(len(G_x)):
         if x_j[k] in G_x[i]:
             cluster_pre_x.append(G_x[i])
 return cluster_gt_x,cluster_gt_y,cluster_pre_x,cluster_pre_y


def get_coordinate_from_prediction_1(pre,G_y,G_x):
 #(1) cenral point among pixels
 y=[];x=[];
 for i in range(len(G_y)):
     y.append(sum(G_y[i]) / float(len(G_y[i])))
     x.append(sum(G_x[i]) / float(len(G_x[i])))
 return y,x


def get_coordinate_from_prediction_2(pre,G_y,G_x):
 #(2) Most strong value among clusters
 v=[]
 for i in range(len(G_y)):
     v_i=[]
     for j in range(len(G_y[i])):
         v_i.append(pre[G_y[i][j],G_x[i][j]])
     v.append(v_i) 
 y=[];x=[];
 for i in range(len(G_y)):
     j=v[i].index(np.max(v[i]))
     y.append(G_y[i][j]); x.append(G_x[i][j]);
 return y,x


def obtain_rmse_and_matching(x_i,y_i,x_j,y_j):
 #Calculate every pair distance
 rmse_list=[]; rmse_list_all=[];
 y=[];x=[];
 num_of_predicted=len(y_j)
 if len(x_i)!=0 and len(y_j)!=0:
     for i in range(len(x_i)):
        rmse_i=[]
        for j in range(len(x_j)):
            rmse_i.append(pow(pow(float(x_i[i])-float(x_j[j]),2)+pow(float(y_i[i])-float(y_j[j]),2),0.5))
        rmse_list.append(np.min(rmse_i))
        rmse_list_all.append(rmse_i)
        jj=rmse_i.index(np.min(rmse_i))
        y.append(y_j[jj]); x.append(x_j[jj]);
     num_of_matched=0
     for i in range(len(rmse_list_all)):
         for j in range(len(rmse_list_all[i])):
             if rmse_list_all[i][j] <10:
                 num_of_matched=num_of_matched+1
     precision=float(num_of_matched)/float(num_of_predicted)
     #precision=float(num_of_matched/num_of_predicted)
     #print(num_of_matched,num_of_predicted,precision)
 else: precision=0
 return rmse_list,y,x,precision

rmse_all=[]; precision_all=[]
x_gt=[];y_gt=[];x_pre=[];y_pre=[];
C_gt_x=[]; C_gt_y=[];
C_pre_x=[]; C_pre_y=[];
for ll in range(d1):
    #print(str(ll)+"th among "+str(d1))
    rmse=[];precision=[];
    x_gt_d=[];y_gt_d=[];x_pre_d=[];y_pre_d=[];
    for i in range(d2):
        label=label_in[ll,i,:,:,:,0] 
        prediction=prediction_in[ll,i,:,:,:,0]
        s1,s2,s3=np.shape(prediction)
        mse_t=[]
        rmse_time=[]; precision_time=[]
        for k in range(s1):
           yy=label[k,:,:]
           pre=prediction[k,:,:];
           h=128; w=257;
           yy=np.reshape(yy,[h,w])
           pre=np.reshape(pre,[h,w])
           min_val=0
           max_val=1
           mid_val=(min_val+max_val)/2.0
           # make a color map of fixed colors
 #          cmap = clr.LinearSegmentedColormap.from_list('custom blue', ['#244162','#DCE6F1'], N=256)
 #          bounds=[min_val,mid_val,max_val]
 #          norm = mpl.colors.BoundaryNorm(bounds, cmap.N)
           # tell imshow about color map so that only set colors are used
 #          img = pyplot.imshow(yy,interpolation='nearest',cmap = cmap)
 #          print("Coordinate")
           y_i,x_i=get_coordinate_from_ground_truth(yy)
           y_gt.append(y_i); x_gt.append(x_i);
   #        for j in range(len(x_i)):
   #            print(str(j)+" th storm"+str(x_i[j])+" , "+str(y_i[j]))
   #            pyplot.plot(x_i[j], y_i[j], 'r+')
        # make a color bar
    #    pyplot.colorbar(img,cmap=cmap,
    #                norm=norm,boundaries=bounds,ticks=[min_val,mid_val,max_val])
#           pyplot.show()
        #pyplot.savefig("ground_truth_"+str(ll)+"_"+str(i)+"_"+str(k)+".png")
#           print(str(ll)+"th batch "+str(i)+ "time "+str(k)+"(2) prediction label")
#           img = pyplot.imshow(pre,interpolation='nearest',cmap = cmap)
#           print("Coordinate")
           G_y,G_x=get_pixel_group_from_prediction(pre,threshold)
           y_j,x_j=get_coordinate_from_prediction_2(pre,G_y,G_x)
           rmse_result,y_j,x_j,precision_result=obtain_rmse_and_matching(x_i,y_i,x_j,y_j)
           y_pre.append(y_j); x_pre.append(x_j);
           cluster_gt_x,cluster_gt_y,cluster_pre_x,cluster_pre_y=cluser_matching(yy,G_y,G_x,x_i,y_i,x_j,y_j)
           C_gt_x.append(cluster_gt_x); C_gt_y.append(cluster_gt_y);
           C_pre_x.append(cluster_pre_x); C_pre_y.append(cluster_pre_y);
  #         for j in range(len(x_j)):
  #             print(str(j)+" th storm"+str(x_j[j])+" , "+str(y_j[j]))
  #             pyplot.plot(x_j[j], y_j[j], 'r+')
           rmse_time.append(rmse_result)
           precision_time.append(precision_result)
 #          pyplot.show()
           #pyplot.savefig("prediction_"+str(ll)+"_"+str(i)+"_"+str(k)+".png")
        rmse.append(rmse_time)
        precision.append(precision_time)
#    np.save("rmse_"+str(ll)+".npy",rmse)  
    rmse_all.append(rmse)
    precision_all.append(precision)

d=[x_gt,y_gt,x_pre,y_pre]
x_gt=d[0];y_gt=d[1];x_pre=d[2];y_pre=d[3];
c=[C_gt_x,C_gt_y,C_pre_x,C_pre_y] #(4,1200)
c_gt_x=c[0];c_gt_y=c[1];c_pre_x=c[2];c_pre_y=c[3];
rmse=np.reshape(rmse_all,np.shape(c[0])) #(1200,)
#print(rmse)
#print(precision_all)

precision=np.reshape(precision_all,np.shape(c[0])) #(1200,)


def bounding_box_error(rmse,x_gt,y_gt,x_pre,y_pre):
    iou=[]
    for i in range(len(rmse)):
        for j in range(len(rmse[i])):
            if rmse[i][j] > pow(2,0.5)*2.0*a or max(x_pre[i][j],x_gt[i][j])-min(x_pre[i][j],x_gt[i][j]) > a or max(y_pre[i][j],y_gt[i][j])-min(y_pre[i][j],y_gt[i][j])>a :
                iou.append(0.0);
            else:
                intersect=(2*a+min(x_pre[i][j],x_gt[i][j])-max(x_pre[i][j],x_gt[i][j]))*(2*a+min(y_pre[i][j],y_gt[i][j])-max(y_pre[i][j],y_gt[i][j]))
                iou_val=intersect/float(8*a*a-intersect)
                iou.append(iou_val)
    return iou


def intersect(a, b):
    """ return the intersection of two lists """
    return list(set(a) & set(b))

def union(a, b):
    """ return the union of two lists """
    return list(set(a) | set(b))

#TO DO: FIX BUG
def pixel_cluster_error(c_gt_x,c_gt_y,c_pre_x,c_pre_y):
    cluster_gt=[]; cluster_pre=[]
    for i in range(len(c_gt_x)):
        cluster_gt_i=[]
        cluster_pre_i=[]
        for j in range(len(c_gt_x[i])):
            #print(i,j)
            cluster_gt_i.append( (c_gt_x[i][j],c_gt_y[i][j]) )
            cluster_pre_i.append((c_pre_x[i][j],c_pre_y[i][j]) )
        cluster_gt.append(cluster_gt_i)
        cluster_pre.append(cluster_pre_i)
    iou_p=[]
    for i in range(len(cluster_gt)):
            U=union(cluser_gt[i],cluster_pre[i])
            I=intersect(cluser_gt[i],cluster_pre[i])
    iou_p.append(float(len(I))/float(len(U)))
    return iou_p



iou=bounding_box_error(rmse,x_gt,y_gt,x_pre,y_pre);
tp=np.count_nonzero(iou)
recall=float(tp)/float(len(iou))
#pre=float(np.count_nonzero(precision))/float(len(precision))
pre=np.average(precision)

iou_tp=[]
for i in range(len(iou)):
    if iou[i]!=0.0: iou_tp.append(iou[i])


print("++++++++++++++++++++++")
print(" Error Analysis with bounding box")
print(" Threshold "+str(threshold))
print(" Bounding Box size "+str(2*a))
print(" Number of TP: "+str(tp))
print(" Number of FP: "+str(len(iou)-tp))
print(" Recall : "+str(recall*100)+" %")
print(" Precision    : "+str(pre*100)+" %")
print(" Average Intersection over Union (IoU) for object detection among TP :"+str(np.average(iou_tp)*100.0)+" %")                                                                                                                                                                    
