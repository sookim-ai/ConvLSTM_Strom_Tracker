#python  error_matric_noplot_v1.py 0.3[threshold] 10[radius] 
import numpy as np
import sys
import random
label_in=np.load("./Y_test.npy")
prediction_in=np.load("./test_result_20.npy") 
d1,d2,d3,d4,d5,d6=np.shape(prediction_in)
threshold = 0.01
a = 20 
input_iou = 0.1 #0.1 0.2 0.5
threshold=float(sys.argv[1]) #0.01
a=float(sys.argv[2])
input_iou=float(sys.argv[3])

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
 #obtain cluster group from prediction image yy by threshold
 Y_y,Y_x=get_pixel_group_from_prediction(yy,threshold);
 cluster_gt_x=[];cluster_gt_y=[];cluster_pre_x=[];cluster_pre_y=[];
 # if ground truth coordinates (x_i,y_i) are in prediction cluster, save coordinate in cluster_gt
 for k in range(len(y_i)):
     for i in range(len(Y_y)):
         if y_i[k] in Y_y[i]:
             cluster_gt_y.append(Y_y[i])
 for k in range(len(x_i)):
     for i in range(len(Y_x)):
         if x_i[k] in Y_x[i]:
             cluster_gt_x.append(Y_x[i])
 # if predicted coordinates (x_j,y_j) are in ground truth cluster, save coordinate in cluster_pre
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



# From the (x_i,y_i) match all cases of (x_j,y_j)  then calculate error matric
def obtain_rmse_and_matching(x_i,y_i,x_j,y_j):
 threshold_rmse=pow(2,0.5)*a*(1.0-pow(2*input_iou/(1.0+input_iou),0.5))
 #Calculate every pair distance
 rmse_list=[]; rmse_list_all=[];
 y=[];x=[];
 num_of_matched=0
 num_of_predicted=0 
 if len(x_i)!=0 and len(x_j)!=0:
     for i in range(len(x_i)):
        rmse_i=[]
        for j in range(len(x_j)):
            rmse_i.append(pow(pow(float(x_i[i])-float(x_j[j]),2)+pow(float(y_i[i])-float(y_j[j]),2),0.5))
        rmse_list.append(np.min(rmse_i))
     for i in range(len(rmse_list)):
         num_of_predicted=num_of_predicted+1
         if rmse_list[i] <threshold_rmse:
             num_of_matched=num_of_matched+1
 matched=float(num_of_matched)
 detected=float(num_of_predicted)
 return rmse_list,matched,detected


rmse_all_gt=[]; 
rmse_all_pr=[];
recall_matched=0; recall_detected=0; precision_matched=0; precision_detected=0;
for ll in range(d1):
    rmse_gt=[];rmse_pr=[]
    x_gt_d=[];y_gt_d=[];x_pre_d=[];y_pre_d=[];
    for i in range(d2):
        label=label_in[ll,i,:,:,:,0] 
        prediction=prediction_in[ll,i,:,:,:,0]
        s1,s2,s3=np.shape(prediction)
        mse_t=[]
        rmse_time_pr=[]; rmse_time_gt=[]
        for k in range(s1):
           yy=label[k,:,:]
           pre=prediction[k,:,:];
           h=128; w=257;
           yy=np.reshape(yy,[h,w])
           pre=np.reshape(pre,[h,w])
           # Find centroids in ground truth clusters
           y_i,x_i=get_coordinate_from_ground_truth(yy)
           # Grouping predicted image
           G_y,G_x=get_pixel_group_from_prediction(pre,threshold)
           # find centroids in predicted clusters
           y_j,x_j=get_coordinate_from_prediction_2(pre,G_y,G_x)
           # Matching ground truth centroids and predicted centroid 
           rmse_result_gt,gt_matched,gt_detection=obtain_rmse_and_matching(x_i,y_i,x_j,y_j) #Recall
           recall_matched=recall_matched+gt_matched
           recall_detected=recall_detected+gt_detection
           rmse_result_pr,pr_matched,pr_detection=obtain_rmse_and_matching(x_j,y_j,x_i,y_i) #Precision
           precision_matched=precision_matched+pr_matched
           precision_detected=precision_detected+pr_detection
           rmse_time_gt.append(rmse_result_gt)
           rmse_time_pr.append(rmse_result_pr)
        rmse_gt.append(rmse_time_gt)
        rmse_pr.append(rmse_time_pr)
    rmse_all_gt.append(rmse_gt)
    rmse_all_pr.append(rmse_pr)

recall=float(recall_matched)/float(recall_detected)
precision=float(precision_matched)/float(precision_detected)


print("++++++++++++++++++++++")
print("IOU based detection error matric")
print(" Error Analysis with bounding box")
print(" Threshold "+str(threshold))
print(" Bounding Box size "+str(2*a))
print(" Number of TP: "+str(recall_matched))
print(" Number of FP: "+str(precision_detected-precision_matched))
print(" Number of FN: "+str(recall_detected-recall_matched))
print(" Recall : "+str(recall*100)+" %")
print(" Precision    : "+str(precision*100)+" %")
print(" IOU :"+str(input_iou)) 
print(" Result, "+str(threshold)+" , "+str(precision*100)+" , "+str(recall*100))

