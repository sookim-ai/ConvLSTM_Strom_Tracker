import numpy as np
import skimage.measure
import random

path="/export/kim79/h2/TC_labeled_dataset/2_new_dataset_for_heatmap_generation_normalized/light/"
def generate_data(path,timesteps,batch_size):
    X=[]
    Y=[]
    index=[i for i in range(205)] #0, 1, 2, ... , n-1
    for i in range(len(index)):
        print("reading "+str(i)+"th data")
        image=np.load(path+"div_image_"+str(i)+".npy") #(65, 256, 513, 6)
        label=np.load(path+"div_image_label_"+str(i)+".npy") 
        a1,a2,a3=np.shape(label)
        label=np.reshape(label,[a1,a2,a3,1])
        image=skimage.measure.block_reduce(np.asarray(image), (1,2,2,1), np.max)
        label=skimage.measure.block_reduce(np.asarray(label), (1,2,2,1), np.max)
        print(np.shape(image),np.shape(label))
        d1,d2,d3,d4=np.shape(image)
        if a1 > timesteps :
            for j in range(int(a1/(timesteps*0.5))-1):
                start=j*int(timesteps*0.5)
                end=start+timesteps
                X.append(np.reshape(image[start:end,:,:,:],[1,timesteps,d2,d3,6]))
                Y.append(np.reshape(label[start:end,:,:,:],[1,timesteps,d2,d3,1]))
        if i%100==0: print(np.shape(X))          
    s1=len(Y)

    random_index=[ i for i in range(s1) ]
    X_r=[]
    Y_r=[]
    #random shuffle the order of dataset
    for i in range(len(random_index)):
        X_r.append(X[random_index[i]])
        Y_r.append(Y[random_index[i]])
    X_r=np.concatenate(X_r,0)
    Y_r=np.concatenate(Y_r,0)
    print(np.shape(X_r),np.shape(Y_r))
    num_of_groups=int(s1/batch_size)
    X_sample=X_r[0:24*num_of_groups,:,:,:,:]
    Y_sample=Y_r[0:24*num_of_groups,:,:,:,:]
    X_sample_last=X_r[s1-24:s1,:,:,:,:]
    Y_sample_last=Y_r[s1-24:s1,:,:,:,:]
    X_sample=np.concatenate([X_sample, X_sample_last],0)
    Y_sample=np.concatenate([Y_sample, Y_sample_last],0)
    
    X_sample=np.reshape(X_sample,[num_of_groups+1,24,timesteps,d2,d3,6])
    Y_sample=np.reshape(Y_sample,[num_of_groups+1,24,timesteps,d2,d3,1])

    np.save("X_light.npy",X_sample)
    np.save("Y_light.npy",Y_sample)
    return X_sample,Y_sample






image,label=generate_data(path,10,24)
