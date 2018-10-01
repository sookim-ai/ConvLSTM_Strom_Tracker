import matplotlib as mpl
from matplotlib import pyplot
import matplotlib.colors as clr
import numpy as np
import skimage.measure
import random
image_in=np.load('./X_test.npy')
label_in=np.load("./Y_test.npy")
prediction_in=np.load("./test_result_1.npy") #(5, 24, 5, 128, 257, 1)

for i in range(24):
 image=image_in[0,i,:,:,:,1]  #[U850,  V850,  PRECT,  QREFH,  TS,  PSL]
 label=label_in[0,i,:,:,:,0] #(183, 24, 5, 128, 257, 6)
 prediction=prediction_in[0,i,:,:,:,0]
 s1,s2,s3=np.shape(prediction)
 print(np.shape(image))

 for k in range(s1):
    inn=image[k,:,:]
    yy=label[k,:,:]
    pre=prediction[k,:,:]; 
    h=128; w=257;
    inn=np.reshape(inn,[h,w])
    pre=np.reshape(pre,[h,w])
    print("Timestep "+str(k))
    min_val=0
    max_val=1
    mid_val=(min_val+max_val)/2.0
    # make a color map of fixed colors
    cmap = clr.LinearSegmentedColormap.from_list('custom blue', ['#244162','#DCE6F1'], N=256)
    bounds=[min_val,mid_val,max_val]
    norm = mpl.colors.BoundaryNorm(bounds, cmap.N)

    print(i,k)
    print("x image")
    # tell imshow about color map so that only set colors are used
    img = pyplot.imshow(inn,interpolation='nearest',cmap = cmap)
    
    # make a color bar
    pyplot.colorbar(img,cmap=cmap,
                norm=norm,boundaries=bounds,ticks=[min_val,mid_val,max_val])
    pyplot.show()

    print("ground truth label")
    # tell imshow about color map so that only set colors are used
    img = pyplot.imshow(yy,interpolation='nearest',cmap = cmap)
    
    # make a color bar
    pyplot.colorbar(img,cmap=cmap,
                norm=norm,boundaries=bounds,ticks=[min_val,mid_val,max_val])
    pyplot.show()

    print("prediction label")
    # tell imshow about color map so that only set colors are used
    img = pyplot.imshow(pre,interpolation='nearest',cmap = cmap)
    
    # make a color bar
    pyplot.colorbar(img,cmap=cmap,
                norm=norm,boundaries=bounds,ticks=[min_val,mid_val,max_val])
    pyplot.show()



