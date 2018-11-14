# -*- coding: utf-8 -*-
"""
Created on Wed Jul 26 19:12:56 2017

@author: Erdig
"""
import pandas as pd 
from skimage.io import imread, imshow, show
import numpy as np
import matplotlib.pyplot as plt


#Read the image 'maas2.jpg' using  imread
img = imread('maas2.jpg',as_grey=False) #true olsa siyah beyaz okuyor

plt.figure(figsize = (16,8))
#Display the image using imshow
imshow(img)
plt.show()

#Display the size of the image and one of the channels
print ('shape of the image'+str(np.shape(img)))
print ('shape of the red component'+str(np.shape(img[...,0]))) 
#0.299R+0.587G+0.114B gibi bir formul var, mesela sadece kırmızıyı gormek için 1,0,0 vektörü ile çarpılır

#Create red, green, blue and gray multipliers
red_multiplier = np.array([1,0,0],dtype=np.uint8) #Should include only the red channel
green_multiplier = np.array([0,1,0],dtype=np.uint8) #Should include only the blue channel
blue_multiplier = np.array([0,0,1],dtype=np.uint8) #Should include only the green channel
gray_multiplier = np.array([0.299,0.587,0.114],dtype=np.uint8) #Should correspond to 0.299R+0.587G+0.114B

fig, (ax1, ax2, ax3, ax4) = plt.subplots(ncols=1, nrows=4, figsize=(16, 40), sharex=True, sharey=True)
#Display the 4 images
ax1.imshow(img*red_multiplier)
ax2.imshow(img*green_multiplier)
ax3.imshow(img*blue_multiplier)
img1=np.dot(img,gray_multiplier) #tum chabeller aynı olunca siyah-beyaz

img[...,0]=img1
img[...,1]=img1
img[...,2]=img1
ax4.imshow(img)
show()

#Read 'maas.jpg' as a greyscale image. Display the shape and the image itself
img=imread('maas.jpg',as_grey=True)
fig, ax = plt.subplots(figsize=(15, 10))
imshow(img)
show()

print np.shape(img)

#Calculate the gradient of img (gradient iki yöndeki değişimlere bakıyor)
x=np.gradient(img)
print np.shape(x)
#Display the horizontal gradient
fig, ax = plt.subplots(figsize=(15, 10))
imshow(abs(x[0]))
show()

#Display the vertical gradient
fig, ax = plt.subplots(figsize=(15, 10))
imshow(abs(x[1]))
show()

#Display the sum of these two gradients
fig, ax = plt.subplots(figsize=(15, 10))
imshow(abs(x[0])+abs(x[1]))
show()

#CHARACTER RECOGNITION FROM IMAGES
import pandas as pd 
from skimage.io import imread
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm

def read_data(typeData, labelsInfo, imageSize, path):
    #Intialize x  matrix
    x = np.zeros((labelsInfo.shape[0], imageSize))
    for (index, idImage) in enumerate(labelsInfo["ID"]):
    #Read image file
        nameFile = "{0}/{1}/{2}.Bmp".format(path, typeData, idImage)
        #Read the file, reshape and add to x
        img=imread(nameFile,as_grey=True)
        img=np.reshape(img,(1,400))
        x[index,:]=img
    return x

imageSize = 400 # 20 x 20 pixels

#Set location of data files , folders
path = 'C:\\Users\\Erdig\\Desktop\\sabanci\\Practical Case Studies\\7.ders'

#This is necessary to retrieve names and labels of training images
labelsInfoTrain = pd.read_csv("trainLabels.csv")
print labelsInfoTrain.head()

#This is necessary to retrieve names of testing images. This time we only read information about test data ( IDs ).
labelsInfoTest = pd.read_csv("sampleSubmission.csv")
print labelsInfoTest.head()

#Read training matrix. Each image is reshaped as a 1 x 400 array
xTrain=read_data('trainResized',labelsInfoTrain,imageSize,path)
#Read test matrix
xTest=read_data('testResized',labelsInfoTest,imageSize,path)

print np.shape(xTrain)
print np.shape(xTest)

#Read and print the colored image
img = imread('trainResized/3.Bmp',as_grey=False)
imshow(img)
plt.title('Original')
show()
#Read and print the greyscale image
img = imread('trainResized/3.Bmp', as_grey=True)
imshow(img)
plt.title('Grayscale')
show()
#Print the inverted greyscale image
imshow(1-img)
plt.title('Grayscale Inverted')
show()

"""
In order to convert a label to an integer, we use ascii codes. map(ord, ....) converts characters 
to their values from ascii conversion table, and map(chr, ....) converts an integer to corresponding
character.
"""
yTrain = map(ord,labelsInfoTrain['Class'])
print (yTrain)
yTrain=np.array(yTrain)

from sklearn.ensemble import RandomForestClassifier

rf = RandomForestClassifier(n_estimators=100)

#Fit, predict and convert the obtained classes to ascii codes
rf.fit(xTrain,yTrain)
y_pred=rf.predict(xTest)
print y_pred
y_pred=map(chr,y_pred)
labelsInfoTest['Class'] = y_pred

from sklearn.neighbors import KNeighborsClassifier

knn = KNeighborsClassifier(n_neighbors=3)

#Fit, predict and convert the obtained classes to ascii codes
knn.fit(xTrain,yTrain)
y_pred=knn.predict(xTest)
y_pred=map(chr,y_pred)

labelsInfoTest['Class'] = y_pred
labelsInfoTest.to_csv('Results_knn.csv', index=False)

###FEATURES IMPORTANCE
features = rf.feature_importances_
features=np.reshape(features,(20,20))

#Plot the importances of features
plt.imshow(features, interpolation='none')
#Add a colorbar
cb=plt.colorbar()
cb.set_ticks([rf.feature_importances_.min(), rf.feature_importances_.max()]) 
cb.set_ticklabels(['not important', 'important'])  # put text labels on them
plt.show()

##IMPROVEMENTS
#Add the negatives of images to the dataset)enrich the datasets
xTrain=np.append(xTrain,1-xTrain,axis=0)
print np.shape(xTrain)
print np.shape(yTrain)
yTrain=np.append(yTrain,yTrain,axis=0)
print np.shape(yTrain)

rf.fit(xTrain,yTrain)
y_pred=rf.predict(xTest)
print y_pred
y_pred=map(chr,y_pred)
labelsInfoTest['Class'] = y_pred

features = rf.feature_importances_
features=np.reshape(features,(20,20))

#Plot the importances of features
plt.imshow(features, interpolation='none')

#Scaling(arka ve önyüzdeki farklılığı arttırıp görüntüyü yükseltiyor)

"""
Used in order to normalize data:
Input: Grayscale image
Output: Same image converted to range white and black (0-1)
"""
def scale(arr):
    """
    Linear normalization
    http://en.wikipedia.org/wiki/Normalization_%28image_processing%29
    """
    arr = arr.astype('float')
    minval = arr.min()
    maxval = arr.max()
    if minval != maxval:
        #Do the scaling
        arr=arr-minval
        arr=arr/(maxval-minval)
    return arr

img = imread('trainResized/3.Bmp', as_grey=True)
imshow(img)
print('Minimum value of a pixel ' + str(img.min()))
print('Maximum value of a pixel ' + str(img.max()))

img=np.reshape(img,(1,imageSize))
img = scale(img)
#To avoid numerical instabilities
img[img >= 1] = 1
img[img <= 0] = 0

print('Minimum value of a pixel ' + str(img.min()))
print('Maximum value of a pixel ' + str(img.max()))

# Print the rescaled image and its negative
#Rescale the datasets
for i in range(len(xTrain)):
    xTrain[i,:]= scale(xTrain[i,:])
    
for i in range(len(xTest)):
    xTest[i,:]= scale(xTest[i,:])

xTrain[xTrain > 1] = 1
xTrain[xTrain < 0] = 0
xTest[xTest > 1] = 1
xTest[xTest < 0] = 0

rf.fit(xTrain,yTrain)
y_pred=rf.predict(xTest)
print y_pred
y_pred=map(chr,y_pred) #asci kodu harfe çevirmee
labelsInfoTest['Class'] = y_pred
#Draw the feature importances
cb=plt.colorbar()
cb.set_ticks([rf.feature_importances_.min(), rf.feature_importances_.max()])  # force there to be only 3 ticks
cb.set_ticklabels(['not important', 'important'])  # put text labels on them
plt.show()


##0-1 gibi değerler ile daha görünür resimlerrr
img = imread('trainResized/3.Bmp', as_grey=True)
img=np.reshape(img,(1,imageSize))
img = scale(img)
#To avoid numerical instabilities
img[img >= 1] = 1
img[img <= 0] = 0
img=np.reshape(img,(20,20))

img=(img>=0.5)*1.0 #true-false matris döndürüyor, 1 ile çarpınca 1 ya da 0 yapıyor
imshow(img)
plt.title('Binarized')
show()


#Neural network
imageSize = 400 # 20 x 20 pixels

#Set location of data files , folders
path = 'C:\\Users\\Erdig\\Desktop\\sabanci\\Practical Case Studies\\7.ders'

#This is necessary to retrieve names and labels of training images
labelsInfoTrain = pd.read_csv("trainLabels.csv")

#Read training matrix. Each image is reshaped as a 1 x 400 array
xTrain = read_data("trainResized", labelsInfoTrain, imageSize, path)

#This is necessary to retrieve names of testing images. This time we only read information about test data ( IDs ).
labelsInfoTest = pd.read_csv("sampleSubmission.csv")

#Read test matrix
xTest = read_data("testResized", labelsInfoTest, imageSize, path)

yTrain = map(ord, labelsInfoTrain["Class"])
yTrain = np.array(yTrain)

from sklearn.neural_network import MLPClassifier
mlp = MLPClassifier(hidden_layer_sizes =(400,400))
mlp.fit(xTrain, yTrain)

y_pred = mlp.predict(xTest)
y_pred2 = map(chr, y_pred)
labelsInfoTest['Class'] = y_pred2
labelsInfoTest.to_csv('Results_mlp.csv', index=False)
