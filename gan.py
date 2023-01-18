import os
import numpy as np
from tqdm  import tqdm 
import matplotlib.pyplot as plt
from keras import initializers
from keras.layers import Input,LeakyReLU,Conv2D
from keras.models import Model,Sequential
from keras.layers.core import Reshape,Dense,Dropout,Flatten
from keras.datasets import mnist
from keras.optimizers import Adam
from keras import initializers


np.random.seed(1000)
randomDim=100
adam=Adam(lr=0.0002,beta_1=0.5)

def plot_loss(epoch):
    plt.figure(figsize=(10,8))
    plt.plot(dlosess)
    plt.plot(glosess)
    out=os.path.join("./gan",f"simple_loss{epoch}.png")
    plt.savefig(out)

def plot_greratedimage(epoch,example=10,dim=(4,4),figsize=(4,4)):
    noise=np.random.normal(0,1,size=[example,100])
    generatorimage=generator.predict(noise)
    generatorimage=generatorimage.reshape(example,28,28)
    generatorimage*=255 
    plt.figure(figsize=figsize)
    for i in range(generatorimage.shape[0]):
        plt.subplot(dim[0],dim[1],i+1)
        plt.imshow(generatorimage[i])
    plt.tight_layout()
    out=os.path.join("./gan",f"simple_gan_loss{epoch}.png")
    plt.savefig(out)

def save_mpdle(epoch):
    generator.save(f'./simple_generator{epoch}.h5')
    discriminator.save(f'./simple_gan_generator{epoch}.h5')

def create_generator_modle():
    
    #create generator modle  
    modle=Sequential([
    Dense(256,input_dim=randomDim,kernel_initializer=initializers.RandomNormal(stddev=0.01)),
    LeakyReLU(0.2),
    Dense(512),
    LeakyReLU(0.2),
    Dense(1024),
    LeakyReLU(0.2),
    Dense(784,activation='tanh'),
    ],name="generator_modle")
    #set optimizer and loos funaction
    modle.compile(optimizer='adam',loss='binary_crossentropy')
    return modle

def create_discriminator_modle():
    modle=Sequential([
    Dense(1024,input_dim=784,kernel_initializer=initializers.RandomNormal(stddev=0.01)),
    LeakyReLU(0.2),
    Dropout(0.3),
    Dense(512),
    LeakyReLU(0.2),
    Dropout(0.3),
    Dense(256),
    LeakyReLU(0.2),
    Dropout(0.3),
    Dense(1,activation='sigmoid'),
    ],name="discriminator_modle")
    #set optimizer and loos funaction
    modle.compile(optimizer=adam,loss='binary_crossentropy')
    return modle



(x_train, y_train), (_, _) = mnist.load_data()


x_train=x_train[:]
y_train=y_train[:]

#convert to numpy array
x_train=np.array(x_train)
y_train=np.array(y_train)
print(x_train.shape[0])
#normalize
x_train=(x_train.astype(np.float32))/255
x_train=x_train.reshape(x_train.shape[0],-1)

#create generator and generator modle
generator=create_generator_modle()
discriminator=create_discriminator_modle()

#combine network
discriminator.trainable=False
ganInput=Input(shape=(randomDim,))
x=(generator(ganInput))
ganOutput=discriminator(x)

gan=Model(inputs=ganInput,outputs=ganOutput)
gan.compile(optimizer=adam,loss='binary_crossentropy')


if not os.path.exists("./gan"):
    os.mkdir("./gan")
dlosess=[]
glosess=[]

epoch=500
bachsize=128
print(x_train.shape)
bachcount=x_train.shape[0]//bachsize

for e in range(epoch+1):
    print("epoch ",e)
    for _ in tqdm(range(bachcount)):
        noise=np.random.normal(0,1,size=[bachsize,randomDim])
        #print(">>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>",noise.shape)

        imagebach=x_train[np.random.randint(0,x_train.shape[0],size=bachsize)]
        #print(">>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>",imagebach.shape)
        """plt.imshow(imagebach*2)
        plt.show()"""

        generator_image=generator.predict(noise)
        print("generator image:",generator_image)
        """plt.imshow(generator_image*255)
        plt.show()"""
        print(">>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>",generator_image.shape)
        x=np.concatenate([imagebach,generator_image])
        ydis=np.zeros(2*bachsize)
        ydis[:bachsize]=0.9
        discriminator.trainable=True
        dloss=discriminator.train_on_batch(x,ydis)

        noise=np.random.normal(0,1,size=[bachsize,randomDim])
        ygen=np.ones(bachsize)
        discriminator.trainable=False
        qloss=gan.train_on_batch(noise,ygen)

    dlosess.append(dloss)
    glosess.append(qloss)

    if e==1 or e%5==0:
        plot_greratedimage(e)
        save_mpdle(e)
plot_loss(e)