# Video Classification

The repository builds a **quick and simple** code for video classification (or action recognition) using PyTorch. A video is viewed as a 3D image or several continuous 2D images (Fig.1). Below are two simple neural nets models:


## Dataset
e-MARG(Road- Surface Video Classification)

## Models 

### 1. 3D CNN (train from scratch)
Use several 3D kernels of size *(a,b,c)* and channels *n*,  *e.g., (a, b, c, n) = (3, 3, 3, 16)* to convolve with video input, where videos are viewed as 3D images. *Batch normalization* and *dropout* are also used.


### 2. **CNN + RNN** (CRNN)

The CRNN model is a pair of CNN encoder and RNN decoder (see figure below):

  - **[encoder]** A [CNN](https://en.wikipedia.org/wiki/Convolutional_neural_network) function encodes (meaning compressing dimension) every 2D image **x(t)** into a 1D vector **z(t)** by <img src="./fig/f_CNN.png" width="140">

  - **[decoder]** A [RNN](https://en.wikipedia.org/wiki/Recurrent_neural_network) receives a sequence input vectors **z(t)** from the CNN encoder and outputs another 1D sequence **h(t)**. A final fully-connected neural net is concatenated at the end for categorical predictions. 
  
  - Here the decoder RNN uses a long short-term memory [(LSTM)](https://en.wikipedia.org/wiki/Long_short-term_memory) network and the CNN encoder can be:
    1. trained from scratch
    2. a pretrained model [ResNet-152](https://arxiv.org/abs/1512.03385) using image dataset [ILSVRC-2012-CLS](http://www.image-net.org/challenges/LSVRC/2012/).



## Training & testing
- For 3D CNN:
   1. The videos are resized as **(t-dim, channels, x-dim, y-dim) = (28, 3, 256, 342)** since CNN requires a fixed-size input. The minimal frame number 28 is the consensus of all videos in UCF101.
   2. *Batch normalization*, *dropout* are used.
   
- For CRNN, the videos are resized as **(t-dim, channels, x-dim, y-dim) = (28, 3, 224, 224)** since the ResNet-152 only receives RGB inputs of size (224, 224).

- Training videos = **9,990** vs. testing videos = **3,330**

- In the test phase, the models are almost the same as the training phase, except that dropout has to be removed and batchnorm layer uses moving average and variance instead of mini-batch values. These are taken care by using "**model.eval()**".



### 0. Prerequisites
- [Python 3.6](https://www.python.org/)
- [PyTorch 1.0.0](https://pytorch.org/)
- [Numpy 1.15.0](http://www.numpy.org/)
- [Sklearn 0.19.2](https://scikit-learn.org/stable/)
- [Matplotlib](https://matplotlib.org/)
- [Pandas](https://pandas.pydata.org/)
- [tqdm](https://github.com/tqdm/tqdm)





## Device & performance 

- The models detect and use multiple GPUs by themselves, where we implemented [torch.nn.DataParallel](https://pytorch.org/tutorials/beginner/former_torchies/parallelism_tutorial.html).

- A field test using 2 GPUs (nVidia TITAN V, 12Gb mem) with my default model parameters and batch size `30~60`.

- Some **pretrained models** can be found [here](https://drive.google.com/open?id=117mRMS2r09fz4ozkdzN4cExO1sFwcMvs), thanks to the suggestion of [MinLiAmoy](https://github.com/MinLiAmoy?tab=repositories).


 network                | best epoch | testing accuracy |
------------            |:-----:| :-----:|
3D CNN                  |   4  |  50.84 % | 
2D CNN + LSTM           |  25  |  54.62 % | 
2D ResNet152-CNN + LSTM |  53  |**85.68 %** |      




