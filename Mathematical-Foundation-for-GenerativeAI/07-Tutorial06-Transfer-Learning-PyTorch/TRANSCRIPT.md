# Transcript — Tutorial 6 : Transfer Learning with PyTorch

> **Source:** https://www.youtube.com/watch?v=ETJG9mmeL5k  
> **Channel:** NPTEL - Indian Institute of Science, Bengaluru  
> **Duration:** ~29 min  
> **Note:** Auto-captions cleaned lightly. Minor ASR errors possible.

---

**[00:03]** Hello guys, welcome to this tutorial in which we'll be looking at how do we load and use pre-trained models. So, till now we have in all our previous uh tutorial sections, now we have seen that how do we work with PyTorch. Now, by now it should be pretty much clear that all these layers which are there now can be thought of

**[00:31]** as LEGO blocks. And then you can uh put them together. The only uh point that needs to be remembered is there should not be any mismatch in terms of the shape. If that is taken care, now we can easily go ahead and fit things. Okay. Now, whether it is needed or not from the perspective of uh architecture, that's totally a different question altogether, but

**[01:00]** as per the programming construct, there is nothing stopping us from doing it. The only issue is there should not be any kind of a shape mismatch. If you're taking care of that, you can mix and match all these layers. So, now on this specific idea, now let's look into the pre-trained models. Now, uh I'm assuming that all of you have the basic idea of networks.

**[01:31]** I'm spe- specifically speaking on different CNNs. So, you are comfortable with different networks like AlexNet, VGG, ResNet architecture. Now, for the sake of completeness, I'll just go ahead into a very brief discussion of the same. Okay. So, now let's look into the the AlexNet architecture. &gt;&gt; So now here, this is the standard

**[02:06]** Wikipedia. So this is the LeNet now which was developed to take care of uh the MNIST data. Now, you have this 28 cross 28. So the number of channels is one now as you can see it here. So and then you can see that the first convolution has 5 cross 5 kernels, two padding. So they haven't specified how many uh uh filters are there. So since the

**[02:33]** number of channels in the output has six, so we should infer that there are six filters and then they haven't specified the stride. Therefore, we should consider that the stride is one. And then we have a 2 cross 2 average pooling with two stride now which will uh reducing the the spatial dimension by half and so on and so forth. You know, you should be able to read through this. This is the AlexNet architecture. Now wherein which uh the input is 224

**[03:01]** cross 224 cross 3. So now uh it should be pretty much clear that why is this uh 224 224 3 is uh quite an important number to us. The reason is this is the the input size of an ImageNet data. Okay. The first convolutional layer has a kernel of shape 11 cross 11 with stride four and then it has 96 filters. Since padding is not specified, so there is no padding. So padding is zero.

**[03:30]** See, uh whenever we are not specifying the stride, now we mean stride is one. Whenever we are not specifying the padding, now it means that the padding is zero. And then we go ahead with 3 cross 3 max pooling with two stride. Okay. See, earlier we had non-overlapping pooling. Now we have a overlapping pooling. So we should be able to compute the size now based on this using the formulas which we have discussed in our earlier sections. Now, so on and so forth. And then you have

**[03:56]** this ReLU, uh, what do you call activation function? And then you go ahead, and this is the final size. And that you flatten it out. Okay? So, that you flatten it out and convert it into a vector. And then that is converted into a 4096, uh, length vector. And then you have a dropout. For the sake of regularization, we are using a dropout of 50%. And then we are using the activation

**[04:23]** function ReLU. And then again, 4096 to 4096, and 4096 to 1000. Now, why is this 1000? Now, because as I told you, these were the networks that were designed for a 1000 class problem. Okay? So, we should, uh, have it a 1000, uh, length vector. Now, we should get it. Now, this is the outcome is a logits. That logits you pass it through a softmax to get the the probability vector. So, now in this specific architecture, now let's take any other, uh, data, uh, which is there.

**[04:52]** Now, let's take, uh, some other data which has four classes. Now, then Now, the only thing that you have to modify is rather than, uh, the final, uh, fully connected layer, the classification head, now which is having, uh, 1000, uh, nodes as the output, should be able to give four nodes. Okay? Should be having the four nodes. So, you should be modifying this last layer. Okay? So, now on the other hand, if your input size is different, now the preferred choice is

**[05:21]** reshape the input to match this. Now, otherwise, all these internal computations needs to be modified. Okay? So, now if it is not at all possible to change the shape from whatever might be the shape to 224, now then you have to modify the whole architecture. Now, then, using a pre-trained model is of no use. Okay? Now, the use of pre-trained model makes sense whenever input shape is, uh, 224 224 3, and then the number of nodes is the only thing that you're

**[05:51]** modifying at the last layer. Okay. Now, this is AlexNet for us. And then one more very important architecture that is being used is uh VGGNet. So, let's look at the original paper itself and then uh try to understand the Okay.

**[06:23]** Now, as we can see from this thing, now VGG comes in six different variants. Okay. So, A, A LRN, this LRN represents uh layer normalization. B, C, D, E. Okay. So, now it has 11 layers, 11 layers with uh layer normalization, 13 layers, 16 layers, 16 layers, and then 19 layers. So, the input is 224 224 RGB image, so which is that.

**[06:52]** In the first layer, now it has a convolutional layer, now which is having a 3 cross 3 kernel, okay, and then 64 such filters. Okay, 64 filters are there. See, here we are not specifying the stride. Now, whenever the stride is not specified, now it should be pretty much clear that it is stride is one and the padding is zero. That means there is no padding. So, the input channel, now we should be

**[07:19]** looking at from the top, so it is a RGB image, so the input channel is three, the output channel is 64. Now, we should be able to declare the network architecture similarly. And then after that you have this max pooling here. Okay. So, max pooling they haven't specified the stride as well as the kernel size. Now, that means that you should be taking a 2 cross 2 uh kernel size with stride as two, the standard one. And then after that you have this uh the input is 64 channels, the output is 128 channels, so three cross three

**[07:49]** kernel with stride one and no padding is this layer. Again, you have the max pooling. Now, you should consider uh the two cross two kernel and stride as two. And then followed by again a two convolutional layers. So, each of them having 256 filters. So on and so forth. Now, whatever is the output, now finally you flatten it out. Post max pooling, you flatten it out. And then 4096 to 4096 to 1000. Okay? And finally you have this softmax.

**[08:20]** Now, you can see it here, the number of parameters it is in millions. So, the number of parameters in A and A LRN is 133 million, in B is 133, in C is 134, in D is 138, E is 144 million parameters are there. Now, this is how uh we go ahead and work with the the VGG. Okay? So, similarly you can read the network architecture. Uh so, I have just taken as a representative VGG 11. Similarly, you can read these. Now, this is VGG. And

**[08:50]** another famous architecture &gt;&gt; [clears throat] &gt;&gt; that we have is uh ResNet. See, the idea of ResNet is coming out of the skip connections. So, I'm assuming that all of you are uh comfortably aware the need of skip connection, why is it needed, and how does skip connection brings in

**[09:17]** stability. Otherwise, this is the right time for you to look at those ideas. So, now the main idea in uh ResNet is now you add after certain transformation, you add the untransformed data as well. Okay? Now, that means that the model can choose now whether this particular transformation is needed. Okay? So, that is the overall idea.

**[09:46]** So, now the way in which it has been done is for every two layers they are giving this skip connection. As you can clearly see from this diagram okay? After each of the two layers now you are taking the untransformed input at that point and then you're adding it. So, now we can see here that the uh

**[10:13]** untransformed uh data untransformed information is there is given. So, the transformed data is added with the untransformed data. Okay? Or untransformed uh features. Okay? So, now like this for every two layers now they will do this. So, now let's look at the representations of this. Now, the ResNet is uh famously with uh

**[10:42]** five variants. Now, these are the five variants that were described in the main paper. It doesn't mean to say that only these five variants needs to be looked into. Now, you can come up with uh so, other variants with let's say ResNet 20, ResNet 22, or ResNet 32. Uh now, that that's totally now fine. Now, using the same idea we can come up with uh &gt;&gt; [clears throat] &gt;&gt; different number of layers in this residual nets. So, now here you have this 3 cross 3

**[11:09]** kernel and then you have this uh number of channels number of filters that are there is 64. Okay? Now, the way in which it works is now first you go ahead with this 7 cross 7 kernel 64 uh channels stride is two padding is not provided therefore the padding is zero followed by a 3 cross 3 max pooling with stride two. It's an overlapping pooling that is there. And then now you have this three cross three kernel with 64 filters.

**[11:36]** So stride is one, padding is zero. And the input is 64 channels, the output is also 64 channels. And then it is off Why is this put in these brackets is offer these two there's a skip connection. Okay. And like this it is repeated twice. So now the four layers here, four layers here, four layers here, four layers here. So totally 16 and this is one 17 and this one fully connected layer now will be there. Now totally this will sum up to

**[12:05]** number of layers to be 18. So now like this now we will have the layers. Now similarly you can look at the architectures of different other variants of residual networks. Okay. Now Now let's try to load these models which are there. For our use Now what we shall be doing is these trained weights are readily available in

**[12:34]** the PyTorch library. So they were trained on ImageNet data. &gt;&gt; [snorts] &gt;&gt; Now you can directly use them and rather than starting with a random initialized random weights now we can start with these trained weights. Okay. Now that will be good. Now you can think of it like this. So one good example that can be taken on this procedure is Now all of us started when we started

**[13:01]** riding uh two uh cycles the major issue that we had was how do we balance? Okay. But once we transferred from the cycle to two wheelers the main issue was not balancing anymore. Now because the idea of balancing was already learned by us. So we transformed we transferred the idea and this is a kind of transfer learning that we can use.

**[13:30]** Okay, already we have the weights that can recognize certain features. And on that we can go ahead and fine-tune. So now when you go for riding a two-wheeler, now you will try to understand the mechanics of two-wheeler. The same thing can be applied for someone who started with a two-wheeler, now which is having these clutch and gear. So when they go to this manual transmission cars, now for them understanding the clutch and then the gear balance will be very comfortable.

**[14:00]** Now compared to someone who is not familiar with the clutch and then the gear system. Now if they have for for their previous things they have only driven some mopeds. So now if they go to manual transmission cars, the major issue for them will be now how do you balance that clutch and gear. But for someone who has already comfortable with that, when they go into the manual transmission car, they will be very much comfortable. Now this is an analogy. So So you can think of this

**[14:28]** analogy now for going ahead and understanding the idea of transfer learning. Now similarly, we already have the weights which are learned on a huge chunk of data. Now ImageNet has 1,000 classes and each class has 1,300 images. Now apparently the whole data itself is around 200 GB to train. Now you have a trained weights on such a huge humongous amount of data. Now rather than starting from a random

**[14:56]** initialized point, now we can start from this and then we can train it. Now at that time we use a different terminology, we call it as fine-tuning. Okay. Now because you are already starting with some pre-learned weights. Now these weights are made available to us. So we just take them and then start from those weights. And similarly, the next architecture which is there is what is known as a ConvNext architecture. Now in that architecture, the main difference is

**[15:24]** rather than having a 3 cross 3 kernels, now because of which the so once you have a image which is of size 224 cross 224, so if you are only looking at a 3 cross 3, it is a it is looking at a very small area. So, what they did is they increase the kernel size. Okay. Now, you can look out for the network architecture of the ConvNext also. Okay. So, now coming back to where we started, now our aim was to somehow load these weights and go

**[15:54]** ahead and try to do a classification. It can be used for any other task. You can change the the head which is there. So, now we in this tutorial section, we'll be going ahead with an MRI data. We'll be taking an MRI data. We take this preloaded weights and then we will perform the classification. Now, here the number of classes is rather than 1,000, the number of classes we have is four. Okay. Now, that means that we should be only changing the last layer. Okay. And

**[16:22]** we'll be reshaping the data into 224 cross 224 cross 3 so that you don't need to change the internal computations. Okay. Now, these are all the standard libraries. I'm assuming that by now you should be very much comfortable with these. Okay. So, now I have defined a simple MLP. So, we start with a simple MLP. So, which is a the image size the in features is 3 into image size I'll specify as 224. So, 2 512 512 to 128 128 to number of classes.

**[16:51]** Okay. This is a simple MLP just to show the contrast. And then I have used a simple convolutional uh neural network. Okay. Now, I have a convolutional 2D followed by a max pooling, convolutional 2D followed by a max pooling. Three convolutional networks three convolutional layers are there. &gt;&gt; [snorts] &gt;&gt; So, where in which the first one is having 32 filters, 64 filters, and 128 filters. These are chosen at random. Okay. So, and then you have a classifier and then

**[17:19]** head now which is there to classify it into the number of classes that we give. Okay. So now now similarly now I have uh gotten couple of uh helper functions to go ahead and load these uh things. Okay. Now before that maybe it would be a good point. Hold on. Yeah, here. Yes. Yeah, I should have gone here. Yeah, sorry for that. That's fine.

**[17:49]** Yes. Ah. Yeah, I should have gone from here. That's fine. So now where does these pre-trained models are? Now from torchvision import models. Okay. Now what are the models we are getting? We are getting AlexNet, VGG19 and ResNet18. Now all the three architectures which we just saw. Okay. So now AlexNet weights uh So is equal to models. AlexNet weights default. Now you load these uh models. So

**[18:16]** you get the instance as well as the weights loaded. Okay. Now what is the data that uh we are using? So we are using this uh brain tumor MRI data set which is available in Kaggle. I'm loading the the Kaggle import. Okay. And uh this is the path in which the data is uh available. Okay. So now it has started loading all those uh networks. Okay.

**[18:45]** So and then here I'm going ahead with the transformations. Torch data loader random split data sets transforms known libraries for us. Now the image size that we have is 224. The train transform what is the transformation that we are going ahead? Now we are resizing the image to image size. Now what is the image size I have given which is 224. Adhering to the notations that we have. So you resize it to 224 cross 224. And then we have what is this known as

**[19:13]** random horizontal flip. Now, this can these kinds of transformations which are there can be looked at as regularizers. So, now what we do is now we take couple of randomly chosen images and then we perform the horizontal flip. Okay. Now, you can also have a vertical flip also. Perfectly fine. So, just for the demonstration, I have used this

**[19:43]** horizontal flip. And then you convert them into a tensor. And then you normalize it with respect to this mean, these three means. So, this is for channel one. It's three channel image, right? Channel one, channel two, and channel three. Now, this is the standard deviation with channel one, channel two, and channel three. Okay. Now, you go ahead and normalize with this. Okay. Now, how did we get these numbers? Now, these are ImageNet stats. So, this is a Since we are going ahead with the preloading of the ImageNet data, it would be a good

**[20:12]** starting point for us. Okay. And similarly, now in the test data, now the transformation which is needed is the shape should match and the normalization should be there. Now, we don't normally go ahead with any of these kinds of horizontal flips and things. So, test data, we'll just take it as it is given. The only thing that we do is the data we will resize it and then we will normalize it. Okay. Now, this can be taken it as a small exercise. Now, have different uh stats.

**[20:41]** Now, mean and standard deviation for training and testing data, have it for the same. Now, do this and see is there any change in terms of the evaluation metrics at the end. Okay. So, I'll leave it up to you as a small exercise that you can take it up. So, and then you will you are getting various in the folder, where is the training data and where is the testing data. And then from that you can get the data set. Okay. You just need to specify the transforms. Okay.

**[21:10]** So now here I just want to remind you that earlier we looked at a way in which we can come up with a custom data loaders, custom data sets. So whenever we specify the index, that is one way. We have an annotation file which is there and then we go ahead with it. This is another way. Now here in this training folder, now all the different classes are segregated. Okay. So now class one is one folder, class two is another folder,

**[21:38]** class three is another folder. Like that the segregation is there. Similarly in the testing as well. So now in those cases you don't need to specify and custom data set. Now you can directly use data sets. image folder and then specify the training directory now from which it can directly take the things. Okay, similarly train and the test data now you have. Now once you have them now you can have the validation ratio. You can split them.

**[22:05]** So okay, and then you convert them into the loaders. Okay. Uh so now you can see here you have 32 in each train batch. The batch size is 32. The number of channels is three, 224 224 and this is the label. Okay. So now these are the the four classes which are there. Okay. So now let us print these models which

**[22:35]** are given to us. Okay. Print. Let's see I'm using the same variable AlexNet model. Okay. Now if I run this now you will see this

**[23:04]** This is an n. sequential. Now, inside this you have all these convolutional layers. After that, you have this classifier. Now, what is that we have to change? We have to just change this. You have to this whatever is the instance that instance of classifier of six. This is what you have to change. Now, similarly, the same thing goes with VGG 19 and ResNet 18. Now, that is what is being done here in these. Okay? Now, you get the pre-trained models.

**[23:31]** So, you go to this model.classifier.six. Now, whatever is the in features which is there, you change it. You change it to in features to number of classes which is there. You get the in features, now which is in our case which is 4096. Now, 4096 to 1000 was there, you change it to 4096 to four. The same thing is being done for VGG also. The same thing is being done for ResNet also. Now, I have used ConvNext also. And then I have used ViT. Now, as of now, since uh we haven't defined uh

**[24:01]** vision transformers, now just for uh the future reference, I have just kept it here. Now, we can just ignore it and we will come back to this uh vision transformer now once we done with the transformer architectures. Okay? Now, these are the uh generic methods that we have used. And then this is the standard training uh thing. So, and then initialization. So, the number of classes is four. I know your MLP, simple CNN, AlexNet, VGGNet, ResNet 18, ConvNext, tiny, and then ViT which

**[24:33]** is there. All of them have initialized. Now, here now uh I want to point out that have we used any dictionary similar to that at the beginning of our discussion? There I specifically said most of our usage will be similar to what we'll be using later. Okay? All the terminologies that we have been using from the beginning. Now, you can see that starting of the Python, we have been consistent with those

**[25:02]** terminologies. Okay? Now, because it it has to give a one 360° view of the same idea that will be resurfacing again and again. Okay, good. So, now you have declared all these things. So, I'm going ahead with just five epochs. So, with this particular learning rate, I have already run this. Okay? So, and then what do you do for each of the model? You go ahead and train this model. Now, with MLP, now you got around 82%

**[25:31]** accuracy. Okay? So, and then with simple CNN that we have, we got 90% accuracy. And then uh Mm, okay. Okay, maybe it is not loading. Now, similarly, now we can see that okay, maybe it is still training, but that's fine. So, maybe uh uh it will be loaded.

**[25:59]** So, what we can see here is as the the network complexity increases, now we can see that the accuracy improves. Okay? That can be immediately seen from this. So, what will be the accuracy of VGG, what will be the accuracy of uh so AlexNet, what will be the accuracy of ResNet? Now, once you run these, so you can go ahead and compare them. Okay? Now, this is how we go ahead with the

**[26:29]** pre-trained loading the weights of the pre-trained models. Okay? So, now please remember this. The only thing that you have to take care is the shapes. Okay? Now, once you're taking care of the shapes, now uh you can use them as LEGO blocks and then fit all of them together. Okay? So, now with this, we come to a nice point in which uh we will stop using

**[26:58]** what do you call not using we will stop discussing all these uh PyTorch ideas to an extent we know all the capabilities that are needed for us uh to be used in our future discourse that we will be going ahead. So now just to give you a small recap of what are the things that we have discussed till now.

**[27:24]** Now starting from creation of tensors till going ahead and using all these different pre-trained models. Now we have seen CNNs, we have seen RNNs, we have seen uh uh how do you load all these pre-trained networks and stuff. Okay. So [snorts] now going ahead we'll be implementing all these state-of-the-art generative models. And these are can be thought of as a

**[27:54]** the prerequisite that is needed. And now even if you uh what do you call ignore the the major aspect that is going to come. Now because that is what we are going ahead in the course, but these bunch of tutorials which are there starting from the introduction of Python till this PyTorch, this can be considered as one block now wherein which we have given a fairly detailed introduction

**[28:23]** on Python, NumPy, and PyTorch. Okay. Now this is where we stop the PyTorch basics tutorials. So in the coming next bunch of tutorials the plan is to go ahead and look into couple of numerical examples. Okay. So as we have been repeating that this course assumes that you have

**[28:51]** already taken a first level machine learning course and then you know couple of ideas. Now we'll be reinforcing them. Now till now we reinforced with respect to programming. Now we'll be reinforcing them again using equations and problem solving skills. That is what lies ahead of us next and then now we will start with our generative models tutorials. Okay? With this I conclude today's

**[29:18]** tutorial. I hope it was helpful. Meet you in the next bunch of tutorials. Thank [music] you all.
