# Transcript — Tutorial 14 Part 2 : Transfer Learning using CNNs

> **Source:** https://www.youtube.com/watch?v=vocN0cxAT7I  
> **Channel:** NPTEL - Indian Institute of Science, Bengaluru  
> **Duration:** ~28 min  
> **Note:** Auto-captions cleaned lightly. Minor ASR errors possible.

---

**[00:06]** Hello all, welcome to this tutorials in which we'll be continuing our discussion on CNN's. The main part of this tutorial is uh understanding how do you get a pre-trained model and then how do you infer from a pre-trained model and then how do you modify the pre-trained model and then how do you extract embeddings from a model. So now first let's try to understand how do you load a pre-trained model. See the

**[00:34]** idea of loading a pre-trained model is all these big big networks that came for that imageet challenge they came up with publishing their weights. They used to give the instance and as well as the weights of the model. Now because of which you can use those weights and rather than starting from a point which is random you can use those

**[01:05]** weights and then you can just fine-tune them. Okay. So now couple of networks that came are in this one is VG. Okay. Now here is how the VG uh networks look like. Now in VG what they did was they stacked a lot of uh uh convolution layers. Now let's try to understand. Now VG had these different versions. A A

**[01:34]** LRN. Now LRN is layer normalization. B C D E. Okay. These were the different versions in which A had 11 layers. Now you can count it 1 2 3 4 5 6 7 8 and then you have 9 10 11 layers totally okay including the layers in MLP. Now this is the same in addition to that there was a layer normalization here you

**[02:03]** had 13 layers so two added here you had 16 layers okay like this so so on and so forth 16 layers and 19 layers so let's try to understand this because now whenever we want to understand a convolutional layers now we have to know those hyperparameters. How many filters

**[02:30]** are there and then what is the size of what is the special extent of each of the filter? What is the stride and what is the padding since they haven't specified stride and padding at all in any of these network description. Now we can consider the stride is one and padding is zero. That means there is no padding. Now that leaves us what is the kernel size? What is the special extent of the filter and then how many filters were there. So you can see here con 364 that means that the size of the kernel

**[02:59]** is three and then there was in this layer there were 64 filters of them. Now this is given to a max pooling. Now whenever in max pooling also they haven't specified the kernel size and then the uh the stride and therefore you have to consider it the default value which is 2 +2 is the kernel size and then the stride is two that is given as an input to this. Now here the input channel is 64 the output is 128. So here you have 128 channels and then that is passed

**[03:30]** through a max pooling and then again you have a 3 +3 kernel where which you have 256 filters that 256 again is given to this layer. Okay this is input is 256 layers. Output is also 256 channels again a max pooling so on and so forth. So now whatever is the final output here that is converted to a vector and then you get 4096 you convert that vector into 496 496 to

**[03:59]** 4096 496 to,000 okay and then you pass it through a softmax to get the posterior okay now similarly we can extend the the idea of weights here also okay now then the question is they started from 11 and then came to 19. Why did they stop at 19? Okay, they would have went on stacking even more convolutional layers, right? What was the issue? Now, this issue is similar to the idea of uh

**[04:28]** the vanishing gradient that we saw in RNN. Okay, in our theory discussions uh because of too much of transformations now we need uh the the gradients become close uh near to zero. Okay. So because of which uh the impact the modification you cannot see in the initial layers because of which the learning will be not good. So now now how do you solve this problem? Now if if

**[04:57]** you stack if you just stack multiple convolutional layers now we used to see the vanishing gradients problem. Now then the whole question that arises is okay now how do you solve this problem? It's exactly the same way in which we solved the problem of uh vanishing gradients in RNN. Now we have a path in which we allow a function to learn identity. Now that means that you take a connection and untransformed input which is there you give it to the

**[05:26]** next stages. Okay, that is what ResNet does. Okay, that is residual connections. Here you can see you get an image. You have this initial convolution layer and then after that for each convolution layer you see that you see a connection here. Now when which untransformed identity operations are allowed okay you have this similarly for each two convolution layers you see this uh connections these are known as the

**[05:56]** skip connections. Now because of which you are allowing the passage of the untransformed uh uh input because of which you resolve the problem of the vanishing gradients. Okay. So now because of which here we were able to have a convolution layer with 152 layers quite deep actually. Okay. So in our uh uh thing we will be just looking at the

**[06:25]** first two things reset 80 and reset 34. We'll not be going with any of these. These are quite huge. Okay. So now then how is this there? Now you can see here first is you have this image which is 224 + 224 + 3 that is the input you have a 7 + 7 kernel you have 64 filters and the stride is two we don't have the uh padding okay now that is passed through a 3 + 3 pooling with stride two now this

**[06:54]** is common for all the networks and then we see for each of those columns now you have this this repeated twice now that means that after these two operations which is having 3 + 3 kernel size and 64 filters. Now stride is one and padding is zero. After these two you have a connection. Now that is the reason why they have mentioned this in a square brackets. Okay. Now after this there is a skip connection exactly similar to that. Okay. After two you have a skip

**[07:24]** connection like this. This is repeated twice. Now that means that you have four convolution layers here. Similarly here also you have a four convolution layer. Here also you have four. Here also you have four. Totally 16. This is 117. And then whatever is the output that you convert it into a,000 dimensional uh value using a fully controlled layer that is 18 16 17 18. So it is known as resonate 18.

**[07:52]** Okay. So you have 18 layers of weight. Now this is resonate 34 when which this particular uh is repeated thrice. Now as you can see here this color which is a 1 2 3 it is repeated thrice. Similarly here 1 2 3 4 you can see that this is repeated four times. The next one is repeated six times. So then three times. So like this now we this is how the rest connections work. Okay. The idea of skip connection now

**[08:20]** when which you are allowing the passage of the identity. Okay. So now let's uh we will not be looking at the implementation of them. We'll be directly taking the the instance which is already there in uh uh the PyTorch repository with the weights and then we'll be going ahead with that. Okay. Now let's do that. Now torch you need and then you need this uh f which is functional. Then you have torch vision domodels.

**[08:51]** Okay. Now what are the models that you are getting? You're getting VG19 and its weights. ResNet 18 and its weights, ResNet 34 and its weights. You're importing all these things. Okay. And then you're putting onto your device. Okay. You are enabling the device and the input is 2 + 32 224 224 224. Okay. Now that is this is your number of

**[09:20]** dimension. Now this is your uh special size 224 + 224. Now what is this two? This two is the batch size. Okay. Now, first always we have the batch followed by the input. Okay. I'm taking two examples, two dummy examples. I'm generating it randomly. Now see once I pass it now, if everything is fine, it will just give me some dummy output. That's fine. No problem. Okay. But that is a straightforward indication that the

**[09:49]** forward process is happening properly. And because the forward process is happening properly, now there is enough uh uh information to perform a backward pass. Okay. Now then how do we uh load the pre-trained models? Now VG weights VG weights dodefault. Now we use this reset weight reset weights. Default we use this for loading the weights. Okay. And then from this

**[10:19]** you create instance. So you got the weights VG 19. Okay, weights is equal to these VG weights and then you put it onto the device two device and then you are putting in evaluation mode. Now that means that uh you're not doing any kind of a training. Okay, similarly you get the weights. If you want to do some kind of a training, don't put this eval. That's all. Okay, you're porting onto the device.

**[10:47]** So because of which now you have uh instances of the VG, ResNet 18 and ResNet 34. And please remember that the input size is still 3 224 224. Okay. This is the imageet size. Okay. We are dealing with this that specific size itself. Okay. And then we are looking at the the forward pass VG model. You pass this dummy X that you have created. The output will be a vector with thousand

**[11:17]** values. This is a logits. Okay, the logits with thousand values. Similarly, you do it for reset 18. Similarly, you do for reset 34. I'm just extending them. And then you apply the softmax f that is a functional that we have used. Okay. F dots softmax of vg logits dimension is equal to one. That means that you consider it as a column and then apply the the soft max. Now this will give you the probabilities. Okay, this will give you the posteriors. Okay, and then you're

**[11:46]** just looking at its uh shape. Okay, all of them will be 1 + th00and uh you had two examples. So it will be 2 +,000. Okay, which makes sense. Okay, that's what you're seeing here. So now uh this is how you just pass the data and obtain the things. Now whenever we are dealing with a thousand class problem now there is one thing that normally people have it is uh they will take top five uh thing see what happens

**[12:15]** is now if you have a 10 class problem now then uh uh on an average now you will be having equ if you go ahead with an equally likely assumption now that means that each uh posterior that is p of y is equal to some i given x now it will be 0.1 if you have a thousand class it will be even far lesser okay and therefore they take top five uh values now I'll just show you how to do it this

**[12:44]** is a simple uh way to do it now we are printing the top five now you get the categories now why do you need the categories uh now because this is thousand class problem and then all these categories are predefined so we know what are those thousand classes okay it's already there so and then now it is seeing that for probability dot size of zero. Now what is the probability dot size? Now that is two of zero that is if you see it it is 2,000.

**[13:13]** So zero is this two. So that means that you're repeating it twice. You get the top five probabilities and the respective index torch top probability of I k is equal to five. Now this will give you the top k now that is top five that you're going ahead and what is the model and which sample you are looking at it. enumerate and then gives you now what is the uh category what is the probability and then what is the class

**[13:40]** ID okay that you can see it here whenever we did it for the VG sample you get it as no velvet which is the ID with 885 and then the probability is 0.04 04 this is the maximum probability this is random images just giving so and then the label is walk the ID is 909 so on and so forth okay this is for sample zero and then similarly for sample one now

**[14:10]** you can see you can get the categories as well as the probabilities so that we can look at them and please remember that these where the probability distribution is uh across thousand examples, thousand classes. Okay. Fine. So this is how you take a pre-trained model. These are the pre-trained models. Obtain their weights and then do the inference. Okay. Now if you want to do the training that is also fine. So you can use the

**[14:37]** same uh model weights and then you can start the training. But here there is one thing that is the number of classes is fixed. Why am I saying the number of classes is fixed? because of the output. Now this is giving you a vector of thousand uh dimension. Now that means that you're using a thousand class problem. That may not be true always. Okay. Now sometimes so let's say that you want to go ahead and do something uh some X-ray or MRI uh classification. See

**[15:08]** in those cases you might not have thousand classes but rather than starting the training from the scratch you can take already pre-trained models which are there and then start with that. Okay. Now what is the use of it? See the major use of that is already whenever it's a trained model that means that it knows how to identify the basic uh shapes. Okay. It is uh empirically known that the early layers

**[15:36]** which are there will actually uh be recognizing the basic shapes like edges, corners and stuff. Okay. On top of it you build uh a thing. So since they are already trained they know how to recognize couple of things. Now rather than going ahead and uh uh retraining everything. So let's start from what we know. Okay. Now, now this is like uh now for example you have learned how to ride a bicycle. Now if you want to ride a

**[16:05]** bike now you don't start a fresh no you know how already how to balance yes the weight may be different but you already know how to balance so you can go ahead with that okay without balancing and then you can go ahead with that okay so now let's do that now how do you modify them see uh to modify them first we should understand so let me print this

**[16:35]** GG 19 model. So what happened? Oh, okay. So, okay, it will give me an error because I haven't defined because this is the runtime is gone. Let me run this. So, it's downloading. Just Just give it

**[17:16]** a moment. Okay. VG model. What is that? VG model. Not 19 model. It's just the So you see this feature extractor first

**[17:42]** and then you see this classifier. Now this classifier you want to modify this last one which is th00and 4096 to,000 which is there. Let's say that you have five classes. You want to modify this 4096 to 5. Okay. Now we are still assuming that your input size is 224 + 224 + 3. Okay, if that is not the case then you have to actually if if your input is of a different dimension you

**[18:09]** had to actually sit through and compute the output so that you change this the final uh feature size input feature size which is there. Okay. Now consider that if your input size is 152.152 now then the output size of all these operations will change because of which the input for the classifier in this change okay because of which you have to compute them again okay this is the so that means that assuming that your input

**[18:37]** is 224 + 224 + 3 you have to modify this sixth layer in the classifier similarly if I take the resonate in a similar manner So is this reset 18 model if I can uh print them and see. So layer four now finally so you have to modify this FC resonate

**[19:05]** model of FC now which is 512 to,000 which is there I have to modify 512 to some other uh size whatever if it is five classes 5. Now the same thing with reset 34 also. Okay, they follow the similar structure. Okay, you have to modify this 512 to 5 whatever. This is the modification that you have to do. You're modifying the classifier head. Okay, so now let's do

**[19:34]** that. Now these are uh the things that we already know. So you got this the in features which is there. now is the in features of the six which is uh if you remember let me print it and show it to you again. Okay, the in features is this six in

**[20:01]** features. This is you are getting and then the out feature is number of classes. Correct? Now I have fixed the number of classes to five. Therefore you're modifying this to 4096 + 5. Okay. and then you so that means that every other layer now will have the pre-trained weights of VGG only this layer is modified with a random

**[20:31]** matrix. Okay. Now there are cases in which they actually keep the whole feature extractor as it is. They will freeze saying that the there is no need to train this. So they freeze the gradients. So and then only train this classifier. There are a couple of uh methods in which they do that. Okay, that is fine. So as of now what has happened is only in this this part now you have uh a matrix now which is 4096 + 5 which is randomly initialized not

**[21:02]** having any pre-trained values. Okay, that's what you have did model VG.classifier of six model VG classifier six you just modified it n.linear linear in features which is the in features of the six to number of classes. Okay. Then you ported the device and then keep it in the training mode. You're passing this dummy data where there are four examples with

**[21:30]** each example having a size 3 224 224. Okay. Now then what should be the output when you pass it for all these four? For each of them you should get a five length vector. Okay. That's what you should get. If I run that, you can see here the output dimension is 4 + 5. Okay. Now this is how you modify the classification head. See and again now one I'm repeating this again and again.

**[21:58]** The input is still 224 + 224 + 3. If you want to modify the input, let's say that you want to do it on MNEST which is 28 + 28. Now you start and recomputee the values here. Okay. Everything has to be recomputed so that the the input here modifies. Okay. Now this is for VG19. The similar thing is for the ResNet 18 and ResNet 34. You load the models. Okay. And you define the in features. So

**[22:26]** ResNet 18 fully connected in features that you change to number of classes put it onto device. The same thing you do you pass it you get the value. This is how you take the classification head which is there and just modify the number of classes only that specific uh layer is uh having the uh the randomly initialized weight. Everything else will have the the pre-trained model. Okay,

**[22:54]** this is how we look into modifying of classification head and then you can use it for any other training that you have. Okay. Now then another important thing that are used that are CNN's are used is for feature extraction. Okay. Or embedding extraction is what it is called. The one of the famous examples that we can think of is actually captioning. Okay. So you have an image you have to caption it. So what do you

**[23:21]** do? You pass on the image and extract the beforeh classification head. Now whatever is the output you get it and then take that and then pass it through a sequence to sequence model. Now because of which you can obtain a uh what do you call caption okay this is one of the uh interesting ideas that are used. So now let's see how do you do it. So now what do you need here is you have to go till the what do you call all the

**[23:53]** features all the features and then you don't need to go through the the classifier okay you don't need to go through the classifier head going looking at here now you have to pass your image till here till this average pooling that is there and then don't go into this classifier okay that means that I just need till here all the weights that are there till here after this I don't need this okay that's what you will do the same thing

**[24:21]** is for resonant based models okay that is what we are trying to do here okay now first you get the weights similarly to the previous idea I'm creating a new class BG embedding where in which I get the base models I get all these resonant weights now features base model dot features now this will give me the whole feature extractor with all those layers and then I get the average pooling.

**[24:48]** Okay. And whatever you get from the average pooling you flatten it up you get an embedding. Okay. Now that will be of size 25088. Correct? Right. That will be the input for this round. Okay. Now you extract this embedding. Correct. Now you pass two examples with 3 224 224. So you'll get two two things with 25088.

**[25:18]** Okay. Now you get the features, you get the average pooling, you get your input, pass it through the feature extractor, pass it through the average pooling layer, and then flatten it up and then return this. You're not performing the classification. You're just extracting the embedding. The same thing is for ResNet also. Now depending on whether it is ResNet 18 or Resnet 34, you load the model. You only have uh con 1. There's the first one.

**[25:46]** Layer 1, layer 2, layer three, layer four and then the average. So let's print the model and see that 18 model. If I print it, you see this first con one you want it. And then you want the layer 1, layer two, layer three,

**[26:14]** layer four. You don't want you even you want the average cooling. You don't want the FC the fully control layer so that you can extract the embedding. Okay, that is what you're doing. First you get the con first one BN Reu max pooling. This is your first operation. So just till here. Okay. Then you get this. This is layer 1, layer 2, layer three, layer four. Okay. See here all of them are

**[26:44]** categorized as four layers. Okay. All the networks layer 1, layer 2 and then pass it through the average pooling. So and then FC is not there. Similarly, you pass it, you get 512. So exactly similar. Okay, this is how you get the embeddings. Okay, so now uh please remember now all these are like Lego blocks. Okay, now

**[27:12]** you just need to find the proper size. If you match the proper size, now once you have matched the proper size, now you can actually work through them comfortably. Okay. Now you just pass a dummy input and see whether the transformations, the matrix multiplications that are there are actually happening. Once it is done, it's perfect. We can go through it. Okay. Now this is where we rest our case on CNN's. I hope you enjoyed the idea of

**[27:41]** CNN's where CNN's can be used as a classifier. CNN's can be used for the idea of transfer learning when in which you take the weights of the previously trained models and then uh you go ahead with that and CNN's uh which can be used as the feature extractors. Okay. So in the next tutorials we will be looking at RNN's LSTMs and GRUs. Okay. So we will be understanding them and implementing

**[28:09]** them. So in addition to that we'll be coming up with the deep iron and deep LSTM and then uh deep gru models. So preferably we'll be looking at them in the view of classification. And similarly now we'll be looking at uh the sequence to sequence in a very generic structure which can be incorporated into any of uh these sequence models. So with this we conclude uh uh the CNN

**[28:37]** tutorials. I hope uh you liked it and then you understood CNN. So meet you in the next tutorials with RNN. Thank you. Bye-bye.
