# Transcript — Tutorial 16 : Implementation of VQ-VAE

> **Source:** https://www.youtube.com/watch?v=NZQzEYuok_c  
> **Channel:** NPTEL - Indian Institute of Science, Bengaluru  
> **Duration:** ~58 min  
> **Note:** Auto-captions cleaned lightly. Minor ASR errors possible.

---

**[00:04]** Hello guys, uh welcome to this tutorials in which we will be continuing our discussion from VAES. So we'll be moving towards another latent variable models which uh is known as vector quantized ves. See uh earlier in the previous tutorial you might have implemented ves and beta ves. So where in which you change the value of beta and go ahead but you are still

**[00:32]** assuming that your latent variable is from a continuous distribution. Okay. So for simplicity we have uh taking into it into some uh scaled and shifted version of Gaussian. But here what is that we are trying to do is we are assuming that the latent distribution is a discrete one and then you'll be learning those discrete latent vectors

**[01:02]** and then going ahead with with the task. So without any further delay so let's uh go ahead and uh understand this particular uh uh VQVE. Okay. So first uh uh all these necessary imports. So I'm assuming that uh there is no point in uh reiterating what are these and uh uh what are these libraries

**[01:30]** doing. We have been using them. We will be using them uh even in the next section of our next coming sections of our tutorials. So device and random seed. So why do we need it? So we are going ahead with emnest uh data. So uh here we are going ahead and uh uh normalizing the values of the pixel between 0 and 1. Okay. So we are keeping uh as it is. So then we are creating the

**[01:59]** uh data loaders and then we are visualizing some of the examples. Fine. So standard uh initial thing. So now let's try to understand the hyperparameters of VQVE. The first thing is each latent vector have this many dimension which is embedding dimension which is 64. Uh so now that means that so we are

**[02:28]** uh dealing with The first thing that we have in hand is this uh embendic dimension which is 64 and then the number of embeddings which is 128.

**[02:57]** Now what does that mean? That means that the learnable dictionary that we have is of learnable dictionary will be of size you have 128 and each of them is of size 64. Okay. So you have 128 uh embeddings and each

**[03:30]** embedding is of size 64. Okay fine. So now the dictionary the learned dictionary that we will have will have 28 of these vectors each of them is with 64 dimension. So it will be 64 cross 128. Okay, this will be the learnable latent dictionary. Fine. So now

**[04:00]** and then we have this beta. So we'll see what is this coefficient of commitment loss. So this is another hyperparameter. And then we have the learning rate. So and then the number of epochs that we'll be going ahead is 20. Okay. Now this is for the demonstration purpose. We are going ahead with small number of epochs. But if you want to train it in proper way so going ahead with uh just 20 number of epoch may not be sufficient for good generation. Okay. So now let's look at the encoder. Now the encoder is we are using standard

**[04:31]** convolutional networks. So the input that you are giving is uh 12828. So which is an eminest image. So you'll be converting it into 324144 because of the convolution and then we are using relu as the activation. then 34 to 64. So this is just standard convolutions. I'm just uh leaving all of them. So the output is what is important for us. The output of the encoder is B. So this is embedding dimension which

**[05:00]** we have fixed to 64 77. Okay. So if I have uh an encoder, if this is an x uh that is given, this is the encoder, the output of this is 64 cross 7 + 7 block. Okay. Now how do we interpret it? So we have to interpret

**[05:29]** it. We will be permuting it. So actually what does this mean is I have uh 49 uh latent vectors I have 49 latent vectors each of the dimension 64 this is what we should be uh interpreting this as okay fine so now let's proceed

**[05:57]** so now we can think of it as whenever you have Each image we can represent it as Z E of given X belongs to R 64 + 7 + 7. Okay. Each vector will be encoded. So the each vector so you have 49 vectors. Each of them will be encoded and looking at the the code book that we have. So we'll be looking at uh this

**[06:25]** code book and then encoding each one of them. Okay fine. So now then what? See the code book that we have here it contains 128 of embedding vectors. Okay. And each of size 64 as we have discussed. Now what does the normal VA does? So the the normal VA

**[06:54]** we have X that is transformed to Z. Okay as you there is a sampling reparameterization happens agree to it. So you get Xhat. This is how a normal V works. Okay. So now here what is happening is the decoder that we will be having. So let me remove this. Okay. So the decoder that uh

**[07:25]** now we will be obtaining now this will be taking these latents that we have in hand. So which is from the finite collection that we have. So now let's see how does this happen with a simple example. Now let's assume that so my latent code

**[07:50]** book which is there which we call it as E has four embedding vectors E1, E2, E3 and E4. Okay. So E1 is vector which is having 1 one. E2 is another vector which is minus1 1. E3 is another vector which is min -1 -1 E4 is another vector 1 and minus1. Now

**[08:21]** these are the embedding vectors that are there and we pass an X through the encoder and then the output that is Z E of X. Now let's say that it is 0.8 8 0.7 this is the output of the encoder now what is that we want we need to find the nearest vector which

**[08:49]** is there now how do I find I need such k now which is arg min of j of z e of x minus ej in the second norm is what I'm taking Okay. In the second norm. So fine. So that will be E1. So the choose the chosen vector will be E1. And that is what we are calling it as a

**[09:22]** quantized vector which will be passing it through the decoder. Okay. So now you have X we pass it through the encoder we get Z of X and then we get the the nearest of the values. Okay. So nearest from where? nearest from the the code book that we have and that is what

**[09:53]** we are representing it as ZQ of X that you pass through the decoder we get this Xhat and this process know which is there now this is the encoder process and this is the the decoder process And then this is the quantization process.

**[10:30]** Okay. Now this is vector quantization. Fine. So now what should be the loss? Now we have to to an extent understand the loss function. know which is there fine. Yeah. Before that let's uh so encoder is done. So let's go ahead. Now let's check the encoder uh input output. Now you

**[10:58]** given uh image which is of size 128 28. The output will be 6477. So which is uh just the same thing that uh we have looked at. Okay. Now comes the the vector quantization. So now let's try to understand what is happening in this vector quantization now which is uh this quantization process. So how is this happening? We just saw a very simple example. Now let's see how is that being implemented.

**[11:25]** Fine. So now here you know we have number of embeddings. See number of embeddings we have fixed to be 128 and embedding dimension that we have fixed it to be 64. And these are learnable parameters. Okay. So that means that there has to be some kind of uh uh back propagation that has to happen through this code book. Now because of which we get the optimal uh values for these uh vectors. Okay fine.

**[11:51]** So now the code book that is there is k cross d. So you have uh 128 of them. So and each of them is of the dimension d. Okay which is 64. Fine. So now we just initialize them with uh the uniform waiting. It's perfectly fine. So now what is the input for this? See the input for the code book. So if I have this uh VQ

**[12:20]** the input for this is B will be there batch size this is 64 77 7. Okay. Now this will be the input. Now we have to convert it into B hwdy. Now because if you remember I just told you that you have 49 vectors each of them is of size 64. 64 is the dimension. So we permute it and then know we get B

**[12:51]** will be there. So so each image will have 7 + 7. we will have 49 vectors of size 64. Okay fine. So now that is what uh we need to look at it. So you just permute them so to get it into the shape. Okay. Now then you need to know how many vectors are

**[13:19]** there like that. Okay. So now here you can see that this is for each image. So let's say that I have eight images. So for each image I have 49 vectors. So that means that if I have eight images so it'll be 8 into 7 into 7. So those many number of vectors will be there. So therefore you can multiply them. So this is B into 7 into 7. This is the number that we have and each of

**[13:48]** these is of size 64. Now these many vectors are there. Okay. So now for each of them you have to go ahead and find which is the nearest vector which is there in the code book and then proceed. Okay. So that is why you flatten it. So you just did the same thing. Now compute the distance. So now you want to compute this distance. So this Z now which is there and E which is the encoder uh embedding which is so this is the

**[14:17]** embedding that we have and Z is the output of the uh encoder that we have. So you just compute the distance. So it's just the square of them. You know how to expand it and uh take it. So this will be the distance. Okay. So it will have three terms. So this will be the distance. And again the distance shape now will be the same size. This size. Okay. So now what is that? We want we want the

**[14:45]** closest. Okay. So we want the arg of this. So those that get selected is what is we are calling it as encoding indices. Okay. So it's like let's go back to the example. So now in this let's say that I have four uh embedding uh vectors in my code book. Let's say that I need to

**[15:15]** find eight embeddings. Okay. I have eight embeddings uh that I got from the encoder and each of them is of size two. Okay. So now for each of these eight that is there I have to be finding now which is the nearest ones. Okay. So now if I do it now what will happen? I get the distances which is there. So I find the

**[15:50]** distances for each of the eight vectors that is there. Now what is the distance between all the four of the encoding vectors? It will be there. This will be the distance. Okay. So now you have uh so now if this is there you have eight rows and then you will have four columns. Okay. So like this. So you will have So now

**[16:16]** what is that we are supposed to do? You take the minimum of this. Okay. So that will be the for the first one. You take the minimum of the next one. That will be for the second uh vector. Now so on and so forth you get the minimum values minimum indices. Okay. These are are the arg min. We are taking the arg min. So you get the indices of this fine. So that is what is called as the

**[16:47]** encoding indices. You got the minimum of them. So now and then what whatever the embeddings you indices you have you fetch the embeddings itself that is what we call it as ZQ and these are the embeddings from the code book which is having the nearest distance from the uh encoding vectors that we got from the encoder. Okay. So now that is of the

**[17:15]** shape B into H into W into D and then you restore the shape. Now so that it becomes B H WD. Now this is the output. So now what will be the output of it? Now finally you permute all of them you get B comma 7A 64 and this is what we call it as ZQ and this which is there is what is called as

**[17:45]** ZE. This is the these are the embeddings that we got from the encoder and post quantization this is what we got by looking at the embeddings which are nearer in the code book. Okay fine. So now now let's look at the losses. Now how do

**[18:14]** you go ahead and take see this these things needs to be learned. Now therefore we need to have some kind of loss. Now let's look at the loss there will be three components of losses that are involved. Now let's look at them each of them together. The whole loss that we have the VQB loss that is there if I call it as L now which is minus log of P of X given

**[18:46]** ZQ of X. This is one the first term. And then we have the stop gradient. So these are the three terms that we

**[19:20]** will have in the loss if you use uh the proper uh uh functions. So this will be turning out to be a the reconstruction loss which we have been seeing in VE. This has a name. This is known as the code book loss.

**[19:46]** And this the third term which is what is known as the commitment loss. Now why there are three terms. Okay. So so now let's try to understand them. Now here now when I mean by e this is uh from the

**[20:12]** code book. These are the embedding from the code book. Either you can say it as E or you can say it as ZQ of X perfectly fine. These are the ones that are near to the what we got from the encoder. And then what is this SG? This is what is known as the the stop gradient. Now why is the stop gradient used? So we will look at it in a moment. Okay. Now these are the three terms that are

**[20:41]** involved. See term one is something that we all know which is the reconstruction term. Okay. So that is needs to be there. Uh so when you go ahead and write this will be x - xhat this will it will boils down to become this. Okay. So now let's look at the the remaining two terms which is the codebook loss and then the commitment loss. Let's take one at a time. Let's

**[21:08]** take the code book loss and try to understand what is this loss which is there. Now from the earlier example that we have the for a given image so the encoder gave us let's say 0.8 0.7 and then the nearest codebook vector that we have E1 which is 1 and 1. Therefore, ZQ of X

**[21:40]** will be 1 and 1. Okay. So, now the question that we are trying to ask in the codebook loss is how should E1 now which is there? Now this is uh E1 right now when we in this current example this E will be E1. Now how should this E1 should be as close as the representation of the encoder that we have got correct?

**[22:11]** Now how do you ensure that if it has to be as this E1 has to be as close as Z E of X. Now what you will do? Now you will just write Z minus Z. This is what you will write. Correct? So now minimize this. these two vectors will become as close as possible. See now the next question is this Z. Now how did you get this Z? Now you got this Z from the encoder. So it will be attached

**[22:41]** to the parameters of the encoder. Okay. So now what you have to do? You should detach it. Okay. That detachment is what is known as the stop gradient. You have to detach it. So you have to freeze the Z. Therefore we call it as in this term. Okay, this is the codebook loss which makes total sense to us. We do not want to update the encoder parameters. Now therefore the second

**[23:10]** term will have a stop gradient for ZE. Okay. So now let's go to the third term which is the commitment loss. So what is this commitment law saying? So you have sum beta into z of x minus stop gradient for e and since the these are vectors you are

**[23:42]** dealing with the ucidian distance. Okay. So now the question is why do I need it? See if you are making your uh code book vector as close as to the outcome of the encoder. Now why do you need it? So now whenever we are going ahead see this is an iterative training. Now what will be happening is now as the parameters of as the vectors of the code book is changing now so are the uh parameters of the

**[24:14]** uh encoder. Now during training the parameters of the encoder changes the all these vectors also will change. So now what is the intuitive idea is see once you reach so once you go ahead and choose a region for this code book so we don't want the outcome of the encoder which is there now to be uh wavering too much. Okay. So now we want some kind of a regularization on the region know which

**[24:42]** is there. Okay. So that is what we are trying to do do once we choose uh a codebook region. So we don't want don't wander arbitrarily far away from the codebook vector.

**[25:22]** Now therefore see we don't want the outcome of the encoder to wave away. So therefore it is somewhere in the sense you are seeing to that you are taking care of the parameters of the encoder. So you are not saying that the modification to codebook uh parameters should happen. So therefore you have a stop gradient for this E. Okay. And you want them to be regularized. Now how strongly you want

**[25:50]** to go ahead and restrict it. Now therefore you have a beta here which is nothing but the commitment loss that we have. Okay. Now that is the reason why we have these three terms in the loss. Okay. So now let's look at them in the implementation. So the VQ loss VQ loss has two things. One is the codebook loss. The other is the commitment loss. Okay. And I just want to bring it to your notice that uh see E and Z Q both of them are same. So

**[26:24]** we can uh just replace them. See the code book loss is stop gradient of Z minus ZQ which is nothing but E. So this is the codebook loss. So Z now which is detached. And what is the commitment loss? ZE minus so stop gradient for ZQ. ZQ is nothing but E. Okay, this is the commitment loss that you have and this both of them put together is what is known as the vector

**[26:52]** quantization loss, VQ loss. So, and then the next question is okay now this quantization which is there how do you do it? No, you do it using argument. See, argument which is there is not differentiable. Now, what is that I'm trying to say is this. Sometimes we are if we write it in equations it is much

**[27:20]** more com comfortable. See what is quantization. The quantization that we have is K is argument of J Z minus EJ. Okay. So now here this ZQ is equal to EK. Correct. So the operation argument

**[27:51]** which is there is discrete. Okay. Now therefore now there will not be any smooth transition from one index to another. Now since this is the case how do you handle it? So we handle it through what is known as the straight through estimator STE. Okay. Now during the approximation of backward pass what do we do is the partial gradient with respect to z for zq we will be considering them as

**[28:22]** the identity. Now pretend this the whole quantization operation which is there so which is there as an identity function for the purpose of sending the gradients the reconstruction gradients back to the encoder. Now how do we do it? So ZQ is equal to Z + Now what is the stop gradient of EK minus Z correct? So now this will be ZQ will be EK. So now when you take the

**[28:57]** the gradient so now this term the gradient when you take the gradient with respect to z this term since you don't have a gradient this will be zero and then the partial derivative this will be one. Now therefore, now what will happen is the reconstruction Okay. And this is what is known as the

**[29:33]** straight through estimator. See the whole idea is now you are just considering that this gradient which is there is identity. Okay. Fine. See that is what is known as the straight through estimator. So uh zqst is z plus zq minus the z permuted plus d attach. So this is nothing but your uh implementing uh this

**[30:03]** particular equation which is there. So and then and convert this finally. Okay, this ZT which is there, this also needs to be converted, converted and then return. So what are you returning? You are returning ZQST, VQ loss, encoding indices, code book and commitment loss. Okay, fine. So now maybe we should uh keep track of

**[30:32]** things what is returning what. So otherwise uh so it it will be we will uh lose it. So now x is the input for the encoder. The output that we are getting is 64 + 7 + 7 and then we have this uh vq for which I'm passing the 6477. This

**[31:01]** output will be the straight through estimate of the ZQ. So we are getting this uh VQ loss and then we are getting the the encoding indices. So we are getting the the code book loss. We are getting the commitment loss.

**[31:30]** Okay. And then we have the decoder. What is the input for the decoder? The Z Q that we have. And the decoder will give you Xhat which is a value between 0 and 1. Okay. Fine. So now decoder it takes uh the input which

**[32:01]** is D77. So that is 6477 you use uh transposed convolutions. So and then you up it up. So you get 324 14 and then 128 28 and finally you apply NN dot sigmoid and the output you get the decoder. Okay. So now the VQA in full so where in which it has decoder encoder and then the vector quantization

**[32:31]** all of them put together. So the first is the encoder so which is from X to Z. The second one is Z to ZQ now which is the vector quantizer. So which needs number of embeddings embedding dimensions and beta. and then the decoder which is taking the embedding dimension. So then how will be the forward? So first you get X and pass it

**[33:01]** through the encoder you get Z then you quantize it you pass it through the VQ so and then get all these things and then how do you uh perform the decode? So you pass the ZQ get the the reconstruction which is Xhat. So now now these are all pieces all these three put together is what is known as the VQVE

**[33:29]** class for which the input is X what is the output that we have now we have uh reconstruction which is Xhat we have uh the VQ loss we have the encoding uh indices. So now we have uh the code book loss

**[34:03]** we have the commitment loss. So just for these are for bookkeeping. So we have this ZE and then we have the ZQ. We have ZE and then we have ZQ. This is what you get as the this this VQVE in itself will have the encoder

**[34:31]** vector quantization and then the decoder. Now these things will be present inside the VQVE. Okay. So now let's train it. Now, now all these things we have got now we have uh created the VQA model. Fine. So, and then uh the the loss the VQE loss the first is the the first term which is the

**[34:59]** reconstruction loss that is the MSSE between the reconstruction and the original that we have. Okay, that is the reconstruction loss. And then we already have got VQ loss which is the sum of the codebook loss and then the commitment loss that we have that is the total loss. Now we can uh uh return them. And then the optimizer I'm using the atom with the learning rate 10 to the^ of minus 3. So just one training step. Now here I get these images port them onto

**[35:27]** the device clear off the gradient. See there is only one optimizer involved. So you perform the forward pass. So you get reconstructions, VQ loss, embedding indices, code books, commitment loss, ZQ and Z. Okay. So now this is the total loss that you have. So then once you get the loss now you loss dot backward you perform the back propagation update the steps and then the remaining things are for

**[35:55]** bookkeeping. Okay. So this is what we have now. Then uh so that is for dummy one. So now we train for one epoch. So total loss, total reconstruction, total codebook, total commitment. So these are all for bookkeeping. So you iterate through the loader, you get the input, clear off the gradients, pass the images through the model which is VQVE which is internally has encoder, vector quantizer uh and then the decoder you get the output and

**[36:24]** then compute the loss, perform the back propagation, update the weights and then store all these things. So total loss, reconstruction loss. So then we can plot them and see. Okay, this is these are for the bookkeeping cases. Okay, see the major see this is the training process will be standard but understanding the loss function and how is what we intend for is being translated using the loss function is

**[36:52]** where our main emphasis should be. Fine. So now then the evaluation now how do you evaluate? you just pass the uh loader and then see now what what are the losses. Now here what do I mean by evaluation is just to see what is I'll pass the test loader I'll see what is the losses that I'm getting that's what the evaluation means here okay now train the VQVE now we had a function to do it for one epoch you do it for uh the 20

**[37:23]** epochs that we have just repeat the same process call this train one epoch 20 times so and then you go ahead and evaluate using the test uh loader Okay. So, and then for the bookkeeping you keep all of them. Test loss, train reconstruction loss, train total loss, codebook loss, commitment loss, all of them you keep in track. Okay. This is the full training process. I have already uh ran this. Now, because it takes some time to run these kinds of

**[37:52]** things and then you plot these losses, train loss, test loss. So, the total loss, you just plot them. This is how it is. Now, this is the training loss and this is the test loss which is there over the epoch. and then individual components like reconstruction code book commitment. So you just plot all of them. Now we can see that now they are uh all of them are converging nicely. Okay. So now let's see the the main purpose now which is

**[38:21]** reconstruction and then finally we will extend it how to sample from this. Now we haven't looked at the sampling. Okay we are still worried about the reconstruction in this case. First I get the images then I pass it through the model. I get these reconstructions. I'm plotting I'm uh porting it onto the CPU and then I'm plotting both the original and then the reconstruction. And this is how the original and the first the top row is the original the bottom row is the

**[38:52]** the reconstructed images that are there. Okay. So now let's see what are these uh uh latent codes are discrete latent codes are. So I take a batch I'll take one image. Okay. So and then I'll pass it for that I get these 49 different uh vectors that are there. Each of them is of size 64. I'm plotting I'm printing them. Now these are the indices that

**[39:21]** were selected 49 of them. Okay. So you can see 1 2 3 4 5 6 7. So 7 + 7. So this is what now we have got for the digit seven. Okay. So these are the indices that got selected. So these are the what do you call uh uh these corresponding to this E will be chosen. Okay. So now uh the same thing we have just visualized just to see is there any uh some kind of

**[39:49]** representation or some kind of heat map that can be generated. But no uh we don't have so then this is the the learned uh uh code book which is there you can see it this is one of the one of the vector okay so which is 128 + 64 okay you have 64 uh uh you have 128 vector each of them is of size 64 so one of the

**[40:16]** codebook vector we have just uh uh represented it here. So now sometimes we want to know these are some of the additional things I'm just showing it because these kinds of experiments should trigger couple of questions. So now how is the codebook usage? See what do I mean by that is how many times the vector one uh I have 128 vectors no how many of how many of times vector one is used? How many of time vector 2 is used? Is there some kind of uh uh what do you

**[40:45]** call clustering that is happening? Now these are the important this since these are so the finite number of vectors that we have in hand we can perform analysis like this okay see some of them being now measure the codebook usage so here what is that I'm trying to do is I'll go through the test loader and then obtain all these uh encoding indices that are there I go on appending them so these are all the indices that are there for uh all and then I get the unique indices

**[41:13]** so then now I know how many uh so each of the image I know what is the indices vector that is there encoding indices and then I know all the indices so I can say that now how many times the code book one uh the codebook vector one is used how many times the two is used how many times three is used so that I can uh easily identify okay so now and then if I ask the question is there any vector which is

**[41:42]** not used now apparently no all the 128 vectors are being used. So now then if I plot a histogram of how many times what is used. Okay. So I just put them into bins and then if I see this is what we get. Now we can see it here that now this is the index one index two so on and so forth. Now I have totally uh these many uh times they are used. So

**[42:10]** you can see it here that uh the distribution of the choice of indices is uh uneven. So so we normally think we get some kind of a uniform distribution over here but no that is not the case. Okay. Now we are getting some uh random usage of vectors. So there is no specific structure in this particular case. So sometimes so in some of the images we might get some kind of a structure. Okay. Now then we can always

**[42:39]** uh uh get another uh idea which is known as the perplexity. See the idea of perplexity is uh it it measures so uh now how useful now each of the vectors are. So now how do we do it is so the code book

**[43:12]** code book perplexity now let's say that we have PK now which is we have the code vector K is used with some probability. Okay. Now we can have H which is minus this is nothing but entropy. Now summing over all the vectors that are there PK.

**[43:42]** So log PK you get this H. So what is perplexity? It is E to the power of H. Okay. So now if things are uniformly used so then we get if uh you get some kind of uniform distribution if PK is 1 / K then e to the power of H the mag the value that you will get is K okay so now this is in case of some kind of uniform

**[44:10]** distribution that is there but we don't have that so we calculate the entropy and then you take the uh e to the^ of this entropy which is there. Okay. So you know you get it the codebook perplexity that we have got is 444.51. The maximum one that you can get is K which is how many uh number of embeddings are there but we are getting it as 44.51 51 which is indicating that not all the uh uh

**[44:39]** vectors are having some kind of uniform distribution. Now there is some kind of gradation which is there fine. So there is just a measure for it. So now now directly uh decode directly from the code book. Now this is a very interesting experiment now which is I get this Z from the encoder. Now you pass this Z. Now you get from the quantizer you'll get these things. Okay. Now get these embeddings now which

**[45:10]** is the quantized vector that are there. Now for each image now it'll be 49 of them. Each of them will be of size 64. Now convert them. So 7 764. This is the quantized vector that we have. So you permute them and then you decode it. Okay. This is the uh standard procedure that we'll use. Okay, which is there directly from the codebook indices. Now can we get it?

**[45:39]** Yes, it's a straightforward same old method. Yes, as it turns out we can. This is the original image and decoded from the codebook indices. They look similar. Now, now what will happen if I directly pass the encoder output? Okay. So, rather than quantization, what will happen if I directly pass the encoder output which is ZE? Rather than passing ZQ, what will happen if I directly pass ZE?

**[46:06]** Yes. So, we got if I get the model, I get ZE and ZQ. So I'm just choosing one of the features. Now how it will look. Now this is the the encoder which will be there quantized feature. This is the from if I'm just passing Z itself. Now this is what will happen if I get ZQ. Okay. So now then we just for

**[46:34]** completeness I'm just checking the gradient flow. So we can just uh know it will happen. So, so like this. So, we have uh we can do lot of experiments. I have done couple of them and the code book uh and this is available in the description. So, one interesting experiment is this. So, directly passing the ZE and passing the ZQ. So now what will

**[47:03]** happen? So if you do it that is this okay you directly pass the Z or you directly pass the ZQ. See what we can see immediately from here is there is no much of a difference. Now why there is no much of a difference now because of the second term and the third term in the loss we have seen to it that so this particular difference is minimized. So we have explicitly seen to it. Now another question can easily come up

**[47:31]** know what what if I just generate some random indices and pass. Now that is what I'm trying to do in this case similar sort of the code. So I just pass the uh random indices generated randomly. This is what you get. Okay. See just because something is working that doesn't mean that you can have some kind of randomized methods it will work. No. Okay. This has a structure to it. Now all of those indices in one group it makes it will start making sense. But if

**[47:59]** you sample them in random and do it see you get gibberish. Okay. Okay. So now all these things are fine. See now the next question that and the very important question that arises is now how do I sample new uh images from this? See this discrete code book which is there the earlier in our continuous latent space

**[48:27]** you can sample any number of example from Z and then you can pass it through the Z decoder and then get the new examples. But here what do we do is we pass the images and then we collect all the uh vector uh Z that is there and then what do we do? sample we we come up with some GMM we fit a GMM to it so that we can sample from this distribution of Z and then those sampled one we will compare that with the codebook we vectorize it

**[48:56]** and then we pass it through the decoder that is what we will do for that we'll be using this GMM so Gaussian mixtures so you need standard scalar because you need to be normalizing before passing it through the Gaussian mixture models so then mattplot lib and other things so first we put it in the evaluation mode. Now why? Because we don't want any gradient computations in this case. So all latent. So you get the train loader and then you pass it through. Now what

**[49:26]** is the this is the ZQ shape. Okay. For each of the image you get this ZQ. So you flatten it out. What do you mean by that? See you have 64. You have 49 vectors each of size 64. So you just uh this is for one image. So that means that one image which is there is a vector of length 3136. So like that for each for batch size of

**[49:54]** image I have. So this is for one image. I'll just uh keep it. I go on appending it for each of them. So and then I keep track of the labels also. Why do I keep track of the labels? So hold on to that question. So I'll show an experiment where which it makes sense. So you have all the latent. So let's say that you have some 60,000 training images. So you have all the latents now. Now uh yes now this is the latent size. So you have 60,000. Now each of them is having 3136

**[50:22]** length vector. Now you fit a uh GMM to it. So first you this will all all be in tensor. So you convert them into numpy. Okay. So and then we standardize them. So subtract the mean and divide by the standard deviation of it. So that is what this standard scalar uh is doing. Okay. So now then we fit a GMM. Okay. So now how many components you take in

**[50:51]** GMM? So that is again another hyperparameter. I am currently taking uh 20 uh components. Okay. So now I fit a GMM. So each uh so your uh you have 20 component uh GMM. So the latent dimension is 3136. So now uh so we are assuming that the uh coariance matrix that we are considering

**[51:20]** we are considering a diagonal matrix. So and then the maximum iteration that we are going ahead with uh is 200 uh it's 200 and then if the log likelihood is very small now that means that it is less than 10 the^ of minus 3. So at that time you can stop so like this. So then we are fitting a GMM. So it ran for I said it to run for 200. It ran for only 110. So the GMM is ready. Now it has converged. So now let's look at the

**[51:49]** uh weight of each of these component. Okay. So it converged is true. So the number of iterations that we have is 110. So GMM weight shape is 20. And then GMM means. So that means that each component is having a mean of uh length 3136. And then since I'm assuming the coarience matrix which is there is diagonal. So the coariance shape will also be 20 cross 3136. Fine. So then I'm

**[52:19]** just uh printing the mixing coefficient. So that means that the component 0 is having the mixing coefficient of 0181 so on and so forth. If you sum up all of them you should get one. These are the uh mixing coefficients. So, so recall your GM discussion that we did earlier and this is how the mixing coefficients are of all the 20 components that we have considered.

**[52:48]** So again here also it is clear that the components the mixing coefficients are not in some uniform way. Fine. So now um we have stored the labels no so we know for which uh 3136 length vector that we have the ZQ that we have now what is the label associated with that. So now what we can do is now we can see that

**[53:17]** now how are the uh the distributions of uh the components for each class of the image. Okay. So now we can do that. So fine. So let let's do that. So for uh digit distribution within each component. So that means that if I take the the first uh uh what do you call GMM component now how many of uh digit one 0 1 2 will come so on and so forth now we

**[53:50]** can obtain that using this code okay see these are all standard uh looping so I'm not going into the details of the code so you can see that here GMM component zero is having uh the number of samples I'm having is 1085 and this is the for digit 0. This is one digit one. So so on and so forth. So it is predominantly component 0 is considering digit two and digit three and digit five. Now so on

**[54:20]** and so forth we can uh look at for the remaining GMM components also. So so let's not uh this is another important uh uh distinction that we can make. Fine. See now comes the important part. So where which we want to sample from the uh uh this GMM. Okay. Sort of we can easily sample. So I want 25 samples here. So I sampled 25 of them. So now

**[54:52]** then what? Now for those 25 of them just one of them I have shown what are the components that are used. Now all these are standardized. Now we just subtract at the mean and divide by the standard deviation. So you want to redo it. Now why? Because so these see what are these samples we are. So since we have f GMM and then the GMM has converged the samples of this is nothing but the uh ZQ. So we are sampling from the ZQ distribution. So now we have normalized

**[55:21]** it. So you you inverse that standardization process that we have done. So that now using these new samples you can pass it through the decoder and obtain new examples. Okay. So that's what you have got. So you have uh performed the inverse transformation and then uh this is the sample latent tensors that we have. So which is of this shape. So now then now this is of shape 3136. Now you have to convert it

**[55:48]** into 6477. No because your decoder input is 6477. That's what you are doing. So you are changing the view number of samples which is 25. So embedding dimension it will be 6477. So this will be the size. Then you put this through the decoder. Okay. So whatever uh the generated uh input that you have, the generated output that you have, you put them onto the CPU. So and then you display them.

**[56:17]** You display the images. This is how it looks like. Okay. So you can see a lot of uh mishaps. Okay. But you see this so these are from the samples obtained from GMM. Okay. So you are directly going ahead and passing those samples through the GMM. So whatever your samples that you

**[56:50]** got from the GMMS you're passing it through the decoder. Now another thing that we can do is we can always go ahead and find the nearest vector on the uh code book and then you can pass it. So that's what the next experiment that we did. So we just perform that thing. So display those things. This is what we get. See there is no much of a difference in the in the generation. Okay. So if you compare both of them.

**[57:18]** See here this is from the directly passing it through the whatever you have obtained from the GMM directly passing it through the decoder and this is vector quantizing it and then passing it through the decoder. Okay. So like this our aim of sampling from the uh ZQ distribution we uh solve it uh using the GMM. Okay. Now this is regarding the uh tutorials on VQVE.

**[57:50]** See the idea of VQE is very important. Now why? Because as we go along uh so we will be implementing VQA on the latent space of the diffusions. Okay. So the next tutorials will be on diffusions. So and then later we will combine both the uh diffusion models and VQAs. So so that we get better reconstructions. That is what the state-of-the-art uh models of generations they actually go ahead and

**[58:20]** do. So with this we conclude this tutorials on VQVE. The next tutorials will be on the diffusion models. So thank you all.
