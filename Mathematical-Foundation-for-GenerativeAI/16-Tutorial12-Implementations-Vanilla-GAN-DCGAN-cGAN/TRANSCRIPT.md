# Transcript — Tutorial 12 : Implementations of Vanilla GAN, DCGAN and Conditional GAN

> **Source:** https://www.youtube.com/watch?v=dBcURX7GrwE  
> **Channel:** NPTEL - Indian Institute of Science, Bengaluru  
> **Duration:** ~78 min  
> **Note:** Auto-captions cleaned lightly. Minor ASR errors possible.

---

**[00:03]** Hello guys, welcome to this tutorials in which our main aim is to understand the implementational details of generative adversarial networks. In the theory sessions, by now you would have been comfortable with the mathematical nuances that are there in terms of generative adversarial network, which is a special case of

**[00:33]** variation divergence minimization. So, we have chosen a specific a function and because of which now we get this particular uh uh thing which is generative adversarial networks, which is a saddle point problem. We'll be going ahead and looking at the implementational details of the generative adversarial network. So, without any delay, so let's

**[01:04]** jump into the discussion. Now, we all know that this is the generator network. So, wherein which you'll be sampling from a normal distribution and then you will have a neural network. So, and then you will pass this through the neural network and then you'll be obtaining the samples from this P theta distribution and then you have this discriminator which is a binary classifier. Now, wherein which

**[01:33]** you give this X or X hat. Now, it should be giving the the decision. Now, what is the decision? This is a probability that this is real given the given images from the real distribution. Okay, it is from PX. Okay, this is a This is what we look at look at it as an output. And then we want the system where in which this discriminator

**[02:01]** actually fails. So that means that post training, so the values that comes out of the discriminator either you give a real image or a generated image, so it should be giving 0.5 now which is an indication that we have obtained good samples where in which the discriminator is not able to discriminate between the real images and the generated images. Okay. So now we all looked at this is the loss

**[02:29]** function which is there and please see this there are two terms here. These are the two terms that are involved. So where in which the first term the expectation is over the the actual data. Now this is over the generated data. Now we can replace this X hat which is there using Z. Now when we Z is sampled from normal zero I that also is the way of

**[03:00]** looking at it. So now now we So there are two sorts of parameters now there is theta and there is this W. So now we will be maximizing with respect to this W. Okay. So we can see this both the terms now contains this W. This this this set of parameters W. So now we can approximate this expectation which is there using the sample averages

**[03:28]** and the batches. So this is the the this is the samples from the real data and this is has to be calculated over the samples from the generated data. Now how do you get this XJ hat? Now we get this XJ hat first we sample from Z and then we will pass pass through the generator and then we get this XJ hat. So, now that means that to go ahead and

**[03:57]** update the weights of the discriminator, now we need the sum of these two terms and then we need to maximize this. Okay. &gt;&gt; [snorts] &gt;&gt; Now, in contrary if you want to obtain the theta, which is the parameters of the generator, now we can see this. Now, this is the same two terms. We can see that the first term which is there is independent of theta, so therefore we don't need to consider this. So, we we only need to go ahead

**[04:26]** with the second term. Okay. So, and then we want to minimize this. Okay. Now, see here whenever I am saying that there is a generator, I'm not bringing in any kind of architectural predispositions. So, the way in which we implement into today's discussion is now first we'll be going ahead and using a very simple

**[04:54]** multi-layer perceptron as generator and discriminator and then we'll be implementing a the vanilla GAN and then using the same generator and then the discriminator setup wherein which we have used MLP as generators and discriminators, we'll be going ahead and implementing the conditional GAN. So, I'm assuming that all of you are pretty much aware of conditional GAN wherein which Y is added as an additional

**[05:22]** input. So, the label information is also provided. Okay. So, we will see the nuances as we go along and then rather than using multi-layer perceptron, we'll be going ahead and using a deep convolutional networks. So, we'll be implementing DCGAN, now which is just the architectural modification of vanilla GAN. Instead of using multi-layer perceptron, we'll be going ahead and using a deep convolutional network and then we'll be using transpose convolutions for uh

**[05:49]** uh the generator and then we'll be using the plain convolutions for discriminator. Okay. Now, we'll see the all those nuances you know as we go along. Okay. So, now uh the first thing so, we need to install a couple of packages. Now, why do we need to install is see once the images are generated, how do you know the quality? Is there any numerical metric that is there? Now, using which we can estimate the quality of the

**[06:18]** generated images. Now, as it turns out, we have what is known as uh FID scores. Okay. So, this FID score now, what we do is we use the Inception net. So, we would have look at the Inception net earlier in our PyTorch CNN discussions. So, we take an intermediate layer. Now, we see the idea is like this. You given image and then you get an embedding out from the middle of a CNN network. Okay. So, now those intermediate representations are used to

**[06:52]** uh compute this FID scores. Okay. Now, that is what this uh computation is all about. So, for that we need special packages. So, we need uh this torch metrics which are there. And uh and one more thing see uh GAN training is pretty time-consuming. So, therefore what I have done is I have already ran all this this thing. So, therefore we'll be looking at only the outcome of this. So, uh we will not be

**[07:19]** live coding. We'll be just looking at the output. Now, this is the the packages uh that are needed. Okay. We install this torch metric package which is there. So, and then uh the next set of uh libraries that are needed is we need this uh torch. Okay. So, torch and then uh optimizer and then data loader data sets transform. So, make grid is used to have a kind of a grid kind of a structure for plotting the images and then you need matplotlib and numpy. These are the

**[07:46]** standard libraries that we have been using overall so on and so forth. So then the next set of code now which is there now is to take care of the device configuration. So first we'll be looking whether the GPU is available or not. If the GPU is available we'll be using the GPU. We have the whole code is implemented with GPU enabled system. So we used the same Google Colab now which

**[08:14]** is there. So that's how we have used. So see running this without GPU is possible. You need to make the network much more shallower. Now but the problem is it will take too much of time. So please be aware of it. So and then we'll be using the MNIST data. Okay. So we'll be using the MNIST data that we have. So here we'll be preparing the MNIST data set. See the need for this specific preparation of MNIST data set is quite simple. See the generator outputs

**[08:43]** so normally we'll be having a tan h activation at the end of the generator output. See and the tan h value will be between minus one and one. But the original MNIST data which is there the issue there is it will be intensities between zero to 255. So then what has happened is we have normalized the data will be normalized therefore we'll be getting

**[09:10]** the values between zero and one. See the actual data which is present is values between zero and one but preferably for better implementational nuances we'll be using tan h at the end of the generator in each cases. So therefore there will be a mismatch in the range of values. Now therefore what do we do? We need to convert this zero one the range of the actual images which is between zero and one into minus one and

**[09:39]** one. That's what we are trying to do. We are trying to normalize this. See, the X which is there, so we'll be subtracting 0.5 and then we'll be dividing by 0.5. Now, therefore the the zero which is there will be converted to minus one and then one which is there will be converted to plus one. So, this is how we normalize. So, you can see it here. So, we have composed the transformation. We'll be first converting it to a tensor and then we'll be normalizing with mean zero and

**[10:07]** standard standard deviation 0.1. Since it is only one channel image, we are we are using MNIST which is only one channel image. Therefore, we have only provided one number here. So, if you are going ahead and implementing this for any other dataset which is having a three channel data, now it is good to use ImageNet statistics so that uh is publicly available. So, any 3D data in the sense any three channel data

**[10:34]** that you're going ahead and using, now it's it's good to go ahead and use the ImageNet statistics that are available. Okay, fine. So, and then you're downloading the datasets. So, the batch size that we are considering is 128. The standard ideas, okay? So, this we have seen again and again even during in our the PyTorch discussions that we have and I'm assuming that these are clear. So, and then what I'm trying to do is I have taken some examples from a batch. So, okay,

**[11:03]** each batch since my batch is 128, I'm getting one batch of things and then I'm only picking 32 of them. So, and then I'll be going ahead and plotting it just to see that now these are the real images that are there. See, real images in this these are the images that are provided to us. So, the actual images we have taken uh out of 128, we have taken the first 32 of them and then here we we

**[11:30]** So, this is just for the sake of reference, so that once we generate the image, it has to be close enough to the way in which it is looking. Okay. Good. So, now now then uh uh we need the hyper parameters. So, for vanilla GAN, the hyper parameters that we'll be using is the latent dimension that we'll be using is 100. So, that means that the Z which is there will be a vector in 100 dimension. It's a normal vector in 100 dimension. And then the image size that we are getting is 28

**[11:59]** cross 28. The learning rate that we'll be using with a 0. uh 0002. We'll be using the same learning rate. See, since uh we all know that uh it's we need to optimize two sets of parameters in GAN. And one set of parameter we need to maximize, and in the other set of parameters we need to minimize. Now, therefore, we can have multiple learning rates, which is perfectly fine. Even in implementation, we can go ahead and do it. But, what we'll be going ahead is we'll be using

**[12:27]** one set of uh one learning rate for both generator and then the discriminator. This is the choice that we are making. Okay. We are going ahead with number of epoches as 25. Okay. So, now let's uh look at the the generator architecture. So, we are using MLP here. So, now let's look at this uh architecture. Okay. So, now first uh nn. sequential, we are having 100 to So, latent dimension is 100. So, we have 100 to

**[12:55]** 256. Okay. And then we are using leaky ReLU as the &gt;&gt; [snorts] &gt;&gt; activation function now which is there. So, which we have used earlier also during our discussions. So, we have uh using leaky ReLU here. So, and then uh then we'll be having 256 to 512. Again, leaky ReLU. And then 512 to 1024. And then 1024 to 28 cross 28, which is 784. Now, this is your generator. And then at the output we'll be having the tan h. So, just to uh

**[13:25]** reiterate, so the the network structure, so it will be Uh the first is 100. So, from 100, we'll be going into 256, and 256 to 512, and 512 to 1024, and then 1024 to 784, and this you'll be going ahead with

**[13:53]** uh tan h operation. So, you'll be having the 784, and finally whatever is the 784, we'll be resizing out of after this tan h, we'll be resizing it, and then you'll be getting an image which is 28 cross 28. Okay. This is what uh uh uh will be the generator. Okay. So, now let's uh look at the discriminator part. Okay. Uh and the forward thing. So, you just pass it through the generator.

**[14:23]** And then the discriminator. So, the discriminator that we are having is you'll be giving an image. An image uh will be having 784 features. Okay. So, 784 to 512, we'll be again using leaky ReLU with the leak of 0.2. And then 512 to 256, and 256 to 1. Why we are going ahead with a binary classification problem. Okay. So, so that means that the architecture this is the This is the generator that we have, and

**[14:56]** then uh the discriminator. Discriminator, it is uh uh this is 784. Okay, I myself forgot. To 512, 512 to 256, 256 to 1. So, and then

**[15:24]** uh and please remember here I have also put it in comments here that after this please do not use sigmoid activation because we'll be using the loss with logits. So, it it will be converted the it will be taken into consideration. Okay. So, so please don't have any kind of sigmoid activation here. We'll be having the BCE loss with logits. So, I'll tell you how how it works. So, then you just pass the thing. So,

**[15:53]** this is just the forward through this NN. sequential that we have. Okay. So, now then we have to create the instances of this generator and discriminator. Okay, you just provide the input as the latent dimension. The latent dimension that we'll be using is 100. Okay. So, and then you create an instance of generator, you create an instance of discriminator. So, and then you print the generator and then the discriminator. And you can see it here. So, 100 to 256, 256 to 512, 512 to 1024, 1024 to 784. So, and then lastly you'll

**[16:24]** have a tan h in the generator. And then the discriminator will be 784 to 512, 512 to 256, 256 to 1. Okay. So, and then there is no activation function post this. Okay. So, now what is the loss that we'll be using? The loss that we will be using is BCE with logits. Okay, this is the loss. I'll define how how does it mathematically mean? And then the real images that are there I'll be

**[16:53]** indicating them with one and fake is indicated with zero. See, rather than using the word fake, which we'll be using anyhow, so it is good to use the word generated. Okay. So, and then the optimizer for generator and discriminator. See, you can immediately see here. So, you're providing the learning rate. This is what I was indicating earlier. We can have different kinds of learning rates uh for generator and then for discriminator

**[17:21]** and then since the these are two different optimizers all together, we can have different optimizers as well. So, you can use Adam in one case, SGD in another case, and then see how does it has impact on the outcome and how do you What do you mean by an outcome? Because these are images. Again, these sampled images uh has to be passed has to be uh what do you call measured using FID score and then that is the the the metric that we'll be using. So, now

**[17:53]** going back to this What is this BCE with logits loss? Okay. So, now let's look at it. So, now uh So, the real examples, we mean it. So, it's a binary classification problem. We mean one. And then the generated examples, so, we mean it by zero. And then what is

**[18:19]** this BCE loss with logits? So, this is loss with logits. Now, let's say that the logits that is coming out we'll call this as uh for as A. A is the logits.

**[18:47]** And then Y is the the label. Okay? That we have. So, then how it looks like is So, it is Y log sigma of A plus 1 minus Y log of

**[19:14]** 1 minus sigma of a. Okay. This is what we mean. So, see, y can take value either zero or one. If y is one, now this term will be taken into consideration. If y is zero, this term will be taken into consideration. And how does it help us in optimization? Now, when we go into the training aspects, I will delve into much more detail. Okay. So, now this is uh

**[19:45]** the the BCE loss with logits. So, now let's look at the the training aspect. Okay. So, we already have the So, the first thing that we'll be doing is we'll be putting this uh uh generator and discriminator in the training mode. And then we'll be going ahead with the whole of the data 25 times because we have decided uh the number of times that we'll be looping through is 25. Running loss, I have kept zero and zero. These are book markers.

**[20:11]** So, let's not worry about them. So, you take a batch of examples. Now, then what do you do? First, you go ahead and get how many examples are there. You know, what is the the batch size of the examples you get? This is again for bookkeeping. Now, then whenever you are getting the images, so that image will be coming from CPU. It will be located in the CPU. Therefore, what do you need? You need to port it into the GPU. That's what you are doing here. Real image.device, we already know this.

**[20:39]** So, now then the image that we have uh will be a matrix which is 28 core 28 matrix. So, but since we are using MLP, uh so what we will be going ahead is now we need to convert it into a uh a vector. Now, why do we need it? We already have the image. Now, the question is uh why do we need it? Now, let's go back to the loss function that we have. Okay.

**[21:09]** So, it takes, as you can see here, it takes xi as an input. Okay, as you can see here, it takes xi as an input. So, you take this xi, pass it through the the discriminator, which is there. Okay. This will be the output of the discriminator. Fine. Now, this is for xi. Now, xi is xi is uh sampled from px, so which is the actual data distribution, so for which we have put

**[21:38]** as one. Okay. And then, you can see this here. This is for xj hat, which is there, the jth example, now which is there. Now, this is the generated example. You take the generated example and pass it through the discriminator, one minus of it. Okay. So, when will this be taken into consideration? Now, it will be taken into consideration when we have the generated image. So, now

**[22:07]** into this, if I have So, if I put the yi and yi hat. Now, what is yi yi hat? It is not the actual labels of the images. Either it is real or fake. If I put this, now this will be only there for real images, so it will be one. So, one minus yi, it will be It will It will be one only when this yi will be zero. Okay. This resembles our

**[22:35]** BCE with logits. Therefore, we are using this. Okay. Now, it it is the choice that we made to go ahead. So, now we you are making it into a flattened vector. Okay. That is what we are going ahead. So, now the first thing we are going ahead and training the discriminator first. Uh since it is going in a loop, there might be some gradients that are accumulated from the previous step. Now, therefore, what do we do? We clear off all these uh

**[23:03]** gradients which are there. So, the discriminator gradients has been cleared. See, we are going ahead and training the discriminator. So, that means that we are optimizing for what? We are optimizing for W. Okay. So, now So, since uh uh we're optimizing for W, so now let's proceed. Now, first you take the real images and then you pass it through the very discriminator, so you get the

**[23:33]** output. So, what did you get? You get D W of XI. Okay. So, let's let's uh look into it. So, you got uh this. You got this D W of XI. Okay. Fine. So, this is what we have got. So, now we are making all these targets. So, real targets torch.ones for the batch size. This is why we needed

**[24:02]** the batch size. Wherein which every entry is one. That is why we have taken torch.ones of a column and then you are putting it onto the device. So, now you use this and then you go ahead and give real output and then real targets. So, all your real targets are one. So, now if you Now, if you go ahead for each example,

**[24:29]** this Y which is there is one. Therefore, you'll only be getting this term. Okay. Now, please remember there's a negative sign here. Okay. So, you'll be only be getting this term. Okay. So, wherein which what is this A? So, this A is logits. This A is nothing but D W of something. Okay. Now, whether it is uh uh an actual image or a generated image, it doesn't matter. So,

**[24:57]** you you have this DW of this. Okay. So, and then these are logits. So, once you take the sigmoid, it will become the decision of that binary classifier. So, this term I get for one example. This is exactly what I wanted for one example. Okay. And I get it over all the uh real examples, I get this term.

**[25:26]** Okay. This first term is what I have got. Okay. Now, please remember there is a minus here. You take that minus. Therefore, this maximization problem will be converted into a minimization problem. And why is this important? See, because when you use this optimizer.step, it only performs gradient descent. Okay. But, this W, which is there, now we need to perform gradient ascent. Now, therefore, that is again one more reason why we have used this BCE loss.

**[25:53]** Now, since you have that minus, so what you'll be getting is of this. So, you can take this. So, this will become a minimization problem. And then then you can use the optimizer.step. See, these nuances needs to be taken into consideration. Okay. Now, these are very important. Otherwise, sign mismatch happens. So, you'll be optimizing totally a different objective. And then you might be diverging from what you want. Okay. Fine. So, I got this first term. And then this maximization, this negative sign I have already told. So,

**[26:21]** let's not uh uh worry about it now. Fine. So, now I got this uh loss. So, which the first term, which I'm calling it as real loss. Now, I need to go ahead and generate the images. So, now what I need? So, I need the samples from the normal distribution. How do I take the samples from normal distribution? I know torch.rand n. So, the batch size, which I'm keeping it as 128. And the network has network is agnostic of batch size.

**[26:50]** That's totally fine, which we know from our earlier discussions. So, latent dimension is when which each example is is a vector of 100. So, and then you pass that through the generator. Okay. So, now we need again Please look at this. You need this XJ hat. How do you get this XJ hat? You get you sample Z and then you pass this Z

**[27:18]** through this discriminator. Okay, to do that first you have to get this XJ hat. How do you get this XJ hat? This XJ hat is obtained by passing this Z through the generator. Okay. Now, that's what we are doing now. First, you are passing it through the generator and then you are getting the the fake images. So, now it has passed it through the generator. Now, please remember it will be having the

**[27:45]** uh computational graph information. Now, therefore it is will be coupled with the parameters of the generator. But, we are going to update the parameters of the discriminator therefore we want to decouple it. Okay. So, now that's the reason why we use this fake images.detach. So, when which we are decoupling the flowing of gradients. So, detach prevents the gradients flowing into G.

**[28:13]** So, we only want to update the weights of the discriminator. Okay. So, now you take this uh so, you detach uh the thing and then you pass it to the discriminator. That will give you the fake output. Okay. Now, that is the logits that we have got. I'm naming it as fake outputs. Now, then what will be our targets in this case? The targets will be torch.zeros. Now, when which all the examples will be zeros. So, now now what has happened here is

**[28:43]** the Y. So, the Y that I'll be getting will be all zero. Therefore, what happened? This term will be off. Okay. So, 1 minus Y will be 1. So, therefore, this will be there. So, therefore, it is log of 1 minus of DW of this. Okay. These are the generated images in the case. So, this is what the term that we wanted. This is the second term that we wanted, which we got. So, log of 1 minus of this

**[29:11]** logits. Uh sorry. Uh sorry. It is not It is a sigmoid output. We will be having of A in this case. Okay. Now, that is what you can see it from here. Okay. And then again, it is sum of two negative terms. So, so this is again a negative term. This is again a negative term each example. And that summation will be sum of all the negative terms. So, you can take that negative sign outside. So, then if this will When you take the sum, now finally,

**[29:38]** this will turn into a minimization problem. So, it will not have any impact on the objective that we want to go ahead and maximize. Okay. Fine. So, we got the targets. And then you get the criteria. So, what is the total loss? Real loss plus loss real plus loss fake. Now, that is uh uh the loss with respect to the actual examples, the loss with respect to the generated examples. So, what do I mean by that? The first the summation and

**[30:06]** then the second summation is what do we mean. Okay. So, now please now please understand that this is not See, the when uh uh when when we give mathematical treatments, we need to use these similar terms that are there. When you go out and see some of the other repositories, uh there you see these terms. So, if you don't introduce these terms, it will be a a different kind of a trouble. So, now please remember this these two terms that we have is summation that we

**[30:36]** have the first term and then the second term is what is being summed. So, what is that I'm saying is you're actually getting this is what I'm saying. Okay. This is because of the samples from PX. Now, this is because of the sample from P theta. Okay. So, so I want to equate this. So, fine. So, you have the total loss. Now, then you what you do? You go ahead and uh

**[31:05]** perform uh loss.backward. So, and then optimizer.dstep. Okay. So, you go ahead and get it. So, this will update the parameters of the discriminator. Okay, you have updated the parameters of the discriminator. Okay. So, now So, what did we do here? See, we updated only the weights of this particular discriminator that we have. We updated the weights of this discriminator. We use these two sums and then we did it.

**[31:35]** So, now we need to update the weights of the generator. Now, then what we want to do? Now, these weights of So, again, if you want to update How do you update the generator? So, you need to compute this particular term. Again, so generator parameters are there and then the generated things has to go through the discriminator. So, then how do we do it? So, we update only the theta with this W. So, even though you pass this through the discriminator,

**[32:03]** so this W will be kept constant. Okay. So, now let's see this in uh implementation. So, now first optimizer.zerograd. So, we clear them off. So, and then fake output. So, how you get these uh fake images uh So, which are there, which we have already uh generated. You pass it through the discriminator. And then here we have torch.ones.

**[32:31]** Okay. Now, here I just want you to give an alternative viewpoint that we will be going ahead and implementing. Okay? So, now we already know that the theta star that we want is uh arg min of

**[32:59]** theta Now, which is one over the B2 This is summation J is equal to 1 to B2 log of 1 minus D W of G theta of ZJ This is this is this is this is log.

**[33:27]** Fine. This is what uh we need to optimize. This is the equation that we should be optimizing. Okay? So, theta, how do you update? Theta minus some learning rate. So, the gradient of this with respect to theta and W. Okay? This is the update rule that we have. So, now we will not be going ahead and implementing this equation. So, even though we got this equation, we'll be going ahead using another objective, now

**[33:57]** which is known as the non-saturating objective. Okay? The reason is uh implementation nuances, it has nothing to do with uh the uh the theory aspect of VDM that we have looked into. So, now let's look at it. The optimization that we'll be going ahead and doing will be &gt;&gt; [snorts] &gt;&gt; we're using this theta star now which is arg min of theta. See, I prefer to

**[34:25]** retain this arg min. So, minus one over B2 the summation J is equal to 1 to B2 log of DW G theta of ZJ. Okay? So, now see uh See that you don't have any equation for log of A A minus B. See, that is not what I'm doing here. Okay? Now, this is

**[34:54]** a totally a different objective altogether. Okay? Now, this objective, now which is there, is what is known as the this objective has a name. This is non non-saturating uh objective. Okay? Now, please remember these are two different things, two different mathematical functions, but they provide

**[35:23]** the same end result. Now, what is the end result that we want? That is what I'll show you. Okay? So, I'm just just making the point. Now, now these are two different mathematical functions. So, two different mathematical functions, but provide same

**[35:51]** end result. How does it provide same end result? That is what we will discuss. So, now see Now, let's consider this as the outcome. Now, not consider, it is the outcome of the discriminator. So, now let's call it as P. So, let's P is uh DW of G theta of Z. And then, once you apply the sigmoid, now this has to be between

**[36:20]** zero and one. Now, what is this? This is the probability that this is from a real example given some X hat. Okay. Now, I'm explicitly writing X hat because we are only considering the generated samples here. And even in this case or in this case, we are only worried about the generated samples. Okay. So, now if you take one example, what does this boils down to? Now, this is 1

**[36:50]** log of 1 minus P. This is minus log P. Now, this will be log of 1 minus P. Now, this will be minus log B will be the thing. Okay. So, now let's call this as uh loss one, which is what we have obtained. Now, that will be log of 1 minus P. And then the non-saturating thing that we have non-saturating optimize objective that we have, that is minus log of

**[37:17]** P. Okay. Now, let's take a case now wherein which the outcome of the discriminator, the P value that we have got, P is 0.1. The P value that we have got is 0.1. Now, let's see what happens in both of these cases. Now, in this case, so in L1, now what is log of 1 minus P? Now, this is uh

**[37:46]** log of 0.9. Now, this is minus 0.105. In the other case, this is minus log P. Now, in this case, now this will be minus log of 0.1. Now, this is 2.303. Okay. Fine. So, what? I got two different values.

**[38:15]** So, what will be your question? Okay. Now, valid question. So, now now what is that we want in both the things? This is also a minimization. We want to minimize this term. Okay. So, if you see, this is summation. So, you want to minimize this term. So, individual terms you have to go ahead and minimize. Okay? So, in both of these cases, so, you want to you want the minimum

**[38:44]** value of log of 1 minus P. Here, also, we want the minimum value of minus log of P. Okay. We want the minimum value that P take. Okay? Yes, we are optimizing the parameters, but the value that P is coming because of the interaction with the parameters. Fine. So, now, so So, what do we mean? So, we want 1 minus P

**[39:15]** to be smaller. Now, then, what do you want to do? Then, you have to increase P. That's what we need, right? In this case, similarly, even if you go ahead in this case, it will turns out you need to increase P even in this case. See, even though the objective functions are different, now, both of them, if you implement, if you go ahead and understand, both of them want to increase the the P value. Okay? So, now,

**[39:44]** so, now, what does this mean? So, both of these functions, so, both of these D of G of Z. Okay? Both of them want to increase this. Fine. So, now, you want to increase Now, how do you increase it? You can only modify the

**[40:12]** parameter. So, we want a stronger push. Okay? So, we want it to go there faster. So, whichever gives you the gradient which is stronger, I'll use it. So, now I want to go ahead and compute the gradient. Okay. So, now what is this P? This P is sigmoid of whatever activation A that we have. Okay. So, now Uh how do we get this? So,

**[40:43]** now what is uh L1 is log of 1 minus P. So, you want to What do we want? We want this the the gradient with respect to the activation and then from there it will go into the parameter. So, we just want uh to see what is this value. So, now this how do you calculate? Now, this is

**[41:18]** So, chain rule. Okay. Now, similarly in this case also this is uh L2 is minus log of P. Now, how do we obtain? So, we want uh this this is divided by gradient uh This is Okay. Now, uh now we want to compute now this is the case.

**[41:46]** Now, what is this? This is P into 1 minus P. Okay. So, now uh this uh leave it. So, you should be able to take the gradient with respect to sigmoid function. I hope all of you know this. This is what will be the thing. So, when you take uh the L when you take the partial derivative of L1 with respect to P. So,

**[42:16]** now what you will get? You will get minus one over 1 minus P and this is P into 1 minus P. And this will be uh minus one over P into P into 1 minus P. Therefore, what will be the gradients? So, in this case the gradient will be minus P. Now, in this case the gradient will be minus of 1 minus P. Okay?

**[42:44]** Now, this will be the gradients. So, now let's take a couple of numerical examples and see what will be the gradient value. Okay? So, now I'll give the P value. So, this is for L1 it is minus P. For L2 it is minus of

**[43:09]** 1 minus P. Okay? So, now if it is 0.1 Now let's start from 0.01, simpler one. So, minus 0.01, this will be minus 0.99. Now, if this is 0.1, this is minus 0.1. This is minus 0.9. If it is 0.5 this will be 0.5,

**[43:37]** this will be 0.5. If it is 0.9, this will be minus 0.9. Okay, there is a minus here, sorry. This is 0 minus 0.1. Okay? Now, we can immediately see from here the even though the direction in all the cases it is same. Okay? The direction is same, but the second case is having a much more stronger magnitude. Okay? Now, therefore, now

**[44:05]** what it boils down to, now therefore I can go ahead and obtain this. Okay, I can immediately go ahead and obtain this. Fine. So, this is the non-saturating objective which we have is a better way of in the sense stronger way, you know, better in the sense stronger way empirically in terms of obtaining the parameters. So, that is what we just

**[44:34]** looked at now. Okay? Now, fine. Then what? Now, there is another issue that happens. See. Now, this is for generated examples. Now, we already know that the Y, the label that we are having for generated examples is zero. But this term looks like this. This term, this is what it is if you use BCE loss.

**[45:02]** Not this. This is This is what we thought for generated ones. So, therefore what do you do? In this specific case, you just even though you are working with uh generated examples, makes Y as one. Simpler thing. Now, therefore this minus of this you'll get minus of this. Fine. And then that you're minimizing. Problem solved. Okay? This is how we actually deal with it. Okay? Now, this and this are similar.

**[45:31]** Now, the only difference is Now, here now Y normally it will be one when it is real, but since we are even though we are dealing with the generated cases, what do you do? You make Y as the uh one so that the problem is solved. Okay? This is what we'll be doing. So, now you can see here targets. That is the reason why in this case the targets the generated targets that you will have

**[45:58]** is torch.ones. This is the story behind this this particular line. Okay? And then you use fake outputs and then the generated targets, and then you go ahead and update the parameters. The the same old story continues. Okay? Now, this is how you obtain it. This is how you optimize for the parameters. Okay? So, we have we have uh updated the parameters of discriminator as well as the parameters of the generator.

**[46:27]** Now, I urge all of you I see here in each batch, one time I'm updating the parameters of the generator and one time I'm updating the parameters of the discriminator. Now, just uh try it out. Try multiple multiple times you train the generator and only one time you train the discriminator. And vice versa, multiple times you train the discriminator and only one time you train the generator. Just just have an

**[46:54]** inner loop inside for each batch. Okay? So, and then try it out. And then see what kind of output will you get. Will it have any uh substantial increase or substantial alteration in terms of the values of uh the the FID scores. Fine. So, now yes, now you can see this is how the loss is modifying. That's fine. Now, we want to generate the examples. Now, whenever you are want How do you

**[47:22]** generate the examples? You just discard the discriminator that you have. You sample from Z and then what do you do? You sample from Z and then pass it through the You sample from Z and then pass it through the generator. You get the images. So, first you need to get those random noise. torch. randn, you're taking 32 examples. Okay? So, 32 we want to generate 32 examples. So, you're generating 32 noise vectors, each of them with uh 100 dimension, and then you're putting it

**[47:49]** onto the device, and then you're passing this noise into the generator. Okay? And so you get 32 cross 784. That has to be converted into 32 128 28. Okay. Now because we are dealing with uh vectors, therefore you want to go ahead and do it. And then what you do? So the values will be between -1 and 1. That again you need to convert it to 01. Now how do you do it? You just add to the generated images. These are

**[48:18]** uh uh numbers. So to that you add 1 and divide it by 2. So that will convert it into 0 and 1. And then you make a grid. You port the the images got generated from the generator. Now you have already ported the generator through the to the GPU. And therefore you need to use this dot CPU so that it can uh be ported back onto the CPU and then you plot it. Okay. And these are those 32 generated uh examples. Okay.

**[48:47]** Uh see I know these are quite bad examples. I I'm with you on that. But you can see at least 019. Okay. Two to certain extent is able to get generated properly. Okay. So this is uh regarding the vanilla uh implementation of the GAN. Okay. Now what happens is now Now let's proceed towards the conditional

**[49:15]** implementation. See in conditional implementation, what happens is everything remains the same. The only thing that changes is whenever you are going ahead and training in each of the time you'll be passing your uh either uh the uh real images through the discriminator or you'll be going ahead and passing the noise through the generator, you will be giving the uh label information with it. Okay. That's the difference which is there. Okay. Now

**[49:44]** let's look at it. So let's look at the the conditional Now as you can uh see here, the only difference is so rather than having uh uh one uh input, you'll be having two inputs. So, one will be the noise, the other will be the label information. Here also, you'll be doing the same thing. See, uh now the question is how do you give Y as an input? Now, that will be our next question that needs to be

**[50:12]** answered in conditional GAN. And still here also, we'll be using uh MLP uh kind of architecture for discriminator as well as well as for the generator. How do you give this Y? See, this Z, now even though you keep it This is a 100-dimensional vector. These are 100 numbers. If you give just one, now it is a So, the label information is just one number in your whole input. Now, here also,

**[50:40]** this is X is a 784-length vector, and if you keep Y as just a single number, it might not have the impact that we want. Now, therefore, rather than giving Y itself, we want to give some kind of representation of Y. So, there are multiple kinds of representations that we can think of. One straightforward representation is you convert each of this label into the respective one-hot representation, and then go ahead and do it. Now, that is

**[51:09]** one way of uh looking at things. Now, rather than that, there is another interesting way which is known as learned embeddings. Okay? Now, what do we do is we learn the embedding of the number which is there. Okay? Whenever I mean learning, so it is not These are Whenever we say anything is learnable, that means that we are saying that there will be some learnable

**[51:36]** parameters involved. So, now Now, therefore, you'll have some learnable parameters. Now, then again, you need to update this parameter. Now, how do you do it will be the question. That I'll show you in the in the program. But this is what you are supposed to do now. Okay? Fine now. So now let's proceed. So everything else all the loss functions and other things remains the same. Okay. So now conditional generator. So the number of classes is 10. Embedding

**[52:04]** dimension I'm taking it as 10. You can modify this. Embedding dimension is what will be the length of the embedding vector that you're going ahead. What will be the length of Y? The rather than one I'm giving it as 10 numbers. So you can what do you call? You can have an increase over there. You can have a much more lengthier vector which when which can be given as an input. Fine. The latent dimension I'm keeping it as 100. So what we had earlier. So now this is what is known as the label embedding.

**[52:33]** So nn.embedding So number of classes is 10. And the embedding dimension is 10. Okay? So number of So what it will does it it will come up with the matrix. Okay? The matrix will have 10 uh rows. And then since the embedding dimension is 10 it will be a 10 cross 10 matrix. Okay? So this is embedding for zero, embedding for uh one, embedding for two, so on till embedding for nine. So So you will come up with the

**[53:04]** with the matrix like this. This will be a 10 cross 10 matrix when which each row represent the embedding of that specific uh label that you're going ahead. And how do we learn this? will be your question. These are the parameters. And these parameters you are putting it in the generator. So how will [snorts] the generator will look like? So now you have the

**[53:31]** the parameters uh So now it will be So embeddings. &gt;&gt; [snorts] &gt;&gt; Embeddings then the generator. And whenever you want to update the parameters of the generator, you should get your gradients from the discriminator. Uh

**[53:59]** So now anyhow you're going ahead and uh uh considering these weights of the discriminator, now which are there, you're not updating them. You're just passing the gradients in the reverse direction. Now don't stop at the parameters of the generator. Just push it ahead to these uh embedding uh weights, where in which each of them are updated. Okay, this is how we do it. See, [snorts] generator will get So this generator which is there

**[54:32]** now will have uh now will get X directly. Now, this is what do we mean by X? You'll get Z from here. And then you'll have another uh input that you'll be getting now which is uh you'll have uh

**[55:00]** embed you will get the Okay. So now you pass the label and then you have the Z. Okay, these are the input. This label which is there will be passed through this embedding. So what do I mean by that? If it is two, I check I take this uh second uh row which is there. And that and this will be concatenated and given

**[55:28]** as an input for the generator. Now this is what I mean by uh uh these embedding uh vectors are learnable. Okay. So fine. So for that you need this embedding. So, nn.embeddings. So, this is for all practical purpose it is similar to nn.parameters. Okay. So, then now what happens is your input dimension will be the noise dimension plus the label embedding. So, the label embedding will be a 10 dimensional vector is what we have decided. So, therefore it will

**[55:55]** be 110. And then uh you have the model which is exactly similar. So, there to our previous architecture, no difference. But in the forward pass there will be a difference. Why? Now, first you get the labels and you pass those labels through the embeddings and then you get the label vectors. Okay. Now, this will be the first step. So, you get the labels. You pass it through this nn.embeddings. You get the

**[56:23]** uh label vectors. Okay. So, and then you get the Z which is already given as an input. You need to concatenate and then proceed. So, you have this uh random noise that you have and then we have the information for the digit that is there. You can You concatenate them. torch.cat. Concatenate them using uh dimension one. So, and then you then you pass it through the model. Okay. So, in this case which is the uh the generator. Okay. Now, the

**[56:52]** conditional discriminator, how does it look like? So, discriminator again that also needs to have a input that needs to be taken into the uh embedding. Now, here also what do you do? Now, here also you do the same thing. Let the discriminator learn its own embedding for the labels. Now, that is why you have the label embeddings here and there is no need that the embeddings that is learned from the discriminator has to match with the No.

**[57:21]** Now, these are the parameters. It will be updated on itself. Now, you'll be only passing the label, okay, which is zero to nine, any integer between zero to nine. That will be converted into its respective embedding and then it is taken into consideration. Okay, fine. Now, uh first you have the input dimension, which is 28 cross 28, plus the embedding dimension, which is uh that uh we are considering it as uh 10. Okay. So, then what we So, it is 794. And then you have a classifier, now

**[57:50]** which input is 794, and then uh it goes on. So, how will be the forward function? Exactly similar. You take the labels, pass it through the embedding, and then you go ahead and concatenate the label vectors that you have got specific to the discriminator, and then you go ahead and pass through the model. Okay. Now, this is how the network structure of uh uh conditional generator and conditional discriminator will be. Fine.

**[58:17]** So, then you initialize them, port it onto the device, uh you use the same BC loss. Now, the everything the obviation will be the same now. Okay. There will be no change in the obviation story. Now, there will be a small nuance that is there whenever you are going ahead and training. Okay. So, you get these optimizer set, all those the same stories now hold even now. Okay. So, now here, you get this batch size, you get the images, as well as you need the labels. See, earlier in

**[58:45]** unconditional GAN, you don't need the labels. So, now you have these labels. Now, then what do you do? Now, you take this uh real images, you flatten them up. Okay. So, that's what you do. And then, how do you proceed with the obviation of the parameters of the discriminator? So, first uh you optimizer.0_grad. Now, you have these uh real images and their respective labels. Here, what do I mean by labels? It is 0 1 2, whatever is the label of that particular batch. You get

**[59:15]** the targets. Okay. So, you get the targets, and then So, the real outputs and the real targets, you just uh pass it on through the criteria. So, which is there you get the the first term in the in the summation. Okay. So, which is with respect to this is log of DW of G of X where X is from PX is what you have got. Now, you take the the fake the noise images you take the noise.

**[59:44]** Okay. Now, for that you have to generate the labels now. Now, because you have you need to pass it through the generator now. So, for that you need to specify the label. So, torch.randint you're starting from zero and then the number of classes it is 10 so it is will be it will be generating random integers from zero to nine for whatever is the batch size. So, now you have got the random label vector. Okay. So, when which each element will be a integer between zero to nine is what you have got here.

**[60:11]** Now, you take this labels that are there and then you pass the noise as well as the labels through the generator. Now, that will give you the images. Now, you need to go ahead and pass this again through the discriminator. Before passing it through the discriminator, you need to detach the what this generated images from the generator. That's what you will do. And then you obtain the the the fake targets using torch.zeros. So, and then

**[60:42]** you compute the loss. Okay. Now, this is the second term in the updation of W. Now, then you add them so you get the total and then you back propagate and then obtain the gradients and update the step. Now, this is with respect to the generator. Now, with respect to this is with respect to the discriminator. Sorry, this is with respect to the discriminator. Now, let's come to the generator part. Okay. So, now to the generator part. Now, first you need to

**[61:12]** clear off the gradient. Okay. Now, again you get uh the fake outputs. Now, you already have fake images and then you have already generated the labels for them. See, whenever I'm using the word fake, so what do I mean by that? It These are generated. Okay, so that is taken into consideration. And then you obtain the labels here with using one. Why one? I have already explained in detail because we'll be not be using the the function. We'll be not be optimizing

**[61:40]** the function that we have. We'll be optimizing a surrogate of it, which has a similar and better implications on the weights. Okay, so and then you get the loss and then you go ahead and update. The same old story holds on. So, now to generate you can generate a specific digit now. Okay, that digit you have to give. Therefore, what I am doing is here requested labels. So, I'm taking a list range 10. Now, that means I'll get 0 to

**[62:10]** 9. I'm repeating it thrice. Now, that is what I'll be giving as an input. So, the remaining things remain same here. Now, you can see these are the images that are generated. 0 1 2 3 to 9. Okay. See, so 0 to 9 I'm able to generate. This is far better generation compared to what we had in an unconditional generation. Okay, this is how we work with the

**[62:41]** conditional GAN, which is implemented using the MLP. Now now DCGAN so the DC GAN is just the implementation of the same thing using the deep convolutional networks. That's the difference that we have. Now, let's look at it. Now, we just need to So, see, the unconditional architecture and then the conditional architecture, everything remains the

**[63:08]** same. The way in which we give the labels, the way in which we use the embedding, everything remains the same. So, we just need to look at two different two different architecture. One is the deep convolutional architecture for generator and then the deep convolutional architecture for discriminator. In the discriminator we will be using the standard convolutional and in the generator, now since we want to increase the spatial dimension, now we'll be going ahead and using the transpose convolution. And how do we implement it?

**[63:35]** We just need to look at the network structure. Okay. So now uh the DC generator the deep convolutional generator. Now first you have the some latent dimension. Let's say that this is 100. You are you're using nn.linear. So using that nn.linear you are converting into 128 77. Okay. Now that means you are getting a vector which is a number which is 128 into 7 into 7. Now then

**[64:02]** Now what will be your convolutional blocks? So transpose convolutional blocks. Now it will be first it will have 128 channels to 64 channels. The kernel size is four and the stride is two and padding is one. Okay. We have already seen how does transpose convolution works in our earlier discussion. So we can you can refer back to that. Okay. So now then, so it will have 128 77. So that means that the vector which is there will be converted into a

**[64:30]** into hyper volume which is having 128 channels and then each channel will be of matrix of size 7 cross 7. So the output will be again a hyper cube which is having 64 channels. So and the spatial uh thing is 14 cross 14. Fine. So and then you use batch norm 2D. So we have already seen what is batch norm and then you'll be using ReLU here. No specific you can use leaky ReLU also, perfectly fine. So then the next layer will be 64 14 14. That you'll be converting into 128 28.

**[65:00]** So in channels is 64, out channel is one. So kernel size is four, stride is two, padding is one. So since it is generator, you'll be having a tan h at the output which is similar to what we have earlier. Now, how will be the forward process? First, you get uh the Z, and then you get this X, which is uh B cross 128 into 7 into 7. Now, that you need to resize into B 128 7 7. So, that's what you do. X.view.

**[65:31]** So, you'll be considering the last three dimensions, 128 7 7. So, the batch size dimension will not be taken into consideration. Okay. So, that you pass it through the convolutional blocks, you get the image. Okay. Now, this is regarding the generator. Now, coming to the discriminator, to the discriminator, the input is uh an image which is of size 28 cross 28, which is having one channel. So, you'll be using the standard convolutional 2D one channel to 64 channel. Kernel is four, stride is two, padding is one. So, we know how it

**[65:59]** works. Now, here I'm using leaky relu uh with 0.2, and in place is equal to true. Again, from now you have 64 14 14. That you convert into 128 7 7. Okay, just for the sake of symmetry, I'm using this. Again, you have the batch normalization. &gt;&gt; [snorts] &gt;&gt; Now, then you have uh finally, you need to put this into a single number, right? That's the point of the discriminator. So, 128 7 7 to 1. You'll have a linear layer uh for this. Okay. And how will be the forward? It's

**[66:28]** exactly similar. First, you get the images. Okay. So, then uh uh you obtain and pass it through the convolutional blocks. The convolutional blocks will give you a hypervolume, which is having 128 channels, and the spatial dimension of each of them is 7 cross 7. That you flatten it out. That's what you're doing it here in line number 72 here. And then, you pass it through the fully connected layer. That will give you a logit. Okay. So, then you go ahead and uh initialize

**[66:56]** the thing. Uh so, training everything is exactly similar that we have. No difference. Okay. So, then I'm taking some I'm sampling some things. Now, this is the images that are generated using uh DCGAN. Okay. So, using exactly the similar process, the only thing that is modified here is the G theta and DW which is there. Rather than it being an MLP, now it is a

**[67:24]** convolutional blocks here. So, now uh we will just do the repeat the same thing you follow the conditional uh thing. Now, again, you will have a learnable parameters. So, then the latent dimension now which is the noise, so will be added to this embedding. So, so your input uh So, it was 100, now it is 110. So, again, in n.linear, you will have a similar uh operation. You just uh go ahead and use the similar structure.

**[67:52]** And finally, you will have a tan h thing. So, how do you you get the labels? Whatever labels you have got, you convert them into the label vectors, and then you concatenate them with respect to the noise that you have and pass it through the the uh the fully connected layer, so that first you get uh a vector which is 128 into 7 into 7. So, that you are converted into hyper volume which is 128 7 7, and that is passed through the convolutional blocks,

**[68:19]** okay? So, to diagrammatically represent maybe that would be a better thing. Okay. &gt;&gt; [snorts] &gt;&gt; Now, first, what do you do? You have a fully connected layer. Okay. That will give you let's say that this is In this structure, it is giving you 100. Now, this will be giving a vector now which is 128 in into 7 into 7. That you are converting it to into an

**[68:50]** uh hyper volume now where in which you will have 128 channels. And this will be 7, and this will be 7 now which is there. Now, this you are passing it through the con blocks. That will give you an image which is of size 28 cross 28. This will be the this structure. So, if you are going

**[69:18]** ahead with the This is the generator structure. If you are conditional, you just add uh 10 here, now which will be attached to the embedding in the earlier notion. Okay, now then how will be the uh discriminator? So, the discriminator, you have this 28 cross 28. That you pass it through the con blocks. That will give you an output

**[69:46]** which is having a 128 channels. Seven here and seven here. That you flatten it out and then convert into a vector that is given to a fully connected layer. Now, this will be a single number. This will be the discriminator. This is the structure that we are using. Okay? So, now let's look at this. Now, how will be the conditional discriminator

**[70:14]** be there? Now, you'll be having this uh uh embeddings. Okay? So, then uh what do you have? You will have this con 2D. Okay? So, first you get all the convolutional operations. And then, you'll have a fully connected layer, which is there. Now, let's see the how is it uh being done here. Okay? Now, first you get the label

**[70:41]** embeddings. See, the label maps which is there is converted into 28 cross 28. Why it is converted into 28 cross 28, not 10 here, is for each image you are just adding it. Okay? So, okay, maybe I should have uh explained it at this point. Okay. See, here the label embeddings earlier we used to have it for 10. Now, here I'm taking it as 28 cross 28. Why? Now, because each image is a 28 cross

**[71:08]** 28. So, what I'll be doing is now this label information which is there that I'll be adding to it. Okay. Now, that is what I'm doing here. So, you get this information and then you concatenate them and pass it through the convolutional blocks. Okay. Uh this is what you'll do. So, okay. What do I mean by adding is I'll be concatenating. Maybe I should have chosen my words properly. [clears throat] So, you have B 1 28 28

**[71:36]** and then labels will be again B 1 28 28. So, your input for the conditional discriminator will be a two-channel image. Okay. That is used and then that you get the uh 128 into 7 into 7 vector and then that you pass it through the fully connected layer to get the things. Okay. So, then the whole training process everything now will be remaining the same as we saw in the conditional vanilla conditional GAN.

**[72:04]** Okay. So, I just want to show you the to generate an image. These are the generated images that we have. Okay. So, now I'm able to generate images. So, from all the four different things now now one is vanilla GAN, it's conditional variant using MLP. Now, vanilla GAN and its conditional variant using the convolutional blocks, deep convolutional blocks. Okay. Now, I have obtained this.

**[72:33]** Now, you can see this these images, the conditional DC GAN images that we have are far superior in terms of its thing. I'm not saying that it's perfect. Now, you can see that the nine here there are there. But anyhow, it is far superior than all the images that we have generated. Now, we are running it only for 25 epochs. Okay, very small epoch. So, it is far better in that case. Fine. So, now how do you obtain the FID scores?

**[73:01]** Okay. So, for that you need to import this uh uh Fréchet inception distance from this torch_metrics.images.fid. So, how do you want to just do it? It's It's quite simple. Now, first uh the images that we have generated will be from -1 to 1. That needs to be converted to 0 1. So, and then Now, here how do we do it? We just clamp it. So, we should clamp it to 0 to 1.

**[73:29]** So, that means that every value which is between which is less than 0 till -1 is just clamped. Okay. So, now it is just a one-channel image. So, now what do you do? You just repeat it for the three channels. Now, because the input for the InceptionNet will be a three-channel data. So, therefore you just So, now you have the repeated image which is there. Now, which is used to compute the FID score. Now, what do you do? You just say that how many features I want to

**[73:56]** extract. See, this inception network which is there has multiple layers. Now, which feature you have to take is what is being represented here. I'm taking this 2048 features. So, to this uh uh you're giving the image. Okay. So, then I'm taking I'm generating 5,000 images and then I'm passing it through and then getting the the number. Okay. That's all is is this whole code about. So, you can uh look at it. Now, you can see here.

**[74:25]** So, the number of images that I got is 5,120. The score that I'm getting is 21.5. See, lower the better. Okay. So, lower score is the better value. So, similarly we did uh the same for uh all the different GANs. The four different ones that we have computed. I have just repeated the same thing. So, it's a thing. So, you can see here. Now, these are the things. For vanilla GAN, we got

**[74:54]** 92.93. For conditional GAN, we got 104. See, this is interesting. See, even though we thought in the conditional GAN's images were generated good, but in terms of FID score, so vanilla GAN is doing better, but it's the same thing as reversed in terms of DC GAN. Okay. I'm just plotting them. So, you can see this here. So, these are the plots. It it it just for the sake of plotting. So, here, now what I have done is I have just

**[75:24]** tried a a [clears throat] nice experiment here. Now, what is it I'm trying to do is I'm generating one noise. I'm getting one noise vector that is of 100 dimension. Okay. I'm getting one of it. And then, I'm trying to generate with different labels. I'm giving all the nine labels to it. Okay. The same noise with label zero, label one, label two, so on and so

**[75:52]** forth. And then, I'm generating this. So, you can see here. This it's the same noise. So, I'm using conditional DC GAN here. The same noise, but the labels are different. Okay. You can see the output here. The same noise, but the labels are different. And then, I reversed it. Okay. Same label with different noise. Okay.

**[76:18]** Now, that gives a output like this. Okay. So, what does this mean? Uh your output is dependent on both the labels as well as the noise. Okay. More predominantly on the labels in terms of conditional GAN. Now, it is pretty much evident from the example that we have same noise we have used and then we provided different labels,

**[76:46]** we are able to obtain different images. Okay? Now, this is what the we have done. Now, this concludes today's tutorials. So, we just implemented in today's discussion and then we saw the mathematical equivalence between the the loss functions that we have obtained. So, you are going ahead with some derivation and then how is it translated into the loss function? We we saw it.

**[77:14]** So, that's what we will be doing throughout the this course. Now, whatever has been discussed in terms of uh mathematical thing, we'll be we'll be seeing its translation and then we'll be seeing its one-to-one correspondence. Now, how is the mathematical idea that we have is correspondingly modified into its coding what do you call things?

**[77:41]** Okay, so that's what we have discussed in today's discussion. We have used a single channel image. If you have compute if you have computer at your disposal, now please go ahead and train it for nice other images which are three channel and then you can make a nice uh uh block out of them. So, this whole notebook the link for this notebook will be made available in the description of this video so that you don't need to

**[78:10]** code it or generate it again. You can use this existing code and then you can go ahead and uh enhance it and let us know in the comments how does the images that you have generated looks like. We'll meet again in the next tutorials. Till then, have a nice day. Bye-bye.
