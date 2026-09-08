# Transcript — W2_T5: Tutorial: Tutorial: implementation of generative adversarial network

> **Source:** https://www.youtube.com/watch?v=iOb8vmlJd8o  
> **Channel:** IIT Madras - B.S. Degree Programme  
> **Duration:** ~41 min  
> **Note:** Auto-captions cleaned lightly. Minor ASR errors possible.

---

**[00:11]** [Music] Hello all, welcome to this tutorials on uh implementing vanilla GAN. So by now you would have looked at the mathematical formulation in detail on how GAN has been formulated using WDM. So now let's start take it from there and then see how do we implement

**[00:41]** the vanilla GAN. So now let's start the input. So we are implementing uh the vanilla GAN. Okay. Now why it is vanilla gan it not any flavor of it. Okay. There are a lot of flavors of the GAN. Now we are implementing the the basic flavor. So it is called as vanilla GAN. Okay. Now in your theory sessions now it would be like just GAN. Okay.

**[01:10]** Yeah. I I just write as vanilla GAN. That's all. Okay. Now the input that we have D is set of input X1 X2 so on until X N drawn IID from the distribution PX. Okay, this is the data that is given in

**[01:39]** our implementation. Now we'll be using data as mst. So in which we have uh images which is of size uh the dimension of image is 28 + 28. The reason why we have taken a very simple set is uh so that now we can run it and then see the outcomes. Now since our computing resource the GPU

**[02:10]** resources that we have is limited we want to take miniature data sets and then see its implementation in working. Okay. Now this is what uh the data that we have chosen. So now we should uh look into the the generator and the discriminator network. Okay. So now the generator network that we have we take

**[02:45]** samples from Z which is normal distribution with zero and I. Now we have this generator network parameterized by theta. Take the Z as input and then it gives it generates Xhat from the distribution E theta.

**[03:12]** This is the the generator network. And then we have the discriminator network which input. This is called as discriminator

**[03:38]** for that reason. It takes X or Xhat as input. And then it will give there will be a sigmoid function at the end. Now it will be giving the probability of y is equal to 1. Now which is with how much probability it belongs to the real data. Okay given the n. Okay. So now now we should choose the network. Now we shall

**[04:06]** be using simple MLP in both the cases. We look at the MLP structure when we go to the collab link. Now the generator network that we have generator network first we should decide on the dimension of Z. Now Z dimension we will take it as 100. You can take something else also but we have taken 100 a random thing. Okay. Now this will be the input for this generator

**[04:35]** network. And then now what will be the output of the generator network? Now it has to be an image. So which is of size 28 + 28. So now that means that I should have an output of 784. Now which I then reshape it into 28 + 28. Now in case of the discriminator network again I'll use one more MLP discriminator

**[05:05]** network the input now will be 784 now which is uh 28 + 28 dimensional image now will be the input the output now will be the probability that y is equal to 1 given whatever the input from the distribution that we have. Okay. Now this is the uh network structure. Now then how will be the loss

**[05:35]** function? Now uh you have derived the overall loss function. I'll just take that uh one equation from there now which will be comfortable to us to look at. Okay. The loss function will be J of theta, comma w equals expectation from the true distribution

**[06:05]** px log of the outcome of discriminator. Now you choose X from the true distribution PX plus expectation of with respect to P theta 1us sorry uh there's a log of log of

**[06:34]** 1us DW of X. Now this X is obtained from the E theta distribution. Okay. Now you have expectations. Now what we will do is now there is a step involved. Now we approximate this expectation with respect to sample mean and then we use the batches. Okay, we use the batches. Now the we the

**[07:04]** approximate expectation using uh sample mean and use badges. Okay. Now because of that now assume that uh now I take from this distribution px a batch of b1 and from this p theta a batch of b2

**[07:36]** okay so now because of which my j of theta w will be 1 / d1 summation i = 1 v1 log of dw of x i. Okay. Now

**[08:10]** plus 1 / b2 summation j = 1 to b2 log of 1us dw of x j hat. Okay. Now uh the way in which we have uh x1

**[08:40]** x2 till x b1 from all our iid iid from the distribution px and then we have uh x 1 hat X2 hat so until X B2 hat

**[09:08]** ID P theta from the distribution E theta. Okay. Now how did we get this fake image as it is called as the fake image. Now what did we do was now we sample it from this normal distribution and then we pass it through the generator network right and then we obtain this fake images. Now because of which what I can do I can

**[09:38]** rewrite this equation especially the second term of this equation just copy this whole thing. I can rewrite the second term in terms of the generator function because of which it will be like uh

**[10:04]** XJ will be g theta of Zj. So where in which you'll be sampling Z1 Z2 so on till Z B2 from IID normal 05 I

**[10:33]** okay now this will be the and then you have to have the minmax there will be the adversarial criteria so where which you have to maximize the parameter of the discriminator and then we should minimize the parameter of the generator. Okay, that is there. So now now let's look into the the

**[11:03]** generator network. I just recap this and then we will go to coding. network. Now we have this objective W star is R max

**[11:31]** of W with the same equation. I can just copy this this. Erase this and then I'll write it

**[12:04]** completely. have this we want to find the arg max with respect to w that will be our w star okay since we have a maximization problem now we will not be doing the gradient descent actually we'll be doing

**[12:31]** the gradient ascent okay now because of which WT + 1 is okay is assigned with WT plus the learning rate with respect to the the discriminator function. Now we can we will be having different learning rate for generative fun generator function and then the discriminator function. We'll be having two different learning rates and then the gradient with respect

**[13:02]** to the generator parameters of the objective function. Okay. And it is uh makes sense that while training the discriminator all the parameters that are involved with the generator all the theta which is involved with the generator are kept constant. Okay. So now how do we do

**[14:08]** it? Okay. Now you have uh this network. Now let's take this network them. How do we do it? Now just for

**[14:35]** reference we'll keep this. Now first we will uh sample Z1 till Z B2 from IID normal 0 I. Okay. Now relook at the objective. You need this. Then you have to pass it through the discriminator function.

**[15:03]** uh first uh zed you sample pass it through generator function you get the fake images that you pass it through the discriminator function that will give you this loss value. Okay. Now then you pass through the generated function G theta of Z because of which you get G theta of Z1 G theta of

**[15:35]** Z2 so on till G theta of Z B2 okay now these will be your fake images now once you have these fake images. You pass these fake images. You pass these through pass through the discriminator network and uh get the second

**[16:04]** term. Okay, you get the second term. So now see it is 1 minus. Okay. Now you should be looking at this probability using BCE loss we can get it. It's a straightforward implementation of uh BC laws. Okay. So now then what do you do?

**[16:34]** You take X1 till X b1. Okay. to take a sample from PX, okay, then pass through the discriminator network, okay, and get the first term. Okay? Now you add these two terms.

**[17:07]** Now your total loss for uh the discriminator. Okay. So total loss for discriminator will be term one plus term two. Okay. Now using which we only update the parameters of discriminator. Okay. What do we do? We only update these parameters using these laws.

**[17:34]** Okay. Now this is how you implement the weight update the gradient ascent. Now this has to be done for the discriminator network. So now coming to the generator network. How do you work with the generator network? Okay. Now let's start with that. Let's take the generator network

**[18:02]** into consideration. In the generator network you have to find the theta star which is the arg of j of theta comma w. This is what we need. uh we can obtain this from the previous

**[18:33]** we can just need to reiterate the same previous equation because of that reason I'll just copy this equation. Yeah. And this you have to find the argument with respect to theta.

**[19:05]** Okay. Now if you closely look at this equation now this first term has uh no theta involved. Now it is just the discriminator. Only the second term is involving the theta. Now therefore we can get rid of this first term since it is uh independent of theta. The first term

**[19:31]** is independent of theta. Therefore your theta star will be ar with respect to theta. How do you do it? 1 / bx 2 summation j = 1 to b2 log of 1us dw of g theta

**[20:01]** of zj of brackets. Now we need to take care of them. Okay. Now you need to update. Now we use the gradient descent in this case because it's a minimization problem. Now theta t + 1 is equal to theta t learning rate it can be same or the okay now whenever you are training

**[20:41]** the generator okay now we are training the generator we have to keep the parameters of the discriminator constant okay now We'll training the generator. Generator keep of

**[21:19]** discriminator constant. Okay. Now because of which your uh gradient passing okay let me put the diagram once we put the diagram it will be much more have whenever you're doing this will do

**[21:52]** is the gradient will pass through this whole network. I'm just taking it the other way around so that okay this is the backward movement while doing it now we don't update the weights of W okay update only

**[22:23]** theta with W constant. Okay. Now, how do you do it? You sample from So, you need this. You first sample from the normal distribution. Then you pass it through the generator function and then pass it through the discriminator function. Okay? And use BC loss with respect to the real images. Now as you need to uh trick the

**[22:52]** discriminator you'll be using the BC laws with respect to the real images your target will be the real the labels of real images which is one in our case. Okay so now now with this much of uh background uh detail in hand now let's go to collab. Now what I have done is now I have uh pre-written the code. The sole reason why I have done that is we can

**[23:21]** just skim through the code once and then I have kept the uh outputs ready. Now we can uh look at uh those outputs. Okay. Let's go to the code. Uh this is a Google collab. Now we have discussed this in our previous tutorials in which we looked into PyTorch. Okay. So now these are the necessary packages. Now torch is to

**[23:52]** import the PyTorch for neural network for optimizer. To get the data is torch vision to transform the data use transforms. To get the data loader you use the data loader and then to plot the generated images we use this PLT. Okay. These are the necessary libraries that I have imported. And then we'll be using the the CUDA device. Now we'll be using the GPU. So torch device CUDA if torch.ca is available. Now if this is

**[24:22]** true, use CUDA otherwise you use CPU. Okay. And then you have the transform here. Now you have composed a group of transform. Now one is now first you convert it to a tensor. Now once you have converted into a tensor you normalize it with mean 0.5 and the variance 0.5. Okay. Now this is the uh the mean tpple. This is the uh variance tuple. Now if you are since we

**[24:52]** are using emnest which is a single channel data we are giving only one value. Now if we are using a multi- channelannel data we should be giving multiple values. Okay. So and then uh you're obtaining the data sets torchvision.datasets.mnest. Okay. Now the data you are downloading into dot / data. Now train equals to true. You're downloading the training data. We don't need we are not doing any inference. So I don't need the test data. Download equals to true. Now once you have

**[25:19]** downloaded the image you apply this transformation. Now what do you mean by that? You convert it into a tensor and then normalize it. Now similarly you create a data loader. Now with batch size of 128 and then you shuffle them each. Okay. Now look at the generator network. Now in the generator network now as we discussed earlier now either uh it's a see finally you want to approximate the function g

**[25:47]** theta. So now you need a function approximator. Now either you you can use any variance of linear theoretically you can use any variant either the choice are now either uh you use uh MLP or you can use uh convolutional neural networks. Now I have used MLP. So I have created a class generator which is a child of NN domodule. Now in which I should be giving in the init

**[26:17]** method apart from the instance itself I should be giving noise dimension now which we'll be fixing as 100 and then the image dimension which will be 784. Okay. Now these are the two parameters which can change and then you're calling the parent that is nn.modules constructor with the generator as uh the parameter and then you are creating a stack n.sequential. Now these are the sequential process. This is the same way

**[26:45]** in which we discussed earlier also. Now first you have nn.linear you have a mlp from noise dimension which is 100 to 256. Now then you use relu activation then you have 256 to 512 again you have relu 512 to 1024 again ru 1024 to image dimension now which is 784 in our case and then I'm applying tan h because we want to normalize the image

**[27:13]** now it has to be in the range minus 1 to1 okay now this is my generator method and then let's go to the discriminator method now where in you have uh similarly you have initiated the discriminator nsequential linear from the image dimension which is 784 to 512 now here I'm simply using leaky relu different activation

**[27:41]** function okay and then uh 512 to 256 again I have leaky relu fight 256 to 1 now why I'm using one because you should be getting the probability so what I'm doing is I'm passing ing it through a sigmoid function. This is my discriminator method. Okay. Now both of them are MLP multi-layer perceptrons or fully connected layers or fully contained network whatever you want to call them. Now the noise dimension is

**[28:09]** 100. Now this is the parameter that we have fixed and then now you have uh image 28 + 28 that is the size of your input image mst 28 + 28. You are creating an instance of the generator by passing noise dimension and image dimension. And then you are putting it to the device. Okay. And then you are having the discriminator with image dimension and then you are putting it to the

**[28:37]** device. And then you have two optimizers. Now because uh uh so from our discussion now it is evident that whenever you are uh updating the values of generator your discriminator values has to be kept constant and whenever you are updating the discriminator the generator value has to be kept constant because of which I'm having two different optimizers now one optimizer is for generator the another optimizer is for discriminator G optimizer and D optimizer now both of them are atoms

**[29:08]** okay and then with learning rate 2 into 10 ^ of - 4 and then the loss that we'll be using is BC laws. Now by now when you look at the outcome of a discriminator as a probability as the outcome of a binary classification problem. Now you can understand the log of 1 minus DW or just log of DW you can interpret it as the outcomes of a binary classification problem and then you can use BC loss.

**[29:39]** Okay, now these are the necessary parameters and the optimizers. And what am I doing is I'm uh using some fixed noise. Now to generate I'm using some fixed noise generator. Uh I'm I'm trying to plot the generated images using this fixed noise. I'm generating it always for all the loops I have keeping this as fixed. Okay. and

**[30:07]** then I renormalize it and then plot it. Okay, this is just a plotting function. I'll not be going deep into this. Now, this is not of a plotting session. No, we will not be going into that. Okay, so now let's look at the training. Now, before looking at the training, now let's look at the function call from where the training has been uh called. Okay. Now as was discussed in theory session now you can have multiple

**[30:37]** variations of training. Now you can update the parameter of generator once and discriminator once. Now that is one way you can update five steps of generator and one step of discriminator or the five step of discriminator and one step of generator. Okay. Now uh now whenever you update generator more now you tend to get good fake images. Now discriminator more it is able to classify okay properly between the real

**[31:06]** images and fake images. Now looking at the images that are generated now you should come up with a stopping criteria. I have just used a simple 50 epoch so that it is just the what do you call a trial run. Okay. So now we will see actually with 50 now you don't get uh very good reconstructions but you will start seeing the uh what do you call the initial stages of images getting

**[31:36]** reconstructed the images getting generated not reconstructed generated to be more specific okay so now let's look at these three cases now all of them I have implemented in only one function okay now you should give train loader And then the number of epochs and which mode do you want to work in one one where in which you have one step of generator and one step of discriminator or you want to work in the mode of five gen one discriminator or

**[32:05]** five disk and one gen okay now this five is not something holy okay now you can change it and then you can do it okay so now let's look at this I have taken the fixed noise now as uh so that I can generate for the same noise images after some epochs and then compare them comfortably. Now for epoch in range number of epochs. Now you are training

**[32:34]** it for some 50 epochs you have uh you have enumerate train loader. Now first you will so whenever you have this enumerate first you will get the index and then from the train loader you get the image and its uh label. Now since we are dealing with MNS the labels are also there but labels are not of consequence to us in vanilla GAN so let's not consider the label I'm calculating the batch size okay and then I'm flattening it okay so batch size

**[33:05]** comma 1 I'm putting it into device and similarly I'm getting the real label start to do once now where which all ones are there what is the size batch size comma 1 I have taken bat size of 128 8. So now 128 ones will be there. That also I'm putting into device fake labels which is start.0 which is an indic 0 is an indication that it is the fake data and that also I have 128. Okay. Now this data is ready. So

**[33:36]** now now let's look at training the the discriminator. Okay. Now let's go back and revisit the equations and see the one toone mapping between the steps that we did in theory and in practice. Okay. Let's look at the training of Now first you take real images which is

**[34:06]** from px and pass it through dw to get the first term in the loss. Okay. Now let's see whether this is being done. Now you take the real images, pass it through discriminator and then you get the output. Okay. And then you calculate the loss using the BC function. Okay. Now that means that the first term is ready. Okay. So now let's take the second one.

**[34:43]** Now what is that you're supposed to do? You have to generate the samples from the normal distribution, pass it through the generator and then pass it through the discriminator. Okay. To get the second term. Now uh this is the step in which tot.rand you're generating from the standard normal. Now with the bat size and then the noise dimension which is 100 that you are putting onto the device you are passing it through the generator

**[35:11]** you are getting the fake data. Okay see here you have got this fake data and then you need to pass through the discriminator and then you pass through the discriminator. The reason why you have put dot detach is that so that you don't need to compute uh uh you don't need uh the fake data which is generated to be there in your computational graph because of that reason you have

**[35:39]** detaching it so that the gradients no need to flow in that direction you don't need to calculate that and then uh you calculate the loss now you have got the outputs now using which you calculate the loss now outputs which is from your discriminator and the label which is the fake label you calculate the fake fake score. Okay. Now you have got term one, D loss real which is term one and D loss fake which is term two. You add both of

**[36:09]** them. Okay. And then you need to update the discriminator values discriminator.0 grad. Now you uh clear off all the previous gradients. D loss which is addition of these two. You do the backward. Now why only? So you're using D loss here. Now please remember that and then you use D optimizer.step. Now that means that only the weights of the discriminator will get

**[36:40]** updated. Okay. This is the discriminator training. Okay. So now let's look at the generator training. Okay. Now let's go back to the generator. How is it done? Now you sample from look at the equation. You sample from Z pass it through

**[37:07]** the generator then pass it through the discriminator and then get the outcomes. Okay. Now get the samples from Z pass it through the generator then pass it through the discriminator to get the output and then to trick the discriminator now as I discussed earlier now we'll be passing the real labels and obtain the loss now and then you zero grad all the gradients

**[37:37]** of the generator or the parameters in the generator you calculate only okay you calculate only the gradients with with respect to the weights of the generator and then you update only them. Okay. So now now this is the training of discriminator and generator. Okay. One one training. Okay. Now then what I have

**[38:04]** done is now if your uh mode of training is five gen and one discriminator. Now you have already done one step of generator and discriminator. So now you need to generate four more uh so you need to update the weights of generator four more times. Now the same code I have placed it here for I in range four. If that is the case this will happen. Now if this five disk and one generator now one gen has already been up updated. Now the discriminator is updated four

**[38:32]** more times here. Okay. Now similarly this five is not anything uh holy. Now you can just uh use it and then for every 10 epochs I'm printing the loss okay and then of both discriminator and the generator and then I'm uh plotting the generated images using the fixed noise. Okay now we have fixed the noise here for the same noise after each 10

**[39:00]** steps I'm showing them. Now when I run this I'll start getting the output. Now this is the training of one step gen and onestep discriminator after 10 epochs. Now this is the discriminator loss. This is the generator loss. And these are uh 8 + 8 generated images after 10. After 20 now you can see some of the images are becoming uh much more comfortable. After 30 you can see more.

**[39:30]** After 40 you see some more structure. after 50 now you see a better uh thing that is happening. Okay this is one generator one discriminator five generator and one here. Okay. Yeah the outcomes are not so great but you can make sense of it. And

**[40:01]** similarly five discriminator one generator one. Okay. Now this is the the overall thing. Okay. Yeah. I hope you had fun. Now what I suggest you is uh couple of simple exercises. Now you can keep we'll be sharing this uh collab with you. You can keep all of them same. Now you can change this MLP to CNN and see how the uh generated images will be

**[40:32]** better. And then one more thing that you can try is rather than using MNEST which is a straightforward or simple data use some other data and see how actually the regeneration happens. Okay. Now I hope uh the tutorials was helpful and fun. Now see you in the next tutorial coding the next variation of GAN. Till then bye-bye.
