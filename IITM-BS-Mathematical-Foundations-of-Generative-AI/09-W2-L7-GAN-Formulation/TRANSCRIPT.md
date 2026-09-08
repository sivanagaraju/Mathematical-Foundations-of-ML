# Transcript — W2_L7: Generative adversarial networks: formulation

> **Source:** https://www.youtube.com/watch?v=pLD5Q5cS4kI  
> **Channel:** IIT Madras - B.S. Degree Programme  
> **Duration:** ~35 min  
> **Note:** Auto-captions cleaned lightly. Minor ASR errors possible.

---

**[00:22]** implementation implementation of a GAN in practice. So here I'm assuming that the audience of this particular course are well versed with uh trying neural networks especially the error back propagation algorithm using uh gradient descent. I will not be going through those algorithms assuming that the audience here know how to train neural networks. Typically what happens is you

**[00:50]** know as I've been saying this course is by and large architecture agnostic. So I'll not be telling you what sort of architecture that is to be used for uh degenerator and discriminator. It does not matter. In general these things uh will happen to be some sort of neural networks. So these can be any neural networks right? This can be a a multi-layer perceptron which is also called as the feed forward neural

**[01:17]** network or it can be a convolutional neural network and so on. So similarly with this you can this is also another neural network with any kinds of architecture you can it can be a CNN it can be an MLP and so on. So we will do some of this during the tutorials but I'm assuming that the the audience here are well versed with training neural networks. So you have two neural networks so that let us write the algorithm. So the input for this that you only have

**[01:48]** data okay you have n data samples so we are given n data samples that are drawn iid from unknown distribution px this is all we have okay so we need to choose an architecture for uh the generator and the discriminator network so this is the the generator network and this is the discriminator network.

**[02:28]** As I said for this particular case, the t function can be equivalently represented using a function that would take x and maps it to a number between zero 0 and one because the d function is defined that way, right? It is the sigmoid of the b function. Uh so the sigmoid is always bounded between 0 and one. Therefore, this can be interpret interpreted as a classifier, right? a binary classifier. So we will see more about it

**[02:56]** in uh in the next module. But yeah, so this is called the discriminator. So what we need to do is input is input that are given are uh just the data set with n samples drawn id from an unknown distribution px. So what we should do is we should first train the discriminate uh the generator and discriminator alternatively. So what do we have the objective is that we we need theta star and w star to be or rather I'll write that uh

**[03:27]** separately. So first we need our w star which is the equivalent of the t function. What do we need? We want this to be maximized. Remember that we got this bound by maximizing uh the f divergence uh the objective of the divergence with respect to the t function. So we need to maximize this expectation. Now I will change the expectations. Maybe I'll do it the next step. So this is expectation of log of

**[03:57]** dw of x with respect to px and you have uh plus expectation of log of 1 - dw of xcap with respect to p theta. So we know that uh while we are uh optimizing these kinds of expectations in practice we'll have to approximate these using sample averages. So this is arg max. So I'm assuming that people also

**[04:27]** here know that uh uh that neural networks are trained using batch gradient descent. Okay. So I will approx I will replace this expectations in terms of batches. This is log of dw of x i plus 1 by

**[04:57]** b2 j is 1 through b2 1 - sorry log of 1us dw of xjc cap. So this is what the cost function is where x i or x1 up to x b1 are samples drawn

**[05:26]** from bx. Okay. And uh x1 cap to x b2 cap are the samples drawn from b theta. So this is the optimization that we should solve. Okay. So now uh how do we do this? Then we do one gradient step. Okay. The parameter upgrade that we do is this way minus some. So note that we are maximizing. So this should be plus some

**[05:57]** alpha 1 which is the learning rate times the gradient of the loss function which is j gan theta comma w with respect to w. Right? This is what we need to do. Okay. So now how do we do this in practice? So this is one gradient step. This is discriminator we train the discriminator

**[06:36]** and while the discriminator is being trained we have the generator network to be constant right so similarly we'll have to look at the generator network as well which is we need to minimize the bound that we are constructing on the divergence we will minimize j theta chet sorry jan jan So we have theta comma w. We have to minimize the same objective. And note that we have this adversarial

**[07:04]** nature here in the sense that uh the uh discriminator is trying to maximize the objective and the generator is trying to minimize the same objective. So if we simply replace that using sample averages then we need to minimize the same objective with respect to the same objective has to be minimized with respect to theta here. Okay. Now where is the dependence uh on theta? If you

**[07:33]** look at it the first term here is independent of theta and we can take that out right. Why we are training the generator? We can take this thermode because this is this is independent of theta independent of of theta. Sorry. Yeah, this is independent of theta. And the second term should be dependent on theta. I mean while apparently one will not see

**[08:01]** the dependence on theta here. This is actually equal to the following. Right? Let me write that down. So this is uh minimizing with respect to theta. I'll just state the first term down here. This is 1 by b2 which is some bad size. J = 1 through b2. We have log of 1 minus dw. What is xj cap? By the way, xjcaps are sample from theta. So what are those? Those are simply xj is equal to g

**[08:32]** theta of zj. Right? This is the way we have defined our XJ caps. So this would be dw of g theta of xj. Right? So now the dependence on uh uh the cheetah function is apparent here. Right? So this is the loss that we are going to optimize for. Okay. So what we should do is one step one gradient step towards the updation of the generator would look like this. So note

**[09:03]** that this is one other step size. The gradient of the loss function with respect to theta is what we should see. And this is this is one gradient descent step. One gradient descent generators. Right? This is one gradient

**[09:34]** descent step through generator. And here we have a step size. This is uh a step size gradient step size. And we have another gradient step size here. So we need to use can use this to be the same or we can use use these to be different step sizes as well. But yeah, so these are uh STM sorry. Yeah, these are uh step sizes for gradient desync right. So here while we are doing

**[10:05]** this we'll have to keep theta as a constant right over here theta is kept a constant while you are while you're optimizing for the discriminator while you are optimizing for the generator obviously w is kept a constant right the parameters of the discriminator are kept a constant okay we have to we'll have to solve these optimization problems alternatively that you uh update the generator step generator

**[10:35]** parameters and by keeping the discriminator parameters constant and then you update the discriminator parameters keeping the generator parameters constant. So please see that while the generator while the discriminator is trying to solve a gradient ascent problem because typically when you do uh when you solve optimization problems you do gradient descent because most problems in machine learning are cast as minimization problems. Here you have a plus here uh while updating the discriminator

**[11:06]** parameters because we are maximizing here right I mean you have plus the parameters here are trying to maximize this function and the parameters here are trying to minimize the function and that is where the adversarial part of it comes into picture. Okay. Okay. So let me just uh write down the uh training step separately. So to train the discriminator to Okay. So let's Yeah. To train the

**[11:37]** discriminator. To train the discriminator. How do we train the discriminator? So we keep the generator keep theta constant. So we are not touching the theta network. Okay. However, we'll just I'll just write down here. So note that to train the discriminator network, we need samples

**[12:07]** both from px and p theta. Let me write that down. We have z samples sampled from normal 01. We have g theta of z. And here we get xcap sample from d theta. Okay. Now what we should do is so note that we are given uh some samples from the data. So we have uh data samples x1 x2 up to xn. So we select a batch of uh samples from

**[12:39]** so first what we have to do is define b1 to be a batch of samples from x1 x2 up to x b1. So it need not be continuous right? It is simply some random samples taken from D. Okay, this is a subset of D. Just take one batch of samples from the given data set. Okay, then so we are training discriminator here, right? So maybe what we are training I

**[13:07]** will uh write that in a different color. So what is being trained? Let us write that in green color. So we are training the Okay.

**[13:33]** dw. Okay. Now note that the loss function for this uh so j theta sorry j gan which is a function of both theta and w is given by this particular equation that we wrote right let me just copy that equation and write Yeah. So, j theta is given by that. So,

**[14:14]** how do we compute the first term here? The first term is computed by simply passing x1 x2 up to x b1 through this. And we can compute this particular term right. Uh sum of i is 1 to uh b1. We have log of dwxi. Okay. We can simply compute this term here. So what we should do is that

**[14:43]** we should simply pass x1 x2 up to xb through the discriminator network. Right? Just pass them through the discriminator network which is also called one forward pass. we pass them pass it through the discriminator network and we get this. So how about the second term? How do we compute the second term here? So note that this in the second term jcap is nothing but g theta of zj right that is our definition of z j. So what we have to do is that

**[15:15]** maybe we'll copy this that we have some space there. Okay. Now what has to be done is first we need to sample sample G1 Z2 up to ZB2. All of these are sampled from normal 01. Okay. then pass

**[15:43]** Z1 through ZV2 through the generator through G theta with fixed theta right so because you know note that we are updating the discriminator parameters here so we'll have to keep the fix the theta so what happens is we have here Z1 1 up to ZB2. We are passing this through the

**[16:13]** discriminator network uh sorry the generator network with fixed theta. Okay, we pass this and what do we get? We get g theta of Z1 to G theta of ZB2. So this is what we get. So here you write fixed theta. So we have a fixed theta here. We're not we're not changing theta at all. Okay. What is

**[16:40]** being triled is written in the green uh green color here. Okay. Now what we do is that once we get g theta 1 to g theta_2 we have to pass this g theta z1 to g theta z b2 again through the discriminator. Okay. And once we do this we can compute this 1 by v2 j is 1 through v2 log

**[17:15]** of 1us 1 - dw of g theta zj right this can be computed I don't think there are three brackets here. Hope that this is clear. So what do we do is that the discriminator I mean the loss function has two terms. One where the samples are coming from ex right the first expectation. The second

**[17:44]** expectation where the samples are coming from d theta. Okay. So for that we need to first get samples from px and samples from p theta. How do we get that? So samples from PX is simply get um some B1 samples get a batch. So like take a batch of samples from the data set itself. That is straightforward. Okay. Then to get samples from Okay. So maybe yeah to get samples from P theta what do you do is that you sample Z1 through ZB2

**[18:16]** uh which is a batch of samples from normal distribution and pass all the all those through the generator with the generator parameters being fixed. So now we will get g theta z1 to g theta zv2 which are the samples you know the what are these actually these are actually samples from p theta of xcap right okay and these are samples from px obviously correct so we have s we now have samples from p theta of xcap we now have samples

**[18:43]** from px of px as well so pass all those and by the way these are not I mean you can you can do it either at a batch level or you can do it at a sample level depends upon what sort of implementation are you doing. So pass samples from px through the discriminator and compute the first term in the loss and pass the samples from p theta through the discriminator compute the second term in the loss. So once you compute these two terms, what is to be

**[19:11]** done is that you compute the gradient okay of this J gan J gan with respect to W right and then uh back propagate this loss back propagate this loss through the discriminator till you get to the input of it till you get the get to the input of

**[19:37]** the discriminator. Right. Once you do this, you simply update the parameters of the discriminator by usual gradient descent but with a plus sign here because what we are doing here is gradient ascent. That's it. This completes one step of training through the discriminator network. I hope that this idea is clear. So while we are doing this we will keep.

**[20:10]** So maybe I will write this. So this is uh green is for uh a forward pro forward propagation and uh the red is for back propagation. Right? Okay. So only only this green network is trained. So we don't uh we will not touch the generator network at

**[20:38]** all while we are training the discriminator. Okay, hope this is clear. So we'll do a similar thing for the generator now. So now to train the generator network. Now how do we do this? Let us uh do the same thing. We have uh the generator network

**[21:06]** here. Okay, I will write this in green now because this is what we are going to train by keeping the discriminator parameters fixed. So we have the generator network here and we have a fixed discriminator. While the generator is being trained, we have the discriminator to be fixed. Okay. And these who take uh samples uh we will write that down

**[21:41]** later. So this is dw. Okay. So now this as we know we'll take samples from samples from Z. This is G theta of G and we have samples coming from P. So now uh here right this is a fixed w because we are only training the generator. So what is the loss function here? As we saw the uh the first the the first term is independent

**[22:10]** of theta. So we can completely ignore that. We will only take the uh second term which is this one. right? We need to compute this term now. How do we compute this? It's pretty similar to what we did the previous

**[22:37]** time. But note that in this case uh we don't need real data, right? Or rather data from PH is not needed at all. while you train the uh generator network. For the discriminator network, you need and while training the discriminator network, we need data both from px and pja. But while we train the generator network, we don't need data from px at all. Okay, because the first we is independent of the generator parameters are independent of the first term. Okay, how do we compute this? Simple. So we

**[23:07]** first sample Z1 through Z B1. Did we call it B1? No, B2. Okay. We sample Z1 through ZB2 from normal 01. Okay. And then we pass all these. So Z1 through Zp2 through this network and what we get

**[23:37]** here is G theta of G theta of ZB2 we get all these samples right which is needed here. Now take all these samples and pass them through the discriminator. Now note that the discriminator uh parameters are held a constant here. Okay, take all these samples and pass them through the

**[24:09]** discriminator. Pass them through the discriminator and compute. What should we compute? We should compute dw of g theta of zj, right? We compute this. Of course, I mean we need to compute the entire thing. Okay. we needed this. Okay. So, let me just

**[24:35]** regrate it. So, what we need to compute is we need to compute uh 1 by b2 sigma j is 1 through b2 log 1 minus dw g theta of zj. So we need to compute this thing. Okay, you compute this. Then what

**[25:18]** do we do? We compute the gradient of this particular thing with respect to to theta now. Okay. So this is what our uh uh J gan is right. This is RJ gan in. So in the case of discriminator also yeah this is RJ gan in the case of generator. This is RJ gan. So we compute the gradient. We pass the gradients from all the way from the output of the

**[25:47]** discriminator okay to the input of the generator. Okay. Please note that while we do this we keep update update theta only with w a constant. Okay. So while the

**[26:14]** gradients pass through the discriminator the discriminator parameters are not going to get updated while you are while you are updating the generator parameters. Okay. So let me repeat what is to be done is that why you want to compute the gradient. Okay. So let me complete that step. So after we do this how do we update? we update theta + 1 the new theta to be the old theta minus

**[26:43]** some other uh learning rate for step size we have this gradient with respect to theta for this particular function right okay so now let's just uh summarize so to get the to update the parameters of the generator we don't need the first term or we don't need the samples from the real data at all because the first term is independent of

**[27:12]** the generator parameters. So we take uh some B2 samples B2 number of samples from the normal distribution. Okay. And then pass do one forward pass pass it through the generator to get the generated samples from P theta. So these are as you know these are samples from P theta right? So we get samples from B theta,

**[27:45]** G, E2 and these are samples from P theta. Okay. So pass all those samples from P theta through the discriminator and compute this particular loss. Okay. This can be computed. Once we compute this, we compute the gradient of this particular loss with respect to theta which are the par the generator parameters. Okay, you note that uh this is actually a function of generator parameters as well, right? Because we

**[28:12]** are doing a forward pass through the generator parameters. So this this entire loss function is a par is as is a function of the generator parameter as well. So we compute the gradient of this loss with respect to generator parameters. pass the gradients through the discriminator by keeping W fixed and then continue passing the back propagating the gradients all the way through the input of the generator network and then here you just simply update theta here

**[28:41]** right update theta using gradient descent. So that completes one gradient step through the generator. So typically what is done is that uh uh the generator and discriminator trading are alternated. So you take one step through a generator and one step through the discriminator and keep alternating between it. Some of the practical tips include that know people don't use uh alternating same number of alternation I

**[29:10]** mean alternating uh training between the generator and discriminator but generator sometimes generator is trained more than the discriminator meaning you know you try you take five gradient steps through the generator and one through the discriminator and in some situations you take five gradient step through the discriminator and take one gradient step through the generator and so on. uh depends upon the uh the uh particular use case that you are looking at. So what is the stopping criteria here? Typically in uh the grand training there is no well definfined stopping

**[29:40]** criteria. The way it is done is that one would look at the quality of the generated data which is P theta using some metrics and depending upon whether those metrics are reaching the satisfactory level that is where the training is stopped. Okay, give you more details about uh how this training is uh stopped when we look at the classifier interpretation of this GAN in the next module. Okay, so let me

**[30:08]** just uh reiterate what we just did. So we looked at one instance of uh variation divergence variational divergence normization as a generative adversarial network, right? Again in practice. So what did we do? We had of course we had a push for network. We have a generator network and this discriminator network. Where did this come from? For all practical purposes, this discriminator network is nothing but our t function. Okay, which would construct a lower bound on the f divergence. So there is

**[30:37]** one f divergence that this is trying to minimize. Remember that why should this work? It should work because we are trying to minimize a divergence metric between p theta and px. Okay. So once we do that then uh sampling through g theta sampling using g theta is equivalent to sampling from px. So that is the whole idea that is still there. Okay. So how do we do that? We start off with an F divergence construct a lower bound using

**[31:05]** this f function and then minimize that lower bound. So have a f function construct a have a create a lower bound and then minimize the lower bound. For this gen in the special case of generative adversarial networks, you start with this particular f function and that would correspondingly give a conjugate and that would give uh an activation function also which would make our uh t function have the proper form that the domain of uh the t function the range of

**[31:34]** the t function has to be domain of f star. Okay. when we plug those when we plug those uh equations to the uh lower bound that we constructed it can be rewritten in terms of another function dw which is simply the sigmoid function on top of the b function that we had. Okay, in that case your uh setup looks like this that we have the usual generator network and the the t network is equivalency written as this

**[32:01]** discriminator network where we have the v function composed with a sigmoid at the output. The final GAN architectures look like this. And here is the loss function. And we know that these expectations are approximated using sample averages. So implementation wise, we'll do it using batch gradient descent where we take a batch of samples and we do we take one gradient step and we run through the entire data which would make it one complete epoch and the training is done for multiple epochs. I hope that

**[32:29]** uh the audience here is aware of what batch gradient descent is. How do we do that? We first keep the generator parameters fixed and uh compute the discrim solve the optimization problem for the discriminator using gradient descent. We just saw how to do that. So for the training the discriminator you need data both from the both from p theta and from the real data. We do that by sampling some samples from the normal distribution passing it through the

**[32:58]** generator by keeping the generator parameters fixed. So we get some samples from P theta pass those through discriminator and compute the second term in the loss and we take one batch of data from the real data and pass that through the discriminator and compute the second first term of the loss add them both compute the gradient with respect to W back propagate it through the discriminator and do one gradient step through the discriminator. So while doing all this the generator parameters are keep kept fixed and to train the generator network you do the same thing

**[33:27]** but by keeping the discriminator parameters fixed. Take a samples Z1 through ZB2 from the uh normal distribution and uh pass it through the generator to get samples from V theta. Generator training is independent of uh samples from px. This is also called the real data. Okay. So now take g theta z1 through g theta zb2 pass it through the discriminator. compute the uh the second term of the loss and take the gradient of that with respect to the theta

**[33:55]** parameters back propagate it through the discriminator all the way to the input of the generator network and while doing all this keep the discriminator the parameter of the discriminator network constant okay and then take one gradient step through the generator network. Okay, this is how the training is accomplished in the this would take us to the end of training the BDM variational divergence

**[34:27]** minimization or one special case of PNS. Okay. So in the next module we will look at how to do inference on this and we'll look at uh a classifier interpretation of this particular special case of GANs and we will look at some of the improvisations for GANs and how that can be used for different use cases. I told you that with different f functions you will get different sort of behavior imposed on this f divergence and therefore the generated data will have

**[34:55]** different kinds of behaviors. So we will look at more on how to uh improvise this to do different things and we'll also start with how to accomplish inference post training for this particular uh PDMS or GANs for conditional generation and other tasks. Okay. Uh thank you.
