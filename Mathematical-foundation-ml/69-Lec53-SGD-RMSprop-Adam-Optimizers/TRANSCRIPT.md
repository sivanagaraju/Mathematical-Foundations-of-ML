# Transcript — Lec 53 SGD, RMS Prop, ADAM : Optimizers

> **Source:** https://www.youtube.com/watch?v=N6F1J-wcE_E  
> **Channel:** NPTEL - Indian Institute of Science, Bengaluru  
> **Duration:** ~19 min  
> **Note:** Auto-captions cleaned lightly. Minor ASR errors possible.

---

**[00:01]** Welcome back. Training neural networks with adaptive learning rate. Okay? Now, neural networks are trained using ERM via gradient descent. Now, if uh

**[00:32]** suppose um R cap of theta is the empirical risk. estimates the parameters as follows. Now, you have uh theta t + 1 to be theta t minus alpha times the gradient of the empirical risk with

**[01:02]** respect to theta. Right? This is what the first-order gradient descent is all about. Uh there are versions to it, right? One version is what is called as the stochastic gradient descent. Where compute uh R tilde of theta

**[01:32]** as on a set of samples, right? Which is uh where where B is a subset of the data set D

**[01:59]** randomly sampled. You already saw that uh this has a regularization effect is what I told you. So, this is called stochastic gradient descent or SGD. Where if you have 1,000 samples you construct a batch of 32 samples or 64 samples, which again have a parameter. And then perform the gradient descent. And one full um walks through the entire training data

**[02:29]** is called an epoch during training. An iteration is where you do one forward pass and one backward pass through uh the neural network that you're training. If you do a stochastic gradient descent and uh pass through the entire data set once then it's called an epoch and training is done for several epochs. That's how the training is done. Okay? This is stochastic gradient descent. Uh but in in both in stochastic gradient descent and gradient descent, right? The

**[02:58]** step size, okay? Alpha is fixed. This is also called learning rate. This learning rate is fixed in stochastic gradient descent. You fix it to a particular value. Okay? And you keep it the same. But uh in optimization, we know that while we are looking at uh gradient descent or rather numerical algorithms

**[03:24]** we may have to take a larger step at times where the gradient has uh uh larger magnitude and you are near the uh the optima. You have to take a smaller step uh at other times and so on. So, basically, instead of having the fixed learning rate it's always a better idea to have adaptive learning rate, a learning rate that adapts. Okay? So, there are techniques that would incorporate this. There are a few adaptive learning rate techniques.

**[04:11]** we look at uh uh a few famous [snorts] uh such things. Uh the first thing is called um Where the idea is as follows. Define momentum MT as

**[04:40]** beta one times MT minus one plus one minus uh beta one times uh GT where so, GT is the gradient of the loss function. &gt;&gt; [cough] &gt;&gt; And now

**[05:07]** you make theta t + 1 as theta t minus some alpha times Yeah? What is this doing? This MT is a running average of the gradients. Moving average of the gradients.

**[05:35]** At t equal to one, it is simply the gradient. Okay? At I mean, you initialize MT M0 to zero, so it's equal to the gradient. As you uh keep moving in time uh the M the the M term will keep accumulating the gradients and taking an average of it as we move on. Okay? Now instead of subtracting the the the gradients, you subtract the

**[06:03]** accumulated gradients over time. Why is this a good idea? You're computing the exponential decaying average of gradients. Which means in earlier times you take larger steps depending upon the gradients and the later steps you take smaller steps if the the gradient because the gradients are being accumulated over time.

**[06:32]** So, this is called SGD with momentum and this term MT is called the momentum term. Okay? &gt;&gt; [snorts] &gt;&gt; The See here what did we do is that we accumulated the gradients. But we know that in optimization uh the first-order gradients are not enough, right? You need to accumulate the second-order terms of the gradients as well. Why? Gives you the acceleration effect.

**[07:03]** Right? So, instead of only looking at the first-order gradients or rather uh the gradients, look at the gradient squared terms as well. Okay? So, this is called another adaptive learning rate method, which is called RMSprop. So, here instead of averaging the gradient, you average the second-order or rather the square of the gradient. Define VT to be some beta two

**[07:32]** times VT minus one plus one minus beta two times GT squared. We are accumulating the square of gradients here. Now, you make theta t + 1 to be theta t minus some alpha divided by root of VT

**[08:00]** into GT. So, why do we need this uh root of VT? VT the dimensions of the VT and not the dimensions are dimensions. It's the physical dimensions of GT is square of gradients. Right? So, root of VT has the dimensions as that of gradients. And GT is gradients. They both cancel and the one that this will be of dimensions of the parameters

**[08:28]** and that's why you can add and subtract them. The dimensionality has to match, right? Yeah? So, that's why this is the uh the RMSprop where what we are doing is that instead of using the first-order gradients or rather the the gradients, we're using the gradient squared terms. Okay? Why do we have to do this? See, we are scaling the gradients. All gradients are now being scaled with a particular value.

**[08:55]** This particular value is inverse of their squared. So, the gradients, okay? Are the parameters for which the gradients are damping uh higher the gradients, they dampen quicker. The lower the gradient values, they they dampen slower. So, here what we did was we dampened all the gradients no matter what their the so-called acceleration is. Here they are being scaled by

**[09:26]** the inverse of the uh damping factor or the acceleration. That's it. Okay? This is about RMSprop. Uh the other update to this, improvisation of this is what is called as adaptive momentum estimation. &gt;&gt; Also known as

**[09:59]** Adam. And by the way, even though I've written it this way, right? If theta are vectors, these are done at the element level. The parameter level. I mean, these things are actually the Hadamard products. Because when you take the square root of gradient, you're you do it dimension wise. Each of the dimension has to be multiplied. So, you please note that if you do this,

**[10:26]** then there there is a different learning rate for different parameter also. You understand? There is adaptability across different parameters. You know, some parameters are are updated faster or larger and some parameters are adapt are are are are changed slower. Because this is happening at the parameter level. Because these are vectors, no? We are looking at Hadamard

**[10:53]** products here. You see that? &gt;&gt; [snorts] &gt;&gt; Okay? So, this adaptive momentum estimation, which is referred to as Adam, actually combines both. It's a combination of both the RMSprop and the SGD with momentum. There we have both the terms. So, compute momentum as beta one times empty minus one plus one minus beta one times GT

**[11:20]** and beta two times VT minus one plus one minus beta two times GT squared. &gt;&gt; And the final estimate theta T plus one is theta T minus alpha divided by root of VT times the momentum.

**[11:49]** So, this is a combination of both the RMSprop and uh the momentum term. You know, you have both the momentum and the second order term. This is called Adam. Now, if you go to the standard optimizers such as PyTorch etc., you have the uh the facility to choose your optimizer. You can choose SGD, you can choose RMSprop, you can choose SGD with momentum or Adam. Adam is the go-to choice for optimizers in today's

**[12:18]** learning because empirically it has been observed to be uh leading to good better convergence. Now, you have to choose this beta one and beta two. They are hyper parameters. Okay? Adaptive momentum estimation. Uh Now that we are talking about optimizing neural networks, there is one point that I would I would make, &gt;&gt; [laughter] &gt;&gt; which is which is better to be known. See, we are using gradient descent here.

**[12:46]** Okay? To optimize over the parameters parameter space, that would be typically a very large size, right? I mean, today they are of billions of parameters. You know, they can be thousands of So, we are optimizing in a very large dimensional space. And the objective function that we are that is being optimized is not a convex function of these parameters. The empirical risk that we are

**[13:13]** optimizing is no convex function. Okay? So, first order gradient descent does not guarantee that that you will go to the true optima. In In fact, it can be is that the the the likelihood of these methods taking you to the true minimizer is very very low. And it can be shown mathematically. Let me just tell

**[13:40]** you that. So, let's say that we are operating in a thousand dimensional space. Our neural network has thousand per thousand is nothing. You know, a simple uh ant level MLP will also have thousand para more than thousand parameters. Let's say that you have a neural network that has thousand parameters. Now, suppose uh there is an optimization surface. When do you know or how do you find out

**[14:08]** whether a given point on this optimization surface is a point of extrema or not? Recall your high school or uh college math. How gradient has to be zero, right? That will only tell you that the gradient vanishes at that point. How do you know that if it's a maxima or a minima? Oh, second derivative test. You'll have to always look at the second derivative test. Now, if you are working with vector valued functions, what is the equivalent

**[14:36]** of the second derivative test? Not the Hessian. Hessian is the equivalent to second derivative, but if a if a point has to be a global optima, then the signs of the eigen values of the Hessian matrix are all all have to be of the same sign. If all of them are positive or in In other words, if the Hessian matrix is positive definite, then that particular point is a minima or maxima? Yeah, whatever, right? It It has to be

**[15:06]** It has to be all the eigen values have to be of the exact same sign. Now, let's say that we are operating in a thousand dimensional space. What is the size of the Hessian? Which is 10 raised to six, a million. We have to We are looking at a million uh eigen values. We're looking at million eigen values. Let's say that we model the sign of the eigen value as a Bernoulli random variable

**[15:33]** with success probability P. What am I saying? With probability P, the sign of the eigen value of the Hessian matrix is positive. With probability one minus P, the success probability is uh that the probability that the sign of the eigen value of the Hessian matrix is negative is one minus P. So, now take this P to be 0.99999. So, with a very high probability, the eigen value of the Hessian matrix is

**[16:03]** going to be positive. Now, what is 0.99 to the power of 10 to the six? Now, 0.99 to the power of 10 to six is a very very small number. Right? Which means that the odds of any point that you land on to on the loss surface in a thousand dimensional space being a global optima is extremely low. This by This with an assumption that the

**[16:34]** probability of it taking a positive value is 0.99. So, what am I trying to say? So, now we are looking at not a thousand parameter neural networks. We are looking at millions and billions of parameters of neural networks. So, almost always, gradient descent will not take you to the point that is a global optima. Then what is the hope? The hope is look at your validation data.

**[17:02]** The only question that we are all the points that we land uh to with gradient descent are local optimas. The only question that is to be asked is is your local optima better than my local optima? &gt;&gt; [clears throat] &gt;&gt; And how do we measure this? By looking at validation data. That's why when you train neural networks, there is no theoretical way to look at what the stopping criteria is. Okay? Just you have That is why people

**[17:30]** look at come up with benchmarks. You have test data. Whatever neural network you train, perform its rather extract its performance on test data and that's that's your uh truth. That's it. Okay? In fact, there is one other nomenclature that is used. When you are doing gradient descent, right? Every single point of gradient descent will give you a different model. Because you're navigating in the theta space, correct?

**[17:58]** And these things are called checkpoints, by the way. So, what is shipped are checkpoints, model checkpoints. A model checkpoint is nothing but a particular set of theta for this architecture. So, when you download a model, a pre-trained model from some of these uh commercially available libraries, Hugging Face for instance, what they actually give you are these theta's with an ERM done on a particular task. Start from it, either do inference on it or you can initialize another neural

**[18:25]** network with that and you know, you do another ERM on top of it or do distillation or do whatever you want. Is okay? So, I think now we are in a position to take up a data set, take any neural network and train it to for whatever task you want. This is the foundational principle of training any uh you know, neural network for any task, including LLMs. And today's LLMs are trained on thousands of

**[18:53]** Not thousands. Hundreds of thousands of uh GPU cards for several tens of months on trillions of tokens of data. That's the scale. With billions of parameters, exactly. Right? That is what it is. Okay, so this completes our discussion on neural networks. We will next go to the classification and regression trees, decision trees, and subsequently we look at ensemble methods.

**[19:22]** Thank you.
