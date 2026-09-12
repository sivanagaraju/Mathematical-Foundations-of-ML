# Transcript — Tutorial 14 : Wasserstein GAN (WGAN) Implementation using Gradient Penalty

> **Source:** https://www.youtube.com/watch?v=Mxad7Wz7ymg  
> **Channel:** NPTEL - Indian Institute of Science, Bengaluru  
> **Duration:** ~20 min  
> **Note:** Auto-captions cleaned lightly. Minor ASR errors possible.

---

**[00:01]** Hello guys, welcome to this tutorial in which we'll be seeing the implementation of WGAN using gradient penalty. See earlier, we discussed the in the previous tutorial, we discussed the idea of WGAN using weight clipping. Now the reason why we wanted is we wanted the parameters of the critique network which are which we represent using W. Now they should satisfy one lip sheet constraint. Okay. So because of which we restricted

**[00:32]** or we clipped the value of these parameters now between minus .01 to plus .01. Okay. So now the weight clipping idea is simple. Okay. So let's let's see why do we need the gradient penalty. See the

**[00:58]** the weight clipping that we have used is simple but not ideal. Now why do I say it is not ideal for these specific reasons. Okay. So now what will happen is if we clip the values to be extremely small. Okay. So if we have the clip value to be small

**[01:34]** if you have small clip values so the critique which is there now now then what will happen is if we are clipping it for very small value, the values that the critique network can take are very much restrictive. Okay. The parameters that you can get is very much restrictive. Now because of which it might not be optimal. The critique will be restricted. Okay. See what do I mean by critique will be restricted? The parameters, the values that each of the parameters of the

**[02:02]** critique network that can take now will be between very small values now because of which we might not be able to get the full-fledged uh scores. So, now then what will happen is now because of which it leads to or of loose capacity. So, now if you remember

**[02:32]** our earlier implementation of WGAN that we did in our previous tutorials, the generations were pretty bad. Okay. Now, I urge the viewers of this video to go back to that code and see what is the value of uh clip value that we have used, modify it, and see whether we get better generations. Now, was it because of the uh optimizer that we have used? Is it

**[03:01]** because of the learning rate? Is it because of the capacity of the network or is it because of the clip value that we have used? Okay. Now, they can go ahead and check it. So, now on the other hand, if the clip value is large, so if we have uh the large clip value, now then what will become is your Lipschitz constraint will become weak. Okay. We will not be able to satisfy the Now, therefore,

**[03:46]** going ahead with the weight clipping is not a good idea. Therefore, we want some other way of doing it. Okay? So, therefore, we want to satisfy the Lipschitz constraint with some other method. Okay? Therefore, we have what is known as the we introduce the idea of the Okay? We introduce the idea of gradient

**[04:19]** penalty. So, what is this idea of gradient penalty? Now, you take the gradient with respect to X. Okay? The W of X norm of it. Now, this has to be approximately to be equal to one. That's what now we want. So, now, what is this X that we have? So, the way in which we do it is

**[04:58]** between the real and for the points between the real and generated samples now, we'll be computing this. Okay? Now, how do we implement this? I'll I'll just show it to you in a moment. Okay? So, now, why is this going to work? The whole idea is for an optimal critique value under one Lipschitz constraint, now, what will happen is the

**[05:26]** gradient norm which is there, now, which actually is what you're computing, the gradient norm, now, which is there, now, will be tending towards one for the optimal transport plan. Okay? Which we actually want. Now, if you have an optimal transport plan, the gradient, now, which is there, of these things, now, will be uh tending towards the one. Now, therefore, if we satisfy this constraint, it is same as satisfying the

**[05:54]** one Lipschitz constraint. Fine. So, now And again, uh these are for uh the functions. So, these are for the differentiable one Lipschitz functions, which we have in our case. So, now, now how do we implement this? Now, the implementation of this is quite simple. Now, we introduce now what is known as uh the loss for gradient penalty. See, the loss for gradient penalty, now

**[06:22]** what is this? We have a term, we have a regularization term. Now, expectation, now we have this of Okay. So, it has to be approximately one. Now, therefore, you're using this This is kind of an MSE loss. Now, the expectation of uh

**[06:50]** the MSE loss is what you're taking. Now, this has to be uh minimized. Okay. So, that uh will be taken into consideration. Now, what is this x hat? This x hat will be some epsilon value of some x real plus 1 minus epsilon of x fake. And epsilon which is there is sampled uniform from zero one. Now, this is a

**[07:17]** linear combination of real images and the fake images. That's what you're doing. Now, as simple as that. Okay. Now, this is what uh we said in between the generated and the real images, now we'll be taking the value. Now, this x hat, now will be used for this particular computation. Now, what will happen this is we want to minimize this whole thing. Okay. Now, since I I to minimize this, now this TW, this this gradient, you know, which is there, now has to be as

**[07:46]** close to one as possible for this minimization. Now, therefore, now what will happen for the critique objective? So, already we know that what is the critique objective? It is P theta of TW of X hat minus expectation of PX TW of X. We have plus since we are minimizing, so plus

**[08:13]** uh some uh uh regularized value into this also GP. So, normally, the value that this is taken, we take it as 10. So, this will be the critique objective. Now, that means that you have to compute this additional term and add it to the objective. Now, what is this uh taking into consideration? So, this will see to it that the weights of the critique network uh satisfy one Lipschitz constraint. So, now will there be any modification in

**[08:43]** the constraint of the generator? No. Okay. Now, we are only worried about the values that are the the the values that the parameters of the critique network takes. Now, therefore, the generator objective So, the the generator objective remains same. Okay. The only modification is we have to go ahead and come up with

**[09:12]** the way of computing this. Now, once we have this, we can add it and then see how does the operation happens. Okay. So, now let's look at the implementation of the same. So, now we'll be computing this gradient penalty. So, what do I need is the critique network. So, you want the real images. Now, you will be the wanting the fake images. Why do I want the critique network? No, because see, in the objective uh So, if you want to get this, see you can see it here.

**[09:41]** So, this is the critique network. You want to take the gradient with respect to that. Therefore, I want the critique network. And then, why do I need the real images and the fake images? So, you need the real images, you need the fake images. You'll be finding the linear combination of them and working with it. Okay. So, this is how you compute the gradient penalty. So, now let's see whether the same implementation is being done here. Now, first you get the batch size, fine. So, then you sample uh

**[10:07]** uniform from uniform distribution, which is of the batch size. See, when I mean uniform distribution, you'll be having a vector uh when which each element is sampled from uniform distribution. Okay. So, then you get this X hat, which is epsilon into real plus one minus epsilon into fake. Okay, this is the interpolated images. This is your X hat that we are considering. And then, we need the gradient uh for this. So, therefore, we make

**[10:36]** require gradients is equal to true. Okay. So, now then we pass this through the critique network. Okay, that's what we want. So, so this you can see it here. We have obtained uh this particular X hat by the interpolation. So, we pass this through the critique network. Okay. So, and then, since we have enabled the gradient, I can use dot grad to get this particular gradient value.

**[11:04]** Okay. So, now you get this. Then, we need to get the gradients. So, torch.autograd.grad, so these are the outputs. Now, the input, so we should be getting the gradients with respect to X. Now, please remember, therefore the input is X. So, you get the graph, and then you get the gradients using this. Okay. See, we are you're not getting the gradients with respect to the parameters of the critique network, You are getting

**[11:33]** the gradients with respect to the given input, the interpolated uh uh images that we have generated. Okay? And then, you flatten those gradients for the sake of updation. You take the norm of them. You take the two norm. Okay? That's what we want. So, please remember this. You take the two norm of it. You can see it here. So, we got each and everything that is there here. So, now, then you compute the penalty,

**[12:03]** gradient norm minus one. So, this is the mean. Now, you get the penalty. Okay? So, now you have computed this LGP. Now, now what are the uh WGAN parameters? Now, I'll be using uh the lambda for GP as 10. Number of critique, now I'm keeping it as five in this case. So, learning rate, I'm keeping it as uh 10 to the power of minus four, and I'm using Adam

**[12:30]** in this case. So, then I'm creating all these generator and then the critic network. And then, I'm initializing them. So, with the same old initializer that we have used earlier. And then, I'm getting the optimizers done. Now, then train the critic network. So, you remember that we have modified the critic objective as of now. So, now first, you get the batch size.

**[12:58]** You get the real images. Then, you get the Z. Then, you get the fake images. You detach the fake images. Exactly the same procedure. You get the real score. You get the fake score. And then, you get the gradient penalty. Now, please remember the term. So, this is the the first term now which we have. Now, this is the the fake score. This is the the real score. And then, we have already know how to compute the gradient penalty. Now, I can

**[13:26]** use it. I can get the penalty using the function that I just wrote. So, I got the gradient penalty. Now, therefore, what is the loss? The loss will be fake scores minus real score plus lambda into this the gradient penalty that we have computed. This is the loss. This is the critic loss. And then, uh, you go ahead and clear off all the previous gradients. loss.backwards, optimizer. uh uh, step. Okay? Fine. So, then, what is the

**[13:53]** generator step? The generator step remains same as we used in our, earlier discussion. Now, the Now, the whole training Now, we'll remains exactly the same. The only thing is, now, in addition to that, we have this, uh, gradient penalty, now, which we should be considering. And then, we should be considering it for the bookmarking. Apart from that, the overall training procedure remains the same. And then, we'll be updating the generator values only for five times once.

**[14:22]** Okay? Fine. So, and the remaining things are bookmarking, I'm just, uh, leaving all of them. Huh, and then, huh, so, this is a right time to point this out. I should have pointed it out in the earlier tutorial itself. But anyhow, that's fine. See, you can see the loss value, sometimes it is getting off to be negative. It's perfectly fine. Why? See, the output of the critic network is something that belongs to R. Okay? So, So, just because we are getting a negative, uh, loss here, there

**[14:50]** is no need to have some kind of an alarm. It's perfectly fine. Okay? So, the output of the, uh, the critic network can be any real value. Okay? It's perfectly fine. So, therefore, since because it's negative, so, we don't need to uh, worry. But, the gradient penalty has to be positive because it's an expectation of a positive random variable. Okay? See, expectation of positive random variable is always positive. So, we should be knowing it. So, this is a

**[15:17]** positive random variable. Okay? Why is this a positive random variable? The reason is you have the square of it. Okay, the square cannot be negative. Uh so, it is non-negative. So, therefore the expectation of a non-negative has to be non-negative. Okay? So, therefore it satisfies. So, your gradient penalty has to be a non-negative value. Fine. So, now then you generate the samples. Generating samples, you don't need the critic network, you can just ignore it.

**[15:45]** Now, you can see some better generations. Okay? Now, compared to what we generated in uh WGAN with clipping, now at least these generations look better. We have trained it for very small amount of time. So, because of the GPU capacities, now we you can go ahead and train it for longer. So, we can have better generations. And this is how we go ahead and get these

**[16:14]** gradient penalty and work with this WGAN using gradient penalty. See, the idea is quite simple. Now, here the only idea that uh we are using is this. Now, rather than going ahead and performing the weight clipping, now which is not a good idea. This whole reason is Why is it not a good idea? Because it can make the critic network to be very restrictive. Now, we don't want that to happen. So, we want our critic network

**[16:41]** to be as good as possible. Now, therefore rather than going ahead and making the critic network as uh restrictive, now we can use other forms of ensuring that the Lipschitz constraint is taken into consideration, and gradient penalty is one such way. Now, the whole idea is modify the critic uh uh network in such a way that the parameters of the critic network, the gradient with respect to the interpolated images is approximate to

**[17:09]** one. Okay? So, how do you enforce it? You just enforce a loss. Uh this is nothing but a some kind of an MSE loss. Not some kind of it is an loss. Not some kind of a reconstruction loss in terms of gradients. So you just take care of it and then add it to the uh the loss function of the critique and then train it. We obtain better images. Now I urge you to modify couple of things even in this. Change the number

**[17:38]** of times you are updating the critique and other things and generate better images. Okay, I leave that as a small exercise. You have access to the code. You have access to the collab notebook. The link of the collab notebook will be given in the description of this video. So you can look at it. You can modify it and then obtain better images and then we'll be really happy if you put those images in the comments sections. So using the method that has been used to get better generations. So it will be

**[18:06]** reinforcing to us that the the content is understood by the viewers of this video. So thank you all. This week conclude the section of the tutorials where in which we have implemented WGAN. See there are a lot more GAN architectures which are there up. So you have CycleGAN, we have bidirectional GAN, so on and so forth. The underlying idea remains the same. Okay. So it is just for different purposes you'll be using different

**[18:34]** things. So I urge all of you to look around the different literature of GAN. So with this much of background and this much of comfortable implementation, now you should be able to take up any GAN paper and then you should be able to implement GANs. See the overall idea is please don't restrict yourself for looking at images. See we have never during our theory sessions, we have

**[19:01]** never considered the modality of the data. The modality of the data is depends on the application that we are using. Now if If want to generate, for example, let's consider an application wherein which we want to generate some embeddings. Okay, you have some kind of data, but some some kind of vector data. Now, but what has happened is they are very small in number, you want a large number of samples. Now, in that case, you can train a GAN

**[19:30]** on those smaller numbers, and then once you have trained it, you can leave away the discriminator, or if you're using Wasserstein's thing, you can leave away the the critic network, and then you can obtain any number of samples from that specific GAN network. So, irrespective of the modality for speech, for some kind of embeddings, now you can use these kinds of GAN structures. So,

**[19:57]** please don't restrict yourself that this GAN which is there is specific for the images. It is generic enough to work for any kind of data data modality, but remember one thing, now whenever you're creating networks, you're working with some kind of LEGO blocks. So, the only thing that you should be taking into consideration is what is the shape of the input, and how are you transforming it. If you have taken them into consideration, it works perfectly fine irrespective of the modality of the

**[20:24]** data. I hope these GAN tutorials were helpful for you, so you are able to implement them. In the next tutorials, we will start implementing variational autoencoders. Now, with this, we rest these tutorials. Thank you.
