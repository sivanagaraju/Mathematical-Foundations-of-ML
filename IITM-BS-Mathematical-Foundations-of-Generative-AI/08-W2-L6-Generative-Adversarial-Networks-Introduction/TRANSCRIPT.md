# Transcript — W2_L6: Generative adversarial networks: introduction

> **Source:** https://www.youtube.com/watch?v=EHhURRwMEPo  
> **Channel:** IIT Madras - B.S. Degree Programme  
> **Duration:** ~22 min  
> **Note:** Auto-captions cleaned lightly. Minor ASR errors possible.

---

**[00:12]** The other consideration that one needs to look at right now is how do we represent this uh t function. Now remember that uh t function by construction was a function that is defined from the space of data to domain of fstar. This means that depending upon the choice of uh f function in the f divergence we need to tweak the t

**[00:43]** network in such a way that the range of the t network uh corresponds to domain of f star. Okay. Now to do that uh this is how it is done in practice that the T network which we called as uh TWW of uh X is represented as a composite function of let's call this uh sigma which is a function of X f and

**[01:14]** another network uh BW we'll call it which is a function of X. So what is this? where sigma f is uh f divergence f divergence specific activation and vw of x is simply a

**[01:47]** function that would take x and map to real numbers. What is this? What does that mean? That means that the T network that we have to approximate or the T function that we will approximate using the neural network is represented as a composite of two composition of two functions where you have one network BW of X which will give you a real number. So that means

**[02:14]** that here is a neural network whose last layer is going to be a linear layer. Right? This is common across uh all the uh m divergences. And there is um another activation that is uh laid on top of it. Okay. which is specific to that particular um f divergence. So this is f divergence

**[02:46]** specific activation f divergence specific activation. So which means that this entire network now from starting from x to the uh the output of the activation now acts like uh the t function approximate the t function that will take x and this will map the x to domain of far. Okay. Now we did this as a composition of two

**[03:14]** functions. Okay. Where we first mapped x to r and from r to domain of far. Okay. from X to R is what is called as the VW network and from R to domain of FAR is what is called as this particular activation. So note that uh uh the specific activation depends on what is the F divergence that is used. Okay. So that means that your final

**[03:41]** objective function now can be expressed in terms of this. Now we have uh an expectation of uh recall that it was uh TWW of X. So now TWW of X will be the composition of sigma F operated on VW of X. Okay, this is what TWW of X was and this the expectation was with respect to PX. This is our uh first function of the loss.

**[04:11]** And the second uh component that we had was an expectation with respect to p theta and uh this thing was um far of f of we had the t uh of x here again so which is sigma f of vw of x right and all we did was represent our t function. Okay we'll just slash this. So what we had here was an expectation of

**[04:40]** TWW of X with respect to PX and an expectation of FAR of TWW of X with respect to P theta. Now this PW was uh TWW of X right because it takes X. Now was uh represented what was this T function? T function was a function that would start that would map the samples from the data space to the space of domain of SAR. So by construction this is uh dependent on the particular the choice of the f function that we do for

**[05:07]** f divergence. So just to make it general uh the way that was represented is that you represent this TWW of X as a composite function where you have one function that is common across all F divergences where it takes data and maps it to some real numbers using some linear layer right uh a neural network that would start from X and ends with a linear layer which is a projection onto a a real number the space of real numbers and from there you have uh an activation. Okay, this activation is

**[05:38]** specific to the particular F diverence. Okay, so then we rewrote the loss in terms of uh the composition that we wrote which is uh sigma f composed with vw of x which is a uh neural network with linear layer. Okay. Now we will look at one particular instance of uh a diverence minimization. Let us call that as uh uh the I mean it's popularly known as the

**[06:09]** generative adversarial networks adversarial networks. Generative adversarial networks or GANs in short. Now these are a special case of uh okay so pardon me there's a typo here. This is generative generative adversial networks

**[06:34]** also known as GANs. So GANs happen to be a special case of this general class of algorithms that we just saw as variational divergence minimization algorithms. Let us just see how that happens. So now as you know uh the first thing that we should start with is a choice for the f divergence. So for follows. So now choosing a particular

**[07:10]** fivergence right uh means that you are choosing a particular f function right so f of u that you would choose for the dan is u log u minus u + 1 log u + 1 okay u + 1 log u + 1 so now this is uh similar to geson Sand divergence but

**[07:39]** not quite exactly the the Jensen Sandon divergence. Okay, it's quite similar to Jensen Sandon divergence but with a constant but this is the f function that is used for uh the uh usual generative adversarial network. Okay. So once you use once you have the f function then you should also write down the conjugate for that. So f* of t for this particular uh uh u the f function convex function is minus log of 1 - e

**[08:11]** power e power t. Okay, this is the far uh function which is the conjugate uh corresponding to this particular u function. So now it's easy to see that uh domain of far is r minus which is uh set of all negative real numbers. Okay that is the domain of far. So now given what uh we just saw we'll have to ensure that we

**[08:41]** choose this uh activation function for the P function in such a way that uh the activation function projects uh uh neural network with a linear layer to negative real numbers. So that is the way we have to choose our uh activation function. So the activation function that is chosen for this particular case you know sigma of f sigma of f see note that the sigma will take a scalar right okay we should

**[09:12]** maybe I should also write it previously sigma is an f divergence specific activation so the sigma of f will take a sample v which is a real number so it will take a sample from real numbers and then map it to domain of star okay so note that the way we have constructed it the sigma function always states a scalar as an input v which is a scalar as an input and maps it to domain of far. Therefore in this particular case

**[09:42]** of uh the apparence that is used the sigma function would look something like this sigma of f is negative of log of 1 + e power minus v. As just discussed, note that uh this is a scalar valued function in a sense that it takes a scalar that is the output of

**[10:09]** the the v function and outputs another scalar which corresponds to domain of far. In this case it is r minus. So one can see that this is r minus because you have log of a positive number and you have minus here. So which means that this is always r minus. So collating all this what one should do is that now we have to write down this particular loss

**[10:37]** function. Let me just copy this. Write down this particular loss function with all of these values. Okay. So now we have everything that we need. We have the activation function. We have the fstar function here and uh we also have uh px and p theta uh the

**[11:06]** expectations that we can compute using the sample averages. So we have to collect all this and write down uh an expression for this particular function for the GAN. So let us do that now. J gan which is a function of theta and w is going to look like this. uh expectation of log

**[11:33]** of dw of x plus expectation of log of 1us dw of x it is expected the first expectation is with respect to px second expectation is with respect to P theta. So where where dw of

**[12:11]** x is given by 1 by 1 + e power minus vw of x. Okay. So people who are uh conversent with uh neural network literature knows that this is the usual sigmoid function. This is the usual sigmoid function. Sigmoid function. Now uh we

**[12:39]** introduce another uh variable called d here uh just to be consistent with the notation that is used in the uh GAN literature. Okay. So note that the original generative adversarial networks paper does not uh motivate uh the uh loss function that they derive in terms of minimizing the particular f divergence. Okay. Now this is uh a more generalized version of how

**[13:07]** to look at the GAN's activation function. So a GAN's objective function. So now there in that paper the objective function that was taken for a GAN was this one. Okay, J GAN and uh just to ensure that whatever we are doing is consistent with the notations in the GAN paper, we just introduce another uh variable here called DW where simply DW is 1 by 1 + e^ minus vw of x which is nothing but a sigmoid

**[13:37]** function. So finally so how did we actually get this? uh one can just easily verify. So I'll leave the algebra for the uh to the reader. So please do the algebra. What one should do is look at this activation for the look at this loss function that we have and simply have to substitute these equations here right for sigma f just substitute log of 1 + e power minus

**[14:06]** b and d is vw of x here right and for f substitute this negative of log of 1 - e power t this case t would be sigma f of vw of x okay if you sub substitute these two things and rearrange the terms By writing dw of x by 1 by 1 + e power minus vw of x we will get this loss function or the equation for the yans. Okay that's that's exactly

**[14:35]** what it is. So now let us write down how would the uh the implementation architecture wise would look like. We have uh as usual the generator function or the generator neural network which would take uh samples from uh an arbitrary distribution. In our case it is normal 01 and this would give us samples

**[15:03]** from p theta of x or xcap. Note that this x and xcap are dummy variables. So this is g theta of z. And we have the the discriminator network here. And the way we have composed the discriminator network is that we first have let me just write that down. This is a scalar. So okay

**[15:33]** it first we have the V network VW of X. Okay that would uh take X and gives you real numbers. Okay, this is our B network and on top of it we compose this with uh one sigmoid layer which is 1 by 1 + e

**[16:10]** power whatever it takes as an input. So this is what we called as v right. So it is 1x 1 + e power minus v. Maybe I will write it a little bigger so that it's easier to with another network.

**[16:37]** This is so we get uh r here as what we call as v and here what we do is 1 by 1 + e power minus v and finally we get the output between 0 and 1. This is by construction. Right? This is what we have. And this entire network is what we referred what we have referred to

**[17:12]** as the DW function here, right? Yeah. In the original GAN paper, uh they describe this uh DW network as the discriminator network where there's no breaking down of uh the the uh discriminator this way where we have a where we have one neural network that would give you the V function and then there is a uh sigmoidal activation on top of it. They have just combined these two into

**[17:40]** one neural network. Uh if you do that then it looks this way. Let me just do that. So we have architecture. The final GAN architectures looks like this. We can just perhaps copy this

**[18:23]** This is the generator network. We have a discriminator network here. So what I will do here is I'll simply combine both the V function and the activation and make it it as one neural network instead of writing it as a composite. So in this particular case what we have is a scalar as the output and the output that we get is

**[18:50]** between 0 and 1. This network is what we write as dw of x. Please note that uh the way we were doing was the critique network in our formulation had TWW as the output right where TWW was taking X and mapping it to domain of FAR right and what did we do we broke down TWW as a composite of some

**[19:21]** activation function operating on another function learnable function that would take data and map it to real numbers. Okay. Now in the case of uh generative adversarial networks, we had a particular activation function that would u approximate our t function that we need to construct the lower bound on the f divergence. Okay. However, when we plug that into the uh the lower bound and rearrange the terms, so we we got

**[19:53]** another formulation or rather another way of expressing it where the t function now is uh represented using this d function which is nothing but the sigmoid of the v function that we are looking at which is uh mapping data to the real numbers. Okay. So the reason I just mentioned that is the output of this discriminator network is between 0 and one. One should not

**[20:21]** confuse this with the domain of far. Okay. So this is not exactly the t function that we are optimizing for directly. But of course we are optimizing for the t function in an indirect fashion because uh the t function that we have is simply a constant or rather uh a deterministic uh composite composition of a function a deterministic function of the d function. So if you know the d function

**[20:49]** then you know the t function as well but their uh ranges happens to be two different things. Okay. So just to make sure that we are on the same page. Okay. So the final architecture would be that this would take uh uh x as input. Okay. And give you 0 and one. So what is the loss function that we have? So let me just copy that as well. Okay. I can write it again.

**[21:16]** So the loss function please note that this is nothing but the lower bone that we have computed on the corresponding f. So we have an expectation of log of dw of x. Now this these x are coming from px. We have another term which is expectation of log of 1 - dw of

**[21:46]** 1us dw of x where let me call it as xcap just to differentiate. So this xcap is coming from p theta right. So this is what we have. So now what we will do is uh we will see how to implement this particular thing uh in practice. Okay.
