# Transcript — W1_T3: Tutorial 3: Introduction to pytorch: datasets & dataloaders

> **Source:** https://www.youtube.com/watch?v=c2gN3TK3U74  
> **Channel:** IIT Madras - B.S. Degree Programme  
> **Duration:** ~30 min  
> **Note:** Auto-captions cleaned lightly. Minor ASR errors possible.

---

**[00:13]** Welcome everyone. We continue with uh our sessions in this course on deep generative models. In this particular session, we will continue from the previous session where we were looking at uh building generative models or rather generative samplers from the tech using the technique of variational divergence minimization. In this session, we will continue from

**[00:42]** the uh previous session where we constructed a bound on the divergence metric that we would want to optimize and then we will see how to use the bound that we have constructed in building the sampler that we wanted to build. Let me quickly recap uh whatever we had done the previous session before we continue. So the idea was that uh we are given data which

**[01:15]** are uh samples drawn uh iid from an unknown distribution. So for the sake of completeness I'll write that all of these uh xis all the data points xis are in some dimensional space and I also denote this as uh x script x. Now we have given n data points uh sampled according to an unknown distribution from a dimensional real space. The goal is to sample from

**[01:45]** the unknown underlying distribution uh build a sampler for the underlying distribution. Now the uh approach that we took was that we start by sampling from an arbitrary known distribution like a unitn normal uh gausian distribution. So we know how to sample from this distribution and we take those samples and pass it through a deterministic function a learnable

**[02:14]** deterministic function which is a neuronet network and uh we want the output samples from this neural network to follow the distribution of interest px. Okay. Now initially we say that uh this particular sample with the output of the neural network uh is denoted by xcap and we say that it has a distribution p theta and obviously this p theta depends upon the parameters of

**[02:43]** this that neural network theta. Then we set out our problem as minimizing or as finding out the parameters of this deterministic function g theta such that some distributional divergence metric between px and p theta uh is minimized. Now uh if we get if we define a good distributional divergence metric that would uh be minimized if and only if these two distributions match then we

**[03:13]** are we have uh reached our objective because if we find the parameters of this deterministic function theta in such a way that uh a a distribution divergence is minimized between them then the the distribution that the output of this neural network follows will be same as that of px and we got a sampler for px which is what we wanted. Okay. Okay. Now uh the next step was

**[03:45]** that we wanted to get an algorithm to minimize uh the divergence metric between px and p theta without knowing the samples of uh without knowing both of the distributions but only having samples from it. So you can draw samples from the distributions but we don't have the distributions themselves right. So how do we get samples from px? It is by the data set. The data set the given data set itself is the sample or data set itself will contain samples from px

**[04:14]** and samples from p theta obtained as the output of this neural network. Okay. For the divergence metric uh we defined a class of a large class of divergence measures called the f divergences right where given a particular uh convex function okay f we have one divergence distribution divergence uh metric uh which is called

**[04:42]** the f divergence. So one can obtain multiple uh distributional divergence matrix by plugging in different uh uh choices for this convex function f and we also saw saw this property of the distribution diver f divergence that it is always a non- negative measure and it is zero if and only if the underlying distributions match. Okay. Now the uh key challenge was that uh how do we solve this optimization

**[05:10]** problem when we do not have access to px and p theta. Okay. So we wanted to get an algorithm that would minimize the f divergence between px and p theta without knowing either of the distributions but only having samples from them. Okay. Now uh the key idea was that integrals involving density functions can be approximated using samples drawn from the distributions. So that is the um key idea that was uh there. Now why

**[05:45]** is this possible? We just saw a classical result from statistics demonstrates that integrals of this form which is uh integrals of functions some functions h of x with respect to uh probability density functions v of x can be obtained or can be approximated using uh the samples obtained from the distribution and this result is kn known as the law of large numbers. Right? So now the key

**[06:13]** idea is that if the f divergence which would uh involve integrals over uh px and p theta can be expressed in terms of expectations of some function with respect to px and p theta then perhaps one can compute the uh f divergence and optimize them over the parameters. That is the key idea. Now then we set out to express f divergence in terms of expectations over px and p

**[06:42]** theta. Uh the way we did that was we first express uh the f function in the f divergence in terms of what is known as its conjugate. So this conjugate is also known as the fential conjugate of a convex function. Uh the definition of convex uh the convex conjugate of a given function was uh given convex function was given and uh the uh the f function that appears in the f

**[07:12]** divergence was expressed uh in terms of the convex conjugate. Okay. And after doing some algebra we just saw that finally we get uh a bound on the f divergence. Okay, we'll not get uh the exact representation of the f divergence in terms of expectations but we get a bound on the f divergence in terms of the integrals that we can compute. So

**[07:42]** finally the f divergence that uh we set out to optimize uh is represented in terms of bound lower bound on the uh lower bound involving the expectations over px and p theta which we know how to compute. So that is where we were in the previous session. So let us continue from there. Now given so what we look at today is

**[08:13]** that realization of variational divergence minimization is what we look at today. So now we know what variational divergence minimization is. uh we'll see how to rearrange that in practice. Okay. Now let's say uh as usual we are uh given uh

**[08:43]** data given data which is a set of samples x1 x2 up to xn and these are drawn iid from px. Okay, this is what we know and please remember the setup that we have uh a neural network. We have a neural network that

**[09:15]** takes uh samples from uh an arbitrary distribution and gives out samples from distribution called P theta. And uh our goal is to minimize the f divergence between

**[09:40]** px and p theta. And this we want to do with respect to the parameters of this neural network g theta. Okay, this was the goal and uh what we did in the previous session was that we showed that the f divergence between uh px and p theta px and uh p theta is bounded by the

**[10:14]** uh supreum. Okay, for the sake of uh ease of understanding I will replace supreum with a maximum. Okay, assuming that uh the supreum can be reached. So now supreum was over a class of functions from a space of uh functions uh denoted by script t. So what we wanted was that we wanted to maximize this expression over the space of functions or these functions

**[10:44]** that we are talking uh the objective function that we are saying is expectation over T of X and this expectation is with respect to PX minus the expectation over uh FAR of TX. So remember what fstar is right we start out with uh an f divergence here right and this fstar is the conjugate for this particular f divergence okay that is what it is and this second expectation is with respect to p theta

**[11:12]** so this is what we uh found out the previous time now what we want is you know we want a theta star right that uh would minimize that would uh minimize the f divergence Okay, that would minimize the F divergence. Now, um because since the F divergence cannot be minimized as you I mean as it is what we did was uh we

**[11:40]** express the F divergence in terms of the expectation that can be computed. Now what we have here is not an exact optimization of F divergence but an optimization of a lower bound on the F divergence. So this is equivalent to minimizing the lower bound that we constructed lower bound on f divergence.

**[12:07]** Okay. Now note that uh these two optimization uh problems are not equivalent because in the above in the above optimization problem what we are seeking is the minima of a particular uh quantity. Now in the second line what we are seeking is not the minimum of this quantity but the minimum of a lower bound of that particular quantity. Okay. Now minimizing lower bound does not mean that you're minimizing the actual

**[12:36]** function itself. But this is a limitation of the method that all that we can do is construct a lower bound and try to minimize the lower bound instead of minimizing the function itself. Okay. The function here is the f divergence. So we cannot do anything with fence because it involves integrals that are not tractable and involves uh density functions that we don't know. What we can do definitely is uh construct a lower bound on this f divergence and optimize the lower bound instead. Okay.

**[13:05]** So that's the best that we can do and that is what we are going to do now. Okay. So now this is equal to uh the minima or theta and this lower bound that we constructed okay involves another optimization or t and what is this lower bound? It's uh uh the expectation of uh t of x minus the expectation of uh far of t ofx or p

**[13:39]** theta. So this is what we have to optimize. This is what we are going to optimize now. So note that. So how did we get this? We wanted the theta star which are the parameters of this network uh of this function such that the f divergence between px and p theta are minimized. Right? Now the f divergence

**[14:11]** cannot be minimized cannot be computed because of uh the difficulties that we just stated. Therefore what we do is we construct a lower bound on the f divergence and we would set we set ourselves to minimize the lower bound instead. Now the lower bound that we constructed on the f divergent f divergence itself involves an optimization over uh some other function t of x. Okay. Now the final objective okay

**[14:40]** involves two optimization problems okay which is uh uh the minimization over some set of parameters theta and maximizations over maximization over another class of functions d ofx of the same objective function. Okay. So now this is uh the outer optimization is with respect to the

**[15:09]** parameters with respect to the parameters of the G function. Okay. And the inner optimization is with respect to a class of functions. a class of functions T of X okay which uh belong to a large family of functions right this is what it is now optimizing over uh

**[15:40]** functions space of functions uh cannot be done analytically because these are we are talking about uh functions that are uh that are not simple here what is done in practice is the following that we represent represent the set of functions that we are optimizing from which is script t via neural networks

**[16:15]** again neural networks so let's call them uh t w of x where where w are the parameters of that particular neural network are the parameters of the neural network. Okay. Now what we did was we

**[16:43]** parameterized this family of functions T using neural networks. Okay. because we know that neural networks are powerful enough and they are universal function approximators. So we just expressed we just express uh t also as neural networks. Okay. Now what happens to the uh above optimization problem with this with

**[17:12]** this the op the following. So we have we have to optimize over two set of parameters theta and w. So we need uh to minimize with respect to theta and we need to

**[17:41]** maximize with respect to w the same objective function. So which is the expectation of uh t w. So note why we have a subscript w here because uh we have represented the t function using neural networks which are parameterized by some parameters w minus the expectation of f* of tw of x and this expectation is with

**[18:13]** respect to pta right so this is our final objective that we would want to optimize that we have uh two neural networks One is the theta uh one is parameterized by theta the other is parameterized by w. So we have two neural networks. Now what is the final setup? Let me write that down. So di diagrammatically this is how it looks like. we have implementing implementing the

**[18:50]** variational divergence minimization for but for uh generative sampling generative modeling I can say. Okay. So this is how you implement finally this variational divergence minimization for generative sampling. So we have a neural network as usual which is the actual

**[19:25]** sampler. So this takes uh this is this is the neural network that converts that uh converts a normal distribution or pushes forward a normal distribution into the distribution of interest. And we have been calling this as the G network right and this is uh what we get here is XCAB which we which are samples from the distribution P theta of Xcap. Okay. Now this neural network has parameters theta. Now we have another neural network here

**[20:09]** function. So this neural network is representing the t function tw of x. Okay. This will take X as input and gives you TWW as uh the output. Right? I'll call this as TWW of X this network. Okay? Now we have two set of parameters that we would want to optimize. So our uh uh cost okay or the loss function

**[20:39]** that we would optimize would be uh let me just uh copy this so that I don't have to rewrite. So please uh remember what is that we are doing here. what we are doing is this is the uh f divergence that we are minimizing with respect to theta. So we have this

**[21:08]** particular cost function and what we want to do is that we need to minimize this uh cost function with respect uh theta that would minimize this and we want a w that would maximize this. Okay, this is what we we have. So we

**[21:38]** have two neural networks. Okay, one representing the sampler, the other representing the uh the t function. And please note what our t function is. You know, we get our t function uh when we construct the lower bound on the f divergence. The idea was to minimize the f divergence. We could not do it. So we constructed a lower bound on the f divergence. And while we are we were constructing the lower bound on the f divergence we landed up having another function t of x uh another class of function t of x and uh we had to solve

**[22:07]** the lower bound itself involved an optimization problem. So to obtain obtain the lower bound we had to maximize over class of functions t of x and of course once we get the lower bound on the f divergence the idea is to minimize the f divergence or the lower bound on it. Okay. So finally what we did was we represented this class of functions T using another set of neural networks. Now finally it boiled down that what we wanted to do is to optimize this particular objective function that

**[22:37]** we have uh and we have two optimization problems. one we need to find the parameters of t uh such that the optimization problem is maximized and once we do that we'll have to use the same objective function and we'll have to minimize uh with respect to the generative parameters theta okay so this sort of problems where uh you have uh uh uh an alternative minimization and maximization problem over the same

**[23:05]** objective uh a given same objective function these are called the saddle point optimization because uh what we are actually seeking is a saddle point. So what is a saddle point? Suppose you take a function uh a little difficult to draw that. Let

**[23:34]** me try that. Okay. Suppose you have uh a function of two set of parameters uh one is theta the other is w. So please note that these are vector valued parameters because you know we are talking about the weights of neural networks here but for the sake of uh uh demonstration I'm writing them as scalless and suppose you have a function that is that you are optimizing. So saddled

**[24:04]** point is a point okay that we are that you are seeking such that if you move along the theta direction at that particular point the function increases and if you move along the direction of w star at that particular point the function decreases right so what we are seeking is one point okay yeah these are the contours of the function so this point is such that it's not well repres presented in

**[24:33]** this uh diagram. But yeah this point is such that the at let's call this as theta* comma w star at theta* comma w star the function okay in this case it is j theta w around theta

**[25:05]** star and decreases decreases around W star. So you have the you have one point okay so if you traverse at that if you traverse in the w direction around that point okay the function decreases and if you traverse around the uh that point in the theta direction the function increases. So the this sort of

**[25:34]** a point is what is called as a saddle point. Okay, let me just take this off. Now what we are seeking in uh the variational divergence minimization is a saddle point. So this is by construction. So typically in optimization while solving optimization problems uh the general guidelines is to avoid saddle points right because uh it's it's ne I mean it's it's it's not uh an absolute optima in the sense that uh uh in one direction the function

**[26:03]** increases the other direction the function decreases but this is one optimization problem okay a rare optimization problem where we are deliberately seeking a settle point okay so that is why you know you might have heard that trying these kinds of uh uh uh networks or you know solving these kinds of optimization problems are pretty difficult because getting to a saddle point is uh is not easy and there can be multiple saddle points for a

**[26:31]** given loss landscape or an objective landscape. Therefore it's not uh easy to find solution for these kinds of saddle point problems. But yeah, so what we have is uh a saddle point optimization problem where there is a function and we want to seek the maximum of that function with respect to some parameters and we want to seek the minimum of that fun of that function of that same function with respect to some other parameters. Right? So what we get here

**[26:59]** these are solution for a saddle point optimization problem. Okay. uh now if uh I mean I'll just jump but yeah so uh for people who actually know how uh an adversarial training or generative adversarial networks work in fact this is this is what a blueprint for an adversarial network is. So any started point problem is called an

**[27:26]** adversarial optimization problem also right? This is also an adverse serial optimization problem in the sense that uh whatever theta network is trying to do right which is uh minimizing this objective function the w network is trying to do exactly the opposite of what theta network is doing and vice versa right

**[27:54]** because there's an objective that theta network is trying to minimize And the same objective is uh being maximized by the W network. And that is why uh these two networks can be seen as adversaries to each other. One is trying to undo what the other is trying to do, right? Uh in terms of objectives and that's why saddle point problems can be called as adversarial optimization problems and hence the name uh adversarial networks, you know, we will see more into it.

**[28:22]** Okay. So with this uh what we will do next is we will see one concrete instantiation of uh uh this particular uh uh setup okay which happens to be called the generative adversarial networks or GANs. or popularly known as GANs. Okay. So

**[28:58]** these are nothing but uh a particular instance of uh the variation divergence minimization that we saw. And by the way, uh this network is typically called as the the generator network. Uh because uh this is trying to uh sample from the distribution of interest and this network is called uh this is the discriminator network and

**[29:45]** this network is called the generator network and uh you can see why right I mean this is this is a critique or a discriminator I'll tell you why it is called a discriminator network in one of the instantiations that we will do this neural network will can be interpreted as a classifier we'll see how that is possibility. This is called a critique network because uh it is trying to uh bound the divergence, right? I mean whatever quantity that this network is trying to optimize uh this network is trying to construct a bound on top of

**[30:15]** it. So it's it's sort of uh uh trying to create uh a bound not sort of it's exactly trying to create the bound which the generator network is trying to optimize and that's why the name the critic network discriminator because in one particular instancation of this uh this network boil down to be a classification network which we will see in a
