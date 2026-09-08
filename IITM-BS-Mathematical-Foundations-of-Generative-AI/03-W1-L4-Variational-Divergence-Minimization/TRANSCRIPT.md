# Transcript — W1_L4: Variational divergence minimization

> **Source:** https://www.youtube.com/watch?v=nfZQYopzv20  
> **Channel:** IIT Madras - B.S. Degree Programme  
> **Duration:** ~26 min  
> **Note:** Auto-captions cleaned lightly. Minor ASR errors possible.

---

**[00:11]** [Music] Hello everyone. Welcome to this session of the course on deep generative models. In this session, we will be looking at a class of generative models based on a principle called variational divergence minimization. The famous generative adversarial networks or uh GANs are also a member of uh this family of generative

**[00:41]** models. Now before we go into the details of uh variational divergence minimization I would quickly recap the concepts that we saw in the previous session so that there is continuity. Now in the previous session we looked at a formal definition of what generative modeling is. Recall what we are g given is a data set which is a set of n samples drawn iid independently

**[01:14]** identically distributed from an underlying uh unknown distribution which we denote by script px. Now given this the goal is to estimate the underlying distribution px and learn to sample from it. As I said in the previous session, estimating PX may be implicit or explicit explicit depending upon what model what model are we looking at. But one of the primary requirements of a

**[01:44]** generative model is to learn to sample from the unknown underlying distribution in this case px. Now we also looked at what is a general principle of uh learning generative models. uh I told you that it involves uh three broad steps as a recipe. The first thing is to assume a parametric family on vx denoted by p theta typically in uh

**[02:14]** modern day generative models. This p theta is represented using deep neural networks and this is what is referred to as a model. Okay. Then once you represent the underlying distribution using some parametric model P theta. Now one has to define and estimate a divergence metric between uh the true distribution px and the model distribution p theta. The next step is to solve an

**[02:44]** optimization problem over the parameters of P theta the model distribution with a goal of minimizing the aforementioned divergence metric that would capture the distance between P theta and px. Now the idea is to set the parameters of this parametric distribution p theta in such a way that it approaches px as closely as possible. We also looked at uh an example of a

**[03:12]** class of uh methods uh that are called push forward methods where the idea is to start uh with an arbitrary random enable from which we know how to sample from. So typical choice would be uh a normal distribution with zero mean and uh unit variance and then represent your p theta as the distribution obtained from the output of

**[03:41]** a deterministic function g theta that would map the samples from the space of z from which the random variable is sampled to the space of data denoted by script x. Now what happens is that by the uh the elementary probability theory we know that uh a random variable when pushed forward put through a deterministic function will lead to

**[04:10]** another random variable denoted by xcap in this case and will have a different distribution as compared to the distribution of the initial uh random variable that was used as an input. Now the probability distribution of the output random variable determines on what sort of function is d theta. Right? So the idea uh in uh these classes of uh denative models which uses

**[04:41]** a push forward mechanism is to represent the function d theta as a deep neural network. Okay. And the distribution that this particular neural network is imposing is denoted by P theta. Okay. Then the idea is to start from an arbitrary distribution uh which often happens to be a normal distribution pass it through a deterministic neural network and obtain

**[05:09]** the samples from some distribution denoted by v theta. Please note that the output of this neural network is only giving us samples from P theta but not the distribution P theta itself. Okay. Then as we saw the objective is set as a minimization an optimization problem over the parameter space of this particular function in this case in neural network such that a

**[05:39]** distributional divergence metric between the true distribution and the distribution that is imposed by the model is minimized. Now the hope is that if the distribution if the divergence distribution of divergence metric is well defined and the optimization problem uh is solved appropriately then when the optimization is solved and you reach the optimal set of parameters

**[06:09]** theta star the uh distribution that is imposed by this neural network P theta would have gone close to uh the true distribution px because the divergence metric has this property that it is always non- negative and is zero if and only if the constituent distributions match. Okay, if that happens then if you sample uh a data point from a normal distribution and pass it through this

**[06:39]** network the trained network d theta star then the samples that we will be getting at the output of this neural network would be samples from px because now p theta star is close to px by construction. Okay, this is the general principle and we start the previous session uh by asking the following questions that uh how do we compute divergence metrics without knowing neither of P theta and px. Right? So now

**[07:09]** all we have are samples drawn from px. We also know how to get samples from P theta because obtaining sample from P theta is easy by design because all you need to do is sample a Z okay which is uh from an arbitrary distribution pass it through the G function and whatever samples you are getting at the output of this neural network are nothing but the samples from P theta. So we have samples from PX we have samples from P theta.

**[07:39]** However, we neither know px nor we know b theta. Okay, we don't know what the underlying density functions are but we have samples from it. And the other question that we ask is what should be the choice of the divergence method that we would be using. The third relevant question is how to choose the particular g theta. Okay, which says that if you choose a p theta then you are choosing v sorry if you are choosing the function g theta then in turn you're choosing p

**[08:08]** theta as well. What choices can one make on d theta? The other relevant question that we asked is how do we solve the optimization problem of minimizing the divergence metric? Okay, so these are the questions that one should ask if one is interested in constructing a jazz model using the push forward method that we just looked at. So now variational divergence minimization which we will be going to uh look at now is no different from uh the push forward method that we

**[08:39]** just saw. Okay. So let us continue. Now we'll be looking minimization. We'll be looking at the

**[09:11]** method of variational divergence minimization. Now to start with let us define the first thing that we should do is define divergence metric. metric between

**[09:49]** distributions. Recall that one of the key ingredients of uh building a generative model is to define a distributional divergence metric or divergence metric between distributions. Now let us do that. Let us define a class of distributional divergence matrix called the F given

**[10:21]** two probability distribution functions with the corresponding denoted denoted

**[11:07]** by px and p theta. The F them. Between them is denoted as follows. It is typically denoted by

**[11:53]** DF and within the parenthesis we write the density functions corresponding to the distributions which we are measuring the divergences between. So please note that uh I have assumed that uh the underlying distribution functions have well definfined uh probability density functions here and I'm also assuming

**[12:21]** that the underlying random variables are continuous random variables here. There are equivalent definitions of f divergences for the cases where the density functions are not well defined and the random variables are discrete. But since most of the generative models that we'll be looking at the random variables are continuous and density functions are well defined and they exist. Uh I am defining the f divergences for this special case where

**[12:50]** we have the density functions well definfined and the underlying random variables are continuous. So the f divergence between px and p theta is defined as the integral the integral of p theta evaluated at x time f of px evaluated at

**[13:18]** x over p theta at dx and the integral is over the space over which the random variables are defined. Now here f of u is a convex an arbitrary convex function defined from positive real numbers to

**[13:46]** real numbers. It's a convex function continuous function with the value of the function at uh Now if you look at the

**[14:31]** definition and where which the distrib distributions px and p thet are supported. Typically this space happens

**[14:58]** to be uh rd the d-dimensional real space because that is where our data is coming from. Now you please note that um uh f here is any function that is convex and left semicontinuous such that f of 1 equal to z. So one can choose an f which has this particular properties and that particular any particular choice of fs

**[15:25]** is going to give you one divergence metric that has special properties. Now uh u here is a dummy variable here. So f is a function from positive real numbers to real numbers. So please note that uh this is a well- definfined quantity because uh what we are taking here is a ratio of density functions evaluated at the same point. Now by definition

**[15:54]** density functions are uh non- negative. Therefore the ratio of two density functions evaluated at uh the same point is going to be a non- negative quantity rather a positive quantity. So now uh and the range space of density functions are also scalar right. So in uh despite the fact that x here is a d-dimensional uh random vector uh when you evaluate the density function at that particular

**[16:22]** point the density function evaluates to a positive uh real number. Therefore the ratio of uh two density functions is going to be another positive real number and therefore f of a positive real number is well defined. Right? And uh the f function by definition is a convex function which is left semicontinuous with uh the value of the function at one being equal to zero. Okay. Now okay. So what are the properties of uh this f

**[16:51]** divergence? See the properties that we need are the following properties of uh f divergence. are as follows. So one f divergence for all f for any f is nonredited.

**[17:25]** Okay, for any choice of f. Well, this is one property uh that we desire because um we want the divergence metric between uh two distributions to be always a non- negative quantity. Right? So this is by construction f divergence is non- negative for any choice of f. The other important property that is of uh importance to us is that uh the f divergence between any pair of

**[17:56]** distributions will be zero if and only if the two distributions are exactly equal. Okay. So we will use both of these properties uh for our uh case while constructing generative models. Now uh recall our goal was to uh build a generative model uh such that the distributional divergence between

**[18:25]** the true distribution and the model distribution is going to be zero. Now if we take the divergence metric as the f divergence right then by these two properties right it is always non- negative and it will go to zero only when the underlying distributions match uh we can use this distribution divergence metric for the case of for the purpose of building the generative model. Okay. Now let us look at some

**[18:53]** examples of this f Now if f f of u is u log u okay then the corresponding f divergence will be the famous ulb

**[19:23]** library divergence or kale divergence okay it's very very easy to see that uh with this particular choice of f divergence the underlying uh sorry f function the underlying uh f divergence become the kale divergence. Just do the algebra very quickly. So df between px and p theta for this particular case will be recall the definition it will be

**[19:52]** integral / x. We have p theta of x f of f of uh px by p theta dx. So this is the definition of the pairs. Now substituting uh u log u for f function will give us the following. So it will be so now u will

**[20:24]** be the ratio of px and p theta. Therefore this would be u that is uh px divided by p theta x * log u. So u is again px divided by p theta dx. Now this p theta cancels away. So

**[20:56]** which would make px theta px / p theta which is the definition of the famous k divergence right so now k divergence is a special case of f

**[21:26]** divergence with this particular choice of the f function. Okay. So now as you can see uh this f divergence will give you access to a large family of divergence matrix. Okay. That can be used for different purposes. Now if uh for those of you who know f diver k divergence is no is is not known to be symmetric. What do you mean by that? uh if

**[21:55]** you consider the K divergence between distributions px and p theta okay this is not the same as the k divergence between the distribution e theta and px okay now there are consequences of uh kale divergence not being symmetric okay so now if one does not want to uh use kale divergence or uh by the way this is

**[22:25]** called the forward scale divergence and this is refers this is referred to as the reverse scale divergence in the literature. Now for whatever reason uh one does not want to use scale divergence as the uh divergence metric while solving or while building the generative model because of the fact that it is not symmetric. Now one can use a different

**[22:53]** divergence metric and uh which would come with the property of its own while building the generative models. So now uh the this motivates us to I mean uh this justifies why one needs to look at a large family of divergence metrics. Okay, because different choices for this f function is going to give us different divergence metrics that would have that would

**[23:22]** possess different properties. Okay, so this is one example of uh the uh f divergence. The other example could be where f function is given by half u log u minus u + 2. Right? So if you choose this f to be

**[23:55]** this way then the corresponding f divergence is called the Jensen Shannon divergence or JS divergence. A version of this is what is used in uh the famous generative adversarial network. We will see that as we move on. So this is the definition of the Jensen Shannon divergence. The other famous choice of a function is half times the modulus of u - one and

**[24:28]** this particular uh corresponding ep divergence is called the total variation distance. Okay. So now the summary is that by and by the way all the functions that we looked at the f function uh obey this these properties that uh it is from r + 2 r they are convex and they are

**[24:57]** left semicontinuous with f of 1 equal to zero. One can easily verify that. Now the summary is that by choosing a particular f divergence one gets sorry a particular f function one gets access to an f divergence with certain properties. Okay. Now different choices for the f function will lead to different uh divergence metric and therefore different properties uh in the way the generative model is going to be built.

**[25:28]** Now the next thing that we should do is that uh given a particular epiverance I will describe a generic algorithm that can that that that is used to construct a generative model using the push forward method that we saw by minimizing any f divergence. Okay. So now after that what we will do is that we will look at a particular example by fixing a particular f divergence and see how does that converge into the famous algorithm

**[25:59]** of generative adversarial networks. Okay. So that is the idea.
