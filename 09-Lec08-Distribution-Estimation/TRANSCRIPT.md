# Transcript — Lec 08 Distribution Estimation

> **Source:** https://www.youtube.com/watch?v=aYb8KG9JYsg  
> **Channel:** NPTEL - Indian Institute of Science, Bengaluru  
> **Duration:** ~28 min  
> **Note:** Auto-captions cleaned lightly. Minor ASR errors possible.

---

**[00:03]** Hello everyone, welcome to this session of the course. In this particular session, we will continue from the previous discussions where we set the context for the machine for defining what the core problem in machine learning is and we looked at the foundations of probability theory in terms of uh random variables, distribution functions, conditional distributions and so on. We will build on those concepts today and we will set the problem the formulation of the

**[00:35]** foundational problem in machine learning and I'll also introduce you to this idea called empirical risk minimization framework or erm framework that happens to be one specialized way of realizing these general class of distribution estimation problems which would pave the way uh to look different algorithms that we'll be looking at this course. Okay. So recall that

**[01:08]** we are given a data set points denoted by X I Yi. You have n such data points and this we say has been sampled iid from a distribution pxy

**[01:39]** which is unknown and we say that this all these x i are in general points in a d-dimensional cartition plane and yis can be points in a k dimensional space or they can be discrete with k possible categories and so on. So this is what we said

**[02:08]** and I said in general this y or the label may not exist. So let me also introduce you to the nomature. Typically right I mean this is uh this nomature may not hold true everywhere but typically x is called the the input or the feature features or data depending upon

**[02:37]** what literature you're looking at. I is typically called as the the label space or output. But please note that for our purpose they are the elements from the range space of the underlying random variable and there is one sample space that exists. Okay. Uh upon which we have defined these random variables. Both X and Y are random variables here. So let

**[03:06]** me write that as well. So x and y are random variables Okay. So this is what we are given. So

**[03:32]** the input or the uh the raw material for machine learning is this data set. You might have heard this term, right? Data is the new oil. So just like oil is the raw material for a lot of industrial applications, data set or data is the raw material for all the machine learning. And uh whatever algorithms that you come up with will be as good as these data points. We will see we'll formalize this notion as we move on. But this is the starting point. You are given end data points that are sampled

**[04:01]** IID from an unknown distribution. So we also looked at two different views, right? You can look into this data set as n realization of a single random variable with a with a particular distribution or you can look into that as one realization of n different random variables which are all statistically independent and identically distributed. I hope that these terminologies makes perfect sense to you. Now I mean I have spent three classes to uh make you understand what these terminologies is.

**[04:29]** From now on remember that that when I say that there exists a data set I'm talking about a a few points that we have gotten from the range space of an underlying one underlying random variable or n the random variables which are statistically independent and identically distributed and there exists an underlying sample space uh which has a probability measure and we have pushed it forward using these random variables

**[04:56]** onto a distribution and it is the distribution that we do not know. Okay, this is the world view that we have to have. Okay, no matter what sort of problem we are solving, this is this is the world view that we have. This is the starting point. Okay, now any questions on this because this is extremely important about the I so I get it like what independence means. So two images are independent of each other. But when we when we come talk about identical uh

**[05:27]** isn't like everything identical. So, &gt;&gt; okay, hold on. See, the question is, so what is this IID all about? Uh, one request that I have is these identity and independence they're not English terms. They are mathematically defined constructs. Okay, I can just give you rough intuition of what it may translate to in practice. So identity means here in IID identity means that if you look into these end points as having one sample from n

**[05:55]** random variables we are saying that the underlying distributions of all these n random variables are identical which means that the underlying probability measure has the exact same functional form. That's what we mean. This is identity independence is statistical independence in the sense that if you have two random variables that are statistical independence the joint distribution becomes the product of marginalss. That's all. Mathematically it is not more than this that's all it is and when we say iid this is all we

**[06:25]** are talking about but if you want an intuition okay a rough intuition and if you look at that image example that we are looking at we are saying that one image is independent of the other image again you know if you ask me what independence mean right I don't know mean I can only define independence in a statistical sense in a in a mathematical sense precise mathematical sense that which we have done ready. Okay. Now, roughly speaking, you are considering each data point as as an independent

**[06:53]** data point. Okay. Which means that uh the labels or the characteristics of one data point, one image does not depend on the other which is a fair assumption to make. The other thing that we are seeing is all these images are being captured uh using let's say same under same conditions such as you know for for this X-ray example that I've been giving. It may mean that they have been captured using an X-ray machine that has more or less a similar uh feature similar device characteristics and so on and the

**[07:23]** sampling is being done um within the same kind of population and so on. That's it. And mind you uh you know this this jump from mathematical uh perfection to some intuition is is rather weak and rough. So you can consider any data to be ID. The only thing is does it make sense practically is the question. I can tell you what happens if you break ID assumption in practice and there are methods that would that are specifically designed to

**[07:52]** handle nonID data. However, it doesn't go beyond the mathematical constraints. So you can assume an X-ray im a data set that contains X-ray images and the pictures of human faces to be coming from the same distribution. That's perfectly valid. Okay. So that's a design choice. So the question is mathematically speaking you know we assume because all the algorithms that we construct will base uh their construction upon

**[08:21]** this in the sense that they assume that these random variables are statistically independent and we use properties but from a designer perspective when you are given data the question that you should be asking is is the data points are the data points that I have within this data within this data set be fairly represented using an ID assumption. That's it. If it's not, then don't use algorithms that are based on this. That's the important. Okay. Right. So, this is the starting

**[08:49]** point. We have we have a data set that is sampled ID from an unknown underlying distribution. And these are these are points from the range space of an underlying random variable that has some probability measure. Okay. So the problem the foundational problem or the fundamental problem of machine learning is

**[09:20]** given D sampled from an unknown distribution. Estimate estimate the distribution. This is one objective. Estimating the distribution is one objective. The other objective is sample from the

**[09:48]** distribution. Okay, these are the foundational questions that uh machine learning is is asking. So we are given samples from an unknown distribution. We want to estimate the underlying distribution. That's it. Okay. And the methods that are designed

**[10:18]** to estimate the underlying distribution but does not look does not bother about sampling are typically called the discriminative methods or discriminative models. I will I I will give you more uh uh intuition on why they are called discriminative models in a while. They are called discriminative models. But the ones that are designed to sample are called generative models so-called geni. Okay. The the foundational problem is that you are given samples from an

**[10:45]** unknown distribution. Estimate underlying distribution. There's one slight variation of this. Okay, I'll come to that in a while. So this is the foundational problem of machine learning and all the algorithms. Okay, that oh all the algorithms that we'll be looking in this course will be mostly concerned on the first part of this problem because uh this course does

**[11:13]** not look at a lot of generative uh samplers. That is for the next course. The next course is completely on generative sampling. This course is mostly on the distribution estimation. Okay, how do you estimate underlying distribution? &gt;&gt; You will know when you will know. Okay, so um I talked about estimating the distributions. The question is what distributions do we estimate? Correct. So let's look at distribution estimation.

**[11:57]** So we said given D is again uh samples drawn iid from an unknown px this is what we start with what do we estimate now depends on what problem you know what is what is your interest okay so depending upon there I'll introduce some nomatures uh you usually used nominatores.

**[12:24]** There are people categorize algorithms in ML based on what distribution are you trying to estimate? At times what happens is it is not the distribution that you estimate but you estimate some functions of distributions. What do I mean by that? I'm I I I hope that you people are aware of these these kinds of functions which are called moment generative functions and moments and so on. Right? So what is

**[12:52]** the moment is the generalized idea of expectation or the mean. I mean I should not be calling it mean it's the expectation of the distribution. So given D some of the problems actually estimate the expectation of the distribution. So you don't need the entire distribution but you only need an expected value of the distribution or let's say the the second moment of the distribution and so on. So this is one other way of putting the problem right. So given samples that you are that you are uh that are coming from an unknown

**[13:20]** underlying distribution estimates first moment or second moment and so on. This is one way to look at it. The other way to look at it is get the entire distribution completely. So you estimate either the distribution or you estimate some function on top of the distribution typically moment sort of functions. Okay. Now, as I said, okay, sorry, this is sampled from PXY, right? So, we chose to u separate one other random variable and we wanted to call that a label. Okay?

**[13:48]** Now, so one example or examples of distribution estimation, right? So, examples can be as follows. So one example can be estimate P Y given X where given D estimate P of Y given X. So what is P of Y given X? It is the conditional distribution of this random variable Y given X. So why is this of importance? Yeah. So when I say Y, it's different

**[14:18]** from this Y. So don't ask me why. Okay. So estimate the conditional distribution of y given x can be one problem. So why is this of importance? Recall that typically in problems we chose to call this as labels some labels. Now what we need to know is we we would like to know the likelihood. Remember that uh the distribution functions can be distribution functions are probability measures and probability

**[14:46]** measures can be looked at or interpreted as likelihoods. So P of Y given X would give you the likelihood of obtaining the labels given that you have already observed the data points. In the example that we have been looking at right which is disease classification given that we have seen an image we would want to know what category that this image belongs to. Does it make sense? So this is this is one way to do it. And if in the bounding

**[15:16]** box regression example that we just looked at right which is uh uh given an image you find out where the tumor is now know where what we are estimating is the the width height and the center of that box where your y is in three dimensions. Even in that case we would like to estimate P of Y given X because given this image you tell me what is the likelihood of this random variable Y taking different values. You see the point? Now these sort of problems right where you estimate P of Y

**[15:44]** given X are called classification Honestly I'm not big fan of these nomines because these are the boundaries are very very thin. So typically classifications are the cases where Y is discrete. Regression problems are the ones where Y

**[16:15]** is in RK. Okay, the these are classification and regression problems and you can estimate other uh kind of problems are where you estimate either px or py or p of xy either the joint distributions are

**[16:42]** marginalss. It's possible, right? So why do you want to do it here? You would want to estimate P of Y given X because one of the use cases is where given an image we would like to know what what is the label uh corresponding to this particular image. You want to do this. Why do you want to estimate PX, PY or PXY in most cases? You would want to do this

**[17:10]** in generative modeling because you want to sample from this. As I said, this is not generative modeling because generative modeling are the algorithms where you implicitly sample. Estimating the marginalss or the joints do not mean that you need to sample. Okay. But you can still estimate the marginalss and joints. Suppose you want to know given data. You would want to let's say that you are given thousand images, right? The question that I want to ask is what is the likelihood of

**[17:42]** seeing a particular disease in a population. Now you should be able to distinguish between okay let me ask this question then we'll go on. Let's consider this example of having 10,000 images and we we have labels. What these labels are? These labels are binary labels. Disease or non-d disease. This is the kind of data that you have. So how does it look like in practice? You have an image and a

**[18:09]** corresponding label, right? So which is a binary label. That's what you have. And you have 10,000 such images. Now I will give you um some possible use cases. Okay. And you try to identify what is the underlying problem that you are trying to solve or rather what is the distribution that you are trying to estimate there. Okay. Obviously, if you want to solve this problem that I want to know what whether a given image corresponds to the disease

**[18:38]** case or not, then what distribution am I interested in? I'm interested in P of Y given X. The the conditional distribution of Y given X is what I'm interested in. Correct? Now, uh suppose I'm interested in this question. What is the likelihood of observing a particular disease in a given population? What distribution am I looking at? P. &gt;&gt; It's P of Y. Can all of you see this? So I I'm not interested in relating the labels to

**[19:06]** images at all. I'm only interested in knowing whether a given population or rather what is the likelihood of obtaining a particular disease in a given population. That's what I'm interested in. And that is PY. Okay. What about PX? What is PX? PXs I'm interested I'm interested in knowing how are the pixel values appearing in a particular image and I

**[19:35]** don't care about the disease at all that's px make sense okay and can somebody tell me what would uh estimating px given y correspond to px given y correspond to I only consider the images that have that have disease and then ask this question how are the pixels distributed in this particular case when the this when the diseases are when when when I know that

**[20:02]** it's disease does it make sense so now as a practitioner what one should do is so you start with a problem you know we all start with real world problems uh the the real power of of being a good machine learning engineer come comes from abstracting out that real world problem into the space of math where you where you where you tell yourself or rather find out what sort of problem am I interested in how do I cast this as an

**[20:30]** ML problem [clears throat] and remember I told you you know this this this notion of label is sort of pseudo h suppose I'm solving this problem of in painting which is given an image okay some of the uh pixels do not exist no they they're just masked and what you are interested in is finding out what those pixels Where where do you fit that? You fit that in P Y given X and those Y's are

**[21:01]** some of the dimensions of this X itself. So now you know right? I mean that's why I call the labels as sudo. So whatever your labels are your labels are you just have to have data that way. Okay. So that is why I'm not a big fan of a nomin that I'm going to introduce you to introduce to you right now. There's this separation of algorithms called the supervised algorithms and unsupervised algorithms. Supervised learning and unsupervised learning. So the definition is the following.

**[21:29]** Supervised learning are those algorithms which use labels and unsupervised learning are those algorithms that does not use labels. Now tell me what is supervised and what is unsupervised? in our nomature right because we chose to see the labels as the dimensions of data or rather one other random variable that is defined on the same sample space what what is supervised what is unsupervised that's why I'm not very

**[21:57]** comfortable with this nomature of supervised unsupervised but that's the definition the conventional definition is if there are labels in the problem and you're estimating distributions they are called supervised learning if there are no labels in the uh in in in the data set then it's called unsupervised learning. Okay. Generally what happens is labels come from human beings. That's why supervised learning is is is considered a costly affair because you

**[22:26]** know some a clinician has to sit down and give those labels. Unsupervised learning is cheaper or because there are no there is no involvement of a human being where you know they have to uh label the data that that's how it is perceived. But as far as a mathematician is concerned &gt;&gt; we don't care because there are just two random variables that are defined on the same sample space. See appreciate this world view. If you can

**[22:55]** see a d-dimensional vector valued random variable as D random scaler valued random variables on the same space, what stops you from seeing one other scalar random variable which you call as a label as one other random variable on the same sample space just that it is being assigned by some human being that does not you know take it out from the formulation that we have made. Okay, did you get it? So now yeah nominator I just wanted to introduce you to the nomin cl

**[23:22]** because otherwise the course will not be called a machine learning. If you go out of a machine learning course and tell tell people that you don't know what supervised and unsupervised learning is then you know people blame me not you what sort of a teacher he is. So now you know what supervised and unsupervised learning is actually we will look at a few algorithms that may be called unsupervised learning and supervised learning and so on. But mostly in the course I mix both because I don't care. I mean in the sense that our formulation does not care whether it's a label or not. It's just

**[23:49]** one dimension in our data and we will encompass that. So for as long as the course is concerned we'll be the problem that we'll be solving is given a distribution sorry given a few samples that are that that are sampled from sampled ID from a distributed you just noticed I use the word sample once as a verb as once as a noun. I said given that you have a few samples sampled from an underlying distribution,

**[24:17]** right? The first sample is verb and the second sampled is sorry the first one is known and the second one is verb. Anyway, so given a few data points that are sampled from an underlying distribution which is unknown, you estimate the underlying distribution. This is the problem that we'll be solving and we'll be looking at different algorithms to do this. So this sampling uh sorry this distribution that we are estimating either it can be joint it can be marginal it can be conditional or it can be some function it can be a

**[24:46]** moment on the distribution and so on that's that's the statistical problem that we'll be solving in this course. Okay and depending upon what the nature of uh uh you know suppose if you wish to call this one dimension of random variable as a label you label the problem as a supervised or an unsupervised learning algorithm. Okay. And uh if you're estimating uh conditionals then it's called if you if you're estimating the the conditional distribution of this sort and if your y which is the label uh random variable is

**[25:17]** discrete then the underlying problem is called the classification problem. If you make it if you make y to be uh in r k or continuous then it is called regression problem and so on. This is the nomature. [snorts] Okay any question so far? So when you talk about Y like the likelihood that the image we have received is &gt;&gt; no there is no image here I'm talking

**[25:45]** exam &gt;&gt; no even in exam that's the precise point that I want to make there is no in if you are estimating py there is no image all we are saying is in this population what is the likelihood of somebody getting disease or not is what py is &gt;&gt; okay so isn't that the same as p of x &gt;&gt; no precisely not so the question is how is py different from p of x given y. You need to understand this. This is very important. P of x given y is what is the likelihood of this particular pixel in

**[26:14]** this image taking this particular value given that I'm looking at diseased image. That's px given y and py has nothing to do with images. Py is not even bothered about images. Okay. So given data so P of X given Y which is the conditional of data given labels is talking about the values that we have observed for the data given that the label is fixed to some value. And please remember one other thing in all conditionals

**[26:50]** P of X given Y right this is the notation but this actually means right as we are saying I have to estimate this and evaluate it at a point X okay given that my other random variable Y is fixed at a particular value Y this is what we are estimating so when you evaluate the conditional distribution. I told this but I I'm reiterating it. Always remember that the conditioned random variable is fixed at a particular value always.

**[27:19]** Okay. Therefore, conditional distributions are function of functions of conditioned random variable. Right? Why? Because if you change this Y, there too many Y's. If you change this Y, the conditional distribution changes. Got it? Yeah. So whenever I write conditionals you know notation wise I will most likely not be writing the arguments of these distributions but you should remember right I mean see whatever is inside this bracket so especially the conditional

**[27:45]** distribution is a little tricky in the sense that this x what is this x this x is the this is dummy in the sense that this is where I'm evaluating this particular distribution okay and this y is not dummy I'm actually fixing this y into a particular value and depending upon that value this evaluates to a different function al together. And what is this x given y that I write here? This is simply saying that it is a conditional distribution involving two random variables x and y. Is my notation clear? I'll be using this

**[28:14]** notation throughout. Okay? And now that I'm looking at notations, I I'll also introduce you to another notation. See, whenever I write the script P, this is this this represents distribution functions. But mostly in machine learning, people work with density functions. not the distribution function. So let me introduce a concept called probability density function. We'll see you in the next lecture.
