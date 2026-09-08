# Transcript — Lec 13 Minimization of KL Divergence

> **Source:** https://www.youtube.com/watch?v=Ij4p5hLbfo4  
> **Channel:** NPTEL - Indian Institute of Science, Bengaluru  
> **Duration:** ~24 min  
> **Note:** Auto-captions cleaned lightly. Minor ASR errors possible.

---

**[00:03]** Welcome back. So now that we have uh a way to quantify the distance between the density functions or distributions, we are good to go. We'll go back to our recipe. Right? So let's go back to our machine learning recipe. Now given okay for now see when I'm formulating this uh the general solution for the ML problem I will assume that we are given

**[00:33]** a particular random variable from a distribution and a lot of samples from it or I can make an ID assumption and say that I have n random variables one realization of each and we have n of them which are identically distributed independent and identically distributed. Let us stick to this view. Okay, let us because the iid assumption is made throughout. Let us stick to this view that we are given n random variables one sample of each all of

**[01:02]** which are independent statistically and identically distributed. Okay. Now this methodology that I'm going to talk about right now is applicable irrespective of the kind of data that you see in the sense that it can be a supervised problem it can be an unsupervised problem or or so on right which means that you can have label you can't have label and this method methodology is is applicable to problems where you are estimating marginalss or if you're

**[01:33]** estimating the joint distribution if you're estimating the conditional It doesn't matter. Okay, that is why I will make it generalized in a sense that suppose we are given suppose we have n samples So let's say that we have uh D which is

**[02:05]** I'll use uh another notation here V_sub_1 E2 V3 up to V N and this is drawn ID from a distribution uh I'll write the density P V. So note that this v can be px pxy px given y py given x anything depends upon what the underlying problem is. I the

**[02:34]** treatment now will be general. Okay. Now the goal is to estimate PV given D. Uh this is our goal. Now we know the recipe. Assume which can be evaluated at B as see this

**[03:05]** V is a dummy variable by the way P theta as B as the model density and compute K divergence. Now between P V and P theta. Okay. By assuming that we are dealing

**[03:33]** with continuous random variables, this is given by the integral of PV divided by P theta evaluated at V and you have DV here and this is over the space that these these random variables are defined. Let's assume that these uh all vi are in some d-dimensional real space.

**[04:00]** Okay, that's the most general observation assumption that we made. This is what it is. Right? Now, what are we interested in? Remember the recipe again. So we want our theta star or the optimal parameters to be the one that minimizes the kale divergence between and p theta. Correct? So this optimization is over theta and you

**[04:28]** let us uh open this term out and we have this integral PV log PV. The second term is uh minus integral p v log p theta.

**[05:05]** So integral is a linear operator and I'm just I can open the log and write this as uh uh push the integral after the negative sign. So note that this first term right this term can be completely ignored. Why? This is independent of theta. theta is the optimization variable and the first term is independent of that theta which is the optimization variable. Now how do you interpret this?

**[05:33]** So I'm saying that this is the entropy of data. I can do nothing about it. This data is given to me. See if I if I have to minimize the the divergence or uh the divergence metric between PVA and P theta the only thing that I can play with is P theta not PB because that's entropy of data and I have no control over it. So I can just knock that term off. Okay. Now this will be

**[06:12]** the negative of the integral EV log P theta P DV this is the optimization problem that I want to solve now okay now remember in the first class somebody asked a question that we aren't we begging the problem here because we started the the entire process by saying that we don't know PV right now when we don't know PV what's the way to compute this integral two

**[06:41]** things the first thing is this integral is intractable in RD it's in very high dimensions it's very hard to compute other thing is that we don't even know what PV is that's the whole problem that we are trying to solve so how do we solve this problem we use statistics so we can write this as arg min is the negative of what's this integral? This integral is expectation of correct. So by definition this integral

**[07:15]** is nothing but the expectation of log p theta with respect to pv. Okay. Now again can we compute this expectation? No, we can't. But here comes the the biggest twist in detail that statisticians tell us that expectations can be approximated using samples. So if you want to compute the

**[07:42]** expectation of a particular function with respect to a distribution and all that you have are samples from that distribution then you can approximate that expectation using something you know something that is computable and there's a theorem that guarantees you that if you have large amount of samples then this estimate that you have actually converges to the true expectation in probability. What's this uh theorem? &gt;&gt; Law of large numbers. So let's invoke

**[08:09]** that. Okay. So if we invoke the law of large numbers, you tell me which one. Weak law of large numbers. Expectation of log of P theta V respect

**[08:46]** to PV can be approximated by what's the estimator expect what is the estimator that would estimate the expectation it's the sample mean okay sample mean is given by 1 / Then where

**[09:15]** VI is sampled ID from PV only if this happens law of large number works. So you have to ensure that all the samples that you have are drawn IID from a particular distribution. Right? So now can we compute this quantity? Now we can because we know what P theta of VI is because that's what we have defined and we have samples from V PB that's it. So now law of large number

**[09:44]** says that this particular estimator now you know Why AI works only with

**[10:16]** terabytes of data? It's lot of large numbers. What you are actually doing is trying to approximate an estimate an expectation using sample average and uh statistics tell us tells us that that only works if you have lot of samples, right? So what is the infinity that is known to human beings? The entire internet. train your models on the entire

**[10:42]** internet. That's the story. This existed centuries before. What did not exist is that we could not traverse this arrow. Right? So this reminds me of a story, right? In in a class of mathematics. It was being told that uh there are infinite numbers between these two, right? and a student came and did this and said I just crossed infinity right anyway right so what what we what

**[11:13]** what are we doing is that by considering a large amount of samples uh we are approximating the expectation using sample averages now our optimization problem boils down to the following so we want to seek the minimizer where VI is sampled IID from PV.

**[11:57]** This became our estimator but in fact you'll be surprised to know that even charge GPT is trained using the same equation. This is exactly what is done. So now how is the optimization problem solved and what P theta is being used is different. Okay. Now this is what I call as the Makes sense, right? Because all we are

**[12:34]** doing is just if you trace back what we did started from the definition of scale divergence, right? The estimator that we have gotten for the parameters theta is the one that minimizes the kale divergence between the true distribution and the model distribution. What the indices and &gt;&gt; come again K and L. What do they say? &gt;&gt; Oh, I I thought I mentioned they are the names Kilbach and Ller two people. Okay. Kale divergence. In fact, the

**[13:04]** generalized version of it is what is called as an F divergence and Kale divergence is one member of that large family. So Kale divergences Kale divergence has its problems. It's not the best divergence to be used. Surprisingly, this is what is being used even today and it sort of empirically works. Okay. So this is the minimal care divergence estimator. Any questions so far? &gt;&gt; Oh, here is it. Okay. So the question is

**[13:42]** why is why are we considering KL between PV and P theta and not P theta and PV. So typically in literature no this is called as forward kale. The term that you refer to is called a reverse scale. Okay. Now if you take reverse scale you will have a term that you cannot compute. You can see that no because there will be a p theta log PV term. We can't do anything about

**[14:11]** it. That's why we use the forward cable. No, this is the expectation right this is the see okay um suppose there's a random variable this goes to uh the

**[14:39]** foundations of probability theory suppose there exists a random variable So look at this. There is a law called law the law of unconscious statistician or lotus it is abbreviated as. Now suppose x is a random variable and f of x is a function of a random variable. So typically or not typically right

**[15:09]** always the function of a random variable will have a different distribution compared to the random the the original random variable. Now if I were to talk about the expectation of the function of the random variable, it's actually equal to the uh expectation of that new random variable with respect to the distribution of that new random variable. However, it can be shown that expectation of a function of a random variable with respect to its distribution is equal to the expectation

**[15:37]** of the function of the random variable with respect to the original random variable. This is lotus. It can be proved. I think it should have been done in stom probability theory classes. Right? It's a okay so what I'm saying is show this result prove this will fix that in next class. I told them [snorts]

**[16:03]** maybe try to just uh push it a little or something. Okay, that is what it is. Yeah. Uh therefore see uh if you look at if you uh consider that then the expectation that we are trying to estimate is the expectation of this function of the random variable V. So

**[16:31]** this is that f of x that I wrote. So log of p theta of v is a function of the random variable v. So we are look we are looking to estimate the expectation of function of a random variable with respect to the underlying distribution of the random variable and we look at the sample averages and law of large numbers tells you that uh the sample average will converge to the true expectation as n tends to infinity. [snorts] Okay. Anything else? Okay. Now this is what I call as the uh the minimal dail

**[17:02]** estimator. But we also know that uh that I can write this as the max or theta of this particular function correct because okay now I'll introduce some nom remember I told you that if p theta is the model distribution that we have

**[17:32]** assumed over uh the true data. If you evaluate that at a particular point, I gave a name for this. If you evaluate the density function at a point, it is called the likelihood. Density functions being evaluated at a

**[17:57]** point is called likelihood. Okay. Now what is this? We are trying to find an estimator theta that would maximize the likelihood of all data points under the model. That's one another interpretation of this and that's why this estimator is also called the maximum likelihood estimator. So I can do the entire math in a

**[18:34]** different way. So what I'll say that is that P theta of VI okay is the likelihood of VI under P theta. So what I want to do is to find the theta that would maximize the likelihood for the entire data set. This is for one data point. If I have to do it for the entire data set, what should I have to do? I'll have to consider the joint distribution with respect to all the random variables

**[19:02]** that we have and I have n of them. See, our interpretation is that our data set has n random variables which are all iid. Correct? If I have to consider the joint distribution between n random variables that are iid, how do I do that? Product, right? So, I the the total joint likelihood of data. Let me write that of the data.

**[19:38]** So let me call that as L of D is given by the product all the distributions, right? uh okay now what do I need this is a

**[20:04]** function of theta of course I need to maximize the uh likelihood of this you know joint likelihood of data now I don't like products so I'll take I'll I'll go through I'll make this go through a monotonic function and log is one such thing. So what I'll do is that maximizing this is equivalent to maximizing the log of the likelihood of this. If I do that then this will get

**[20:33]** converted into a sum. Okay, then I seek theta star as the one that would maximize a scaled version of this. And people call this the maximum

**[20:59]** likelihood estimator. I'm not a big fan of this because why should you maximize the likelihood? To me, this estimator is the one that minimizes scale divergence. Mathematically, they are exactly the same same thing, right? The one that maximizes the likelihood over the data is the one that minimizes the K

**[21:26]** divergence between the true distribution and the model distribution. I would like to see that as the minim the minim the minimal K divergence estimator but I will call it as maximum likelihood estimator to ensure that that I'm following the convention &gt;&gt; exactly the same no that's why I wrote the math as well but remember that all we are doing is minimizing the K divergence between the model distribution and the distribution even

**[21:53]** when we maximize [clears throat] the so-called likelihood Okay. &gt;&gt; Oh, I assumption is always made and law of large numbers tells you that that this is a this is this is a good minimiz this is a good estimator for the minimizer. That's it. See the handwavy explanation that people give is the following. Right? What is P theta of V? P theta of V is the likelihood of

**[22:20]** obtaining V under T theta. So what do we need? We need a model that would maximize the likelihood of obtaining this particular data set and that's why we maximize the likelihood of uh this particular data set under the model distribution and we seek the parameter in such a way that this would maximize it. So that's that's the handway explanation that is given why should this be a good estimator and so on is a different question altogether but I like to see that as the one that minimizes

**[22:49]** the distribution of divergence between the true and the model distribution. Okay. Okay. Now one question that remains is any any questions so far? Okay. See one. Yeah. &gt;&gt; Less likely an event is the more

**[23:16]** information it gives us. So if we have a data set which is very less likely &gt;&gt; and if we maximize the likelihood it should give us a better like it will end up giving us a worse model for the entire thing but it but as per more information &gt;&gt; okay uh I'll sort of answer your question but I'll make a comment with your uh permission. This is precisely why I don't want to see this as uh uh maximum likelihood

**[23:45]** estimator because there's no there's no mathematically grounded interpretation to what is being done. So to your question, see look at what likelihood are we maximizing? We are maximizing the likelihood of seeing the data computed under the model distribution that we want to maximize. You understand? So given that uh I am treating my data coming from the model

**[24:13]** distribution give me that model distribution which would maximize the entropy sorry which will maximize the likelihood not entropy. Okay. Yeah. But again as I said uh seeing this as maximizing the likelihood is not something that I'm a fan of. I want to see this as the one that minimizes the clear divergence. Okay. So wherever you see this maximum likelihood estimator remember that it's the minim minimal K divergence

**[24:41]** estimator. Okay. Uh we will stop here and continue in the next lecture.
