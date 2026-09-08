# Transcript — Tutorial 10 : Review of Machine Learning 1

> **Source:** https://www.youtube.com/watch?v=wjSKM1xFoSU  
> **Channel:** NPTEL - Indian Institute of Science, Bengaluru  
> **Duration:** ~47 min  
> **Note:** Auto-captions cleaned lightly. Minor ASR errors possible.

---

**[00:01]** Hello guys, welcome to this bunch of tutorials. In which our major aim is to understand numerically mathematically how does we formulate things, how does we solve problems. Okay. Now most of the contents that we will be looking in these bunch of tutorials now are considered to be prerequisites. So you can think of these as a small

**[00:31]** recap of whatever ideas that have been discussed in the first level course of machine learning. So what are the topics that we shall be covering? Now we expect from the viewers of these tutorials that they are comfortable with the idea of maximum likelihood estimation with EM algorithm. Back propagation numerically, so not

**[00:59]** having an intuitive view of how does it happen in a system using dot backward, loss dot backward. That is not what we expect. We expect that the students or the viewers of these uh tutorials have been equipped with necessary mathematical foundations to understand these things. Okay. So without any further delay, so now let's jump in.

**[01:27]** In the first numerical problem, our main aim is uh to understand maximum likelihood estimation, MLE estimation. So now the problem goes like this. Let I have this X, now which is uh from a normal distribution with mean mu and variance one.

**[01:55]** Now from this distribution, now we have uh uh We have n IID realizations of x. Okay? But, what we have done is, while taking the observation, okay, we did not take down the actual value of the observation.

**[02:22]** We only noted whether it's positive or not. Okay? So, value is

**[02:58]** positive. Okay? Now, suppose suppose m of n observations that we have recorded Find the

**[03:25]** maximum likelihood estimation of the parameter mu Okay? I'm assuming that the uh problem setting is pretty much clear. I'm having n IID uh realizations from this distribution, which is there. And this distribution

**[03:52]** has only one parameter mu, and now from the writing it should be clear that these are scalar valued random variables which we are dealing with. Okay. So, now while going ahead with that, we only noted whether it is positive or not. Now, out of these n realizations that we have, now m of them is negative. Now, what is the maximum likelihood estimation of this mu which is there? Now, how do we go ahead and calculate

**[04:22]** this? So, now let's go into the solution. we know that the probability of uh x taking the value less than zero is uh probability x minus mu divided by one less than zero minus mu divided by one.

**[04:51]** Okay. Here, the standard deviation is one. Now, this is uh we are standardized x. Sorry, standardized uh x. Now, this will lead to This is probability of uh z less than minus mu. Now, here, since we have standardized, z is from the distribution 0 1. Okay. And we know this value from

**[05:19]** the properties of normal distribution, and we know this value. This is phi of minus mu. Okay. So, now now similarly now the probability of uh x greater than zero is one minus probability of x less than zero. &gt;&gt; [clears throat] &gt;&gt; This is one minus phi of minus mu.

**[05:49]** This is fee of mu. Now here, now at these points we are using a This is using inverse. See, I'm assuming that uh these ideas which are there are clear and comfortable. Okay. So, this this directly means that you would have

**[06:17]** uh undergone a rigorous course in probability theory and then a rigorous course at a first-level machine learning. Okay. So, now coming to the data, now we have M now which is uh number of uh negative observations. Okay. So, and then N minus M is uh

**[06:47]** number of positive uh observations. Okay. Now, for each of the observation, now which which which we can immediately say that that observation take what is the probability of it being positive or being negative for uh each observation. positive is uh

**[07:22]** fee of mu. Probability that this is uh negative is fee of minus mu. Now, with this, how can we go ahead and find the likelihood of the observed data? Okay, so now now observed data uh that I represent with

**[07:56]** uh L of mu is probability that X takes value less than zero have uh M of them and probability that X takes value greater than zero I have N minus M of them. Okay. Now this is fee of minus mu to the power of M

**[08:25]** into fee of mu to the power of N minus M. Okay. So now this is the likelihood. Now what do we do? We normally take the log. Okay. So now taking the log Now taking the log that we represent by L of mu is

**[08:53]** log of capital L of mu. Now this is This is log of fee of minus mu to the power of M into fee of mu to the power of N minus M. So now by using the log properties, now we can immediately see this as M

**[09:23]** log fee of minus mu plus N minus M log of fee of mu. Now just for the sake of simplicity, now we prefer to go ahead with some kind of substitution. So, now uh Let's say that P is uh

**[10:01]** probability that X is less than zero. This is phi of negative mu. And 1 minus P is probability X greater than zero. This is phi of mu. Now, we can easily substitute these. Now, because of which how does uh the likelihood term looks like? So, now

**[10:39]** the likelihood becomes I'm saying likelihood, not log likelihood. Now, it becomes L of P P to the power of M 1 minus P to the power of N minus M. Okay? This is what uh it becomes. Okay? Does this ring a bell? Does this formulation ring a bell? This is exactly the likelihood of

**[11:08]** Bernoulli. Okay? So, wherein which I'm considering the negative uh term is treated as success. So, you have M success that has been observed in N trials. Okay? So, now this is uh Bernoulli model

**[11:42]** where the negative is treated as success. Okay, and then we have m success that has been observed out of n observations or n trials. So, n success are observed

**[12:08]** in n trials. Now, what do we do? First, we find the MLE for p and then we convert back to mu. That is the the standard idea. Now, let's take the log likelihood. L of p is m log p plus n minus m log of 1 minus p. Now, how do we go ahead with that? We

**[12:36]** differentiate with respect to uh p and then uh equate it to zero. The standard idea. Okay, we want to get the maximum likelihood. So, you get the log likelihood. So, and then differentiate it with respect to whatever the parameter that we are looking for. So, and then equate it to zero and then simplify it. So, now now Now, differentiate

**[13:03]** with respect to p and equating to zero, d by dp of this l of p is uh m by p minus n minus m divided by 1 minus p. So, this is the thing. Now, you equate it to zero. Now, equate to

**[13:32]** zero, this is a m by p minus n minus m divided by 1 minus p is equal to zero. Now it will be m into 1 minus p equals n minus m into p. Now this will straightforward uh say that p is equal to m by n.

**[14:01]** Which implies the MLE observe uh thing for this p hat, that is how we represent. And this is m by n. Okay? Now this is something that is straightforward known to us. The MLE is uh the probability of getting a negative observation. So you have got m negative observations, so out of them n. Okay, now now this p hat which is there, now we have to convert

**[14:35]** back to mu hat. Now what is this p? The value of this p is phi of minus mu. Okay? That's how we started. Now we have to represent them in p hat terms. P hat is mu of minus mu hat. Now this is uh m by n is phi of minus mu hat. Okay?

**[15:01]** So now if phi of a equals to b implies a is equal to phi inverse of b. Now in this case a is equal to minus mu hat, b is equal to m by n. So, if we substitute this so we get uh

**[15:32]** negative mu hat is phi inverse of m by n. Okay? Now, which is uh mu hat is negative phi inverse of m by n. Okay? So now, uh we know that uh negative

**[15:59]** phi inverse of Q if we have this is same as phi inverse of 1 minus Q. These are the standard properties of uh the standard normal, which is there. Okay? And then it should be very much clear that we are dealing with standard normal. Now, we have this mu hat is uh phi inverse of 1 minus m by n. Now, this is therefore this mu hat

**[16:34]** is equal to phi inverse of n minus m by This is the MLE estimate that we have obtained for this mu hat. Now, this is uh uh intuitively clear for us. Okay? Now, if we go back to this term, which is there.

**[17:05]** Okay? So now, if we have many negative observations, so what does that mean? If we have many negative observations so then what will happen is m over n, now which is there, will be large enough. So therefore, uh this term fee of minus mu hat now will be large. Okay, this means that this mu hat minus mu hat which is there now is positive. So that means that

**[17:34]** the estimated value the estimated maximum likelihood estimate that we have obtained for this uh particular mu hat is negative. Now this makes total sense even for us. If many observations are negative, so then it's a straightforward indication that the normal distribution should have a negative mean. That is what now we are trying to look at this point. Okay. Now this is how we go ahead. The standard method is now which parameter you want to go ahead, find the

**[18:03]** the likelihood, find the and then get the log likelihood, differentiate with respect to zero, and equate it to zero. This is the the standard method of using MLE estimate. Okay, I assume that this is clear. So in addition to that, I just want you to bring it to the notice that you should be very much comfortable with another estimation which is map estimation. So I'll leave it to the viewers to have a recap of that as well.

**[18:31]** Okay. So now in the next uh problem that we shall be dealing with now let's go into one of the very interesting ideas now which is EM algorithm. Okay, expectation maximization algorithm. So now let's look into it. Now it will give you a nice recap of how the EM algorithm works and then how to solve and get the closed form solutions for the parameters. Okay. So now

**[19:02]** uh uh the density of X is uh is a mixture of two exponential densities. densities, okay. &gt;&gt; [snorts] &gt;&gt; So now then for any given observation X which is parameterized by theta, how

**[19:32]** does this look like? You have this pi beta one e to the power of minus beta one X plus one minus pi beta two e to the power of minus beta two X. Now here, what are the parameters? What are the theta represents? So theta is pi beta one beta two, okay.

**[20:01]** Now this pi is uh the probability of choosing from the component one, the first exponential density, and one minus pi is the probability that you're choosing from the second exponential density which is there. Now therefore, this pi has to be between zero and one, and this beta one and beta two now has to be uh greater than zero. So this pi which is there is between zero and one.

**[20:31]** And this beta one is greater than zero, and beta two is also greater than zero as per the exponential densities. Okay. Yeah, we can use equal to here. That's fine. Okay. So now, what what are we supposed to do on this? So we are supposed to use EM algorithm. So use expectation maximization algorithm.

**[21:01]** So now how we should do it? Now use uh binary Z Okay? as a latent variable. So, now you can think of it like this. I have a coin. I'll just uh toss it. If it is head, I'll choose from component one. If it is tail, I choose

**[21:29]** from component two. So, equivalent of saying that. So, E step, Q function, and M step. So, that means that just go through the whole EM algorithm once is what is being said here. Okay? So, now let's go into the solution.

**[21:58]** So, I I assume that the question is clear. So, now you have uh a mixture of two exponential densities. So, now you have to go ahead and obtain uh the forms for these parameters which are there. Okay? So, now let's go ahead with that. Okay. So, uh Let's look into the solution.

**[22:25]** Z is there. Takes the value one if X component one. It takes the value zero if X component two.

**[22:52]** Okay? This is what we have. So, this is a straightforward indication that Z Z is a pi. Okay? So, now Now, we have this uh f of x parameterized by theta.

**[23:21]** Okay. Oh, sorry. If z x z. So, we want the complete data. Okay. Now, what do we have is uh x1 x2 so on till xn observations. Okay. So, now for each xi there is a hidden latent variable zi.

**[23:49]** There is a hidden variable zi which is there. Okay. Now, therefore, what does the complete data look like? Now, therefore, the complete data is x1 z1 x2 z2 so on till

**[24:19]** xn zn. Okay, this is what we have. So, now all these observations are IIDs. Now, therefore, the joint complete data distribution data sorry data density sorry data density is represented by this f of xz parameterized by theta is the product of i is equal to 1 to n f of

**[24:47]** xi zi parameterized by this theta which is there. Now, this we can substitute. We know this. This is uh I is equal to 1 to n. Now, how it will be? This we can write it as pi beta 1 e to the power of Sorry, beta 1

**[25:15]** x i to the power of zi into 1 minus pi beta 2 e to the power of minus beta 2 xi to the power of 1 minus zi. So, it is quite simple. See, what are the values that each zi can take? Now, each zi can take uh zero or one. Now, let's say that zi is

**[25:44]** one. Okay? Now, then this will be one, okay? Because of which it is coming from component one. Now, therefore, 1 minus zi, it will be zero. Therefore, this whole thing will be one. Okay? This will become one. So, therefore, you'll only be considering this. Okay? Now, if uh zi is zero, that means that it is coming from component two. Now, then uh this will be zero, because of which this whole thing will be one. Now, this is 1

**[26:12]** minus zero is one. So, therefore, you'll be choosing from this. Okay? So, this makes sense, I guess. Now, then what you're supposed to do? Now, this is uh the complete data density. Now, you take the log. Okay? Now, uh l of theta is

**[26:40]** log of Z parameterized by theta. Now, this is a product. So, now you can immediately apply the rules of

**[27:08]** logarithms and then uh this will become summation I is equal to 1 to N log of pi beta 1 e to the power of minus beta 1 xi raised to the power of zi plus 1 minus pi

**[27:40]** into beta 2 e to the power of minus beta 2 xi So, this is what So, the log term will become. Now, we can further simplify this by taking uh the log inside. So, now this will be summation I is equal to 1 to N This will be zi log pi

**[28:09]** plus log beta 1 minus uh beta 1 xi plus 1 minus zi into log of 1 minus pi plus log of uh beta 2 minus beta 2 xi Okay?

**[28:39]** I'm using the uh rules of logarithms I assuming that these things are clear. Okay? So, now what is that? Now, we are using EM algorithm. We don't know what is this ZI. Now, in E step, what is that we are supposed to do? In E step, we compute the expected value of ZI. So, now let's start doing it. in

**[29:09]** E step, so we compute compute expected value of ZI for a given XI, okay? So, given current parameter

**[29:48]** estimate. See, just for the sake of clarity, I use another thing what is known as theta old just to indicate that the old parameters and the updated parameter. Update parameters, I leave it as theta. Okay? So, now this is beta one old and beta two old. Okay? So, I know that the equations will become more cluttered,

**[30:16]** but I'm assuming that all of you are capable enough to take care of these things. Okay? So, now what is this mixing coefficient which is there? This is expected value of ZI given XI parameterized by theta old. Now, how does this will be? Now, do we know this expected value?

**[30:44]** Yes. Now, this will be probability of Z I equals to 1 given X I parameterized by theta old. And can we compute this value? Yes, apparently we can compute this value going ahead with the the Bayes rule. This is f of X I given Z I is equal to 1 parameterized by theta old into probability of

**[31:14]** Z I taking the value 1 parameterized by theta old divided by f of X I parameterized by theta old. This is from Bayes rule. Okay? I assume that those basic ideas are crystal clear. So, now what value it will take? Now, this is pi old. What is this? Pi old is because of this.

**[31:42]** Okay, taking the value 1. And Now, given it is from the component 1, now it has to be beta 1 will be the parameter. Now, this is beta 1 old into e to the power of minus beta 1 old X I. This is the numerator term. This is for this. Okay? And this pi old is because of

**[32:11]** this. Okay. Fine. Then this divided by in the denominator I'll have a pretty big term. This is pi old beta 1 old into the power of minus beta 1 old X I plus 1 minus beta 2 old

**[32:43]** e to the power of minus beta 2 old into x i Okay, fine. Now this is This is the the probability that for a given x i Now, z i is taking the value one. Now similarly Similarly, we have probability that z i

**[33:11]** taking the value zero given x i parameterized by theta old is one minus the coefficient. Okay? And this is same as the expectation of one minus z i theta old

**[33:41]** which is one minus the coefficient. Okay? Now, this completes the E step. We have computed this expectation now which is there. Now let's go into the the Q function now which is there. Okay? This completes the E step. See, in couple of treatments they don't

**[34:19]** use this Q function. They don't They directly They don't name it and then they just go ahead with that. Maybe people whose treatment was not using this term Q function. This is not a standard term which is there. But it is an intermediary step that comes in. Okay, this is uh U of theta given theta old. This is uh expectation of Z given X parameterized by theta old of the log of

**[34:49]** theta. Okay. And this No, this is expectation of &gt;&gt; [snorts] &gt;&gt; Z given X parameterized by theta old. This is log of f of X {comma} Z theta

**[35:19]** given X parameterized by theta old. Now, we know this already. We know the inner term. Okay. Now, this is there is an expectation. The log term we already know. This is a summation I is equal to 1 to n Z I into log pi plus

**[35:47]** log of beta 1 minus beta 1 X I plus 1 minus C I log of 1 minus pi plus log of beta 2 minus beta 2 X I

**[36:18]** Okay. And this is uh using the rules of expectation. So, expectation of summations you can take the expectation inside each one of them. So, what I'll do is I'll turn this whole thing as AI and I'll turn this whole thing as BI. Okay? So, now now then what happens is Now, we can take that expectation inside. So, expectation of

**[36:50]** ZI AI the expectation with respect to ZI Z given X. So, this will be expectation of ZI into AI. We already know this value. Now, this is into AI. Now, similarly expectation of 1 minus ZI into BI. This is uh

**[37:19]** expectation of 1 minus ZI into BI. We already know this also. This is 1 minus of the coefficient into BI. Okay? Now, using these you take all those uh expectations uh inside. Okay? Now you take this expectation uh which is there inside of this.

**[37:49]** This expectation of for each term of this whole thing this is ZI into AI. Uh so, it will be coefficient into AI. This is 1 minus of the coefficient into BI is what you will get. Okay? So, now therefore the whole term will be this whole summation I is equal to 1 to N coefficient into AI which is uh

**[38:20]** log of pi plus log of beta one minus beta one xi plus this into log of one minus pi plus log of beta two minus beta two xi

**[38:49]** You get this. Okay. So now next what? Now you want to maximize this. Expectation step is done. Now this is the outcome of the Q function. Okay, this is what is known as a Q function. People may know it without that name as well. That's perfectly fine. So now what is the next thing that is supposed to do? Now this which is there this fun this term which is there now you have to

**[39:17]** uh go ahead and differentiate with respect to each of the parameters. You have three parameters. pi beta one and beta two. You have to differentiate with each of those parameters and then equate it to zero to get the values. Okay. So now whenever you are going ahead and differentiating now with respect to let's say that you take the pi term so only this will come. Okay. Only this term will come. You you

**[39:47]** take and multiply inside only these two terms will come. Okay. Now if you go ahead with beta one so you get this multiplied by this. And and this. Yeah. And this. Sorry. And this. So these terms now will come. So whenever you're differentiating with this this whole thing since it is independent of beta one so this you will not be taking ahead and similarly with respect to beta two. So now what I'll do is I'll just simplify this for each of the term. Okay? So, now

**[40:19]** Now Now, let's do the the M step. Okay? Now, let's do this M step, now which is there. Now, what is the first thing? This Q function which is there with respect to pi Now, how does it look like? Now, this is summation I is equal to 1 to N. You have this

**[40:48]** coefficient log of pi plus 1 minus of this into log of 1 minus pi. This is what you have. Now, then you have to take this and take the partial derivative with respect to pi of this Q pi. Now, this will be summation I is equal to 1 to N. Minus 1 minus

**[41:18]** the coefficient divided by minus of this. And then you equate it to zero. So, equating this to zero, now you get uh equating to zero we'll get uh 1 minus pi summation I is equal to 1 to N the coefficient is 1

**[41:45]** divided by 1 minus pi. This is summation I is equal to 1 to N 1 minus this. So, now let's substitute something. S is summation I is equal to 1 to N. I Now, this immediately goes that the summation I is equal to 1 to N 1 minus of this will be N minus s.

**[42:13]** Okay? Now, if you substitute this, by pi equals n minus s by 1 minus pi. Okay? Now, therefore, if you uh solve this, so now for the new value, you get 1 over n

**[42:41]** summation i is equal to 1 to n this coefficient is uh what you get. Okay? Now, this is for the pi value. Now, let's uh look at for uh beta one. Now, now m step for uh

**[43:10]** beta one. So, similarly, I write this q function with respect to beta one, so it will be summation only the first term will come into picture here. Coefficient into log of beta one minus beta one xi. Okay? Now, then what you're supposed to differentiate with respect to beta one and equate to zero. Okay? Standard terms. Now,

**[43:38]** now differentiate with respect to beta one and equate to zero. Then what do you get? Summation i is equal to into 1 over uh beta one minus xi equal to zero.

**[44:06]** Now, this will be 1 over beta 1 summation I is equal to 1 to n 1 to n this. Okay? Now, therefore, now beta 1 new will be I is equal to 1 to n Okay? This is how beta 1 new will look

**[44:45]** like. And similarly, you perform a similar uh si- thing. Similarly, for beta 2 you get beta 2 new is summation I is equal to 1 to n 1 minus of this summation I is equal to 1 to n 1 minus of this into xi.

**[45:16]** Okay? This completes the the whole thing. Okay? Now, this you go ahead in uh uh what do you call in loop and then you solve for the parameters. Now, EM in an iterative fashion, you go ahead and do it. Okay? Now, this section of tutorial we will stop at this point. So, in which we just saw now, how can you

**[45:44]** go ahead and uh solve in terms of MLE and in terms of uh expectation maximization. See, the overall aim of this is to reiterate or re-emphasize the ideas that uh you might have to an extent uh what do you call uh sunken inside. So, just to bring it to the surface so that we'll be needing these ideas as we go ahead with respect

**[46:13]** to the the generative models. Okay, that is the the sole purpose of this. Now, this section of tutorials is we end at this point. In the next section of tutorials, we'll be working on couple of uh uh neural network based uh numerical problems. I hope uh this section was not too heavy. And then you are comfortable with uh solving the idea. The standard idea is this. Uh you get the likelihood, differentiate uh uh

**[46:40]** with respect to whatever parameter that you have, and then equate it to zero. Okay, this is the standard thing that we have used. I hope that that is perfectly clear, and then you are very much comfortable with using these uh uh uh what do you call these uh tricks of the trades. Okay? Uh that is you'll be able to understand the mixture densities, you'll be able to understand how do you go ahead with uh the standard differentiation and other

**[47:09]** things. The standard ideas which are there. Okay? Now, with this we conclude today's discussion. So, thank you all for uh going through this session. Now, we will be meeting again with respect to couple of numerical problems on neural networks in our next section of tutorials. Till then, bye-bye. Have a nice day. Thank you.
