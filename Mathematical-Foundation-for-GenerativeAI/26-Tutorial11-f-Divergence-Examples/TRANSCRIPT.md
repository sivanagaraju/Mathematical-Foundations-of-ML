# Transcript — Tutorial 11 – f-Divergence and Examples

> **Source:** https://www.youtube.com/watch?v=GjxuVZeMSfE  
> **Channel:** NPTEL - Indian Institute of Science, Bengaluru  
> **Duration:** ~48 min  
> **Note:** Auto-captions cleaned lightly. Minor ASR errors possible.

---

**[00:01]** Hello, guys. Welcome to the tutorial in which we'll be understanding couple of F-diver examples of F-divergence and then the properties of F-divergences mathematically. So now, let's in our lecture section Professor Pratush sir would have defined F-divergence. I'll redefine it and then we will prove all those properties and then we will take couple of examples and then we will say that we will

**[00:30]** mathematically prove that why the KL divergence which is there is not a distance metric. Okay? Now, this is the overview of what we shall be doing in this tutorials. So without any further delay, let's proceed with understanding in-depth the properties of F-divergences. So F-divergence, let's define it. So F-divergence, the definition goes

**[01:02]** like this. Given two probability distribution functions So we have been given distribution functions. Okay? Given probability distribution functions with the

**[01:34]** corresponding density functions. Okay? See, we have the distribution functions and then their corresponding density functions. So what are the distribution functions that we have? We call it PQ which we have and then their corresponding density we represent it by small case uh P and Q. So now then then the F-divergence

**[02:08]** between them the F-divergence between these distribution functions between them is uh DF between the distributions P and Q is integral over the support Q of X

**[02:37]** F of PX divided by QX DX. So QX which is the likelihood in terms of Q and then the likelihood ratio which is there now we are considered here. Okay. So now again what are the properties now the F of U should be in R plus to R. Okay. When I say R plus I'm assuming zero is included here. So that's the

**[03:07]** what do you call uh assumption that we have zero is there. So and then this function is convex. And then it is left or lower semi-continuous. F of one is zero. Okay.

**[03:34]** Under these assumptions okay. Now let's proceed. This is the definition of the F-divergence. Okay. So now let's first take now F of one is equal to 0. Okay. Let's first take that into consideration. Okay. So, now f of 1 is equal to 0.

**[04:03]** How why is it needed? Let's look at now. Now, assume P the distribution function P is Q

**[04:30]** so that the likelihood ratio P of X divided by Q of X uh is is well defined. Okay. Is well defined whenever needed. Okay. So, I'm assuming that all of you are comfortable with these terms. So, whenever needed. So, now let's represent this likelihood

**[04:59]** ratio which is there. So, let's consider this likelihood ratio. Let me call it as R of X is uh P of X divided by Q of X. Okay. Now, this is the likelihood ratio that we are considering. And then Now, we are already under the assumption

**[05:24]** that F is a convex function. Okay. So, F is convex function and we can use Jensen's inequality. Now, remember Jensen's inequality? We have defined Jensen's inequality in the review of probability.

**[05:52]** Please go and look at it. Or another thing that you can look into is uh if you are not comfortable with the definition of convexity or Jensen's inequality, there is a very uh beautiful Wikipedia pages that are there on Jensen's inequality and then the proof for Jensen's inequality. In our earlier review of probability tutorials, we have only defined the Jensen's inequality, Markov inequality, or Chebyshev inequality. We had just defined them.

**[06:20]** So, we haven't looked at any of their proof assuming that uh the viewers of these uh uh videos are equipped enough with those proofs. So, if not, so you can uh at least for Jensen's inequality, there is a very beautiful uh Wikipedia page uh which is there. Uh please, you can go and look at the proof over there. And then even uh for what is a convex function, so if you have questions, you can visit that Wikipedia page and clear yourself the

**[06:47]** ideas. Okay? So, now we can use Jensen's inequality. inequality. Now, how then please be under please be very clear that the the f-divergence that we have defined is actually an expectation. Okay? I presume that by the definition

**[07:16]** it is pretty much clear that the f-divergence that we have defined is a expectation with respect to Q. Okay? So, I'm assuming that and then uh f is a convex, so we can go ahead and use the Jensen's inequality definition. So, now now because of that, how does the definition will turn out? See, the expectation

**[07:46]** with Q, then you have this f of R. This is what we have. Now, because of the Jensen's inequality, it will be less than or equal to f of expectation with respect to Q of R. So, why? Because we are assuming f is a convex function. So, this particular idea holds over there. So, now

**[08:16]** Now, let's try to understand now what is this Let's try to uh evaluate what is this expectation of this likelihood ratio that we have. Okay. Now, what is this? Now, this is integral Q of X into What is R? P X

**[08:44]** divided by Q X D X. Okay. So, now this will be integral P X D X. See, P X is a valid density function. So, therefore, its integral has to be one. Now, because of which, now what will happen? This expectation of the likelihood ratio will become one.

**[09:11]** So, now, I think from here it is pretty much straightforward. So, now from here, so what is uh the f divergence with respect to this Now, this is This is expectation with respect to Q of f of R. This is clear. Now, by Jensen's inequality, now this will be less than or equal to f of

**[09:44]** expectation of r. And this is equal to f of 1 from our earlier thing. And then, so this will be zero. So now, now because of the assumption that f of 1 has to be zero, see what happens is my the divergence, the f divergence that we have, will be always be greater than or

**[10:13]** equal to zero. Okay? This is a the first part. Okay? The f divergence which is there is always the greater than or equal to zero. Fine. So now, let's proceed with the the next one. The next question that we want to ask is now when will when will be the

**[10:48]** divergence be zero? Under what conditions will this divergence be zero? Okay? So now, under certain conditions suppose the f which is there, the function that the f that we are considering is uh strictly convex. If it's strictly

**[11:14]** convex, again, so for the definition, please visit the strictly convex at u is equal to 1. Okay? Now then, the equality in Jensen's inequality, see remember Jensen's inequality now has a greater than or equal to symbol. No, I'm asking when will be the equality holds

**[11:43]** is what I'm looking into. Okay. So, the equality in the Jensen's inequality only when the the random variable that we have under consideration, now which is R of X, okay, which is there, is constant

**[12:15]** Q almost surely. this will hold. So, now now only in that cases now the F-divergence now between the distribution P and Q, it will be zero. Now, what does this mean? Okay. Now, what do we mean by this? So, that is now when will be the R of X be constant? So, let's assume that it is

**[12:44]** a constant and proceed. So, that we will be able to understand with respect to that. Okay. So, now now this implies now this uh this implies the likelihood ratio P of X divided by Q of X which is there is a constant. Okay. Is a constant for Q almost surely.

**[13:19]** Now this should hold, but we already know that we know that from our earlier discussion we already know that the expectation with respect to Q of this likelihood ratio, which is P of X by Q of X, that we have, which we have represented by R, okay. P of X divided by Q of X which is there, is one.

**[13:47]** Now, this therefore C equal to one. Now, what do you mean by C is equal to one? So, that means that P of X divided by Q of X is equal to one Q almost surely. So, now what does this mean? So, which means uh which means So, P of X is equal to Q of X

**[14:19]** almost everywhere. So, now therefore the divergence between the distribution P and Q will be zero only when P is equal to Q. Okay?

**[14:44]** Now Now, this now we will be able to understand the two conditions that we have put for the F divergence, now which are Now, what were the condition if you remember? DF between uh two distributions now has to be always greater than or equal to zero and then

**[15:12]** DF of two distributions is zero only when Okay? Now, we have looked at both of these properties, and then we have mathematically proved. So, when will these conditions hold? Okay?

**[15:41]** So, let's proceed. Now, this is This is regarding the conditions of the F-divergence. So, now let's go to the couple of examples of F-divergences. Now, let's look at Examples of F-divergences. Now, in our

**[16:12]** uh lecture, now sir has already described one F-divergence, which is there, which is uh the chi KL-divergence. So, now let's start with that, and then see. Okay? Now, what is the definition? Again, let's recall the definition. The divergence uh between uh two distributions P and Q, now it is uh Q of X

**[16:40]** F of the likelihood ratio P of X divided by Q of X DX. Okay? So, now uh Now, let's change these F-functions and see. Let's use different F-functions and see what are the different divergences that we get. This is the the standard F-divergence formula. Okay, this is a class of divergence metrics. Okay?

**[17:08]** So, now let's look at the first one, now which is the first one. Now, the we choose here, in this case, we choose f of u, where u being a dummy variable, u log u. Let's choose this as the f function and then see what will happen. So, in that case so the divergence between p and q is integral

**[17:37]** q of x This is there. Now, we want to replace uh f with u log u. Now, what is u? U is the likelihood So, now then what will happen? This is px divided by qx log of px divided by qx dx.

**[18:03]** Now Now, I think uh the remaining things all of you are capable of uh simplifying px log px divided by qx dx. And this is uh what is famously known as the the KL divergence. Okay. Between p and q. Okay.

**[18:30]** And I just want you to remember that what is the relationship between this and the MLE estimate that we have done earlier. Okay. Maximum likelihood estimation that we have done. Now, can we look at the maximum likelihood estimation as minimal KL divergence estimation. Please uh take it. Now, if you have taken any rigorous course in first entry-level machine learning course, this should have been

**[18:59]** the first proof that you have seen. Okay. So, now this is the first f divergence that we have considered. Now, let's consider another the second choice for f divergence. Let's choose this F of U to be minus log U. Okay? This is what we have chosen. So, now

**[19:27]** then, what will happen? The F-divergence now between uh P and Q, now it is integral Q of X minus log of PX divided by QX DX. Now, from here I assume that it's pretty much crystal clear. It is

**[19:55]** Q of X Use the properties of log. log QX divided by PX. Okay? DX. Now, this is DKL between Q and P. And this is what is famously known as the reverse

**[20:22]** KL divergence. Okay? See, the choice of the F is leading to us to different divergence measures. Okay? This is the the reverse KL. Now, let's take the next one. The next one that we take Let's choose

**[20:53]** F of U to be half into the absolute value of U minus 1. This is the choice of the F function that that are taking. Now, then the F divergence is equal to integral of Q of X half of PX

**[21:21]** divided by QX minus one into DX. This is uh integral of Q of X half PX minus uh QX divided by PX DX. Okay. Oh, sorry. Uh in the denominator there is QX. Sorry. Sorry. It's not PX. In the denominator it is So, now uh see Q is a density function.

**[21:58]** So, and you know the properties of density function now because of which uh PX minus QX. Okay. DX. Now, this is what is the total variation distance between P and Q. Now, please remember I'm using the term distance. I'll justify the

**[22:26]** uh use of the term distance now maybe in the last part of the discussion of this tutorial. Now, this is the the total This is the total variation distance. Okay. Now, this is the choice that we have taken for the F function that led us to the total variation distance. Okay. Fine. So, now let's choose one more.

**[22:57]** The famous one which sometime down the line you will be seeing Now let's Let's choose the f of u to be half u log 2u / u + 1 plus half

**[23:25]** log 2 / u + 1 Okay. Now this is a choice of the f that we have. So now let's look at it now. Now the u which is there, now it is uh px / qx. Now let's take each of the term and then uh let's do it. Uh First I'll just look at the term. So this term

**[23:53]** now will be half px / qx into log of px divided by px + qx. See, I think this is pretty much clear. Now you have u + 1 u + 1. Now u is px / qx. So you qx. So it will

**[24:23]** be px + qx / qx and in the numerator also you have px / qx. So I have cancelled it. So I'm assuming that these simple arithmetic operations you should be able to do it. Okay. So now then this [clears throat] is the first term. Now the second term now will be half log of 2 into qx divided by px +

**[24:51]** qx Okay. Now this is what it becomes. Now this is the f of u. Okay, so this is the f of u. So now and then you have a q of x outside. Now let's look at the divergence. Uh So now d f of p q Now this is integral q of x

**[25:19]** f of p of x divided by q of x d of x. Now you can immediately see here that you have a q x here. Now once you replace this whole f of u here there is a q of x that will go inside. Now therefore you can rewrite this whole thing into

**[25:48]** half integral of p of x divided by p of x plus q of x by 2. Okay. This

**[26:16]** into d x. Okay. Plus again half q of x log of q of x divided by p of x plus q of x divided by 2. Now now this is pretty much evident that

**[26:45]** now this is KL divergence now between two distributions. This is pretty much evident. So this if I represent this whole thing, let's say with some mx, if I represent these things with respect to some mx, so I can write it. So, now this will become, let's say, m is a p plus q by 2. And then we can write the small mx,

**[27:15]** which is px plus qx by 2, okay? While calculating the likelihood ratio. Now, therefore, now what will this will be? Now, this is half into integral of px log px by mx dx plus

**[27:43]** half Oh, sorry. There There's sorry. There There has to be an integral here. I may be missed an integral here. So, since I'm writing it as two integrals, I missed an integral here. So, integral of qx and then there has to be a dx here, okay? qx log qx divided by

**[28:09]** mx dx, okay? Now, this is half DKL between p and m plus half DKL between q and m. This is what is famously known as the

**[28:44]** Jensen-Shannon divergence between p and q, okay? Now, if you remember Sir explicitly took this example. Now, what is the difference now in terms of KL and reverse KL. Now, what will happen? And what points are given more weightage in KL divergence? What points are given more weightage in reverse KL? And then he explained that we need a combination of that. Now, which is nothing but

**[29:13]** Jensen-Shannon divergence. We explicitly took this example in the class. I hope you remember that and then you would be able to understand it. Now, with this in a much more clearer manner. Okay? Now, these are a couple of examples that we can look into. Now, there are So, as you remember there was a very nice joke that was discussed that Now, we have A-GAN, B-GAN. So, all

**[29:42]** the alphabet GANs are there. So, the the point that was made during that, which was emphasized during the lecture class also Now, which is Now, change the F-divergence. So, you the the the F function Now, you'll have a different divergence measure, which is there. Now, because of which we get different classes of GANs that we get. Okay, what is GAN? We will look at it up sometime down the line.

**[30:10]** So, now in the next part, now what I want to emphasize is We made a very interesting remark that the di- the KL divergence, now which is there is not a distance metric. Now, why did we make such a statement? What was the intention of that statement? Now, what do you mean by a distance metric? What are the properties a distance metric

**[30:39]** should satisfy? Now, let's look at it now. Okay. we discussed so we made a statement that KL divergence is not a distance metric. So, let's see why. Now, why is DKL not a distance metric? is not a distance metric.

**[31:10]** Now, let's look at it. Why is it so? See, any distance metric which is there has to satisfy four properties. Okay. Now, we will define each of those four properties and then we will see that what properties are not satisfied by the KL divergence. That's what we shall do. Okay. So, now let's take

**[31:55]** each of those properties and then we will see. Suppose the distance between two objects P and Q is D of P {comma} Q. Okay. then

**[32:26]** D of anything is a distance metric is a distance metric if and only if if and only if it satisfies these four properties. It satisfies this property.

**[32:58]** Now among that, let's take the first property. Now which is non-negativity. So what do you mean by non-negativity? So the distance between any two objects P and Q has to be greater than or equal to zero always. And we have just in the earlier case we just proved it that the KL divergence

**[33:24]** satisfies this. KL divergence satisfy this property. So now out of four property, one property is satisfied by KL divergence. Okay. So now let's take the the second property. Now which is there. The

**[33:50]** identity of indiscriminables. Now what do you mean by this? Now the distance Okay. So we have to use the distance metric between the objects is zero

**[34:21]** if and only if these two objects are same. Now we already know this. Now even the KL divergence satisfy this. Okay. So, the DKL satisfies this. Now, out of four properties, the two properties that we have seen Now, both the two properties are satisfied by KL divergence. So, now let's go to the third property.

**[35:00]** The third property, now which is the property of symmetry. Now, what do you mean by the property of symmetry? Now, the distance between P and Q should be same as the distance between the objects Q and P. Does KL divergence satisfy this? Okay? This is not satisfied. Okay? We already We trivially know this from our

**[35:28]** earlier example. The forward KL and the reverse KL are two different things. Okay? KL divergence So, know this. So, the DKL, now between P and Q is not same as the DKL between

**[35:57]** Q and P. Let's take a very simple uh uh numerical example to ensure this case. Now, let's consider some Bernoulli distribution. Okay? Consider P, and now which is uh .1 Q, which is .5 and 0.5. See, these are masses. Okay? So, I'm

**[36:25]** uh I'm using So, why let me write it properly okay What is the issue with that? It's Let this be P and this is Q. Fine. So now Now then in this particular case now what will be the DKL? The DKL will be the DKL between P and Q

**[36:53]** will be 0.9 log of 0.9 divided by 0.5 plus 0.1 log of 0.1 divided by 0.5. This is approximately 0.368. Okay. I'm assuming that all of you are very much comfortable in doing this. Now then what is uh

**[37:22]** DKL of Q and P? Now this is I have already computed this. This is 0.511. This is straightforward simple numerical example not to show that see the DKL of P and Q is not same as the DKL of

**[37:52]** Q and P. Okay. This is pretty much uh evident for us. So now Now why is this the reason? No because of the overlapping points so that happens now that was discussed earlier in the class. Let me just make a point. Okay. So now

**[38:20]** if you consider the DKL between uh Q Now, this is expectation with respect to P log of P by Q, okay? I'm abusing notations. I'm assuming that all of you are able to understand my

**[38:50]** abuse. So, what does this mean? Now, this is where P is large matter a lot. It's where P is small. See, when I say P, I mean So, the likelihood that is being

**[39:19]** computed, okay? Hardly matters. Now, but what will happen when you take DKL between uh P and Q, what is that we are asking here? Now, we are asking how surprised Now, it's kind of average of that, okay? How surprised is the

**[39:50]** How surprised is the model about actually occurs. Now, on the other hand, if we take DKL between uh Q and P, now this is how much

**[40:24]** probability does the model assigns where data Now, from this explanation, it should be straightforward clear that Now, why do we need a combination of both forward KL

**[40:53]** and reverse KL? Okay. If we want to sample, now we want both of these things. Okay. So, therefore, we want a combination of this. Okay. Now, how do you get this combination is by Jensen-Shannon divergence. Now, what is Jensen-Shannon divergence equivalent to? Now, hold on. Now, we will get to that uh &gt;&gt; [snorts] &gt;&gt; in our discussions. Now, therefore, from the what do you call uh from understanding of these equations itself, it is pretty much clear that uh these represents two different things.

**[41:23]** Now, therefore, uh by It's very natural that symmetry doesn't hold. Okay. Now, this is one of the straightforward reasons Now, why the KL divergence, which we have, is not a distance metric. Okay. Now, see, this is the third point, which is symmetry. But, if you remember, it has actually four points, no? Now, what is the fourth point? Now, that will be our next question. The fourth point that needs to be looked into

**[41:54]** is the idea of triangular inequality. The fourth point is Okay. See, roughly, what does triangular inequality means? See, you want to uh So, let's say that you want to go from P to P to R. Okay. So, the distance that is there from

**[42:22]** directly from P to R and then the distance that you take from P to Q and Q to R. Now, the distance directly from P to R has to be lesser. Okay. So, if you have a point P here, you have a point R here and then you have a point Q here. That is why it is called as the triangular inequality. Now, what is that being said here is the distance for P and R Oh, sorry. It has to be less than.

**[42:53]** Sorry. Not greater. Okay. Yeah. So, this is Jensen's inequality's Prabha. Okay. Yeah, I'm writing it the other way around. So, it is less than P of plus distance between Q and R. Okay. So, now let's look into it. P to R

**[43:37]** can never be can never be longer than Q. Okay. Now, this is the idea of triangular inequality. Now, let's see whether this is satisfied. Now, let's take uh P again some Bernoulli I'll write it small P. Okay.

**[44:07]** Let's p as uh 0.1 {comma} 0.9 and q Let me say 0.5 0.5 and r which is 0.9 0.1. Now Now, what are the uh DKLs that we want to compute? Now, DKL between uh p and r. Uh that is uh

**[44:35]** approximately 1. 758. Now, DKL between uh p and q is approximately 0. 368. And DKL between uh q and r is uh approximately 0.879. Now, from this, it is uh

**[45:05]** directly clear that 1.758 Now, which is from p to r is greater than 0.368 + 0.879. Uh sorry. 0. Okay. Sorry. So, sorry. Sorry. Sorry. I just Sorry. Sorry. Sorry. This is not 879. Now, this is uh 0. 0.511.

**[45:33]** Therefore, whose sum will be 0.879. Okay. Yeah. 0.879. Now, this is violating the triangular inequality. So, which violates the triangular inequality. Now, therefore the KL uh which is there is not uh what do you call a valid distance metric. Okay. Now, and one more thing is you

**[46:01]** cannot have a kind of this geometrical interpretation for KL divergence. That's another reason. Now, this is what we wanted to conclude. I just want to give you a very small exercise. Now, is uh a valid

**[46:29]** distance metric? Is total variation distance, now which we have just looked at earlier, is it a valid distance metric? Okay. Now, from the question itself it has to be clear. But, what I want you to do is just go through all the four steps and compute the things and convince yourself, yes, total variation distance, which is there, is a valid distance metric.

**[46:58]** Okay. So, with this, we conclude today's discussion. So, wherein which we understood the ideas, the properties of F-divergence. And we understood them in mathematical terms. We proved all those things and then we took couple of examples of KL divergences. Now, by varying examples of F-divergence, KL divergence was one. So, we took different F functions

**[47:25]** because of which which led us to different uh uh divergence metrics, like F-diver uh sorry, KL divergence, reverse KL, total variation distance, Jensen-Shannon divergence. So, we came across these different divergence metrics. And then we understood why is KL divergence, which is there, is not a distance measure. Okay. So, these are the things that we have looked at in this tutorial. I hope it was helpful and complimentary to the points that were discussed in the

**[47:55]** lectures. Now, we will look at uh the next section as we go along the weeks. Thank you all. I hope you enjoyed it. Meet you next time. Thank you. [music]
