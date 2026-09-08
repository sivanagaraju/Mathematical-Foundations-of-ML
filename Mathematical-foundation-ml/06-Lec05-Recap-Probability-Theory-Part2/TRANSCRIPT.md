# Transcript — Lec 05 Recap of Probability Theory Part 2

> **Source:** https://www.youtube.com/watch?v=R69wew8RrPo  
> **Channel:** NPTEL - Indian Institute of Science, Bengaluru  
> **Duration:** ~21 min  
> **Note:** Auto-captions cleaned lightly. Minor ASR errors possible.

---

**[00:03]** Welcome to this class uh in the course mathematical foundations of uh machine learning. In this class, we will continue with our discussions on the fundamentals of probability theory and relate the ideas of uh vector valued random variables and joint distributions to the problem of machine learning with examples. Last time I give an introduction on why one should look at the problem of uh machine learning from a probabilistic standpoint and we also defined some foundational

**[00:33]** ideas of probability theory including sample space uh event spaces and the probability measures and I I also introduced you to the idea of uh random variables and why random variable is needed and what we work with in practice. are realizations of random variables as they call it. Mathematically speaking, what we have are vectors from the range space or range set of the function

**[01:05]** called random variable. Okay, that's what we measure from. Okay. And whenever we say that there is a random variable and you have some measurements done from a probabilistic standpoint, you have to always remember that there always is an underlying sample space that gave rise to this particular random variable. And uh with a sample space also comes a probability measure which has been pushed forward

**[01:35]** into being a probability distribution function. Okay, which is also called a cumulative distribution function or CDF. You have to always remember this. Okay. So from there now uh when we defined I defined the random variable as a scalar valued function in the sense that you have uh sample spaces and from sample spaces uh from the elements of sample space you map them onto real numbers. Now one can

**[02:04]** generalize this idea onto vector valued random variables wherein the range space of this function called random variable is no longer r but a d-dimensional real space. Okay. So let's define that first. Now the

**[02:41]** range set or the co co- doain set of the random variable of the function See note that I've been repeatedly using the word function for random variable because that's what the object is. Okay. Uh despite the misnomer, the range set of the random the function called random variable is RD which is a d-dimensional real space where D is a scalar.

**[03:23]** Okay, which means that from sample space omega you have the random variable X that is mapping the sample space to RD. So these are also called the vector valued random variables where from the sample space you map it onto RD okay with the D- dimensional real space. Okay. Now this can be viewed from two different angles. One as a function or a

**[03:53]** single random variable that maps your sample space to a d-dimensional real space or one can see this as a collection of d scalar valued random variables. Okay. So to have that view let us first define what is called as you know multiple random variables or joint distributions. I'll tell you why this uh viewpoint is

**[04:25]** uh uh important and how does this how does these two viewpoints converge. Let Sample space corresponding to some random experiment. So there is one sample space that we have. Okay. So define So call them

**[05:00]** x1 and x2. Note that I've been using lower cases now for these random variables. Uh I'll tell you why. There's I'm doing this deliberately. I'll tell you why I'm doing this. Define two functions x1 and x2 which are x1 is a function from the sample space to r. So this is one scalar valued random variable and x2 is another function

**[05:28]** on the same sample space. Remember that or rather note the fact that one can define multiple random variables or multiple functions between two pairs of sets always right. You can define multiple functions between sets. And because you can define multiple functions between sets, there can exist multiple random variables on the same sample space. Okay. Now, corresponding to these, there also

**[05:57]** exists the respective distribution functions, the corresponding distribution functions. No, it goes with B2 because the moment you define uh a sample space, there is a measure. Okay, and the moment you define a random variable, there's always a distribution function corresponding to it. So you have two distribution functions. Okay. Now recall that uh in the set theoretical view for

**[06:28]** probability theory, we can define the probability over set operations of uh events. Now what what do I mean by that? If there are if there exist two events A and B you can define and they have probabilities corresponding to them right the measures corresponding to them we saw by definition this probability measure can also accommodate the unions and intersections of these

**[06:56]** events. So we can define probability of unions and probability of intersections and so on. So which means that if you define two random variables, okay, because they by definition their inverse images always map back to some uh element in uh the event space f and because you can since you can talk of intersections and unions in the event space, there should be a way to combine

**[07:24]** these random variables. Does that make sense? Right? So that's exactly what the idea of joint distributions are. Okay. So probability distributions. Yes. So now P

**[08:03]** X1 X2 okay now you evaluate this at two points correct evaluate this at um yeah for the lack of notation I'll have to write it at right as A and B let's say that you evaluate at that at two points A and B what is this given as this by definition this is the probability this is the measure okay that corresponds to an event E which is

**[08:34]** the inverse image that corresponds to the the intersection

**[09:03]** of um so let's say that E is the event which is the inverse images off under

**[09:46]** X1 and X2 respectively. Make sense? So there are two random variables that we are talking about and inverse images of both of them correspond to some events in the uh the original sample space. Take the intersection of both of them and you get another event. The probability measure corresponding to that particular event is what is defined as the joint

**[10:14]** distribution. Okay, make sense? Now you can extend this idea to n random variables. So it need not be two random variables. You can extend this idea into n random variables in which case so you have a joint distribution between n random variables or d random variables that are all defined on the same sample space.

**[10:44]** Right? So that's exactly what the idea is. So now this the above idea can be extended to D scalar random variables extended to the number of scalar random

**[11:17]** variables. Okay. Now remember that we also said that there can exist vector valued random variables know where uh the function which is the random variable maps from the sample space to a d- dimensional real space. Right? Since we can define d number of uh scalar valued random variables there seems to be a connection between these two. turns out that they

**[11:45]** are exactly the same. Okay, so a a vector valued random variable can be seen as a collection of D scalar valued random variables that are all defined on the same sample space. Okay, that is what the idea is. So for for the rest of the course without the loss of generality I assume that we are working with a d-dimensional vector valued random variable. So when I

**[12:15]** say that there exists a d-dimensional vector valued random variable you know you should know what is happening. So either you can see that as one function [clears throat] from the sample space to RD with the corresponding vector valued uh with the corresponding underlying distribution. Okay. See if it's a if it's a d- dimensional vector valued random variable then uh the uh the interval that we see here happens to

**[12:44]** be the cartition products and we just saw that the last time right it's the cartition product or if you want to go back to the sample space it's you can see that as the intersection of all the events corresponding to the individual scalar valued random variables that could be seen as defining d different functions on the same sample space. Okay. Okay. On top of it, right, once you define multiple random variables on the same sample space,

**[13:11]** you can define know things like conditional distributions since there there exists conditional probabilities. Now I told you the last class, right? whatever applies from the set theoretic point view viewpoint of in for the probability uh theory can be extended to something in the space that is pushed forward by a random variable a function called random variable right so now define [snorts]

**[13:44]** conditional probabilities and conditional distributions corresponding If A and B are two events

**[14:16]** corresponding to a sample space then we all know that you can define what is called as probability of the event A conditioned on B as the probability of the intersection of these two events with respect to the probability of the conditioned event. So this is the definition of the conditional probability. And what does this correspond to? I mean if you want to interpret given that the event B has already

**[14:46]** occurred evidenced by something, what is the likelihood or the probability that the event A occurs? That's the uh way to interpret it. Okay. Now since everything that we do from this theoretical angle can be translated to in distributions you can define what are called as conditional distributions. Okay. So we define conditional distributions as P suppose

**[15:15]** suppose X and Y are two random variables defined on the same sample space. See, please note that here X and Y are generalized in the sense that they can both be vector valued random variables. Okay, from now on whenever I say that

**[15:43]** I'm dealing with a random variable, I always mean that I'm talking to I'm talking about a d-dimensional vector valued random variable where the function is from omega to RD. Remember that. Okay. So there are two random variables. And by the way, the range space of these two may be different in the sense that one can be RD, one can be RK, doesn't matter. You still just have two vector valued random variables. Can define the conditional distribution P of X given Y.

**[16:19]** So evaluate that at X at some X. Okay, it evaluates at some X given Y is fixed to some value Y. Always remember that when we define the conditional distribution, the conditioning random variable is fixed at a value. Okay. And that's why the conditional distribution is always set to be a function of the conditioned random variable because once you fix it, the sample space get shrunken. Okay? And you fix that and then define a

**[16:47]** distribution over it. So always see from now on whenever I write so I don't write this y equal to y from now on I'll just write p of x given y but when I write that always remember that the conditioned random variable is fixed to a value a given value given that this is fixed to a particular value the question that we are asking is what is the underlying distribution so how is this defined so this is defined as divided by

**[17:21]** what is called as the marginal distribution. So this is called the marginal distribution because by definition if x and y are two

**[17:50]** random variables and please note that I'm assuming Assuming that the underlying random variables are continuous when I write this definition. Okay, two random variables. So marginal assuming that this integral is well

**[18:27]** defined. So by the way, does anybody know why this is called a marginal distribution? It's pretty interesting. Suppose these two are discrete random variables. Discrete random variables can be written as a table. The joint distribution of a discrete random variable can be expressed as a table. Right? Now if you look at the evaluate the margins along the margins of that table whatever you get is the marginal distribution for all of those excess. So that's why it's

**[18:56]** called the marginal distribution right it's literally the margin the one that gets evaluated at the margin. So basically what we are saying is that if there are two random variables that are defined on the same sample space if I integrate out one of the random variables or rather I would nullify the effect of one of the random variables by adding uh all the values corresponding to the other the other random variable can take I will get the behavior of only one of the random variables and that's

**[19:23]** what is called as the margin. Okay, now we know what joint distributions are. Now we know what marginal distributions are and what vector valued random variables are. Okay, again as I said this is a very quick primer on the probability theory and the notions that we need for this particular course. I'm assuming that that all of you have done this course rigorously. This is just refreshing your memory. Okay, this is not a full probability theory course by any uh imagination. Okay, so now we know similarly you can define the marginal on

**[19:52]** y as well. So marginal on y can also be defined as you take the joint distribution and you integrate out uh x and by the way when I write this integral y integral x this is not a one-dimensional integral right depending upon what the range space of this random variable is you'll have to if it's let's say r3 you are talking about triple integrals and if it's rd it's a dimensional integral and so on so for notation I'm just writing it as

**[20:21]** integrating over x okay which In fact, this is sort of abuse of notation when I say that see this y denotes a function. Okay, you cannot integrate over a function. You can integrate over sets. So by this I mean that this is the I mean the range of y that that's what I mean. I mean to be precise let's maybe we'll write that. No this is &gt;&gt; these are capital X and Y right

**[20:53]** &gt;&gt; come again &gt;&gt; these are capital X and capital Y &gt;&gt; capital X and capital because they are functions but integration is being done over the range of X and range of Y when you compute the marginals. Okay. Right. Now comes the interesting part. Now you remember our problem, right? Initially our problem was that we have an X-ray and we wanted to know whether that X-ray corresponds to the disease case or non- disease case. You know how are all these

**[21:21]** related to this? Okay. Okay. So now uh let's try to get that example uh grounded in in these uh these under with these notations and languages. Yeah, I'll do that and then we will go to the density functions. Okay, we'll see you in the next lecture.
