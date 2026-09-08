# Transcript — Lec 11 Entropy

> **Source:** https://www.youtube.com/watch?v=P6wjLz4dRTs  
> **Channel:** NPTEL - Indian Institute of Science, Bengaluru  
> **Duration:** ~17 min  
> **Note:** Auto-captions cleaned lightly. Minor ASR errors possible.

---

**[00:03]** So welcome to this uh new class on mathematical foundations of machine learning. A bit of recap. We last time cast the problem of machine learning as that of estimating the distribution function. And there's a disclaimer that I wanted to convey to you today that whenever I speak I always say that we are dealing with distributions because all definitions and the sense of probability and so on are are only valid with distribution functions but when I write

**[00:33]** and when I define the math equations I'll write the density functions assuming that there is a onetoone relationship between both of them which is by and large true. So but remember that every definition that I write based on density functions has an equivalent definition based on the distribution function and vice versa. Uh I say that we are estimating distribution functions because objectively that is what we are doing right because that's what we want to do

**[01:01]** because that's our probability measure but algorithmically and mathematically we will be estimating the density functions. So please remember this throughout the course. Okay. So we cast the problem of machine learning as that of estimating the underlying distribution function given samples from it. Okay, I also gave you a recipe of how this is done. So there was a three-step recipe that you start by assuming a parametric

**[01:30]** functional form on the distribution that you want to estimate. Okay. And there was a slight mistake last time. One of you actually noticed it. See here there is a dimensionality mismatch. If W1 and W2 both of them are RD. So W1 transpose X is a scalar and W2 if it's in RD then uh a scalar cannot be added to a vector. Correct? So W1 is in RD

**[02:05]** and W2 is a scalar otherwise dimensions will not match. Okay. Yeah. So assume parameterize P theta using some parametric family of functions and estimate their parameters. How do you estimate the parameters? You define or compute the distance metric between the true distribution and the distribution that we have assumed. And my nomature will be that P theta which is uh the distribution that we have assumed on

**[02:32]** reality. This is what I'll be calling either as model distribution or model Okay. Or the assumed distribution. So I'll be using those words. So P theta is the model distribution. So there is some distance metric which takes uh two distributions and maps it to a positive real number. We find or estimate the parameters of uh the parameters theta by solving an optimization problem. That would say that the optimal parameter theta of this

**[03:00]** model distribution is the one that minimizes a divergence metric between the true distribution and the model distribution. So that is the general recipe is what we saw. Now in the rest of the course by and large we will be looking at different ways to solve this problem. What distribution functions what models how is the what is the uh divergence metric that we'll be using and what is the way to solve this optimization problem. We'll be looking at multiple ways to do it. Okay. So with that recap, let us look at now

**[03:33]** defining. So one of the key factors of this recipe is that we need to first find a way to quantify the distance between distribution functions. Okay? Cuz that is what we are trying to optimize over. Right? So our parameters are chosen so that a distribution a divergence metric between a pair of distributions are distribution is minimized. So we'll have to first define or come up with a notion that would

**[04:01]** quantify how far or close a pair of distribution functions are. So let us do that now. Distributional divergences. So there are multiple ways to do this. In fact, this is a topic in itself. How do you quantify distance between distributions? There's one large family of distributional divergences by name f divergences uh which will be studied in my next

**[04:28]** course right when we study generative models. I will take one particular example of the f divergence that is often used in in machine learning and we will use that as diver divergence metric and move on in this course. Okay. Now objective is the following. Objective Distribution functions defined on the

**[05:06]** same sample space goes without saying. or quantify This is our objective. So we are we are

**[05:35]** given two distributions that is defined on the same sample space. We want to quantify how close or far they are. Okay, to do this we need some background in information theory because uh historically speaking people defined distances between distributions using information theoretical approaches. So we look at some of the interesting ideas of information theory and then uh we will see how to use those measures to define distance

**[06:04]** between distributions. Okay. Now given or it is in the sigma algebra or event space. the information content in A

**[06:39]** as negative of log of probability this particular event A. So this is called the information that is associated with this particular event A. Why is it? Now if you interpret probability of an event as the likelihood of appearance of that

**[07:08]** particular event. Okay. What do I want to quantify here is that I want to quantify what is the amount of information that is being conveyed by that particular event. Now think about it intuitively. If I tell you something that is obvious, do you think am I conveying something uh or or is that particular statement of mine is it conveying a lot of

**[07:34]** information or it's not conveying a lot of information? It is not cuz if I tell you that the sun rises in east then I'm pretty much telling you nothing. Okay? But if I tell you that uh tomorrow you will win a lottery of 1 cr rupees or something right. So that's a lot of information. So now events that are more likely convey lesser information and events

**[08:02]** which are less likely convey more information. So the measure that we come up with have to be designed in such a way that more the likelihood is of a particular event the less the metric has to be. This is one thing. Second thing is if you extend this let's say that we know that even sample space omega is a is a member of f. Now what is the information that is conveyed by the entire sample space?

**[08:34]** It is zero information. Why? Because I'm saying that something will happen. If I say something will happen of course, right? I mean I'm not I'm not saying you anything. So that has zero information. We have to come up with we have designed the metric in such a way that the sample space should convey no information and the null event have to convey maximum possible information. Correct? If I say that a null event is going to happen then it has to have infinite information. The other requirement that we have is if there are

**[09:04]** two statistically independent event the information combined information conveyed by them combinedly have to be equal to the the sum of the information combined individually. Okay. If you look into these three requirements then negative of logarithm of the probability of that particular event satisfies all this. It's very easy to s look at because if a is the sample space then p of a is one p of sorry if

**[09:34]** if a is sample space yeah if a is sample space and p of omega is 1 negative of log of p of 1 is zero. So if it is the null event then P of null is zero and this will become infinite no very large value and if there are two events A and B if they're independent uh the the probabilities will get multiplied and the log of them will get added and there's a negative sign here because more the the

**[10:03]** likelihood is the lesser the information has to be. So I of A is the negative of log of the probability associated with a particular event. This is also called surprisal in the information theoretical uh language. This is called surprisal of an event. Okay. Now this is for one particular event. Okay. Suppose I want to quantify. See why was this important historically?

**[10:32]** I'll tell you all these ideas came because the uh the objective was to transfer information from one place to the other communication. Now when you communicate there was no band there was the the bandwidth was far below today's uh numbers given a very very limited bandwidth they had to convey or they had to communicate the maximum possible information. Okay. So

**[11:01]** the question was given that I have bit constraint I have only these many bits that I can uh transfer what should be the best information that I have to convey in such a way that my loss is minimized. So this was the question that communication engineers came up with and to answer that they had to come up with these ideas. So given a particular distribution now tell me what is the amount of information that it that it uh embeds onto it. Okay, this is for one

**[11:29]** particular event. Suppose I'm looking at uh looking for looking at an entire distribution. For now, let's assume that uh we are dealing with discrete random variables. It is easier that way. Then what is the way to quantify the information associated with an entire distribution? So we know how to quantify the information associated with one particular event. If I have to do it for an entire distribution, how do I do it?

**[11:57]** What is the most intuitive way? So suppose X is a discrete random variable. Then the information or surprisal probability of x i. So I can write it as

**[12:42]** probability of x i because this is a mass function. It's a discrete random variable and mass function is a valid probability measure. log of P of X I is the surprise assoc associated with the event denoted by the the value X I. Now I'm interested in doing it for the entire distribution. How do I do that? So what I do is that I take an average of all of this. Okay. So now what is the notion of average in uh in terms of

**[13:13]** probability theory? You take the expectation. So you look at the expected value of this particular function with respect to distribution px. So this is the average information that is contained in a distribution px. Okay. Because it's a discrete random variable. This is given by this. This summation is over i. What is i? i is denotes the indices all the

**[13:41]** values that this discrete random variable takes. You have minus x i log probability of x i make sense Right? So if you're looking

**[14:14]** at one particular event and you you know how to quantify the surprisal or information associated with one particular event and if you want to do it for an entire distribution, you just take the average. The notion of average in uh in in from from a statistical sense is that you look at the expected value of uh that particular function. And uh I'm I'm I'm assuming that all of you know that expectation of a particular function of a random variable if when you take the

**[14:44]** expectation you multiply it with uh so I think I made a mistake. No. Yeah. This should be pxi. Sorry. Yeah, this because

**[15:09]** expectation of a function of a random variable with respect to a distribution X is given by sum of PH * FX correct over all X this is the definition H okay now this became the average information that is contained in this entire distribution this has a name. Do you know what this is? This is the entropy associated with distribution px. The average surprisal that is associated

**[15:52]** with a particular distribution is called the entropy associated with the distribution. [snorts] We all know that lost or thermodynamics tells us one of the fundamental laws of thermodynamics is that the entropy of the universe keeps on increasing. So what does that translate to? It translates to saying that universe never stops surprising us. Why is that of importance? Now there's a

**[16:22]** lot of fear-mongering that AI has come up right uh now it'll take all the jobs and all that right true it will happen a new world order will emerge but because the fundamental of thermodynamics says that the entropy of the universe keeps increasing there will be new problems that would emerge because entropy is increasing just that the world order changes so anyway every time some significant

**[16:52]** technological breakthrough happens. What happens is a new world order will emerge but because of the law of thermodynamics, new problems will emerge and there will always be scope to solve these new problems and come up with new solutions. The fundamental idea of uh you know human value exchange would not change. Meaning society will reward people who will create value. The notion of value changes as a function of time. What's valuable today is not valuable tomorrow.

**[17:21]** Okay. But because the entropy of the universe is always increasing, the only way to survive is to adapt. That's what evolution also tells you. No, it's not the smartest that survived. It is not the strongest that survived. It is the fittest that survived. The definition of being fit is that you can adapt. Okay. Anyway, so this is entropy of a particular distribution. Uh we will stop here and continue in the next lecture.
