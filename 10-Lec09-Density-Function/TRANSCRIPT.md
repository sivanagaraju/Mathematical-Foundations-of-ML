# Transcript — Lec 09 Density Function

> **Source:** https://www.youtube.com/watch?v=_QrezNPmxDk  
> **Channel:** NPTEL - Indian Institute of Science, Bengaluru  
> **Duration:** ~8 min  
> **Note:** Auto-captions cleaned lightly. Minor ASR errors possible.

---

**[00:05]** Welcome back. So given uh a continuous random variable with a distribution function px density function is a function density function I will denote it with small p and an x to denote that x is the underlying random variable. Okay, this

**[00:31]** is my notation. PX, it's a function from the range of the random variable, range space of the random variable to R plus it's a non- negative function. Remember that density functions are simply non- negative functions. They are not bounded between 0 and 1. I'll tell you I'll talk about the properties of this. a function which is from X to R such that if you evaluate this function at some point X

**[01:00]** this will give you the I'll write it the other way out. So if you take the distribution function and evaluate it at x, this is equal to the running integral of this [snorts] This is the definition of a density function. Okay. Now by definition, please note

**[01:29]** that suppose you evaluate the density function at a particular point. What does that give you? It'll give you a positive number, right? It'll give you a positive non- negative integer, but that does not correspond to probability. Please remember this. It's a trivial mistake that lot of people do. Evaluating density function at a point is not a valid probability measure. &gt;&gt; Come again.

**[02:00]** &gt;&gt; No, you can evaluate it at a point. See, &gt;&gt; no, it will not be zero. It'll it'll evaluate to a particular value, right? You take the gian density function, plug in a value to x, you will get some value, &gt;&gt; huh? Yeah, probability is zero. That's what I'm trying to say. See, you can evaluate, see two different things. Density function is a function. You can evaluate the density function at a point, but that does not give you a probability. That's all I'm trying to

**[02:28]** say. If you integrate the density function over a range by definition it it is a valid probability measure because integrating a density function over a small range corresponds to a distribution function and distribution function is a valid probability measure because it's a push forward measure but evaluating the density function at a particular point is not a probability measure. I mean one trivial example that that I always give people see when I talk to people and conduct interviews etc. Um this is one

**[02:58]** trap that I throw towards people. They'll say this thing goshian density it's pretty easy right I mean you can actually it's bounded bounded and therefore value can be interpreted people interpret this as probability. So then I'll ask them to write down the density function for a uniform random variable that is uniform between 0 and half. What is the density function for a uniform random variable that's uniform between 0 and half? It is two. So now it's always more than one right.

**[03:28]** So how can you interpret that as probability? If you evaluate the uniform density of uniform random variable between 0 and half, it'll evaluate to two at all points. But probability measure if you remember is a function that is bounded between 0 and one. And this is one you know example where you can clearly show that the density evaluating density function at a point does not give you valid probability measure but integrating it would. Okay. However, the value or the eval eval evaluation of

**[04:00]** the density function at a particular point is called a likelihood. This is the nomin that people use. Okay. Anyway, I I'll use that nomin later. The reason I define density function is most algorithms okay operate on density functions not on distribution functions just for mathematical ease. There's no other reason. And all of you already know that not all random variables can have density functions. So there can exist random variable where the density

**[04:29]** functions are not even defined. See this is for the case of a continuous random variable. For the case of a discrete random variable evaluating density function is a valid probability measure. Okay. You actually for a discrete random variable you can interpret the you know the the density function. It's it's not called the density function. It's called the probability mass function. Evaluating the probability mass function at a particular point is a valid probability measure by definition. Not

**[04:58]** for the continuous that's why I defined it for the continuous random variable. In general, we will I mean all most of our data can be interpreted as continuous random variables. That's why we'll be working with density function. So we have to change our problem. I said given d estimate the distribution function. Right? Now I'll say given D estimate the density function. I'm assuming that all the data that we will be having can be modeled using those random variables for which the density functions are well defined. That's a

**[05:26]** fair assumption to make because most well- behaved random variables will have density functions defined. Okay. So we will work with density functions uh from from now on we will work with density functions. Okay. So all these problems right now we'll be instead of estimating uh the conditional distribution we'll be estimating the conditional densities P of Y given X and P of X and so on. Right? So please note the change in notation. If I write P

**[05:55]** with you know two lines it's a distribution function. If I write it with one line it's the density function. And throughout the course rest of the course from now on I'll be working with density functions not with distribution functions. It's not in the real world [clears throat] there can be sort of [cough] distributions which can't be described in mathematical function. &gt;&gt; No, that's not what I'm asking. That's not what I'm saying. That's a question that we'll be seeing later. Um see all I'm saying is in real world all data

**[06:24]** that we have can be assumed to have valid density functions. That's it. Valid density functions. What I'm talking about is there are random variables for which the density functions cannot be defined. So I'm saying I'm we are assuming that all the data that we have that we get in practice can be defined uh or rather will have or can be defined using random variables can be represented using random variables for which the density functions are well defined that's all. Okay. So if if you people do not know

**[06:51]** what density functions are please read up uh but yeah this is what it is. It's another function that is defined from uh the range space of uh the the random variable x to r plus they're non- negative functions. The range of uh the range space of the density function is always positive real numbers. And the definition of it is that it's a function which would evaluate distribution function at a point. If you consider it running integral between minus infinity to x and this integral by the way is not a single integral. Right? If it's a

**[07:20]** dimensional uh random variable that we are talking about then then it's a integral over rd usual integrals. Okay. Okay. Right. So I think we have now laid the uh foundation that that we that we needed for defining uh the machine learning problems. As I said it's mostly density estimation problems, conditional marginalss and joint densities and so on given samples permits. So how do we do that? Think about it. It's a non-trivial problem to solve

**[07:50]** because okay let me write it down. the challenges with ML. &gt;&gt; We'll see you in the next lecture.
