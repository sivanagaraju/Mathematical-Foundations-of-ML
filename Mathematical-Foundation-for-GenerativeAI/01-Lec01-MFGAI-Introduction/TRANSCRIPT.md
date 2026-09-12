# Transcript — Lec 01 Introduction

> **Source:** https://www.youtube.com/watch?v=H05WDy9Mngk  
> **Channel:** NPTEL - Indian Institute of Science, Bengaluru  
> **Duration:** ~70 min  
> **Note:** Auto-captions cleaned lightly. Minor ASR errors possible.

---

**[00:02]** Hello everyone, welcome to this NPTL course on mathematical foundations of generative AI. In this particular course, we will be focusing on the mathematical nuances and the underpinnings of all the generative modeling algorithms that power today's AI world. Now as all of you know artificial

**[00:32]** intelligence specifically the generative part of it has revolutionized the way science and engineering is done. So we are now immersed with algorithms that are fueled by the generative modeling paradigm. We cannot now imagine a life without the generative AI. So take for example the algorithms that we

**[01:00]** interact with on a daily basis or the systems that we interact with on a daily basis such as the GPDs and geminis and clouds and so on. All of these are examples of systems that are built using the generative AI. So this is a significant paradigm shift that has happened over the last decade in the field of uh machine learning. What was a dream

**[01:27]** the la during the last decade has now become a reality. So what was a science fiction the last decade has now become reality where machines can actually think, interact and solve problems for us. All these capabilities are possible because we use specific type types of algorithmic design combined with massive amounts of data and

**[01:54]** good training paradigm that have made all these algorithms a possibility. So now there are multiple sources of content that is uh out there that would uh teach you the way these algorithms are built. these algorithms are used and so on. But this course uh I have designed it with a unique objective that if there is somebody who is interested in not only taking the algorithms off

**[02:24]** the shelf and using it for their particular uh use case but to understand what is happening under the hood and even come up with newer algorithms that would uh create new possibilities or solve the existing problems uh in the current ecosystem. then knowing the mathematical details of these algorithms becomes a an unavoidable necessity. Now most of these

**[02:56]** algorithms are designed by mathematicians and engineers who could think from the mathematical first principles and then use those mathematical tooling to envisage new ways of doing things. So now this with this with that preface I designed this course in a way where we look at the mathematical nuances of a broad family of uh generative modeling

**[03:25]** generative models the state-of-the-art generative models so that somebody who completes this course will be equipped with the mathematical arsenal that is needed to either either understand the details of the existing algorithms or to even create new generative algorithms using those tooling. So with that as the uh general introduction uh in the first couple of

**[03:56]** lectures we will look at some of the foundations foundational mathematical tools that are needed upon which uh generative modeling or rather machine learning is constructed and that happens to be uh the probability and statistical toolkit. Now I will just give a broad overview of how one would look at the problem of uh machine learning in

**[04:24]** general and generative modeling in particular from the lens of uh probability probabilistic models and then I will talk about the general recipe that is used in gener most of the generative models in constructing uh the um algorithms that they use. So that is what we are going to do in the first few lectures. So and then moving forward we will look at

**[04:53]** several classes of generative models. So let let us uh discuss about those classes of generative models uh that we will be covering in this particular course broadly. I'll just take the topic name. So the first and the style of this course, the teaching style of this course will be uh the old school way of chalk and talk where I'll be writing down all the equations uh that that we derive. So we

**[05:23]** start from the first principles and then I derive all the underlying equations and I write down each and every equation that I uh you know that I talk about but not uh use a preconstructed set of slides and then we'll build everything from first principles you know that's the uh methodology with which I've uh adopted to teach this course and this course will also have a practical component uh which will be dealt with by

**[05:52]** my teaching assistant Chandan. So he will be walking you through all the programmatic or implementation details of all the algorithms that we will be studying in the uh lecture part of it. So there will be accompanying tutorials for each of the uh lecture content that we do. So that every algorithm that we study in the lecture part will be

**[06:21]** implemented practically using uh the frameworks such as PyTorch that are built on Python so that uh a student would understand how to translate all the mathematical constructs that we have discussed in the lecture part into algorithms that can be used in the uh the real world via implementations. Now broadly uh I will start the course with an introduction

**[07:05]** and then we will go to variational methods. variational methods that are built upon this idea called latent variable modeling. Under these variational methods, there will be multiple algorithms that we'll be looking at. One, we start with the

**[07:33]** mixture models. An example would be the classical gian mixture model which is one of the classical algorithms that are used for generative modeling. We will also look at the variational autoenccoder family which are abbreviated as VAEs. Then

**[08:07]** under the variational methods we will also look at these family of models called the diffusion models. sometimes also called as DDPMS, denoising diffusion probabilistic models. And then we will look at the adversarial family of models of which the

**[08:40]** generative adversarial network which are also referred to as GANs. This was one of the classical one of the early and classical uh or rather early methods in the neural family of generative models that created a lot of

**[09:08]** uh uh buzz around the world and then we will look at auto reggressive models. Within auto reggressive models, you have the transformer-based LLMs that we see today, transformer-based language models or the

**[09:39]** large language models and we will look at another family of models called the state space models or SSM. So these state space models are uh a very alternative uh or a very very attractive alternative to the auto reggressive large language models because of uh uh some of the advantages that they would offer over the uh the auto reggressive models. Then we will

**[10:06]** also look at the normalization flow-based models and so on. So this is the broad family of models that we will be uh looking at. So we'll start with this will this is the course outline. We'll we'll start with uh introduction to probabilistic ML. Then we will look at the variational methods uh which are uh latent variable models the mixture models mixture

**[10:34]** densities variational autoenccoders and its uh variations. We will look at the diffusion models and for the adversarial family we will look at the generative adversarial networks. We look at the family of auto reggressive models which are transformer-based large language models. We will look at state space models and also discuss normalization flow-based models. So this is not an exhaustive list of everything that we are going to cover in this course. But broadly this is uh these are the models that will definitely be covered in this

**[11:03]** course. And then uh at the end of the course we will also look at uh some of the reinforcement learning methods such as RLF and uh the DPO, PO and so on which are used as the post uh uh training methodologies for large language models. So that's the broad uh plan the course plan that that I have envisaged for this course. So with that let us uh start looking at

**[11:32]** the probabilistic way of uh approaching the problem of machine learning. So we will look at the probabilistic machine learning. So perhaps before we continue with the

**[12:01]** the further topic uh I would uh encourage all the students taking this course uh to have the background or rather acquire the background that is required to take this particular course. So if I have to list the prerequisites uh if one has enough background in the probability theory and the random processes that would serve the student

**[12:28]** uh uh a lot in this particular uh course because all the algorithms that we will be looking at in this course uh will be dealt with from a probabilistic uh standpoint. So having a strong prerequisite on on prob probability theory and statistics would definitely help the student. The other prerequisite would be if if the student uh also know a bit of classical machine learning you know where they understand uh the way the

**[12:57]** classical ML algorithms are built such as the linear models and the linear regression and the neural networks and the kernel methods such as SVMs and the graph based methods and so on that would uh help the course help the student in this particular course and a bit of programming background especially in Python uh language is much needed. because all the implementations that we will be doing of these algorithms that we study will be uh based on Python. So even though we will have uh a couple of

**[13:27]** uh initial tutorials on uh uh the libraries such as Pyarch upon which we build uh these entire course having some background on the fundamentals of programming uh would definitely help. So the treatment that we will be uh taking in this particular course for machine learning will be probabilistic. So I assume that the students that would take this course are already equipped with the the fundamentals of the probability theory

**[13:59]** and then build the foundations from foundation starting from there. [snorts] So now what is probabilistic machine learning? The fundamental problem that we encounter in machine learning is that of uh modeling the uncertaintity. Right? So the foundational problem that we'll be looking at the entire machine learning would be that we have to model the uncertaintity. So what do I mean by modeling the

**[14:32]** uncertaintity is that we want to understand the system. You know you can take the example of understanding the uh the human language as a system. So that's what we do in language models. So now there understanding the human language typically whenever you want to model a particular system in engineering you would go to the physics of the physics of the problem which is the uh the foundations of how the uh the boundary

**[14:59]** conditions or the systems are built and then you try to model that system using the known physics or mathematics. An example can be suppose you want to understand how a particular rigid body moves from point A to point B. Then you have to know ideas such as the momentum, acceleration, the position and so on and you measure all of them using a physical device and then build math which is the Newtonian mechanics that has

**[15:28]** deterministic math around it and you can completely specify uh the trajectory of a particular uh rigid body that moves from point A to point B. So that is an example of how you deal with uh problems from the first principles of physics. Now if you are dealing with problems such as uh modeling let's say human language or trying to understand what is there in an image uh which is a natural scenery and so on. Then there are things

**[15:58]** that are non-measurable. You know for instance the idea of whether some person is present in a particular uh image is not is not physical. It cannot be measured. Or another concrete example can be let's say that you have uh uh the problem of classifying whether a given email is spam or not. Right? So in that particular situation the idea of what constitutes a spam and

**[16:27]** what does not constitute a spam is not something that can be physically measured using instruments. Correct? So that has uh a lot of uh perceptual association with the ideas. So whenever you have these kinds of situations where you are interested in measuring things that are abstract or rather quantifying things that are not measurable and abstract, one would resort to statistics or

**[16:55]** probability uh probabilistic way of looking at things where we model the uncertaintity or the non-measurable So an example can be classify Now classify a body of text as let's say

**[17:27]** spam or not a spam. So now this idea of what constitutes a spam or not is is very abstract and non-measurable. So how do we deal with something that is not measurable? So another example is so given an image. So this can be the first example. The second example is so let's say that we are given uh an image. identify a particular person or an

**[18:08]** object from the image. So, so for instance, we might be interested in uh u finding out whether one particular person exists in a given image or if we are looking at let's say a clinical image uh such as an X-ray we are interested in finding out whether a particular tumor is present in the

**[18:35]** object. Now the idea of a person or a particular tumor or or another object being present in an image which is uh which is the measurement of the amount of uh light that a particular surface reflects is very abstract. So what do I mean by that? You cannot measure whether a tumor is present in an image or not. So there is no physical device to measure whether measure the presence of

**[19:04]** a tumor or otherwise in a in an X-ray. So that is a perceptual idea. So now whenever you have to deal with things which are not measurable uh but abstracts what do I mean by an abstract? it is something that is that cannot that cannot be physically measured or rather know which has a perceptual association with it. How do we deal with such a situation is the question. Now it

**[19:32]** turns out that under these kinds of situations where uh it is impossible or rather difficult to completely specify uh a given system using the physics or the first principles. One particular tool that has helped uh engineers and mathematics mathematicians to deal with is to so model these systems via

**[20:03]** repeated observations. So this is something uh that has that has worked. In fact, you know, if you look at uh the uh today's landscape of generative models, it is this paradigm that has made everything that you see uh in in in in today's engineering paradigm around the generative models a

**[20:30]** possibility. Now what what is the idea here? Okay, let's say let's say that I uh that I'm interested in classifying a body of text as spam or not not as spam. uh I don't know what constitutes a spam or not. Okay. But what can I do is I can accumulate hundreds and thousands of body of text. Let's say these are emails. I can accumulate hundreds and thousands of these emails and make an expert label that as spam or not a spam. So now I

**[21:00]** have a lot of instances of a body of text being a spam or not. Now I use that which is uh a a collection of multiple objects or multiple scenarios of what I want to uh build and use that repeated observations to build a system. Okay. So this is the crux of or rather we we can we can do the same thing for the second example as well. Let's say that I'm

**[21:28]** interested in finding out whether a particular object is present in an image or not. Okay. What I would do is that I would collect or rather I will do a lot of repeated observations where I see an image and I see that particular person or I take an image where that person is absent. Now I model this idea of presence or absence of a particular person or an object in an image by collecting enormous amounts of

**[21:56]** observations that I make regarding the system that I'm interested in building. A naive example can be that suppose I'm interested in knowing whether uh a role of a particular dice turns out to be of a particular number to face then the entire process of uh rolling a dice and what side that it turn out to be is a function of multiple environmental

**[22:26]** variables that that are not measurable. In that case, what do we do that we roll that dice multiple times, okay? And make repeated observations and see if we can model the system or the uncertaintity and abstractness that we have using repeated observations that we have made. So this is the central idea okay behind all the probabilistic and statistical methods. at least the frequentest way of doing it

**[23:05]** where if you are trying to model a system where there are uh variables that are not physically measurable then one of the ways to model that system is to try to emulate that system multiple times and then using those repeated observation. See if you can fit a mathematical observation or a mathematical model to the repeated observations that we have made. So that's the crux of the frequentist way

**[23:32]** of the probabistic approach. Now we have to concretize all these ideas. You know it's it's a lot of talk. We have to concretize all these ideas. So how do we do it in uh in in in in probability theory is that we will start with uh the set. So let us uh define. So I'll give you a very gentle and a quick introduction to probabistic machine learning. So I'm just repeating it again. I am

**[24:06]** assuming that the students here are convincent with uh the ideas of probability theory. If you're not, I think I would strongly recommend you to take up a course on probability theory before doing this course because this course heavily relines relies on uh the knowledge of the fundamentals of probability theory and statistics. Now as I said in uh uh the the central crux

**[24:31]** of probabilistic machine learning is that we model the system that we are dealing with using the ideas of probability theory. So let us look at what do what are the foundations that go behind the probability theory that that we would look at that that we would consider to build the algorithms of generative modeling. Okay. So in in uh in probability theory we start with an experiment

**[25:00]** which is also called the random experiment. So there is a random experiment that somebody is conducting. So this random experiment can be um tossing a coin or writing

**[25:45]** an email and so on. So it's the choice of a practitioner or or an engineer to look at every systemic design that we have from uh the lens of probability theory. So every kind of system that that we encounter can be modeled as a random experiment. Now this random experiment is conducted

**[26:18]** several times is the underlying assumption that we have uh in in probability theory that we have a random experiment that is conducted multiple times. And what does it give? A random experiment outputs So all the possible outputs of a random

**[26:47]** possibilities of uh the random experiment are enumerated in a set called set of outcomes. Okay. And if you take a set denoted as omega and this set will have all the outcomes that are associated with the random experiment. So example can be that if you are looking at the uh the example of tossing a coin what we have

**[27:16]** as outcomes are uh the usual head and tail as two possible outcomes. There can be another random experiment whose outcomes are let's say 1 2 3 uh up to six. So this is this is for coin toss and this is for say die roll. Okay. And suppose we are looking at uh the experiment of taking a picture. The

**[27:44]** possible outcomes are let's say that we are looking at um um taking the picture of uh human faces. Then it's pictures of all possible human faces all humans. So this can be the set of all the outcomes of uh the random experiment or it can be set of all possible texts

**[28:25]** and so on. Okay. So in fact uh while we work in practice right you know we assume that there is an underlying random experiment uh that is being done and what we what we get to observe are the outcomes of uh the uh that that particular random experiment. In fact we will see uh later that we will not even have the raw uh elements of the the outcomes of the sample space. you'll have uh you know

**[28:52]** something else that we define uh uh right now which is called the random variable but anyway so the idea of The idea of

**[29:32]** a random experiment is that that there exists you know this this random experiment may be conducted by uh a person or a set of people or the nature. So it is every all the systems uh that we see or the data that we get can be looked into as the possible outcomes of this random experiment and the way we quantify the random experiment is through a set which is called the sample space. So outputs of a

**[30:01]** set of all possible outcomes is what is called as the sample space denoted with omega. So sample space is a set of all possible outcomes just complete this random experiment. So

**[30:41]** please note that I have not formally defined what uh randomness is in this whole um discussion. In fact it is not a well- definfined idea. So what uh randomness is? We we assume that that it is random. You know actually we should put it within the code. It's it's random uh because we are not dealing it from the first

**[31:09]** principles of physics. Rather than that it does not have any mathematical significance and this is the only randomness that that figures in the entire probability theory. Everything else uh is is is well defined and formulated from the measure theoretic principles. This is the only level of randomness or so-called randomness that we deal with anyway. [snorts] So the starting point for uh all the probabistic uh treatment of

**[31:37]** machine learning is this idea called sample space which is a set of all possible outcomes of a random experiment. Now we don't stop there. Now once we have the sample space what do we what do we uh do with it? So sample space as we saw is is a set. So on this particular sample space we define what is called as a probability measure. Define measure

**[32:12]** on the sample space omega. Now what do you mean by uh a a probability measure? So let I'll not go into the definition of what a measure is but roughly the idea of measure is that if you are given a set if I have to quantify the length of a particular set okay or rather suppose I have a set and I want to compare two subsets of a particular set then I'll have to have a metric okay where I can measure uh quote unquote

**[32:44]** measure the size of the subset of a particular set. I'll give you an example. So we all know that uh we have the set of real numbers, right? And suppose we take uh two subsets A and B. Let me write those sets. the idea of a measure. So a measure is actually a function

**[33:15]** uh let me call that as m. A measure is a function from the subsets of a particular set. So let's call it's from subsets to positive real numbers. So this is the broadest definition of what a measure is and there can be multiple measures that one can define. [snorts] We already know uh the idea of measure from the le the usual lebe measure that we use. Let me

**[33:42]** just uh illustrate that. So what's the idea of a measure is that suppose we have uh subsets of consider the set of real numbers and a and b be two subsets of the set of real numbers that we are looking at. So an example can be you have uh a is a set between 2 and 4 which is a subset of real numbers and b is another set

**[34:11]** between uh let's say 6 and 16. Okay. Now if I'm interested in knowing which of these two sets are bigger which of these these two sets are smaller. So suppose I want to compare of A and B. So what do I do? I have to

**[34:38]** define something which translates a particular set into a positive real number so that I can compare positive real numbers. Right? [snorts] So that's the idea of a measure. So if you have to compare the size of two sets A and B, we can define a size measure. So define what is called as a size

**[35:03]** measure as suppose you have a set uh A which is a subset of uh real numbers. So let's say A is defined as uh closed A and B. Then uh the definition of the measure of a let's call that m of a can be defined as the the modulus of the absolute value of difference between b and a right in fact

**[35:32]** we know this right so this is what the the usual uh the league measure that we use or the length measure so this quantifies the idea of size or length of a set in uh if we are considering the subsets of real numbers right so now uh given the sample space which is a set of all outcomes that we have so the question is can we define so now the question is associated with

**[36:36]** with outcomes. So this is what we are interested in. So we are interested in quantifying and measuring uh the uncertaintity associated with outcomes because you know remember our our whole goal is to quantify or model a system that has uncertaintity in it. And we started by enumerating all the possible outcomes uh

**[37:04]** that the system throws at us. And then if we have to proceed further, we'll have to associate a particular measure upon the subsets of the sample space or the outcomes that we have of the experiment [snorts] and this measure has to quantify the uncert uncertaintity that is associated with each of the outcomes. Right? So that's the idea. So to do that so first we will define what is called as uh the event space f which is set of all subsets of

**[37:35]** omega. So just like we define uh the le measure or the length measure on subsets of r we'll have to define a measure on the subsets of omega which is uh uh defined as the event space. This is also called as the event space where an element A which is a member of

**[38:00]** uh this event space is actually a subset of uh the sample space omega. Right? You consider any subset of the sample space omega that becomes an element in the event space. Okay. Now we define what is called as define on the elements of the event space F

**[38:32]** as follows. So define a measure call it P. I denote it with P. So this is a measure that takes an element from the set of events and maps it to a number between a real number between 0 and 1. So this measure is called the probability measure. So think of it like the idea of size or

**[39:05]** length. You know just like we have the idea of length or size upon defined upon the subsets of real numbers. You should think of it like the size uh of every element that is a subset of the sample space. Okay. So we look at uh the outcomes of the experiment one outcome or uh multiples multiple outcomes of experiments that we are doing and associate a number between 0 and one on each of these uh elements

**[39:36]** of the event space. Okay. Now any measure should have a particular property I mean a set of properties. So uh we assign a set of properties on this particular measure that we define on the probability measure and some of the I'll not list all the properties but I'll list a few important properties properties of the probability measure is as follows. Now if we take the first

**[40:05]** property is that uh the probability measure associated with every element of f is always greater than or equal to zero for every member of the event space. Okay. And the second property is that if we consider uh the measure of the entire sample space then it is one and the measure associated with a a null event is zero.

**[40:38]** The third property is that if A and B are two elements of the sample space such that the intersection of A and B is null, then the probability measure associated with the union of these two elements is equal to the sum of the individual probability measures.

**[41:13]** Okay, so these are the uh properties of the probability measure. So now what's the idea? The idea is that given a particular element of the event space which is a subset of the sample space, we associate a number between 0 and one with each of those events and call it a probability measure. And this has these properties that uh the probability measure associated with every element uh in the sample space is always greater than or equal to zero. The least value that it can take is zero. The

**[41:40]** probability associated with the entire sample space. Now note that sample space is also an element of f. Right? That is equal to one. And the probability associated with the null event is zero. And if there are two events of two events such are such that they are they do not intersect then the probability of their union is the sum of the individual probabilities. Now this is a measure that we have defined on the subsets of the sample space. Now

**[42:08]** how do we want to interpret this measure is left to us in the sense that just like we see whenever we look at the league measure we interpret this league measure as the length of a particular subset isn't it so if you take a subset of a and b I mean a subset of real numbers which is a b closed abbec measure that we defined can be interpreted as the size or length of the set isn't it similarly

**[42:36]** probability ity measure is a mathematical construct in the sense that it's simply a function from the subsets of the sample space 2 close 01. Now [clears throat] it is for us to interpret whatever it means. So remember what was our goal? Our goal was to define a measure that quantifies the idea of uncertaintity associated with the outcomes. Now we just defined a measure which is between 0 and one. The reason it is defined between 0 and one is because we want to interpret that as something that is quantifying

**[43:03]** uncertaintity. Right? So if the probability measure associated associated with a particular event is one then we interpret that as an event that has zero uncertaintity in the sense that it is a certain event and that is why the probability of the sample space is one in the sense that we are saying the uncertaintity associated with sample space is zero or rather uh yeah because uncertaintity is uh uh is

**[43:32]** what we are trying to quantify with P. it it is the the uncertaintity associated with the um the sample space is zero or the probability of that happening is one uh because something will always come out if you do the random experiment right I mean we are saying what is the uncertaintity ass associated with the sample space we would say that it's zero similarly the uncertaintity associated with the null event is zero because there is nothing that you can be certain

**[44:01]** about the null event right And also here we are saying the uncertaintity associated with the union of two events is equal to the sum of the individual uncertaintities as long as these two events do not intersect. Okay. So now the way I just presented you know looks like uh the probability measure is the inverse of the uncertaintity. Of course, it's the inverse of uncert uncertaintity and that's why probability measures are also interpreted as the likelihood or uh the

**[44:29]** the uh the uh likelihood with which a particular given event would happen. Right? So one can interpret this probability measure uh as the metric that would quantify the uncertaintity associated with every element which is every element of uh the the uh the sample space. Okay, which is uh the

**[44:58]** outcome of a particular experiment. Okay. So now what did we uh land on to? we had. So given a random experiment, given that we are conducting a random experiment, There is a

**[45:30]** a triplet. a mathematical triplet that we constructed which is set of sample spaces and set of subset of the sample space called the event space and we have a measure that we have defined on top of this. This triplet is what is referred to as the probability [clears throat] triplet. In fact, this is the starting point of

**[46:03]** all the machine learning uh that we would do that we start by assuming that there exists a random experiment whose outcomes are enumerated in the set called the sample space and we have a set of subsets of uh subset of the sample space called the event space and we define a measure called the probability measure uh which has the properties that we just discussed uh and defined on the subset of the uh outcomes of the sample space. So, okay, this is

**[46:34]** the sample space and this is the probability [clears throat] measure. So I've still not told you what's the

**[47:03]** practical relevance of uh defining the sample space, event space, and probability measures. You know, we'll get to that in a while. But remember what's our goal? Our goal is to model systems with unknown measurements or rather abstract measurements. And we want to study the behavior of those systems. You know what are some of the concrete uh use cases or the questions uh that we will be asking while modeling these systems is something that we will see later in this course. You know one example can be that

**[47:31]** can we emulate that particular system which is what the the question that we will ask in ask in generative models. That is one question. The other question is that you know can we probe the system and uh see what sort of a behavior that the system would uh u describe or system would demonstrate if it is presented with a particular type of an input and so on. We'll ask some of those questions as we move on in this course. Uh mostly the emulation question is what we'll be asking because we are looking at the

**[47:58]** generative part of it. But yeah, so the idea is the summary so far is that if we are g given systems uh that have abstract and non-measurable parameters, one of the best ways to quantify that is to model it using repeated observations. Now how do we do that is by assuming that uh that we have uh a random experiment. Okay, the system is a I mean the the the system gives out the

**[48:28]** uh the outcomes. A system is conducting random experiments and it gives out uh some of some outcomes and these outcomes are enumerated in a set called uh the sample space and we define a measure called the probability measure on this particular set that would quantify how certain or how likely a particular outcome of uh this particular uh sample spaces. Okay. Or random experiment is okay. So from there we will move on. See

**[48:58]** what will happen is in practice you know we'll have to now uh come to the practical world. So often times or most of the times in practice access to the probability triplet which is right. We don't get to see what the

**[49:29]** sample space is. we'll not know what the outcomes are and of course we will not know what the underlying probability measure is. Now to do that I mean but what we will have is some surrogate of the random experiment which which is a measurable thing. Let me tell you what I mean. So if you take the example of uh let's say um gathering an X-ray or taking an X-ray image as the random

**[50:03]** experiment. So do we have the outcome of uh this taking an X-ray as a random experiment? We don't because it has abstract. Now you can still say that oh we do measure uh an X-ray image and what we have is a measurement uh that we see as an X-ray image but but the the the uh the counterargument to that is that what we

**[50:30]** have is not the outcome of the the sample space but a surrogate on top of it. What do I mean by that? This entire process of a person uh getting in front of an X-ray machine and the radiations being uh shined on that particular person and uh you measure the uh the X-ray that is uh the the X-ray radiations that are passing through that particular object and you know capturing it on a particular screen and converting

**[50:59]** that into something that that can be read through computers. If you take that entire process as the random experiment, so we don't have the outcome of it. But what we actually have is the final set of numbers that we read on X-ray that we read as an X-ray image in in a computer. Okay. So which means that what we have as the outcome is a surrogate. Let me give you another example. If you take the example of uh

**[51:27]** creating a textual sentence and uh constructing that to be a spam. Let's say that you know somebody is typing a a a paragraph that would constitute a a spam. Now what would what would be the elements of the sample space in that case would be that somebody thinking that you know he or she would create a spam and then constructing a set of words uh uh that that they would type out and this entire process of typing

**[51:57]** something out constitutes the the random experiment. what do we get to observe are a set of uni- code characters that we measure uh in in our computers which is not the same as the outcome of the the random experiment but a surrogate on of on the sample space. So what do I mean by that is that we yeah we would not have access to the probability triplet. We get to measure So what are these surrogates? These are

**[52:48]** actual measurements that we do the actual measurements that we get to work with are the surrogates that I'm talking about. Now we want another mathematical tooling that would capture this idea that we don't have the uh elements of the sample space themselves but we have what we have are surrogates of the sample space. So now on omega

**[53:32]** to describe the surrogate or the measurement that we What do we mean by that? We start with omega and define a function on top of this omega. Okay, which maps it into real numbers. So what we actually finally get to measure are

**[54:01]** real numbers, isn't it? Suppose we we are looking at an image. The elements of those image are real numbers. So finally what we work with are not the elements of the sample space but we work on a surrogate of uh the sample space and what do I mean by surrogate? It's a representative of the elements of the sample space which are real numbers. So finally we get to work with real numbers uh which we call our data. It can be an image or a text or any a speech signal or anything and these are not elements

**[54:29]** of sample space irrespective of the random experiment that we are that we are conducting or that the nature is conducting. Finally, what we get to work with are real numbers which are our measurements which is our data. Now, we need a mathematical construct that would translate the elements of the sample space onto the real numbers that we work with. Okay? And that mathematical construct is a function that takes the elements of the sample space and maps it to real numbers. And this function is

**[54:58]** what is referred to as the random variable. This is random variable. So I keep saying this multiple times. You know random variable is such a misnomer that random variable is neither random nor a variable. Right? I mean it's actually a deterministic function uh that we look at. It's actually a function. A random variable is a function from sample space Now in general uh it can be a function

**[55:40]** from the sample space to a d-dimensional uh uklidian space or d- dimensional real vector uh which is that you know where d is any arbitrary positive uh real number. So what does this random variable uh give us? This random variable takes the elements of the sample space which are abstract which we do not get to observe and maps it to a real vector that we actually observe in practice which we work with image now uh will be an

**[56:11]** element from a d-dimensional real space. Right? So that element from the dimensional real space now can be seen as uh an element from the sample space being transformed into a dimensional real number under this function called random variable. Okay. So now once we have the a function that we have defined on the set omega what happens to the probability triplet? So once

**[56:39]** a function random variable a function which is called the random variable again I keep saying this because you know people generally uh mistake this idea of random variable they think that it's a it's a variable it's actually not a variable it's it's a function that we define on sample space. So once a function called random variable is defined on omega the probability

**[57:05]** triplet that we talked about So what is meant by that is that we start with the probability triplet uh omega and the event space f and the

**[57:34]** probability measure uh p and then we define a function on top of omega. Once we do that, omega gets translated into a d-dimensional real space. Okay. And then the event space gets translated to something called Borel sigma algebra which is I I'll just talk about it and the probability measure gets translated into uh something called a distribution

**[58:02]** function. Okay. So let us define all of these. So of course we know that x is a function that is anyway defined from omega to rd. This we know. Okay. So this is converting each element from the sample space to an element in RD which is a d-dimensional uh real vector. So given an element in the sample space this will the random variable as a function will give you a vector in RD. Okay. So what is B? The sigma algebra or

**[58:31]** the Bal sigma algebra is now so now F used to be the subsets of [snorts] omega correct. So F was the subset of omega and therefore the boral sigma algebra B is a subset of a set of subsets of RD. So if you take the real numbers and take the subsets of

**[58:58]** the real numbers then you get the sigma algebra B and what happens to the probability measure that we defined. So so P gets transformed into something called the distribution function that I talked about. So what is a distribution function? Let's define that distribution function. Okay, defined at a particular value X. Now please note the notation here that uh I write script p for denoting the distribution function and this x which

**[59:27]** is the subscript subscript on the distribution function tells us that this is the random variable uh that we are talking about. This is the function which is called the random variable. I I mean I cannot emphasize this enough. A random variable is a function. Okay, we it's just a bad nomature calling it a random variable. It's actually a function. So this probability uh sorry prob distribution function is something that is defined via a particular random variable x and it is getting evaluated

**[59:57]** at a point x. So please note that this x and this x is different. Maybe let me just uh write it using a different uh notation here. Uh let me use small x to define that I'm measuring it. So this is I'm measuring this function evaluating this function at a at this small X and this is defined under the random variable capital X. So what is this defined as? This is defined as the probability. Okay. So note that the probability

**[60:26]** measure is always defined you know by definition probability measure operates on the elements of F which is the event space. Okay. So now whatever goes as an input to the uh probability measure has to be an element of the event space. Okay. So now this probability it's it's the probability of an event A. What is that event? It is that event which gets mapped to a particular set which is minus infinity to x which

**[60:58]** [snorts] is the small x under the random variable x. What do I mean by this? So it it looks complicated but it's not. Let's say that I have a random variable x okay that takes an element of omega and maps it to real numbers r. Okay, we are we are looking at uh the uh the case where the domain or co-domain of the range space of the random variable as a function is uh real numbers. We have a case like that. Now

**[61:28]** imagine uh a set A okay which is uh an element of the sample space and this set in R

**[61:54]** under the random variable isn't it? So whenever we take a random variable which is a function which takes an element of uh the sample space and maps it to real numbers every subset of omega which is an element of f gets mapped to some subset of r or some element in the sigma algebra that we talked about. Okay. Now any every subset of f gets mapped to a subset of r under the random variable x because it's a function. Now suppose

**[62:35]** which is uh an element of uh the event space gets mapped to this particular subset which is um open minus infinity to closed X under X. Now what am I saying is that if you take uh a subset of the uh the sample space or the an element of the event space such that it gets mapped to

**[63:05]** this particular subset of R under the random variable that is possible right so what happens is that every element in the sample space is now getting mapped to a real number. Now if you take a subset of u uh the sample space which is an element of the uh event space that also gets mapped to some subset in R under the random variable. Now suppose there is there exists a set in F such that it gets mapped to this particular subset of R. So this is this thing is a subset of R, isn't it?

**[63:40]** So this particular subset is a uh this set this set is a subset of R. If that gets mapped and to that particular subset then the probability distribution function under the random variable evaluated at this small X. Please note that the small X here which is an argument under the distribution function is same as this X that we are talking about. This is defined as the probability measure okay of set A.

**[64:08]** What is the set A? Set A is simply that subset okay which gets mapped to this particular subset under the random variable. So what we are saying is that what is this X inverse? X inverse is the inverse image. So whenever you define a function you can always define an inverse image of that function. So take this function random variable okay and take the inverse image. So what does the

**[64:39]** inverse image of a random variable do? Well, while random variable takes an element of the sample space and maps it to R, the inverse image of this function takes an element of R and maps it to omega. Isn't it? That's the inverse image. So therefore you consider the inverse image of this particular subset of R. Okay, that gets mapped to some subset in uh omega which is an element of F. You

**[65:08]** consider the probability of that particular set which is well defined because we define the probability measure on the elements of the subsets of the omega which are the elements of the event space. Okay. So this probability becomes a well- definfined measure. So that probability is what we refer to as the probability distribution function. So this is the probability

**[65:46]** that we evaluate at x. So let me just uh write it uh uh completely so that the picture is clear. Now the probability distribution function at

**[66:17]** small X. So this is how one should read this. Huh? So now this is what we uh denote as px evaluated at x. Okay. So this is how how is this uh read? This is read as the probability distribution function. Okay. Under the random variable x evaluated at small x. This is how one reads that. So this is this is equal to is the probability

**[66:54]** of the event A. Okay. that probability of event A that gets mapped to a particular subset mapped to this

**[67:20]** particular subset under the random variable X. Right? So probability distribution function of a random variable evaluated at X is equal to the probability of [snorts] the particular of a particular subset A. Okay, that gets mapped to minus infinity to X under the random variable. How do we write that? Write that as the inverse

**[67:48]** image. So if you take the inverse image of the random variable uh of the set minus infinity to x then whatever subset that you get of omega compute the probability of that particular subset that is the distribution function evaluated at x. Okay, see it looks like a very complicated definition but you know it is defined in this particular way because the idea is if you consider the probability of this particular subsets under R then the

**[68:20]** probability associated with any subsets can be computed. Okay. So it is with this idea that these particular subsets of the real numbers are considered. Okay. So now what happened finally is that we started with this particular triplet which is omega f and p we don't have access to this and then this gets transformed or push for pushed forwarded into another set which is r real numbers and then we have the the

**[68:50]** subsets of real numbers and the probability distribution function. Okay. Now what happens in practice is this is not accessible right? But this is what we measure and work with. We measure this and work with this not accessible and abstract. So this is

**[69:15]** the non-accessible and abstract uh part of it. So we assume that every system that we are modeling has this uh framework under it that there is a random experiment that is being conducted and there is a subset of the set of uh all the outcomes of the random experiment and we define a probability measure on top of it and then we don't have access to it to ensure that we are actually uh working with something that

**[69:42]** we measure we define what is called as a random variable. So this function is the random variable which is a function as I uh talked just talked about. So this is a random variable. We work with a random variable and convert the probability triplet omega f ofp into real numbers the plural sigma algebra and the distribution function. Now you will see in the uh in the uh upcoming uh sessions of this course that the entire problem in generative

**[70:11]** modeling is to estimate uh the underlying probability distribution function. So if you estimate the distribution function it is equivalent to estimating the under underlying probability measure. And this is what we had to do from the beginning, right? I said that our whole goal is to quantify the uncertaintity that is associated with the system. And this is the way we quantify the probability measure is the way we quantify the uncertaintity. And the whole problem in machine learning as we will see later is

**[70:40]** that we want to estimate the probability distribution function. Okay. So with this I will conclude uh today's session.
