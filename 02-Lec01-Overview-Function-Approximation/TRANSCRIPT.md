# Transcript — Lec 01  Overview of Function Approximation

> **Source:** https://www.youtube.com/watch?v=G2h7nD_Stxg  
> **Channel:** NPTEL - Indian Institute of Science, Bengaluru  
> **Duration:** ~47 min  
> **Note:** Auto-captions cleaned lightly. Minor ASR errors possible.

---

**[00:03]** Hello everyone, welcome to the NPTEL course on mathematical foundations of machine learning. Let's begin the first lecture. So you have seen that in today's era machines have started thinking. It's a ethical and philosophical question whether it's a good thing or a bad thing that they have started thinking. Nevertheless, they have started thinking. Now without going into the ethical and

**[00:33]** philosophical issues of learning machines in this particular course we will look at algorithms which would make these machines think. Now this is a series of two courses of which this course that you are a part of currently is the first one. So here we will build the fundamentals that are required to build the

**[01:03]** state-of-the-art generative models or even machine learning models. In the next course, subsequent course that would be taught in the next semester. We will be looking at the state-of-the-art algorithms. Now think of it more like the foundational laying course where we will look at the probabistic foundations of what does it mean to make a machine learn most of these are English terms right a machine is learning thinking and all

**[01:31]** this so we'll have to put some rigor to it by writing equations now in my world mathematics is nothing less or more than a language the problem with colloquial languages such as English or whatever is that they are too fragile. You can interpret the language in 100 different ways and there is so much of ambiguity and so on. But for a practitioner who is an engineer, if you

**[01:59]** want to implement stuff and make something work, ambiguity doesn't work. So you need to make things more precise. And to make it precise, you need a language which is designed for precision. And that is mathematics. Now most of you have studied or rather studying a course on probability theory. When you study the first course on probability theory, most of the results that are taught there seem very strange,

**[02:32]** right? So there are things like conditional expectations, law of large numbers and so on. One question that often comes to one's mind is what's the use of all these algorithms? Right? What are the use? What what is the what's what's the point in even you know looking at convergence of random variables or transformations of random variables functions of random variables and so on. It turns out that without those tools you cannot make a GPT or or

**[03:04]** or a Gemini or large language models work. And in this in this course which is uh a two-part series at the end of this course you will start appreciating why these algorithms are important you know why one has to talk in a precise mathematical language and how do you implement all these and hopefully by the end of the second course you would also be in a in a situation where you can develop some of

**[03:32]** these algorithms. Okay. So now the objective of this course, the entire objective of this course is to differentiate the learners of this course uh from what I would call uh an internet data scientist. Right? Now everybody and their grandmother is a data scientist nowadays. But now what differentiates somebody? Uh one analogy that is often given is that

**[04:02]** of course you don't need to know how an internal combustion engine works to drive a car. But you need to know the inner workings if you want to build one and if you want to build one that's more efficient, right? And that's more faster, that's more lucrative in any sense. So that's the objective of this course. So uh the uh at the end of these two courses students should be in a position where you pick up any of the literature the

**[04:31]** state-of-the-art literature from machine learning you should be in a position to understand what's happening there. Most of the times what happens is students or people or practitioners get intimidated when they read papers because there's lots of theorems you know a lot of math and lot of jarens and so on. So the entire objective of this course, one of the objectives of this course is to demystify all that and make you read those papers as you read a novel, right? I mean with with ease that's what I mean and in my experience

**[05:01]** this has happened you know you can ask the previous takers of the course and this has happened the course is designed that way. So initially as I said in this first course we will we will lay the foundation and I'll try to make this course uh self-contained as much as possible right it will be self-contained but of course uh some of the deeper nuances and the research aspects of machine learning will only be discussed in the next course because it cannot be fit in one one full course that's the objective

**[05:30]** okay so having said that let us start and as I said I'll not use any slides in this course this will mostly be board work uh because you know you can ask any language model to actually create slides. So now that the course is being recorded you can put it in a language model and ask it to create slice and it would do it. what if there is some u use that is left with humanity it is with things like you know I can actually

**[05:56]** write make mistakes correct them and go back and so on right so we'll do that so it will mostly be board work that's that's what we will look at okay so now let's let's look at uh machine learning and some definitions and the other thing that I'd be doing is this course would mostly be looking at the algorithms from a probabilistic standpoint point. This course will mostly be

**[06:30]** looking at the ML algorithms from a probabilistic standpoint. I'll elaborate on this and you'll be able to appreciate and understand what I mean uh when I say it's probabilistic standpoint. Now you all of you have dealt with the problem of function approximation right most problems in science and engineering can be looked into as function approximation. So what is function approximation? From the time of people

**[06:57]** like Kepler and Gau and so on, they have been asking this question of uh function approximation. And one uh famous problem that bogged a lot of uh mathematicians uh in early 17th uh 18th century is to find a function that would estimate the positions of planet. Right? So they observed the positions of

**[07:24]** uh planets at different points in time. They wanted to know where these planets are going to be in some other points in time where the observations are not made. Okay. So this mathematically can be formulated as a function approximation problem. So what is a function approximation problem? You have two sets call that X. This is the domain set. We'll not use the word input because you

**[07:58]** know it's at an abstraction level it's just a random variable. We'll come to that in a while. For now it's it's elements of some set. Call it domain set. And there is one other set that we call the range set. You can roughly call that as an output. So what is given are

**[08:26]** pairs or tpples of this sort. So let's call this another set where you have points of this sort. I have x1 comma y1 and x2 y2 till xn. where

**[09:02]** XI Yi is a pair of observations. Now what's happening is that you are given n pairs of observation x i comma y i where you can simply at a at a mathematical abstraction level you can

**[09:31]** simply look into this as elements from two different sets for every x i you have a corresponding yi. Now we assume so we make an assumption assume that there exists so this symbol stands for there exists there exists a function f assuming that all of you know

**[09:57]** what a function is right a function is a mapping between two sets x and y so from set x to y so what does it do given a particular x i it'll will give you the corresponding yi. It'll operate on an element of x script x and it'll give you a yi and there exists a function. The whole problem is that the underlying function The whole problem is this. So we are

**[10:39]** assuming that there exists a mapping between the input set and the output set or the the domain and range while we have observations right which are evaluations of the function which we represented by this uh uh set D. We don't know what the underlying function is. Okay. So now given this so given d

**[11:08]** find f. So this has been a classical problem in science and engineering. Right? So there some for some underlying function exists. We don't know what the function is. However we have observations from that function. So which means that there is an independent variable and a function acts on that independent variable and gives some dependent variable and we have pairs of uh these values. So given these pairs we would

**[11:37]** want to estimate what the what the underlying function is. As I said this has been a problem that is sort of being studied well studied for years together right in different context of science and science and engineering. And one example that I gave is that given that I know the positions of uh the the planet at at different points in time I'd want to estimate what would be the uh the location or the position of the planet in some other point in time. So in this

**[12:04]** case your time can be seen as X and your positions can be seen as Y and so on. So you can look at multiple examples. Now how are people solving this problem? If you look at historically see one way to solve this first of all can we solve this problem? So it's actually you know it's a it's an illpose problem or a or a blind problem because we know nothing about f

**[12:32]** the underlying function that connects the time to the position of a planet is completely unknown while you are making the observations. Now when you have no information about the underlying function but only given observations from that function how do you estimate the underlying function is the question. How do you do that if you're totally blind? How do you solve that problem? Anybody? &gt;&gt; Not not present.

**[13:03]** &gt;&gt; Well, that's uh that's a bad way to do it, right? I mean, the one suggestion is that uh you just define the function to be equal to yi is at every point in x i. But that's not a good uh estimate because uh see the whole point why do we want to find the function in the very first place? We want to we want to find the function because we need to know what the output of that function is for x that we have not observed. Now if you define your function to be uh equal to

**[13:32]** the values that it has taken at all observed points then there is no I mean that that does not serve our purpose. In fact we will nice that you put this idea out because we will formalize this idea. This is what is called as overfitting. Right? We will see that idea later in the course. We'll formalize that idea. So we don't want such an approximation. But I'm asking you for the you know the highle philosophy or idea of how do you solve a problem if you are not given anything and you're completely blind in

**[13:59]** a scenario in this in this case. How do you how are you going to solve this problem? &gt;&gt; Yeah but you have to start from somewhere right? What do you optimize on? See optimization etc comes later. The first thing that you should start with is &gt;&gt; uh yeah see we have not even discussed about what is an idea of error. See this is what this is exactly the point that I'm trying to make. We have we have some idea of how to solve problems but we

**[14:28]** don't have the precise idea of how to solve problems. So this is the gap that I'm trying to fill in this particular course. So optimize error minimization all that is fine. So know what what is optimization? What's what's the notion of erroration? Right? This is what we are going to define as we move on in the course. But at a philosophical level, so what you should do is that in this these kinds of situations. Okay. So start So you first start with some guess on f.

**[15:07]** Okay, just say that fine. I mean I make an assumption that my f the underlying function is a parabola. This is one example. In fact people like Kepler actually did that. You know they said that planetary uh path is an ellipse. So you make an assumption that it's an ellipse. Now given that that I've made an assumption that this is an ellipse. Now what is the

**[15:37]** question that I need to answer? So have I answered the entire question? I have not. Why? I've already told you that it's an ellipse. So what else remains? &gt;&gt; I'll have to tell you which ellipse is it. So there can exist infinite ellipses. So given that that I have made an assumption that my reality is an ellipse. Now tell me amongst infinite possible choices that I can

**[16:07]** have for uh I mean infinite ellipses that can that can that I can choose from which ellipse should I choose in this particular situation. So this is the question that we'll be answering. Now every estimate that you get that you make for f is called a model. Okay, there's a colloquial uh saying proverbial saying in machine learning community which says that all models are wrong but some are useful

**[16:37]** right so our objective right is to get a useful model so now we'll define the notion of what useful is now one thing that I want all of you to understand is that every estimate that we get for this f is a model is an estimate right so we have no access to reality we have no access to what was the underlying function that actually generated this sort of data. So we'll only get an estimate. So now the idea is start with some initial guess on f and

**[17:05]** refine the guess using d. So this set D which are the observations that we have is what is often referred to as data and you people have already

**[17:31]** heard this that data is the new oil right we will see later that you know why that statement is not an over statement and why the quality of your data determines how good your F is and so on right so this is the philosophy so what happened is that we are given some pairs of observations. One from two sets basically and we assume that there exists a mapping between these two sets. One is called the domain, the other is the range. Uh if you want to call it input output, call it input output. Two

**[18:00]** sets mapping given the observations made from that particular uh function. The objective is to estimate the underlying function. Okay. The way to do it is start with some initial guess on f and then refine the guess using the data. Now I've told you what we'll be doing in this entire course in nutshell. Now the question that we'll be asking is what guess to make. Okay. And how to refine

**[18:29]** this guess using data and how do we know whether the guess that we have made or the refinement procedure that we have uh resorted to is a good refinement procedure. So there are 10 ways to do it you know 100 ways to do it. Every refinement procedure is an algorithm and every guess that we do is a model is a is a model family. So there is a there is linear models linear family of models and there are you know there are quadratic family of models and there are kernel machines and there are neural

**[18:57]** networks. All of these there are decision trees. All of these are the guesses the initial guesses that we make on f right and there are algorithms to refine the the guess that we make and these algorithms will give rise to a machine learning method basically. So one is the error back propagation. The error back propagation is an algorithm that is that is used to uh do this refinement on a family of models called neural networks

**[19:25]** which you'll be seeing and so on. So depending upon what sort of model choice that we make the algorithm the refinement algorithm will will change and that's what we call as learning. We'll formalize all these ideas as we move on. So because at at a broad level this is what we are going to do in this course. Okay. Now when I started I told you that we will be looking at the probabilistic viewpoint.

**[19:54]** So what do I mean by that? So this is very important. So today we will actually look at this. So people often um don't appreciate why one would need probabistic viewpoint. See while whatever I said so far or rather the entire machine or rather most of the machine learning algorithms can be explained away from the the deterministic viewpoint that I just told you. There are things that would not fit well if you do not have the

**[20:21]** probabilistic viewpoint. See for instance generative models. Generative models are best explained from a probabistic standpoint. Okay. And probabistic standpoint is so general enough that it can encompass both the discriminative and generative framework. We will we will formally define those term terminologies as we move on. But yeah, so probabilistic view framework is more broader in the sense that it will encompass all kinds of formulations of machine learning. So that's one way to

**[20:50]** look at it. The other way to look at it is as follows. Think about it now. Let's say that the problem that we are looking at is that you are given a picture. Okay? Or an X-ray. You all of you understand what an X-ray is, right? An X-ray image image of somebody's lungs.

**[21:19]** And you are also given the outputs or the uh the y set of it. I mean the the elements from the y set or the range set which would tell you whether this particular x-ray has a particular type of disease or not. Okay. In this case let us uh uh write that down. Now consider

**[21:52]** XIS to be X-ray images. Now note that when I said that I am looking at uh functions there is a domain and there is a range. I did not restrict the elements of this domain and range. Right? So now the elements each element of this set X can be a vector. What do I mean by that? I'm talking about I'm talking of vector valued

**[22:21]** functions. So all of you understand what a vector valued function is, right? A function that would take a vector as an input and gives a vector as an output. It I mean and scalar is also a special vector. All of you know that, right? So I did not restrict these X to be uh scalers. Now in the case of X-ray you imagine each of the X-ray image as a vector. I'll tell you how to imagine each of uh each of an X-ray as a vector. So now for abstraction sake let us say that each of these X-ray is a vector.

**[22:50]** Okay. Now what do I have is that I have XIS as as my um X-rays and um okay so maybe I'll just concretize it right away so that you have a better image of what I'm seeing. So this is an image right? An X-ray image. What is an X-ray image? Hold on. Okay. So now this is an X-ray image. So

**[23:23]** what we have are we have P pixels here and Q pixels here. So what is a pixel? When you represent an image in a in a computer in a in a digital system what you get is a matrix right so each of the elements of this matrix is a number between 0 and 255 if you are looking at an 8 bit image okay now I have P * Q number of

**[23:55]** pixels each of which can take a value between 0 and 255 does it make sense That's what an image is. Now what I do is I will stack each of these numbers along the columns. I take the first column. Okay. This is the first column. I take the second column of the image and keep stacking it. Okay. And how many columns are there?

**[24:27]** &gt;&gt; There are &gt;&gt; P columns. Pth column. I stack them all. And now this becomes a large vector. And what would be the size of this vector? &gt;&gt; PQ. The size of this vector is PQ, right? PQ sized vector. You have a PQ sized vector. Okay. Mathematically speaking, I can say that this is a point in a

**[24:59]** RPQ dimensional space. So now this is a visualization that I want all of you to have right now that our data or every element of this X set that that we are talking about is a point in a very highdimensional or rather in a highdimensional ukidian space. So for people

**[25:26]** uh who want their memories to be refreshed a ukidian space is an is an extension of the two-dimensional cartitionian plane. In a two-dimensional cartitionian plane you have two numbers in three dimension you have three numbers. In n dimension you have you know as human beings we can't visualize anything beyond three dimensions of course but this is a point in some PQ dimensional space. I will generalize that going forward and say that my every xi in the set that I have is actually a

**[25:56]** point in some ddimensional uklidian space. This is the notation that I'll be using. An element in the set x which is my input set is a vector in a d-dimensional real space. I'm assuming that all the data that we have all of them are real valued. Does it make sense? Okay. Yeah. So this is for uh uh so we were looking at why do we need a probabistic viewpoint. Let come let's let's let's get back there.

**[26:24]** Now let's say that x i are these d-dimensional images that we have and y i okay is simply one of two possibilities 0 and one. Okay, where zero is diseased and one is benign

**[26:55]** right or you want to say non-d diseased okay now we are dealing with a problem so there exists an Okay, that would take elements of x and projects it to space y. So now note that what is this function f doing? This function is looking at a point in dimensional space or pq. I mean our pq is here d right it is taking a point at

**[27:23]** dimensional space and assigning it a value of either zero or one. Does that make sense? Right. In fact uh in in in real world this function F is a pathologist or a radiologist. He or she is looking at the images these X-rays and diagnosing whether this X-ray correspond to a a disease case or a

**[27:51]** non-desease case. Make sense? Now, now given that you know somebody has given us uh uh some thousand uh x-rays with these labels, the problem is to find a function f okay or mimic this radiologist. So you might have heard right everybody will be out of job. This is what they mean. So we started by

**[28:19]** learning what a what a clinician or a radiologist do. At the end of the course, we will try to learn a function that would mimic anybody's job. So that is the universal function approximation idea. So basically learn any function that that's how it translates uh to mathematically speaking. But anyway, so we want to learn this function given thousand x-rays. So given thousand x-rays with the corresponding uh uh labels, our objective is to learn that underlying function that would look at an image and

**[28:50]** gives it a label of zero or one. So now you interpret it the way you want. I mean you call zero as disease, one as non disease or anything. Basically it's a function approximation problem. Now here is the catch. Now suppose you want to connect. So what is this function doing? This f is actually mapping one observations with the other. Now suppose you want to connect mass with acceleration. Both are measurable and we know how to

**[29:17]** connect them right and force I mean I meant force and acceleration there is force and acceleration we know how to connect them how to connect them &gt;&gt; they are linearly dependent that's it easy and suppose you want to connect velocity with auxilation and we know how to do that right because both are measurable now what do we want to connect here is the question think

**[29:45]** Think about it. It's very subtle. Now this X that we have right now which is which which we call as image okay is actually the measurement of how much light or rather how much radiation that a particular surface is reflecting or observing absorbing. Not observing absorbing. If you take a picture of some surface right what what are you actually

**[30:12]** measuring? If you go at the level of the sensor, what you are actually measuring is the amount of reflectance that the the the surface that is being uh measured is offering. Now why would that correspond to an abstract idea called diseased? Do you see the question

**[30:38]** more so? So we know that all of I mean machine learning today can do gender identification looking at images correct. So gender is an idea that's very synthetic. Why would the amount of light that somebody's face reflect correspond to a synthetic human created idea called gender? Now you extend this idea to anything. So

**[31:09]** now today's ML, today's machine learning algorithms does speech recognition, intent recognition, right? So what is a microphone measuring? A microphone is simply measuring the amount of pressure gradient that is being, you know, that that's being observed in the environment. That's what a condenser microphone will measure. Now, why would that correspond to something that is a semantic meaning?

**[31:40]** You get what I'm saying? So now the problem is that if there are things that are measurable, I keep joking all the time. So it turns out that learning a function that is that you know that is useful to even fly an aircraft is easy because you can you can actually fly an aircraft using Newtonian mechanics that does not need more than four or five parameters. But to find out whether somebody's picture is is that of a male gender or a female gender, you need a 100 million

**[32:08]** parameter model. Why? Because human beings are complicated and human constructs are very very complicated by construction. So one of the assignments or experiments that I'll make you to do is the following. Right? So let's say that we take the pictures of uh handwritten digits. This is one uh standard data set that people use no called the eminis data set. So where you take pictures of handwritten digits. Now one experiment that I often ask my students to do is do is that if you randomly label those data

**[32:39]** sets, can you learn a machine learning model that would explain that random labeling? Turns out that yes, you can. You will see that during your assignments. Do you see the point? Right? So the idea of semantics is very very very complex and very non-trivial. So if there are measurable quantities like force, acceleration, mass etc. then a deterministic worldview is enough.

**[33:09]** But when you are trying to learn functions between quantities that are not so well defined and very abstract, having a problem having a deterministic worldview won't help. I'll give you another example. Let's say that I want to the task that I have is that given a coin. Okay. I want to predict or rather I want to say I want to predict the outcome of the toss of that coin

**[33:37]** and I toss it. Can I solve it? So I let's say that I give you the outcome of 100 coin tosses. What are the outcomes in a coin? There are only two possible outcomes that I'll give you the I'll give I'll give you all the outcomes of the coin toss and every coin toss I make some five measurements what measurement I'll do so I'll measure how uh how what is the mass of that what is the weight of that coin and I'll measure what's the atmospheric pressure

**[34:05]** and I'll measure what is the pressure with which the person toss the coin and so on the question is given that can you give me a function that would predict what would be the next coin toss. Can you perhaps it depends it depends on what measurements are you doing right now. Historically speaking uh what happened this happened in you

**[34:35]** know mid80s. Uh so when people used to uh solve the problem of speech recognition. So what was speech the problem of speech recognition is similar. You can see you can all of you can see that problem of speech recognition is similar to what we cast right now. Right? Because what we measure are vectors. What are these vectors? These vectors are speech signals. What are these speech signals? When somebody speaks, there's a microphone that is stacked in and you get a vector of values. What what are actually those values? They are the

**[35:04]** amount of pressure gradient that is occurring when somebody is speaking. So when when you speak, you know the physics, right? the air pressure changes and that's what gets captured from from the microphone and you are given that. Now what you need to do is map those pressure differences into an into an abstract idea called phone. So this sound a e or a b c etc that we say they're all abstract right they're not they're not they're not measurable and if you measure this the pressure

**[35:32]** difference while I say the sound ah and you say the sound ah they are very different they are two different vectors now when there are two different vectors how do I ensure how do I map those pressure differences to an abstract idea called phonium was the question that bogged people for a long time you know how people were solving it they modeled the entire speech generation process using physics. Okay, they they uh they modeled the

**[36:02]** speech generation process using physics as concatenation of several waveguides. Okay. And they studied the boundary conditions all the boundary conditions and they said that if a particular phone is perceived then the then the constriction that should happen at the vocal folds has to be in a particular way and for decades generation of speed scientists studied this problem of

**[36:30]** inverting the system. Now given the speed signal go back to those those properties of waveguide. So they modeled this entire speech production process as a process of dynamical waveguide. So basically when you say ah there is one sort of construction that's happening. When you say e there is other type of construction happening. So it's bas it's like concatenate multiple pipes of different diameters. Okay. And blow air to through it. Depending upon what sort

**[36:58]** of constrictions that that that that are there you get different sounds. If somebody is uh familiar with uh playing a flute, you know what I'm talking about, right? So that's how they modeled it. So now they modeled the entire recognition problem as an inversion problem where given that this is the measurement and this is the corresponding phone that has been perceived invert the process and tell me what sort of waveguide gave rise to this particular thing. So there that this is how speech recognition problem was being

**[37:26]** studied. Okay people, similar thing happened to uh image recognition, object recognition as well. I mean you could looking at an image you wanted to find whether there is a person here there's a there's a there's a see what is the ultimatum of an image a scene rec scene understanding problem that's what today's ML will very easily do right upload a picture it will tell you oh here is a person here is it does all of this was the holy grail of uh image image recognition or image

**[37:54]** image processing engineers for decades the way they used to do it is looking at an image they first used to detect primitive uh primitive pictures such as edges and lines and image gradients and all that then hierarchically construct different objects and then try to do it and so on. So this is start basically they are trying to learn this underlying function right but doing it this way now what happened I'm just I'm giving you this history just to ensure I mean just to

**[38:23]** make you appreciate how important it is to look at the probabilistic viewpoint okay now in mid 1980s or rather late 1980s 1980s correct 1980s indust in people from industry they solved this problem of connected or rather isolated digit recognition, speech recognition using statistical methods. Now what do you mean by statistical

**[38:52]** methods? I'll we we'll see that uh in great detail uh later. But the core idea is don't look at the physics of the problem. Just do repeated observations, right? And deduce statistical outcomes from it. If I go back to the example of coin toss that I just told you, I cannot predict I mean the the the the objective is to predict what the outcome

**[39:20]** of the coin toss is. Right? Now I don't know how to study the physics of the problem. So what I will do is I will toss this coin 10,000 times. Right? And I'll count how many of them turned out to be heads. And then I use statistics to answer questions such as you know what would what is what are the like what are the chances of my next toss going to be heads and tails and so on. Right? Same thing happened with speech recognition. People used statistical

**[39:47]** methods to solve the problem of speech recognition. And uh the joke is uh or rather the cruel joke is that people speech scientists physicists that were studying uh the problem of speech recognition from first principle physics based first principle called this as ignorance modeling because they said since you don't know what's happening you are modeling your ignorance in a statistical way. I mean engineers said fine fine okay we don't

**[40:16]** mind but it works that's exactly what's happening today you don't know what are the you know the nuts and bolts of the modern day machine learning algorithms are in the sense that it's very poorly understand what the inner workings of these other than the engineering nuances of it we know how to build systems but we don't know how to understand them at one philosophical school of argument is who cares you know we don't understand human brain as well but it works

**[40:48]** An engineer would say it that way, right? I mean, yeah, I mean, today machines are thinking, right? We will eventually maybe we'll understand, but at least they are thinking today. So, this is why you need the statistical way of doing it. Long story short, I'll write that now. The thing is f is complex from

**[41:18]** the physics of the problem. I mean most of the problems which we call as the thinking problems you cannot estimate if from the physics of the problem as I said the example is why would the amount of see the question that I'm asking is the amount of light that gets reflected from some surface

**[41:48]** how is that related to an idea called gender we don't know okay but what we can definitely do is take a lot of pictures and ask somebody to rate whether or rather annotate this so as to whether this is a male gender or a female gender and try to use statistics to build something that works. So that's the whole idea. Okay. So because f or the underlying

**[42:17]** function is complex or difficult to be modeled using the physics of the problem. we need to resort to the statistical methods where the idea is to do repeated observations. So now resort to statistical methods. So this is why we need probability

**[43:02]** theory. So what is probability theory all about? That we make repeated observations and then try to learn our estimates. Does it make sense? Okay. Now, if we do this, then we'll have to formulate the this problem of function approximation, right? What was our problem? The problem in machine learning is pretty simple, right? Given D finder. So, there are two

**[43:31]** sets mappings from one to the other and there is a function that's mapping one to the other. Given multiple observations from from that particular underlying function, you estimate the function. That's all the problem is right now. This is from a deterministic standpoint. Now from a probabistic standpoint, things change. Now, now this X and Y no longer happen to be deterministic

**[43:57]** sets. They happen to be random variables. define

**[44:29]** or rather impose random variables. In fact to be precise they are the support set of random variables. What does that mean? All of you have taken a first course in probability would probably know that a random

**[44:57]** variable is neither random nor a variable. It's a deterministic function. It's a deterministic function from sample space to real values, right? And it's just not a mathematical construct, right? It it was not defined that way just for fun, right? And uh I'm assuming that all of you uh at least have heard terms like boral sigma algebra and measure spaces

**[45:24]** and so on, right? That's the first shock that you get when you once you come to IAC. But they're very very needed. So you need a measure which which we call as probability measure. So I we will end the class here but I would urge all of you to refresh your probability theory basics when you are coming to the next class specifically these these these concepts. The idea of sample space

**[45:54]** okay the idea of what a random experiment is the idea of a random variable. Okay. And the idea of a probability distribution function of course there's a probability measure and then you have a probability distribution function. Okay. And of course density functions and so on. Density functions. You know that density functions may or may not

**[46:21]** exist for all random variables. Please review your probability theory basics very very thoroughly. What we will do in the next session is that I will connect all these function approximation problem that I just said to the ideas of random variables and the distribution functions. You know, in fact, people who can see can already see this problem of function approximation gets translated translates to a function of distribution estimation.

**[46:53]** Right? And they have parallels everywhere. They have one is to one parallels. uh there are parallels between estimating a function in the deterministic worldview and estimating a density function or rather the distribution function in the probabistic framework. You're not yet done. You can wait to pack your backs. So please review your probability theory basics very very thoroughly and it is going to get rigorous. Okay. So um yeah so we are going to look

**[47:24]** at the course from a purely probabic probabistic standpoint but I will give you the deterministic uh uh equivalence at all points but the course will be based on the probabistic framework. So please review what uh your probability theory fundamentals are while you're coming to the next class. Okay we will stop here today. Thank you.
