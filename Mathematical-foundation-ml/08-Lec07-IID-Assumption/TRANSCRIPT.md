# Transcript — Lec 07 IID Assumption

> **Source:** https://www.youtube.com/watch?v=C83xmx80tMo  
> **Channel:** NPTEL - Indian Institute of Science, Bengaluru  
> **Duration:** ~30 min  
> **Note:** Auto-captions cleaned lightly. Minor ASR errors possible.

---

**[00:03]** Welcome back. What is this IID? So, this means that this is Okay? Again, I'll have to um tell you one other interpretation.

**[00:34]** Let's say uh that we have two random variables, D-dimensional random variables, okay? That are defined on the same sample space. Okay? There will be two distributions corresponding to each of them. What if both the distributions are exactly the same? So, it's like having It's like observing

**[01:01]** the same random variable twice. You can see that as either observing the same random variable twice or you can see that as observing two different random variables having the exact same distribution, correct? Okay? So, that's the when I when I say that they are identically distributed you see these all of these N points as one realization of N random variables having the exact same underlying distribution PXY. That is what it is.

**[01:32]** Okay? So, it is that you have either you have N realizations of the random variable or you say that you have one realization of N random variables, all of which which have the all of which have the exact same distribution PXY. Okay? So, that is one thing. You know, they're all identically distributed. So, whenever we do machine learning we make this assumption that all the N data points that we have are coming from the exact same distribution,

**[02:00]** which is which says that the underlying experimental conditions are not changing. So, that's why non-IID data, you know, dealing with non-IID data is a well-known research problem in ML community. So, what happens when the IID assumption breaks? Okay? So, that's why if you create a machine learning algorithm, build a machine learning algorithm on one data set, let's say that you do it on lung X-rays.

**[02:28]** So, you can't use the same uh same algorithm to work on let's say brain MRI because they're not identically distributed. The underlying sample space or the random experiment that we are doing is not the same. Okay? That's one thing. The other thing is, you know, you're saying that they're independent. So, what do I mean by that? I actually did not define independence earlier. I'll do it. See, uh recall your probability

**[02:57]** theory classes. If there are two events, there is something called statistical independent that is defined between two events, right? When? When are those two events called statistically independent? Two events are called statistically independent if Yes, the intersection uh the probability of the intersection of those is equal to the Sum or product?

**[03:28]** Yeah. So, no no, I'm I'm not asking about about the distributions. I'm asking about the probability measures. The measures. So, I'm saying if two events A and B are independent if A intersection B is equal to P of A? Yes. This is the definition of independence, right? Same thing happens with the the distribution. If the joint distribution happened to be product of marginals, then they are called

**[03:55]** then the then the random variables or the underlying distributions are called statistically independent. So, this this is a technical term. This is not nothing to do with the English independence or anything, okay? Even though they are sort of related. Now, what are we saying here is that I told you that this can be seen as one realization of N random variables. This can be seen as

**[04:22]** one realization of N random variables, each which have which all have the exact same underlying distribution. What we are assuming or what we are saying is that each of these random variables are statistically independent. Which means that if I have to write the joint distribution of all N data points, I can write that as product of the individuals. Okay? Please note

**[04:48]** that I'm not saying that Okay, so hold on. Let me come to that. You know, this creates a little bit of confusion. See here each of these random variables, right? XI. They are in D dimensions, right? Initially, I told you that even as even a D-dimensional random variable can be seen as combination of D scalar random variables. Even amongst those random variables, we

**[05:17]** can define define statistical independence. Do you see? Am I saying that here? No. Very very important. Okay? Independence in the IID assumption does not mean that the vector-valued random variable here is independent across its dimensions. The independence here only means that

**[05:42]** there is statistical independence between the individual data points that we have drawn. make that clear So, if you if you want to, you know, I'm saying that image one is independent of image two. I'm not saying that the first pixel in image one is independent of the first second pixel in image two. Sorry, image one. Do you see the difference?

**[06:10]** See, we can make that assumption, by the way. In some of one of some of the algorithms, we do make an assumption that the there is statistical independence across the dimensions of data. But in general, we don't make that assumption. In general we don't make that assumption that there is statistical independence among across the dimensions of data. We only make an assumption that there is statistical independence across multiple observations of data. But in general, different pixels might

**[06:38]** have Not might, they will almost always have dependence. So, we don't model it that way. See again, very very important, right? You know, we have two variability here, you know, two or two axes of variability, right? One is that every single random variable, which is a data point, okay, is a D-dimensional random variable which has which can be seen as amalgamation of D scalar-valued random variables. Okay? And we are not talking about statistical independence amongst those D scalar-valued random variables that

**[07:06]** constitutes a single data point. We are talking about statistical independence between different data points. We also saw we we we said that every data point can be seen as one element of the range space of one random variable, and there are N of them which have exactly the same distribution. Okay? So, those N random variables are statistically independent. That's the IID assumption.

**[07:36]** Okay? Let me write that. So independent &gt;&gt; [snorts]

**[08:01]** &gt;&gt; And they're identically distributed. What does that mean? That you have one observation of N N random variables. Each of them have exactly the same underlying distribution. Or you can say that See, if you if you have the viewpoint that you have N observations of the same random variable, this identically distributed becomes redundant, you know, you don't have to say that they are identically distributed because there is only one random variable that you're dealing with anyway. There are N observations from of that random variable.

**[08:29]** Do you see this? You know, two viewpoints. Basically, there are two different ways of looking at the exact same thing. When you see a textbook or when you read a paper they will say that you have data set and this tilde corresponds to sample from. The symbol tilde corresponds to sample from. So, it is saying that the random experiment What do you mean by sampling, by the way? Sampling is conducting multiple trials of the underlying random

**[08:55]** experiment that gives rise to the sample space. That is sampling. So, this means that we have got tossed the coin or rolled the die capital N number of times. Okay? And that has given rise to an underlying sample space. And that sample on that sample space, we have defined a random variable. And we have N observations of that. Okay? And the underlying probability measure that we have defined on the sample space have given rise to a distribution that we wish to uh estimate later.

**[09:30]** I'll take questions in a while. Is this viewpoint clear? See, I'll From now on, I'll just write that we start with a data set which has N tuples sampled from an underlying distribution. You should know what it is. Okay? You can see this as one realization of N random variables that have the exact same distribution. Or N realization of one random variable which has a distribution. Same way to look at it. The independence is across different realization of or rather different

**[09:59]** random variables. Right? And uh not across the data dimension. These are the points that I want you to understand and know. Okay? Okay, let's let me take questions now. No, independence is not interpret. Identity is interpret. Sorry, implicit. Not independence.

**[10:33]** See, independence only makes sense if you have multiple random variables. If you have one random variable, then there is no need to actually even either even Okay, I think I I should correct myself. If there is If you see that as multiple realization of a single random variable, then you neither have to talk about independence nor you have to talk about identity because there's only one random variable and n realizations of it. In the other interpretation, you need IID because you have n random variables defined on the same sample space. All of them are identically distributed and

**[11:01]** statistically independent. One realization each of multiple random variables, all of which have exact same distribution and which are all statistically independent. Yes.

**[11:33]** The other one was n realization of one random variable. IID at all. Yeah. Yes. Of course. Of course. So, basically saying, you know, all of your patients are going to the same hospital or using

**[12:00]** the same machines. See, again, this is for This is up for interpretation. Okay, this is a good question that you made. See, uh typically what happens is let's say that you are you have two data sets. Okay? One which is which contains images of human faces. The one that contains the uh images of buildings, let's say. Now, do you want to see them as uh same underlying sample space or same random experiment or not is a design choice.

**[12:32]** Suppose typically what happens is, you know, suppose you take a data set that has uh the human faces and build an algorithm. Okay? And try to test it on the building images, it won't work because by assumption, you're you're you're uh you're uh by construction, you're assuming that the random variables are identically distributed. So, when you get when you get the building image, your assumption breaks. However, suppose, you know, you combine both the data sets and build an algorithm

**[12:59]** together. You can see both of them as the outcome of the same uh random experiment. So, that's design choice. So, that's why if you include all data in the world and create a data set, there's no non-IID. That's exactly what the current LLMs are doing, you know? So, I joke always that if you can learn a I mean, or construct a lookup table for the entire universe and come up with a good hashing function, then machine learning is a solved problem. All you need to do is learn a huge

**[13:28]** lookup table for the entire universe. So, if you look at, you know, petabyte scales of data and combine all internet, you know, for all you know, you might be learning a you know, creating a big lookup table. Anyway, so point is what do you want to call as your data set is your design choice. Okay? But most, you know, uh often than not, what happens is you start with a data set. But mathematically speaking, when we

**[13:56]** design algorithms, as you will see later, we will use these uh these properties that we have uh imposed. Now, we'll use the property that these are statistically independent and we'll use the property that they are identically distributed. But it is up for interpretation. Or rather rather, it is up for uh the construction. Now, you construct it the way you want. But ensure Make sure that, you know, uh the the so-called test data, I'll define what test data is.

**[14:23]** Uh assuming that you know what test data is. You know, basically, you know, this is this is data that you build your algorithm on. You have new data that you want to predict upon, right? You'll have to ensure that that also comes from the same distribution. Because you have made this assumption implicit. Because you will be estimating the underlying distribution anyway. And if you if you start, you know, uh giving your algorithm points that have not come from this distribution, the algorithm is deemed to fail.

**[14:56]** Now, uh one one request that I have is use the precise terms that we have defined, you know? So, what is similarity? All we are saying is that by construction or by assumption, you know, whenever you make an assumption that your training and test data comes from the same distribution. That's the way you construct the algorithms. That's it. Yeah. Anything else? Good question. Very good question. Uh

**[15:29]** the question is what is the intuition behind, you know, combining uh or rather taking interpret interpreting the data and label as two random variables, you know? Why not see that as one single random variable and deal with it? Mathematically, you can. Right? But actually, you know, I I I was going to say that later, but now that you asked me, I'll tell you. See, people typically design or not give nomenclatures to problems. So, when you

**[15:57]** Suppose you have a problem where you have data and some labels, people call that as supervised learning. Okay? And suppose you have only data and no labels. Honestly, I don't understand what that is. Precisely because of the point that you asked. To me, it is some random experiment that's happening and some random variable sitting and you want to estimate the underlying distribution. Now, how does it matter if it's a label or data or anything? But

**[16:26]** historically, there has been a a nomenclature that people have made. You know, for instance, okay, there is an image and somebody says that So, what happens is the following, right? I mean, to be fair to people who did it, see, an image or typically data points are uh humanless measurements. In the sense that uh some machines make an assumption. You take an image and there is no human that is involved. But when you apply a label, there is some human judgment that comes in, you know?

**[16:55]** That's why you want to see that as a different random variable. But I'll tell you, there is a uh there is a uh a field of subfield of machine learning called self-supervised learning. All they do is they take a few dimensions of data and call that as a separate random variable and estimate the conditionals there. It's only an interpretation issue. That's why I told you I don't like to call algorithms as supervised learning, unsupervised learning, and all that.

**[17:23]** Okay? But but people do take that nomenclature. They call this Y as label. Now, why? So, I'll tell you one other interesting thing. Now that you asked that question, suppose there is an image, okay? Okay? And my problem is, you know, we know that some some of the pixels of this image is bad for some reason. It's some old picture and some And the task is to uh find out what these what these miss

**[17:51]** miss missing pixels are. Okay? This problem is called inpainting, by the way. Inpainting. So, you have an image and you inpaint it, right? I mean, now nowadays, some of these, you know, state-of-the-art models, right? Uh they can do they can erase a a background and they can add things. These are all same problems. So, now, what is the label here? The missing pixels. So, when you take data, how do you train such kind of models? You take an image, randomly mask

**[18:20]** some of these things, and ask it to reconstruct it back. In that case, you know, there are the the data some of the dimensions of data itself can be seen as the label. That is precisely why I did not start this course by saying that machine learning is divided into three parts called uh supervised learning, unsupervised learning, and all that. That's not what it is. To me, a sample space, a random variable, you want to see that as two different random variables, call one data, call one labels. That is why, you know, if

**[18:46]** you look at my writing, see, typically, a label is taken to be another random variable defined on the same sample space. You don't want to call it label, you don't call it label. And by the way, now that he has asked the question, uh problems or algorithms where label exists are called supervised learning. Really? Okay? Problems where the label does not exist are called unsupervised learning.

**[19:13]** And problems where some of the dimensions of data itself is made as label is called self-supervised learning. Very pseudo distinction, right? Now, you see why generalization helps. Okay? So, typically, that is how it is written. Okay? Yeah. Good that you asked that question. I would have anyway told you that. No, it does not. See, I've still not

**[19:42]** defined what ML is. Okay? Maybe some of you have the lot of curiosity right now. I'll tell you. See, finally, it comes down to what is the distribution that you would expect? I'll tell you what happens is all machine learning is this. Given D, estimate P. That's it. All machine learning is given that you have sampled from an unknown underlying distribution, estimate the underlying distribution. That's the problem. And the end rest of the course is simply to estimate that. In fact, this is

**[20:09]** discriminative machine learning. Generative machine learning is you not only estimate the underlying distribution, but also learn how to sample. That's it. The problem is given D, estimate P. Now, you can estimate multiple P's. You know, for instance, you can estimate the conditional P of Y given X. Then this is a classifier. Okay? And if Y belongs to RK, it's

**[20:34]** called a regressor. If Y is discrete, it's called a classification problem. And if you estimate P of X given Y, it is called conditional data estimation problems. Okay? If you estimate PXY, it's called generative problem. But basically, given this, you estimate marginals, conditionals, joints, and learn to sample. That's all machine learning. Guys, I'm I'm formally defining that in a while. But that is why I I want you to

**[21:03]** understand this. Hello. Random experiment, sample space, random variable, distributions. That's it. And we have samples from these distributions. And you do 100 things on top of those distributions. Makes sense? Yeah? Any questions? Any more questions? Yeah. So, can we now also define notion of something as good label space and Good label space? Uh no, you can't. No. What do you mean How

**[21:30]** do you define a good label space? Like label can be anything. So, how can we say that some label is a good descriptor of See, that becomes, you know, too philosophical a question to answer. Now, the question that you're asking is what is the good way to divide labels? I don't divide images, divide society, divide people. Divide anything. So, do you think men and women is a good label?

**[21:58]** Or do you think that, you know, whatever, right? You know, people have divided things into based on any random. That's why, you know, I keep saying that this notion of label is very very very Good that you asked that question. You know, it's extremely synthetic in most of the good times. That's why I keep telling this. See, to even fly an aircraft, you know, which is an engineering marvel, all you need is a four five parameter

**[22:26]** model. Newtonian mechanics will take care of your aircraft. And it's only three four parameters is all you need to estimate. Kalman filters and particle filters. That's it. But to identify whether a picture is that of a man or a woman, you need a 100 million parameter model. Why? Because the idea of semantics is very very pseudo. See, nature is simple. You can fly an aircraft only by using

**[22:53]** four five parameter models. But to identify whether somebody is you know, something is a spam or not, you need a 100 billion 100 million parameter model. Because the idea of spam keeps changing. That's why one of your assignments will be the following. I'll give you some random data set and ask you to randomly label it. And train a neural network to solve that. It will solve with 100% accuracy. That only shows

**[23:19]** how fragile and meaningless human divisions are. I actually made this experiment. I have a 3-year-old son, right? So, whatever whatever image that we identify as two, three, or something, you know, jumble them and give them give some random label to them. They will learn it. Right? So, that's all. I mean, see, as engineers, we are not concerned with philosophy.

**[23:48]** Our job is that you are given something, somebody has labeled it, asked to ask me to give an algorithm that would do this. We will do this. That's it. Okay? Just a question. Like you said about the jumbling of the of the numbers and giving it any random label. So, like that randomness is across every time if three is coming, we are giving the random label or every time that label becomes constant like if three we are Too philosophical. So, what do you mean

**[24:16]** by three? Like three is one of the numbers. I don't know. When I'm labeling, I don't know if that is one number. Like this particular shape, you are giving it label. Very good. Every time No two images are same. Yeah, that's it. &gt;&gt; So, I decide to label them. So, you give Suppose there are 100 images. You give them 100 different labels. You can You can make a machine learning model to learn it. Understood. See, that's the whole point that I am

**[24:43]** trying to make. So, the idea of similarity is very very It's a pseudo idea. You can see every image as different. So, somehow we have learned to uh Okay. So, I'll tell you now that you're asking. Hold on. &gt;&gt; [snorts]

**[25:09]** &gt;&gt; Five or eight. It looks like six. You got the point. That's it. Just like what you're predicting like when you're See, the learning The algorithm will do something. Yeah, that's what The question is you can make the algorithm learn anything. That is That is the point. &gt;&gt; Accuracy part you said like Uh you define. See, you Again, we have to define accuracy metrics, right? So, now we have That we

**[25:37]** will define. So, that's why we need a test data. We will say, "Okay, is this matching with them?" All that we can do. Right? But the idea is that, okay? Any other question? We are actually running out of time. Yeah. Uh so, uh can you distinguish between like the first D dimension within the same let's say the pixel and then this like two different set of random variables. The D dimensions and then Okay. By the way, uh this is only an interpretation.

**[26:04]** The way you interpret is the way I mean, you can see that. See, if you want this pixel thingy, right? I mean, image you're looking at. The dimensions of data are different pixels in the same image. In the constant I'm saying how it was different. Let's say we defined IID. Uh in in IID, we are never saying that the independence is across data dimensions. We are saying that the independence is across data points. Rather, you know, we have N data in in the interpretation So, okay. See, this

**[26:32]** IID thing only applies if you look into your data set as one realization of N random variables, right? Because there are N random variables, we can talk of statistical independence among those N random variables. So, we have N cross D random variables. If we view this Yeah. Yeah. Yeah. I mean, if you if you want to see all random variables as scalar random variables, then you have N cross D random variables. Correct. Amongst which there is statistical independence amongst set of DD random variables or

**[27:01]** rather Yeah, D dimensional random variables, not across the scalar dimensions. That's the way to look at it. Yeah. Sir, sir, this idea that the test data should come from the same distribution it's sampled. Sir, isn't it self-defeating? Because this can lead to overfitting anyway. For example, the tumor problem that you said. We don't want to learn the tumor classification from a particular hospital or particular experimental conditions, Hold on to that question. We have the entire course to answer that.

**[27:29]** Good question. Hold on to that question. Yeah. Anything else? Earlier when you described the joint distribution function, you said you're using small lower case X1, X2. Was it because you took a particular case of small A and small B? No. Yeah. Good that you reminded me of that. Uh see, I said that each of this is a D dimensional random variable, right? So, each of the dimensions I represent by lower case later in the course. Each of these XIs are X1,

**[28:01]** X2, X3, X4 up to D. So, that's why I wrote the scalar random variable as lower case. See, I will soon run out of happen. So, lower cases are for scalar alphabets, I told you, So, that will valued random variables and upper cases are for vector valued random variables. And in general, my random variables are always vector valued. Okay? So, I think let's stop here. Uh just a couple of minutes. Uh I mean, I wanted to actually define the problem, okay? Uh which I will Maybe I'll just do it right now.

**[28:30]** Give me 2 minutes. See, all problems in machine learning given D, sampled from an unknown

**[29:02]** underlying distribution PX. See, note that I'm writing PX here, not XY, because you can combine that Y also in this X. Doesn't matter. PX. estimate PX

**[29:32]** or So, given that you have in realizations of uh random variable with an unknown underlying distribution, estimate the underlying distribution

**[29:59]** and learn to sample from it. I mean, you don't have to learn to sample from it all every way every time. So, sampling is the generative modeling problem. You don't have to learn to sample either estimate the underlying distribution or some conditional or marginals on top of it and learn to sample. With this, we conclude this particular session on basics of probability theory. Now that we have defined everything that we need uh to go to the actual definitions of uh the machine learning problems. From the next class, we will start with

**[30:28]** defining rigorously what the problem of machine learning is and look at the algorithms for machine learning one after the other. Thank you for attending.
