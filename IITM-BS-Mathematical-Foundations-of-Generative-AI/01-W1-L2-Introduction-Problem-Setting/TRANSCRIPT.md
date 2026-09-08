# Transcript — W1_L2: Introduction & problem setting | generative AI basics explained

> **Source:** https://www.youtube.com/watch?v=HUunmwZfGzc  
> **Channel:** IIT Madras - B.S. Degree Programme  
> **Duration:** ~58 min  
> **Note:** Auto-captions cleaned lightly. Minor ASR errors possible.

---

**[00:11]** [Music] So with that as the introduction uh let us now uh dwell directly into the uh the course content. So let us in this module let us define what the problem of generative modeling is from a mathematical perspective. Okay. So generative modeling. Okay. So, generative models are pretty

**[00:53]** much uh everywhere uh that you would see these days, right? So, uh the examples for the generative uh applications such as chat, GPT, right? and uh the Google's uh Gemini you have claude and so on where these are

**[01:23]** conditional text in the sense that given a particular

**[01:48]** text as an input or rather prompt as it is called. So this would generate a response okay with corresponding to that particular text that you have given as an input. So that would constitute a conditional text generation task. This is uh one of the example that you see. So this text can be in natural language or this can be computer

**[02:21]** code and so on. So you might have seen a lot of applications of uh some of these uh techniques in generating computer codes okay that work for different kinds of uh tasks. So these are conditional text generators basically where conditioned on a particular text given that you are you have a particular text given as a pro as an input uh these should generate uh a text as

**[02:52]** um the response okay uh or the output. The other examples that you might have seen are uh image generators such diffusion. So these things are conditional image generators. image

**[03:25]** generators. So now what do these do? Given a particular description of the image that you want to generate, these models would create an image corresponding to that particular description. So you might have seen a lot of examples where people have generated different kinds of images uh corresponding to the prompt that they give as an input. Okay, this is another example of what the generative model is.

**[03:55]** The other class of models that you might have seen generators. So given a particular text, right, what do these do? given a particular text, they would generate a wave file, right? Or uh a speech

**[04:24]** utterance that corresponds to this particular text, right? So these are several examples of what uh uh generative models are capable of which you might have seen in uh action in uh current day scenario. Okay. Now mathematically speaking what is the problem that we have at hand? Now we will start with what is called as data.

**[04:53]** So the starting point as with any of the machine learning techniques, the starting point would be what is called as data. Okay? where I define data as a set of points which are not particularly ordered. We have n data points okay that

**[05:26]** are sampled iid according to a distribution px which is unknown. Note whenever I use script letters like this I mean that I mean probability distribution functions and whenever I'm using non-scripted letters such as this I mean

**[05:56]** probability density functions. Now we'll I'll I'll make it clearer as we move on. Now this is the starting point for any of uh the generative modeling task or for that matter any of the machine learning task that you are given n data points or samples which are drawn I mean this symbol uh denotes that these are sampled okay iid where iid stands for uh independent and identically

**[06:28]** distributed from a distribution px that is unknown. Okay. Now, typically all these data points X I they lie in some D-dimensional real space. Okay. Where D is called the dimensionality of data of the

**[06:54]** data. Okay. Uh typically this D is going to be very high. For instance, examples, you take examples. Suppose uh your uh xis are images. Images. So images are represented as the color image is represented as a as a tensor. You

**[07:24]** have r number of rows. You have c number of columns. And typically you have three channels each corresponding to uh the red, green and blue which are the primary colors. So typically uh the value of R and C would be of order of uh hundreds if not thousands and that into three would make your data lie in tens of thousands of dimensions and hundreds of thousands of dimensions no if not

**[07:52]** more. So that would be the dimensionality of your data. So let's say that your R is uh some 400 pixels and your C is 400 pixels and uh uh three. So now we are talking about uh 16 * 3 which is uh your D here would be 48 and four zeros. This would

**[08:19]** be 480,000 dimensionality of the data. Right? assuming that I uh represent each of the pixel as one dimension of data. So these images correspond to uh 400 480,000 dimension dimensionality of data. Right? So the point that I'm trying to make here is that the dimensionality of the data uh we denote that by the letter D and it is typically

**[08:49]** of order of tens of thousands if not more. Ruby we are dealing with high dimensions of uh data here. Okay. Now uh all these uh excise the data are because they are samples drawn from an unknown distribution. We say that uh this all the x's they are instantiations of some random

**[09:22]** variable random variable with a distribution with a distribution px. Okay, we assume that our data points are actually being sampled from an underlying unknown distribution and we also assume that all the data points that we are that we have have been sampled independently from uh from one

**[09:50]** distribution that we have. Okay, that's what is meant by the iid sampling which means that all of the data points that we have are independently sampled and they are identically distributed and coming according to one particular distribution which we denote as px which is unknown. Now what does that mean? If you go back to the images example suppose our data set has 1,000 images. Suppose D has 1,000 rem

**[10:22]** uh X write them denote them as X1 X2 up to X N and we have N is 1,000. Here what we are saying is the first image denoted by X1 and the second image denoted as X2 are statistically independent. Okay. And uh we are saying that both of them are

**[10:50]** actually sampled from the same underlying probability distribution uh which is denoted by px. Okay, we are saying X I is statistically independent to XJ and we have X X I and XJ both are drawn or rather I have to say X I and XJ every X I is uh drawn according to one distribution PX and of course the

**[11:18]** underlying distribution is unknown so please note that I'm not saying that the the pixel pixels are the dimensionality of the data are independent of each other. Right? I'm not saying that the first pixel here and the uh let's say the thousandth pixel here they are independent of each other. That's not what I'm saying. What this ID assumption says is that the first image that we have and the let's say 100th

**[11:49]** image that we have are statistically independent. This is a fair assumption to make in practice because suppose you are dealing with images then uh there's no reason why you should uh not believe that uh that the first image and the 10th image are statistically independent because they are collected independently. They're both captured independently at different points in time. However, unless otherwise specified, in general, we don't we do

**[12:18]** not make an assumption that the uh the data is independent in the dimensionality sense in the sense that the uh different the different dimensions of the data are not independent. We are not claiming that the uh dimen data points are not statistic are statistically independent within the

**[12:46]** dimensionality. We are only saying that across the samples there is a statistical independence in the sense that one image is independent of the other. Okay, I just wanted to make this point clear that whenever we say that the data has been drawn iid, we only mean that there is independence across the different across different data points that we sample and they are all sampled according to a same underlying distribution. Okay. So why do you have

**[13:14]** to make this assumption that they all sampled from the same underlying distribution? It is because of uh uh mathematical ease in the sense that if you have multiple distribution that your data is coming from it is it's going to be uh difficult to uh estimate multiple unknown distributions at once. So now if you assume that all your data is coming from one single distribution it is much easier to tackle the problem that way. Okay. Now in the case of uh GPT etc uh

**[13:44]** which are uh the large scale uh chart based generative models this data is a set of all documents okay a large number of documents uh or articles that are there in uh the internet okay so what we are saying is that all the data that we are getting okay data points that we are getting are independently sampled where the independence is across

**[14:15]** different data points and all of them are coming from one distribution and in the case of uh model such as GPT this distribution that we are talking about is the distribution that would capture uh the the entirety of everything that is there in internet. So I think you know you understood the point that it does not matter that uh uh this px is restrictive to a particular subdomain or anything. It just for the

**[14:43]** mathematical convenience you make an assumption that all our data points are drawn from one distribution that you want to estimate. Okay. Sure. Okay. So let us uh uh continue. So now as I said all of these uh x they are instantiations of a random variable. Okay. and they are drawn according to the distribution and uh since we are talking about uh x that is there in uh uh some d-dimensional real space x i okay are

**[15:18]** instances of a vector valued random variable uh of size D right because we are saying that all the um data is lying in a d- dimensional real space okay so now what are we

**[15:46]** saying we are saying that there exists a probability distrib distribution over the data points where all the data points are instances of a d-dimensional random variable. Okay. So please have this uh world view because this is very very important uh because this is the kind of mathematical formalism that we will be using in the rest of the course.

**[16:15]** Now I'm assuming that uh the audience in this course are aware of uh what is uh meant by a vector valued random variable. Just like you have distributions over scalar valued random variables, you have distributions over vector valued random variables. And uh the data that we have are modeled as instances of vector valued random variable drawn from one particular distribution which is unknown. Okay. Right. With this

**[16:45]** let us now define the problem of generative modeling relative modeling. How do we define the problem of generative modeling? So now what are we given are these data points which are instantiations of uh uh

**[17:23]** random variables drawn iid according to an unknown underlying distribution. Given this our objective our goal in generative modeling is to estimate estimate px. So we want to estimate the underlying probability density function or probability distribution

**[17:51]** and learn to sample from it. Okay, this is how you define the problem of generative modeling in machine learning that given n samples that are drawn iid from an unknown and so this will be typically an unknown

**[18:24]** distribution. So given that you are drawn n samples iid from an unknown distribution the objective is to estimate the underlying unknown distribution and more importantly learn to sample from the underlying distribution. Now this estimation some of the models uh do not estimate the underlying distribution explicitly. Okay, they would estimate uh px in an implicit manner but almost all the models generative models would learn how

**[18:52]** to sample from the underlying distribution. So just to contrast these this with the discriminative family of models. Now in in a discriminative model you would estimate the underlying distributions you know you typically estimate the conditional distributions of form p of y given x in the case of discriminative models. However, we do not necessarily learn to sample from the data distribution in a discriminative model. While in a

**[19:20]** generative model, the primary objective is not only to estimate the underlying distribution but also learn to sample from the underlying distribution. Okay. Okay. So now what is the general principle of solving this problem? Let us look at how this problem of generative models how do we

**[19:47]** go about solving this problem is as follows. Yeah, this is not uh some sacroscient way of doing it. But looking at all the generative models, right, one can abstract out a general principle of how these generative models uh how the realm of generative models uh work. Okay. So first thing is that

**[20:19]** start off assume a models. Let me just not call them models yet. So assume a parametric family on

**[20:53]** px. Okay. Now the goal is to estimate the underlying distribution. From now on I'll just switch on to the probability density functions assuming that the distributions that we are dealing with have well behaved density functions. Okay. Uh just to make the uh calculations and math a little more easier. Okay. So assume a parametric family on PF denoted typically denoted by P theta denoted

**[21:20]** by P theta. Okay. Now typically this P We will see specific examples for all

**[21:50]** this uh as we uh navigate through the course. Now this p theta which is uh an assumption that is made on px. Okay. uh are typically represented pardon me there's a spelling mistake here represented these are typically represented using deep neural networks this is uh owing to the fact that uh the modern deep neural networks are known to

**[22:18]** have uh very good expressive power in the sense that they can approximate a large family of functions uh in fact any function to arbitrary closeness. So this is what the universal approximation capability of deep neural networks are uh capability of deep deep neural networks is. So now given that they are so powerful uh in representing any function. So the model or uh uh the distribution or the

**[22:48]** density function is represented using a deep neural network. Okay. And this is what is called as a model. Okay. Now, this is not a standard term whatsoever, but I'll be using this term across the course. So, whenever I refer to something as a model, I'm actually talking about uh the

**[23:16]** parametric function. Okay, which we would assume on the density function that we would want to estimate. Okay, now that is what I'll be referring to as model. These things would be represented uh using deep neural networks uh at least in this course and uh in all the uh the modern uh way of looking at the generative modeling all of these are mean p theta which is the representation of ps is

**[23:47]** done via deep neural networks given their expressive power okay whenever I say model I'm talking about p theta here okay now once you do compute a

**[24:19]** diver a divergence metric or a distance metric. between the distribution P theta which the model is giving you and PX which is the true distribution. Okay. Now you define an estimated divergence metric between P theta and PX.

**[24:48]** Now you might have the question that I mean aren't we begging the question because uh we all started by saying that we don't know the underlying distribution px. Now how do you estimate or rather uh compute a divergence or a distance metric between p theta and px? That's a very very valid question which you would be taking up uh in the uh in the rest of the course. But at an abstract level, so assume that that you can define uh a distance or or

**[25:19]** a divergence metric between a pair of probability distributions which would tell you how close or far a given pair of distributions are. Okay. Now once you can measure how far or close p theta and px are the the final step is to solve an optimization problem

**[25:54]** over the parameter space of P theta of parameters of P theta P theta to Right. So that's it. So what you do is

**[26:25]** any uh generative model that we are going to uh look at in this course will roughly follow this recipe where uh given that you are you have data that are drawn ideally from an unknown distribution. Now the goal is to estimate the underlying distribution and learn to sample from it. As I said this estimation is done explicitly in some of the models and it is done implicitly in other in some other. We will see that.

**[26:54]** Now the recipe to solve this problem uh contains roughly contains three steps where you first start by assuming a parametric family on px uh denoted by p theta which is represented using deep neural networks. Okay. And then you define and estimate a divergence metric or a distance metric between P theta and PX. Which means that given a particular P theta that your neural network is representing with certain parameters uh

**[27:23]** you measure how close that particular P theta is to the true distribution PX. Okay. Now once you can measure that then finally you solve an optimization problem over the parameters of P theta to minimize the above divergence metric. So in other words, you keep tweaking uh the parameters of this deep neural network that represents your p theta till a point where the divergence metric between p theta and px reduces you know becomes uh close to zero in the sense

**[27:53]** that you learn the parameters or you adjust the parameters of this this neural network representing p theta such that your p theta approaches px as closely as closely as possible. Okay. So this is the general recipe of how uh any problem on generative modeling is solved. Let us now look at uh an example of how this is realized. Please note that uh the goal

**[28:31]** of uh generative modeling is two four. one we want to estimate the underlying unknown distribution of data and we also want to learn to sample from it. Okay. Now how is sampling accomplished? Now the procedure that I just told you is estimating b theta but how is sampling accomplished is something that we would see now. So let's say that there is a there is a random

**[29:05]** variable Z okay which is in some k dimensional real space and has some arbitrary distribution has some distribution. Okay, what are we saying

**[29:35]** here? Uh we are saying that we start from uh another random variable Z. Okay, which is in K dimensional real space which has some arbitrary distribution but it is known. An example can be that your Z can be in a gshian distribution with zero mean and unit variance which is in K dimensions. So it

**[30:03]** can be a k dimensional gian random variable with zero mean and unit variance. Right? Now we know that suppose g theta of Z. So let us yeah let us call this G g theta of Z is a

**[30:31]** function from the space of Z to the space of X in which we have defined the random variable. Okay. the traing the range space of g okay okay has a different distribution different

**[31:08]** distribution than that of X then that of Z then that of Z and the distribution of G the function G

**[31:49]** theta right so what are we saying here suppose there exist an arbitrary random variable okay and there exist a deterministic function that is operating on top of it. So let's say that G is a sorry so G is a deterministic function that is operating on an arbitrary random variable. We know from probability theory that if you transform a a random variable using a

**[32:18]** deterministic function, the range of uh that deterministic function will also be a random variable. Right? And the distribution of that random variable is not same as that of Z but depends on E theta. Right? This is a known thing. Now suppose let's suppose g

**[32:50]** theta of z is a neural network let's say that g theta of z is a neural network. Okay. Now let us draw that. Okay. So maybe before that don't say that this neural network we take Z from

**[33:32]** normal 01 as the input samples from normal 01 as the input. This is the G theta of Z network and this is given as output XCAP. Okay. So suppose d theta of z is a neural network and let's also denote the distribution of xcap which is equal to g theta of z okay it

**[34:09]** denote the theta. So I'm reiterating please note that uh uh I'll be using the word distribution function and the density

**[34:39]** function in a sort of interchangeable manner even though mathematically strictly mathematically speaking they're not exactly the same but uh for the sake of uh ease of understanding I'll be saying distributions and d using the words distributions and density functions in interchangeable manners. Please note that they're not exactly the same. Okay. Now we uh denote the density function of g theta using

**[35:11]** uh p theta of xcap. The density function of xcap as p theta of xcap. Then the output of this neural network we can interpret that as samples coming from p theta of xcap. Right? Okay. Now this neural network uh what is

**[35:42]** it trying to do? Given samples drawn from a known distribution, it is transforming that into some other distribution which is denoted by PT topics. Okay. Okay. Now suppose of px and p divergence

**[36:26]** measure between px and p theta. Okay. Now uh remember the recipe that I said you start by um assuming a parametric family on p theta then you define an estimated divergence metric between p theta and px that we have denoted by d here. Okay. What is the next step? We just need to solve an

**[36:55]** optimization problem. What is the optimization problem that we are solving? We want to set theta okay in such a way

**[37:26]** that the defined divergence metric between px which is the two data distribution and p theta which is the distribution that our model is giving is minimized. You note that this divergence metric will be a function of theta and therefore the optimization would be over theta. Okay. So now we want this theta star to

**[37:55]** be that theta which would minimize some divergence metric between px and p theta. Okay. So uh it is uh a notational custom that uh a divergence metric between two probability distributions is denoted by D uh and within parenthesis you have two distributions separated by two vertical

**[38:24]** bars. Okay, this is just denoting uh a divergence measure between two distributions. Okay. Now if we do this so upon the successful solving of this optimization problem now what happens is that your p theta becomes close to px. Suppose we choose this theta in this particular way. Correct. Okay. We'll also see later that one of the properties of this divergence metric is

**[38:52]** that it is always we choose the divergence metric such that it is always non- negative. Okay. So the minimum value of this is zero. Okay. And that happens if if and only if the underlying distributions match with each other. Okay. Then we will see these properties when we define different family of divergence metrics. But yeah, so the idea is that if you have a divergence

**[39:20]** metric, okay, that would be zero if and only if the underlying distributions match, okay, and you solve this optimization problem which is to find the parameters of this uh neural network ga in such a way that this diverence metric is minimized then we are pretty much done. Why? Because we have estimated

**[40:25]** estimated by the G theta function. Right? Because the distribution that we are getting from the output of g theta is nothing but px because now p theta has p theta has matched with px. Okay. Not only that it is being estimated uh the distribution px is simply estimated by g theta network. Well how is that possible?

**[41:08]** Now let's say that we have trained the g theta network to solve the optimization problem. Okay, let us denote that as okay let me just take the same uh picture here and reuse it. Let's say that uh we have solved the

**[41:37]** optimization problem and I'm calling this as g theta star because we have solved the optimization problem and this is p theta star and this is close to close to px because that is the way we have solved this optimization problem. Right? Now what happens is you can take a sample of Z okay and then pass it through

**[42:06]** uh the neural network here a sample from the normal distribution okay which you know how to sample network. GT network would produce a

**[42:34]** sample from PS would produce a sample [Music] from from P theta of Xcap. I mean to be precise but now because P theta of X is close to PX P theta star of Xcap okay which is close to PX by construction okay

**[43:08]** close to PX by the virtue of solving of this optimization problem we end up px. Okay, now that's it. So now uh recall what was our objective? Our objective was that uh once you are given uh some data samples that are drawn ID from an unknown distribution. We wanted

**[43:37]** to estimate generally in distribution and also learn to sample from it. Okay. Now what did we do? We started from uh an arbitrary distribution from which we know how to sample and we use the fact that uh any random variable passing through a deterministic function will produce another random variable whose distribution depends on the particular function that this random variable is passed through. Right? Now we use this

**[44:06]** property and we use this idea that we represent this g theta function using a deep neural network. Now given that a deep neural network can approximate any arbitrary functions which is the universal appro universal uh approximation capability of a neural network. We use we exploited that and we said that okay you would fix the parameters of this neural network in such a way that some distributional divergence metric between the true

**[44:34]** distribution px and the distribution that this neural network is imposing is minimized. Now upon learning what will happen this neural network would not only estimate the underlying distribution px but also give you a way to sample from it because there's no way there's no uh uh limit on how many samples that we can produce from a gian distribution right because that's random number generation correct so we know how to generate random numbers once we

**[45:04]** generate random numbers which are sampled from gian 0 i uh you can pass those samples through this trained neural network which would produce samples from p theta star of xcap which is close to px which is the underlying density function of the true data okay by the way one other side note uh the notation that I use is the following when I write a density function p the

**[45:34]** subscript here denotes the random variable under consideration and the value within the parenthesis is a particular value okay uh the symbol within the parenthesis a particular value that that particular random variable take okay that's my uh notation here when I write p theta star okay I mean that this particular distribution is parameterized by theta okay I'll be using this notation p theta

**[46:01]** star all along the course that means that uh it is it is a distribution that is parameterized by theta. Okay. And it is upon the random variable xcap. I mean strictly speaking I have to write p xcap and denote some denote theta here but that would make it a little cumbersome and that's why I write it as p theta* xcap of xcap where whenever I write uh something within the bracket I mean that uh it is take it is being evaluated at a

**[46:30]** particular value xcap. Okay. and the subscript typically uh represent the random variable or the parameters. Okay, getting back to what we were doing upon training. Okay, what we do is that we sample a random number from a distribution that we know how to sample from and then it is passed through the deterministic function G theta* here denoting that it is trained. uh then that would

**[46:59]** uh sample from G pass through G theta would produce a sample from okay sample would produce a sample from P

**[47:26]** theta star is cap which is close to PX. Okay. Uh yeah sample from Z passed through D theta star would produce a sample from P theta star which is close to px. Okay. And in turn we end up sampling from px which is our objective which is the objective that we would want to accomplish right. So this is the general principle of lot of algorithms that we are that we would uh uh study in

**[47:54]** this course starting from uh generative adversarial networks and variational autoenccoders and diffusion models and auto reggressive models and so on. This is a general principle. Okay. So uh we will be looking at different instantiations of this general principle. The questions that are to be asked now is given this the questions that are to be asked are the following. Uh the questions the questions that are to be

**[48:27]** asked are as follows. Okay. Now how to metric divergence matrix. Okay. without knowing px and p theta. Please note that

**[49:02]** in the uh example that I just discussed here, we neither know p theta nor we know px. Correct? we know we we we don't know either of the distributions uh that are used uh to solve this optimization problem. I mean one can guess that this divergence metric will be obviously a function of p theta and px. But we neither know p theta nor we know px. Now

**[49:30]** all we have all we have are samples from p theta and px. So do we know do we have samples from px? Yes we do because that's the data that we have. Do we have samples from P theta? We do have samples from P theta because by construction we can just take a sample from an arbitrary distribution and pass it through G theta and we get samples from B theta. Right? So one important question that is to be asked is how to compute the divergence methods without knowing P theta and PX.

**[50:00]** Now this is one question that we are going to uh ask and answer in this course. The second question that we should ask is what should be what should be the choice of the choice of the divergence metric divergence

**[50:27]** metric D. Okay. Uh yeah. So the other the question that we asked is ask is what should be the choice of divergence metric and different choices for the divergence metric will end up uh with different generative models with different properties. Okay. So what should be a choice of divergence metric? The other question that would be that we would ask is that how to choose how to

**[50:59]** choose the g theta function. Okay. Or in turn the the distribution P theta, right? What should be the parametric form for the P theta that one should look at. For instance, you know, P theta for a GAN is very different from that of VAE that is very different from what is done in uh diffusion models and auto reggressive

**[51:27]** models and so on. So these are typically the questions uh that would come up which you would answer in this course that now we know that the general principle is to start from an arbitrary random variable and pass it through a deep neural network so that uh the distribution of the output of the deep network matches with that of the uh underlying unknown distribution. The question that you would ask is uh how to compute the divergence metric uh without knowing both px and p theta and what

**[51:57]** should be the divergence metric itself that is to be chosen which would give different uh uh properties that that would infuse different properties on the generative models and how to choose this particular model distribution p theta and there's also one other question that that would be take taking up is How to problem of minimizing the divergence

**[52:36]** metric, right? Minimizing the divergence metric. Okay. So now we'll be looking at uh several ways of answering all these questions. Uh and uh each of one each of each of the choice that you make to answer uh these parts of these questions would uh give rise to different kinds of

**[53:05]** generative models. Okay. Now that would uh uh bring us to the end of uh this particular session. uh I'll just quickly recap. So now as I said in this course uh we will be looking at uh several families of deep generative models. Uh we'll start from adversarial networks. Then we'll go to vatium autoenccoders. Then we'll be looking at diffusion models. Then would come to auto reggressive models then state space models and some uh reinforcement

**[53:34]** learning based alignment algorithms for the uh AR language models. uh we started from defining what generative models are and some of looked at some of the examples such as stat GBT which are conditional text generators. Okay. And there are uh uh conditional image and the speed generators such as dolly stable diffusion and so on. Okay. Now then we formulated the problem of generative modeling that uh the problem

**[54:04]** is that we are given n data points that are sampled ID from an unknown distribution. Okay. And uh the data points are typically taken to be random instantiations of random variables that are defined in a d-dimensional real space. Example can be an image or a text or a speak or any of that sort. Okay. And we assume that the data points are statistically independent of each other. Okay. And drawn from one single

**[54:33]** distribution. Now formally speaking, the problem of generative modeling is that given this particular data that has been sampled ID per unknown distribution, we want to estimate the underly distribution and learn the sample from it. The general principle of solving this problem is that we start from assuming that uh the p theta that we would want to estimate uh has a parametric form sorry px that we would want to estimate has a parametric form p theta and this is typically uh

**[55:01]** represented using a deep neural network given their uh expressive power and then we define and estimate a distributional divergence metric between the true distribution and the uh the model distribution and solve an optimization problem over the parameters of the model distribution uh such that this divergence metric is minimized. Now the idea is that if we bring our model distribution p theta close to that of px then the problem is solved because we want to estimate p theta and we do it

**[55:30]** from a uh uh model uh model building perspective. Now here is an example where you start from an arbitrary random variable that you know how to sample from. Typically it is a goian random variable and define a neural network g theta of Z which is a function on this random variable uh from uh the input arbitrary random variable to the uh the dimension in which the data lies.

**[55:58]** Okay. Then you solve an optimization problem uh over the parameters of the neural network. Okay. such that the distribution of divergence between px and p theta which is the distribution of the output of this particular neural network is minimized. Now if you do this then you have practically solved the problem because upon solving you have implicitly not only implicitly estimated underlying

**[56:26]** distribution but if you take uh a random number sampled from the input distribution and pass it through this G network it would go it would give a sample that is from P theta which is now very close to PX and therefore we are sampling from the underlying distribution. Okay. Now um the samples that we are getting from the output of the neural network are not a part of the data set. Okay, obviously because if we are not interested in just

**[56:54]** reproducing what is there in the data set and this is because uh we are sampling from the underlying distribution but not just reproducing what is already there. Okay. So every time you sample from a distribution uh you get different points different uh data points in accordance with the uh the uh corresponding likelihood or the probability of each of the data points. Okay. Now with this framework the questions that we would be asking and

**[57:22]** answering in this course is first of all what should be the choice of divergence metric that should be used and then because it looks like the divergence metric is a function of both the distribution px and p theta without knowing px and p theta how do we even compute the underlying distribution of divergence then how do you choose the model itself g theta of z okay and then given that uh that there is a model, there is a distributional divergence

**[57:50]** metric that we can compute. How do we solve the underlying optimization problem of minimizing the divergence metrics? Okay, these are the questions uh that we'll be asking in the rest of uh this particular course and every different choice that we make uh or rather every different algorithm that we take up to uh answer some of these questions uh end up uh in having different generative models that are used in different uh uh

**[58:20]** scenarios. So that brings us to the end of this particular model module. See you in the next session.
