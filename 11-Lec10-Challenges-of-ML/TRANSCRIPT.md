# Transcript — Lec 10 Challenge With ML

> **Source:** https://www.youtube.com/watch?v=767MLwniPKE  
> **Channel:** NPTEL - Indian Institute of Science, Bengaluru  
> **Duration:** ~35 min  
> **Note:** Auto-captions cleaned lightly. Minor ASR errors possible.

---

**[00:11]** So please be informed that if I write px pxy py given x or py it doesn't matter mathematically it's the same problem that I'm trying to solve just that interpretation changes. Are you all comfortable with this? You should be super comfortable with this because sometimes I will write px sometimes I'll write px given y sometimes I'll write p y given x and so on deliberately because you know it's the problem that we are trying to solve right depending upon what problem we we

**[00:39]** are trying to solve we will estimate the underlying density right so this right is I think let me just push it a little to my left so that it comes in the same &gt;&gt; see this boxed item is what is the

**[01:07]** central problem of machine learning it's an nomin again where is a machine here what is learning and all that I've not told you that right I mean there is I mean there's a machine which is a which is a compute unit right that's what that's all it is but anyway so this is the central mathematical problem that we are talk talking about and as I I as I told you what the what is this what is the main problem associated with this is that pxis the main challenge is so px

**[01:38]** is unknown both from or or I'll write it as completely unknown So we know nothing about px. We have zero information about the underlying distribution. But we still want to do it. Seems like a herculean task, right? Cuz if you if you know something about uh

**[02:08]** the distribution, you can perhaps think of estimating it. No, we have no information about px. However, are we totally blind? &gt;&gt; We are not totally blind. Why? we have samples from it. So that's the central problem. So given samples from an underlying distribution, can you learn the underlying distribution? That's it. So if the problem were well defined, okay, and uh uh completely tractable, then you don't need hundreds of different algorithms to solve it.

**[02:36]** And by the way, uh I'll introduce another nomature. Uh every algorithm or the estimates that you get for this underlying distribution is called a model. &gt;&gt; [clears throat] &gt;&gt; you get an estimate for the underlying distribution, right? That's what a model is. So in large language models, the model is actually an estimator for the distribution. And I've told you, I mean, anybody who has studied the estimation theory 101 knows that all estimators are wrong by

**[03:04]** definition and therefore all all models are wrong. But some estimate estimators are useful and therefore some models are useful. So here is the statement. All models are wrong. Some are useful because they they're density estimators. Because they're density estimators, there cannot be one correct density estimator, right? But you can compare density estimators depending upon their metric, some metrics. Challenge is that the underlying

**[03:38]** distribution is completely unknown and we have to get it from the samples. So this is the central question that

**[04:26]** we'll be asking. So given a distribution, how do we estimate given samples from an underlying distribution, how do we estimate its sample? So that is the central question that we are asking. So let us see how to do that. Okay, I'll give you the broad framework of how to do this. Then we will see several examples of how is how is it done. Okay. Now let's consider. So consider data set D

**[05:06]** which contains which are ID drawn from unknown underlying density px. this is also a sort of misnome. It's not completely mathematically correct because we cannot you you cannot sample from a density function. You can only

**[05:34]** sample from distribution functions. But yeah, so I will make that leap of faith and write it. Okay. So sampling ID from an unknown density and this is what we are starting from right. The goal is to estimate the underlying density. Okay. Now any estimation problem remember in the very first class I talked about a function approximation problem right you have x and y which are samples from two given sets and we are learning a function with between those

**[06:02]** two sets and uh all we have are samples from that. We don't know anything about the function. So what has been the historical method to do it? Historical method to do it is that you start by making some assumption on the unknown function and then try to solve an optimization problem. That's what I told you how to do it. H because if you're totally blind, you can't do anything. You have to start with some uh some some stick or something right totally blind is completely lost.

**[06:30]** Now mathematically speaking the way to do it is assume You assume a functional form on px or uh

**[06:57]** better way to say it is Assume a parametric functional form 1 px. Okay. Uh denoted by p theta. We are estimating the underlying density function. Okay, we don't know anything

**[07:24]** about it. What we say is that let me assume that my underlying density comes from a parametric family of functions call it P theta. Please remember what are we trying to estimate? Remember that we are trying to estimate a function from the range space of the random variable 2 R plus by definition. This is what we are trying to estimate. Okay. And we assume that this function has a particular parametric form P theta. So what do you mean by a parametric uh

**[07:52]** function? A parametric function is a is a mathematical function object that has a few parameters you know one example can be a line. So let me give examples right example can be so p theta 1 or uh what do I call it p a is of x because it's on x can be running out of alphabets I told you so

**[08:19]** let's say that this is w1 * so x is a vector so let's say that x uh is in rd if it's in rd then this is w1 transpose time x + uh w2 where w1 and w2 are also in RD right so here theta is w1 and w2 this is one example now I

**[08:48]** can so this is one example that I can take for w1 and w I mean uh the parametric family the other example can be that I assume this to be a gshian distribution right with this particular density function and this can be so this is uh I mean this this n represents that I'm looking at I'm

**[09:16]** assuming that all of you know okay maybe just for sake of completeness I'll tell you see when we talk of density functions there are a few wellestablished standard density functions So examples can be a gausian density function right and an exponential random variable density function or if you look at discrete random variables then you have the Bernoli random variable and so on right we will look at a few examples later these are some standard examples

**[09:43]** of density functions so you can take them you can take those to be uh your choice for your functional form so it can be a gosh distribution which has parameters uh mu and sigma okay in this case your theta is the set of mu and sigma right? uh mu is another vector in RD and sigma is a vector in R d /d correct this is a goshian distribution

**[10:13]** and so on you I mean see this is your model choice you have to choose a model and we will see 10 different examples for these these this p theta in this course okay we'll also study why why one model is better than the other and so on right what advantages can it offer and so on and the so-called neural network or deep neural network is one such example for this p theta we'll see what sort of

**[10:42]** examples we can take so that's the design choice so given a problem so what you are actually given is this right you are given n samples from an unknown distribution right you want to estimate the underlying distribution what you do is you start by making an assumption on p theta and that's your model choice you choose a linear linear regression or you choose a logistic regression you choose a neural network, you choose a kernel machine, you choose a GMM, you choose a transformer. All of these are PTA theta choices for this PTA. That's it. Make sense? We will see multiple models

**[11:13]** in this course. Now that's that's each model. Okay? So you make an assumption. So then once you do this, now you have already fixed a particular model. Okay? The degree of freedom is already lost. How? You are making a big leap of faith that the underlying density function actually has this functional form. So somebody asked a question 10 minutes back. How do we know whether that is the particular density function that density

**[11:41]** functional family that my density function belongs to? We will definitely not know. However, thankfully we have a few functional forms that are known to be universal function approximators in the sense that if you choose the parameters of those functions properly, then they can approximate any function to arbitrary closeness. Mixture densities are one such family which we will see later and neural networks are one such family. So that is why today

**[12:09]** the go-to choice for this P theta is a neural network because the neural networks are families of functions that can approximate any function to arbitrary closeness. That's called universal function approximation capability. That's why you choose such models in practice. However, once you choose a model P theta, you are making this big leap of faith that my underlying density function is a member of this family. For instance, if I choose my underlying uh model family or p theta to be w this this particular

**[12:37]** form, this is called a linear model. If I choose it to be this form, then I'm saying that my density function can be represented as a hyper plane or a line. It's not a line because line is in one dimension. It's a hyper plane in three dimensions. You assume that it's a line that that that my density function is. Now, what is what if your density function is actually a quadratic function? Uh you're lost. It's a it's a very bad model. I told all models are wrong. This is a very very wrong model

**[13:05]** or rather a worse model. But you are stuck with it. You know the moment you make a choice for P theta, the algorithm will not take care of the model. It will only take care of given that I have made an assumption that my reality or the underlying density function is a line. Tell me what line it is. That's an algorithm. Okay. So how many lines can exist in

**[13:32]** practice in RD or rather in theory in RD infinite line can exist because every line is separated by these two parameters right the in intercept and the slope. If you change the intercept and slope you'll have a different line. So once you do this so assume this is the recipe. Okay the recipe to solve an ML problem is the following. Right? The first step is assume a parametric functional form on px denoted by p theta. This is the first step. So this is your model choice. This is also called the model choice. make a model choice

**[14:00]** or choose a model. Okay, once you do this, what do you do? Chosen a model. This next step is huh? What do you mean by that? Mathematically, they have not even talked about an algorithm. I'm abstracting things out even at a at one level higher. Okay. The next thing to do is define or compute between

**[14:42]** not define actually you actually have to compute a distance metric between px and p theta. See what is the problem that we are solving right now. Uh we have made we have assumed that uh my underlying density has a particular functional form. The question that we are asking next the next logical question to ask is given that my density function is a normal density tell me which normal density it is

**[15:11]** cuz there can exist infinite normal densities. Right? If you change the mean and the variance of a normal density you will get a different normal density. Now having fixed that my underlying density function is a normal density the next logical question that that I'll be asking is amongst infinite possibilities for this normal density tell me which normal density should I choose to do that I need a notion okay of [clears throat] how good or bad a particular choice of normal density is.

**[15:42]** Suppose I choose my normal density. What do you mean by choosing a normal density? fixing the mean and variance of the normal density or this theta right these are called parameters let me write that so this theta are typically referred to as parameters so if you fix a particular value to this parameter you will get a different model suppose you fix a particular value to this parameter you will get a model now I need a way to tell you how good or bad a particular

**[16:11]** choice of this P theta is how do we do it mathematically if there were a A to quantify the distance between density metrics. The larger the distance is the poorer the model is. Correct? So first we need a way to define a distance metric between the true density that we are trying to estimate and the model choice that we have made. That's what we need to do. So let's call that

**[16:38]** as I was supposed to use capital D for this. Now again I have used D already for data sets. Let me use small D. So let D denote the distance metric

**[17:03]** [clears throat] Did you notice that I wrote distance metric within quotes? Cuz the metrics that we'll be using in practice are not metrics

**[17:30]** in the sense that a distance metric has to have a few properties such as being symmetric and uh and uh uh and obey the triangular law of inequality and all that. The metrics that that are used currently in machine learning do not have those properties and therefore they are not metrics. some notion of divergence or rather distance between distributions that's why I put this thing within quotes they're not matrix metrics mathematically speaking right so it's a function what is that it's a function d

**[18:00]** which takes px and p theta as inputs okay and uh maps it to some positive real number right so it's a function that takes two distributions as input. You see, so it's the cross productduct of two distributions that it takes as input and maps it to positive real numbers

**[18:27]** because we need the notion of distance, right? It cannot be negative. It should always be non- negative. So therefore, this is the kind of function that we define. Okay, fine. Uh let's say that we define this and we there's a way to compute most of the ML algorithms, right? So this is easy. Making a model choice is very easy. Why? Because you just choose a function. Okay. The the algorithmic difficulty, okay, comes when you comes in the

**[18:55]** subsequent steps. The second step is to define this distance metric. Now, this definition is also easy. Just like you choose a model, you choose a distance metric between them. And by the way, what differentiates one ML algorithm from the other is the choice of this function obviously and the choice of this metric also. In fact there is the third in the in the recipe we have the third step where we find the parameters by solving an optimization problem. The algorithm that is used to choice to to solve that

**[19:24]** optimization problem is also what differentiates one ML algorithm from by the other. So I give you the recipes put one here one there one there you get one different ML algorithm that is what we will see several examples of this you know several uh algorithms in this course but this is the general recipe right. So choosing a distance metric is not difficult. In fact, what is difficult is to compute this distance metric because we're sort of begging the problem. If you look at it, we started by saying

**[19:53]** that we don't know PX, right? But the distance metric that we defined, we'll take PX as an input. Now, aren't we begging the problem? Aren't we coming back to step one? Because we don't, we started by saying we don't know PX. Now, we need PX to compute P theta. Turns out no. So there are statistical methods where you can compute. See this is where things like you know law of large numbers and the the other probability theory ideas that you studied comes handy. We will see

**[20:21]** that as we go in the as we move on in the course. So they are not mere useless mathematical constructs. They they're constructed for a for a reason and we will see the usage. Huh. So now yeah hold on hold on let me complete. So now uh we have to compute the distance metric. Assume that we we will come up with algorithms to compute it. So we need a distance metric. So now what what have we done? We have chosen a model. We have a distance metric assuming that and we know that we can compute it. So now

**[20:48]** given a model, it will tell you whether how far or close this model is to reality is fine. What is the third step? The final step is I'll complete this and then take your question. The final step is by solving the following optimization

**[21:23]** problem. What optimization problem we'll solve? We will seek a theta which I call theta star the optimal parameters such that the distance metric that I have defined between px and p theta

**[21:53]** is minimized and you know what this is referred to as in ML community this is called training. training training the model. Okay, I want to write that. I can write that. [clears throat]

**[22:22]** See this course won't teach you anything different than what is already there. But I'll tell you the story from a completely different angle which is so broad that you can en encompass hundred other different stories within this framework. That's the objective of this course. So everybody knew that models will be trained in ML, [clears throat] right? But yeah, so this this this is the framework that we construct here. Okay. So that that's all it is. So how do we do now our problem boil down to given D solve this

**[22:52]** optimization problem. And I'm assuming that all of you know this notation. I'll be using this notation multiple times. Argu [clears throat] for the benefit of uh the online audience, I'll just define it. See arg min is the argument. Yeah. Not argument minimum. So if I write arg of a particular function, the question that I'm asking is suppose I minimize this particular function. Okay. With respect

**[23:21]** to this parameter theta, at what point in theta space does this function achieve its minima? The argument at which the function attains minima. It's not the minimum value of the function that I'm seeking. I'm seeking the minimizer or the argument of the function that minimizes this function. I'll be writing this. So that's the recipe. Let's walk through it. Now, this is the entire recipe of what we'll be doing in this

**[23:48]** complete course. We'll be we'll be seeing 10 different methods of doing this. Okay. The input is that we are given some data set D. Okay. Which contains samples that are drawn from an unknown distribution and our interest is to estimate either the distribution or some surrogate of it depending upon what the problem is. What is the uh recipe that we use is that we start from assuming a parametric functional form of

**[24:15]** the underlying distribution that we are trying to estimate. Okay. And then define a distine and compute a distance metric between the true distribution and the assumed model distribution and find the parameters of that model by solving the optimization problem that would seek the minimizer of the distance metric between the true distribution and the model distribution. That's ML for you. Okay. So what we'll be doing in this

**[24:42]** course right now is we will ask questions like you know what model what choice of P theta should we take? Okay. What is the choice of distance metric that we should be looking at? And how do we solve this optimization problem? It's non-trivial because most most of the distance metrics uh are non-convex. So you cannot use any of the you know well-known optimization techniques to solve this. You'll be surprised to know uh in today's time and age, the go-to standard choice for P theta is a neural

**[25:11]** network. Okay. And the go-to choice for this optimization problem is first order gradient descent. For all optimization that we have studied in our in our previous courses, all the conjugate gradient methods and second order methods and so on world to ML is artificial general int general intelligence is being achieved by using a simple first order gradient descent. [snorts] Yeah. So that's all. So we'll we'll see

**[25:38]** that in the course later. So we'll solve this optimization problems using the numerical gradient descent method of the first order and the typical amount of parameters that are there in today's models like know like GPTs and so on is of order of billions. We are now please appreciate the scale we are solving an optimization problem in a billion dimensional space because number of parameters are

**[26:06]** billion. Of course we'll not start the course with billion parameters. we'll start with five six parameters and so on. Now you know the current scale is that of billion parameters. So this reminds me of uh an anecdote that happened uh when I was in Delhi um right in front of IT Delhi uh there's this institution called Indian statistical institute ISI okay a lot of economists and statisticians and so on. So I was I was having a conversation with with an economist once and you know they were

**[26:33]** saying um yeah I mean our our models I'm I'm I'm trying a larger model for a problem that he wanted that he wanted to solve. He's saying I'm trying a very large model and see a larger model is not a good model right because you don't need too many parameters in your model you need lesser number of parameters. He was telling me that oh I'm I'm using a larger model now and I want to reduce the number of parameters and so on. Curiously I asked him how many parameters does your model has? He says oh it has it has gone to eight now I

**[27:02]** have eight parameters in my model said what eight so we are talking about 8 billion parameters right so [laughter] so that's the scale so initially all the linear models etc they were talking about you know 10 model 10 parameters 15 parameters if your data is in 100 dimensions then you have the linear model will have 100 parameters 100 plus 100 100 for slope and 100 for intercept 200 parameter model today's models have billion parameters but what's The most interesting you know

**[27:30]** and satisfying thing is that no matter how uh the models have changed and the and the landscape has changed in terms of scale the underlying mathematical problem that is being solved remains the same and this is the problem that is being solved. The problem that we are after is that see this px that we are tackling right now is the amalgamation of all human knowledge that has been seen in history that's agi. So imagine that the random experiment that we are

**[27:58]** considering and the sample space that we are considering is the entire universe and all documents, all text, all images, all kinds of data that has been produced in in in the entirety of human race is what are sample spaces and we are assuming that there exists an underlying distribution and throwing our machinery with a large lot of scale and too much compute and data comes out AGI. Why is still an open question. So is this the only way to solve crack

**[28:26]** human intelligence? Perhaps not and most likely not because a child does not needs this much of data and this much of compute to learn. We are also learning an underlying distribution. Even human learning can be cast as this problem. You know there is every time we can we make a mistake we get a feedback and then we you know we go back and so on. Right? Maybe we are estimating distributions but we don't look at the amount of data that today's models look at. Right? and and the power consumption that our brain has

**[28:56]** is is orders of magnitude lesser than what today's uh models take. So maybe there is a parallel model for intelligence. Yeah, that's for some other days discussion. But this is the recipe. Okay. And we'll be looking at several ways of solving this problem. Choose different models, choose different distance matrices, choose different optimization algorithms. And that makes you know if you choose I'll tell you right if you make this P theta to be a linear model and you make this to be the KBA labor divergence and you

**[29:24]** solve this optimization problem in a deterministic way then it becomes a linear regression if you make it compositions of sigmoids and uh if you make this metric again Kbach labor divergence and you solve this using numerical gradient descent using an repeated application of chain rule which is called back propagation then this becomes a deep neural network and so on right if you use uh the dual methods of optimization use under a kernel op kernel method formulation then that

**[29:52]** becomes a support vector machine and so on we'll see all this you know don't worry about these terms these are jarens so basic idea is that you make a choice for p theta you make a choice for resistance metric you solve the optimization problem and that's your ml okay any questions on the formulation yeah &gt;&gt; that's a question that we will answer the question is I mean I told you right we begging the question since we do not know px how do we calculate uh the

**[30:20]** distance metric that is what we will see next statistics help you there your probability theory will course the probability theory course will help you do that okay &gt;&gt; of this distance &gt;&gt; r plus r plus includes zero Let's say

**[30:47]** &gt;&gt; yeah the question is uh we jump from solving or estimating distribution functions to density functions does it change anything does it change the scale etc or anything no it does not because the density function and distribution function in the cases where they are both defined have a one to1 relationship right so we estimate we work with density functions that's it That's for mathematical convenience. See for instance gshian density is easy to compute and so on or goshian distribution is a little difficult.

**[31:14]** That's where you need q functions and so on. Remember those tables and all that density functions are easy to work with. That's the only reason we work with density functions. Okay. See the there's before we move on with this uh there is one detour that I want to take. Okay. and see [snorts]

**[31:39]** okay uh not written that thankfully the detour that I want to take is that in the first class I told you that there is a deterministic way or nonrobabilistic way of looking at machine learning correct right where you have see even there the the starting point is the same you have x and you have y and you want to learn a mapping now I cannot not talk about unsupervised learning in that framework. Now you see why it is so difficult to stick to nonproistic ways. Okay. But

**[32:09]** anyway, so let's say that you have X and Y and you want to learn a function and you make an assumption on that function and still solve an optimization problem, right? That's what you do there as well. Now does that assuming that this X and Y are vectors from some uh some space which is RD and RK and all you need to learn is a function between those two. assuming that there exists some function. Now we have this world view that those x and y's are actually the range spaces of

**[32:37]** two random variables. Right? Suppose we give up that world view and say that okay these are two sets and we want to learn a function between these two and there is some framework which is called risk minimization framework or function learning framework. Something comes up. The question is that something which comes up how does it connect to whatever we have seen? It turns out that they are exactly the same thing. So I want to show you that. Okay. The framework of empirical risk minimization

**[33:06]** or risk minimization which is what is taught in an ML 101 class. I want to show that empirical risk minimization is exactly the same of distribution estimation and uh and and and and getting you know getting the parameters of the distribution in a uh in a in a in this uh framework that we just saw. I want to show equivalence between those two because as when we move on right when you read a textbook for instance some textbooks choose to stick to uh the

**[33:35]** non-probabilistic way of looking at things or for instance there is this textbook by um Christopher Bishop right pattern recognition neural networks mo the treatment is mostly non-probabilistic but there is this there is this book by Kevin Murphy the title of the text itself is probabilistic machine learning okay uh so there uh Both treatments are seen in the in the literature. I want to sort of first fix the equivalence between those two so that when we move on you know that we are doing the same

**[34:03]** thing. Okay. So we'll do that in the next class. Uh so when we come to the next class yeah thanks for attending this class. So this uh ends this particular class. So just just recalling right we formally today defined what the problem of machine learning is. So the form the the problem of machine learning is that that we are given a few samples that are drawn iid from an unknown underlying distribution and we want to estimate either the distribution or some of the

**[34:31]** surrogates of it. That's the that's the entire problem and uh the challenge with the problem is u that you don't know the underlying distribution. So the way to do it the the general recipe is that you start by assuming that the underlying distribution has a parametric form and define and compute a distance distance metric between the true uh distribution and the model distribution or the parametric functional form and solve an optimization problem. It always happens

**[35:00]** that when I summarize the the class for a day, it looks like why did I spend 90 minutes on this topic? Always happens that way. But anyway, so that's that's that's for today and when we come to the next class, as I said, I will start from the risk minimization framework and I'll show you how risk minimization framework is exactly the same as this and then we start by looking at the linear family of models. Okay, thank you for attending this.
