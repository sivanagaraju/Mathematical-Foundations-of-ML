# Transcript — W1_T1: Tutorial 1: Forward pass & backpropagation

> **Source:** https://www.youtube.com/watch?v=VxRIqenOoQw  
> **Channel:** IIT Madras - B.S. Degree Programme  
> **Duration:** ~42 min  
> **Note:** Auto-captions cleaned lightly. Minor ASR errors possible.

---

**[00:17]** algorithm for f divergence minimization. Okay, this is what we are going to do now. So please recall that in the setup of generative modeling. The setup that we have is the following that we are given a data set. Okay, a data set

**[00:47]** denoted by D that has samples X1, X2 up to Xn. So we have n uh data points that are all sampled iid from an unknown distribution px. Okay. Uh unknown distribution px. Okay. Now we have this particular setup where we wish to build a generative model. So goal is to learn to sample from px.

**[01:25]** I want to sample from the distribution and how do we uh do that? So recall the general recipe that we had. So we have uh a function represented using the neural network a deep neural network. We call that g theta of Z where Z is coming from any

**[01:56]** arbitrary distribution which we know how to sample from. And the output of this neural network will give you samples denoted by Xcap. And we told that this we assume uh is coming from a distribution which we denoted as p theta of xcap. Okay. Now uh the goal is to I mean this will serve as a sampler for px or a

**[02:24]** generative modeler for px given p theta of xcap is same as that of px. Right? So how do we ensure that uh the above setup the above setup becomes a sampler for P theta becomes a sampler for P theta sorry sampler for PX

**[02:52]** if if my P theta is same as that of PX right so that is the idea and how do we do that we do that by solving this particular optimization problem where we fix the parameters of the neural network such that it minimizes the f divergence or any distribution of divergence metric between px and p theta. So we will choose f divergence here

**[03:19]** uh to do this. Now note that the algorithm that we are going to look at now is independent of the choice of f function that we make. So this algorithm works for any choice of f function that we make that we that we make. So this algorithm is general in that sense. Okay. Now the objective is to uh come up with an

**[03:55]** algorithm minimize minimize the f dials. Okay. Between between px and p theta. Okay. Px and p theta. Okay. Now uh the other constraint that we have is okay

**[04:38]** both right so please note that this is the most uh important challenge that comes up while building models. This is a common problem that comes up in any machine learning paradigm that you need to minimize or rather you need to optimize all functions involving distributions and density functions. Okay, but one would not have access to the actual density functions or distribution functions. So that is the key problem. No, but what do we have? We have without knowing both of them

**[05:23]** right so now we have samples that are drawn okay from both the distributions because how uh for we have samples from px because that's our data we have samples from p theta because given any theta, we could sample as many uh points that we need from Z and pass it through the G network and get samples from PTA theta. So maybe I'll just write that. So now samples [Music] from okay uh yeah samples from PX is

**[05:56]** nothing but the data set data set D. Okay. So samples from from P theta what are those? So they are the output outputs of the G function of G neural network in this case G theta of Z with or rather four different for

**[06:31]** different okay that's all so we have samples from both the distributions but we do not have the underlying distributions right now Note that the definition of f divergence involves computing an integral okay over the space of data and the integral has both p theta and px in it. Okay. Now the challenge is that we neither know px nor we know p theta and uh computing this integral is also intractable because the

**[07:00]** integral that you are computing I mean even though I've written that as a single integral this is an integral over a d-dimensional real space right you have to integrate over each of the dimensions separately and a multi-dimensional integral that p with very high dimensions is practically impossible to compute right so now the question is without having access to p theta and px and the fact that we are dealing with very high dimensional data. How do we solve this optimization problem

**[07:29]** involving divergence method? Say that is the question that we'll be asking right. Okay. Now the key idea is the following. See this is a recurrent idea that comes up with all the generative models that we are going to look at in this course. Okay. I mean this idea is actually uh a very old statistical idea that is that comes up recurrently with all machine learning algorithms is that um

**[08:13]** functions. Density functions in integral involving density functions can be approximated using samples. These can be uh approximated using samples uh drawn from the distribution. Right? Samples drawn from samples drawn from the distribution. Okay. Uh you all know this

**[08:42]** uh the idea is the following right? Suppose suppose uh we want to compute we want to compute the following integral. Okay compute the following integral. Okay so what integral do we

**[09:08]** need to compute? Let's say that we are interested in computing uh integral hx uh px dx hx px dx or x. Okay. So now this is the integral that we are interested in computing where h is any arbitrary function. Okay. H is a function or any function. It is a function on the random variable

**[09:40]** and px is as usual the density function. Okay. Right. Suppose we need to uh compute this integral. Let us call that integral. Let us call that integral some i. Okay. This is what we need to compute. Okay. What do we have is that we have we have samples drawn from

**[10:08]** PX. Samples drawn IID from PX. Okay. We have IID samples that we have drawn from PX. Okay. Let us call them as uh X1, X2 up to XN. And these samples are drawn IID from PX. Suppose we have only this and we do not we do not know what px is. Can we compute that integral is the question. Turns out we can because the integral is nothing but the

**[10:39]** expectation of the function h of x with respect to distribution px. Right? I hope that all of you know this definition. Right? So if you want to integrate a particular function h of x with uh multiplied by a density function okay over the space of uh the random variable then uh that particular integral is defined as the expectation of that particular function

**[11:06]** with respect to the underlying distribution px. Now this is called the law of the unconscious statistician abbreviated as lotus. Okay. Right. So now what we need to actually compute is the expectation of a particular function with respect to a distribution. Right? Now we also have this powerful result from statistics called the law of large numbers. weak law of [Music] large that would say

**[11:40]** that now if you have uh x1 x2 xn drawn iid from a distribution px then the expectation of rather the sample mean. Okay. Uh all of these XIS are

**[12:11]** actually coming ID from I'll write in a different line. So now all of these uh x i are being drawn iid from this the the distribution with respect to which we need to compute the integral. Then if you have enough number of samples uh at the limit as n tends to infinity uh this particular sample mean converges or rather is equal to

**[12:40]** approximately equal to the okay uh under the expectation this can become an equality but if n is small then I mean n is not sufficiently larger then this becomes an approximation that this is equal to to the expectation of this particular function h of x. Right? So this is one powerful result that is used often in machine learning that uh

**[13:10]** the two expectation okay of a function of a random variable with respect to underlying distribution can be approximated using uh the sample averages okay where the samples are actually coming from px. So note that if these excise are not coming from px this is not not true. This is true only if uh the uh samples upon which we are computing the average are actually

**[13:39]** coming from the distribution with respect to which we need to compute the expectations. Right? So that's the idea. So now if we use the law of large numbers then the integrals that involve expectations of functions with respect to distributions can be approximated using sample averages. Right? So now why is this uh relevant to the discussion that we are doing. So all we are saying is if the f divergence or any divergence

**[14:11]** metric can be expressed expressed in terms functions with respect to px and p

**[14:37]** theta. then one can compute and optimize again right so this is the idea okay now let's recall so our f divergence uh involved uh integrals yeah see the definition of f

**[15:04]** divergence involved integrals uh with p theta and px. Okay. Now this cannot be computed because as I said we neither know px nor we know p theta and these are integrals over uh over the space of random variables which are very high dimensional. But we also have this idea in statistics that would say that uh integrals involving density functions can be expressed in terms of

**[15:32]** expectations and expectations can be approximated using sample averages. Right? So we only need samples from the distribution but do not need the distribution to compute expectations. Now putting all the pieces together all we need to do is somehow represent the f divergence in terms of expectations involving p theta and px. If we can express the f divergence with uh using expectations with respect to p theta and

**[16:00]** px then perhaps we can use the samples from p theta and px which you already have access to and compute the f divergence and optimize f divergence. So that is the goal. So it turns out that uh one can actually represent uh the f divergence in terms of the expectations over p theta and px. So that is what we are going to uh see right now. Okay. So expressing

**[16:42]** expressing the F divergence in terms ations over px and pja. See this is what we need to do right now. Okay. Uh we'll see how to do that. Okay. Now uh let us start from the definition of f divergence. So this is

**[17:12]** going to be a little uh pedantic. So please uh bear with me. So dup the definition of a divergence is as follows integral over uh the space of random variable we have p theta of x time f of px evaluated x divided by p theta evaluated at the same point time dx.

**[17:44]** This is the definition of f divergence. Okay. Now I'll give you a a rough overview of how this uh math is going to be. Now please observe that we have p theta of x which is a density function right and we have some function okay so please if you compare this with uh let's say the the integral that I had written before which is x some density

**[18:12]** function times some function of the random variable x. So this the definition of f divergence does not look similar to this. Why? Because while there is a function f which is similar to this some random function uh some arbitrary function x h. Now the arguments for that uh function here was the random variable or rather samples drawn from random variable. But here the arguments for the f function uh is the

**[18:42]** ratio of the density functions themselves. Okay. So now it's you cannot write this okay as an expectation over a p theta straight away because it is not in this particular form. Okay. So to do that what is to be done is so we have to somehow decouple this f function right from this px and p theta. we somehow have to remove this px and p theta outside of this f function so that we can get this f divergence uh

**[19:13]** in this particular form which can be expressed as expectations. Okay. So that is the uh whole idea of the algebra that we are going to walk through right now. Okay. So now how do we do that? To do that we we need to first define what is called as a conjugate or a complex conjugate function. uh a conjugate function for any convex function. Conjugate conjugate function conjugate function. So any convex function okay

**[19:51]** would have a conjugate corresponding to that particular convex function. How do we define this? If f of u is a convex function, please note that u is a dummy variable here. f of u is a convex function. Then there always exists then there

**[20:37]** function denoted by f star of t. So please note that t is also a dummy variable here of t defined as follows defined as follows. Okay. Now f of t has a pointwise definition. F of t is equal to the supreum or the maximum over the domain of f. Supreum is of this particular

**[21:07]** function which is u t minus f of u. Okay. So this is the definition of uh what is called as a complex uh a conjugate function of uh a convex function. So let me explain what this is doing. Now um this is a pointwise definition in the sense that to get every point of the conjugate one needs to solve an optimization problem uh

**[21:36]** involving the original convex function. Right? determin this is what is known as uh a pointwise definition where uh so you don't get far of p just as some deterministic function of f of u but to get every point of this conjugate far you need to solve this optimization problem over this particular uh function okay that is the idea now what is this uh roughly what happens is suppose uh you you have a function f of u which is

**[22:07]** a convex function here let's say that f of u is a convex function. What we are actually doing is constructing multiple lower bounds okay for this function f of u at every point and the value for far of t would be that particular uh lower bound okay which is the tightest okay supreum uh corresponds to I mean you can look into that as uh the the maximum

**[22:34]** operation if the uh the the value is attained okay Now just for the understanding sake I can actually write this as uh max over uh uh the domain of f. So what is u here? I mean u is actually the optimization is over the domain of f. Okay. Now as I said u is uh a dummy variable here uh which is uh signifying the domain of the original convex

**[23:03]** function. So what what what one is doing is put it inside the parenthesis. What one is doing is constructing what is called as a conjugate function for a given convex function. Okay. In a pointwise manner. So how do you do that? at every point. Okay. Uh U one constructs uh multiple lower bounds on that particular U and chooses the tightest of all those lower bounds

**[23:31]** as the value of the conjugate at that particular point. Okay. So that is the definition of uh what a complex a sorry a convex conjugate is. So now every function uh every convex function would have a corresponding conjugate. Okay. Now so there are a few properties uh that we need to look at of far. So properties of star properties of the

**[24:11]** Uh by the way I did not define what a convex function is. Uh assuming that uh people are aware of it. So maybe just for the sake of completeness let me define what a convex function is here. Let me just write it here. A function f is convex. A function f of u is convex. So if and only if for all

**[24:44]** [Music] u1 u1 u2 belonging to the domain and some constant alpha 1 alpha 2 uh which are real numbers. Okay. uh alpha 1 * f of u1

**[25:10]** plus alpha 2 * f of alpha 1 u1 + alpha 2

**[25:41]** u2. Okay, this is the definition of uh what uh the convex function is. Maybe I can make it uh to be convex combination. So what I wrote was a linear combination let us make it a convex combination. So any convex combination for any convex combination both the alpha 1 and alpha 2 have to lie between 0 and 1. So you take any number

**[26:11]** between 0 and 1. Okay. And this is u2. Then for any points u1 and u2 that are in the domain of uh uh the functions uh alpha * f of u1 plus alpha alpha 1 * f of u1 plus alpha 2 * f of u2 hat should always be greater than or equal to f of alpha 1 alpha 1 u1 plus alpha 2 u2. So this is the definition of what convex function is. Okay. So let's get back to uh what we were looking at. So

**[26:41]** now every convex function will have uh what is called as a conjugate corresponding to it. So coming back to the properties of the conjugate far of uh t. So the properties are that uh far which is the conjugate is also convex. So one can when one can verify these

**[27:09]** properties. The other thing is the conjugate of the conjugate because the conjugate is also convex. One can take a conjugate of the conjugate itself will give back the original function. So these are the two properties uh that are uh that we'll be using. So this implies that um um f of

**[27:43]** u what is f of u now it is the conjugate of the conjugate now can be represented in terms of a supreum over the domain of far which is the conjugate of f Okay, please note that writing this tu

**[28:08]** or ut does not have any uh you know uh preferential orderings. So they are both dummy variables. All we are saying is that given a convex function there exists a conjugate defined by this and given the conjugate which is also convex one can take the conjugate of the conjugate which will get back the original function which means that original function now will be will become the conjugate of the conjugate which is defined this way right so you

**[28:37]** can define the conjugate for any convex function this way now far is the conjugate and we are defining the conjugate of the conjugate conjugate which means that my original function now can be represented in terms of the conjugate of itself. Okay. Okay. So now let's what is the relevance of this is this to what we are doing. So let us come back to the definition of the f divergence. The definition of the divergence was as follows.

**[29:10]** We have integral p theta f t. Now this is okay. this we'll take

**[29:37]** this as uh the uh scalar u okay so now this is this can be written as integral okay this is over x maybe I'll skip this every time so p theta of x * f of u dx right where u is the ratio of these densities the ratio of these densities

**[30:03]** now we know that f of U which is a convex function can be represented in terms of the supreum or the maximum over T which is the domain of FAR and the supreum that we take is of this particular function. Right? This is something that we just saw which means that I can write my f divergence not write the arguments every

**[30:33]** time in terms of integral p theta x times there's a max or sorry there's a supreum I actually write that as max but yeah there's a supreum over uh t which is the domain of far and you have t * u - f of t. So please remember that u here is the density ratio

**[31:07]** dx. Okay. Right? Now we'll uh unroll it. So this is uh uh integral / x p theta of x * prim / t. And uh what we have is t which is a scalar time you have uh px of x divided by p theta of x

**[31:36]** minus far of dx. Okay. Right. So now remember what was our objective? Our objective was to represent uh the f divergence in terms of expectations. Now if we were to do that then we need to have the form where there is a density and some function and

**[32:04]** we still don't have it. We have a supreum here. Right? So now to represent this integral in terms of expectation we somehow want the supreum outside the integral. Right? So that's what we will do. However you look at this we can't just put the supreum outside the integral. Why is that? Now note that this optimization problem okay is over t. However the um function that we are optimizing over

**[32:32]** has x in it. Okay. And the integral is with respect to x. So which means that I just cannot bring out the supreum outside the integral because the supreum uh that I'm computing involves x. Right? Now think about it. If you solve this optimization problem, okay, uh for a particular x, you get some t. Which means that the solution for this uh inner optimization problem is a

**[33:01]** function of t. Therefore, if I bring the supreum outside the integral, okay, now the supreum is not with respect to t but with respect to a function of x. So why is this a function of x? This is a function of x because the uh solution for this optimization problem that we have that I'm having will be a function of x because this has

**[33:29]** x as its argument. So let me write that down. This will be some functional d of x px vt of x minus sar of some function p of x. This is because because the inner optimization problem involves

**[34:08]** X involves X and the dependent or rather a function of x. Right? I'll I'll reiterate what we just did. This t was a scalar that we were optimizing over. Now suppose you

**[34:38]** solve this inner optimization problem with respect to t. You will get some solution for it. Now that solution is a function of x. Why is it a function of x? Because the uh objective function for this inner optimization problem involves x right now if we push the supreum outside the integral then for every x right uh the uh the supreum has to evaluate in such a way that if you plug

**[35:08]** in that particular x it will give you a t that is solution for this optimization problem. Right? because of the dependence of uh uh this inner optimization problem, the objective of the inner optimization problem on x. If I push the supreum outside, then the uh the optimal or the search is not over a scalar t but over a set of functions t of x such that if you plug in a value for that function t, it has to give the solution for this optimization problem.

**[35:38]** Okay. Okay. So now this search is over a functions script t. So what is script t? So script t is a space of all functions. So this this this should be

**[36:07]** from uh x to domain of far because know the uh the uh t I mean small t was a variable that we were using to represent domain of sar right. What is this? These are a space of containing or rather space of uh yeah

**[36:40]** space containing containing uh solutions for the inner optimization problem. Okay, there a space of function containing uh solutions for in optimization problem and t of x is one member of this whole family of functions. So no note that our uh uh

**[37:11]** optimization problem now is not over a spa uh scalar p but over a space of functions. So what does that mean? It actually means that we are searching for a particular function from a space of functions. Now think of this script ty as uh a bag of multiple functions. Okay. And from all possible functions from x to domain of sar we are we are searching for that one particular function that would give me the solution for this

**[37:39]** optimization problem at all x. That's all the supreum I mean that is all uh the supreum over t of x signify. Okay, but there's a problem. So the issue is that while this is true, this may not be

**[38:04]** a strict equality. Okay, this will become an inequality. So remember that our RHS was the divergence measure F divergence. Now this will become an inequality. So why does that become an inequality? Now the space of functions of functions T.

**[38:34]** Okay. That we are optimizing over. we are optimizing over may not contain the optimal T star optimal T star of X

**[39:01]** that is the solution for all X the solution for the optimization problem right that is the solution for the inner optimization Okay, what we are saying is uh that while we are searching over a space of functions from all functions from X to domain of F star, there's no guarantee that this space that we are

**[39:29]** searching over will actually contain that particular T star. Okay, which would solve this optimization problem for all X. Which means that this integral that is there or rather the solution for this optimization problem that you calculate may not be exactly equal to the f divergence. Okay, it can at best equal to the f divergence or it'll be more than that because we are

**[39:57]** solving a supreum problem here or a maximization problem. Right? So that's the idea. The idea is that uh this uh using the convex conjugate. Okay. Okay. So let me just uh copy this. What did we show so far is that we created a bound on the up divergence.

**[40:27]** Okay. So let me just uh complete this. This will be a supreum over t of x. Okay. And what is the uh integral here? If you just uh do the algebra, this would be integral p theta x * dx dx minus integral sorry this would be px right p theta would cancel away. So psd and

**[40:56]** integral p theta of x uh time f* of tx dx right. So now if you can appreciate so what do we have inside the square bracket are nothing but

**[41:23]** expectations. So we have expectation of this function P of X with respect to PX minus the expectation of this function F star of T of X with respect to P theta of X. This is what we have. So which means that we are we have uh uh achieved our goal that we have expressed the f divergence in terms of expectations over px and pa.

**[41:53]** This is exactly what we started with. No. Uh let's look at our goal. Our goal was to express the F divergence in terms of expectations over px and p theta. We accomplished that by constructing a lower bound on the f divergence. Okay. Okay. So now uh this is the uh the objective that we wanted to achieve uh in this particular

**[42:21]** lecture. So now we have expressed the f divergence in terms of uh uh the expectations over px and p theta. Now the in the next module we will see how to use uh the uh this formulation okay that we have come up with in building the generative model that we are interested in. Okay so this brings us to the end of this particular module. Thank you all.
