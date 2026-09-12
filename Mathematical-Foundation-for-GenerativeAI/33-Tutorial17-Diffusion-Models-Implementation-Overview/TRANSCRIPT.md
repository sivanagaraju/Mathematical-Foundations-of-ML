# Transcript — Tutorial 17 : Implementation overview of Diffusion Models

> **Source:** https://www.youtube.com/watch?v=Jw01N9Efubw  
> **Channel:** NPTEL - Indian Institute of Science, Bengaluru  
> **Duration:** ~38 min  
> **Note:** Auto-captions cleaned lightly. Minor ASR errors possible.

---

**[00:03]** Hello guys uh welcome to this tutorials in which we will be going ahead and uh coding up diffusion models. So now uh as uh done in theory class now we will be seeing uh a lot of equations and then we'll be seeing how these equations get translated into the code. Now which is the exact way of uh coding in all these uh uh tutorials that we have done. See for this I'll be using uh uh a review

**[00:33]** paper which got published in uh 2022 August so almost uh 4 years before so from the point of recording that is happening now which is in 2026 September. The name of this uh review paper is understanding diffusion models a unified perspective. Now this will be the uh uh what do you call uh the primary reference for equations. Uh this whole reason is this.

**[01:01]** See because since it is nicely typed so and each equations are labeled so it is easy for us to refer to which equation we are dealing with. Okay. So all these is nicely referred. So so if I go there if I say that okay now in this particular part of the code I'm implementing equation 28. let's say that even though that is a simple thing okay so if I say that I'm implementing this 28 so it is easy for you to go and

**[01:30]** understand now what is happening okay now that's how my whole code structure now in this uh diffusion models is okay so now before going there I just want to uh go ahead and uh and give you a very nice uh perspective of how things are done in this code okay so now we are uh looking at uh it's a hierarchal VAE okay so in which

**[02:09]** the latent space at uh each time stamp is of the same uh size as that of the data in in its uh vanilla implementation see we have implementations where in which uh we have a VQVE encoder we get the ZE and then we perform the diffusion process over the latent space okay and then we finally have a decoder of VQA so where in which uh this uh is converted

**[02:38]** into the uh image okay that's another thing but that is not what we are implementing in this particular tutorial discussion uh so we are implementing the vanilla version which is there in the uh paper that I in the review paper which I just showed. Okay. Now here uh the forward process is fixed. Okay. So now there is stoasticity

**[03:09]** involved and what do I mean by that is the parameters are fixed. Okay. And the the reverse process is learned E is fixed. Okay. Sorry. This learned.

**[03:37]** So now what is the forward process? See the the forward process that we have is uh what is uh x1. See we have uh x0 so on till xt again till x capital t. Okay. This is

**[04:05]** what we have. okay so this is the isotropic goian. So and then this is the forward process. See reverse process. This is Q of

**[04:33]** X1 given X not. Now this is q of xt given x tus 1. This is q of xt given x capital t minus one. Okay. Now this will be the reverse process. Now how am I uh labeling it?

**[05:03]** Now this is P theta of XT given XT + 1. This is P theta of X1 given X2. This is P theta of X not given X1. The forward diffusion is fixed but the reverse diffusion is what is learned. Okay fine. So now

**[05:32]** now how do we uh so better uh image is I'll show you from the paper itself. This is okay this is the forward diffusion and this is the reverse diffusion process. This is the noising process. This is the denoising process. Okay you are go adding noise in this case uh I'm assume. So what do I mean by noise? You are just adding a sample from the Gaussian distribution. So scaled and

**[06:00]** shifted. So here you are removing the noise. So that is why it is called as denoising diffusion probabilistic models. Okay. So so this this is the this is what I'm trying to tell. Fine. So what is x1? So x1 we are fixing it to some square root of this alpha 1 * x plus 1 - alpha 1 * epsylon epsylon. Okay. We can call it as epsylon

**[06:28]** 1 I guess. epsylon 1. This epsylon 1 is a sample from zero i. Okay. Now, similarly, if I go ahead, I can write it as xt is alpha t that is there into x t - 1 + 1 - alpha uh t. Okay, just d

**[07:06]** into t. Okay, so this is what we have. So this is q of x t given xt minus one. This is from a normal distribution xt alpha t square root of alpha t. This is the mean. So this is the the linear scaling and shifting of the

**[07:34]** Gaussian. So 1 minus this is the the variance. Okay. Now this is uh exactly the 32nd equation in the uh paper that we are referring. See all my uh equations will be numbered and then that numbering is from this specific paper. Okay. So this is the sorry 31st sorry 31st not 32nd it is 31st equation

**[08:07]** this is the 31st equation that we have okay fine so so good so now and then uh what is that uh we want so we want to minimize the elbow. So all those elbow constructions is good. So we have this log of px. I'm taking one example.

**[08:35]** So this is less than or equal to. So this is the full we have uh expectation of q of x1 given x into log of p theta of x given x1 this p theta this is the first term now minus expectation of q of x tus 1 given X into

**[09:08]** the KL divergence between these two distributions XT given X TUS 1 into P of XT the K divergence between this again minus there's a summation involved t = 1 to tus And

**[09:36]** there is an expectation of Q of X T - 1 comma XT given X into DKL of Q of XT given XT minus one with respect to P uh and P theta of XT t x t + 1 this is what we have got and uh

**[10:09]** what is this equation number so this is that okay this is the term this is equation this is uh equation 45 okay so now what are these three terms so this is the these are the three terms that are there the first one is what is known as the reconstruction term. So which is similar to VAE that we had.

**[10:38]** Okay. The second term which is there is what is known as the prior uh prior matching term. See I'm just foregoing all the algebra and then I'm directly taking the equations. See the whole reason is we have to understand what is that we are trying to do first. Okay. So and then this is what is known as the third term is what is known as the consistency term. These are the

**[11:09]** three terms that are involved. So consistency. So what is this? What is this consistency term doing? Now this is having some kale divergence. So the dnoising step. Now what is the denoising step? So this p theta of xt given xtus 1 should [snorts] match should match the noising step. Okay,

**[11:38]** that is what is there should match the noising step. This is what is this uh uh the third term now which is there is saying which is the consistency term. See now, now if you keenly observe this, there is a very uh interesting problem that is involved with the third term which is this. Now what is the problem here? See the

**[12:08]** whole problem here in this third term is now this is expectation over two random variables. Okay. Now we can immediately see that this is expectation over XT - 1, XT. Now there are two random variables involved for every time step and please see that there is a summation here and then here there is an expectation over two random variables. Now this is a uh very uh troublesome not in terms of calculation. Okay. So I

**[12:35]** don't I'm not saying that it is troublesome in terms of calculation. See what happens is since you have two random variables involved see the variance can increase actually. Okay. So now the question is is there any way in which I can convert this expectation which is there into one random variable so that we can go ahead. Okay. So the problem here is because of two random variables that is there and then there's a summation the variance will be large. Now is there a way in which I can have

**[13:05]** uh the expectation over one random variable is the question that we are trying to ask next. Okay. So now why am I doing it? Now hold on. Now why why are we revisiting the equations in the tutorials? Hold on to that question. See we can actually do it with the step. So this is xt given xt minus one. Now you can

**[13:32]** see this here that is what we have xt given xtus one. Okay, I'm considering that. now this these are first order marco. So therefore you can XT given XT -1 comma X not you can write it like this and then this will be

**[14:01]** Q of XT given so XT minus one Q of XT okay maybe I'll write it much more neatly so Q of X T -1 given XT comma X Q of XT given X divid by Q of XT -1

**[14:35]** given X not okay now this is what it turns out to be see now okay now this is very much easy for us now E this is what you're saying. This is what this is equation 46 that we have. This is equation 46 that we have now. Now as it turns out okay now we can

**[15:06]** actually compute this. Okay. If you put this now what will happen to the terms that are involved in the elbow is so log of px again uh there will be [snorts] three terms. What are those three terms? So the first term is expectation of q of x1 given x into log of p theta of x given x1

**[15:41]** now minus dkl see please so dkl of q of xt given x p of xt. This is the prior matching term minus summation t is equal to 2 to t expectation of q of xt given x

**[16:15]** into dk of q of expectation of this dkl of xt minus one. This is matched with p theta of xt minus one given xt.

**[16:44]** Now these are the three terms that are there. The first term is what is known as the this is the reconstruction term and the second term is the prior matching. And what is the third term? The third term is what is known as the not the consistency term. This is what

**[17:13]** is known as the denoising term. See please try to understand what is this saying. So let's say that if I have an XT here okay now XT this is X T -1 and this is X T + 1. Okay. Now what is it saying? XT - 1 given XT. Now XT - 1 given XT. So you

**[17:43]** are going like this. Now but it is with respect to Q. Now this is the noise that needs to be removed and then this is this is P of P theta of XT minus 1 given XT this is fine. So see this now which is there XT. So you can compute it. Okay. Now what is that we have here is Q of

**[18:13]** XT given XT minus one is what this step is okay but what we have is this now from this equation we can actually compute it okay so that means that there is no what do you call uh mixing that is happening so it is perfectly fine this is the den noising term but it has achieved a very interesting thing Now what has it achieved? See here

**[18:47]** now my expectation has only of one random variable. Now because of which my variance issue got removed. Okay. So now this is the equation that we'll be implementing. So this equation number from the paper. This is 58. This equation that we have is equation 58. This is what we'll be implementing actually throughout our uh uh discussion we'll be implementing this. Now how we will be implementing this is the

**[19:15]** question. Okay. So now see look at the couple of terms involved. See what is this this specific term saying. So given x not what is the noisier version at the t step is what it is saying. Now this is saying given x not what is the noisier version that is there at t minus one. So that means that so can I given a t and then the input image x not can I get the noisier version is the question that is being asked. Now as it turns out we can do it

**[19:42]** very easily. So now how do we do it? So we already know that XT equals square root of this XT - 1 + 1 - alpha T into so

**[20:10]** so this we know right now what we can do we can substitute for XTUS 1 so if I substitute for XT - 1 it will become X XT -2 now so on and so forth if I do it in the recursion So xt is nothing but square root of this. Now what is this uh this is uh alpha 1 into alpha 2 so on into alpha t square root of it into x plus

**[20:42]** 1 minus alpha t bar into epsilon not. What is alpha t bar? again in a similar manner. So now that means that given the the initial x not now I can obtain the noisier version at any given time stamp t. Now this I can do now because of which what will happen? Now I can compute this q of

**[21:13]** xtus 1 given xt comma x not I can compute this. Why am I computing this? Now look back at what is the equation that we are implementing throughout our code. The equation that we'll be implementing is equation 58. So which is kale divergence between xtus 1 uh given xt x not into this p theta not into uh so k divergence between these two distribution. Okay. So now what is this? We can compute this.

**[21:45]** This is q of xt given xt -1 comma x q of xt - 1 given x divided by q of xt given x. This term can be computed from the above. The

**[22:14]** other term also can be computed easily. Now these are all this is what is this? This is a normal distribution. So you can actually so this is uh This is of some mu q of xt and x knot comma some sigma

**[22:44]** q of t. This is this distribution. Okay. So now I know in this okay in this equation I know what is the distributional form of this particular term. Now similarly this is this also I assume that it is of also of similar uh uh distributional form. The only thing is I assume that the mean is what I need to compute. So the zigma which is there I

**[23:12]** keep it as same. So what I'll be doing is you compute this. This is equation number 70. Okay. See, I'm skipping most of the algebra. Then I'm just going towards the the important uh uh ideas that are involved here. Now, this is uh equation number 70.

**[23:45]** Okay. So now this is of this format. Okay. This is the mu. Okay. This is the whole mu and this is the uh zigma. Okay, we're assuming it to be a diagonal matrix. So when these are the the values of the diagonal. So and then and then so what is that we want? See okay the third term which is there is the denoising term which is the KL divergence.

**[24:14]** So what is that uh we have in that KL divergence that is that is the dkl of q of xt -1 given xt comma x you're taking p theta of xt -1 given xt so I already know this this is nothing but dkl of some normal

**[24:43]** distribution XT minus one parameters by this is muq okay which is there and then some zigma q what is this p theta this is okay this is also I'm assuming p theta to be again a normal distribution which is there of xt minus one so what is the mean the mean I'm assuming is to be mu theta

**[25:12]** and then I'm assuming that the variance is the same. Okay. Now this is what the I I need to find the scale divergence between two Gaussian distribution. See we already know how to find the scale divergence between the two Gaussian distributions. So this is the third term. So this is the third term of the uh uh the elbow that we have derived. So now if I go ahead and uh solve this. Now this will turn out to be this is

**[25:43]** this is approximated with mu theta minus muq squared okay now this is now what is we call it as mean estimate so what is that I'm trying to do I'm trying to find the mean mu theta here. Okay. So now the same mean know which is there. Okay. Now we can perform some

**[26:13]** other operations which is elaborated in the paper. The same thing. So the same this is equivalent for this scale divergence. Okay. Now we can obtain it like X theta hat of XD minus X. It turns out to be proportional to this. Now this is what is known as the the sample estimate. Okay. Again there is another thing that

**[26:42]** we can do. So we can obtain now what is known as the noise estimate. Okay. This is what we call it as the the noise estimate. Okay. after some operations see I'm saying that these are equivalent representations is what I'm saying so the same thing now you can do with this is what is known as the the score

**[27:23]** estimator now all these four which are there are equivalent for this K divergence that you are obtaining. So any of the thing that you do, you know what is that you're trying to do? You're trying to go ahead and find the KL divergence between these two distributions using any of those estimate. Either you can use the mean

**[27:51]** estimate, sample estimate or you can use the noise estimate or you can use the score estimate. All four are equivalent. Okay. So fine. So that is what you have to do now. Okay. So now how do we do it is the question. See now now let's uh look at the overview that we are trying to do before we proceed. Okay. See we will have a network. We

**[28:18]** will have a we'll have a unit kind of a network. I'll explain uh unit as we go along. Now this is given xt comma t. Now you have to get some output. Now this is the model that we are considering. Okay. Now the neural network that we are considering see if we consider to estimate mean. Okay. So okay hold on. So what is the model under

**[28:52]** consideration? What is the the network output? See modern sense either are we going ahead with mean estimate, sample estimate or what and what is the target that we have. Okay. And uh what are the equations from the paper? So that these are the equations that we are implementing. So we should be very much

**[29:19]** clear on that. The first one is if we are going ahead with mean estimate the network output will be mu theta of XT comma t you give XT and T as the input to this it will give you the mu theta. Now what is the ground truth that you want? You want it to match to muq now which is a function of xt and x knot. Now this is implemented using the equations 92 and 93. So now let's look at that equations 92 and 93 for the

**[29:51]** completeness. Look at 92 93. See this is the equivalence for that KL divergence. The full-fledged algebra is here. So 93 is this. So given so you can given XT and X not. See these alphas are fixed. So given XT and X not you can get it. Now how do I get XT? Now we already know given X not and the time T I can obtain

**[30:20]** XT. Now this is nothing but scaling and shifting of X not which I know. So I can compute this and this is what the mu theta is what is my network is doing. I know this muq this term also I know I can compute it. This is the I'm assuming my coarance matrix is diagonal. I know how to compute this as well. So I can obtain this which I can minimize. Okay, this scale divergence I can get it

**[30:50]** that is the mean estimate. So now then the another estimate is we can go ahead with the the sample estimate. Now where in which the outcome is X theta of XT. Now here the ground truth is X not. This is as per 94 and 99. Equation 94 and 99. So now let's look at it. What is

**[31:16]** equation 94? This X theta now which is there. Now this is what what is 99 saying? See here X theta which is the output of the network. X not is known to us. This is the input and I can compute this whole thing. So this is what I'll be minimizing. If I'm trying to go ahead with the sample estimate model, this is the equation that I'll be going ahead. This is given by my neural network. This is I already have and these things are

**[31:45]** fixed. So I can compute them. This is what I'll be minimizing. Now if I go ahead with the the noise estimate, what is the output of the network? The output of the network is mu theta hat which is XT comma t. The input the target is this. Now this is as per the equations 115, 135 and 125 and 130. Now what is equation 115? So now let's look at them.

**[32:18]** What is equation 115 telling me? So you can see it here. See if I rearrange these terms my x knot now which is there can be obtained. This is rearranging. So earlier we know how to get XT from X not. Now given uh the XT I can get X not also. This is just rearranging these equations which is 115. Now what is 125 saying me? This 125 is saying that this is how the mu theta and then the epsylon theta is

**[32:47]** related and what is this 130 it is saying that now this is epsylon not I know what is the noise that I added at the first place and this is that what is neural network is g giving me and then these are the terms now which are computable easily this is as per the noise estimate now then the fourth one which is There is uh what is the the score estimate where in

**[33:16]** which we need to obtain s theta of XT comma t and what is the this is epsylon divide this is the negative scaled of the uh the noise which is as per equations 143 148 and 151. So we'll be implementing with respect to these equations. What is 143? I'm skipping all these algebra. See 143

**[33:45]** is how is the score is the predicted score is related to mu theta. See I'm showing in each of these cases how is the mu theta related to the outcome of the uh the neural network. Now why is this needed? because you'll be finally implementing the KL divergence which is in terms of mu theta. Okay. So that is how they are related. So that is 143 and what is 148. Now this is the loss function that

**[34:12]** you'll be minimizing your uh uh network to be. Okay. And how do I get this? This is the true score. How do I get this true score? No, this is epsilon not I know. So this is the negative scaled version of the uh the initial noise. So this is what is ST is what is given by the network and then I can compute these terms comfortably. Okay. So now

**[34:43]** what is that finally everything boils down to now if you have this XT and T you have a neural network you can get X theta hat. Okay. Now using equation 94 you get mu theta. Now you have this xt comma t you pass it if you are going ahead with this is the uh sample estimation. If you go ahead with noise estimation you get epsylon theta and using equation 128.

**[35:15]** Okay. So you can go to mu theta. Okay. So and then you have XT comma T use the neural network if the score prediction is happening you get S we get this S theta. Now that using equation 143 you get mu theta. See in all these different forms you get the you can go to mu theta. We

**[35:46]** have equations to get this mu theta. And then once you get this you you you do it either in sample estimation, noise estimation or score estimation. You can finally whatever might be the output of the neural network. We have equations for deterministic ways of getting mu theta. So now once we have that now what we can do is so once we have that mu theta

**[36:14]** okay now we can compute this which is nothing but the k divergence now once I can minimize this okay now all these are equivalent and then this is how we will be doing things okay now is it clear now this is the process of implementation of diffusion models is what we'll be going ahead head. Okay, this page gives you the full-fledged uh uh nuances that we'll be dealing ahead.

**[36:45]** So you'll have a neural network for which you give XT and T as the input. So and then you get whichever noise mean sample score whatever may be the estimate we can get mu theta and then I know ways of computing muq directly. So I'll get the uh the MSSE loss between them and then go ahead and back propagate and update the weights of the network. Okay, this is the procedure that we'll be solving.

**[37:14]** So the whole aim of this discussion was that I wanted you to show not to look at these things as four different implementations. It is one different implementations which is there in different ways. This is what I wanted to uh what do you call make it clear. Okay. So with this we complete uh this section of tutorials. Now now in the next section we will start implementing. So today we just laid the uh ground that is needed for us to go ahead with the

**[37:43]** implementation. In the next uh section of the tutorials we will start the implementations and see how it works. Okay with this we conclude today's discussion. I hope now you got the unified perspective. why it is called as unified perspective. So I am assuming that uh now you are uh agreeing to the name that is given to this paper. Now when which all the four different uh perspectives of diffusion models are unified okay so now we will be

**[38:11]** implementing this unified perspective and proceeding. Okay that's how we prefer to implement. So with this we conclude. I hope this tutorial was informative and then you unified all the perspectives. So in the next tutorials we'll be implementing them. [music] Thank you.
