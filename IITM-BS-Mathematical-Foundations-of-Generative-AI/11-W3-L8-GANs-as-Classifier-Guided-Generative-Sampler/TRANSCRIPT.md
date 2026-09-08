# Transcript — W3L8: GANs as classifier-guided generative sampler

> **Source:** https://www.youtube.com/watch?v=ga8VOW6pPeA  
> **Channel:** IIT Madras - B.S. Degree Programme  
> **Duration:** ~41 min  
> **Note:** Auto-captions cleaned lightly. Minor ASR errors possible.

---

**[00:11]** [Music] Hello everyone. Welcome to this module of the generative AI course. In the previous session, we looked at how to train a generative sampler using the variational divergence minimization algorithm. We also looked at a special case of VDM methods which was called the generative adversarial network. In this session, we will continue from there and the objective of this session is to look at some of the

**[00:41]** improvisations over the knives generative adversarial network. I will also discuss a couple of famous applications of generative adversarial networks in tasks like domain adaptation or uh distributional transformation and so on. Okay, that is the objective of this particular module. So let us start this module. So

**[01:14]** here the first thing no just a quick recap of what we were doing. So in the generative adversarial network which is a special case of variational divergence algorithms. So we had two neural networks. One we called the generator network and there was a discriminator network. The idea is to start with an arbitrary random variable and use a neural network to transform it to random variable of interest by minimizing a distributional divergence metric. So we used f divergence as the

**[01:42]** distributional divergence metric and minimizing the f divergence directly is not feasible. Therefore we constructed a lower bound on the f divergence which turns out to be this particular objective. So we optimize we we minimize for that lower bound and while constructing the lower bound the construction of lower bound itself involves an optimization problem which is uh solved which is solved over the class of uh functions which is represented by the

**[02:11]** discriminator network. Okay. So we alternatively solve a maximization problem which is finding the lower bound on the f divergence and then once we construct the lower bound on the f divergence we minimize that and we alternate between these two optimization problems. Right? So this is the broad description of what we saw so far. Okay. Now I would start today's module by giving you another interpretation of the generative adversarial

**[02:41]** network as classifier guided generative models. So let us interpret interpretation of interpretation of a GAN as classifier guided classifier guided generative samplers.

**[03:20]** Okay. Now what we have what we saw so far was that again was a special case of variational divergence minimization algorithms where we start with an F divergence construct a lower bound on it and then optimize for it. Right? So this is another interpretation of a GAN as a classifier guided generative sampler. So let us look at it. As usual, what we have is a data set with n samples of data which are drawn iid from px. This

**[03:50]** is our as usual setup. Now we also have this uh sampler which we are calling as uh the generator which would take samples from an arbitrary distribution like a gshian distribution. take samples from a normal distribution g theta of G and we get samples from the output of the generator network

**[04:20]** which we designated as P theta of X. Okay. Now the goal is [Music] to what is our objective? Our objective has always been that we want p theta of x xcap to be close to px or rather p theta oft1 and p theta of x to be uh equal to px and we

**[04:49]** do it by minimizing the f divergence. We know that wait we know that interpretation of uh a GAN right now let's say that there exists a classifier suppose suppose there is a classifier there is a binary classifier such that now let's call this

**[05:17]** as DW the classifier as DW so we have DW W such that DW is 1 if X is coming from P theta. Let's call this DW of X and it is zero if X is coming from okay so maybe the other way around it's easy to interpret it that way. Whenever x is coming from p x this

**[05:47]** classifier will give one as the label and whenever it is coming from p theta this classifier is giving zero as label. So now DW DW is a binary of between the samples

**[06:18]** of PX and P theta. So let's say that we have a binary classifier that would classify between the samples of px and p theta. Now the question is so let's say that the question that we would want to ask is the following that can can dw of x which is the classifier that we just described be used

**[06:54]** P theta and px close to each other. Right? Now the question that we would want to ask is can we use this classifier to make p theta and px close to each other. So it turns out that a classifier can be used for making uh p theta and px closer. So let me write the classifier here. Let's say that we have this classifier which is a binary classifier. So this would take uh x or

**[07:38]** xcap both as input right dw of x or xcap. And what this would give is one if it is x 0 if it is xcap. So this is the classifier that we have right now. The question is can this sort of a classifier be used to make px and p theta closer? Okay it turns out that it can. So the answer is that suppose you have a classifier.

**[08:07]** Okay. Now let's say that so tweak tweak theta okay which are the the parameters of g [Music] theta parameters of g theta till the classifier fails till the classifier fails to

**[08:37]** distinguish to distinguish between between the samples of between the samples of px and p theta. Now this can be a strategy that can be used. So what we are saying is let's say that we have a pre-trained classifier okay that would classify that would give one if the samples are coming from px and it would give zero if the samples are coming from p theta. What am

**[09:06]** I saying is that you one should keep changing the parameters of the generator till this classifier fails to distinguish between px samples of px and p theta. So think about it. When does the classifier fail to distinguish between samples of px and p theta? It's it's when the I mean one of the cases when the classifier would fail is when px approaches p theta or rather

**[09:36]** p theta approaches px because if p theta becomes same as px then the samples from px and samples of p theta becomes indistinguishable and therefore the classifier fails. Okay. So now but the thing is while I said that uh one of the failure cases of classifier is when px approaches p theta classifier failing

**[10:07]** need not imply pH being close to p theta. The converse is not true. So what does that mean is that however [Music] however failure of the classifier failure of the classifier classifier does not imply or need not imply need not imply the distributions

**[10:38]** matching. Okay, px being equal to v theta. So why is that? Let me just uh show you a uh a pictorial example. Let's take a take uh the counter example to show this. Okay, let's look at a counter example. So let's say that our data is in two dimensions. Okay, this is for uh ease of understanding. Let's say that we

**[11:06]** have our data in two dimensions. So this is the space in which the data lies and uh I would uh say that let's the data is distributed this way. So let's say that this cluster represents uh data from px okay or rather let's say that this represents data from p theta and we have the data from px represented uh in in using this

**[11:37]** particular cluster let's say that this data is from px okay we have uh the crosses and dots dots represent data from b theta and crosses represent data from px Okay. Now let's say that uh there there was a classifier. So this was the classifier that was initially there. So let's say that this is uh let's call it uh DW1 which is the classifier that we

**[12:10]** had. Okay. Now what we are doing is we are we have the uh we are tweaking the parameters of theta. Okay. such that this classifier is failing. Now what do you mean by changing the parameters of theta? as one changes the the parameters theta right the this cluster moves around right because the uh if one changes theta p theta changes and this moves around okay now okay so let's say that this is being sampled from p theta yeah now when we change our theta such

**[12:41]** that this classifier fails all that the uh uh the uh optimization changing theta should do is move this cluster right somewhere here. Okay, let me call this P theta 1 for the first set of parameters and this shall be P theta_2 after we say that okay we want our classifier to fail to distinguish between PX and P theta 1. So

**[13:12]** note that the way this binary classifier is working is all the samples that are to the left or above this particular line okay given by specified by this classifier belongs to one class and everything that is below this line or to the right of this line belongs to the other class because it's a binary classifier. Now when I change when I request uh my optimization problem optimization problem to change the theta

**[13:40]** in such a way that this classifier fails it's very easy for the optimizer to push my theta in such a way that all the data that was coming from P theta 1 is just moved to the other side of the classifier so that the classifier fails right when The data comes to this side. Right? Here in this scenario, for this scenario, the classifier fails to

**[14:15]** classify rather I should say the classifier DW1 the particular classifier that we have DW1 fails here to distinguish between P theta 1 and P theta_2. Okay. So this means that when my px overlaps with p theta we know that the classifier would definitely fail. failing this does not imply P theta to

**[14:47]** be rather okay I'll have to say that these not this does not imply that P theta is matching with PX right so now this is the counter example where you can make your classifier fail even without making px and pt theta to be the same. Okay. Now what do we do? So now this algorithm that we just saw, right? You to keep tweaking your theta till the classifier fails to distinguish between samples of

**[15:16]** px and p theta no longer holds because one can make the classifier fail. Okay. In spite of p theta and px not being equal to each other. Okay. So now how do we solve this problem? The final I mean one way to solve this problem is you know one therefore hence the classifier has to

**[15:44]** [Music] be simultaneously simultaneously tweaked or changed simultaneously change along with the generator. Right? So now all this analysis that we did was for the case where the

**[16:12]** classifier was fixed. Now let us not fix the classifier. Let's say that our classifier can also be tweaked. So now what will happen? Let's come back to this diagram. Now when we have our classifier when we have the flexibility of changing our classifier to then what might happen is that once we move our data to P theta_2 from P theta 1 what we can do is we can simply change the classifier as

**[16:41]** [Music] well the classifier can be this. So let us change the classifier and call this DW2. Now under DW2 the the classifier DW2 does not fail to distinguish between PX and P theta_2 right. So now P theta_2 has to be moved again to some other place to ensure that the classifier DW2. So let's call this uh uh new

**[17:14]** cluster P theta 3. Okay. So from P theta_2 the data has to move to P theta 3 to ensure that the classifier DW2 also fails. Okay. So this is the idea that one would keep alternating between a classifier. Okay. Maybe I just take this down because this might be confusing. Or maybe I'll draw the

**[17:44]** coordinate axis in a different color. Let me just do that. uh this is the coordinate axis right. So let me call this uh we have a two-dimensional feature. This is uh x1 and this dimension is x2. Our data x now is in r2. Okay. So every x i here has

**[18:15]** this particular form that it is. So every data point x i now has this particular point particular form. It is a tuple of two data points. Call this uh x1 i and x2 i. Okay. So we are talking about two dimensional data. So the idea is the following. Right. So what we just said is that suppose you want to make p theta close to px. You can use a classifier to

**[18:45]** do so. The idea is to build a classifier that would distinguish between samples of px and p theta and keep tweaking the parameters theta till a point where the classifier fails. Okay. Now we saw that if you fix the classifier then it is very easy for the uh optimization problem to optimization to simply find out uh a theta such that the

**[19:17]** classifier fails but p theta is not equal to px. Right? So now when the distributions become equal to each other the classifier would definitely fail. However classifier failing does not mean that px is equal to p theta. Okay. So now what is to be done is that one does not have to just keep tweaking theta by having a fixed classifier. The idea is to keep moving the classifier as well such that every time you move your data

**[19:47]** so from one point to the other till a point where p theta gets p theta gets overlapped with px such that there is no classifier that can classify between these two points. That is the idea. Now one might get this question that the uh data okay or the generated data or p theta this cluster can keep alternating between alternating between multiple points and the classifier one there

**[20:14]** there can be a classifier that would fail for one particular configuration of p theta and uh then you know to make sure that that classifier does not fail you move your p theta to a different location and uh the classifier gets alternate to the previous location and this can get this can get stuck for eternity. This is a possibility. Okay, this is what is called as the mode collapse problem in GANs. Okay, where the optimization problem that we are

**[20:42]** solving which is the the min and the max problem that we are solving would not get to a good saddle point that your pta may not get overlapped with px at all. So that's a possibility right? So that is why it's one of the known failure modes of GANs where so you start with some P theta okay then there is one classifier that is there and now you want this

**[21:10]** classifier to fail you know move this P theta 1 such that this classifier fails and you move it to this point okay this classifier failed then you learn another classifier okay DW2 and uh now you want to move your P theta 2 such that this classifier fails Now, now your algorithm can just put your theta back to this particular place. Okay? Where both of these the second classifier also fails. Now you change your classifier again to ensure

**[21:37]** that this fails and it'll come back to this P theta_2 and keep alternating between P theta_2 and P theta 1 and never approach PX. That's a possibility. Okay. If you translate this idea to the variational divergence minimization framework, what we are actually saying is that we are alternating between constructing a lower bound on the F divergence and minimizing this. Right? The lower bound that we are

**[22:05]** constructing is similar to learning this classifier. Okay? So the tighter the lower bound that we construct on F divergence, the better this classifier is going to be and the better the minimization of the F divergence is going to be. However, practically speaking, it is very much possible that the lower bound that we construct on the F divergence or the classifier that we are learning is very weak in the sense that when the minimization of that weak lower bound that is constructed may not

**[22:35]** make the generated data p theta_2 come anywhere close to the true data px. Okay. and it can keep alternating between different kinds of uh uh saddle points okay which are not making my p theta and px close to each other. This is a known failure case of GANs. Okay. So please keep this mind keep this in mind. However, the idea here is that we not only change the in the classifier guided interpretation for

**[23:04]** a GAN. What we are doing is we are not keeping the classifier fixed but we are also retraining the classifier simultaneously along with the generator. So that is the idea. Okay. So now with all this let us formulate the cost function with this interpretation. So formulation formulation of classifier guided sampling. So we are formulating a

**[23:38]** classifier guided generative sampler. Now suppose suppose dw as I said is a classifier. It's a function that would take samples from X and maps it to 0 and one. So it can give a real number between 0 and one. Now we'll say that let dw of x

**[24:16]** likelihood the likelihood of the sample x likelihood of the sample x coming from px. X. Okay. So in our case, this DW is a neural network. It will just give you the likelihood of I mean we want that to represent the likelihood of the sample X coming from PX. Okay. Now log of so we know that the objective is to

**[24:49]** maximize the likelihood of log likelihood of the sample x coming from px. So now we want to find our W. Okay, such that the objective is okay. So let me rewrite objective. The objective is

**[25:21]** to is to maximize maximize the log likelihood log likelihood of of X coming from PX. So for any classifier right the objective is to maximize the likelihood

**[25:50]** of data coming from the distribution right distribution of interest that is the objective of any classifier. So we want our um classifier to be learned in such a way that we just want to maximize the likelihood of the data coming from px. Okay. So what we want is our W star for the classifier has to be such that we maximize the likelihood of obtaining the data from

**[26:17]** PX. Okay. So that's how we will train the classifier. Now since this classifier also sees sample from Xcap. Okay. We want what is this? This is maximize the [Music] likelihood likelihood of X sampled from PX. This is what this

**[26:47]** objective is doing. Okay. There's another objective that we would want to optimize which is we also the classifier. So classifier should

**[27:16]** also should also maximize maximize the not coming

**[27:41]** from not coming from vx when when xcap is sampled from p theta. So what does this mean? The classifier has to also maximize the likelihood that xcap does not come from px. Okay. Whenever it is sampled from p theta obviously right. So how do we write that mathematically? Now when if dw of x represents the likelihood that x is

**[28:09]** coming from px then 1 minus dw of xcap would tell you the likelihood that xcap is not from px. Right? So what is this? This is the likelihood likelihood that xcap is not sampled from px. This is that likelihood. Okay. What do we want to do? We want to maximize this as

**[28:37]** well. So we want the log of log likelihood of this under the expectation whenever the I mean the expectation is now taken over P theta. Right? Because whenever the samples are drawn from P theta, we want the log likelihood of those samples not coming from PX to be maximized. Right? So that's that's what the classifier is. So now we want our W star also to maximize this particular likelihood

**[29:11]** which is the average or expected log likelihood of Xcap not coming from PX whenever Xcap is sampled from P theta. Okay. So now basically what we are saying is what is this? Let me write this. This is this is maximize This is maximizing the likelihood the likelihood

**[29:41]** of xcap not being sampled from px. Okay. Whenever it is sampled from p theta. Right? So we want two things to happen for this particular classifier that assuming that dw represent the likelihood of a sample x coming from px. We want two things to be happening that whenever the sample comes from PX, we want the log likelihood of that to be maximized. Right? So we just take the

**[30:08]** expected log likelihood of DW of X whenever the samples are coming from PX and we want that to be maximized. Okay? So I can write this as uh whenever X is being drawn from PX. Okay. Now we also need I mean note that this DW also takes Xcap as input where Xcap is coming from P theta whenever XCAP is drawn from P theta this DW of X should give us zero right I mean that is

**[30:37]** how we have designed our DW which means that it has to maximize the likelihood that Xcap is not from PX okay whenever it is coming from P theta we represent that as the expected value of log of 1 - DW of Xcap whenever Xcap is coming from P theta. So 1 - DW dw of X represents the likelihood of an X coming from PX. 1 - DW of Xcap would represent the likelihood that whatever this argument is Xcap is not from PX. And we want that

**[31:08]** to be maximized whenever my XCAP is actually not from PX or rather it is coming from P theta. Right? So we want these two objectives to be satisfied whenever we are learning W star. So let us combine that. So the final objective the combined

**[31:38]** objective for the classifier training. Classifier training is as follows. What do we need? Whenever the samples are coming from whenever the samples are coming from PX, we want that objective to be maximized, the log likelihood to be maximized. And whenever the samples are

**[32:07]** coming or not coming from px or rather they are coming from p theta we want the log of 1 minus that particular likelihood to be maximized. Okay this is my final objective. So what do I need is that I want whenever my samples are coming from px the log of that particular likelihood has to be maximized and whenever the samples are coming from v theta the

**[32:35]** likelihood one minus the likelihood log likelihood of that particular thing has to be maximized right so we want my w the classifier to be such that this objective is maximized W star has to be such that this particular objective has to be maximized with respect to W. So I mean one can very easily note

**[33:05]** that this is nothing but the lower bound that we constructed on the F divergence. Right? So this is the lower lower bound that we constructed for the F divergence. So let us call this uh J theta W. Okay. Now what is the objective for the generator? Now the objective for the generator objective for the generator

**[33:30]** network or the G theta network is such that the classifier has to fail. Right? That is the way we have such that the classifier has to fail. Right? We have to tweak our theta

**[33:56]** in such a way that the classifier has to fail. Now, how do we do that mathematically that we simply invert? So, invert the optimization problem that is for the classifier. That's all. invert the optimization for the classifier. We do that then we'll ensure that our theta will be such that the classifier objective is inverted. So

**[34:26]** what do we do? We just minimize the classifier objective with respect to theta. Now while the classifier is trying to maximize a particular objective with respect to its parameters, we will just minimize. What do you mean by inverting? change the optimization problem from maximization to minimization that we find out the generator parameters simply by minimizing the classifier objective. Right? So finally what happened is that we get our theta star and war which are

**[34:58]** the generator and the classifier parameters such that you have one objective function and the generator is trying to minimize that objective function and the classifier is trying to maximize that same objective function. Right? And what is this objective function? This objective function is try trying to classify between the samples of px and p theta. Whenever the samples are coming from px it is giving you one and whenever the samples are coming from p theta it is trying to give zero. What we are doing is we are trying to find

**[35:27]** the generator parameters such that the classifier fails and we are trying to make the classifier learn the decision boundary such that the generator uh the generated samples are being able to be distinguished from the true data. Right? So there is an adversarial game that is going between the classifier and the generator and hence the name the adversarial networks. Right? One of the reasons why this is called the adversarial networks already told you

**[35:54]** why this is called the adversarial networks because what we are trying to do is trying to solve a saddle point optimization problem which involves one objective function and two set of parameters. uh while one set of parameter is I mean one set of parameters is being being chosen such that the objective is being maximized the other set of parameters are being chosen such that the objective is minimized right that's why the name the adversial optimization so I can tell you this this is the adversial

**[36:33]** part okay this is the classifier guided interpretation for the GAN so note that from mathematical standpoint and what is happening right algorithmic standpoint whatever we saw as variational divergence minimization is absolutely the same as classifier guided training as well but this is just another interpretation of how one can perceive the knife uh gang okay please note that this interpretation is not generalizable across different other f divergences right why for other f divergences

**[37:03]** there's no guarantee that the T function that we use to bound the F divergence does not have an interpretation that the T function is a classifier. only with this particular choice of DF divergence that our T function has a represent has a has an interpretation which which which which aders to that of a classifier right and once this can be interpreted as a classifier one can see this entire process of adversarial

**[37:31]** optimization as a way to guide the generator or the sampler using this classifier. So what is the story? We want to transform a gian random variable into the random variable of interest which is px in this case. And we do that via a classifier. What is this classifier trying to do? This classifier is trying to classify between the samples of px and p theta. The moment this classifier fails to sample between uh sorry classify between

**[38:00]** samples of px and p theta we are done. Correct? But if you fix the classifier then it is very easy for the generator to make this classifier file and still not make px to be equal to p theta. So what do we do is that we not take a fixed classifier but also change the parameters of the classifier along with the generator parameters such that we keep changing alternative alternating between the classifier and the generator

**[38:29]** parameters till a point where t theta actually overlaps between px diagrammatically speaking what's happening is that we are moving this cluster around right I mean first you have some p theta1 there is a classifier which is classifying between these two p theta and px Right? And then you move your P theta to some other location P theta_2 such that this classifier fails. Then change the classifier from DW1 to DW2 such that the classifier is again

**[38:57]** able to classify between P theta_2 and PX. Okay. Then change your P theta or theta in such a way that P theta_2 goes to P theta 3 such that DW2 fails and keep doing this. So this is the idea from an F f divergence minimization perspective. What we are doing is we start with some generator parameters theta. Okay. With that we construct a lower

**[39:23]** bound to the f divergence between px and p theta and then we change the p theta depending upon that particular lower bound that we have constructed. Once we change the p theta the up divergence also changes right. So therefore we'll have to construct another lower bound between that new p theta and uh the existing px and we alternate between these two. So basically what is happening is that start with a theta

**[39:51]** construct a lower bound on the current I mean on the current f divergence and then minimize that f divergence. Okay in the hope that the uh the minimization would actually make the lower bound the new lower bound that we construct a little tighter. Okay. Then you change the theta again, construct another new lower bound and minimize that and keep alternating between these two processes from a classifier standpoint. Move your P theta such that the current classifier fails. Okay. And

**[40:21]** then tweak your classifier such that under the current theta you can distinguish between the real data and true data and keep alternating between these two processes. Okay. So as you can see uh one can get stuck in uh a place where uh the classifier keeps fooling the generator and the generator keeps fooling the classifier so to so to speak right classifier makes the generator fail and the generator make classifier fail in the alternate

**[40:48]** steps and that is a known failure case of GANs where they can get simply stuck between these two and uh you will never reach px by tweaking p theta that's a possibility Okay, that's why people say that training a GAN is is is is a non-trivial task. Okay, and it is the GAN table is known to be sorry GAN training is known to be uh notoriously unstable. Okay, so now we will move on

**[41:18]** to different kinds of uh improvisations of GANs.
