# Transcript — Lec 52 Transfer Learning and Knowledge Distilation

> **Source:** https://www.youtube.com/watch?v=5f2_TocU_CU  
> **Channel:** NPTEL - Indian Institute of Science, Bengaluru  
> **Duration:** ~37 min  
> **Note:** Auto-captions cleaned lightly. Minor ASR errors possible.

---

**[00:02]** Welcome back. So let me introduce you to this idea called transfer learning. So you recall that neural networks are functions, right? So given uh data X in RD, let's say that you also have Y in R doesn't matter if it's a uh supervised or unsupervised. You have this. Suppose we learn a neural network. Suppose learned

**[00:51]** via ERM. Okay. Now there is some task and uh we have uh trained a neural network and learned it. Okay. Now recall that uh a neural network is a composite function, right? Uh it's a composite functions of linear and nonlinear transformations. So you can tap the output of the neural network at any of the compositions.

**[01:22]** What do I mean by that? So since GV is a composite function the outputs can be tapped at any layer. Okay. Now I'll define Z to be

**[01:59]** G of X. Okay. At layer L. I can do this uh write this as G star. uh denoting that uh star is the set of parameters that you have gotten after performing erm &gt;&gt; of course yeah erm is training after you train you get a set of parameters that's that's what I'm indicating by g star now

**[02:29]** z can be the output that we have obtained on data at the l layer of the neural network and this l can be anything. Okay. And this is typically referred to as the embedding of X. Okay. Um from you might have heard of these terms,

**[03:01]** right? BERT embedding is exactly what is done. So BERT is a particular architecture that is trained for a particular task. Now you train that okay and by the way this G star okay is also called as a pre-trained neural network and the standard libraries today will give you the parameters of neural networks that are pre-trained on several

**[03:29]** data sets and tasks. So this is a pre-trained neural network. Now what do you do with a neur pre-trained neural network? You can tap the output of the pre-trained neural network at any intermediate layer. Okay, for a given data and do what with it? You can use Z for any further downstream tasks. It can be supervised, unsupervised. Now

**[04:03]** think of Z as a feature that you have extracted from that neural network and use it for any downstream task. Okay. Now this is one form of uh embedding extraction and transfer learning. Typically this is not what people refer to as transfer learning. Uh okay any questions on this uh so far? Yeah, &gt;&gt; like when we using any feature for the downstream task, how do we understand that which layer or which feature

**[04:31]** &gt;&gt; that's a hyperparameter? So for instance, uh I think you know one particular layer of imageet uh I mean inception trained on on imageet data set is taken uh to compute a metric called fidets inception distance. Okay. So now how did they come up with that particular layer is through experimentation. So they saw which feature corresponds to the human perception the most and took

**[04:58]** that particular layer. Okay. This is a hyperparameter. Okay. So now this also s can transform your data. Let's say that you are using images. Okay. Can convert images into vectors. Okay. So why is this useful is because what happens is even though the distributions are different let's say that you are looking at two different data sets okay uh one is uh the data set of uh uh normal images or rather uh free form

**[05:30]** images the other is the data set of let's say particular type of uh uh medical images or something even though these two data sets are being sampled from two different distributions and I assumption breaks if you do Since these are both coming from the same image domain, perhaps the features that have been learned in the initial layers of the neural network that is that is trained on the free form images are useful for the other domain as well. So what you do is that take the neural

**[06:00]** network that is pre-trained on the large scale data set and tap the the uh the features from one of the earlier layers and then use it for the downstream tasks. So this is very similar to doing this, right? Let's say that we have this function G. this is layer L and then we get Z here.

**[06:30]** Correct? This is all fix. This is all uh freezed and you are not training this. You can append another neural network here which is some f theta on Z. Okay. And perform erm on f theta for your particular task. You get it right. So this is what is typically referred to as transfer learning where I mean you're sort of transforming uh the the learning. Right. So you take the network that is

**[06:59]** pre-trained on some other data append another neural network. When you do back propagation how do you do this? the forward pass does not change to toward I mean in when you do the uh backward pass uh you make the parameters of uh this neural network freeze right or non trainable you treat it as a fixed function that's it so it is like you have uh you have x that is being passed through a deterministic

**[07:26]** function gv and then you have a neural network on top of it and and all these standard uh uh libraries such as pyarch will give you uh capabilities of freezing a particular layer. Okay, only train f theta. So this is transfer learning. When you do transfer learning, there's no there's no hard and fast rule that you should uh you know freeze only a few uh you should always freeze a few layers of the pre-trained neural network. You can

**[07:53]** take a pre-trained neural network, use that as an initializer and train the entire neural network as well. You see what I mean? So this so what is why is this useful? Typically in a neural network when you train we know that uh that would be trained using uh gradient descent and gradient descent is an iterative algorithm and we need initialization. Random initialization is what is done typically. Okay. But now instead of

**[08:22]** doing a random initialization it is better if you initialize the parameters to be the parameters of a neural network that is trained on a similar data set. that I mean in the parameter space search space is always the parameter space in the parameter space you're already starting in a region which corresponds to some no images some useful distributional data and from there you navigate whatever data that you need that's also another way to do transfer is that is that okay yeah in fact uh one

**[08:53]** can do in this context right I mean is as I said these can be used as Lego blocks one can actually do uh what is called as multitask learning where use the same uh set of features to do some classification here and some bounding box regression here and multiple tasks on the same

**[09:22]** set of features that are being run. Now when you do back propagation there is one path towards this task and there is one other path towards this task and so on. The only thing is the gradients will get added here before it goes to the input. That's it. So this paradigm is called multitask learning. In fact you can do this from scratch as well. Meaning you have a neural network. So

**[09:50]** basically there are two functions two composite functions. Okay, but initial compositions of these two functions are same up to a few layers. That's it. You're learning two composite functions together using gradient descent. That's the idea. What is the intuition behind it? The intuition is that suppose you are solving uh two different tasks on the same data. The initial set of features that are to be learned to uh solve a particular task on the same set of data are the same. So you share the

**[10:19]** parameters across different tasks. All right. So, this is about transfer learning and pre-training and multitask learning and so on. Different paradigms of learning. Any questions on this? &gt;&gt; Well, you are transforming the

**[10:54]** parameters because the initial set of parameters are being transferred by not training them at all. &gt;&gt; Yeah, I mean you are doing learning on uh the you this is also a sort of regularizer, right? out of thousand parameters that are there I'm saying 500 parameters are fixed to a particular value which is given by another neural network that's it backation freezing means not keeping track of &gt;&gt; oh okay it depends on what layers do you

**[11:24]** freeze suppose you freeze layers suppose you freeze some uh intermediate layers and try I mean not freeze the earlier layers then you have to keep track of those gradients because those gradients will be with respect ect to constant functions will be constant. You'll have to carry them through because if there are layers that are be behind the layers that are frozen then you have to remember the gradients otherwise you don't have to remember the gradients depends upon what layers to freeze see here in the in the version that I wrote

**[11:53]** all the layers between input and the representation is sort of frozen you don't have to do that you can freeze intermediate layers as well which is which is not which is seldom done it's not often done typically the the some of the layers from the initial uh input put two of you uh a particular layer in the neural network is what is frozen in in which case you don't have to uh see you still have to uh get the inputs to get the final loss and the gradients

**[12:24]** right so yeah it's a matter of detail anything else okay so this is see nowadays uh most of the ML tasks or neural networks are not trained from scratch. Okay, most of them are transformed because and and okay, so one interesting question that I thought some of you would ask is as follows, right? Uh which was not asked is that I told you that

**[12:53]** this GE is trained on a supervised task. Now if it if it's trained on a supervised task would the features okay so first of all one point is the the kind of features that are learned or the transformation that is learned by GV is completely dependent on what sort of task it is being &gt;&gt; trained for and we know that all supervised learning the goal of all supervised learning in ERM is to

**[13:21]** estimate the posterior density of y given x if we do a supervised if we solve a supervised task. We are estimating P of Y given X. Why should a a composite function that estimates P of Y given X be good on some other task where Y does not exist? Nobody asked that question. That's actually pretty important. In fact in most of transfer learning uh

**[13:51]** this GV is not trained on not trying to estimate P of Y given X. This G is trying to estimate the marginal PX. All methods that are designed to estimate marginal PX are called unsupervised learning methods. We will see examples of some of it later in the course. One example can be that you have a neural network that would simply reconstruct the data back or den

**[14:19]** noiseise the data start from X. Okay, add some noise to data and ask your the task of the neural network is to estimate PX from PXcap where Xcap is the pter version of X. That's called autoenccoding. We will see examples of it later. So those kinds of neural networks are the ones that are used for that are used as feature extractors for transfer learning because you're estimating it's it only is very natural right if you if you if you have a method

**[14:46]** that would estimate the marginal then you can perhaps use some of it features to solve the conditional tasks but if the neural network that is that is built to estimate the conditional distribution you can't use it for the other kind of conditional distributions therefore most of the uh the backbone or the embedding extractors are trained on an unsupervised learning task which is equivalent to saying that they are trying to estimate the marginal distributions and there are multiple ways of doing it. One way of doing it is

**[15:15]** as I said pertub the data right and ask it to d noiseise the data. So that is called mast encoding right? You just take a sentence, take a sequence, remove some of the tokens and ask your neural network to predict those tokens. Fill in the blanks or give it t minus one tokens, ask it to predict the next token. That's the auto reggressive way of doing it. And that's exactly how the modern day LLMs are trained.

**[15:50]** Okay. So there is another learning paradigm where a pre-trained neural network is used. Okay, which is called knowledge distillation. &gt;&gt; Self-supervised tasks which is that

**[16:16]** there are multiple task on which it was trying. one was that you rotate the image and predict the angle of rotation. Okay, you mask the image and predict the uh the mass pixels and so on. Perhaps it's a good idea to talk about uh self-s supervised learning as a paradigm in this course. I'll do that later in the later in one of the classes. I'll spend some time on

**[16:45]** self-supervised learning techniques. Now one after unsupervised learning we'll do self-supervised learning as well because most of the modern day ML uh is done in a self-supervised manner. And by the way another side uh remark here I told you that all the methods uh that are that are designed to estimate the marginal distribution can be used as feature extractors. Right? All generative models by definition are marginal distribution estimators. And

**[17:14]** that's why all generative models the side positive side effect of all generative models is that they are also embedding extractors. So I'll show you this mathematically in my next course right when I talk about generative models marginal estimators all of them are embedding extractors. Okay. Okay. So let's look at uh the idea of knowledge distillation. Uh I'll only talk about the empirical way of doing knowledge uh distillation. uh but there

**[17:41]** is a very nice basian framework on why this should work and so on again I'll do it in my next course the idea is as follows so suppose so there is some data again x y that's sampled from pxy and a neural network g is trained on a task

**[18:16]** Now let f theta be another neural network. than

**[18:49]** GV. This is a typical case but the theory does not stop you from using another neural network with arbitrary number of parameters. So we have two neural networks. One is GV and the other is F theta. Typically F theta will have lesser number of parameters much lesser number of parameters compared to GV. And this is referred to as the teacher network. And this

**[19:16]** student network student will have lesser number of parameters compared to a teacher typically. Right? So there are exceptions always. Okay. Yeah. So what's the task here? The task is why do we do this? uh most of the cases knowledge distillation is done because you know you want to embibe the knowledge that is in the teacher network onto the student

**[19:44]** network. By the way this was actually proposed by Jeffrey Hinton in 2012. Uh the math followed later a very nice math in terms of you know vision uh uh okay I I I'll just tell you what it is doing in a while. So the reason we do this is that suppose you have a very large neural network. Okay. And uh you have a hardware which cannot support that larger neural network. You want a neural network that is smaller in size but have

**[20:13]** a similar performance as that of the larger neural network. What you do is that you construct another neural network f theta which is a student neural network. And suppose you have access to some data or some samples from uh pxy. What you can do is simply do an erm on f theta. Right? Because you have access to data. Do erm on f theta. But however this may not give you performance that is as good

**[20:41]** as a teacher network because teacher network is one larger in number of parameters and two it has seen more data compared to the student network. Now the question is the question here is how to get the performance of the student network. right with this question there's this

**[21:32]** these class of methods called distillation knowledge distillation that came okay so how is it done is as follows so let's say that uh typically in most cases uh teacher network is not trainable Okay. So the parameters of the teacher network are fixed. You have this uh g star and uh you have a smaller student network f theta

**[22:00]** and we have access to sum of the y. So what is done is as follows. So you tap the output of the teacher network at some layer. &gt;&gt; This is pre-trained. This is pre-trend from Lith layer. Okay. And now you take the Lith layer. Let's call this ZT. This is ZS. Z G Z is uh uh the output of the student

**[22:31]** network at the LTH layer. Now ensure that the dimensionality of ZT is same as that of ZS. See, have you noticed this? Uh if you misspell something and write it, it visually looks something bad. Have you noticed this?

**[23:01]** See what I mean to say is I just missed this uh N. Okay, I write it. I wrote it as this. You don't even have to read the misspelled word. You have a visual feedback that would say that this is wrong. Have you noticed that? &gt;&gt; Exactly. So some internal representation that the teacher network has already done, right? Which not even audiary. I'm just saying the visual representation is completely gone. you have to go back and

**[23:29]** see what was the mistake then right you don't have to even read I mean the very first visual feedback will tell you that this is wrong anyway so ensure that these dimensions are the same now what is done is while you train the the the training objective or training of uh I'll write it the other way around so regularize the teacher network oh Sorry. The

**[23:58]** student network expected divergence okay between the distribution of ZT and the distribution of ZS is minimized. Take the expectation out.

**[24:35]** You understand? Now this ZT is a random variable, right? Because we are sampling X from a distribution, the neural networks. And okay, by the way, one other point to be made. I don't know if you noticed this or not. Neural networks are deterministic functions. What do I mean by that? Once it is trained, okay, they give you the exact same output given a particular input. There is no stoasticity associated with it. Okay, the only source of stoasticity is in the data.

**[25:05]** Neural networks are deterministic functions. Remember this there is no sampling with there is no stoast stoasticity associated with neural networks. Now what am I saying is since neural networks are deterministic functions and X is a random variable when you pass it through the neural network if you tap the the embedding at some else layer that also is a random variable which has a particular distribution because it's a function deterministic function of a random variable that has a distribution PZT and

**[25:35]** this will have a distribution PZs right now what am I saying is that while I train the student network. I'm regularizing the distribution of PZES to be the same as that of distribution of PZT. What is the objective? What is the intuition here? The intuition is that whatever feature that this neur this teacher network is learning at layer L has to match with the features that the student network is learning at layer L.

**[26:05]** Now, how do we use this? How how do we calculate this divergence metric between these two distributions? Again using statistical methods assume some distributional form on this. If you assume both of them to be normal normally distributed then k divergence between these two will become the squared error. If you have two two random variables that are normally distributed with zero mean and unit variance you if you assume them to be having a particular

**[26:33]** distribution then k divergence is simply a squared error loss. So you just add this additional square error loss while you are training the student network. So final uh the uh empirical risk for the student network will be so minimize. So now we have to get our theta that would minimize this divergence between PZT and PZs

**[27:01]** plus the usual empirical risk uh or the usual loss function that we define on the student. The second term is simply the supervised training. Right? The first term is is called the uh the teacher matching term where you match the distributions of the representations that have that the

**[27:29]** teacher has learned and the one that the student is learning. Do you see that this do you see that as regularization? We are not regularizing in the parameter space. We are regularizing on the activation space or the embedding space. And I told you right regularization can be done either on the parameter space or on the activation space and so on. This is a type of regularization but the regularization is coming from the distribution of the teacher. Okay. Now

**[27:57]** today in the commercial world uh when a company releases uh LLMs they release LLMs of different sizes. So there's 100 billion parameter model, there is 170 billion parameter, 30 billion parameter and so on. Right? These smaller models are the distilled version of the larger models. Okay? Uh distillation comes with a

**[28:23]** catch. The catch is that it'll you know degrade the performance because the number of parameters in the student is lesser. But you'll have to deal with it. I mean the hope is that while while you distill uh the performance is not degrading but it'll always because the number of parameters are much lesser compared to the student network sorry the teacher network and what is it doing mathematically is that there is a there is why should this work it can actually be theoretically shown that uh that this entire process is actually minimizing the k divergence between the

**[28:54]** true posterior that we would want to minimize okay and uh a posterior by assuming ing that there is label noise. Don't worry about it. I'll do it you know in detail in my next course. But yeah, so there is a very nice when Hinton first came up with this idea. This divergence was called as dark knowledge. Okay. And but in 2021 there's a paper that came that uh analyzed this and you know it it talked about the the title of

**[29:22]** the paper is a basian uh framework on distillation where they give theoretical reasons so as to why this this thing is actually working but yeah empirically this is what it is. All these smaller models that you see today are the distilled ones. Okay. And this can be done for any task as you can see right you can you can distill a CNN you can distill an RNN you can distill a transformer or an MLP or anything doesn't matter yeah all these are architecture agnostic any questions

**[29:52]** on this &gt;&gt; three elements that are students &gt;&gt; come come again &gt;&gt; one will be teacher the other will be students in fact I should not be saying this on camera but the thing is there have been acquisitions that uh one particular geog geography uses the LLM strain by another geography and distill them. You are not supposed to do that. I mean when you sign the uh the user agreement most of these LLM providers will will

**[30:22]** make you sign will make you agree to the fact that you will not be using their inference to do anything further. Okay. But yeah and please note that if you were to distill there should be a way to tap the output from the intermediate layers of the teacher network which may not be the case always commercially available LLMs will only show you the final output right not the intermediate output so

**[30:51]** distilling is not easy so to speak but you can use some other techniques to do that as well okay &gt;&gt; is this only one Good question. Not necessarily. You can distill multiple layers. The question is can the distillation be applied to multiple layers? Yes, it can be applied to multiple layers. &gt;&gt; Yes. Yes. As different K divergences and in fact I in one of my R work we have shown that doing it at multiple layers will uh give some performance boost and

**[31:20]** we also have theory associated with it. Is there any distinction? &gt;&gt; Not necessarily. But at least the question is should the architecture be the same? Ne not be. But uh to for for this distribution to be computable, you need the dimensionality of those to be same. Right? If you have two different random variables which are supported on two different spaces, you can't compute divergences, right? That's okay, right? That's a

**[31:48]** hyperparameter choice. You anyway make it to be equal. That's it. Yeah. &gt;&gt; Um yeah, depends upon the parameters. You know, there are distillation techniques where teacher and the student network may have more parameters compared to the student network. As I said, theoretically speaking, these are all uh I mean there's no restrictions so as to why the student has to be smaller

**[32:16]** having smaller parameters that can happen. Dillation may actually increase the performance. Yeah. Yeah. &gt;&gt; No, no, no. It need not be some intermediate layer of it need not be. It's it's always not the final layer typically because if you're solving a supervised task then you the final layer has to be the label, right? It should be softmax if you're solving a 10 let's say 10 classification problem and so on.

**[32:44]** Yeah. Why not see uh you can let's say that X is 100 dimensional right I have 100 layers here I only have 50 layers here and I make the 45th layer to be having the same dimension as that of the student

**[33:13]** &gt;&gt; number of layers need not be the same only thing is whatever layer you are distilling Whatever features you are disting they have to have same dimensions no otherwise the kid &gt;&gt; doesn't matter doesn't matter yeah typically distillation when it is done it is done at the layer of logits which is you know pre uh find pre- softmax you're solving a classification task there's one layer after which you put a softmax to make it bounded between 0 and one right just before the softmax and

**[33:42]** that's called logits uh in in the community you do you distill the logits in most of the cases But there's no restriction. You can do it at any layer. You had a question. Yeah. &gt;&gt; Not necessarily. But whatever layers you are distilling have to have the same dimensionality. That's it. The math will go through, right? I mean the only thing is the divergence metric has to be computable and that if that has to be

**[34:09]** computable, the random variables have to be supported on the same space. As long as that is happening, it doesn't matter. Okay. Fine. We said that we sample some samples from the training data of &gt;&gt; sample some samples. One is a verb, the other is a noun. &gt;&gt; Yeah. &gt;&gt; Correct. Both you have to do it both. So

**[34:39]** the you take some data uh do a forward pass through the student uh the pre-trained the teacher network get collect a few ZTS and use the same data you have to use the same data and pass it through the student do a forward pass through the student network collect zs and then do one one batch of back prop and update the parameters repeat it iteratively and do the same thing with the student network again sorry teacher network again get the new set of that

**[35:07]** would that won't change you can store Right? But if you do one backward pass through the student network, the parameters changes. So ZS changes. So you comput recomputee the divergence back propagate and multiple uh iterations is what you do. My question was a model of the same size as that of the student model and it is on all samples that the teacher model was but it parameters which perform

**[35:37]** depends on the task. question is a good question right theory wise what can be shown said said is uh yeah the question is if if you make uh the student and teacher the same size and let's say dist will the student do better than teacher right that was that the question &gt;&gt; yeah um that paper that I mentioned right actually says that if you distill with the same size the dist model will do better because it encompasses the label

**[36:07]** noise is the climb Okay. And empirically most of the cases you can actually see that. So dillation can be used as a way to regularize. Okay. &gt;&gt; Both must have the same task. &gt;&gt; Uh both must have the same task also is not necessary. You can train the teacher to estimate the uh the marginal and you can use the student to estimate the uh conditional as well. doesn't matter that

**[36:36]** that that is also a possibility. Yeah, dark art elicitates a lot of questions, right? Oh, more question. Okay. &gt;&gt; Oh, yeah. Yeah. Can teacher and students be of different They will be of different architectures, right? Or I mean are you saying that one is a

**[37:03]** CNN and the other is an MLP or something? Yeah. Yeah. No restrictions have been put here. Okay. Okay. So this is about uh distillation. So now we'll do the last piece that I wanted to talk about today which is uh training neural networks with adaptive learning rate. Uh we will stop here and continue in the next lecture.
