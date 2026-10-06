# Transcript — Lec 46 Back Prapogation in RNNs and Vanishing Gradients Problem

> **Source:** https://www.youtube.com/watch?v=TVkaROL2FLw  
> **Channel:** NPTEL - Indian Institute of Science, Bengaluru  
> **Duration:** ~29 min  
> **Note:** Auto-captions cleaned lightly. Minor ASR errors possible.

---

**[00:02]** Welcome everyone to the class. Okay. So now we have to look at how to perform back propagation uh in in for this sort of an architecture. So what changes? See math wise nothing changes because all we are looking at is uh multivariable chain rule. But what's important to note here is that when you have an architecture this way suppose you are solving a classification task which is this. Okay, you are where you are tapping the output at uh this the the output the the last RN then the

**[00:33]** gradients have two parts right one is right to left the other is top to bottom especially if you have multiple layers the gradients corresponding to every hidden state and every variable that we are looking at here has two uh paths okay see one path is coming from right to left or left to right the other path is coming from top to bottom. Okay. So, yeah.

**[01:12]** You can still see them as different features, right? Capturing different features at different depth level. It is simply a simply having multiple neurons so that universal approximation theorem uh is obeyed so that you have multiple parameters to cater for you need to anyway do function approximation at the end of the day. You need to ensure that it happens that way. That's what the depth is. Yeah. So now the gradients corresponding to uh each of the parameter that we are looking at. Now we'll we'll have two parts right one

**[01:41]** from right to left and top to bottom. So when you are doing back prop back propagation you'll have to ensure that these gradients are taken care of that's it. When you are adding the gradients the paths that we are looking at in MLP what happens is when we define the delta the delta had only one path right the output to input so it did not have the other path now we have two paths. Okay. So this algorithm where uh you have to just do back propagation from two different directions is called back

**[02:09]** propagation through time. Okay. It's called BPTT. I'll leave the algebra as an exercise. Do that. So what's more important is this BPT the this architecture leads to a particular degenerative problem in RNN which we will discuss in a while. But please uh look at the math of BPDT. It's it's the same as the back propagation that we had done for an MLP. But you'll have to ensure that when you apply that multivariable uh chain rule, you have to take both the directions into

**[02:36]** consideration. Okay, please work the math out. Okay. Now, So there are two things that's happening

**[03:19]** in an in an RNN right. So one the hidden state is maintained across time okay which is what is capturing the the temporal information and there is parameter sharing across time it's happening because of this. So what might happen is now as the length of the input sequence increases when you back propagate from output to the input okay the gradients the magnitude of the gradients depending

**[03:46]** upon the the the weight matrices or the the singular values of the weight matrices will keep reducing we'll show the we'll see the math in a while the idea is the following right I mean if I have to roughly translate in a in an intuitive non-mathematical way uh the longer the time elapses the lesser our memory memory uh remembers right lesser our memory can capture if you look at these hidden state HT as a memory okay of that the recurrence right uh on the

**[04:16]** sequences is captured using the HT which is the hidden state the longer the sequence is okay the lesser is the influence of the the past on the current hidden state and that should come out come up in the math somehow the way it comes up is comes up is as the magnitude of the gradients while you are doing back propagation. See what happens is remember what back propagation does right it's gradient descent at the end of the day if the gradient of the uh the

**[04:46]** empirical risk with respect to a particular parameter is very less then that that parameter will not get updated at all because the gradient is very low. Now we should not be ending up in a case where the gradients of the the the empirical risk or the loss that we are computing will be very low as we progress longer in time and that happens in an RNN by construction. Why? Because there are two things here. One there is parameter sharing across time which

**[05:13]** means that you are multiplying the same parameter every time across time and there is all the all the information about the sequences is captured and embedded in the hidden state which is recursive in nature. Okay, that is what is infamously referred to as the vanishing gradient problem in an RNN. We will see the math of what vanishing gradient problem is and we'll see what can be a a respite to such a problem. But did you understand

**[05:41]** what's happening? Because of the recursive nature u the gradients as you accumulate over time will have lesser magnitude. We'll see the math here. Okay. So BPTT leads to So what's uh the math? So at each time

**[06:13]** step tanishing we have the following equation. So zt is

**[06:48]** w1 ht -1 + w2 xt. I'll leave the bias in a y for uh ease of algebra. we have ht to be equal to um sigma * zt. So this is the the current equation that we have right. Suppose LT be the loss

**[07:18]** at time t. capital LT uh L capital T yeah be the loss that we compute at the final time step so and we are computing the gradient at that let's say that we are looking at a classification problem here and LT will be the loss at time t so we have do lt by

**[07:45]** ht where ht is the hidden state at some time point t can be decomposed as Right? Do LT uh depends on HTT through H capital T, isn't it? &gt;&gt; Come again. &gt;&gt; What should be XT?

**[08:22]** &gt;&gt; I mean here this is not X2. This should be XT. This is what you're saying. So, is this clear? So, loss at the last time step. Okay. The dependence of the loss at last time step on HT is through the hidden the last hidden state HD. Right? Now consider

**[08:59]** this particular term. The question that we are asking is what is the dependence of the last hidden state. Okay. Uh on the previous hidden states of previous time. What is that? that is the product of all the Jacobian matrices or the derivatives this simply chain rule. So what am I

**[09:28]** saying is if you are considering a time step small t okay then the last hidden step depends on the penultimate hidden step and the penalty made depends on the previous one and everything till time t. It doesn't depend on anything that is prior to time t because we are only looking at time t here and that dependence can be written as a product of several jacobians and the reason I'm saying jacobian as all these are vectors

**[09:56]** right and we are looking at jacobians here uh let me just write it this way this is the product this is uh k = t2 capital t minus one correct because k + 1 is what I've written here. It's t minus one. Is that all right? Yeah. Okay. Now consider the this particular jacobian del

**[10:32]** hk + 1 by del h k. So we have H K + 1 to be you look at this hk + 1 is sigma zk and

**[10:58]** zk is this okay I will unroll it and write it this way it is sigma uh w1 or w2 W. &gt;&gt; Okay, let me write it. Hold on. So, this is H K plus uh X K + 1 because HK + 1. So, what what are these W's? This is 1 and two.

**[11:26]** Okay, two different parameters, right? So, this is how it is. H. Now if you look at the derivative of this. So now consider h k + 1 derivative of this with respect to hk. If I write this, this would be just I have done the algebra. I take this is the diagonal of sigma dash. Okay, that is evaluated at z k + 1

**[11:59]** times. So w1 okay where this diag is a diagonal matrix. So what I'm taking take a diagonal matrix and put sigma dash or the derivative of zk + 1 in it in it to its diagonal and multiply it with w1. Okay that's the derivative. Okay. Now typically what happens is okay let's look at the norm of this. Consider the norm of this particular vector simply use the

**[12:44]** okay not this let's look at this itself we are interested in so made a mistake so because we we are finally interested in this derivative right here we are looking at the derivative of the loss with respect to the hidden state by the way why am I even considering this term I'm considering this term because it appears in the back propagation through time see finally we need to compute delt

**[13:15]** by dw1 and the dependence on LT or dependence of LT on W1 is through the hidden states multivariable chain rule and and this term comes up in the back propagation equations and that's why we are looking at this term that's why I asked you to actually write down the BPT equation so that you see that this this term comes up this is very similar to the delta term that we

**[13:43]** looked at recall I mean in the MLP there was a delta term right which defines the error so this is this term is very similar to that. Okay, the problem of vanishing gradient comes up because we have this sort of a term. Finally, what we need as the gradient of the the empirical risk with respect to W's which depends on this through the chain rule. Okay. Anyway, so we need the norm of this particular term. Now we will use the Kosis squ inequality and you have

**[14:14]** product of these many gradients here, right? And we computed this gradient as diag of this. And you have two terms here. So I will use the inequality and write that as this. You still have this product. So this norm of this matrix DAG sigma -ash computed at zk + 1 times norm of

**[14:46]** w1. &gt;&gt; inside what? &gt;&gt; Of course. Okay. Now see diag or rather this one the norm of diag this is bounded. Okay this is always bounded. Why

**[15:18]** sigmoid is between 0 and one. Right? Typically if you take the sigmoid function or tan h or something this norm is bounded. Okay. And now let few sigma already. What other symbol? Use lambda. Let lambda max be the largest singular value of w1.

**[15:53]** Okay. And we know that norm of w1 is lambda max isn't it? The norm of a matrix is equal to the largest singular value. Which means that this gradient now upper bounded by this is the first one

**[16:28]** is uh less than one right and uh uh the second one is equal to lambda max and you we have t minus t of them which means that this is bounded by lambda 1 to the power of t minus t we have those many of them is that okay see this is a very nice result okay so as

**[17:03]** the gap t minus t increases lambda power lambda 1^ t minus t decays exponentially. &gt;&gt; Did I write? What did I write? Lambda. Yeah. No, it's lambda max. So that's it. So this is the vanishing

**[17:50]** gradient problem. As the length of the input sequence increases, okay, the gradients corresponding to the earlier time steps will be insignificant. Therefore, see why why do we care about this term? We care about this term because the dependence of the uh the loss on the parameters is through this term. Okay. Now if this term decays exponentially then

**[18:21]** the loss or the gradient with of the loss with respect to the parameters decays exponentially as the sequence length increases and this is the the infamous vanishing gradient problem and if your lambda max is uh let's say greater than one then it's the other problem called the exploding gradients okay both are dangerous so that is why see when see RNN existed for a long time. I mean an anecdote is my bet tech

**[18:49]** thesis was on RNN written BPD and all that used it for some small problem but the thing is uh they could not work with work for uh longer sequences because of the vanishing gradient problems. So something has to be done to solve this. So that's where we have uh the architectural uh uh tweaks that one would do with like GRUs and LSTMs that that are constructed to solve this particular problem of vanishing gradients.

**[19:17]** Okay. So vanilla RNN has this particular problem. So why does this come? I mean it's just because of calculus multivariable calculus. Write down the gradients. You'll see that the gradients are strictly upper bounded by some value that exponentially decays. That's it. Okay. Any questions on this? Max how do we ensure that all &gt;&gt; see that is through that that is through

**[19:44]** uh equations no hard code them &gt;&gt; that's okay see that is why the dependence of your loss on the parameters will be through the hidden states you need to add the gradients with respect to each of the hidden state while you're conclude that that is why you need that that sum. So it's basically the loss or the empirical risk depends

**[20:12]** on the parameter through all the states states at all times &gt;&gt; come you need to add all those gradients that's exactly what BPT is all about okay &gt;&gt; we didn't want to w elements so this says that when we take the linear combination across all the states the gradient with respect to some of the terms it's like if we

**[20:41]** propagate this to the WDs that would be zero that that so this in some sense says that for the HTS that are closer to the last state the W's there would not be zero &gt;&gt; agreed sure see the observation that he is making is that this only says that the gradients vanishes uh exponentially as you go farther and farther from the about out the outermost state. Agreed? But this is not only

**[21:10]** about the outermost state, right? Suppose you are doing a sequence of sequence task, the loss has to be computed at all time steps. At any given time step, as you move further in the time, the the the gradient decreases. That means that these W's will be learned. Okay. By by by ignoring some of the time steps, &gt;&gt; earlier time steps. &gt;&gt; Exactly. So if you ignore the earlier time steps so you I mean your sequential model is not correct. See today's models

**[21:38]** have a context lens that are that are of that that that run into millions of tokens isn't it? And by the way you might have heard this word token right to token is every time step every vector that we have in in our sequence is what is called as token. And by the way, in natural languages in today's uh state-of-the-art models, it's not the word that is taken as the uh the the foundation fundamental unit. It is a token. So what is a token?

**[22:08]** There are multiple tokenizers as well. Now so one thing is this bite pair encoding tokenizer where uh you combine you start with characters as each tokens and you know you combine uh characters that appear multiple times into one and so on. run this algorithm iteratively to get tokens. So put all the tokens in what is called as a vocabulary and today's vocabulary size is about millions of tokens and a sequence is a combination of tokens or rather yeah

**[22:36]** it's it's an ordinal set of a token whatever we have written as x is what is called as a token right so that's why you don't need you don't want this longer the sequence lengths the gradients are vanished you don't want that to happen sometimes you do I mean that's why basically the idea is the following Sometimes you should not forget. Now

**[23:02]** wisdom is to understand what to forget, when to forget, and what not to forget. How does that translate to mathematically? You need some of the gradients to go to zero. You need some gradients not to go to zero. How do you do that? Just add an identity function. That's exactly what is done in the residual networks as well. Right? That's what we will see. Now mathematically speaking what's happening is the dependence of h here right uh let me show you that

**[23:31]** maybe I I'll write you write that so so ht recursively depends on ht minus one right all ht depends recurs recursively on the previous hidden states and that dependence is through this linear matrix w now because of the presence of the w and their their singular values that the gradient is going to zero Now what if I also give a parallel path where instead of the recursive dependence being through W I make the

**[24:00]** recursive dependence through identity which means I have a path where HT is equal to some constant* HT minus one then if I also have a scalar HT HT is equal to some alpha * HT minus one by changing that alpha I'm I'm I'm telling whether this has to be forgotten or to remembered mathematically what happens is the gradient have two paths. One path where it is vanishing the other path where it

**[24:31]** need not vanish. So there in fact the name is also there's there's there's one uh modification to RNN which is actually named as highway networks. Highway networks highway because there's a path from the output hidden state to the input hidden state without any multiplications of these W ms. I'll write all the math right now. That's the idea. The classical idea in any linear dynamical systems is if you have a recurrent relationship between uh the variables in

**[25:00]** a system when you are taking the gradient if you want to break the recurrence add identity to that. That's all. You can have thousands of architectures around this idea. I will show a couple of them. Okay. So that will facilitate you for forgetfulness. That will also facilitate you remembrance. Have both. So that let and let the network decide what to remember, when to remember, how much to remember. See how much to remember comes

**[25:28]** as a learnable scale. What to remember comes as the matrices and so on. So that's why some of the variables in this LSTM are actually called forget gate. They're called memory gate and cell state and all that because of this very reason. So I but I am not a big fan of calling them I mean making them non-mathematical and giving some non-existent biological uh intuions. I'll only write math. Now you got the

**[25:56]** basic idea right? Whenever there is a linear system which has a recurrence relationship through some linear transformation if you apply chain rule through the recurrence because of the depending upon the igen values or the singular values of this recurrence matress uh the gradients will die down to ensure that that will not happen you'll have to create a path that does not have multiplications with this matrix stru that's the that's the idea okay let's look at it right now any questions here

**[26:25]** so far Yeah. &gt;&gt; Well, not because uh it's a good question. Um the question is can there be explicit regular raisers where you ensure that your lambda max is not &gt;&gt; equal to one? &gt;&gt; I'm not saying it's equal to one. You know it can &gt;&gt; then oh exactly equal to one. Yeah, we will do that.

**[26:53]** We see that in other words what you are saying is can I do an architectural change such that there exists a path where the when would lambda max be equal to one &gt;&gt; only when your matrix is identity then your singular values are equal to one that's exactly what we will do now it's nice that you asked that question we will put a regularizer architectural regularizer right now in such a way that we will have a path where lambda m max is equal to one that's a nice way to

**[27:20]** look at it how do you do that. If you only make it one, there is nothing to learn, right? The dependency is only recurrent linear. We want we want a capability where we want some things to be forgotten as well. So, we will have that sort of an architecture which we will see now. Another question. Yeah. in direction.

**[27:46]** &gt;&gt; Yeah, &gt;&gt; that is the thing that should be. &gt;&gt; See there here you cannot control it at all. No. Yeah. So what you can Yeah. You can only control this because this zed here the dependence of zed is also on w1 and w2. Right? The recussion is coming from this zed only. See if you look at it those two terms but for the sigma they are exactly the same.

**[28:16]** This nonlinearity has to be there to ensure that the universal approximation capability. If I remove this all this will become linear functions. No. And linear functions can only we know that the linear discriminant functions have their limitations. Only by having the sigma you can approximate any function to arbitrary closeness. &gt;&gt; That will happen that that you can't

**[28:43]** hand that you can't do anything about it. Yeah. It's similar thing. No, this happens even in an MLP whenever there is a sigma this is going to happen. But here we can control by making that lambda max equal to okay. So let's do that. The exploding gradient can't be installed by &gt;&gt; why same thing you you have one you still have that skip connection thing you know doesn't matter in fact you can

**[29:11]** normalize the uh the values such that exploding gradient becomes vanishing gradient thing it's just a matter of scale okay uh we will stop here and continue in the next lecture
