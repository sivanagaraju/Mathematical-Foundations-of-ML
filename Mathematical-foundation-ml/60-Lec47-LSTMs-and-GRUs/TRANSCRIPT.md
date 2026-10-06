# Transcript — Lec 47 LSTMs and GRUs

> **Source:** https://www.youtube.com/watch?v=Pkuwu4EMRj8  
> **Channel:** NPTEL - Indian Institute of Science, Bengaluru  
> **Duration:** ~22 min  
> **Note:** Auto-captions cleaned lightly. Minor ASR errors possible.

---

**[00:03]** Welcome everyone to the class. See general mathematical objective to solve the vanishing gradient problem. we have the hidden ht to be equal to

**[00:31]** some function f of ht minus one and xt. This is the definition of recurrence isn't it? Current state is a function of the previous state and the current input. Look at our equations. No, ht is a function of ht minus one and xt. Of course, there's the sigma. I mean, all that is captured within this function f now. So, there is linearity. There is it's a composition. This

**[00:58]** capital f now is a composition of linearity and some nonlinearity. It doesn't matter. But what is important is it's function of htus 1 and xt. Correct? So this implies that this particular Jacobian which is a product of these terms. The dependence of the final hidden state

**[01:35]** on any previous hidden state is through this product. Oh, sorry. Yeah, this comes up because of the recurrent nature. Correct. Now in a vanilla RNN is multiplicative We have uh diagonal of

**[02:28]** the sigma sigma dash time w because of this repeated multiplications of w we get that particular term. Okay. The solution is to just approximately equal to identity. If

**[03:06]** you make this then doesn't matter right. I mean this is the degenerative or naive solution as he said make lambda max equal to one so that there's a path from the outermost hidden state to the any hidden state without without any uh linear multiplications of w right now let's let's look at um general principle to do that is to

**[03:36]** achieve this should be additive in in terms of instead of multiplicative. How do we do that? Make ht to be equal to some alpha t *

**[04:05]** ht minus one. Actually I'll write it as hard product here element wise multiplication because alpha t and htus 1 are both vectors. I'll multiply each of the elements by something different value. Beta D * H T tilda where HT

**[04:31]** minus one is the previous state. H T tilda is some function of W1 XT plus uh some W2 HTUS 1 plus some bias. So alpha t and betat t

**[05:07]** are modulators some okay that's all. So now what am I doing is look at this. Now the dependence of ht on ht minus one okay is no additive. It's not multiplicative anymore. See what was it? Ht was dependent on htus one through sigma through this equation

**[05:34]** previously. Now there is also an additive path here. You see this? So now ht depends on htus one. Previously HT was dependent on HT minus one through this matrix and some nonlinearity. This F is some nonlinearity. Okay. I actually I can write it as some sigma some nonlinearity. Now to that we have also added a I added another additive term here.

**[06:04]** Do you see this? Yeah. Now what happens because of this is that the derivative of hk + 1 with respect to hk will now have two things we will do that as well. One there will be a term that still has this w term which the vanishing gradient thing and there will be another term that only has this alpha t times identity because this htus one the dependence is only linear here. You see this that's it. So you are creating a an un

**[06:35]** uh stopped or rather unrestricted conveyor belt kind of a thing from between states HT minus one and so on. We'll do the math. We I'll show you the equations in a yeah &gt;&gt; which one? &gt;&gt; The alpha meaning &gt;&gt; yeah that can happen those are also learnable. I'll complete the story in a while. Okay. Now what happens is uh

**[07:09]** suppose you make betat to be equal to 1. Okay you get a particular type of an RNN called highway network. Okay if you make your uh beta t to be 1 minus alpha t then what you get is what is called as gated recurrent units grus. Okay. If you make both alpha t and betat t to be learnable vectors, then it is an LSTM. In fact, this is not the only way to do

**[07:38]** this. That paper which I'll share with you has shown thousands of ways to do this. Okay, that will ensure that you know all of them will have the same effect. But what is to be done is that you know the only thing is that you'll have to ensure that there is a there's a there's a linear path so that the derivatives will not will not die down. Let's let's look at the ma math now. Aren't we having the learning by manually saying that it should be related to &gt;&gt; no no we are not we're not restitting

**[08:05]** learning because we still have this no we we are not changing anything this depend this is like that skipped connection in reset we still have that path we have now two paths there is a highway or there is a linear connection and there is a nonlinear connection as well so the gradients flow from both the ways there's a you yeah I mean this alpha and beta will tell you how much of the information to forget and how much of the information to remember if you look at it look at it that way.

**[08:34]** &gt;&gt; So out of the two parts which part depends on the problem you leave it to the neural network. So you are learning alpha t and beta t as well. Right? So network will decide which one to what what to remember what to forget but you are facilitating the neural network to forget selectively forget and remember through this alphas and betas that's it see I could have actually written the

**[09:02]** equations of grus and showed you know all the complicated stuff this is this is actually the the most abstract way of looking at things so you can derive thousands of architectures from this okay let us look at the calculus right now we let's look at what happens to the you know the derivative so that this will not lead to the vanishing gradient problem. That's where we stop. Okay. we ht

**[09:31]** &gt;&gt; it doesn't matter I think. So you know this is what what did I write? &gt;&gt; Okay it's just a parameter right? Call it whatever it has no bearing whatsoever. Just two set of parameters. It doesn't matter what it is. Okay. And by the way, some people call this right this H tild as candidate states. Okay. So they have there is some candidate state and then you multiply that with lots of terms, right? Forget gate, input gate, that gate, they call

**[10:00]** them gates and so on. Basically this is the idea, right? So yeah. Okay. So let's let's look at uh the the math. What happens uh at the math? Now See, please don't complain. Don't tell

**[10:27]** me that, you know, GRUs and LSTMs was in the syllabus, but I did not teach them. I've actually taught the grandfather of GRUs and LSTMs. So you derive those things from these. Okay. &gt;&gt; Long shortterm memory networks. There's a long-term memory, short-term memory. This HT is the short-term memory. HT minus one is the long-term memory and so on. It's idea. So just go and look at LSTM

**[10:56]** equations and convince yourself that this is a special case of whatever I've written here. Okay. The power of abstraction. You see, always abstract. so that everything in the world becomes a special case. Okay. Okay. So let's consider this derivative, right? So h is consider. uh the diagonal of uh vector alpha t.

**[11:35]** So you understand what what is diag means right? So it's a diagonal matrix with this alpha t as its diagonal. Okay, that's what I mean. And since we are looking at uh uh gradients, sorry, vectors, I'll have to write these this way. H t -1 * diag of ht -1 plus do

**[12:00]** beta t * h tilda divided by d ht minus one. See whatever I'm doing it for this particular uh abstraction can be done for GRU separately can be done for LSTM separately and that's a homework. You understand what I'm saying? Write down the equations for GRUs and show that it does not have anything gradient

**[12:29]** problems. I'm doing it right now but you'll have to do that special case. This is the most general case. Okay. Now this can be written as so diag alpha t plus some chalon t. So all this all these terms simply calling them as

**[12:56]** chalon t. uh why am I doing that is because see here there is dependence on ht tilda and ht minus 1 and so on which will have w1 and w2 terms ter terms you see what I'm saying see ht here depends on htus one and ht tilda ht tilda depends on xt and htus 1 through w1 and w2 right so these terms which I'm saying uh as e or epsilon t

**[13:26]** this actually depends on w1 and w2. I don't care about that. All I'm looking at is diag of alpha t now which is independent of w1 and w2. This independence is coming through this particular term. See when I take the derivative of ht with respect to ht minus one there is diag of alpha t here right and the other terms and these other terms I've simply use chain rule here right the first term

**[13:54]** plus the derivative of the second derivative of the second has product so this these two terms come from the product rule that's it correct and I'm combining all the terms that are coming from the product rule into something that I don't care okay so now there is diag of alpha t term now when I do the take the derivative of the last state with respect to any time step t which will now have the products of right now if I do cautious thoughts and

**[14:42]** look at the norm of this &gt;&gt; t to t minus one correct tus one some identity here because the diag of alpha k + 1

**[15:10]** you take alpha k and it will become an identity which will be one. That's it. There will be that term. See now what happened now? There is this additive term that comes up. Okay, this first term it can still have gradients. Okay, without caring about the second term which has dependence on W. You see that right? So because of the

**[15:36]** presence of this particular term here depends on what alpha is of course but since we are making alpha to be a learnable parameter if it needs to be identity you can make it to be identity right yeah &gt;&gt; of course that's the best that you can do now at least that's what I told you right we have facilitated the neural network &gt;&gt; to be able to learn the long-term dependencies that's it That's all the

**[16:05]** point is there's no see that's all every regularizer will do. No any regularizer is a is is is simply facilitating the neural network to operate in that high bias low variance region. That's it. Over parameterize and increase the bias. Now you see why this is regularizer. Right? So I told you in when I looked at bias variance decomposition that every single architectural tweak that we do is a regularizer. Now going from RNN to GRU or an LSTM is an explicit regularizer

**[16:32]** where you have that additional path it's like this I I'll come I'll take another question it's like this right so you make Y to be equal to X plus some function f of X y was f of x right now you made it to be equal to x plus f of x where x is the identity this is the resonate idea this is the residual skip connection see this idea

**[16:58]** is so important because even the modernday transformer architectures has skip connections in it. All deep neural networks have a skip connection in it uh in them because you need to be able to learn this identity. That's it. The gradients will simply vanish. And please note that the identity identity that we are talking about in a recurrent architecture is through time not through space. But in a resnet the identity is through

**[17:29]** space not through time. So when you have neural networks that are I I'll just tell you what I mean. See let's say that you have an MLP like this right? This is X and this is Y. You have connections this way. Suppose I take this and put it here and connect it to this layer without any weight. Sorry, this is plus. This is how it is typically done. Take

**[18:04]** one layer the other layer of neural network connect one the the output of that to the next layer. So that's skip connection which is done in the residual neural networks or reset. Even in transformers this is done. And now we are doing it in spatially. What what we mean to say is that in the it is done through in the uh the the depth dimension but in an RNN we read it through the temporal dimension depends on where your vanishing gradient

**[18:33]** happens. Right? See you ask the question how is an MLP related to an RNN? So I'm giving you a hint. In an MLP when you go from one layer to the other the gradients flow from output to the input. If you look at each layer as a time step then there is a vanishing gradient problem as you go deeper in the neural network. Isn't it? That's exactly the problem the ResNet solves. If you make the neural networks you know you know what the ResNet paper shows it shows

**[19:00]** that deeper is not better. If you go deeper and deeper the neural network starts performing worse because of this exact same problem because the gradients have to flow. Now how do you solve that? Through the same idea wherever you have this sort of a thing ensure that the function also has an identity path so that the the gradients do not go to zero. Simple chain rule. So they did residual connections across space. Meaning when you are doing neural

**[19:28]** networks when you are constructing neural networks which are deeper ensure that every layer has residual connection show that there is a uninterrupted path so that the gradients can flow. Do the same thing across time in uh in an LSTM. In fact to make things complex you can have an RNN okay which has longer context and you have deeper RNN as well. You realize what I'm saying? And through the depth you can have residual

**[19:57]** connections there as well. You see what I'm saying? There is an RNN okay where there is parameter sharing across time and each RNN cell can be a deep neural network across space and those deep that deep neural network can have residual connections across space as well have the same effect. I mean gradients actually are flowing from two directions right and please note that across depth there is no parameter

**[20:24]** sharing this is w1 this is some other wcap there is another theta and so on but across time the parameters are shared that's now you can see why this is an MLP with a certain architecture isn't it get it so typically an RNN uh the way it is implemented is I'll that will be a part of your second assignment anyway the day it will be done is you have a deeper RNN across space and you already

**[20:52]** have parameter sharing across time by construction of an RNN. Okay, any questions? You had a question. So just in the last step you're applying what &gt;&gt; see yeah so so you have you'll have to have other term also there is a plus here and uh you'll have you'll have the norms of these individual terms bounded and I'll take this one term and then there is this dag term which will have one as a gradient that's what I meant

**[21:23]** you have norm of sum of two vectors then you have to bound that norm of sum of two vectors can be decomposed into individual norms Yeah. So then you take one of the individual norms and then do it. That's it. Yeah. Okay. So this completes uh the the discussion on the recurrent architectures uh the LSTMs and GRUs and thousands of them as well. Right. Please read that paper. I'll share that with you. Okay. Uh next class uh we will look at the the attention mechanism and the

**[21:52]** transformers which is the the state-of-the-art architects and and how how are these neural networks trained? still erm still erm with the same gradient descent that we are looking at and you can still have all regularizers that are there and remember that we are still trying to get that P of Y given X by minimizing the K divergence so that does not change okay thank Thank you.
