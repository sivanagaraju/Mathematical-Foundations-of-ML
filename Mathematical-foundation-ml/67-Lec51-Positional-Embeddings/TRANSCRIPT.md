# Transcript — Lec 51 Positional Embeddings

> **Source:** https://www.youtube.com/watch?v=xbkJIeLoGyw  
> **Channel:** NPTEL - Indian Institute of Science, Bengaluru  
> **Duration:** ~13 min  
> **Note:** Auto-captions cleaned lightly. Minor ASR errors possible.

---

**[00:04]** Welcome everyone. Um we'll be continuing our discussion on transformers and then we will today look at some of the methods to train the neural networks. We have already looked at uh gradient descent the vanilla gradient descent. Uh I will extend the idea of vanilla gradient descent to the ones that are currently being used. uh and we'll also look at some of the meta learning techniques uh such as transfer learning and distillation in this particular uh class. Okay. So um before we move on to

**[00:35]** the the learning algorithms, there was uh one little piece that was uh missed in the last class. See if you look if you go back to the architecture of transformers, we are looking at sequential data, right? So know we transitioned from uh the recurrent neural networks to transformers as an architecture and we are looking at the sequential data. The idea of sequence, okay, which is a temporal ordinality is embedded into a

**[01:05]** recurrent neural network through the parameter sharing and the the hidden variable state variable, right? That has a time transition to it. However, in a transformer, all of the data tokens in a sequence are processed in an independent manner. Okay? Since there is no hidden uh vector explicit hidden vector in the construction of the transformer, the transformer as an architecture does not

**[01:36]** have the idea of ordinality of the input data. Do you understand what I mean? Suppose you have a sequence. you have a tin sequence in an RNN. What

**[02:02]** happens is you have the hidden states. We start from X1 and X2 etc. And each of the hidden each of the input sequence has a different hidden state. And because of the temporal parameter sharing these hidden states will embed the information that X2 succeeds X1 and so on. Right? XT succeed XT - 1 and so on. Whereas in a

**[02:30]** transformer so each of these tokens are fed or given as an input independently. There is no temporal ordering information that the transformer emits. Okay. So to ensure that or what do I mean by that? uh the transformer does not treat or rather treats the first token exactly the same way as second token and third token and so on. So it does not know that the third token uh succeeds the second token and second

**[02:59]** token succeeds the first token and so on. So temporal information is lost. How do you do that is the question. Okay. So to incorporate of the input sequence accommodated or added into the system.

**[03:55]** So explicit time variable has to be added into the system. Okay, we'll have to say that first token comes first and second tokens come comes later and so on, right? So, we'll have to do that. How do you do this? The easiest way to do it is to have a time variable. Okay? So, let's call it as uh uh small t. It is simply 0 1 2 3 etc. You define new data X as or X I at the

**[04:25]** IAT token as the ayat token plus the ayat time token. That's it. So this will tell you that the first token is you know because it has that particular additional information into it. But the problem with this is uh xi as we know is typically in dimensions right it's a dimensional ukidian space which is a very high dimensional space. And if you add a scalar to it, okay, uh

**[04:54]** empirically it has been observed that uh this information will be totally ignored. Why? Because in a 10,000 dimensional space, no, if you added one-dimensional scalar, that one-dimensional scalar will not be taken significantly by the network that is there. This will be ignored as a noisy point. I mean, if you're looking the data as an image, if you have a spec in the image, that would get ignored while processing. So that would happen if you add a scalar. Therefore, instead of adding a scalar, what should you do is

**[05:22]** that so scalar t and therefore a fixed vector of dimension D So we need a function okay f or we need

**[05:57]** a function f that would take the scalar t and give you a a t bar which is from r to rd. So we need a function of this sort where given a scalar t it would give me a d-dimensional vector okay that would have the same information as that of scaler. How do you do that is the question. There are multiple ways to do this. Okay. Uh one way to do it is called uh

**[06:26]** okay and by the way this thing right this function f it's called positional embedding. simply a d- dimensional vector that corresponds to the temponal orality. So now how do you convert? We know I mean often times we do the other way around right we take data from a higher dimensionality and project it to a lower dimensional we have to do the other way around now. So how do you do that? There are multiple ways to do this. One way to

**[06:54]** do this is by using uh ss and cosiness cosine functions. So example for this is what is called as a sinosoidal embedding. uh

**[07:24]** okay so let me just explain that and then write the math. So what is the basic idea here is as follows. construct sinosoids of different frequencies. Okay. So now how do we do that? So let's say that we have one sinosoid like this and you have another sinosoid and multiple sinosoids of multiple frequencies. Okay. So on what we do is that we take capital

**[07:54]** D of them. you get capital D of them and at every time t you sample from the all these sinosoids if you do that you get a D- dimensional vector at all points in time right do you see what I'm saying you take capital D so not capital D so D number of sinosoids okay where D is the dimensionality of the vector the input vector construct D sinosoids of different frequencies okay and sample

**[08:24]** capital T of them so you capital t number of d-d dimensional vectors and each of these correspond to a time time instant you see this this is t=0 you sample at t equal to 1 you sample and so on and you sample up to t equal to capital t okay and how many sinosoids will we have &gt;&gt; yeah small d number of sinosoids is what we have this is the idea that's it I'll

**[08:52]** write the math for it this is called sinosidal positional embeddings Okay. So now we'll write that. So if you take uh so it is done differently for the uh the odd times and even times. Okay. So this is uh I comma 2T. We have sin I divided by some large number n to the

**[09:19]** power of 2t comma t. See I comma 2t + 1 is cosine of the same thing. This is it. So you get a uh you get capital t number of d-dimensional vectors. Okay, you add them. So now what happens? The same thing, right? So you replace xcap of i

**[09:47]** as x i + tcap of i. That's it. So now the the input has this ordinal information right the temporal information uh which is being explicitly fed into the model right now. Does that work? Okay, this is about positional embeddings and uh &gt;&gt; you take different frequencies, right?

**[10:24]** The reasons you take different frequencies along every D is for the precise reason. But of frequencies &gt;&gt; no you take s and cosiness in such a way that they are not the same. Even if they are same all d values will not be same right. You need a unique vector. That's it. Yeah you take different frequencies to avoid the precise thing that you said. Question is if you take sign and cosiness because they are periodic will won't they get the same value because you just to avoid that you'll have

**[10:51]** different frequencies that are being taken. Okay. Right. Okay. And another point to be made is uh whatever attention mechanism that we saw the last class right it is typically referred to as the self attention okay there is another idea which is called cross attention where if you have two neural networks one which is called the encoder

**[11:18]** neural network the other is called decoder neural network we'll talk about it in a while okay when we talk about autoenccoders later in the course uh in the context of unsupervised learning. I I'll revisit this and talk about the encoder decoder models. But suppose you you remember I think I had mentioned about encoder decoder in the context of uh machine translation right there is an encoder. Most of the models today that we have uh the LLMs are decoder only models which which only compute self

**[11:46]** attention. Basically the idea of cross attention is suppose you have two neural networks okay or two functions. You compute the query from one neural network and uh you compute the value and key from the embeddings of another neural network and then compute attention. That's called cross attention. Get it? Right? So you take different embeddings coming from two different neural networks. You can I mean it need not be encoded decoder at all. It can be

**[12:13]** any two neural networks. Right? So you take one of the you know you have three projections that we need to compute attention. One of the projections you compute from one embeddings of one of the neural networks. The other two you compute from the embeddings of other neural network and then compute attention. And this thing is called cross attention. Okay. Just had to mention this uh uh for the sake of completeness. Okay. Let's now switch gears and move to the uh the topic of uh adaptive learning rate. Learning from

**[12:42]** adaptive learning rate. Uh okay. So training &gt;&gt; pardon my uh back and forth. So let's let's do the other topics first and then get to the adaptive learning rate because you know this is all encompassing the adaptive learning rate can will be used for

**[13:09]** training all kinds of neural networks. Let's do a transfer learning and then we will talk about distillation and finally we'll go to adaptive learning. Okay. So let me introduce you to this idea called transfer learning. Uh we will stop here and continue in the next lecture.
