# Transcript — Lec 48 Attention Part1

> **Source:** https://www.youtube.com/watch?v=00DOOSYFyJA  
> **Channel:** NPTEL - Indian Institute of Science, Bengaluru  
> **Duration:** ~23 min  
> **Note:** Auto-captions cleaned lightly. Minor ASR errors possible.

---

**[00:02]** Welcome everyone. In this lecture we will look at another architecture that is often used in the modern deep learning or machine learning paradigms which is called the transformer. Okay. Last time we looked at we started with multi-layer perceptrons or full feedforward neural networks and then we looked at convolutional neural networks. We also last in the in the last session we looked at uh uh

**[00:28]** recurrent neural networks that were which were the architectures designed for sequence data processing. So continuing from there we will look at uh another set of architectures called transformers that are based on this idea called attention mechanism. We'll look at what attention mechanism is and then I'll look at transformers. In fact uh most of the models today including the large language models are primarily driven by this architecture

**[00:56]** the transformer architecture. &gt;&gt; [snorts] &gt;&gt; Now before we dive deeper into what the transformer architecture is uh I want you to uh understand or appreciate a couple of ideas. See if you recall uh the way we looked at kernel SVMs which is that we started by saying that the data is separable and there exists a mm linear decision boundary linear separability and then we extended that

**[01:26]** idea to the cases where there is no linear separability. Okay? Now what was the basic idea there? The idea was to project the data onto some other space okay? And uh hope or assume that the data will become linearly separable in that space. Okay? Now this idea of projecting the data I mean we have been doing this since a long time right? We have been even when we did the looked at the generalized linear regression we said

**[01:54]** that the data that is not linearly separable or are not explainable by a linear model can be linearized with a transform. And so there is some data you transform it to it into some other space and you expect different behavior post transformation is what we have been seeing. So what's the idea? Start with data X. So start with data space and learn a transformation okay? Uh fee to another space Z okay? And then

**[02:24]** you learn your hypothesis or functions on Z. This was one thing that we have been doing correct? Now this kind of transformation of data from one space to the other okay? Is typically referred to as uh representations. So start from some

**[02:56]** space uh which is the given random variable and then make your data undergo a transformation fee and you get another you project your data onto some other space typically represented by this variable Z and call that the representation or the embedding and so on. We have already done this remember? Uh we had this fee to be a fixed transformation where it can be a polynomial projection or some other projection and in the case

**[03:24]** of uh SVMs or kernel machines we saw that this fee can be different transformations. We looked at Gaussian transformations and the RBF kernels and so on okay? Now so far most in most of the cases that we have seen this fee which is the function that transforms data from one space to the other has been fixed and user defined. I also gave you examples from the classical signal processing community for this right? Fourier transform for

**[03:51]** instance is one such embedding learning. Given some data you transform it into some other domain okay? All the classical transformations Fourier transform Z transform etc. are actually learning this sort of a representation. Now if you carefully look at what we did with neural networks okay? We had a composite function. So we said h of X for a neural network was started with some transformation

**[04:21]** and then there is another transformation and so on. So this is what we said a neural network is. Now this entire function right? The entire operation can be seen as some non-linear projection of data from one space to the other. Isn't it? In fact each of the I mean because neural network is a composite function every inner function can be seen as one projection of data. So W1X is one projection of data okay? And then you

**[04:50]** have W2 times that projection. I mean you project that projection onto some other space and so on. You keep doing this. So every composition that we do can be seen as embedding learning. In that sense a deep neural network is learning embeddings in a hierarchical manner. So every layer of the neural network will give you a projection or an embedding of data. Do you see why?

**[05:18]** Because I mean what is an embedding? It's basically some function of data some projection of data onto some other space. Then you start from the data space and the first layer of the neural network is some projection. And from there you project that it onto a second layer onto a third layer and so on. So that way every layer in a deep neural network can be seen as some embedding that we are learning onto the data. Do you see this? Okay? In fact what is done then done is in some of the applications you train a

**[05:48]** neural network let's say for some task. Let's say that you train a neural network for a classification task. What one can do is during inference you take the data point okay? Pass it through the neural network. You tap the output of the neural network from some intermediate layer. And use that as a representation of data and you can build another classifier on top of those vectors. So this is what is called as embedding embedding learning.

**[06:16]** So given a pre-trained neural network as it is called right? Trained for some I mean what do you mean by pre-trained neural network? You have some neural network and there is a there is a ERM that we have done. So once we do the ERM you can use that neural network and use this neural network to get projections of data by tapping onto the intermediate representations from the neural network. Do you see what I'm saying? Yeah. Uh Okay. This is also called

**[06:45]** representations or embeddings. Any composite function for that matter can be seen as especially the neural network can be seen as hierarchical representation or embedding learning. Okay? Now the difference between uh let's say a kernel machine or an SVM and the neural network is that in classical frameworks such as transforms or even SVMs we fix the transformation. But in a neural network

**[07:15]** the transformation the function that transforms the data is learned. And that depends on these parameters isn't it? So that way a neural network can be seen as a non-linear basis change of basis with learnable transformations. That's why it's powerful. I mean instead of I mean what's the difference between let's say a Fourier transform and a deep neural network is that in Fourier transform you fix your basis onto which

**[07:42]** you project your data. In the neural network the basis that is projecting your the basis onto which the data is being projected is being learned. Okay? Through parameters. Why am I saying this is because most of the architectural changes or designs that we do is an attempt towards learning a better embedding or a better representation isn't it? Because all architectures are all neural

**[08:09]** networks now can be seen from this lens that you are learning a projection of data. So any change that we do to the architecture is learning a projection onto the data. Does it make sense? I mean we need this worldview because one it will it's very easy to see a transformer from from this angle that all we are trying to do is to learn a a particular type of a projection or an embedding such that some behaviors are imposed. So now connect this with our earlier discussion on Bayesian learning

**[08:36]** as well. Every choice that we make on the architecture okay? Is a certain type of regularization that we are doing which in turn impacts what sort of projection are we learning on the data. Okay? That's what we are doing. Right? Okay. So now let's look at this idea called uh attention mechanism. I I'll just give you an overview of this and then I'll mathematically define

**[09:03]** this. The idea of attention is as follows. Now recall that if you have uh sequential data where the the input is sequence and output is a sequence. Okay? We saw how to handle the sequence to sequence uh task using recurrent neural networks. Remember that? So how did we do that? Suppose we have an input sequence X1 X2 up to XT

**[09:33]** okay? So we had these recurrent neural networks recurrent cells as we call them RNN cells. And then we tap the output in an auto-regressive manner. This was Y1 Y2 up to Y

**[10:03]** T dash. So, note that T and T dash are different. Meaning, you know, the length of the input and the output sequence need not be the same. &gt;&gt; [clears throat] &gt;&gt; And because we are doing a parameter sharing across time, this is not an issue at all. You know, we can easily do that. Okay? So, this is what we saw. And what was this? This was HT. The T-th latent representation or T-th hidden state. Uh and the last time, last class, uh

**[10:32]** the a student made a very interesting comment that uh all the information about the input sequence is now being enforced to be embedded onto the last hidden state. And that is being uh propagated through the output generation. And everything about the input, the burden of uh compressing all the information about the input is put on one vector HT. Uh you told you you made that comment

**[11:01]** last time, right? So, which which may not be a a great idea. So, people saw this that okay, this was happening. And to ensure that each of the output uh vectors at all time frames actually is looking at all the input time frames. So, they defined they designed this algorithm and they did they did they defined they came up with this idea called attention. What's the idea, mathematically

**[11:27]** speaking? So, this was H1, right? This was H1 H2 up to HT. They said, "Tap all of these, okay? And add them with a particular linear combination. And give all of these as an input to these things now." Understand? What we are saying is YT, okay? At every time, now becomes a function of

**[11:57]** uh the previous Y, of course, okay? And also a linear combination of all the hidden states You see what is being done here? So, you tap the hidden uh representations from each of the time steps in the input side. Scale them with alpha ones to alpha T. And all these alpha one to alpha T are learnable.

**[12:26]** Okay? And then every at the the output at every time step, make it a function of the previous output, of course. That we have done by architecture. Also, the the linear combination of all the hidden states corresponding to the input side. So, now why was this made? This was made to ensure that uh the the burden of

**[12:52]** carrying the information all the information about the input is not uh laid only on the final hidden state, but you take into consideration all the hidden states of the input and take a linear combination of them and give it as an input to uh the the output uh transform output uh LSTMs, right? And by the way, I I gave a a nomenclature also last time, right? This part is typically called the encoder and this part is called the decoder. So, this is the encoder-decoder

**[13:21]** architecture where uh you have you take all the input tokens and compress it into a particular hidden state and then decode it, so to speak. Now, this is what is called as the attention mechanism where you attend, I mean, to all the input hidden states. I'm I'm I'm I'm only giving you a historical perspective of how this idea was arrived at. Uh we will look at I mean, the attention mechanism is very similar attention mechanism that is used

**[13:49]** in a transformer is very similar to this idea, but it stems from this particular thing that instead of uh burdening the last hidden state of carrying all the information about the input, you take an inner linear combination of all of the hidden states and give it as an input to all the outputs, says the attention mechanism. This idea was there. Then people thought, "Why do you even have an RNN?" You know, because we know that RNNs are bogged with this problem of the vanishing gradient or exploding

**[14:16]** exploding gradient. Despite the fact that we have some uh hacks like uh the the gating mechanism, etc., to ensure that it does not happen, but even then, the long context was still an issue. So, people thought, "Why not do away with the recurrent architecture completely and come up with an architecture that only has attention?" Now, after all, what is this attention doing? If you look at what this attention is doing, it is trying to capture the

**[14:45]** the dependencies or the the interactions between every token or every input time step through these alphas and H. Now, why not capture this completely using only the attention mechanism and completely do away with the recurrent architecture was the question. That's why the the research article or the paper that introduced this architecture is titled as "Attention is all you need." So, what is not needed is that the

**[15:15]** recurrent architecture is not needed. All you need is attention. That's how they proposed this idea. So, this is only a historical, you know, account of how and why the attention mechanism came. So, attention mechanism is not was not proposed by this transformer paper. The idea was there. This was what was called as attention. But now, the contribution of that particular research article was that they completely gotten rid of got rid of the recurrent architecture and found out an

**[15:44]** architecture that can deal with the sequence data without even having parameter sharing across time and recurrent architecture. Okay? So, this is only a historical account of, you know, why and how this thing was was was made. Now, let us look at the math of what the actual architecture is. Okay. So, now what are we given? So, let me we work with a single data point. So, given a data point, So, please note that this is a single

**[16:24]** data point which has T uh time steps. So, now every XI is a D-dimensional vector. This is what is actually called as a token in the literature, typically. &gt;&gt; [snorts] &gt;&gt; So, we have T tokens, which is one sequence uh one sequential data that we have. Now, the objective is to learn

**[16:57]** a projection that encodes This is the goal. Now, given this data,

**[17:30]** learn a projection fee, okay? Such that the uh the the output of this projection has to be in such a way that it tries to capture the interactions between every pair of token. So, that is what the goal is. Now, how do you accomplish this is the question. Now, start by representing, let capital X, okay? Be a matrix which is uh

**[18:10]** &gt;&gt; [snorts] &gt;&gt; in RT cross D uh having the data point. So, now what I do is that uh I take this data. See, this entire data point that I have can be represented as a matrix, right? Because every token is a D-dimensional vector, okay? I stack them all in a matrix and uh I get

**[18:39]** a T cross D dimensional uh matrix, okay? Which is a single data point, mind you. So, I have the single data point. Now, I want to learn a projection onto all on this data point such that the interaction between every token is captured. That's my goal, okay? Now, let's now first define of X

**[19:11]** &gt;&gt; [snorts] &gt;&gt; using three matrices. Let's do that. Let Let's define a projection Q X times WQ. And this shall be this we have T by D and this will be uh D by

**[19:42]** DQ. &gt;&gt; [snorts] &gt;&gt; And another projection X times WK. And These are learnable learnable

**[20:18]** projections. Now, what am I doing is see please remember the goal, huh? The goal is to learn a projection or a representation such that the interaction between the tokens are captured. That is what our goal is, huh? So, towards that first we will define three projections of this data, linear projections of the of this data X or a data point X by our three learnable matrices WK, W Q and WV.

**[20:49]** Is this okay? So, these for now these are simple uh these these are simply projections of data, okay, onto some linear uh subspace defined by WQ, WK, WV. See, you you all of you know that multiplying two matrices are equivalent to taking uh the vectors in the column of one matrix and projecting it onto the space spanned by the other, right? I'm hoping that all of you know that. So, from that angle you take this the data points and project onto the space spanned by

**[21:21]** the columns of WQ and same thing happens for all three, correct? Yeah? So, Q, K, and V are now three projections of data given by I mean our projection onto the space uh given by the the uh the columns of these three matrices. Okay? That's what it is. Okay. Now, why are we doing this? We are doing this to get to the point where uh we learn a projection where the interactions are

**[21:49]** captured. Is that all right? Yeah? Okay. Now, I mean uh I mean for ease of understanding, right? I mean let's say that DQ, DK, and DV are all same. See, they need not be in the most

**[22:15]** general treatment, they need not be the same. I mean I'm avoiding writing three different and just saying all of all of them are same, okay? &gt;&gt; [snorts] &gt;&gt; So, let's let's assume these three are same. Now, Okay. Now, please note that uh every row of these matrices, okay? Uh corresponds to the projection of one token in the data. The way we have defined these matrices

**[22:44]** is that every row of the of of this matrix is actually the projection of a token of data onto the space spanned by WQ, isn't it? Why is that? Because look at this X. So, X has T rows, okay, and D columns. So, every row here is one token and uh you take that row and multiply with this WQ, which means that every row of Q

**[23:13]** is projecting one token onto the space spanned by WQ, correct? Okay. So, now let us define what is called as the attention vector. Okay. Uh we will stop here and continue in the next lecture.
