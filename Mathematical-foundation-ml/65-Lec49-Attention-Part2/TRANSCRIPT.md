# Transcript — Lec 49 Attention Part2

> **Source:** https://www.youtube.com/watch?v=TGzZ8jCX4y0  
> **Channel:** NPTEL - Indian Institute of Science, Bengaluru  
> **Duration:** ~33 min  
> **Note:** Auto-captions cleaned lightly. Minor ASR errors possible.

---

**[00:02]** Welcome back. So now let us define what is called as the attention vector. So given a row of Q uh small Q that is there in uh R DV. Okay.

**[00:44]** Compute. Okay. to get the attention. Okay. attention is again you know it's it's mathematically a vector that we define okay let's define that in a while the first thing that we do is so define si to be skranspose

**[01:13]** ki for all k in matrix k okay what do we do take please note that uh we are defining the attention vector corresponding to one row of Q or rather one one token every row of Q, K and V corresponds to

**[01:41]** projections of one token of data. So we are considering one token of data by taking one row of Q. Okay, you take one row of Q and define attention for that. You know we'll extend this for all tokens in a Y. So for now let us define the attention for one token of data by considering one row of Q. So you take one row of Q. Okay. And then compute the inner product of that one particular row that we have considered with all the row in matrix K.

**[02:14]** Do you see what is being done now? Okay. Okay, we are computing the inner product of one row of uh Q with all row of K and calling them SI. And how many SI do we have now? &gt;&gt; We'll have capital T of them. Okay, so this I is equal to 1 to T. Is that all right?

**[02:41]** Now let S be equal to Q transpose capital K transpose. Okay, which is a T length vector. the

**[03:17]** similarity &gt;&gt; Q &gt;&gt; uh dimensionality, right? Q is a row vector. No. Q is a row vector. This is 1 cross DV and this is

**[03:54]** DV cross T. Correct? Because Q is a row vector. Now if it's a row vector then it should be qranspose, right? Okay. So now you get this Tlength vector. Okay. corresponding to one row of Q. And this vector S is having the similarity between one token or one row of Q. Okay. With all the other tokens in K.

**[04:26]** Is that okay? Yeah. where &gt;&gt; yeah see Q transpose is a Q is a row vector right and uh &gt;&gt; K is a row vector so this becomes an outer product then no no

**[04:54]** &gt;&gt; Q into &gt;&gt; yeah because these are all rows right okay fine Yeah, we want a scalar at the end of the day. So if we treat Q as a row vector then it's qki transpose correct. Okay. So now this &gt;&gt; no so the see I is the running index. No see we are considering one row of Q. One row of the Q matrix is what we are

**[05:26]** considering to define the attention completely. So we take one fixed row of Q which is what I'm calling as Q small Q right then I take this and get an inner product with all the rows of K that is what SI is and I stack them all and make it a vector which is which is S correct yeah so this qranspose Kranspose so please note that this Q is small and K is the matrix so we are taking inner

**[05:54]** product of one row of Q with all rows of K. Okay, which to get this S. Then we scale this and uh define SAP as so S divide it with uh square root of DV. So this scaling is done because you know D is typically very large. So if you don't scale it

**[06:22]** then it'll run to the numerical issues and you know you want this to be uh meaningfully conveying the interaction. So you divide it with uh the root of dv. Then the vector s. Okay. What is this vector s? This vector s is quantifying the similarity between &gt;&gt; one. Yeah. &gt;&gt; Yes. Q transpose.

**[06:51]** &gt;&gt; See Q &gt;&gt; Okay, let's let's let's get this correct. Now Q is in R DV, correct? So if we let's let's fix one thing. So if we treat Q to be a row vector

**[07:20]** then it should be Q transpose &gt;&gt; you mean here &gt;&gt; this is qk transpose. Yeah. Okay. Correct. That's correct. Yeah. If you treat Q to be a row vector, then

**[07:52]** this should be Q transpose. It's correct. Yeah. Okay. So now um this S contains okay or quantifies the similarity between one row of Q and or or rather one token with all the other tokens. Correct? However, we want all of this to be bounded between zero and one. Why? to normalize you know to ensure that

**[08:20]** these will correspond to the relative uh scaling or relative importance of each of the pairs of uh the uh the Q and each of the pairs of tokens and if you have a vector okay so how do you ensure that it is bounded between zero and one just do a soft max right so let's do that let's call alpha to be equal to soft max of between

**[09:23]** Q and all the elements of K. Is that okay? We are doing all this for one vector uh in Q. So you take one vector. What are we doing after all? Right? We're computing inner product between one vector and all the other vectors in K. Scaling them so that the similarities are bounded between 0 and one sum to

**[09:52]** one. That's it. Okay. Finally, what we do is now define attention. So attention vector A for Q as now this AQ is the

**[10:31]** linear combination. of all the vectors in V. that's it. Understand? Okay. So now what are we doing in starting with given data given a sequence. Okay. We projected this sequence onto three subspaces and got

**[11:00]** three vectors Q, K and V. I mean yeah not three uh we got uh uh three matrices Q and V which had the vectors corresponding to each of the token. Now to define the attention corresponding to a particular token what did we do? we considered the linear combination of okay all the other tokens. So basically attention corresponding to a a token is given by the linear combination of

**[11:31]** the V vectors corresponding to all the other tokens. Got it? Now attention corresponding to each token is always the linear combination of the V vectors for all the tokens. So what changes from one token to the other is how much should you scale each of the V vectors okay is determined by the inner product between the Q vector corresponding to that particular token and the K vector

**[12:00]** corresponding to all the other tokens. That's it. Do you see what is happening here? And by the way uh people give names to this. The Q matrix or Q vector is called the quiry vector. and K is called the key vector or key matrix and V is referred to as the value matrix. See to me all three of them are

**[12:28]** projections of data. Okay, you're just projecting the given data onto three different subspaces. One subspace is called the query subspace. The other is called the key subspace. The third is called the value subspace. Now to compute the attention corresponding to a given token, okay, you use the the query representation, okay, and attention corresponding to each of the

**[12:55]** query token is simply the linear combination of the value tokens always. And the scaling of this linear the the linear combination of value tokens is given by the inner product between that query vector whose attention you are computing and all the other key tokens. That's it. &gt;&gt; So what is alpha I like alpha was defined for one

**[13:24]** &gt;&gt; the iat yeah the iat component of alpha. Say alpha is a tlength vector. No alpha is a tlength vector. You're taking the linear combination of t vectors in the value matrix. Each of the uh each of the vector in the value matrix is scaled by one component of alpha. That's it. &gt;&gt; Who is asking?

**[13:55]** &gt;&gt; It need not be the same. In the most general form, it can be they can have different dimensions. But DQ and D &gt;&gt; they are to be the same of course. Yeah. Otherwise the inner products won't work out. Okay. But yeah did you understand the the attention mechanism? This is what is called as attention. So now did we are we uh done with the objective that we started uh with? So we wanted to get a projection of data that would capture the interactions between each of the each of the tokens. Did we do that?

**[14:25]** We actually did that. How? Through alpha. Okay. So our hypothesis is as follows. The representation corresponding to each of the token in a sequence is given by the linear combination of some other projections of each of the tokens. That's the hypothesis. That is what attention is all about. Why? Because remember this treatment right here we are saying it's a linear combination of representations of hidden states

**[14:53]** corresponding to each of the inputs. That's exactly what we are doing here. The representation corresponding to each of the input token is given by the linear combination of another projection of data is what we are saying. Now how do you define what linear combination should be taken is through the interactions of that token with all the other tokens in the sequence. That's it which is given by alpha. Hold on I'll take questions in a while. I'm just trying to algebraically tell you what's

**[15:21]** happening here. So is the algebra clear? And by the way let me complete the story. So this is attention corresponding to a single token and you do it for all tokens. Okay, if you do it for all tokens typically, so attention mechanism, attention corresponding to Q, K and V, okay, is given by see all these are matrices. Okay? And

**[16:00]** the multiplication that we are doing here are matrix multiplications. This is the equation that you see everywhere, right? So what you get after this when you do this, what do you get? So this is basically you see this entire thing as a projection of X start with a T cross D matrix. What you get is a representation which is in the DV space. That's it. The

**[16:29]** output of this so-called attention block, okay, it can be seen as a function on your data that projects onto another space of dimensions DV. That's it. You see what's happening here, right? So, we started with this goal, right? We wanted to learn a projection of data onto a space such that this projection captures interactions between each of the individual tokens in the sequence. And

**[16:59]** we did this through this particular set of transformations. All this is I mean except for the softmax operation here everything is linear. Right? Definition of Q, K and V all of them are linear projections. Okay. And definition of attention also is linear because you're taking a linear combination of the columns of or rather the elements of V or the rows of V. uh but there is a softmax in between you

**[17:28]** know that that is defining what should be the scale with which all of the value vectors have to be scaled so that you get the attention corresponding to each of it you see this so look at this entire attention mechanism as a black box that would take your data which is in dv dimens dimensional space and projects projects that onto a dv dimensional space and this projection is defined in such a way that the interaction between

**[17:56]** each of the tokens within the data point is captured. Okay. So give a given a t cross d sequence it will give you a t cross dv sequence. Okay such that the interactions are captured and this is the uh algebra behind it. So let me answer questions now. &gt;&gt; You had a question. Yeah. Uh so whenever we are planning to go ahead with the projection so we can

**[18:24]** actually multiply by X with W which is P cross DV and then we can directly get this projection. &gt;&gt; Mhm. &gt;&gt; So we took three matrix no which also takes to this. Now even if we take five kind multiplications also it will take to this. Why did we stop with three is my question. &gt;&gt; Oh okay. Uh the question is what is sacroscent about this particular choice of three projections. See you the idea is this right? You want an

**[18:54]** inner product to you somehow the somehow there should be an inner product between the pairs of tokens. &gt;&gt; That is &gt;&gt; right. So you need two of them. &gt;&gt; Yeah. &gt;&gt; And then you you scale them with the input itself. In fact, there are studies where where people show that you can do away with the V matrix and have only two of the projections and scale the X itself to get the similar effect. Does that answer your question? Yeah, people have shown that. Okay. Inner products

**[19:22]** have to be taken and then use that inner product scaled inner product to uh take a linear combination of several tokens. Instead of taking linear combination of tokens, you take the linear combination of projections of tokens. That's it. Okay. And there have been studies where people have shown that you can do away with the value so-called value matrix and have only two projections and still get the same effect. So there the definition of attention will be alpha i * x i.

**[19:49]** &gt;&gt; Yeah because each of the vector is attending to the other vector through inner products. mean &gt;&gt; see when I if I ask for your attention I'm asking you to align with me right so alignment mathematically sort of inner product I mean that's the that's the closest explanation that I can give

**[20:20]** &gt;&gt; us we mean by attention that we filter out most of the information and only attend to it &gt;&gt; well no I mean see I there's this is all it it is to it and as you know I can I only care with the math right so there are inner products general linear combinations. That's it. Now a more interesting question to ask is why would this architecture okay so lead to uh good approximations in terms of empirical risk minimization. There

**[20:48]** have been results that would show that this is a universal approximator. In fact this is not a transformer yet the transformer architecture still has a few more ingredients on top of it which I'll let you know which will make it a universal approximator. So as long as it is a it is a universal approximator I don't care. However, what has been shown is that this sort of an architecture scales with the length of sequences. This does not have the problem of

**[21:17]** vanishing gradients. Why can somebody see that? Because there is no recurrence. There are only inner products, right? There is no there is no recurrence here. And there is no multiplying with the same matrix multiple times. And the when when the gradients flow they don't flow in I mean that direction which would create the vanishing gradient problems you know that's why it could scale for larger lengths of sequences and that's why this is the state of I mean go to architecture for sequence modeling now

**[21:45]** okay as long as it's a good regularizer and it is a universal approximator that's all I care for okay yeah any other question &gt;&gt; sir couple minutes back You mentioned the in the definition of attention that um for a specific query vector uh we are actually like doing some kind of um weighted sum of all of the value vectors

**[22:13]** of the entire uh &gt;&gt; not some kind of exact linear combination of value vectors for each query vector. &gt;&gt; Do we include that specific query vectors uh token as well in the summation? You do know because if there are t number of tokens &gt;&gt; you'll have to do that as well. &gt;&gt; So basically can attend to itself. &gt;&gt; Of course it will right.

**[22:40]** &gt;&gt; Yeah. So why are not only considering the alpha alpha? Why we are multi another vector? So this is see alpha is a scalar. Alpha I is a scalar. &gt;&gt; I ala &gt;&gt; they are all scalers. &gt;&gt; VI's are vectors right? So we are looking at linear

**[23:09]** combinations of value vectors. What you need given a vector q you need a vector that corresponds to the representation that that that that corresponds to representation of this vector. So look at this right. Finally the projection is that given a vector you need a vector alpha I is a scalar right so I need a vector &gt;&gt; alpha 1 you're saying the entire vector itself

**[23:39]** I'm sorry alpha itself as a representation is it one can do that one can take only alpha as a representation corresp as the representation for a given query, right? But the thing is uh it's not the it's not only the I mean I can only give you an intu intuitive way to do it. I mean mathematically speaking it's possible you can use alphas themselves as attention and go ahead. However, my intuition says that

**[24:09]** what you actually care about, right? What the hypo as I said the hypothesis is that the representation corresponding to each of the token is a linear combination of representations corresponding to all other tokens in the sequence is the climb. So to do that I mean it's not the attention that is a better representation. It is the linear basically I'm saying that every token in my sequence has information about every other token. So if I have to represent

**[24:39]** uh one token I will represent that as a linear combination of all other tokens. Right? So that linear how much information does every other token has uh with respect to one given token is captured by alphas. So I scale okay each the representation of every token in my sequence by the amount of information that it corres that it contains uh regarding the other token and take a

**[25:06]** linear combination that's the idea again as I said no it's only intuitive so now if you stop at stop stop at alpha and say that this is the I I will use this as a representation can be used mathematically nothing nothing change I mean it does not prevent it's there's no it's not wrong to use alpha as a representation yeah Actually we got away with the recurrence matrix HD. &gt;&gt; No recurrence was w right. I mean the recurrence was yeah recurrence through hidden state is is gotten rid of

**[25:35]** &gt;&gt; hidden state &gt;&gt; correct. &gt;&gt; How does it ensure that it captures &gt;&gt; I told you right? Universal approximation that's it. So this is a function approximator stops there. Yeah, I mean all that was being done using the hidden states is now being done using this alpha matrix. That's it. Or the value matrix. &gt;&gt; At least that's the hope and claim. &gt;&gt; Yeah, go ahead.

**[26:02]** &gt;&gt; Uh we can calculate the attention by using only X &gt;&gt; instead of V. &gt;&gt; Instead of taking three different transformations, &gt;&gt; you do everything with X itself. &gt;&gt; Yeah. But like doing a learnable transformation &gt;&gt; does that like help to learn more or learn faster? &gt;&gt; Correct. Learn more. See if you do not have those WQV and WK, right? Uh there's nothing learnable here,

**[26:36]** right? But you need to learn the projection, isn't it? The projection has to be learned and that's why you have those matrices. I still did not get what is the difference between K and &gt;&gt; mathematically there is no difference there still that's why I did not I mean I did not start the entire discussion by saying that this is query key and value there are three names that's it to me there are three different linear projections of data project data onto three linear subspaces

**[27:04]** okay use two of them to define inner products use the third one to take inner the linear combination that's call them whatever you want when it's called query key value because you take a vector start with a query okay I mean match it with all the keys and then use values to take a whatever right a linear combination or whatever I mean to me start with data project it

**[27:32]** onto three subspaces linear subspaces use two to define another products use the third to take linear combination that's That's all attention is about. Yeah. Question &gt;&gt; this. No here. How are we getting V? By projecting data onto WV. So how are we getting WV WKV WQ?

**[28:02]** &gt;&gt; ERM. We have a loss function. We'll do gradient descent to get this. That's it. That's what training is all about. You know when if you train an LLM all you are doing is getting this WKWB and and you you hear that there are you know 30 billion parameters 100 billion parameters and so on. They're actually the entries of these matrices elements of this matrix actually. So what is point of scaling s to scap?

**[28:30]** &gt;&gt; I told you right uh if you don't do it then uh since dv or these vectors are of very large dimensions it'll it'll run into numerical error. So you'll have to ensure that they are scaled alpha won't be affected. Huh? Alpha won't be affected. But what happens is if you take these soft max with vectors that are not scaled then all of them might converge into one point right I mean you will not you'll

**[28:57]** not get it spread okay so ensure that uh it's it's pretty well spread you'll have to divide it with the dimensionality for a large value that's it &gt;&gt; so here we fix the capital T and this &gt;&gt; capital T is always fixed that's your data &gt;&gt; uh number of tokens in one second. &gt;&gt; Oh no, the question is how do we deal with different number of tokens in different uh data point? Okay. Uh zero padding in transformers what you do is

**[29:26]** you take the highest length. Okay. And then zero pad it. That's it. And that's what is called as context window these days. And uh the state-of-the-art models right now has a t that's equal to 1 billion tokens. Not billion sorry 1 million tokens. Right. Yeah. So 256 tokens and 512k tokens and so on. Now it has reached a million tokens just uh zero pad them and then do it. That's how it is done in practice.

**[29:53]** &gt;&gt; Yeah. &gt;&gt; The initial matrix is the x &gt;&gt; that is also learned. &gt;&gt; No no that is your data. No input data that one hot that's one hot representation. &gt;&gt; Start with one hot representation. &gt;&gt; Okay your point is correct. Typically what is done is even before doing this there's one other linear projection that is learned which is called uh the embedding right so you learn another projection before

**[30:21]** you do this doesn't matter you can start with mathematically speak you can start with a one hot vector and do all this as well &gt;&gt; let's say we want to do &gt;&gt; no you do it one point at a time right See this batch processing comes up in uh this thing gradient descent not here. &gt;&gt; I can't retention for multiple data

**[30:49]** points. &gt;&gt; That's a different question. So you can you can I mean that's shredding and also how do you parallelize training etc. That's beyond the scope of this particular course. So the way to do it is consider each data point separately and do the training. That's it. Which is the batch gradient descent that we do SGD. doing it individually would &gt;&gt; that's as I said those are all questions that are valid which will not be discussed in this particular course there are 100 ways to there's also this

**[31:16]** KV cache problem if you look at it this is uh there's a order of n squ already know or order order t^ squ because we are computing inner products here see if t is very large this is not the most efficient way of doing it okay and there are these methods called you know kv cache uh and all other methods where people have tried to you might have heard of this these things called flash attention and so on right you know where the whole problem is to get rid of this t² and try to bring it down to

**[31:46]** close to linear computations there are there are ways to do that but not for this course this course is just to introduce maybe the next course I'll talk about all those for now treat each data point individually and compute attention for a data point and then do gradient descent that's it &gt;&gt; if you concatenate &gt;&gt; point taken But not for this course. Yeah. Anything else? Yeah. &gt;&gt; So this is a universal function approximator. Why don't we use this instead of a neural network? You only

**[32:13]** have to learn. &gt;&gt; That's exactly what we are doing right now. &gt;&gt; Today's LLMs are transformers. &gt;&gt; There's only three learnable. &gt;&gt; Well, yeah. I mean it doesn't matter, right? Depends on number of parameters. Now there are I mean there are billions of parameters instead of using see this does not give you lesser number of parameters or anything. It is just another type of regularizer as I keep saying right. It's an architectural novelty. That's it. Yeah.

**[32:40]** &gt;&gt; I wanted to introduce nonlinearity to each of them. &gt;&gt; We'll do that in a while. Transfer. This is not transformer. This is just attention projection &gt;&gt; all these three vectors which we can apply individually to each. &gt;&gt; Instead of doing that they put an MLP after this. Right. That is what a transformer does. Yeah. So you need that otherwise is almost linear. No except for the softmax

**[33:09]** computation here. In fact this is the uh the fact that is used to try to bring it down to linearity because if you can approximate this softmax with something that is linear then this entire operation becomes linear and you can perhaps reduce the computation. That's what is done in some of these methods as well. Okay anyway let's not get there. Is this all right? I mean algebraically what is being done in attention? Okay. So now let's let's look at what happens in the uh in in

**[33:38]** transformer, right? Um okay. So there is another uh small bit of information that I need to tell you. There's something called multi head attention. Uh we will stop here and continue in the next lecture.
