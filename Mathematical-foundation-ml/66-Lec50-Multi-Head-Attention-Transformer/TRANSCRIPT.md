# Transcript — Lec 50 Multi-Head Attention and Transformer Architecture

> **Source:** https://www.youtube.com/watch?v=NQyNZ6Plxzs  
> **Channel:** NPTEL - Indian Institute of Science, Bengaluru  
> **Duration:** ~27 min  
> **Note:** Auto-captions cleaned lightly. Minor ASR errors possible.

---

**[00:02]** Welcome back. So there is another uh small bit of information that I need to tell you. There's something called multi head attention. So which is learn multiple So Z1, Z2 up to Z M.

**[00:39]** So where uh every ZJ is a projection on instead of learning one projection, learn M number of projections. Okay. And what changes &gt;&gt; W's, right? The uh the the protection matrices changes for each of these.

**[01:08]** Learn all of them and define uh the multi head attention take all of these Z1, Z2 etc. concatenate them and multiply it with another matrix. That's it. So this you can see this entire multi

**[01:37]** head projection as one other composite projection right of taking data and projecting it onto some some other space that will give you. So this also gives you uh a t cross dv vector right you'll have to adjust the the dimensionality of wz that way compute m different uh attentions and concatenate the representations of all of them and use another linear projection to take it to a t cross dv space. So that is what is called as a multi head attention.

**[02:10]** All right a transformer uses a multi head attention architecture. Okay. [snorts] &gt;&gt; DV only all of these attentions are uh of dimensions DV. No. add to it.

**[02:42]** &gt;&gt; What do you mean by add to it? &gt;&gt; So after we have this, the dimension of X still does not match with the dimension of the output from all the multi. &gt;&gt; No, this is multi-headed attention. No, I mean the single head attention will give you a DV dimensional vector given a token. compute m of them, concatenate them all. So you get a m cross dv

**[03:11]** dimensional uh representation corresponding to one token. Bring it back to dv dimensions by multiplying it with the matrix wz. That's it. I mean you see the entire thing as a dv dimensional dv a sequence of uh length t with dv dimensional vectors getting back another sequence of sorry a sequence of uh uh tl length sequence of length d each each and get a uh get another

**[03:40]** sequence of length t with dv dimensions that's it okay you'll have to adjust the dimension of wz right such that all these concatenations gets you to T cross DV. That's it. WV is also runnable. Yeah. &gt;&gt; That also can be done. But this is how it is done. You concatenate and then project.

**[04:13]** &gt;&gt; This is how it is done. Okay. Um Okay. So this is multi head attention. One other piece that that is needed to complete. Yeah. depends on how do you define the dimensions. &gt;&gt; This entire thing is &gt;&gt; T cross &gt;&gt; M D &gt;&gt; M * DV correct? Yeah. Now you multiply post multiply

**[04:40]** this with W. Fine. And adjust it that way you get it. Depends depends on how do you concatenate also. If you concatenate the other way, you can get the you can multiply it premultiply it as well. Depends on dimensions. You have to ensure that the dimensions match. That's it. Yeah. &gt;&gt; X is &gt;&gt; like part. &gt;&gt; No, no, no. It's not done. It's not done that way. It is take the same X uh project uh I mean get get multiple attentions and then concatenate them and

**[05:10]** merge them. Okay, that is how it is done. what you said also is done at times you know where suppose your dimen data dimensionality is very large split your data dimensionality into uh parts and learn attentions individually to all of them and then merge is also done but that's not often this is how it is done in most of the cases okay [snorts] so there is one other missing piece that is needed go on &gt;&gt; what do &gt;&gt; yeah so the question is what's the need

**[05:42]** to do multi head attention why not we do single head attention uh the empirical claim is that every head will learn different every head is expected to learn different features of data it's very similar to having multiple kernels in a CNN in a CNN we have we learn multiple kernels at each layer isn't it multiple filters very similar to that that's it you can you can learn with one filter empirically it has been observed that

**[06:11]** having multiple multi heads will will lead to a better uh accuracy, better performance. That's it. There's no there's no mathematical way to answer this question and that's why I refrain from answering this, right? You it's it's it's only it's mostly empirical. [snorts] See, I'm not a big fan of teaching architectures for this precise reason because because I don't know what to teach. I

**[06:39]** mean other than algebra okay there is some regularizer that is being see please realize that at a philosophical level these questions are exactly the same as asking why are we choosing this particular prior for basian learning do you see the equivalence exactly the same question we put prior because we believe that this is how the data is that's it and that has that that's that's only a belief

**[07:08]** Okay. So these are all basian regularizers that have been put and empirically it has been observed that that this will lead to good performance. That's it. Okay. Uh and yeah so finally there is one other uh type other type of regularizer that is done regularization that is done. See in in in all so far the regularizers that we have that we have imposed are on the parameters of the hypothesis isn't it? Which means you

**[07:36]** have a function h theta of x. So we had h theta of x was the hypothesis that we were considering which is a function from x to y and we were regularizing right but if you look at it if h theta of x okay e is a composite function let's say that h theta of x is a composite function this way where w1 and w2 are your theta correct

**[08:06]** theta W1 W2 correct this is our theta now suppose I call this as W2 * Z okay where Z is simply sigma W1X so we I already told you right I mean you can see uh the neural network or any composite function as repeated projections of data onto different subspaces okay Now,

**[08:38]** now this entire h theta, okay, h is now a function of both theta as a function of theta and z as well, isn't it? So h can be viewed as a function of theta and z as well. Now if we can

**[09:08]** regularize theta, why can't we regularize z as well? So I'm I'm saying that I restrict the way this sigma w1 of x is being computed. I mean I've already restricted by making functional form assumptions. I can put more assumptions on uh on on Z right. So this regularizing the representations

**[09:37]** okay uh is something that can be done is also called activity regularizers and so on. So these are these are parameter regular regularizers. These are activity or representation regularizers. Okay see mathematically both of them have the same effect because h is a function of both theta and z. Now if you can regularize theta by putting restrictions on theta, you can regularize z as well by restricting z. Okay. See all the techniques uh that are or rather a class of

**[10:06]** techniques that are used to regularize z are called normalization techniques. Okay. And there are several of them. No. Okay. There are several type of

**[10:34]** normalizations. one is called batch normalization and uh this thing is called layer normalization and so on. Now what's the idea? Replace Z okay with Z minus mu Z divided by sigma Z actually you uh scale this with some

**[11:07]** gamma and add this with some beta mu and beta where uh mu z is simply uh 1 by b zis and uh sigma z is variance of zi. Okay. Now what am I doing is that after I compute uh Z for see what is what what is happening here in a neural

**[11:36]** network let's say that you take one particular layer the output of one particular layer that's what we are calling with Z okay now before we pass it through the next layer we will we will do this operation which is subtract it with the some mean and divide it with some variance and then pass it to the layer. Now the question is what how do I compute this mean and variance is what differentiates batch

**[12:06]** normalization and layer normalization. So first understand this normalization idea. So what am I doing is that in a neural network. So let's say that I have a neural network this way and this is my x this is my first layer z1 z2 z3. Correct? Okay. Now if I am regularizing uh Z1 then before I pass my Z1 to Z2 I do this normalization operation which

**[12:33]** is a fixed operation you see this operation is a fixed operation there's nothing when during back propagation there is nothing to learn here correct is that so actually not this new and beta are learned I'll come to that in a while for now just remember this just look at this let's Let's ignore this for a while. There is nothing to be learned. What you do is that you remove the mean and scale it with the variance. Why is this done?

**[13:01]** Do you see this as regularizer? First of all, why is this a regularizer? Because I'm restricting what values that this Z1 can take, right? After it goes, I am subtracting the mean and scaling with the variance. Why is this done? it it is done to ensure that all the the distribution of this Z1 before it goes to the next layer has zero mean and unit variance. Typically this problem of changing mean and variances across layer is referred

**[13:29]** to as covariant shift. And to avoid that there is no covariant shift okay or avoid coariant shift you do this zcoring or normalization. That's the idea. Okay. Now you also see a a new and beta term here, right? These are learned. Now why are these put? Because suppose this neural network does not want

**[13:57]** normalization. Then there should be a facility to unlearn whatever you have learned here. That's why there are the see you can you can uh uh uh you can fix new and beta in such a way that this entire effect will be undone correct this should be one by this should be &gt;&gt; sigma and this should be mu that's it so that's why you learn this now typically this is put as a intermediate layer

**[14:28]** so this you put a normalization layer here see what is a normalization layer you just do these operations and when you back propagate you have to take the gradient with respect to mu and beta as well that's it do you understand this okay now the question is this mu and sigma how do we compute I mean what is the mean taken over in batch normalization you compute the

**[14:58]** mean over all the data points that come in a batch while you're doing gradient descent Is that okay? In layer normalization, okay, the mean and variance are taken across the dimensions of data in that particular layer, not across different data points. Do you understand the difference? The difference between the batch normalization and layer normalization is

**[15:26]** that in batch normalization you compute the mean and variance by considering the data points by considering different data points across a batch. In layer normalization the mean and variance are computed across different dimensions of data for a single data point and then the means and variances are subtracted for each of the dimensions. That's it. That is layer normalization. Hen

**[16:04]** &gt;&gt; is this is this clear this this is called activity regularization right I mean where you are restricting the values that your Z can take by normalizing it [snorts] &gt;&gt; is it like the models will learn &gt;&gt; through mu and beta yes they have to they have to learn Does this also help? &gt;&gt; There's no it does not. No, unless you

**[16:33]** have an identity, it does not help in that way. You need that identity matrix, right? We saw that the vanishing gradient problem can be solved only by having that identity connection, right? That only that should happen. I mean, yeah, that also is there in transformer which we will write in a while. But this is not for that problem. This is to avoid that covariant shift problem, right? Where you want uh the the distribution of all the intermediate representations to have zero mean and unit variance. And by the way, I said

**[17:01]** that these Z's, right? They are called embeddings, right? They're called embeddings representations are also and also called features. Different people use different terminologies for it. Is that [snorts] all right? Now we are we have all ingredients to write a transformer. Yeah. we use layer and CNN we use like how do we choose between the two &gt;&gt; I refrain in answering that question so empirical

**[17:31]** there is no uh there is no mathematical answer to that question of course the observation is in some architectures batch normalization layers are used in some architectures layer normalization are used is used now how do we know what to choose M &gt;&gt; sir uh when we do training on complete data let's say for all batches will mu and other &gt;&gt; the beta yeah

**[17:59]** &gt;&gt; be same for all normalization layers &gt;&gt; no see mu and beta are learnable this mu and sigma will be same for all normalization layers correct if you do batch norm &gt;&gt; so let's say &gt;&gt; typically right this batchnom is only effective only if you do sddd batch training not the full data training So at the end of the training we assume that it will be equal to standard deviation. No, the question that is to be asked is right how do you compute this mu and sigma during inference

**[18:28]** during training you can compute it over batch how do you do it during inference typically during inference what is done is you take a random batch of data compute that and then use that for inference that's how it is done random batch of training data you do a random sampling from the training data compute the mu and sigma and then use it during inference that is how it is done during inference during training anyway you have a batch upon which you are training isn't Okay. Right. So finally we will write this this transformer architecture with

**[18:57]** all this putting all these uh Lego blocks together. So you start with uh start with X. We have you have X. You have a multi head

**[19:26]** attention here. plus normalization block. See what is a residual layer? I've already told you right that skip connection that identity learning is what is called as the residual layer. So you have a residual connection and a normalization on top of

**[19:55]** it. Okay. And then you have So to get the residual you need this skip connection also right. So this is it is done this way. And then finally

**[20:23]** you have a linear plus softmax and this fully connected layer right you have a different fully connected layer for every token of data for multihat attention you get you do it for all tokens right together because you need to compute the attention then you have t number of DV vectors on each of these DV vectors you put a fully

**[20:50]** connected layer on top of individually. Okay. So then you have a linear and softmax. Why do you need a softmax at the output? Because you are doing a a classification on K tokens at the output, right? So this is one transformer block. See, you have several layers of these

**[21:16]** in an LLM and that's what makes it billions of parameters. This is the the the workhorse of the the modern LLM architectures actually right it's it's mostly a transformer [snorts] now two questions right how do you back propagate I mean it's pretty simple because we know how to back propagate through an MLP right and also these residual and normalization terms and multi attention all of them are except

**[21:44]** for that softmax all of them are linear operations we know how to take derivatives with respect to them so we back propagate using the erm that's it [snorts] Okay, you can solve all problems uh using this this transformer in the sense that you know you want to solve a classification problem. I told you right. Okay, so one last piece I'll tell you. You can solve even uh the the uh the image or computer vision problems using a transformer. How? There is something called a vision transformer.

**[22:13]** Okay, where see you need a sequence to look at a transformer, right? What is done is given an image you divide this image into non-over overlapping patches okay and treat every patch as a d-dimensional token in a sequence okay the paper that introduced this this vision transformer the title is an image is worth

**[22:40]** &gt;&gt; 16 &gt;&gt; 16 by 16 words or something right so basically you make you represent an image as a 16x 16 16 cross dimensional token right and uh use the same transformer architecture instead of a CNN to solve image remember I told you in the last class right every data topology can be used with every other architecture just that you need to represent that that data using this particular form that

**[23:08]** this architecture demands better &gt;&gt; so vitart for image tasks now not CNN's Okay. So I mean it think about it right? What's what what is it doing? It is trying to capture the interaction between several patches in an image. See I keep telling you. This is a universal function approximator right? You can approximate any function and you have erm with you. This is a

**[23:36]** function approximator. There is gradient descent. Okay. Everything else is an art. Okay. entire basian learning is an art. So you choose what prior you want to put. In fact, there have been studies where people have shown that performance comparable to the modern uh day LLMs using transformers can be achieved using an MLP. That's what math says. No, all you just

**[24:06]** have to do it. I mean that that's an art. So what architecture to use? What should be the hyperparameter choices? How many layers should I use? Can't I take this architecture this way? There is no clear mathematical answer to those questions. Everything is empirical. Okay, there's just several architectures, several ways to get to the same end. Several means to go to the same end. Yeah. So today transformers are the ones that are used because you

**[24:34]** know see there are engineering hacks here, right? I mean you got rid of the the recurrent uh problem, recurrence problem which led to the long context. Now this these can handle long context but there are other problems in uh that that that come with it that the computation here is quadratic. Now then people said oh okay fine we solve the long context problem now how do we solve the quadratic problem and then people started asking you know flash attention and linear attention and so on that's the next frontier there are also these in fact after transformers

**[25:04]** came other class of architectures came which are called state space models have you heard of these these architectures called mamba and so on right okay fine so these are called state space models I'll not be talking about systems in this course maybe I'll talk about them in the next course. So SSM are again recurrence coming back. State space models LSTMs and RNNs are recurrence recurrent models. They are actually state space models. So people

**[25:32]** showed that uh the the quadratic problem in transformers can be solved using recurrence again. But what do you do with what do you do for the the vanishing gradient problem? You know they had other ways to do it. That's it. So it's it's a sort of cycle, right? When you come up with a new architecture, show that empirically it works, solves multiple tasks. All you need to do mathematically is show that this architecture is a universal function approximator which can be done because there is a fully connected layer

**[26:01]** here. If you take out the fully connected layer here, this is only a linear function and linear functions cannot approximate nonlinear decision boundaries. We know that. Okay. In fact, if there is a take-home message from this particular course, this is what it is going to be. Don't get fixated on architectures because they are all basian regularizers. Okay. So focus more on the problem that you are solving and whatever architectures as long as it is a univers

**[26:29]** universal approximator and solves your problem it's fine. So some problems CNN's do and and in most of the cases right I mean but for the engineering gains that a particular architecture gives you if it's a universal approximator theoretically you can solve any problem using that that that's the idea okay so we are done with uh transformers the next class uh I'll be maybe half of the class I'll spend on uh how to use some of these

**[26:58]** architectures for you know some of the uh usually occurring problems you know there is this thing called transfer learning multiclass learning and you know uh uh other kinds of training paradigms we'll talk about that okay and uh from the class after that we'll go to decision trees okay that is what the plan is thank Thank you.
