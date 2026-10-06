# Transcript — Lec 43 Local Receptive Field and Parameter Sharing

> **Source:** https://www.youtube.com/watch?v=rm0VmbTQE8Y  
> **Channel:** NPTEL - Indian Institute of Science, Bengaluru  
> **Duration:** ~36 min  
> **Note:** Auto-captions cleaned lightly. Minor ASR errors possible.

---

**[00:03]** Welcome everyone to the class. So last time we looked at a fully connected neural network or a multi-layer perceptron and we also looked at how to conduct empirical risk minimization on such an architecture using gradient descent. Specifically uh we walked through the back error back propagation algorithm which is a repeated application of chain rule to find the gradients of an architecture which is called neural networks. That is what we saw the last time. In this class, we

**[00:32]** will look at a few improvisations over the multi-layer perceptron. Okay. I told you the last class that these uh the choice of the function h theta that we make is also called as an architecture uh in the neural network uh community. So we will look at a few architectural changes uh for neural networks. So specifically I'll I'll talk about three broad classes of architectures. One is the convolutional neural networks or CNN's. The other is the recurrent neural

**[01:02]** networks of which uh gated recurrent units and LSTMs are uh members. And then we'll also look at transformers as uh which is the state-of-the-art architecture that is used in modern day machine learning applications. Okay. So let us start with uh convolutional neural networks or CNN's before we go on to the specific of this architecture right I wanted to make a a few observations

**[01:31]** which are applicable across different architectures of neural networks or general observations for one universal approximation theorem already tells us that all we need is a neural network a two-layer neural network with a single hidden layer to approximate any function to arbitrary closeness. So theoretically speaking all problems in machine learning can be solved using a single hidden layer neural network. Right? So then why do you need deeper

**[02:00]** neural networks? We saw the argument the last time it is mostly for computational convenience and also better convergence. So this is universal approximation theorem is a theoretical result. Empirically when you are solving erm on these kinds of architectures the kind of functional choices that we make for H theta matters. How you see that erm is an optimization problem over the loss landscape or the

**[02:28]** uh the empirical risk landscape and the number of parameters that we have is typically very high right of order of tens and hundreds and thousands or even billions these days. So in such high dimensional spaces when you are when you are sampling when you are searching uh using let's say that you have two choices one to search over one class of functions and the other is to search over the other class of functions which would approximate the same objective function the choice of search space that

**[02:57]** you make okay has a considerable impact on what point on the loss landscape or the objective function landscape that the optimization algorithm take you to does That makes sense. Theoretically speaking, a single hidden layer neural network can take you there wherever the optimal uh point is. But if you restrict, see, you can see that right any kind of uh restriction that we do or an architectural choice that we do is

**[03:24]** actually restricting the search space in the the space of all continuous functions. So you're saying in amongst all possible continuous functions that I can choose from, I will search only within a small subspace of this space of continuous functions. So every subface becomes a particular architecture. So what people have found out is that if you make particular choices of the class of functions that we are looking at for certain kinds of problems, okay, uh erm

**[03:55]** would be feasible with certain kinds of architectures or rather this feasibility can come in terms of uh faster convergence or better accuracy and so on. So there have been a lot of studies that would show that whatever you can do with a CNN can be done using a multi-layer perceptron just that you need uh a lot more training and you know uh more uh careful choices of hyperparameters and so on. Okay. So whatever architecture we see is uh a uh is a choice that is made to ensure that

**[04:26]** practically when we when we conduct erm on these kinds of function spaces they would better converge on the particular kind of problems that we are looking at. So this is the general observation. So the idea the the point is that while choice of architectures make practical uh sense theoretically speaking they you know they they do not give you any more guarantees of convergence than a single hidden layer neural network or an MLP. So you should remember this. Okay.

**[04:55]** Having said that as I said practically speaking uh making certain choices on the architecture or the function spaces that you are searching on can lead to better convergence and accuracy and so on. So therefore it becomes important to look at those meaningful choices that we make okay on the architectures for for certain kinds of problems. Okay. Now that's point number one. uh point number two that I wanted to make is that any

**[05:24]** kind of architectural choice that we make or some restriction that we put on the function space is equivalent to choosing a prior on our parameters. So why is that the case? I'll show you examples when we go to these architectures. But broadly speaking, here is the point. So what is an architectural tweak that we make? The raw vanilla architecture is that of MLP. Okay. Is a multi-layer perceptron where

**[05:52]** every neuron is connected to every other neuron in all layers. Now let's say that I make a uh I I I I make a choice saying that the weights connecting neuron 15 to neuron 16 okay should always be zero. You understand what I'm saying? Some take two arbitrary neurons and say that there should not be a connection between these two. What is that I'm doing? If you are if you if you think about it, I am imposing a direct delta prior on that

**[06:21]** particular parameter in the parameter space saying with probability one choose this particular weight to be zero or whatever value. You see what I'm saying? So every I mean you can see all all tweaks that I make on a vanilla MLP in terms of parameters as a basian prior that I'm putting on the parameter space. So that way a CNN is a regularized multi-layer perceptron. We will see how exactly it is done. A CNN

**[06:50]** is a regularized MLP and an RNN is a regularized MLP and so is a transformer. So any kind of so you take the most general function and start saying that okay this parameter can take only this value with the certain probability this parameter can take this value with certain probability and so on that becomes a new architecture which is nothing but imposing bayan prior on MLP. So that's why remember when I when we discussed regularization I told you that one other way to regularize uh is to make architectural choices.

**[07:20]** Okay that is why CNN all architectural choices that we make can be seen as regularizers. Now why are regularizers important? We know that for particular type of problems that we choose if I know how the parameters are going to be then it's a better idea to impose that on your modeling choice so that erm becomes easier. We know why we impose bias on our models, right? So that's exa exactly why and we also know that if you have infinite data then having a bias does not matter because maximum

**[07:49]** likelihood estimate also converges to the uh the true estimate of the of the parameter as well. you this is I wanted to I mean I just spent five 10 minutes on this topic because I whenever you see an architectural choice that is being made you should always see that as a choice of regularizers that we are making on the function spaces so that it chooses it it it it converges I mean whatever prop uh properties that are uh uh desired are imposed on the particular

**[08:17]** function spaces okay is that all right okay so with that let us look at convolutional neural networks so these are um I'll actually write this as regularized MLP suited for grid like topologies. Now if I have data that is gridl like so

**[08:48]** what do I mean by grid? Uh it's image is an example. So you have you can arrange your data semantically in like a grid you know a tensor basically. So see one can argue that uh uh that uh even a vector is grid right because it it is it is a one-dimensional tensor that way I agree with you. So that's why you have a 1D CNN. But to start with uh the motivation comes from the fact that if you have grid-like topology

**[09:17]** of which image is an example right then you have to have certain regularizers on MLP and if you do that then what the architecture that you get is a convolutional visual network. Okay. Now example are it's it's an image and so on. So these are gridl like

**[09:44]** topology right these can come in multiple applications. Now can't we use an MLP on image? We can we can definitely use uh an MLP on image right we're treating that as a vector but as I said this is a specific regularizer which is custom made for these kinds of data. Okay. So now what's CNN right? So CNN's so it's an MLP with

**[10:11]** two specific regularizers. I'll say hard regularizers on I'll tell you what that means. So one the idea is local receptive field it's called local receptive field. This is one type of regularization that

**[10:42]** is done. The other type is parameter sharing. across topology. So these are the two ideas these are the two regularizers that are that are put on MLP and if you do that the kind of architecture that you get is called a convolutional neural network. I'll tell you why it is called a CNN as well. Okay. So what are these things? Let us look at this. So let's say that we have data in uh in RD for

**[11:10]** now and we still have a vector kind of topology. we still have we still don't have this grid kind of data right now but this is for uh understanding s x x is in rd so we have uh dimensions of data and then let's say that we have neurons so this is an MLP

**[11:40]** so typically what happens in an MLP is that every dimension of data is connected to every neuron. Correct? So connect this all dimensions of data to this neuron. Right? And uh this neuron is also connected to all data dimensions and so on. Right? And this happens uh subsequently with all neurons. You know that right? Suppose I tell you this that I don't collect connect all data

**[12:11]** dimensions to a particular neuron but I only connect a subset of all data dimensions to a particular neuron. Does it make sense? So this is an architectural choice that I'm making. So why am I making this? I'll tell you in a while. Right? There is some intuition that is that is given which I'll uh which I'll talk about. But did you understand what is it? Instead of connecting all data dimensions to uh every neuron, I will only connect a subset of data dimensions to every neuron

**[12:40]** &gt;&gt; in each layer. &gt;&gt; In each layer in in to every neuron yeah in each layer for to every neuron I only connect a subset of data and this subset right uh is a hyperparameter. So what I do is instead of connecting this I only connect these four data dimensions to first neuron and the second four dimensions to the

**[13:09]** second neuron and third four dimensions and third set of four dimensions to third neuron and so on and I do the same thing in the next layer as well. Right? So these two are connected to this and then these two are connected to this and so on. Do you see this idea? So this thing is exactly what is called as local receptive field. So just a second

**[13:38]** is data or activations depending upon what layer are you in and by the way uh a neuron. So we we all

**[14:11]** know what a neuron is right it's that wrpose x operation that we do is what is referred to as neuron. So now every neuron has this thing called receptive field. So what is receptive field? Receptive field is simply the data dimensions or the activation dimensions that are connected to this particular neuron from the previous layer. That is called receptive field of a neuron for obvious reasons. Right? Yeah. So there is a neuron and it is looking at whatever uh uh activations or data in

**[14:40]** the previous layer is called the receptive field of that particular neuron. Now what you are doing is you are making the receptive field to be local. Instead of making it a neuron with a global receptive field, you are making it local receptive field. Is that idea clear? Any questions on this? So now how many how many neurons do we connect to the uh the next neuron is a hyperparameter and uh will there be an overlap not non-overlap is also a

**[15:09]** hyperparameter as we will see you know it's called stride in the CNN language we'll talk about it in a while right but the idea is to only select a subset of the previous layers neuron and connect it to the next layer so that's the idea of local receptive field any questions How many data points?

**[15:39]** &gt;&gt; That's a nomature question. Typically what happens is for a given uh neuron you only look at the previous the most previous layer and then talk of receptive field. That's just a language, right? So yeah &gt;&gt; question is can there be uh any neuron that is not connected to anything? No &gt;&gt; in the upper layer as well as the lower

**[16:06]** layer. &gt;&gt; Come again. &gt;&gt; It should have at least one connection. No, always in the the the uh immediate next layer till we talk about something called skip connections which we will do in a while. &gt;&gt; Yeah. &gt;&gt; Subset is a hyperparameter meaning you

**[16:32]** select which which subset you'll have to connect to. But once you do that you know you uh it it's there's an algorithm to determine meaning you know you do it in a systematic way. Yeah. It's not random. Yeah. Is that okay? Yeah. See now now you can see this right. So what I'm actually doing is if you look at the weights as I said no w1 w2 up to wp if there are p number of neurons here. If I draw a distribution of WP, right? Or uh distribution of all parameters in the WP

**[17:01]** space. By doing this, what am I doing? I'm I'm choosing a particular distribution where I'm saying that four dimensions of this W can take any values, but all the other dimensions have to be zero. So the likelihood or the probability of u p minus 4 dimensions okay taking any value other than zero is zero or with probability one these things

**[17:29]** will take a value of zero. So that's why you see this as a very hard and strong regularizer where you are saying see in most of the basian regularization case what do we do we allow all values for the parameters with certain probabilities depending upon the strength of the regularizer here we are not even I mean allowing uh uh the value of some of the parameters to take any other value than zero. So probabilistically speaking we are saying that we are putting a hard regularizer

**[17:57]** by saying that the probability of uh a few weights taking any other value than zero is zero. You see that it's actually a regularizer. Okay. So this is one idea. The other idea that is uh that is imposed on CNN as I said is uh parameter sharing. Okay, I should tell you what is the and

**[18:26]** by the way with this this guy Yan Likun who proposed this idea right of CNN the the uh intuition that is given behind putting this kind of a regularizer you know that all regularizers are man-made in the sense that you know you that's a design choice that we make so why was this design choice made is that in in an image okay the kind of features that we look at are mostly local in nature. What do you mean by that? So, so let's say

**[18:54]** that we have a picture of a home. I mean, whatever, right? Oh, just went away. Okay. So, let's write it. See, typically in uh classical image processing, the problem is that you give an image and you try to make sense of it. Ultimate problem of computer vision or image crossing is that you are given an image you understand what the scene is or you say that oh here is a here is a home here is a person here is this

**[19:22]** object that object and so on and delineate that's what human eyes do isn't it now to solve that problem which is a very high level semantic problem what people used to do in image processing community classically is to break down the problem into pieces and say that let's start from identifying primitive features like let's let's identify edges okay then let's identify Let's let's identify lines. Then let's let's identify complex pictures like uh uh like triangles and regular polygons

**[19:51]** and then try to hierarchically uh uh increase our understanding of the image to finally to come to a particular conclusion. Right? That's how that's why there are lots of algorithms in image processing just to detect these primitive uh features such as edges, right? And contours and corners and so on, right? So now the assumption is uh that these kinds of features or these kinds of building blocks for semantic

**[20:19]** construction generally appear within subsets of data dimensions. But if you look at images there are only you know the edge is only here. The entire image is not an edge. If you want to look at an edge you only look at one part of the image and so on. So if this neuron is supposed to fire what do you mean by firing? Wrpose x for that particular thing is greater than a particular threshold. We know that every perceptron can fire, right? So if this neuron has to fire for a particular feature, then

**[20:48]** it better only look at a small subset of data. So that what feature that it is looking at is a uh is locally uh restricted that way. So that's the intuition that is given for this make sense. And now what happens is as somebody was asking as you move deeper and deeper into the neural network right the receptive field of uh the neurons in the deeper layers actually keeps increasing isn't it because this has a local receptive field and the the the neuron in the next layer

**[21:16]** has a local receptive field but all both the neurons at the previous layer has a larger receptive field and so on. Now what happens is as you move deeper and deeper into the network the hope is that these neurons will fire for more complex and hierarchical uh entities semantic entities that that that has been observed as well uh uh empirically that the initial layers in a CNN uh fire for primitive uh features such as edges,

**[21:45]** lines, colors and so on. As you go deeper and deeper they fire only for semantics objects is what has been observed. that's that's where it is coming from right so people try to mimic those kinds of things in in an MLP now I'm I'm repeating it because you know I am not a neuroscientist I'm an engineer so uh and a bit of mathematician so as far as the function approximation is concerned you need all you need is an MLP but if you

**[22:13]** put these kinds of regularizers people have seen that the empirically they they better result okay layers what guarantees &gt;&gt; there is no absolutely no guarantee &gt;&gt; in a given layer &gt;&gt; even that is not guaranteed the questions are like what is the guarantee that hierarchical features are learned in an MLP there is no guarantee right what what is the guarantee that

**[22:40]** different neurons in a layer learns different features there is no guarantee okay &gt;&gt; so we have arranged our data getting associated with set of line, &gt;&gt; hold on to that question.

**[23:21]** I I understand what you're asking. Just five minutes it'll be answered. Yeah. &gt;&gt; To extract or take some subsets of data points like how &gt;&gt; I didn't quite get the question is the question. How do you decide what subset of data is to be connected to what neuron? That's a hyperparameter. I told

**[23:49]** you that's a design choice. As a as a designer, you you instruct the neural network to look at this particular subject. Okay, your question will be answered in a while. Just hold on. Okay, this is about local receptive field. And the second kind of regularizer that we impose is what is called as parameter sharing as I said. So what is this idea? Idea is let's uh redraw this diagram. Okay, let me do it again.

**[24:19]** we have dimensions of data and we have neurons being connected here. Now uh the first kind of regularizer that we impose is that only a subset of these neurons are connected to everyone. Right? Now if you go back to the intuition which led us to choose only a subset. What was the intuition? We said that features that occur in the images or any data are locally uh restricted

**[24:50]** and that's why we want to look at only a few set of uh data dimensions. Now suppose you are designing uh a neuron or you are you are expecting a neuron to extract or fire for a particular feature. Let's say that it's an edge. This edge, okay, has to be deducted irrespective of where it occurs in an image. An edge detector has to be an edge detector no matter where this edge occurs in an image.

**[25:20]** Make sense? Suppose this yellow weights that we have no not yellow. What is this color? Orange. Yeah. Did I become color blind? So this this orange neurons or orange weights that that we have connected uh is doing edge detection. Now don't ask me how do you know whether it's doing edge detection. I'm just giving you an example. It does something. Okay. And let's say that it is it is for edge detection. Now this edge has to be detected no matter where it occurs in the data. Now how do you

**[25:47]** ensure that through a regularization here by ensure by making that all the uh neurons in the first layer right are connected with the exact same weights. So what I do is I take this orange and connect all these neurons, okay, with the exact same weights &gt;&gt; always because the idea of local

**[26:14]** receptive field still exists, right? But uh I connect multiple subsets of data using the same set of weights to different neurons. So basically what I'm saying is detect edge irrespective of where it occurs in the image. Does it make sense? So this is let's say this is W1, W2, W3 and W4. This will still be W1, W2, W3 and W4.

**[26:49]** Now from a probabistic standpoint, what am I doing? So I said that in the in let's say 100 dimensional uh weight space, I'm saying but for four uh weights, 96 of them are zero. So that's one regularizer I'm putting. Okay, even I'm putting another regularizer where I'm saying that uh that okay these four of them right the the next four of them also should have this exact same value with probability one that's other regular I'm saying

**[27:20]** okay so this idea is what is called as parameter sharing this is um &gt;&gt; Yeah. See this happens everywhere. So

**[27:55]** now again here if I connect this with green, this I'll connect with the same green. This is theta 1, theta_2. This is again theta 1, theta_2 and so on. The idea of parameter sharing is there across the layers. &gt;&gt; Of course, I mean that's what we learn using erm &gt;&gt; see you are not fixing these W's, mind

**[28:24]** you. You are just doing this connection and still doing an erm. Nobody's fixing these W's and thetas. They are learned through ermity &gt;&gt; be restricted. Right? The question is won't the expressivity of this neural network get restricted? Well, there is one other layer to it that I'll talk about. I mean not layer in the sense of there is one other layer to it or rather one another dimension to it that I'll talk see all these terms are

**[28:51]** mathematically well defined &gt;&gt; one aspect to it right which I'll talk about in a while that will answer your question both of your questions will be answered at once right are we doing disservice to the topology and isn't it restrictive is something and by the way there is a universal approximation theorem for CNN as well okay so that way it does not uh restrict the explicity but in I And I was anyway asking the question myself. If we do this then aren't we saying that all the

**[29:19]** neurons in a particular layer are only looking at one particular feature. That is a valid question which I'll answer in a while. But did you understand the idea of parameter sharing is all I wanted to ask you. &gt;&gt; No, it is like if you want to if you if you if you are looking at uh detection of a particular feature then this feature has to be detected no matter where it occurs in data. So this is what you are doing right anywhere you are

**[29:45]** connecting a a subset of data to a particular neuron. Now this is this can this neuron can detect an edge that appears only in four dimensions of data. Now suppose this this particular feature appears in the fifth four fifth set of four data dimensions then it has to detect the exact same edge irrespective of where it appears. Now if you look at this home image that I draw drew, let's say that this is an edge detector. An

**[30:13]** edges here, edges here, edges here, edges here, edges here, right? And it has to detect at all points. So that's why I need to repeat the have the exact same parameter across different data dimensions. All of them are connected to a local subset of data. the first layer in the first layer. If it detects an edge there in another location, then the second

**[30:46]** detected. &gt;&gt; Exactly. Exactly. So the depending upon you know which neurons receptive field that particular edge falls into &gt;&gt; see we have a neuroscientist in the class you know there's a name for it. It's called retintopy is what he's saying. Yeah. So I mean I don't know that's not my statement but yeah you can contact &gt;&gt; erm same thing. So you just restrict the

**[31:12]** architecture you still do backrop I'll write the equations in a while right you still do the so this is a neural network no this is h theta you still have the labels you still have uh uh the the h theta you still have labels still have erm do backdrop and get it that's it nothing changes yeah good observation I mean maybe he's

**[31:40]** coming from the classical signal processing background the question that he's asking is the weights are same isn't it the filter that is there in absolutely correct &gt;&gt; it's a two dimen yeah right now it's it looks like a one dimension filter I'll show you how it's a it can represent it can be represented as a grid in a while &gt;&gt; it's essentially all the same. &gt;&gt; Yeah. So now correct that's correct. See

**[32:08]** because you made that statement I I I'll tell you a short you know anecdote. See uh in classical image processing what people were doing were coming up with uh these W's okay to do a particular function. For instance there are a lot of edge detectors. There's one example is this gabber filter. Right? The cany edge detector and so on. They are nothing but 10 numbers or nine numbers in a grid and you need to do

**[32:36]** that wrpose x operation. Now people spend their life in coming up with those nine numbers so that a particular function happens. Now neural network comes and says okay everything can be learned. Told you in the first class remember ignorance modeling right. So you let your neural network model whatever it wants. So you you're exactly correct. This is actually a filter a learnable filter &gt;&gt; not necessarily I told you right

**[33:11]** hierarch you have depth so in second layer it's not an edge it may be something else you're combining edge information to do something and by the way most of the times it is not even interpretable see there are techniques like you know grat cam and so on where you can see what this neuron is firing or right but all those are empirical you don't know what a neuron is doing and that's the interpretation problem and there are a few methods to do that but exactly what this neuron is looking at

**[33:40]** it's very hard to know that yeah mathematically speaking as I said right I mean for us it's an MLP with strong regularizers that's it and then we do erm that's all we care but of course these interpretations are pretty valid so this is a filter and it is a filter with learnable weight and so on any any other We want to remain back does that ensure that these because

**[34:11]** these get updated independently. &gt;&gt; Okay, good question. Uh the question is how do we do back prop for this right? we'll have to ensure that these uh these regularizers are imposed when you are doing back back problem right take it as a homework I mean you can do it the way to do it is when we define that uh delta right where we have that error function set the forward and the backward processes that we define right we'll have to ensure that the summations are done only over that and yeah only over

**[34:40]** only over a few subset okay and then when you are updating you'll have to update in such a way that if you are doing it for one uh one subset the other subset to which these filters are connected are not updated it's hardcoded maybe in tutorial yeah see uh one that was another point that I wanted to make I thought I would make it later every regularizer that you put on architectural regularizer that you put on MLP will change the forward and reverse uh equations no

**[35:09]** we'll have to ensure that those hard reg those regularizers hardcoded by the way you you you should notice one other thing these are hard regularizers the sense that these are hard coding that you do. Okay. Uh there are some soft uh architectural level regularizers as well that I'll talk about. There's something called dropout. Okay. We'll talk about it in a while. But yeah, so as far as these hard regularizers are concerned, how do you impose them mathematically is by changing the forward and the backward. If you change the forward

**[35:37]** equations, the backward equations also will change. Right? So you have to ensure that those things are taken care when you are when you are doing backdrop for a CNN. Yeah, that's how you implement it. By the way, anyway, I I I would have told you later. Anything else? So, you understood these two things, right? Uh these are two major uh regularizers that are put on MLP to change it to a to a CNN, right? Local receptive field and parameter sharing. Okay. Uh we will stop here and continue

**[36:05]** in the next lecture.
