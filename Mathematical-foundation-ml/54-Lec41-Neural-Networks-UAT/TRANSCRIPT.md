# Transcript — Lec 41 Neural Networks and Universal Approximation Theorem

> **Source:** https://www.youtube.com/watch?v=npYHSFuqnzs  
> **Channel:** NPTEL - Indian Institute of Science, Bengaluru  
> **Duration:** ~30 min  
> **Note:** Auto-captions cleaned lightly. Minor ASR errors possible.

---

**[00:02]** Okay, welcome everyone. We will continue our discussions uh in this course looking at different kinds of classifiers. We just finished uh max margin classifiers and support vector machines as an example for that. And by the way, one side uh note that I wanted to make is SVMs with slacks are also called soft margin classifiers. If you see that word somewhere else, I think you know you just have to know that they are the same thing. SVM with slacks are called soft margin classifiers.

**[00:30]** Now we will go on to uh another classes of uh classifiers uh or function approximators called neural networks. Now neural networks are uh the today's go-to choice for uh uh classification, regression and all kinds of machine learning tasks. Basically the distribution estimation or empirical risk minimization tasks. Now neural networks are versatile uh in the sense that uh they can approximate

**[01:00]** any function. We will see that formally in a while. There is this very powerful theorem called universal approximation theorem. We saw in our earlier discussions that the choice of the hypothesis class that we make is critical when we are solving the problem. Right now the one suggestion was to choose uh from the family of functions that are expressive enough. Now expressivity depends I mean can be defined in multiple ways and one way of

**[01:27]** defining it is how good can it approximate the underlying function or underlying distribution what whatever way you want to put it. We als we already saw one universal approximator which was gian mixture models in terms of density estimators. But we saw that the algorithm or the the iterative algorithm that is used to uh estimate the parameters may get stuck in local optima which we do not want. Now and that is from a probabistic standpoint.

**[01:55]** From a erm standpoint one good choice for universal approximators are these class of functions called neural networks. Okay. So we will look at neural networks. There are multiple improvisations that have been made on a simple neural network. We will look at uh a few iterations of it. Uh we will start from what is called as a perceptron and then we'll go to multi-layer perceptron which is also called a feed forward neural

**[02:23]** network. Then we look at convolutional neural networks and recurrent neural networks and of course the transformer which is the the current golden standard for machine learning. Okay. Now what is a neural network? So we we still have the same problem uh of uh empirical risk minimization. Same and by the way as we have been seeing in this course these these hypothesis functions are independent of the fact uh whether you are solving a supervised problem or

**[02:52]** an unsupervised problem because ultimately you're estimating distributions. We all know that right now h theta of x is a neural network. So we we all have seen so let's say let's define uh the data first. So let's say that x uh is in rd and y is in r for uh convenience. So h theta of x we know

**[03:21]** that w transpose x is a linear decision boundary or or or a linear function. Let's say that we make it pass through a nonlinearity sigma where the sigma can be the usual sigma that we have that we have seen in the case of um logistic regressor or a logistic classifier. Now let's say that that I take another weight matrix or

**[03:53]** weight vector w2 and multiply that uh with multiply the outcome of this uh logistic sigmoid with another weight. Okay. Now this I make sure that this also passes through another sigmoid and so on. So now this becomes a composite function. So this thing is a composition or composite function Right. So what we are doing is take the

**[04:45]** data use a linear matrix and make it pass through a linear uh transformation and then apply a nonlinearity you get another vector and then pass it through another uh linear transformation and then there is nonlinearity and keep stacking it. So this is called a neural network. This class of function is called a neural network. Now

**[05:15]** this sort of operation okay which is a linear transformation of data is often referred to as a perceptron of course there is a you can have a nonlinearity I mean this nonlinearity can be the sign of this decide based on the sign of wrpose X. Right? So what is it basically doing is

**[05:45]** you are taking every dimension of data and scaling it with a particular scalar or rather weighting it with a particular scaler and that's why people notionally use W to denote the parameters weights. So waiting each weighing each of the dimension data dimensions and then deciding upon deciding based on a linear combination of different uh dimensions of different features or dimensions of data. Right? So this is

**[06:12]** called a perceptron and this idea of perceptron was first proposed in 1960s. So this is not a new idea at all. So this is this has been around. Okay. [clears throat] Now people thought that this has vague connections with uh the way neurons fire. I'm sure that uh neural scientists neuroscientists do do not agree with that. But you know there is vaguely the thing is neurons fire based on the weights uh of

**[06:44]** the features or the different kinds of measurement it does. I mean vaguely people say that but again there is there is no uh uh there's no proof uh so as so so to speak on this but anyway so for us it's it's a mathematical function so this is called a single layer or a perceptron where you take a linear combination of data and and decide based on the sign of it. So earlier people used to uh learn a perceptron. So what do you mean by learning a perceptron? It

**[07:11]** is know finding out this w not based on erm but uh based on an algorithm called perceptron learning algorithm where the idea is very simple. Now suppose you are given a data set okay start initialize with a particular w okay and take a data point and see whether the given data point is correctly classified or not correctly classified. If it's correctly classified, leave W and move to the next point. If it's not correctly classified, then you change your W by adding that

**[07:42]** particular data point. So, please note that the dimension of W and the data point are same because the inner product has to be uh between the vectors of same data points. You understood the algorithm, right? If it makes a mistake, then add that particular data point. So, basically there is a decision boundary. uh if it makes a mistake then tweak the tilt the decision boundary which is and the tilt is given by the magnitude of this particular data point okay keep doing it and iterate keep iterating over

**[08:11]** data so there's a nice theorem that would show that if the data is linearly separable this particular algorithm convert this in finite number of steps I think this will be done in tutorials so it's a it's a classical algorithm this is called the perceptron learning algorithm okay you understood the algorithm right so the WT + 1 is WT + XT where XT is the TH data point if it makes a mistake.

**[08:39]** Okay. And that plus can be you know minus XT depending upon whether it's it it has come up I mean the decision is based on the sign if it's above then it's plus and if it's below then it's minus. What can be shown is if the data is linearly separable then this particular algorithm convert this in finite step which will be done in tutorials. Please have a look at it. This is perceptron learning algorithm. See, however, this perceptron is limited only to linear separability and also it does not approximate all functions. So, people moved uh away from perceptron and

**[09:09]** then uh they came up with this idea called a multi-layer perceptron. MLP is also called a feed forward neural network or fully connected neural network as it is called. Okay. So what is the idea? So here it was wrpose X decide based on the sign in a multi-layer perceptron you start with some W1. Okay. So this is W1 * X.

**[09:41]** Okay. So instead of having a sign so you have a sigma. Now if W1 uh okay let's let's do that once. So let's say that X is in RD and Y is in RK the most general setting. Now now here W1 W1 is a matrix of size. So this is uh let's let's write it as W1 itself. So

**[10:11]** this is dx1 and this should be [clears throat] l1 by d correct? Yeah. X is DX1 and if W1 is L1 by D and uh the sigma that we do is element

**[10:36]** wise then this will be of dimensions it's an L1 length vector correct now I'll have another W2 which is a matrix now W2 is a matrix of dimensions are L2 cross L1 right so we get a L2 dimensional vector okay and then I will do another Okay. Then uh let's say that we have

**[11:35]** another W3 here and W3 can be R L2 cross K, right? So that you finally get a H. So here theta is W1 and W2 and W3. Is this all right? This is how you construct a neural network. So this is a

**[12:05]** three layer neural network. But I've heard about deep neural networks, right? the number of layers that you have, every W is called a layer. Okay? And number of W's you have uh is called the depth of the neural network. Okay? So this is the most general case. Suppose you want your uh uh the output to be a to be for a classification problem then you have to have a soft max

**[12:35]** after this w3. You understand? So it'll become a klength one hot vector. And if you are solving a regression problem, you stop here at W3 and so on. Okay. So, sigma here is so element wise nonlinearity. So, one example of this can be sigma of

**[13:02]** t is 1 by 1 + e power minus. This is one one example of a sigma and we'll see multiple examples as we go later. uh further in the course any questions on this the construction of a neural network I mean the number of elements in all these W's are called the parameters you have here how many parameters do we have l1 * d * l2 * plus l2 * l1 plus l2 l2 * k correct so these many number of

**[13:31]** parameters is what we have this is a multi-layer perceptron or a feed forward neural network or a fully connected neural network and so on why is it called a fully connected neural network. We'll get to know that in a while. Okay. Any questions on this? Yeah. &gt;&gt; Did not make any of such statements. It is just a composition composite function of data. That's it. You know, which involves linearity and nonlinear

**[13:59]** transformations. I did not say what does it effectively capture or anything. &gt;&gt; You can't say anything about it. You know, we know what it is, right? It is just a hypothesis function. when we do erm we are minimizing the empirical risk that's it we should stop there and we have every theory that comes up right we have the bias various de composition and we know how the error distributes and so on so yeah so it's it's a mathematical construct what is more interesting is the kind the theorem that I that I'm

**[14:27]** just going to talk about in a while yeah but do you understand the mathematical construct of what a deep neural network is this is effectively what is actually driving AI. Okay. Yeah. &gt;&gt; I told you right. This is one example. Oh, you mean every layer it can be? Yes, it can be different in every layers, but typically it is chosen to be the same for convenience. Okay. Uh people have empirically found out that having

**[14:54]** different nonline nonlinearities at different layers uh does not lead to gains in terms of uh performance. Yeah. should be &gt;&gt; k L2, right? Yeah, because finally we need a K dimensional vector. Thanks. Anything else? Okay, so let's move on now. See why this particular form, right? I mean this

**[15:22]** seems arbitrary, isn't it? So you have uh you have some linear transformation of data, then nonlinearity, then linear transformation. Why I mean why should this be so powerful is because of this particular theorem okay which is called uh this came in 1980s this this theorem is called universal approximation theorem I'll just state it without uh the proof okay but let's write that write that down so there's a theorem called universal approximation theorem It says let X

**[16:05]** uh be a compact subset of Rn can say uh R D also R is fine and the space of

**[16:33]** all continuous functions. Continuous functions f from X to R I'm doing it by assuming that r I mean the uh the domain or range is r can be generalized to rk as well. So then let sigma be a function from r to r such that

**[17:06]** uh [clears throat]

**[17:41]** asymtotically goes to one and zero at both ends. This is called an activation function. Okay. Now you can choose anything. Uh it doesn't matter how it behaves in in between. And the example that we just gave here has that property, right? One end it goes to one, the other end goes to uh zero. Okay, this is a function this way. Then for any

**[18:07]** given target function, f that is there in the space of continuous functions and a threshold epsilon greater than zero. MLP

**[18:36]** with a single hidden layer n neurons such that by the way I I didn't define what a neuron is. So um if you look at this this is a matrix right if you take one of the columns of this matrix what's happening is it's multiplying or scaling each of the feature

**[19:05]** dimensions with w right and then uh taking a sum of all of it right which is wrppose x kind of operation and there are how many of them here l1 of such things here so every wrppose x operation is called a neuron this thing is called a neural network cuz you can write it that way right so here this I I wrote it in the vectorzed form write it as scalers start from dimensional uh space and then there is one neuron that would

**[19:35]** do that would take one column of w and multiply it with all x so write it as a neuron there's another neuron that takes the second column of w1 and multiplies and scales and so on there are multiple neurons and there is a network of all these neurons and that that's that's why the name neural network okay so now Here we're saying that uh there exists an MLP with a hidden layer with one hidden layer. Okay, just say instead of a hidden layer, say a single with a single hidden hidden layer

**[20:03]** with n neurons such that the network's output f of x or we'll write it as h theta h of x h of x is given by 1 through n. we have uh some uh beta

**[20:29]** I * sigma of condition that uh the supreum of

**[20:56]** the difference between this beta and w are parameters. So this is why neural networks are powerful. So what is this theorem saying? It is saying that you take any continuous function. Okay,

**[21:28]** you can choose a beta and W. So what is this? This is actually a single hidden layer neural network. Why? Look at this. This is wrpose X operation. There's nonlinearity and there is another W here. I mean I I can actually write it as just to be consistent with our notations. I can write this as W2 and here W1. That's also fine. Yeah. So this is a single hidden layer

**[21:54]** neural network. I mean this thing. So this is called a hidden layer because the output is hidden from the input. Right? So the x is hidden from w2. That's why it's called a hidden layer. So it's basically a neural network with one layer. Now this theorem says that any continuous function can be approximated to arbitrary closeness. Choose any epsilon. Okay. with a single hidden layer neural network by choosing this W1 and W2. Now please note that this theorem is not

**[22:22]** guaranteeing that any learning algorithm will get us to that W1 and W2 and it is not even telling us that telling us how many neurons should be there in the hidden layer. In fact, yeah, &gt;&gt; the summation should be over. Yeah. I mean I wrote it in the vectorzed forms, right? Hold on.

**[22:50]** &gt;&gt; Huh? No, no, no. It's just Okay, I'll write this. It's fine. And it's it's all vectors, right? It's fine. But I'll have to &gt;&gt; No, I have to somehow bring this n let me tell you. So this W1 So W1 is a R L1 cross D vector and matrix and W2 is R. Yeah. So we have uh output is one, right?

**[23:19]** Yeah, this is correct. We're assuming that uh the domain is R here. You see this is R. So if it's R, the output has to be in R. So the dimensions match. Now they do, right? W1 is X is R. X is in RD. So let's say X is in RD. If it's in RD then uh D by1 W1 is &gt;&gt; L1 by D and this has to be

**[23:54]** uh L2 cross not L1 it should be the output should be one-dimensional let's match the dimensions now so so there should be n uh uh layers in the hidden no the the hidden layer, right? The first layer has uh D and then you have uh W2. The first dimension of W has to be D, right?

**[24:23]** This is let's write it down. So this is R D by 1. So this should be D by N, right? And this should be we get an N dimensional vector. This should be n by one. See, I'm assuming that my output is one-dimensional. &gt;&gt; Oh, this is W2. Sorry. Sorry. So, this

**[24:50]** should be I was just matching this. Hold on. Uh, so this is W1 W12. It should be d by n. So, you get an n dimensional vector. It should be n by d. Should I put a transpose here? This is n /d. This is n byd. So you get So you have n neurons [clears throat] in the hidden layer. Correct. And this

**[25:17]** should be &gt;&gt; 1 by n. Correct. Yeah. Should be 1 by n. Yeah. This is w2. Correct. Yeah. Now you have a neural network that has n uh neurons in the hidden layer. Okay. And this theorem tells you that you can approximate any function to arbitrary closeness by this sort of a function. And uh if I mean as I said it is not

**[25:46]** telling you how many neurons should be there in the hidden layer. Neither it is guaranteeing that a learning algorithm like erm will take you to those w1 and w2 that will make this this difference bounded by epsilon. It is it is an existential proof. It is simply saying that you can find a function that of that sort. Does that make sense? Now the question is while this theorem tells you that all you need is a single hidden layer neural network, why are we looking at deeper neural networks?

**[26:15]** Today we we we have deeper neural networks, right? We don't have shallow and wide neural networks. We have no we don't have u yeah this is this is this is deep &gt;&gt; and this is shallow. This is wide and narrow. Yeah. So, [laughter] so we don't have shallow and wide neural networks. We have deep and narrow neural

**[26:45]** networks. No. Okay. Let me write that. So, this is depth and this is this is what this is deep versus shallow. Correct? And this is or you want to call it wide and narrow &gt;&gt; this direction. So what we have is deep and narrow neural networks not shallow

**[27:14]** and wide neural networks. Why? &gt;&gt; Not because of that reason. It is purely because of the computational convenience. See what people have found out is that uh if you want to approximate uh with ifro want to approximate naturally occurring functions with uh the neural networks with limited width then you need a lot of neurons in single layer. Are you people aware of this algorithm called

**[27:41]** fast 4year transform? Do you know what the idea there is? The idea is actually this. So the computations that you do in one uh one layer so to speak, right? Basically when you have composite functions okay you can approximate a complex function by having multiple layers of compositions is what you can show with lesser number of parameters. The same idea that is used in a fast 4year transform FFT kind of algorithms

**[28:10]** is what the idea is. However theoretically speaking people have demonstrated that whatever results you can obtain with a deeper neural network can be obtained using a shallow neural network as well. just that the number of number of neurons in the particular hidden layers have to be increased. Okay. Come again. &gt;&gt; There's no theorem. This is all empirical because theorem only tells you that that all you need is a single

**[28:38]** hidden layer. These are only empirical uh observations. And by the way, do you know the number of parameters that are used in today's neural networks? They're of orders of billions. Yeah. So I I also tell you I mean in a couple of classes I'll tell you the two today's neural networks uh how is the and by the way uh the the particular functional form that is used for this neural network is referred to as architecture.

**[29:13]** So this is a three uh layer architecture neural network and so on right see the architecture that is used in today's neural network we will see that as we move on in the class but yeah so that's the idea. Yeah. &gt;&gt; Good question. [laughter] The question that's nominator. So the question is if you call this a three layer neural network then the one that is used in

**[29:41]** universal approximation theorem has to be a two layer. It is a two-layer neural network with single hidden layer. This is a threelayer neural network with two hidden layers. This is just to save my face, right? [laughter] So I can actually call it a two neural neural network. Depends upon nominator. That's it. Yeah, good point. Okay, so let's move on. So now we will use deep neural networks, right? See, one thing that is to be seen is how do we perform an erm on this. So finally we'll have to start with a loss function

**[30:10]** and then calculate the empirical risk and then find out what the uh I mean find find out an algorithm to find these parameters. Right? Now how do we do this is the question. We do it using gradient descent. Okay. So let's look at uh erm on neural networks. Uh we will stop here and continue in the

**[30:36]** next lecture.
