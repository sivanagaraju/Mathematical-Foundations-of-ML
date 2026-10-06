# Transcript — Lec 42 ERM on Neural Networks and Error Backpropagation

> **Source:** https://www.youtube.com/watch?v=dONDRwX_83E  
> **Channel:** NPTEL - Indian Institute of Science, Bengaluru  
> **Duration:** ~47 min  
> **Note:** Auto-captions cleaned lightly. Minor ASR errors possible.

---

**[00:04]** Okay, welcome everyone. Let's look at uh erm on neural networks. So we start with uh x in RD and y in some rk get r also. So we have data as usual which are samples drawn id from the distribution

**[00:40]** h theta of x is the neural network with w1 * x sigma here and w2 let's say we have this neural network for ease. So we need our theta star to be equal to the minimal arg min of the empirical disk which is 1 / n i is 1 through n. We have some loss function y i right this is our standard erm that

**[01:08]** we do. Now how do we do this for neural networks? We use the gradient descent. We already seen what gradient descent is. This is some for times we have the empirical risk. Let's call this rcap rcap of theta. It's a function of theta. Rcap of theta. Take the derivative of

**[01:38]** this with respect to this is standard gradient descent that we do. Right? Now what is important is how do we compute rcap the derivative of rcap theta with respect to theta for these kinds of functions. Right? The question the gradient of the empirical risk with respect to theta for neural networks.

**[02:08]** The solution is the chain rule. Composite functions are the gradients of composite functions are uh computed using chain rule. We all know that right? The gradient of e power sin x squ. Start from the innermost function take the derivative keep hopping. That's exactly what we do. [snorts] Okay. See chain rule for neural network has a fancy name. It's called error back propagation.

**[02:45]** and we'll we'll work uh uh the back propagation algorithm. This is this is the algorithm that the Hinton came up with right so it's sort of uh made uh the community try neural network so to speak or I mean you know what training is simply doing radian descent within okay let's look at uh chain rule for these kinds of composite functions that we have which is called the error back propagation algorithm. See to do that uh just to ensure that uh

**[03:14]** that you know you appreciate the equations that are there no I will resort to the scalar form even though the implementation standard implementations today are done using vectorzed operations I will do it uh assuming I mean I I will assume I mean I will resort to uh the scalar form when I work out the back propagation because it it's called error back propagation for a reason. So those kinds of uh nice equations come out only if we look at the scalar form of things. Okay. So let

**[03:42]** us work uh error back propagation. So this is back prop for a neural network. So there's a lot of notations. I will just write that notation now. So notations So maybe before I write the notations, I

**[04:14]** will write the diagram. So we have data which is uh X which is in RD right? So we have D dimensions of it 1 2 3 4 up to so this is X in RD. So note that I'm now using scalers to do everything. Okay. So there is there are a few neurons here in the first layer,

**[04:43]** neurons in the second layer and so on. Okay. Now what happens is I'll take this as the lith layer. L we'll take this as the lith layer. Hold on. Just don't write yet. I'll complete the notations and then you can copy. Uh this is L minus first layer. L layer and we also need an L L + first layer. Okay.

**[05:16]** Then what happens is that this is connected to all of this Do you see what's happening here? Okay. So, in our vector notation, see, please note that every dimension of data here

**[05:47]** is connected to one neuron. You see what I'm saying? every dimensions of every dimension of data here right is connected to all neurons individually. Okay. Now in our vectorzed notation that forms one column of the W matrix. So every layer of neural network will now have one W matrix. Okay. And how many columns will this W matrix have? It'll have those

**[06:15]** many columns as there are number of neurons in a layer. Okay. every all data dimensions connect to every single neuron. Okay, whose weight with the weights that are given by one column of the W matrix that defines a layer. Is this clear to all of you? Okay. And there are as many W matrices as there are number of layers. Now, if I write all these connections,

**[06:44]** the diagram will look very messy. That's why it's called a neural network. Have you seen that diagram? Right? I mean they show that there are all so many connection that will happen. So it it'll look like this. If you want me to do this is connected one here, right? And this you see what I'm saying? This is how this is why it's neural network so to speak. Yeah.

**[07:13]** &gt;&gt; That's a hyperparameter. How do you choose the number of neurons in every layer? It's a hyperparameter. That's a design choice. Okay, there's no one typical answer to it. See, please note that uh if you look at this part, right, uh this the output of this particular layer can be seen as input to another neural network that's going on. I mean that's the nature of any composite function, right? That's how it works. Is this clear? Okay. So now this is how it is. Let me write down the

**[07:41]** notations. So let L so notations index

**[08:07]** of a neuron in the Understand? Right.

**[08:38]** And let's say I is the index of in the L+ first layer. Now this is important. We have W J K L. This is weight

**[09:06]** connecting to neuron J. Please note that neuron K is in L minus first layer and neuron J is in L layer. So W JK K L is the weight that connects

**[09:43]** K to J. So this is how it is right. So uh index of this is in L minus one K to J. Yeah. So let's say that this is the neuron that we are talking about it becomes too messy. So if this is the K or J, what is this? This is J. This is J neuron and this is K neuron. Okay. And there is a weight that connects these two. And this is W JK.

**[10:13]** So this is an element of that particular matrix that we are looking at. Is that all right? Also note that the elements of the column of this matrix that defines a particular layer. Okay. Uh those contains the weights that connect a particular layer in the next lay next neuron. All those weights are stacked into a particular column. Okay. If

**[10:42]** you're looking at scalers, then there are these many connections. W JKL defines the weight connecting neuron K to neuron J. Is that all right? Okay. Uh so we have uh uh so B J L is the the bias Okay.

**[11:15]** Now let me define a few more things here. Define A K L minus one plus B J L and this is

**[11:44]** over K where uh a J of L is sigma of CJ L. Okay. So now these two things are called the activations and uh uh the pre-activation uh of a particular

**[12:13]** neuron. What is this? See what is this operation? And this operation is wrppose X. Right? So now we know that the kind of operations that every neuron does is take the inner product and then pass it through the nonlinearity. Okay? So we'll define a variable called this pre-activation uh pre-activation output. Okay, which is simply the linear production of data before the activation function or before

**[12:42]** the nonlinearity and whatever comes out of the activity uh sorry the uh uh the uh nonlinearity is called the activation of that particular neuron. Is that all right? So this operation is nothing but wrpose x right that's exactly what it is. Now let us uh with this and by the way please note that for the first layer this a k or a jl is simply the input that all right okay so let us define this uh so z jl

**[13:13]** be the pre-activation of neuron J in layer L and we have A J L is the output or activation it's also called

**[13:41]** activation or output of J neuron H Is it all right now? Notations are clear. Let me just uh flash them again so that they're all on the same page.

**[14:15]** Any questions on notations? Now, what do we want to do? We now have to find uh the gradient of the empirical risk RC cap with respect to all these wj kls. That's what we will do now. Is that all right? See why did I write it as a scalar is because um when you write chain rule for multiple variables right do you know how chain rule applies for multiple variables? You have to take the derivative huh you

**[14:43]** have to take the derivative with respect to each path and then sum them through right so that comes up only if you write it as an explicit summation here that's why I wrote the scalers let me let me write that notations are clear let's move on now okay so now there are uh two things that one needs to do okay uh when you compute the gradient uh let's to compute the gradients Suppose

**[15:20]** so rcap denotes the uh empirical risk. Okay, we want the derivative of RCAP with respect to W. This is what we need. Correct? We get

**[15:48]** this then we can do it for every each of the parameters and then do the gradient descent. Correct? We want this. So to do this, in layer L.

**[16:24]** in layer L as follows delta J L. So write this as the derivative of R the the function that we are trying to uh take the derivative with respect to with this is with respect to this Okay. So let me just give you an

**[16:56]** overview of how we go about solving this problem before we go with the algebra. See the idea is very simple. Now here is uh where we have uh after many layers we have we have to compute rcap right we have to compute the rcap. Now if you want to find the derivative of uh rcap with respect to some weight here uh the dependence of this particular weight on rcap is through every path this wjk has seen till rcap isn't it right so we have

**[17:28]** to traverse that path that's all it is so that's what chain rule does isn't it if you have a composite function if you want to take the derivative of the outermost function with respect to a variable that is inside the composition then you traverse the entire path path back that's the whole idea so to do that what we have to do is that that we'll have to come back I mean that's why the name back we we'll see the algebra very nicely so we start to take the derivative of the loss function or the the empirical risk with respect to the

**[17:58]** pre-activation which is Z okay and then then then we use this to compute uh the the derivative of this with respect to the the layer before it and so on till we reach that WJ. That's the idea. Okay. So let's do that. Let's do the algebra. This is a very important variable. This is the the derivative of the empirical risk RCAP with respect to the pre-activation of the neuron J at layer L. Is that all

**[18:26]** right? Okay. So this is very well valid because this ZJL is a function of WJ. Right. And uh the RC cap the final RCAP will be a function of function of ZJL as well. Do you see that? Okay. And by the way, what is RCAP? uh

**[18:55]** H theta of XI, YI. Correct? Now since h theta depends on w jk and it depends on zj and that's why the derivative of rcap with respect to zj is well defined correct yeah okay so let us look at this now to compute and by the way again one more uh point to be made is the way we will go about this attack this problem is we will

**[19:23]** establish a relationship between this error component in the lth layer and the error component in the l minus first layer. You see what I'm saying? So this this so-called error is now defined for every neuron in each layer. Isn't it? Okay. Now what we will do first is that we will establish a relationship. Okay. Between the error in the L + first layer and the error in the L layer.

**[19:56]** Okay. So to reach to the derivative of or or rather to reach to the weight of JK at the layer L, we will do it through the error in the layers that come after that. You see why it it is to be done that way because the the empirical risk is computed at the final layer. The dependence of this empirical risk or rather the dependence of the weight in some layer in between uh on the empirical risk is through every neuron

**[20:26]** that comes between these two and therefore if we compute the error with respect to the uh the last layer and then relate that error to the error in the previous layer and so on perhaps we can reach to uh the gradient of this uh loss with respect to a neuron that is in some intermediate layer. That's the idea. Now the first thing that is to be done is we'll have to compute or rather we'll have to establish a relationship between the error in the l layer and the error that is computed in the l minus

**[20:53]** first layer and so on. To do that we'll have to start by computing the error in the final layer. Let that's what we will do now. Is is it clear? Okay. So to compute error for the final layer So to do this we'll have to compute uh del j l. Note that we do this for every single

**[21:22]** neuron. Right? So this is uh del rcap by del z j l. This is equal to this is okay. Why is this the case? The

**[21:51]** dependence of RCAP on ZJL, right, can be written through A as well. And please note that a capital L is the output of the neural network. H theta of X is equal to A. You see that maybe just write let me just write that by definition.

**[22:21]** Correct. Yeah. [snorts] &gt;&gt; because there is uh there is an activation in between. No, we'll just see that. So we'll have to take the derivative of that as well. Let me write that. So now this is equal to the activation that is computed on ZJL.

**[22:49]** The derivative of this with respect to ZJL is simply Do you see this? See, since Al is simply

**[23:14]** or &gt;&gt; yeah, this is Alj. Okay. The derivative of a LJ with respect to ZJL is simply the derivative of the activation computed at ZJL. Correct? That's that's obvious. See, since we are looking at derivatives of the activations, no, these functions have to be differentiable. Okay, you see this? So, we got this

**[23:44]** particular term. So, now we know how to compute this particular term. What about the first term depends upon the last function? So now first term let's let's do that. Now del rcap by del a j l can be computed y is the derivative of think it's better if we do it for single

**[24:21]** data point right so already too many uh this thing so this is equal to um ALJ minus loss computed not not minus hold on just don't write it huh alj and you have yj the j component of the particular output

**[24:49]** you see this this is what it is depending upon what your loss function is you compute it this way and this this can be computed Because we know what the loss function is and depending upon what the loss function is, this can be computed. Let me write this. &gt;&gt; Come again. Come again. &gt;&gt; I mean assuming that Y is a vector, I'm talking about the J component of the Y. If A is also a K dimensional vector, Y is also a K dimensional vector. So loss is computed dimension wise. I mean

**[25:18]** that's why I've written as derivative of this with respect to A LJ. All scalers right now. Make sense? Yeah. So can be computed. See now this means that we now know how to get this DLG correct. So whatever we are computing let me just put a tick mark there. So this we know how to how to compute this now. Is that all right? Okay. So now let's to compute the error that is there in some

**[25:54]** hidden layer lith layer for the J neuron. See the algebra seems a bit clumsy and involved but all we are doing is chain rule repeated application of chain rule. Okay. So now we'll do that. So we know that uh del LJ is

**[26:24]** the derivative of the pre-activation of the J neuron in the L layer. Correct? Now please look at this. So dependence of uh the output RCAP okay on ZLJ is through multiple variables that come in between RCAP and the Lith

**[26:54]** layer. Okay. So now if you have uh a composite function and the dependency of uh the output is through multiple uh variables and there are multiple paths parallel paths then multi- variable chain rule tells you that you'll have to add the partial derivatives along all those paths correct. So now let us do that. Now let me write that down. the derivative of

**[27:32]** the activation uh what uh index do I use? I, J, K, L, everything is taken. M huh? &gt;&gt; T is already taken. &gt;&gt; No, no. If if it's not used, no, we can

**[28:03]** take M. So this is uh m in l + 1 and this is uh del z m l + 1 uh z j l and this is summation is over m. You see this the what am I saying here? The

**[28:33]** dependence of RC cap okay on Z LJ is through every pre-activation that comes in the layer L + one. Does it make sense? Yeah. [clears throat] So then you have to add all those partial derivatives that come along the path. Is that okay? Okay. Now what is this equal to? &gt;&gt; One to the number of neurons in the L+

**[29:06]** first layer. Thanks. Yeah, I is already there for

**[29:39]** that in our notations. On a lighter note, when you're doing algebra like this, error in my brain gets propagated through. &gt;&gt; [laughter] &gt;&gt; Is it okay? Now we already know that this term the first term what is the first term? First term is del l + 1 &gt;&gt; i. Yeah, it's correct

**[30:10]** by definition. And to compute the second part, let's let's look at what the second part is. Consider what is this equal to?

**[30:36]** uh to do this we'll have to write Z I L + 1 is given by J W I J A J L + B I L + 1. So we wrote all this very carefully last

**[31:04]** night. So so that the indices should not be mistaken. Okay. Now this is equal to [snorts] w i j l + 1. What is aj? Aj is sigma because we need it in terms of z. So plus B I L + 1.

**[31:35]** This is Z I L + 1. [snorts] Okay. Now if I take the derivative of this with respect to Z I Z I at L. What will I get? W I J at L + 1 times the derivative of this sigma -ash time

**[32:03]** evaluated at Z LJ. Is this all right? Why is this the case? Because all the other terms here go to zero. You see that? Why is that the case? Because yeah there the dependence is only on one particular weight and everything else will go to zero. So we got this particular term as well. Okay. Now here was J. Was it J? Yeah, it is J.

**[32:38]** Thanks. Yeah, there's an idea called mixture of experts in LLMs or rather even in neural networks committee of experts. We will see when we look at boosting algorithms. Uh so this this class is like an MOE. If one makes a mistake, the other will point it out. That was a joke by the way. [laughter]

**[33:08]** Okay. So now we got this. What why did we do this? We'll have to substitute this here, right? So this implies are um so d L no d W I J L + 1

**[33:42]** So I put bracket there. Okay. Into See this is the most beautiful part of back propagation. Okay. Look at what's happening here. If you ignore this particular term. Okay. All we are saying is the error associated with the neuron the J neuron in layer L

**[34:13]** okay is the linear combination of the error associated with all the neurons in the L++ first layer and the linear combination is weighted by the weights that are connecting uh the layer L and L+1 you see that see if you look at uh the data flow from the input to the output. At every layer, the activations or the pre-activations

**[34:42]** are given by the weighted linear combinations of the activations in the previous layers. This is the forward pass. While you are computing the error, the errors also get linearly combinate combined. Okay? But in the reverse direction, that's why the name the errors are propagating in the backward direction. Right? and the the propagation is weighted by the same weights that that propagate the data from the first layer

**[35:11]** to the last layer. This that's why this equation is so elegant. You see the errors are getting propagated backwards because the error in the layer L okay is given as the linear combination of error in the layer L+1 weighted by the same weights that connect the neurons from the layer L to layer L+1. And of course there is this term just ignore it. I mean for the sake of interpretation. Does it make sense? Yeah. That's why the name error back

**[35:40]** propagation. Yeah, error gets propagated backwards

**[36:13]** uh by the weights and therefore the name back propagation. Okay. No, small L is capital L is only for the &gt;&gt; layer &gt;&gt; last layer. Everything else is small L. Oh, does this look like capital L? No, this is small L. That that's small L. Yeah, capital L is only for the last layer, right? This is for all layers. Once you basically the idea is you compute the the error at the last layer and then use the this

**[36:41]** recursive thing and back propagate it, right? So once you have this, so finally to compute. So finally we want rcap the derivative of this with respect to this thing right this is what we wanted now that's given by again use the chain rule the dependence

**[37:15]** of uh this is through uh the activations time del Z J L divid by del W J K L okay. Uh do you notice notationally I've

**[37:50]** been using this do for derivative and delta &gt;&gt; for the error. Do you notice that? &gt;&gt; Okay, I've been consistent, right? &gt;&gt; Yes. &gt;&gt; Okay, this is D L. And what is this? The derivative of uh that's that's actually pretty easy. What is that? That's a K L minus one. Please

**[38:17]** uh verify that it's pretty simple because this is given by sum over uh no a k is given by sum over all this right no z jl is given by uh the sum over this akl minus one and this multiplied by w jk and to take the derivative only that particular term will survive and that's why it is this okay now we know both of this do we know dl delta lj we know how to do it this is computed via

**[38:48]** computed via back prop. in fact computed via what is called as a forward propagation. That's it. So this completes

**[39:16]** backdrop. So we have the gradients that we need and then do a gradient descent. So summary is as follows, right? So let's say that we have a neural network that we build this way. Now I'll vectorize. This is X. We have the first layer W1. We have W2. We have W3. And we have Y here which is H state of X.

**[39:45]** Do you notice that uh the length of the lines that I have written are decreasing? So typically when we solve classification regression problems, the number of neurons progressively will keep decreasing. That's the typical choice. Okay? Because typically what happens is the dimensionality of the input is much lesser than sorry the output is much lesser than the input. Right? So you keep decreasing the dimensionality and show that your output layer matches uh the that of the the output uh variable.

**[40:16]** Now to get one gradient step okay so we'll write that to get one gradient step. So two things have to be done. So first you do what is called as a forward propagation. To get a

**[40:51]** or the output for that particular input. So once you get this what do you do? Do one step off. Is that all right? So to get one

**[41:27]** gradient step we have to do one forward pass with a given data point get the output compute the loss and then do one full backward pass okay to obtain the gradients or two this is for one step of the gradient descent. So basically the computation will get doubled when you are training. This is training. This is basically erm right erm on a neural network. One forward pass one backward pass is one gradient step. By the way

**[41:54]** when people uh implement gradient descent today all of them are vectorzed. Otherwise these summations see you see this right? The kind of operations that we have here are simply linear products inner I mean linear combinations. All linear combinations are vector multiplications. Okay. Now if you want to implement this as scalers then you'll have to write for loop over I that's too much computation instead of that you can implement all of these as vector vector

**[42:24]** inner products basically right so you vectorize it and that's how you implement them so these GPUs are specialized hardware that will do these kinds of operations which is vector products they're they're they're designed to uh do these kinds of operations that's why they are very efficient it. Okay, they're all matrix vector multiplications and uh and sums and products. That's it. Okay, so this is how back propagation is done. Every

**[42:51]** data point you take it one forward pass, compute the output and then back propagate the error using uh this particular equation here layer by layer. Then you compute the gradient and uh I mean everything else is the same. This is training a neural network and as I already told you when we do it when we use let's say stoastic gradient descent do it over batches and one complete pass through the data set

**[43:21]** that you are given is one apoch and multiple apoch of trainings are done typically even your uh you know the the latest LLMs are trained this way what changes is the architecture that's it and the scale of data but the essential training procedure Is this error back propagation? Okay, any questions on this? &gt;&gt; No.

**[43:55]** &gt;&gt; No, no, no. See, for every data point, see to do one gradient step, you'll have to invariably do one forward pass and one backward pass. No, if you don't do the forward pass, you can't compute those gradients at all. That's precisely what we saw today, right? To compute the gradient, you'll have to find those error terms. To find the error term, you'll have to find the error term in the subsequent layer. And to do that, you need the output. So take a data point, get the output, compute

**[44:22]** the error, and uh get the error of the previous layer, get the error of the previous layer and so on. And the derivative of the parameter with respect to the loss uh is dependent on this layer at the error at that particular layer through the equations that we saw. So to get the gradient you'll have to do one forward pass and one backward pass. Okay, is that all right? So this is the famous error back propagation algorithm uh which is the workhorse of uh all the

**[44:51]** the AI models that you see today. But as I said the scale of the data and the scale of the number of parameters is is humongously large. And by the way, when they say that they are training neural networks or training these AI models for 6 months over the data centers that have tens and thousands of hundreds and thousands of GPUs, all they are doing is running this error back propagation algorithm. Okay. So now if if if some some of you

**[45:20]** can actually come up with an algorithm that would reduce the computational complexity of this and can be done without matrix multiplications then the then the monopoly of some of the companies will be just gone. So I keep saying right some of these companies are one good algorithm away from bankruptcy. Their hardware will not be sold at all. Right? Everything is h I mean it there's so much demand for those hardware because all you are doing is

**[45:47]** doing vector multiplications and those hardares are are built specifically for vector multiplications. Right? So if there is a way to do erm on these kinds of composite functions without vector multiplications then of course you may get an phrase as [laughter] well. Okay, so this is about error back propagation. This is how you do erm. So next time when we come and by the way just to complete the story uh you can add regular raises to this loss

**[46:15]** function and nobody is stopping you. You see the so loss function plus let's say any kind of regularizer you want and the same thing happen same thing applies the only thing is the gradient will change to obtain the gradient with respect to the composite function which is neural network you need backrop for the gradient of the uh the the regularizer you don't need back prop you know you can just compute it and then add it only the gradient descent equation would change because gradient is a linear operator right if you add

**[46:42]** some one other term the gradient will just get added up so it doesn't matter that way you see that so you can try a neural network with regularizers as well explicit regularizers as well. So all the problems that I've given you in your assignment can be now solved using neural networks. Only thing is you'll have to code up backdrop you know I will ask you to code up backdrop as well as a part of your next assignment. Okay so that's it. Next class what we will do perhaps is uh I will talk about different architectures uh and you know

**[47:11]** how you can see architectural remember I told you that any kind of choice that we make on H theta can be viewed as bijian regularizers. I will show you that convolutional neural network is actually a regularized heavily regularized MLP. Yeah, we will see a couple of architectures maybe CNN, RNN and transformers and then we will move on to the other algorithms uh the ensemble methods and so on. Okay. Okay. Thank you. That's it for today's class.

**[47:38]** [music]
