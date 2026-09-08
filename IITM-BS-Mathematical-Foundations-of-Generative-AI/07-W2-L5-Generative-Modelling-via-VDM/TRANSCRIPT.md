# Transcript — W2_L5: Generative modelling via variational divergence minimization

> **Source:** https://www.youtube.com/watch?v=stZC0Zk5KYo  
> **Channel:** IIT Madras - B.S. Degree Programme  
> **Duration:** ~54 min  
> **Note:** Auto-captions cleaned lightly. Minor ASR errors possible.

---

**[00:11]** [Music] Hello, welcome all of you to this first tutorial in uh the DGM course deep generative model course. Now in this tutorials now uh now we shall be discussing predominantly on a simple MLP and how the forward pass will happen and how the back propagation using the error will happen. So a couple

**[00:39]** of prerequisites that I'm assuming is that so I'm assuming you guys know the the basic structure of MLP so or uh the fully connected network that is there and then I'm assuming that you are pretty much familiar with the task

**[01:10]** of regression. Okay. So where in which it's a form of a supervised learning task in which you will be going to predict a number. Okay. Now these are a couple of prerequisites uh that I'm assuming that you know for this specific uh tutorial session. Okay. So I'll give a brief introduction about uh the supervised learning as a whole. See in

**[01:38]** supervised learning you know what is the right answer that is the underlying uh part. So you'll so you have the data that the data will be split into two parts. Now one is the training data one is the training data. So this is the data now using which you try to understand the mapping between your uh input. Now the mapping between your

**[02:07]** input to your output. Okay. And supervised learning you know the right answers. So you want to understand this mapping. So now let's call this uh mapping as function f. This is what you have learned. Now how do you learn this? Now this is learned using the training data. Now how do you represent the training data? The data training data we represent it

**[02:36]** as D. So when which you have uh pairs of X I Yi. So when which XI is a feature and YI is your target label and then you have M such uh X Y pair. Okay. X I is the input feature and

**[03:06]** Yi is a target label given. Okay, target label. Okay. So depending on the size of the dimension now you can assume that X is in some real space and Y since we are currently working under the assumption that we are going ahead with the regression task. You can assume that that is also in some real space. Okay.

**[03:35]** Now what is the goal of this whole thing? Now the goal the overall goal is so we want to minimize you want to minimize some error. Okay is a goal of the overall supervised learning. You want to minimize some error using the training data. data. Okay. And uh how does it look

**[04:13]** like? So you have a neural network. Okay. I'm uh preferably considering an MLP here. You'll be given the input XI and then you'll be predicting Yi hat. Now x i is your input and this yi hat which is there now this is your prediction. Okay this is your

**[04:43]** prediction. So we can represent this uh yihat you can represent this yi hat or you use the function f on this data x i and then you get the prediction. So now how do you characterize the error? So the error or loss is called as

**[05:12]** loss. So you can say that this is some L loss L whichever between your actual target label Yi and the predicted label Yihat. Okay, this is the loss. Now which you want to minimize. Now this is the standard setting of your uh any supervised task. Now to understand this so with a very simple intuition. Now

**[05:41]** let's take a very nice example of football. Okay. Now you have a football field. So in which you are training. Okay. This is a football field that you have and uh so you're trying to do some kind of a penalty kick or something like that or hockey or whatever. Okay. Now this is your goalpost. What happens is you're training. This should be your actual

**[06:09]** trajectory. Now you should hit your uh ball here. This is your actual trajectory. But what is happening is uh now you are uh going off a bit. Okay, you're going off a bit and then you're hitting till here. So now this deviation which is there now this deviation which is there is actually the error. Now how do you so the aim is now

**[06:35]** to reduce this error. Now how do you reduce this error? Now you go on looking into practicing it. Now go on uh trying it using the training data. Now I'm I'm going switching back to the context of supervised learning that we have in hand. So you go on practicing and finally this error will reduce. Okay, this error will reduce. This will not become zero. This error that is there. Now this error will reduce actually. Okay, good. So now now given the input you go ahead and

**[07:08]** predict the answer of this process is actually called as the forward pass. This is called as the forward pass. And then you go ahead and then from the outcome you calculate the error and try to update things. And this is called as Now how does this happen? That is the

**[07:54]** overall idea of uh so this is forward pass is from input to output. Now you have your input and then it will convert it to output and then in your backward pass or back propagation. Now what do you do? You calculate error. It's from error to weights. You update the weights. Each model is uh having some weights you update it. Okay. So now uh

**[08:24]** for this this is the brief introduction. Now what we shall do is with this much of uh uh background in hand we shall take a very simple uh neural network structure or an MLP structure. Okay. And then we will try to understand what are the some of the intricacies that are there and how does this backward propagation back propagation or the forward pass it

**[08:54]** actually happens. So let's try to look at both of them. Okay. So now for this purpose what I shall do is I shall take my X I with two features. Now one is X1 the other is X2. My input has two features. Okay. Or you can say that I to represent that this is the I example you can have a subscript there. So now these are my

**[09:25]** input X I1 and X I 2. Okay. Now normally what will happen is we have something called as input nodes. These are called as uh input nodes. So these are we will not be having any node any weights per se. Now

**[09:55]** just the inputs will be given and then you'll be having some hidden nodes. I'm taking two hidden nodes in this uh I'm having one hidden layer. You can have multiple hidden layers like you have one hidden layer like that. You have multiple hidden layers like this. Okay. So and then uh here for a very

**[10:23]** simplistic uh approach. I'm just taking one hidden layer with two nodes. Okay. Two hidden nodes in the hidden layer. Okay. and then I'll have a simple output model. So this is my output model. Okay, this is the the structure which is there. Now let's uh put in some weights. This is a fully connected

**[10:54]** network or metal layer perceptron. So this both the inputs everything from the previous layer will be connecting to everything in the next. Now that is the overall idea. Okay. Okay. This will be there. Then so let's make it more colorful. And then finally we will have

**[11:25]** an output here output will be yi hat. Okay, good. So now let's try to put some names to this. Now we haven't done any namings for this. So now this I shall call that I'll use the

**[12:03]** same coloring convention so that it is comfortable. Now this is a in the first layer first node. This is a in the first layer second node. Okay. So now all these nodes I'm calling it as a. Okay. A is this is let's say this is I J I have now this is

**[12:38]** the the layer number. This is the node number. Okay. And similarly for the remaining things also this is a second layer first node. This is a second layer second node. And this will be a third layer first node. I have named each of those node. I'm using a

**[13:06]** just to specify that it is an activation. Okay. That reason I'm using the word a. So now let's uh try to put on uh the names for the weights. How do how do I represent how do I name these weights? Okay. Now I use W as weight. Okay. Now from which node to which node it is connecting? Okay. Now you can say

**[13:38]** that from the first node it is connect from the first node of the first layer it is connecting to the first node of the second layer. Now what I shall do is I'll call this 1 comma 1 okay of layer 1. So let's try to understand these indices. So now my notation will be W I comma J comma K. Now this K that is there. Now this

**[14:09]** will be the the which layer the I can call it as the previous layer. Now this I which is there I call it as the destination node and this J which is there we call it as the source node. Okay. Now because of which uh so this

**[14:38]** will be and this will be W. It is going to the second node from the first node of the first layer. Okay. And this will be W from the to the first node from the second node of the first layer. Now this will be to the second node from the second node of the first

**[15:07]** layer. Okay. And similarly for this W. So the layer will be two. Okay. This will be from to the first node from the first node. This will be to the first node from the second node. Now similarly you can extrapolate uh these things. Okay. I presume that these uh uh

**[15:37]** nomenclatures are pretty much clear. Now these are the standard uh uh nomenclatures that are being used. Okay. So now now how do you compute these uh values of a now I have these w there will always be biases also it will be there. Okay. Yeah I'll do it when I go ahead with writing the activation functions. I'll name them. Okay. So now

**[16:06]** let's take uh how do I compute this a 21 is it a 2121? Yes. How is this A21 computed? Now for A21 that is this outcome you have this and this you have this X I1 now which is the same as A11. Now since it's the input

**[16:35]** node this value and this value will be same this the first feature now it will be multiplied by its respective weight. Okay. Now we can use it X I1. So the convention is first the weight will come W1A1 1 into X I1

**[17:07]** + W 1A 2A 1 into X I2 plus I need some bias right now I should be indicating that bias as well. Now how do I indicate that bias? Now I'll give it the term B. Now I should be representing the node and other stuff. Right? Now what I'll do is now this is at layer one the first node. Okay. I

**[17:37]** shall be using the same convention. This is an additive bias that I shall be using. Now to this we need an activation function. Now uh let's say that I'm using the activation function G. So we all know that the famous couple of famous activation functions that we have are one is sigmoid activation function sigmoid when

**[18:04]** which the input is x okay or okay let's not confuse x we have already used for input let's uh consider it as some z if z is my input it is 1 / 1 + e ^ of minus Z. This is sigmoid. And then we have RLU rectified linear unit. Now here G of Z. Now what will be the value? This will be

**[18:32]** zero if Z is less than or equal to zero or it will be Z if Z is greater than zero. Okay. Now why do we use these activation functions? These activation functions are specifically used for smoothness or to model more complex relations between input and output. Now we normally use this activation function. Okay. Now similarly you have

**[19:00]** tan h you have many such uh activation functions. Okay. We'll not be going in detail with that. Okay. So now this is uh how you calculate this. Similarly you calculate uh uh this. Okay. So now normally what do we do is this summation which is there now we

**[19:28]** normally call it as zed. Okay preferably small zed. Okay fine no problem. Zed which is there. Now this zed uh you want to name them the same indices of uh a we use it. Okay, this is this is one more representation that is needed for us uh going ahead and similarly let's compute for

**[20:02]** uh a22. So now a22 will be so again uh so you can use the same activation function or you can use a different activation function but people prefer to use the same activation function throughout. Okay that is overall idea. Now g of w from layer 1 we are connecting to the second layer from the first

**[20:30]** one x i1 plus w22 from the first one into x i2 + b12. Okay, this is the one. And then finally you want to compute the outcome. So now uh

**[20:59]** a13 how do you compute this is w2 1 1 into a 21. Okay, this into this outcome. This into this plus this into this. Okay. Plus W from the second layer 1 22

**[21:30]** into A22. And then you need an additive uh value in the second layer. The first node this you pass it through an activation function. Now since this is the last uh node that we have this is what we represent it as uh y I hat this is our predicted value. Okay this is our predicted value. Okay now this is

**[22:01]** the overall process of forward propagation. Now this is if you have multiple layers accordingly you go ahead. So I presume that uh this is clear. Okay. So now then what do we do is now here uh it is evident that this values are fixed. Now why are

**[22:28]** these fixed? Now this these are nothing but input. So these are fixed. Okay. So now if you want to choose uh uh if you want to reduce the error now these weights that are there now all these weights are the ones that needs to be modulated. Okay. Now this depends on this. This depends on this this which is internally depend on

**[22:57]** this. You cannot change this input. So you can only change the weights of it. Now similarly you can extrapolate it to everywhere. So now what do we do is now all the parameters that are there now we put it as a list okay we call it as theta okay so where in which we put all the parameters okay so all the all the parameters of

**[23:29]** the network that we have because of which we represent y i hat equals the function f which is having the parameters theta x. Okay. Now this function can be uh anything. Now this function can be a linear regression. This function can be a polinomial regression. It can be a neural network. Now depending on what is

**[23:56]** your choice you go ahead. Okay. Now this is the this is the forward pass. Okay, this is the forward pass that we have. Okay, so now now the next step now once you have done the forward Okay. So

**[24:37]** now see the idea is you need to reduce the error using the training data. Now that is pretty much clear for that. Okay. So now what is the error? It depends on the task that you are going ahead. Now if it is a classification task uh that you are going ahead we normally use CC loss. Now if you're going ahead with some uh reconstruction task now that is what we shall be looking into in most of our generative cases. Now we'll be looking into the MSSE laws or the reconstruction

**[25:05]** laws that we have. We have KL divergence. Okay. So on and so forth we have lot of loss loss functions depending on uh the structure or what is the aim that we are trying to achieve. Okay. So now in this current case now we shall take a very simple loss. Okay. Now I'll call this L. Uh we will be using the mean squared error loss famously called as MSE uh mean square

**[25:46]** error that we have now we should be using this. Now how do I have this this loss for one specific instance? Now we are only taking one data point. Now if you're having multiple data points, you just accumulate and take out the average of the thing. Okay. Now for that specific uh instance that we have 1 /2 yi now yi is your actual target label minus whatever

**[26:15]** uh uh the prediction that you have got. Okay. F theta of X or this is same as into Yi minus Yi hat your prediction also. This is our MSSE laws that is there. Okay. So now we want to reduce this specific MSE loss that we have in hand. Okay. So I

**[26:46]** presume that that is clear. So now I want to just uh for completeness sake so I want to reiterate that uh we are If you have multiple training points now

**[27:16]** what you will you'll accumulate all the error you take a sum of it and then you divide it by number of points that you have considered now which will give you the average of okay now you're assuming one thing and then the error that is there this is uh this is what we want to minimize this is the error that we want

**[27:43]** to minimize Now how do we minimize this? Okay, standard trick that is known to all of us that is we we minimize a function by computing its derivative with respect to what we can optimize. Okay, now you cannot modify or change you cannot modify or change the input. The only thing that we can optimize is the parameters that are involved. So finally

**[28:11]** so we want to minimize this by computing the derivatives with respect to the parameters. So we change. Now this is uh famously written

**[29:07]** written as a optimization function. I shall not be I'm not assuming that so you are familiar with the optimization. Okay. So I'll not be going ahead and writing any optimization equations. Okay. So now we can only modify theta. Now that two I take a specific uh instance of it. Now let's take that uh so we taking a specific

**[29:35]** parameter wig J K that is at the K layer. Okay. The previous layer is K layer. The next layer is K. That's one layer. Now from the J node of the K layer I'm connecting to the I node of the K + 1 layer. The weight is what I'm considering. I have the loss L. Now I compute its gradient do L by D. So I want to find this gradient. Once

**[30:07]** I find this gradient I'll be using the gradient descent which all of us are uh actually aware of. So then we perform gradient descent. How does this gradient descent

**[30:34]** will work? the weight is updated via you take the previous weight and then you use a learning rate into the the gradient or the derivative that we have computed. Okay. Now this is the standard uh uh back propagation that we do using gradient descent. This alpha which is

**[31:03]** there now this is uh what is called as the learning rate. Now this alpha that we have here this is learning rate we need lot of uh uh uh knack in choosing this learning rate that we shall learn in our uh coming tutorials. So learning rate this will this belongs to so the positive r okay this is positive real number okay now we shall

**[31:33]** be going ahead with the what we shall be doing is back propagation we shall be doing with gradient descent Okay, this is what we shall be doing. Okay, so now what does this uh derivative actually say? What is the intuitive uh idea behind

**[32:03]** this uh derivative? So now if you if you increase this uh uh parameter that you have considered. Okay, if you change this now how does it affects the loss? Okay, now that is what we want to actually find. Okay, so JK. How

**[32:47]** much does No change that is what uh we are looking at that is rate of change of loss with respect to weight. Okay. So now now what we shall do is now to understand uh uh the back propagation in much more detail. Now let's take one specific uh weight uh in mind. Okay. Now let's that

**[33:20]** reason now let's consider the weight uh w1 1 comma 1. Okay. Now let's consider this weight. Let's consider uh this Okay. So now what is that we want to

**[33:53]** find now? Now we want to see that how the loss will get affected as I change this uh weight. Okay. Now in that the first thing that we should uh need to do is now we should know what is the path uh this W11 is being involved. Now as you can see here this is the path in which W11 is involved. Okay let's highlight that path. So just for

**[34:23]** uh okay this is too dark. This is okay this is too light. I shall do is this is 2 mm. Okay. This is the path in which uh it is going. This is the first thing. Now first we should look at now what is the path uh that is uh uh involved. Now let's uh handle this.

**[34:54]** Once we have the path from this uh weight under consideration to the output then we can look at it. So now now it is uh I I am assuming that all of you are again uh know the basic idea of uh chain rule here. this output which is there is affected by this value of a13. Okay. Now

**[35:25]** in this path a13 is affected by this weight. Okay. And this weight is uh in the chain and this this okay there's a chain that is happening now that now we should be taking into consideration okay now let's write it down how does it propagate so the loss that is

**[35:56]** there is uh directly dependent on the output. Okay, which is same as A31. Okay, this is the first step. Okay, so now the loss which is there is directly dependent on this A31. Okay, now how is this A31? Now this A31 which I have is related to this.

**[36:26]** Okay. So now that means that I'll A31 Z 31. into now this Z 31 now which is this

**[36:53]** value. No, which is there in the path. Okay, it is dependent on A21. Okay, a22 will not come in the path. That is this Z31 which is there it is dependent on A 21. Okay. And then the A21 which we have computed this A21 is dependent on this

**[37:26]** Z21 and this Z21 is dependent on this W11. Okay. Now you should actually trace back the path. This is uh a 2121 which is there is dependent on this Z 21 and this Z21 which is there is dependent on this weight

**[37:56]** W11. Okay. Now this is the full back propagation calculation now using the chain rule now which we are able to do. Now I have premp computed uh all these values. I have just uh taken the same equations and uh written them again. And then what I have done is I have taken a very simple feature input now which is one and zero. Now this is your input right and I'm considering all weights and

**[38:25]** biases that I have is one. And then if I go ahead with this computation now a to 1 this will be the computation. Now your weight into this one plus weight into zero you get this value. Similarly now I have computed my output. Okay. And this is my loss. So now now let's

**[38:58]** uh go back to our laws uh gradient uh equation. Now let me copy this. I have written it in a very way. Copy this. Okay. And let me paste it here. This is the same equation. No. Okay. So now one thing that I can change here is this is

**[39:28]** uh this is nothing but by I had. So what I can do is I can replace this okay for all practical purposes I can replace this with yi hat. Okay. So now let's uh go ahead and compute the gradients. The first thing that I want to compute is how the loss will change with respect to my

**[39:58]** yihat. Okay. Now I just want to write the loss computation. L=/ into yi - yihat. Okay. If I take the derivative of this L with respect to this Yi. Now this will be oh there is a square here missing. Yeah. Now this will be yi

**[40:28]** minus yi hat. Okay. Now you can this is like uh d by dx of x² is 2x. Okay. So you get two into this term. This two and this two will get cancel. So you get the term. Okay. Okay, I'm assuming that you know all those basic uh uh rules of uh differential calculus. Yeah, I have computed this

**[40:56]** now. So now the next thing that I want to compute is now how is my yi hat I have is dependent on this Z31. Okay, just to give you a reminder how does this yihat is zigmoid of this Z3

**[41:27]** correct? Now, now you should be taking the derivative here. Now it will be the derivative will be this is sigmoid of P1 into 1 - of this is same

**[41:55]** as yihat into 1us yihat Okay, I have computed two terms. Now this term has been computed. This term has been computed. So now now let's take uh how my Z31 is dependent on a 21. Okay. To understand that first

**[42:28]** let us compute how is uh Z31 equals second layer from 1 to Um

**[43:02]** okay go there. Let's go back to equations and then compute W into plus there is a bias term B this is a term. So now when you

**[43:28]** compute when you take the derivative of a to1 okay it will be the weight this will okay now you have computed this term this ter and this term how is this related so now you should be calculating a1 how is is dependent on

**[43:59]** Z21. Okay. Now what is this uh A21 is sigmoid of Z 21 that means that this will be change the color be uniform in terms of color. This 1us. This is same

**[44:39]** as a 1 into 1us a21. Okay. So now the next term. So this term we are done. This one we are done. This round we are done. This round we are done. How is this Z 21 dependent on W11 is the next question. Okay. Now how is the Z21 dependent on my W 1 one. Okay. This

**[45:12]** is my next question. Now how does it happen? Now let's compute it. Here Z to 1 = W11 into X I1 + W 1 2 1 into X I2 +

**[45:46]** D11. Okay. because of which this will be oops I okay so now we have effectively calculated the all of them so now now let's put all of them together into one equation this the laws this will be let's calculate each

**[46:15]** of those take each of the term So yi - yihat this is the first term this term for 10 to this okay this this okay if we want we can label it this is one this is two this is

**[46:45]** three This is four. This is five. Okay. 3 4 5 just to multiply each of them. Okay. So this into the second term

**[47:17]** is yihat into 1 - yihat into the third term as w12 into a21 into 1 - a21 into X. Okay. Now this will be the gradient of the whole thing. Now you just need to

**[47:50]** compute this particular gradient. Now similarly uh once you have computed the gradient then what do you do? You apply this uh updation method. So now because of which your weight will get updated. Now this you do it for all the weights that are there in your whole set of theta. um you do it because of which your weight updation know will happen. Okay. So now how does the overall now how does overall training

**[48:18]** procedure look like? How does the overall training procedure look like? Let's shed some light on procedure. So what we do is rather than taking each example at a time now we take a batch of

**[48:46]** them the overall data. Now that we have m examples now what do you do? you divide them into uh examples of B batches. Okay. So now let's take that I have a I have a batch size of B and then you have M examples in each batch you'll have B examples let's take say say that I have some K batches now my overall

**[49:16]** data now will be a set like D1 till D2 okay with which each bi Right now we'll be having B examples. Now I'm assuming that M is divisible by P. Now if not in the last batch there will be some small number of examples. Fine no problem. So now what do we do is now and then this is one right now we want to decide on how many times you want to

**[49:45]** iterate over the whole data iterate over the whole data. Okay, that is called as EO. Let's say that I have decided that I want to go ahead and iterate over the whole data max if okay I want to iterate over the whole data max of okay so now then what will happen is for E I'm assuming that you know Python

**[50:16]** and I'm writing it in a Python way okay E in range EOP. Okay. And then for DI in D. Okay, you take the one DI. Each DI will have a bunch of X's and Y. Okay.

**[50:49]** Now all the x put together I call it as capital x and capital y where x is your input and y is your feature. Now what do you do now? You calculate yhat which is nothing but you take this function and then pass this x. Okay, you get this and then you compute loss

**[51:17]** of y comma y hat. Okay, you compute this. This is what is called as the forward pass and this Okay. And then the third step, I'll just

**[51:45]** write it in uh bits. The third step is you obtain the gradients. Obtain the gradients. Okay. Now once you have obtained the gradient, so then what do you do? You update the weights. Okay. Now this is the standard procedure

**[52:14]** that we shall be following in most of the cases. There will be some mix and match here and there but overall structure remains the same. Now now you'll be running over this whole data. Okay. This is one time you will run over the data. Now you finish off uh this k times the k times this will happen. So the number of times in which the weights are updated that is called as iteration and how many number of times

**[52:42]** you want to iterate over the whole data that is called as maximum epochs. Unless specified this is the standard training procedure that is used and then you use the test data to calculate the predictions and then you measure it with whatever metric that you have in hand or have in as per the choice of the problem that you have. Okay. This is the overall idea of the training procedure the

**[53:11]** forward bus. So what did we discuss in today's tutorials? Today's tutorial was predominantly on the structure of simple MLP that we have considered. Okay, we considered the idea of uh mapping. Okay, and then the idea of error and then we took a very simple uh network into consideration and then we understood the components that are there in this network and how do you put their names and then we understood the forward pass.

**[53:40]** Okay. And then how do you compute the error? And then we understood the idea of gradient descent. And how do you apply the chain rule in obtaining the gradients and once you have and then this is the forward pass with a simple example. And once you have that now how do you compute the gradients? Each of one of them we took uh separately and then we computed the gradients and then we computed the total gradient for one

**[54:09]** weight. So the same thing will be repeated for all the weeks and then uh you update it once and then we looked at the overall training procedure of uh the supervised learning in most unless specified. Now there'll be some here and there some small modifications will be there apart from that the overall structure almost. Okay. Now this is what we have discussed today. Now thank you for patiently listening to me in the next session. So we shall be in the next

**[54:38]** tutorials we shall be looking into the pyarch. Thank you.
