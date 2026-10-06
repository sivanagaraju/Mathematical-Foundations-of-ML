# Transcript — Tutorial 12 : Pytorch - Building MLP and Auto Grad

> **Source:** https://www.youtube.com/watch?v=SEEQ2A2WN9g  
> **Channel:** NPTEL - Indian Institute of Science, Bengaluru  
> **Duration:** ~42 min  
> **Note:** Auto-captions cleaned lightly. Minor ASR errors possible.

---

**[00:03]** Hello all, welcome to this uh second part of our uh PyTorch tutorial. Now earlier uh part of our PyTorch tutorial in which we understood the idea of tensors and what are the attributes of the tensors, how do you create tensors, what are the operations that you can perform on the tensor and uh we understood the loading of data sets. Okay, so now we have loaded the data set. Now let's try to write the code for a neural network. Okay. Now here in this

**[00:35]** particular tutorial, now I'll be restricting myself to the MLP. Okay. Now in the coming tutorials, we will be looking at uh CNN's but in this specific tutorial uh my concentration is on creating an MLP. The structure of the MLP that I have uh uh considered is like this. Uh see my data set that I have in

**[01:05]** consideration is MNEST. Okay. The data set under consideration is MNEST. So now each class so now our Y now belongs to 0 to 9. Okay. Now that means that the output now which is there should be a probability vector now of length 10. Okay. Now now how do we do it? Now we

**[01:35]** pass it through a softmax function. All of you know that. But in terms of the structure of MLP. Now one thing is uh sure that my output now this which I have now this belongs to this is a vector in R10. Okay but this is not a probability vector. Okay I have to convert this into a probability vector. Now how do we do

**[02:04]** it now? We pass it through a we pass it through a soft max function and then comes a probability vector let's say P now where in which you have P 0 till P9 okay now this will be the vector that

**[02:32]** you get and then what does this indicate now this is the probability that y =0 given the input x. That is what this term indicate and this is probability y = 9 given x is what it means. And then when you sum up all these elements now you should get one. Okay. And each of them should be uh uh what do you call non- negative. Now all those properties should be

**[03:00]** there. But my whole point is the output now should be having 10 nodes here. Okay. Now since I'm going ahead with an MLP. Okay. Now how should my input look like? Now the input for this MNEST is 28 + 28. Okay. This is a matrix. Okay. It's 28 + 28 matrix. So now

**[03:29]** now since we want to use an MLP now if I want to use an MLP now this I have to convert it I perform a raster scan and then convert it into vector which is 784 + 1 28 + 28 okay if you multiply it is 784 + 1 I convert it into vector now therefore now what happens is my input okay my input will be a So this actually belongs to R 784. Okay.

**[04:03]** Now then then you have to make choices. Now how many hidden layers uh now I should have. Okay. So and then uh what should be their dimensions. Okay. Now all those things. Now I will make a very uh simple case now where in which I have two hidden nodes. Now each of them belongs to R 512. Okay, the output belongs to R 512. Now

**[04:34]** that means that I have 512 nodes here. The output of this layer will be a vector in R 512. Now that's what I mean by this thing. Now, now that means that now I have this is one matrix. This is another matrix. This is another matrix. Now let's call this as w1, w2

**[05:02]** and w3. Now this uh will be okay. This belongs to r 784 cross 512. Okay. Now there will be bias term for each of those neurons. I'm not uh going ahead and considering it. Okay. Now just for ease of uh computation now this is R 512 + 512 and this belongs to R 512 + 10. Okay. Now this is W1 W2 and W3

**[05:37]** matrix. Okay. Now how will the the forward propagation happens in this case? Now you have an x now which is uh 784 + 1. Okay. Now you have this uh what we will have? we will have uh w1 transpose x to that you pass through a nonlinearity

**[06:18]** okay that you have w2 transpose that you again pass through one more nonlinearity and then that you multiply again And you have W3 transpose. This you pass through the soft max. Okay, this is how it will be. Now let's try to get this. Now this what is the size of this? Now this is W1

**[06:46]** transpose. Now W1 is 784 512. Now this is now 512 + 784 this into 784 + 1. Therefore this whole thing will be 512 + 1. Okay. And then you apply element wise uh uh nonlinearity. Okay.

**[07:16]** Now whatever nonlinearity that you want sigmoid relu or whatever. Okay. And then uh you go to the next one. The next one is uh 512 + 512. Now you take the and this is 512 + 1. Therefore the output that you get is 512 + 1. This is the once you apply this you get a 512 length

**[07:49]** vector and then you apply this matrix again you get a 512 length vector. Okay. And then you pass it through this the third one. Now this is if you take the transpose it is 10 + 512 into 512 + 1. Now this will be a 10 + 1 vector and then you pass it through the soft max. Now once you pass it through

**[08:15]** the soft max you get this probability vector where which this is p 0 till p9. Now please excuse me for uh using the abuse of notation. Now whenever I mean p 0 here now what do I mean by this is p y is equal to0 conditioned on the given x. Okay. Now that is what I mean by this. This is the probability vector that you get the post passing through the softmax function. Okay. So this is the full

**[08:46]** structure of the the MLP that I'm considering. Okay. Now first what am I supposed to do? Now first I should have I have an image. First I should flatten it up and then perform the series of uh uh matrix multiplications and then get a vector. Now the output of this point is what is known as a logit. Okay. Now you have this and then you pass it through a softmax method. The output is what is

**[09:13]** known as a probability vector or it is also known as posterior which totally makes sense. Okay. This is a posterior. Okay. So now now this is the structure of the network that I'm considering. Now let's try to code it up. Okay. And then we'll be doing it. So on a device, we'll be doing it on a GPU device. Okay. So now let's do it. Now

**[09:55]** the first thing that I should do is first I should uh import all these methods. Now okay before that runtime runtime. Now here go to this runtime and then you can see change runtime here. Now please select the GPU. Okay. Now then the current runtime will be disconnected. That means that whatever you have stored

**[10:22]** in your working directory will be erased. Now we have to rerun the data loader. Okay. Please remember that. Okay. So this is for collab. Okay. Now that is not the same case whenever you are having a physical uh device with you. Okay. Now I have enabled the the koda device and then the same thing is uh uh replicated here at the the bottom right corner. You're using a T4GP. Okay.

**[10:51]** So now you need to what are the uh things that you want? You want OS. Okay. Now import torch is needed. Now from torch import nn stands for neural network. Okay you use this n. So and then uh so

**[11:22]** these are the three things that are needed. Okay. I'll run this. Okay. Now then I have to use the device. Now I have to use the now CUDA if CUDA is available. Now this is uh the boolean outcome. Now if it is available then the CUDA will be considered. If it is not available now we will consider the CPU. Now I'll rerun it. Now using the CUDA device. Okay. That means that as of now I'm using the

**[11:51]** the GPU which is there. Now I have to rerun this uh whole data loader. No because uh it would have erased the previous case. So I'm just rerunning it so that I have my data loader ready with me. Okay. So now let's go ahead. Okay. Now I I have to create the neural network. Now you have to create a class for this. Okay. Now class let's say

**[12:22]** neural network. Okay. Now that has to be a child of this NN domodule. Okay. Now because whenever you are creating what do you mean by creation of neural network? That means that you're having couple of weights and biases. Correct. Now that means that those weights and biases has to be initialized and then for those weights and biases you have to perform the back propagation. So you have to update the weights. Now all these methods you don't need to explicitly

**[12:50]** code them. Now these are already present in this nn.m domodule. You just need to inherit it directly. Okay. Therefore we have to create uh it as a child of n.module. Okay. And then we have to define the the init method. Okay. Now yeah if you write it will start autofilling things. So you use the super that is uh you're calling the init

**[13:19]** method. Calling the super that means you're trying to access the parent. Now you're trying to call in the init method. Now I have to pass the name of the the class. The first thing that you should do is now first you should flatten. Now that means that you have to convert your matrix which is which is a 28 + 28 into a vector. Now that is what you're doing. self do.flatten n.t flatten. Here you're just initializing things. You have not come to the thing. Okay. Now then then then comes the important thing.

**[13:48]** Now if we understand from here now post flattening now once I give the input. Now this you can consider it as one operation. No you can consider this whole thing as one operation. I give the input and then I get the output. Think of it like that. The because the the input is here. The output of this operation will be given as input to this. The output of this operation will be given as input to this. Now I can think of this as one whole conveyor belt kind of a thing. I can think of it as a

**[14:18]** stack of operations. Okay. Now this uh neural networks allows us to create stacks like this. Okay. Whenever you have the series operations that means that there is no branching condition branching that is involved. Now you can always have this uh uh series operations put together. That is what is known as stacking. Okay. That is done now using this idea known as n.sequential. Okay. Now let's create that. Now self dot

**[14:49]** so linear relu stack. Okay this is the variable name that I'm using. Now linear uh is I have all these mlps. Now relu is the nonlinearity that I'm using is the relu. Now you can use other non nonlinearities like sigmoid or tanh. Now sigma and tan h are preferably now these are all empirical ideas. Now whenever you're doing image work im working on uh images preferably we use ray loop okay and

**[15:19]** whenever we are dealing with sequences that is when we use sigmoid and tanh okay now there is no proof for it these are empirically evolved ideas okay so n dot sequential okay now the first thing that you should uh declare is post flattening. Now you have

**[15:44]** to declare this. Okay. Now this is a linear layer. Nlinear the input is 28 + 28 which is 784 cross 512. Now that means that you have created this matrix. Okay. And then after this what is that you have supposed to do? After the application you have to use a relu. Now which makes total sense the output of this suppose this transformation you have to apply the relu here that is what this n.trelu means now then you have

**[16:16]** again nn.linear and then you have relu and then 512 + 10. Okay. Now you have initialized all these three matrices W1, W2 and W3. Okay. Now please don't confuse with see it has two hidden layers. Okay. Now but the way in which we are initializing is with respect to

**[16:45]** this matrices. Each matrix is initialized with one n.linear. Now please remember that. Okay. So now and then the nonlinearity that you're using is n.reu the relu nonrearity. Now there is no hard and fast rule that whatever nonlinearity you have used here the same one you have to use here. Now you can always mix and match. See all as sir told no these are Lego blocks. Now think of it think of them like that. These are Lego blocks. Now you can mix and match them but the

**[17:12]** compatibility with respect to the size should be there. Okay that's all is the requirement. Okay. Now you have declared this. Now your init method is done. Now you have to define the forward process. Okay. So now in the forward the forward process has to be defined that is the forward pass. Okay. Now but you don't need to specify the backdrop. No it has what is known as uh it will create what is known as a directed asyclic graph.

**[17:39]** Now which we shall look next in the same in this tutorial. Okay. When which things are taken care so you don't need to worry about it. Okay. So now now you have to create this forward method exactly with the same name where the instance is the first argument by default it is there and then the input which is there is a second argument. Now first you flatten it. Okay. Now then you get logit

**[18:07]** equals self.linear stack array loop. Now if you understand this first you get an input first you flatten it up in this place and then you give it here this whole thing is one you'll pass this and then you get the logics okay that is what is happening now this is the forward method that you have to have okay so now you have uh defined the class and then now how do you create an

**[18:37]** instance you have to create a model some variable to store So now that you'll use neural network and then you have to port it. See whenever you create an instance it will be there on the CPU. Now you have to port it to the GPU. How do you do it? Two device. Okay. Now you have it. Now device is CUDA device. Now you'll be

**[19:06]** porting it onto the GPU. Okay. So now whenever you perform any operation now the model is on GPU. Now all your data should be on GPU itself to perform the operations and the output will also be on the GPU. Okay. Now then you can print the model here. Here you can see that the input is 584

**[19:35]** 784 and then 512. Now bias is true. That means that for each of the output neuron of 512 you'll have a bias. So in total the bias will be 512 512 10. Okay. 1024 and 1034. Okay. That will be the number of biases and the number of weights you can compute. Okay. Now that is 784 into 512 + 512 into 512 + 512 into 10 + 1034. That is the number of weights that you have to optimize here. Okay.

**[20:06]** So now that means that in such a huge dimension plus one that is the loss curve and then you have to perform your uh empirical risk minimization over the loss surface in such a huge dimension. Okay. So now let's take an input. Let's take x. I'll create a dummy input. Okay. Toss dot random. I can specify the size 1 comma 28 28. Okay.

**[20:39]** Now and then what is the device? The device that I want to put it here. Device is the argument. Now for that I'm putting the value which is the same name, same variable has been obtained. So logit equals model of x. Okay. So then let's print this

**[21:07]** logits. Okay. Now whenever I say model of X, now the forward method will be called. So your forward method will be applied. Now this 28 + 28 will be first converted into a vector of say 784 and then it will be passed through this whole thing. Now you can see this. Now this is the logs that you have got. Okay. Now this logit switch which is there has

**[21:35]** to be converted into the uh probability vector. Now spread probability. Now how do I do it? I use the softmax function. Now how do I do it? nn dot softmax. Okay. Now dimension one uh that means across the column you get it. Okay. Now and then

**[22:06]** you print this prediction probability that you have okay now you can see this this is randomly initialized. So the only thing that I'm trying to say is now the propagation the forward propagation is happening without any mismatch in terms of size. Okay. Learning is not happening as of now. I'm just saying that the forward propagation which is there is happening without any issues of

**[22:33]** mismatch. Okay. Now this is my probability vector. Now how do I choose the class? How do I make the decision? Now I have to take the arg max of this. Okay. The index where which the maximum value is there. that is the value my y prediction equals the arg max the first arg max of this okay and then I'll be printing the predictor class is two now as you can see it is on the device now because model is on the device input is on

**[23:02]** device everything is happening on the device now please remember that okay fine now is this clear any any questions on Oh. Oh, sorry. It's not question. Okay. I thought this is what happens. Normally we teach in the class and then we say that any questions. Oh, I actually forgot that I'm live recording this in front of a system. Okay. So, yeah, we can uh actually print these

**[23:34]** things. Okay. Uh obtain some values. Wait a minute. Okay. So now let's see how the ReLU is working. Okay. Just to show you that how the ReLU works. Okay. Now I just have uh just create layer one which is uh NN dot

**[24:06]** linear. Okay. uh in features is 28 and then uh have some have to first flatten it up, right? Okay. So, wait. Okay. and then dot flatten off the x that I

**[24:40]** have. Okay. then run one. Okay. There's some error. Flatten equals n.flatten. Flatten

**[25:11]** just Oh, again. Okay. What is the size of this? X

**[26:07]** Let's do one thing at a time. Let's first flatten it up. Okay. So now flatten equals n.flatten. Okay. It's not on the device. Yeah.

**[27:00]** This layer one has to be ported onto device. That's all. These are the common mistakes that happen. Okay. Yeah. So see whenever something is created as I told you it will be on uh the CPU. You have to explicitly port it. Now you can see that now this is what happens. Now you get a vector like this. Now what will happen when I apply the ReLU? Okay.

**[27:30]** So now now C equals NN.relu of hidden one. Okay. Now as you can see here now ReLU is max of uh 0 comma each element. Okay. Now that means that all the negative values which are there will be converted to zero and you can see it here. This is how the

**[27:59]** ReLU operation actually happens. Okay. So uh yeah parameters. Yeah we saw it here. Good. Okay. Now this is how actually a neural network works. Now till now whatever we have seen is just the the forward propagation. See we haven't seen yet uh the back propagation. Now we'll be just seeing it

**[28:29]** as a a sample case for one operation and then that we can extend it for the the whole MLP. Okay. Just to uh understand how does this work. Okay. So now what uh we shall do is now we shall uh create a very small operation to understand the whole thing. Okay.

**[29:01]** Now let's say that I have my input X. Okay. this input x I'm multiplying that with this w and then post multiplication I'm adding this d okay this is what I'm calling it as z

**[29:32]** and then I'll apply loss function. Now let's say that I'll apply the cross entropy over the logits. Okay. Now for that cross entropy to apply I have to supply Y to this will give the the loss. Correct? Now what is happening is now I have uh wx + b

**[30:06]** is your output. This is your zed. Okay. And then what you are trying to do is know we are compute the c loss with z and y. Now this is the operation that we are performing and then it is uh very much evident from here. Now these are the these are the parameters that we want to learn.

**[30:38]** Okay. So now let's try to understand for this simple case. Now how does the the back propagation works? how does the gradients are obtained for this particular simple case and then now this is nothing but for one neuron and then you can extend it up for the further cases okay I'm just trying to understand with one specific cell okay so Now

**[31:20]** this is uh we are dealing with the automatic differentiation. So now Now the go-to algorithm is back propagation. Now as of now unless someone comes up with one good algorithm which can actually replace back propagation till that point we'll be

**[31:47]** using the back propagation. Now let's take X. Now X let's say that it's uh torch dot once of size five. Okay. This is our uh input tensor. Okay. And what will be my y? My y will be torch dot zeros of size three. This is the

**[32:19]** the expected uh output. And then we have w. Now w now has to transform from five to three. Right? So it has to be a matrix of size no 5 + 3 okay and then you take wrpose x okay and then you have to obtain the gradients. So now we use this requires grad is equal to true now which will take care of uh what is known as a computational graph. Okay

**[32:50]** now this is a built-in differentiation engine that is used in what is known as autograd. Now it supports supports the automatic computations of gradients by any of these computational graphs. Okay. So now w and b you have and then what will be the value of zed? Zed will be uh the multiplication of x and uh w plus b. And then what will be the loss?

**[33:22]** the loss you have to compute now which is uh n dot functional binary cross entropy with logits okay and y now this is the loss that you're using okay now you got something now let's uh see what are these okay so now now what happens is each of this now we'll have what is known as a gradient

**[33:50]** function Okay, which is known as the grad function. Okay, now where in which we compute its derivative during the backward propagation step. Okay, and then it is stored in this grad function. So now let's print and see now how are the gradients stored. Now I'll use the formatting f now. Now you can see this now it is

**[34:21]** stored in this grad function in these locations. Okay. Now how do we compute the gradients? Okay. Now you have to use now what is known as loss dot backward to obtain the gradients. Okay. This is the back propagation. Now for that what it will do is now with respect to the loss now it will be computing the gradients for w and b okay

**[34:52]** you'll be obtaining the partial derivatives with respect to the parameters of the loss okay so now x and y are fixed so it doesn't make any sense to go ahead with that so we can look at now what is the grad value okay so now how do we do it once you do it we have this print W.grad grad will be there. Now you can see that

**[35:19]** this is the gradient value. Now it is very much evident that see the gradient value and the value should be same. Now otherwise know how do you perform the stoastic gradient descent equation. Now which is when which w minus alpha times the gradient. Okay. Now alpha is a fixed value. Now you'll be going ahead. This is how you obtain the gradients. Okay. So now if whenever you have a larger network in

**[35:51]** picture now it is not needed that for every parameter okay you need gradients and every time you need to compute the gradients okay now the gradient computation now can be disabled in certain cases. Now one is during your inference. Okay. Now during your inference you don't need to keep track of any of the gradients. Correct? No gradient should

**[36:20]** be taken with respect to what you don't you I'm not saying about computation. I'm saying about tracking. Tracking of gradient itself can be disabled. Okay. So now uh how do we know that? Now let's let's do that. Now we just obtained Z. Right? Now Z equals again let me compute tors dot so it is matt null of

**[36:50]** X comma W and then you can say that Z dot requires grad now this will be a boolean value it is true now that means that as of now you are enabled the gradient tracking okay but on the other hand now we can disable this how do you disable this with torch dot no grad. Now whatever operations you do now

**[37:21]** for this outcome which is there no you are not tracking the gradients. Now because the operation has happened where in which I'm saying that I don't need any of the gradient tracking. Okay. Now therefore for certain operations you can actually disable the gradient tracking and then work with it. Correct. This is one way. The another way is using what is known as detach. Okay. Using uh if I compute this. Okay.

**[37:52]** Now let's say that uh I just copy it. Z determinant uh just equal Z dot detach. Okay. So now if you put now what is the gradient of this? This is false. Dot detach will also exactly do the same thing. Okay. The same result that we

**[38:20]** have obtained earlier. It is just doing the same thing. Now then what happened to zed's gradient. Okay, that also we can track. Print zed dot requires grad. Yeah, that will be true now because you have detached it for this specific case.

**[38:49]** Okay, this variable is not holding any of the gradient tracking. Okay, fine. Zed still has the gradient tracking but the DT doesn't have the gradient tracking. Okay, now why is this needed? See, sometimes whenever we are having a huge network, we would have frozen couple of parameters and only want to update the weights of certain other parameters. Now in those cases for those parameters, you don't need to

**[39:16]** track the gradients. Okay, in those case times you can actually say that no I don't need it. I don't need to keep track of the gradient in these cases. Now all these are achieved with what is known as computational graphs. Okay. Now how does it do is this autograd which is there it will keeps the record of the data of all the executed operations along the new tensors that have obtained. Okay. And how does it do that? It does it with

**[39:44]** what is known as a directed asyclic graph. Okay. Now this is similar to the graph that I actually wrote the one that I Okay, I'll just show you it. This is the directed asyclic graph. Okay. It is stored uh like this. Okay. Now then what happens in this DAG? Now we will leave the the input tensors now which are which are having a path from input tensors now roots to the output tensors. Now this will so by tracing

**[40:13]** this graph from root to leaves. Now we can automatically compute the gradient. Okay. Now in the forward pass now the autograd now which is there will simultaneously perform two things. Okay. In the forward pass. Okay. Now run the operation which is there. Okay. And maintain the operations as per the diagon. Okay. It will maintain the gradient function as per the diagon. Now once you perform the backward option.

**[40:44]** Okay. Now it is called on the DAG root which is there and autograd then what does it do? Now it will compute the gradients on each of the value now using the gradient function. Okay. And accumulates the respective gradients with respect to grad gr. So and then how does it do it? It uses it now the chain rule the standard chain rule which is hardcoded. Okay. Now now this chain rule propagates all the way to the leaf nodes. Okay. Now

**[41:14]** this is where this is the grad this is the grad function. Okay. Now compute the gradients for each of these grad function wherever it is needed and they're accumulated at these points. Okay. Now that is how it actually computes the uh the gradient. Okay. So now the key takeaway is not every parameters needs the gradient of data. So now depending on the need now

**[41:42]** you can actually freeze couple of things. Okay. And all these gradient operation I did on CPU. I haven't quoted any of these onto GPU as you would have noticed. Okay. Now with this we conclude this second part of the tutorial in which we actually wrote the first neural network and then we did the forward propagation till prediction also. uh so we converted the logits into the the posterior probability and then we

**[42:11]** obtained the the prediction as well and then for a very simple case we understood how does we can play with gradients. So in the next part of the tutorial we'll be seeing how to optimize the weights of these new things and then how does the whole training process works and then how do you save the model. Okay, this is what we'll be seeing in the next part of the tutorial. Thank you all for watching. We will meet in the

**[42:40]** next tutorials.
