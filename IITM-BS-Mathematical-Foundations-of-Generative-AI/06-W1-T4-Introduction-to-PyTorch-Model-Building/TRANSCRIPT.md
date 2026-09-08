# Transcript — W1_T4: Tutorial 4: Introduction to pytorch: model building

> **Source:** https://www.youtube.com/watch?v=h1hEddM0aVE  
> **Channel:** IIT Madras - B.S. Degree Programme  
> **Duration:** ~44 min  
> **Note:** Auto-captions cleaned lightly. Minor ASR errors possible.

---

**[00:16]** So now uh let's look at uh the transformation. Now what are the different transformations that are available? They have uh just listed it up. Now the input uh so you have downloaded the data. Now the transformation that we preferably use is two tensor which is mandatory. Now there are some more transformations. No it's not elaborated here. There are some more transformations that we shall be using like random crop

**[00:43]** resize all those things we will look at in uh so whenever we get the time of it whenever we are looking at that code then we will be telling you how uh your input features has been transferred. Okay. Now this is the first step. The data is ready. Okay. Now you have curated the data. So now comes the next thing which is your build the

**[01:11]** network. Okay. Now uh now we should be building the network. So now whenever the next step now which is uh build the network first you should uh know what are the choices of network. Now I'm assuming that we are in this task. Now we are uh dealing

**[01:49]** classification. Okay. But almost similar principles holds for other modalities as well. Now whenever you have image classification now the very crudest form that we can do is uh we use an MLP or you use CNN's okay or nowadays you can use even uh the vision transform as VTs now this is VTS are for future okay

**[02:19]** this is for uh so we'll not be looking into VTS as of now vision transformers preferably we'll be looking into multi-layer perceptrons and convolution neural networks okay so now whenever I say convolutional MLP now let's take the first case of MLP okay that is MLP now there are some choices that needs to be made now the choices

**[02:49]** are the first thing is uh how many layers nodes in each layer that also needs to be decided. a activation function that needs to be

**[03:22]** used. Okay. Now these choices know needs to be dealt with. Okay. So now let's uh take these uh choices into hand and then create an yeah now let's build a neural network. The first thing is OS. Now why do you need OS? Now if you are retrieving anything from the system the torch for the PyTorch. Now torch import NN. Now

**[03:51]** this NN is stands for neural network. Now data loader is needed to get your data and then data set and transforms are needed to transform your data. Now these are the libraries that we have already seen. Okay. Yeah. Okay. This is something that we haven't looked at. How do you get your device? Let it run. I'll rerun it again. I should have told you how to get the device first. It will take some time to initialize all these libraries. These

**[04:19]** are very big big libraries. Need some time to initialize. and establish the connection. Yeah. Okay. So now uh if you click it here, click it here and then you can see here change runtime. Now when you look at this, these are the available devices for free. Okay, a CPU is there, a T4 GPU is

**[04:49]** there and V2A TPU is there. I'll use this uh T4 GPU. It will say that it will terminate your current session. Okay, let it terminate. No problem. I've saved it. So now everything that has stored in the previous thing, all of them will be removed. So I need to rerun this so that all these libraries are reloaded again. Or you can change this. You go to this runtime. You can change this uh change runtime and then change it to

**[05:18]** GPU, CPU, back and forth. You can work it out. Yeah. Now torch. Current accelerator if it is there use it. Else you use a CPU. Yes. Now we are using the CUDA device. Now CUDA is uh the name that is given for this uh all these GPU devices. Okay. Yeah. Fine. I'm using the CUDA device now. So now let's create a neural

**[05:46]** network. That is what uh so we are intending to do now. Okay, let's do it. Now uh we have to create a class of this okay class neural network whatever name that you want to give and that has to be the child of the parent called as nn dot module that is the name of the parent. Okay fine. So now in the init method in the

**[06:14]** constructor you have the self. If you want any other arguments, you can pass it on. And then you call super. That means that you are calling the parents init method. Okay. Now, init method is used to initialize all the parameters that are there. Now, what is the first thing that you are supposed to do? Okay. Now since you're doing an MLP and your input is a we are taking fashion Mness which is cross 28 matrix as a single channel

**[06:53]** image and then the input for your MLP will be a vector. Now you need to convert this into a vector. We need to flatten it up because of which the outcome now will be a vector which is cross. How do I get this 784 that is 28 + 28. Now how do you do it? You take

**[07:21]** the first row put it here. Take the next row put it here. The next row put it here. And so on and so forth. you put it and then you get a 784 + one block and this process is what is called as flattening. So now you have a vector which is of size 784. Okay. So now your uh input dimensions are fixed. Okay. Now which is 784 and your output dimensions

**[07:50]** are fixed which has to be 10 because you have 10 classes. So the final it should have 10 thing you get the logit out of it. Now I presume that you know the theory and then you pass it through a softmax method. You you pass it you pass this [Music] through softmax you pass this

**[08:16]** [Music] through softmax now and get a probability vector out of it okay of size 10 + 1 we are taking a the case of FMS which is a 10 class problem. So we'll have 10 + one. So now the thing is what is there in between? Now you need to look into it. Now now this is what you should decide how many layers should

**[08:43]** be there. Now your input nodes are fixed. 784. Now you have this 784 things. Your input nodes are fixed. Now how many hidden nodes and how many nodes at each thing now has to be fixed. Okay. So now let's look at the code that has been given. Okay. Now first you go ahead and do self.flatten where in which uh the image that you have you're flattening that

**[09:11]** specific image and then what you're doing is you are creating the next thing that you are doing is you're creating a pipeline. Now the idea of pipeline is pretty much simple. Now you have uh multiple processor multiple processes. Okay, you pass your input. The output of the first will be the output of the next like this. Like this like this. Now what you do rather than

**[09:40]** creating each one of them separately, you create it as a single entity. Okay? You create this whole thing as a single entity and then you can make use of it. Now this is the idea of pipelining. Okay. That is what is being used here self dot linear ru stack I'm using the activation function ru so now ru stack n dose sequential now that means that as I told you it's a

**[10:10]** pipeline the output of one thing is input for the next stage I'm using nlinear so this is the first thing now you create one layer here. Now he's saying that 28 + 28 comma 512. Now what does that mean? You have this 28 + 28 now which is your boot. Now then the next thing that you have is

**[10:37]** a fighter. Okay. Now this is one thing. This is one. Okay. Now this is normally this will have a weight vector now which is uh is a matrix which is of size this belongs to R 784 cross 512 and then you have

**[11:06]** uh the bias which is in R first one in R5. Okay. Now this is the first layer that you have specified and then the output of which after this transformation now 784 into 512. Now using an MLP you multiply it. Now how do you multiply

**[11:34]** that? Okay. So now all the MLP you can think of it like if you have your input X. Okay. Now you have this W this Wrpose X is one forward uh propagation that you have. Okay. So now if you case if you take any X now in this case my X will be in it will be a 784 + 1. Okay. And what will be my Wrpose? My W transpose will

**[12:06]** be 512 cross 784. Okay, I can multiply these two things. So the outcome will be a 512 + 1. This will be the outcome. Now to this you'll be adding this B1 now which is a 51 length vector. Okay. Now the output will be a five + one vector. Okay. So this is one step.

**[12:36]** Okay. So that means that every propagation in the neural network you can think of it as a matrix and vector multiplication or matrix matrix multiplication in case of convolutions. Okay. So and then the output which is there the output that you have got the 512 length vector. Now that you pass it through a ReLU activation. Now we know what is ReLU activation. Just to give you a recap. Now if your input is Z okay it will be

**[13:07]** zero if this Z is less than or equal to zero. Z if Z is greater than zero. Okay. Now this is the rectified linear unit. Now ReLU activation that has been used. So now this 512 length vector will be passed through this ReLU. That means that each and everything now you have first element now it will check whether the first element is less than zero or greater than zero. If it is less than zero it will become zero like that

**[13:36]** for each of the element it will check individually that you are applying this railway activation. Now after that again you have one more fight to fight. Now that means that you have fixed this fight. Now this is the input. Now from here again you have a five. Okay. Now that means that you have one more matrix. Now

**[14:06]** here W2 just please bear with me here W2. Now this is of the matrix of size 512 cross 512. Okay. And then your biases two know will be for each node there will be a bias. Okay. So now you have uh let's call this with some you have

**[14:36]** a 512 + 1 that you multiply with w2 transpose. Okay. Now, which is again 512 + 512. Now you multiply this you get a 512 + 1. To that you add B2 which is a 512 length vector. It will be a 512 + 1. Okay. The output of this

**[15:08]** is a 512 + 1. Okay. Again you use a ReLU activation here and then the next layer is you have 512 to 10. That means that you have 512 here to 10. This is the next matrix that you are considering. Let's consider this as W3. Now this is R 512 + 10. Your B3 is a vector size 10. And then what happens? You have

**[15:43]** this 512 + 1 that you multiply with W3 transpose. What is this W3 tan transpose? It will be 10 cross 512. Now this outcome will be 10 + 1. To that you will add B3 which is a 10 length vector. So the output will be a vector of size 10 + 1. And then this is what is called as

**[16:10]** logits that you pass through the soft max function. Okay. To get posteriors now which we call it as P. This is soft max of whatever the vector that is there I just it here since I haven't given any names to it. Now this will be your probabilities. This will be a posterior probability. Now this is now to be more specific this is P of Y given

**[16:39]** X okay for a given X what are the probabilities that we have has been obtained from this neural network okay and please remember that for the last layer you should not use any of these activations don't even put or activations or sigmoid or anything you should leave it up like that okay now this is your uh whole structure. Now here see you can use different

**[17:08]** activation functions, you can use different number of nodes. Now all those things are up to your choice. Okay. And then the forward method now which is your forward propagation. Now first you flatten it. Now that means that you have this 28 + 28 matrix which is converted into a uh what do you call it 784 dimensional vector and then you pass it through this relu stack. Now that means that first it will be transformed to 512 51 to 512 and then 512 to 10. Now you

**[17:36]** get the logits and then you return this logit. Okay. Now this is the class definition and then here you are creating the instance of the class called as model. Now the neural network you're creating an instance and then you are pushing it into device. Now that means that till now it was on CPU. So now you're saying that now you deploy this on the GPU which you have run

**[18:06]** this. This is the network that we have. Okay. So now let's take a simple uh input and then let's see how it works. I take uh an input which is 1, 28 + 28. This is my input and then I'm putting it I'm placing it on the device. Okay. Directly placing it on the device. This is a random input. I don't know what it means. And then model X. Now this X uh you're passing it through the model.

**[18:37]** Now that means that it will undergo this forward pass which is nothing that transformation and then you get the logits. Now those logits you pass it through the softmax function. Okay. Now that will get you the probability vector and then you choose the highest of that probability vector. Now which will your class you have seven is your class. This

**[19:02]** is a random input. So now we don't need to worry about it. It doesn't have any uh what do you call significance here. Now once we take much more uh uh classic example it will be clear. Now this is how you build a neural network. Okay. So now let's go back. What are the things that are ready now?

**[19:28]** Okay. The things that are ready as of now is uh your data is ready. Data is okay. You could have data is ready. and then your uh network is ready. Okay. Now these two things are

**[19:58]** done. So now the next thing that is needed is how do you optimize this thing? How do you solve this? Okay. Now for that uh you need an optimizer. So I will not be specifically going into this automatic differentiation which is your uh how do you calculate the grads and string all those are internally taken care I'll not be delving deep into it because we'll not be needing those ideas we can look into this it's a

**[20:25]** pretty simple straightforward thing how do you optimize the model parameters let's directly go into that melo how do you optimize these model parameters how do you obtain these values Okay. Now, yeah. Now, this is the prerequisite code. I presume that you are comfortable. These are necessary libraries. You're loading the data sets. Okay. And then you're creating an

**[20:52]** instance the model. All those things are ready. Okay. Setting up the connection. It will take some time do. So, let's go back. Uh let's once it's done, let's look at it. Now uh whenever uh so we want to uh uh optimize it. So we we will be using gradient descent. Okay. Now what is the equation

**[21:22]** of the gradient descent? Now w equals w minus sum alpha * whatever is the loss respect to the w all those i j I have just removed it. Now this is the gradient descent. Now if you're using a batch of it, this is stochastic batch gradient descent SGD as it famously called as. Now you have this alpha which is your uh learning

**[21:58]** rate. Okay. Now we have many ways in which you can solve this. Now one is what is the plane SGD is one way and we have this uh RMS prop we can this is one more solver I'll not be going deep into this I just giving you the names that you can read through how is one different from another and then we have this Adam

**[22:28]** okay now this adaptive method with momentum that is why it is called as Adam. Similarly, these are some of the famous methods. Now with using one of these, you know, you can solve this problem. Yeah. So for that we'll be needing learning rate also called as step size which is 10 ^ of - 3. 1 e - 3 is 10 of minus 3 and then you have a

**[22:56]** batches of 64 then you are running it for five e okay your things are done now okay now and then you need a loss you're using the cross entropy loss okay the CC laws as it is famously called as now you need the optimizer okay now you are optimizing on these parameters all the model dotparameters all the parameters of the model if you want to only do it on

**[23:24]** certain set of parameters that can be done. Now we'll look at how to do it in the coming sessions. Okay. Now torch.optim.sgd now you're using stoastic gradient descent and all the parameters. Now what is the learning rate integral of minus 3 which we have declared just in the previous and then let's look at the training loop. Now now it's interesting. Okay. Now let's look at the the overall training loop. Okay, this is a function.

**[23:54]** Now the function will be called only when the function call is made. So now let's go to the path first. Now you're creating an instance of your loss entropy loss. Optimizer you're using SGD. Now epoch is 10. Now that means that you're running it for 10 epochs. Okay. Now for each epoch you'll be doing this. What are you doing? First you print something. Okay. Fine. Now then you are calling train loop. Now that means you're calling a

**[24:22]** method. You're calling a function. Sorry, it's a method. It's a function train loop. Now to that you are passing the test loader train loader. You're passing the train loader model loss function and optimizer. Now these are the parameters that we are passing. Now let's go there. Okay. So now we are in the train loop. Now you have the data loader model loss function and optimizer. First you have calculated the length. Okay. uh let's let's try to understand this

**[24:51]** overall thing much more clearly. Uh we are in the train loop. Okay. First we have uh the first step that we have done is the size. Okay. This is number of examples. Okay. We have obtained that.

**[25:17]** Now model train. So now every model that is there now will be in two phases. One is either in training phase or it will be in the evaluation phase. In the training phase now you need to actually go ahead and compute the gradients. In the evaluation phase you don't need to look at the gradients. Okay fine. So now you're saying that uh put the model into the training phase and then you're saying that okay now enumerate enumerate

**[25:46]** on the data. Now what happens when you say enumerators? Now enumerate uh this D I'm using this data loader using this enumerate. Now what happens? What is this D? Now D1 D2 so on till DK right now how are the index now this is index 0 1 so on till minus one. These are the

**[26:14]** indices. Now what are the values? Now it is saying that batch comma x and y this is what it is saying. Now first this zero which is there will be assigned to this batch because of enumerate the index will go here and this d1 will come to this. Okay then with x and y there okay now you have got the batch index as well as x and y. Now this is

**[26:43]** what enumerate does. Now then you compute the forward pass. Then you perform the forward pass get the predictions. Now once you have the predictions you calculate the loss. Okay you calculated the loss. Now if you remember those are the steps that we looked at even in our earlier tutorials. Just for a recap I'll just open that thing.

**[27:12]** You have your uh x and y. Now first you calculate the prediction and then you calculate the loss. That is the second step that uh we need to be doing. We calculated the loss. Then loss dot backward. Now this will be computation of gradients. Then optimizer dotstep is where you update the values. Okay. the four steps the prediction

**[27:42]** computation of loss obtaining the gradients and updating the weights and then optimizer.0 zero grad. Now what it does is now there will be an accumulation of gradients that happens after each gradient step. What you should do is now you should refresh the gradients and make it as zero. Okay. All the gradients have to be make it as zero otherwise it will be having issues. Okay. So now that those are the steps that you're doing. Now when your batch is

**[28:09]** divisible multiple of 100 then you loss item that means that you extract only the numerical value from this loss. Now why only numerical value? No because this loss is not just number. No it has something more to it. There is something called as computational graph that is involved. I don't need that computational graph. I just need the number. So just los item you get the number and then you say that batch into batch size plus length of x

**[28:38]** and that will give that know how much is the current size of this is just to print the statistics. This is the overall training loop. Training one epoch. Now this you are doing it for each epoch. You are doing this. Okay. Then you have test loop. Okay. Now let's let's go to the test loop. Data loader model and loss function at the input. Now then you give model eval. Now that means that you are saying that okay. So now how many uh

**[29:07]** sorry you are saying that I'm working in the evaluation mode now. Okay. So and then uh what do you do? You go to the test loop first. You compute the size which is number of examples. Okay. Now then you have number

**[29:33]** of batches. How many batches are there? That also you take it into consideration. size and number of matches. So now you should be evaluating right you should be evaluating how much is the loss and then how much is the correctly classified examples for that you need these two placeholders now with torch.n No grad. Now, now why is this dot node grad is used as now? This ensures that no

**[30:02]** gradients are computed in the block which is in uh indented this. Now that means that in this whole block there will be no gradients that are computed. Now you'll ask the question why do you need this? Okay. Yeah, probably we only need this. This is unnecessary but this is added for best practices. Okay fine. So now you iterate over the data loader which is the test loader. Now you get X and Y. Then you go ahead and pass it through the model. You get your

**[30:31]** predictions. Okay. And then you get your loss and then add it to the test loss which is there. You just need the item. You just need the numerical value of it. Good. So now you have added the loss. No, but now we are dealing with a classification problem. Now for us the accuracy is much more important. Now what do you mean by this accuracy? Now that means that how many examples have I correctly classified that I need to

**[30:59]** calculate. Now how do I calculate? See I have obtained a probabilities. Okay. Now what do I do? I have obtained a probability vector. Now let's take have P1, P2 and P3. This is I just have a three class classification. Now if I calculate the highest now which one is high which one is

**[31:30]** highest to that class I'm putting right now which one is highest that will give you a value but I need the index right this is zero index first index or second index I need the I I'll be needing the and what is the what is index of that is an expression. Okay, I get the index of okay now let's say that your P2 is high is no and then you get the index as one

**[32:00]** now then you say that this is correctly classified now when this one now this will be your Y hat this is same as Y which is your actual value then only say that it is correctly classified right now I want to do this specific operation now do that correctly Squ that means that you are increasing it with the spread which is your predictions dot arg max now you

**[32:31]** are taking the index of it the first one the top one is that same as the y now you're thinking that is my prediction is same as the y okay now then it will be true or false right now you will get a vector which is true or Okay. Now you have a batch you will get a list of true or false. Now that true or false you convert it into a torch

**[33:02]** float. Now that means that now true will be converted into one and false will be converted into zero. Okay. So now in a batch now if there are 10 correctly classified things I have 10 ones and the remainings are zero. Now if I sum all of them now I'll get 10. which is number of correctly classified example I just take the item of it that means I just take the number of it okay because of which my current will be updated accordingly okay so now and

**[33:32]** then to run this so that I can have the loss I can show you that also now test loss now we are calculating per batch so now you divide it by number of batches so that you can know that now what's the test loss per batch and then the correct you know how many examples are correctly classified that you divide by size that you get the accuracy and then you multiply that with 100 will give you the accuracy and then you'll get the test score of it. Yeah, you can see

**[34:02]** that how the for each 100 thing you'll have are incremented and accuracy is 42.4 four and then your loss is 2.16 and similarly you can see that the accuracy is increasing and meanwhile the loss is decreasing. Okay. Now in addition to this now normally what one more thing that we do is we will have a list where in which we append the losses and then

**[34:30]** we get a curve which is this is epoch and loss values. Now it is uh we need a trend like this. Okay. Okay. We need a trend something like this. That means that as the EPO goes the loss which is there is reducing after that after some time you know it will flatten that means that there will be no much of

**[34:59]** a change. Okay. Now the next question that comes is you know how many times you should train it. Now there is no standard answer for that. Now we'll be looking at these questions when we start uh handling the uh tutorials on uh the generating models while coding them. Yeah. And how do you figure it out? All those things we will how do you in a simple way we will figure it out. Don't worry on that as of now. Okay. As you can see

**[35:34]** that we have done 10 epoch. So it will run for 10 epochs. The accuracy that we have achieved as 68.7. The accuracy accuracy should increase but the loss should decrease. Okay. Yeah. That should be evident from the discussion that we have. The planning is done. Okay, this is the overall process. Now, let me reiterate this. You prepare your data. If the data

**[36:04]** is available from PyTorch, let take it. If it is not, you create a custom loader. You create the model. Yeah. What are the parameters that are there in the model? You should be aware of that. You take the hyperparameters. What is the loss that you're using? What is the optimizer that you are using? Train loop when the steps are first. You predict, compute the loss, obtain the gradients, update the weights, make the gradients as zero. This is one additional, this is a

**[36:31]** programming step to be more specific. This is not a critical step. And similarly, you go ahead and perform the test and run it for whatever number of epoch you want to do it. So this is how you optimize your uh now that means that you have obtained the optimal weight. Okay. Now you have obtained those theta which is optimal. Okay. When the loss is least in this optimization problem. Okay. So now the last step that is

**[37:01]** involved is how do you save these? How do you save this? because every time whenever I want to reuse it, I don't want to rerun it again. Retrain it. Okay. Now you should be able to save it. Okay. So now I don't need this. Okay. I just need these two statements. Now whatever model it can be that you have saved that you have now to do. Whatever model this is the name of the model.

**[37:30]** This is a variable name. You can use whatever name you want that in the current path you save it as model.path. torch. Model you save it as model pH. Now that is the extension. PTH is the extension of this. You store it like this. Now this is how you save it. Now if you want to load it. Now how do you do it? Now you have the model. Okay. So and then torque.load. You have created the instance of the model. Now

**[37:58]** model.path which is the name of this code. And then you only copy the weights. Wait only. Okay. You're not only copying the width. So whatever is involved in that just copy everything. Okay, this is how you load the model. These are some of the steps that are involved in looking at it. Now we have uh till now we have looked into the idea

**[38:27]** of how do you create a model? Preferably we have created a MLP model. Now at the back end of uh the second tutorial what we should be doing is now we shall be looking into how do you create a CNN model. Yeah. Uh now that is the important thing. Now whenever you have uh I'm assuming that you are pretty much comfortable with

**[39:11]** the CN models. Now the first question that arises whenever you're considering the CN model is how many layers? Okay. Now once you have fixed this question, the second thing is okay. The second thing that arises is you should specify in each layer. You should specify in channel that depends on the input that

**[39:41]** you have. Then you should specify the out channel. This out channel is same as number of filters that you have. And then you should be specifying the kernel size. Okay, we'll be using square kernels. So even if you say five now that you're using a 5 + 5 kernel and

**[40:11]** then what is the stride specify and then you specifying the padding. Okay. Now these three these five options you should be specifying only the input channel here depends on what is the previous stage's input that it depends other than that you should be specifying these four parameters and then once you have specified them then now we are still dealing with an image

**[40:38]** classification problem that means that you should be Okay. Deciding on the classification. Now all the CNN models has some convolutional blocks. Let's say that I have convolution one have this convolution two. I have this convolution three which whatever

**[41:08]** parameters that you have is there. Now after this now you get a block. Now you get an an image block in which you flatten it up and then you need an MLP. Okay. Now this uh logs of convolution layer it is famously called

**[41:37]** as feature extractor. Okay. These are called as feature extractors. And then this MLP which is there this is what is called classification. Okay. Now we need to decide even on this before going ahead and whenever an MLP

**[42:05]** comes into picture. Yeah. The problems of how many layers it's not like you have the input layer and then you have the output layer or how many hidden layers how many nodes in those hidden layers all those things also needs to be decided and what is the activation function similarly activation functions all of these layers also needs to be decided okay yeah and then you need to code it accordingly okay so yeah now the only thing that changes uh in that case is

**[42:40]** uh all your training loop loaders everything remains same only the neural network that is there now that will be changed to a convolutional block yeah so now uh whenever needed now we shall be looking at this convolution layer as of now let's uh stop today's uh tutorials on these ideas just to give you a small recap of what are the things that we looked at today. Now we looked

**[43:09]** at tensors. What are what are tensors and how do you work with them? We looked at the data loaders. We looked at the transforms. How do you build the neural network? Specifically, we are looking at an MLP in that case. Now in an MLP, what are the parameters that are involved and how do you work with them? We understood that. So okay. And then the forward pass and then the training loop, the test loop. Then how does uh you can have even a CNN

**[43:38]** model also in the as a model. Okay. So yeah with this let's uh conclude today's tutorials. From the next session now we shall be starting with uh the interesting stuff. We'll start coding GANs. Okay. Thank you. Bye. See you in the next videos.
