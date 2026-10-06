# Transcript — Tutorial 13 : Pytorch - Training the Model

> **Source:** https://www.youtube.com/watch?v=5PruFG5g1C4  
> **Channel:** NPTEL - Indian Institute of Science, Bengaluru  
> **Duration:** ~33 min  
> **Note:** Auto-captions cleaned lightly. Minor ASR errors possible.

---

**[00:02]** Hello all. Welcome to the third part of the tutorials on PyTorch. In this part, now we shall be looking at the training procedure and then the evaluation procedure. So, now what do we mean by training? Now, we need to obtain the optimal values of the parameter. And how do we do it? Now, we perform the forward propagation, compute the loss,

**[00:31]** obtain the backward perform the backward propagation so that we obtain the gradients and then we update the weights. Okay? This is the standard procedure. Okay? Now, and how will be be doing it on all the examples? No, we'll be using a batch of the examples. So, now let's just make a note of it. So, now the first thing is how many times you want to iterate over the whole

**[00:58]** data, which is the epoch idea. Okay? Now, for e in range number of epochs. Okay? Now, I'm using Python-like struct code here and then so for so many epochs you want to iterate over the data. Okay? And then we will be dividing it into batches. Okay? With the whole data

**[01:28]** is divided into batches. Now, since the whole data is divided into batches, for each batch I'll take it, perform the forward pass, perform the backward pass gradients, update the weights. And then for the second batch, for the next batch, I'll be going ahead with the updated weight. Okay? Now, then let's say that there are batches. Okay? Now, my whole D which is there is my whole data which is there is having small batches. Okay? Then for D in

**[01:58]** D. Now, here D is a list of all the batches. So, one after the another, so it will be assigned. So, now what is the first thing that I'll do? Now, I have X for the given batch. Now, this X which is there is features. Now, first we predict Y hat. Now, how do

**[02:23]** we predict? Now, it is We have declared a model. So, we use model of X. Okay? Now, you get the prediction. And then you compute the loss. Okay? Now, how do we compute the loss using Y {comma} Y hat? The actual and then the predicted value. That will give us the loss. Okay? And then once we have the loss, now then

**[02:53]** we use backward. Now, we use backward to obtain the gradients. Okay? And then we update weights. Now, this you will repeat again. Now, for the second batch that you get, you will be working with the updated weights. Now, so on and so forth. Now, how many batches are there? Now, in one epoch, so many times you'll be updating

**[03:21]** the weights. Okay? So, normally this is what is said to be iterations. Now, one iteration is one weight update. So, in one epoch, you can have multiple weight update depending on the batch. So, now to perform this, now there are certain things that you have to decide. Okay? Now, which what we call as the hyper parameters. Now, what are the hyper parameters that we want to decide? Now, the hyper parameter see What do I

**[03:52]** mean hyper parameters? Now, these are not learned parameter. No, these are just you are considering them. It's a value. Okay? Now, the first thing that we should consider, the first hyper parameter that we should consider is a learning rate or the step size. Okay? Now, then you should decide on the the batch size now we have to decide.

**[04:22]** And then how many epochs now we have to do. Now, these are the choices that we have for the hyper parameter, the basic choices. And the basic architectural choices, okay? The arc choices that we have uh number of uh hidden layers uh

**[04:51]** And then number of nodes in each layer number of nodes in each layer we have to decide. And then the activation function. Now, these things needs to be decided in terms of the architecture. Now, since we have created the model, now we have uh made our choices. The number of hidden

**[05:18]** layers we have taken two. So, number of nodes in each of the hidden layer we have taken 512 512. The activation function we have taken the the ReLU activation. So, these things we have decided while making the architecture. Now, then the optimization choice, okay? Now, then comes the choice of See, there are multiple uh

**[05:48]** now variations of the gradient descent. Okay. See, one problem that you can see with the gradient descent is, now across all the epochs, for all the parameters, the learning rate actually will be the same. So, now the question arises, is there a way in which we can dynamically, uh, shed, we can dynamically shed, obtain the value of the learning rate, now based on the previous gradient.

**[06:15]** Okay, based on the previous, uh, directions and the and the steepness of those directions. So, now we can, can we actually obtain the, uh, gradients? Uh, the, sorry, the learning rate. Okay. Now, that gives rises to the choice of what is known as, uh, different optimizers that we have. Couple of famous optimizers that we have is Adam. There is one more thing called as, uh, RMS prop.

**[06:43]** And SGD, which is the vanilla one. Okay. So, now we'll be shedding more light on this. So, now in the theory sessions, okay, now we'll be discussing this in detail in our, uh, So, we'll be discussing this in detail. Okay. So, for now, let's just use Adam. Okay. Uh, because that is the one which is, uh, very nicely performing optimizer. And then we will see how does from SGD,

**[07:11]** uh, Adam actually develop. Okay. Now, these are the different choices that we'll be having before going ahead with the thing. Now, couple of choices that we have already made is this. We have made the Adam choice. Now, the only choices that are left with are the hyper parameters, now where in which I have to look at the learning rate, batch size, and then the epoch. Okay. So, now &gt;&gt; [clears throat] &gt;&gt; let's, uh, do that. So, now I'll just need to

**[07:43]** run all these ones and see whether uh change runtime T4 is there. Good. This is It would have disconnected because we haven't used it for some time. So, I just need to rerun the things. So. It is getting connected.

**[08:09]** These are the prerequisites code. I need the CUDA device and then uh the network is needed. And then I have created the network. Okay. Now, this is done and then I have uh I have the data with me. Okay. So, now here and one important thing that we should remember is we have the data set. So now, we haven't converted them into batches. Okay. Now, this part is what is known as creating the data loader.

**[08:40]** Okay. Now, whenever we have data loader, now that means that we have created them into small segments of batches. The data set which is there, that we have converted into loaders. Okay. Now, so now let's do that. Train loader equals &gt;&gt; [snorts] &gt;&gt; torch.utils.data.DataLoader. I'll take the training data. Now, batch size, now I'll just uh

**[09:07]** keep it as 128. And shuffle equals to true. Now, what do you mean by this shuffle equals to true in the sense? See, for example, let's say that example one is in first batch in the first epoch. Now, second epoch, it may not be in the first batch. It will be shuffle shuffling. Okay, it might go to some other batch. Okay. That is what is known as the shuffling equals to true. So, what whenever we are dealing with

**[09:35]** the test loader, which is just the evaluation. You just evaluate it once. That is what is inference. There we don't need the thing to be shuffled. Okay, I'm considering the batch size as 128. And please remember that the network that we have is independent of batch size. Okay, whether you have batch 32 or batch 64, the same network will work. Okay, now because batch is another

**[10:03]** dimension. So, whenever you put all of them together, you have 128. Now you have 128 128 28. That will be the size of your data. Okay, now the network that we have is independent of batch size. Okay, so now we have all these things in hand. So, now the next thing that we need to be doing is these hyper parameters.

**[10:37]** Now, what are the hyper parameters that we have? The learning rate. Okay. So, the learning rate. Now, normally, now we have certain standard starting learning rates. Now, for a classification problem, we always take 1 e to the power of minus four. Okay. For classification problems. Okay. So, and then batch size we have already given it as 64. And then I'll take the

**[11:06]** epochs as 10. Okay. And then we have to Next we have to decide on the loss function. Now, since we are dealing with a classification problem, the loss function that we shall be using is cross entropy. Okay. So, how do you do it? nn. cross entropy. So, and then please remember for this

**[11:35]** cross entropy the input is logic. No. You to give the logic it will internally convert it into soft max. You don't need to convert it into soft max and give it here. Okay, then the cross entropy you just give it. Now then the optimizer is uh torch.optim Adam is the optimizer. Now for all the parameters I'm using the same optimizer. Now that means that you can have multiple for the same network you can have multiple optimizers for different

**[12:03]** parameters. That is possible. Okay. Uh theoretically it is possible. Okay. So then you define the the learning rate. Okay. So now we have to write the the training loop. Okay. Now we will do it for one epoch and then we'll have a for loop and then repeat it multiple times. Now d f

**[12:32]** I'll write the function. Train loop. Now what are the inputs for this? The input is the the data loader. That is the first one. And then you need the model. You need the the loss function. We need the optimizer. Okay. Now these are the basic things that are needed.

**[12:58]** Okay. Now first uh Now we have to have some kind of a bookmark. Now just to see how the loss value is changing. Okay. Now to do that Okay. Now we should know the total number of examples that are there in a given data loader and then so that you can divide it and then come to know now

**[13:25]** what is the loss? Okay. Now for that I need the size. Now this will be Okay. This is the length of data loader dot data set. Okay. So now for batch X comma Y in enumerate data loader. Now enumerate, now will give you two

**[13:52]** outputs. Now one is the first it will give you the index and the second one will give you this X and Y. Now this is your features. Now this is your target. Now this will give you the batch ID. Okay. Now you're iterating through your data. Now what is the first thing is your prediction. Now your prediction is uh model of X. Okay. This is your uh prediction.

**[14:22]** And what is the next step? Now you have to compute the loss. Now loss is uh loss function of the loss func the loss computation. Now then we have to look at the back propagation. Okay. So now while doing the back propagation, there is one thing that we should

**[14:52]** remember. Okay. Now that is optimizer dot zero grad. Now what is this optimizer dot zero grad? So now this particular thing, this particular uh method what it will do is for all the parameters in that optimizer, now in our case the whole model Okay.

**[15:20]** All the gradients of the models are reset. Okay. Now, why it is needed? Now, there is an issue of gradient accumulations. Now, gradients by default always adds up. Okay. To prevent double counting. Okay. Now, therefore, what do we do? We explicitly zero them after each iteration.

**[15:47]** Okay. Now, otherwise, what will happen if you do multiple iterations? No, it will by default adds up. Now, we don't want that to happen. Why? We are not doing any kind of gradient accumulation. Now, if you want to do some kind of gradient accumulation, now what do you mean by gradient accumulation? See, whenever you have a very huge model, sometimes we will run the model multiple times, accumulate the gradient, and then do back propagation. Okay. Now, for each update we are doing the gradient gradient back propagation. So, the update, so we don't need to go ahead

**[16:16]** with gradient accumulation. Okay. Now, that's this is to zeroing out all the gradients. Okay. Now, then loss.backward Now, we'll create the uh what do you call? We will obtain the gradients. Okay. And then optimizer.step, now it will perform the update. Now, this is the This is the gradient computation. And this is

**[16:47]** weight update. Okay. And then what do we do? Now, we need some kind of a tracking, otherwise we don't know what is happening. Now, if the batch that we have is of some multiple of some hundred, so just to keep track of the thing. Okay.

**[17:14]** So, loss item will go here. Batch Okay, I don't need the current value. I just need to see that what is the loss at that point. That is enough. Okay,

**[17:48]** let's compute this. Current is equal to batch into length of thing and then you divide it. Okay, now that will just give you an indicator of now how for how much you have obtained this. Okay. Fine. Uh okay. Fine. Now, this is the the training

**[18:14]** loop. Now, there is one thing that we always put here. Now, which is model.train. Okay. So, this Why do we do this model.train? Now, this is to set the model in the training mode. Okay. Now, whenever you have dropouts, uh those are regularizers and then batch normalization, whenever in the model you have these kinds of

**[18:43]** designs of dropout or batch normalization, at that time you'll be needing to do this. Okay. But, in this situation we don't have dropouts or normalization, but it is just the best practice to put it. Okay. Now, this is the the training loop. Now, then how do you perform the evaluation? Okay. Now even for the test loop, again,

**[19:18]** you want the model loss function and then the data loader. You don't need optimizer because anyhow you are not going ahead and uh updating any weights. Now therefore, you don't need the optimizer. So, number of batches and size you can obtain and then So, we have to keep track of loss and then how many

**[19:46]** values we are able to correctly predict. Okay. Now, what is the loss and then how many examples in the whole data loader Uh this is the total size. How many of them I'm able to predict properly? Now, we just need to keep a track of it. Now, this is when you have size, the overall examples, and then you have the correct number of totally correctly classified examples, now you can easily find the accuracy. Okay. Or okay, if you're dealing with

**[20:15]** any binary classification problem, well, then you can go ahead and find precision, recall, or F1 score that we have discussed earlier. And then you can obtain your uh ROC, AUC, all those things metrics that you can easily obtain. So, we have dealt deep into those ideas in our earlier tutorials whenever we are dealing with the base classifier. Now, if you're not having a two-class classification problem, then how do you do it? Now, you can consider one versus rest idea and then do it. Okay. So, now okay. Now,

**[20:44]** let's proceed. Now, whenever we are evaluating, now we have to put the model in eval mode. Now, again, this uh this is a best practice. Okay. So, now and then we put it in torch.no_grad. Why? Now, because I don't need any kind of gradient tracking. Now, why do I don't need any kind of gradient tracking? Now, the sole reason is we are doing inference. Okay, the gradient tracking with respect to the

**[21:10]** computational graph is only needed now whenever you want to update your parameters. Now, in this case you're not trying to update the parameters, so we don't need any kind of gradient tracking. Therefore, we put it torch.no_grad. Okay. So, now then inside this for x and y in data loader first you predict. Okay. And then the total loss which is there, you go on adding. Okay. Now, loss.item, that means that this loss which is there

**[21:39]** is computed on the device. Now, please remember that. Okay. It is computed on the device. Now, therefore now you have to put it as an item so that it converts into Python thing. Okay. Now, then how many of them are correct? So, now correct plus equals, that is correct is equal to correct plus. Now, you take the prediction and then you take the argmax of it. Okay.

**[22:08]** Now, your prediction you have a batch, so this will be a vector of some 128 dimension. Your y is also a vector of 128 dimension. So, this whenever you take the argmax, now you get the labels, the predicted labels. Now, that you are performing a relational comparison with the y. Now, you have a vector of size 128 that is your prediction. You have a vector of size 128 that is your actual value. Now, individually you obtain. So, a relational operation will give true or

**[22:36]** false. Okay. And true is considered to be one and you convert it into float. Now, whenever you convert it into float, what happens is true is becoming one and false is becoming zero. Now, that means how many ones are there, how many examples that are predicted properly. And you take the sum, that is number of examples. dot item, just to convert it into a Python numbers. Okay. Now, this will be able to tell us how do you uh obtain the number of

**[23:06]** correct examples. And then finally, now once it is done, then you will compute the loss for a batch. And then you'll just make a print. Okay. Now, average loss is uh Now, test loss this. You just make a print of this. Now, this is the test loop. Okay. The training loop. Now, wherein which

**[23:34]** prediction, loss computation, optimizer.zero_grad, so that the accumulated gradients are reset, and then loss.backward, so that we obtain the gradient computation, optimizer.step, wherein the weight update happens. Now, here prediction, loss computation, and bookmarking of the loss, so that we can understand how much overall loss is there, and comparing and obtaining how many in the given batch we are able to predict properly, and

**[24:01]** accumulating them, and dividing by the total size, so that I know the accuracy. Okay. Now, this is the training loop, and then the testing loop. Okay. So, I have created the functions. Now, the thing is now I have to actually just run it. So, now for e in range of epochs,

**[24:29]** now print Now, just print that f e plus one of epochs. Okay. And then then then you have the test for after each

**[25:00]** epoch I'm performing the test. And then I'll print training complete. Okay. Uh runtime error, good. All tensors should be expected on the same device. Now, why is this error coming? Okay. Now, the reason is no model we have put it on to the device, but X which is there we haven't put it on to the device. That was not done.

**[25:28]** So, we have to add X is equal to X.to_device, Y is equal to Y.to_device. If you don't have it, you'll get this error. Okay. Now, as I keep on telling, all the model and everything should be on the same device. If you have everything on the GPU, have everything on the GPU. If everything is on CPU, have everything on CPU. Okay. Now, you cannot have in between like that. Now, you can see

**[25:55]** that. Epoch for every 100 thing. Now, you can see that the loss is decreasing. Now, the test accuracy after one epoch is 91.9. Okay. The accuracy should increase and the loss should decrease. Okay. Now, an MLP is able to solve the the MNIST problem.

**[26:27]** See, now the loss is decreasing and the accuracy is increasing. Okay. See, sometimes there might be a very small fluctuations in terms of loss. So, we don't need to worry about it. It happens. As you can see here, from 0.12 it is going to 0.16, but that's fine. It happens. Okay, small fluctuations like this will happen. Okay. Now, this is the the first

**[26:56]** MLP that uh Now, we shall be running on the MNIST data. Okay. See, now we have significantly reduced. So, the training is complete. So, after 10 epoch, we have got 97.3 accuracy. Now, that means that out of 100 out of 1,000 examples, I have got 973 examples

**[27:25]** correctly classified. Okay? That's what it means. Okay? Now, the training is done, and then the inference we are doing it for each time. Okay? So, now Now, how do you save this model? Okay? Now, the model Now, what do you mean by a model? A model is nothing but bunch of weights. Now, see, each of the updates, now, what is happening? The parameters

**[27:52]** are getting modified. Now, you have to save this parameter. So, that if you want to use it again on the same data, now, you don't need to retrain it again. You can take the trained model and work it out. Okay? Now, how do we do it? torch. save Okay? You can say that all everything You can save it like this. model.state_dictionary

**[28:19]** Now, you're saving it as model.pth. You can save it up like this. Okay? And then, how do you load it? Now, loading is again simple. You create the instance of the model. And then, you you have to use what is known as uh So, okay, this is how you save it. Okay? So, now let me create a new model.

**[28:52]** Model two equals uh to device. Okay? And you create it. And then you load the model, model two equals from model.pth.

**[29:28]** You just load it. Okay? So, now now what we can do, we can directly run the inference. Now, test loop. What are the things that we need? We need the model. I'm sending the model two. And then the test loader. the loss function.

**[29:57]** Okay? Is it in the same order? Okay. The order is different. So, This is model two. I'll just run it. Okay. [snorts] Okay. Okay. Okay.

**[30:31]** Uh this is the reason. Okay. Okay. Got it. Got it. So, I have saved it in a different format, now because of which this is happening. So now Now you have saved it as saved dictionary. That is should not be the case. Most of the times we just save the model. Okay. And rerun it. Save the model. Created this.

**[31:29]** Oh, sorry. This weights_only should be false. Sorry. Okay. Yeah. weights_only should be false and then you run it. So you get this 97.3. Okay. And this is how you save the model and then you can reuse it. So you just need to create the instance of it and use it. So now that means now this gives us a straightforward idea that now any of the This is what is known as the pre-trained models.

**[31:58]** Okay. Now whenever we hear these terms pre-trained models, now they release the weights and then if you have the instance of the thing, you can just load the weights and then you can perform like this. So you don't need to worry about the training. Okay. Now that is what we shall be looking at in the next thing. Now wherein which we'll be taking we'll be considering the pre-trained weights of one of the very famous one of the very famous data sets known as ImageNet.

**[32:27]** And all these huge models like VGG network, ResNet, and then we'll be reusing them. Okay. Now this is the overall idea. Okay. So you should have the data set, model, and then the training procedure that is there. Now, perform the training, save the model, and then that save the model in essence, save the weights of the model, which can be reused for inference like this. Okay? So, with this we conclude the idea of MLP.

**[32:55]** The next thing that we shall be taking in the next part of this PyTorch tutorial is we'll be looking into CNNs, convolutional neural network, for the same problem of image classification of MNIST. And then, post that we'll be considering some of the famous uh architectures like VGG net, ResNet, and think. And then, we'll work it out. We'll code them up. Okay? That's comes in the next part of the tutorials. Hope you enjoyed this

**[33:22]** part of the tutorial. Now, welcome to the club. You have uh ran the neural network for the first time. Okay? Now, welcome to the club. Thank you, all. Bye-bye.
