# Transcript — Tutorial 5 : RNNs using PyTorch

> **Source:** https://www.youtube.com/watch?v=k6zF2NsvVrk  
> **Channel:** NPTEL - Indian Institute of Science, Bengaluru  
> **Duration:** ~38 min  
> **Note:** Auto-captions cleaned lightly. Minor ASR errors possible.

---

**[00:03]** Hello guys, uh welcome to this tutorials in which uh we will be continuing our discussion from where we left in our last time. See earlier in the last tutorial section that we saw we looked at CNN's. So we understood how uh we can tackle the idea of classification uh using CNN's image classification using CNN. For that we used a specific data set no which is MNEST. We looked at the training and then the evaluation. The

**[00:31]** evaluation metric that we have used since the data was balanced in that case was a straightforward accuracy. You can uh as we as I uh recall so I just said that you can use uh precision recall f1 score a curves and all those things. So you can go ahead and look into those aspects. So from there we shall continue. Now in today's discussion we'll be looking at sequences. Okay. So

**[01:01]** I'm uh assuming that all of you are quite comfortable with the idea of sequences. So where in which we have a temporal data. Okay. So now let's start discussing on sequences and how do you tackle the problem of sequences handling sequences by using recursive uh neural networks RNN or the updated versions of them which are uh LSTMs and then GRU. So

**[01:31]** now first let's uh create a sequence. Now I'm creating a sequence of batch size four. Okay, I'm having four instances. So I have uh here I'll just prefer to go ahead with uh x1 which is the first one x2 x3 and x4. I have four instances four examples and each example is having a sequence length of five. Okay. Now that

**[02:01]** means that this x1 now we'll have uh x11 x12 x13 x14 x15 something like this. Similarly for the remaining and each x i j which is there belongs to r3. Now this is what is our setup that we have as of now. Okay. You can

**[02:30]** see here I have a batch size of four. I have four examples and each examples have five temporal entities, five uh uh things and each of those uh uh t is having a is in R3. Okay. Now like that you create an X which is of that specific shape where which the dimension one is bat size. The dimension uh dimension zero is bat size. Dimension one is the sequence length and dimension two is the input

**[03:00]** uh dimension. So now now once we look at this the sequence input shape it is 4 53. Okay. So now let's proceed. Now let's look into the idea of RNN. Okay. Now how do we tackle this problem of uh uh understanding sequences? Now we use the first one which is the recursive neural networks RNN. So now we use NN. RNN. So in going ahead and defining it. Okay. Now since I'm running it for the

**[03:28]** first time, I should have uh Okay, since it's a sequential thing should have gone there and got the Py basics. So I would have gone here and uh ran this first which is uh all the libraries and then obtain the obtain the GPU since I'm doing it for the first time so that I can uh continue

**[03:58]** from there. Okay, these are the things that we already have seen. Okay. Yeah. We are here. We just ran

**[04:29]** this. We have created a tensor. So now let's look into the uh an RNN. Okay. So the input size know that is each XT what is the uh so the RNN will go ahead and uh what do you call process each time stamp the input of each time stamp t at a time. So each of them is of size three and then that you are transforming it into an eight

**[04:56]** dimensional space. So like that I have how many layers of RNN you have? you are having only one layer of RNN. Now always whenever any transformation happens the first indices that is the zeroth. So when you use that shape dot shape the first one has to be batch. Now that is what this batch first actually means. Okay. So now let's look at how does this RNN work. Okay. So now let's look at this uh

**[05:24]** recursive neural networks RNN there. Now uh X T which is there see I'm not using any superscript so just to indicate that anything you can look at it X is in R3 and any H the hidden state that you have will be in R8. Okay this is what I'm saying. So how does this work? So you have ht is equal to now

**[05:54]** preferably we use tan h tan h of weight that is associated with x into x at that specific instance plus weight that is associated with respect to the h which is the hidden state into the the previous hidden state plus a bias term. Okay. So now now see now since I have fixed HD to be R8 now I have to perform this operation in

**[06:24]** such a way that the output now has to be a vector of size 8. Okay. So now let's see that how does it work into picture. So now this uh uh you have uh this XT. So each XT is of size uh 3 + 1. See this you have to transform. See whenever you are adding all the three terms should be of the same dimension. So this wx will be of size 8

**[06:53]** + 3. The ht minus one will already be of size 8 + 1. Okay. So now this will be of size 8 + 8. Now this will be a 8 length bias. So now the outcome will be this is 8 + 1 plus this is 8 + 1 plus this is a 8 length vector. So now you can go ahead with that and then tan h is uh performed

**[07:25]** individually on each of them. So therefore the hidden state the updated hidden state that you have ht is coming to be an eightlength vector. Now this is what is being done in this uh specific RNN cell. So now then what happens here? Now you can uh remember now if you have an RNN cell okay now you get H0 now which belongs to which will be a

**[07:54]** zero to that you give X1. Okay now you give X1 now that will give you H1 and then again you use the same RN and cell. What do you mean by use the same RN and cell? Now I mean that use the same weights and biases. Okay. Now then the same iron and cell now you give x2 this will give you know h2. So so on and so forth you repeat the same thing and finally you have u

**[08:23]** h uh n minus one go here you get uh h xn. Okay. Now this is what the whole processing is. uh see normally what happens is at an each cell you can uh get an output. Now how do we uh do that in the cases is now y at any instant t. Now what do we do is now we use a weight

**[08:54]** why okay so to that ht plus b y now we do this operation so that at each output now this you get y1 this you get y2 similarly you get here yn now this is how the standard uh rnn will work okay that you can See here now you are going ahead with four 53. So four is the batch

**[09:24]** five is the number of sequences three is the size of each one of them. So the output okay so this is y. So now now whenever you this returns two things as you can see here one is the output the other is hidden. Now output what is this? When I mean output in this case, what does it mean is you get y1 no y2 so on until yn. This is what will be the output. And when we say

**[09:52]** hidden now that will be h1 h2 so on till hn. Now this is what you will get as the output and hidden. So now we can look at the shape of X's 453 and then Y will be for each of those four examples for each of those sequences. So you will have five sequence in each example you get a eight length output. This is your Y now

**[10:24]** which is this output and then this hidden now which is there for each of the four examples you get the eighth length vector. Okay. Now this is the the overall way of working. Okay. This is ironn and similarly now let's uh look at the LSTMs. Okay. So now See uh the reason why we needed LSTMs

**[11:01]** was uh the problem of vanishing gradients. So here now we have uh couple of transformations. There is some popular uh version the way in which uh this is being discussed now which is you have this what is known as the the forget gate. You have this uh input gate. We have this uh candidate gate and then we have this uh

**[11:30]** output gate. Now these are the four uh different uh transformations that are there. Okay. See unlike uh RNN now where in which you only get one state as uh the output. Now here now we will get two states as output. Now one is this HT now which is uh called as the the hidden state and then in addition to that we get the CT now which is uh what

**[11:58]** is famously known as the cell state and this is now what is known as the the long-term memory the hidden state it is also called as a short-term memory now that is why the name of this unit now which is there is known as the LS LTM long uh long short-term memory. Okay, LSTM unit. So fine. So now what does this each of these transformations does? Now this is

**[12:27]** in common perspective. What does it does is this forget gate which is there it decides now what old information that is there now you can forget. So the input gate decides what new information needs to be stored and then this candidate cell gate create the new candidate information and output gate will say that now how much new information it's combining the old memory and then the new memory. So this is the colloial way in which they say but we need to look into them much more in a better way as a transformations. So now

**[12:58]** now let's represent this forget gate as F. This is I this is C and this is O and these are for each time stamp. Now what will be the output? Now let's go into that the FT and the input for this. So the input is uh at any given time stamp htus one CT minus one and uh XT. Now these are the inputs that you have. Now

**[13:28]** how do you go ahead and perform this computation of FT. Now we use the sigmoidal activation. So now you can uh think of a similar transformation. I'll not be going ahead and mentioning the size of each and every matrix that are involved weight matrices that are involved. Now similarly the way in which we computed for RNN you can go ahead and compute for LSTMs as well. Okay take it as a small exercise. It will be really helpful now if we can look into it. Okay. So now

**[13:58]** now this is now what is the weight that you are giving it for the input in the forget gate that is what this term is into xt see whenever I use this uh now think of this as the dot product which is there plus now what is the weight that I'm assigning in the forget gate for the hidden state into ht minus one plus what is the bias term that is involved in the

**[14:27]** forget gate. Now similarly the outcome of input gate that is also a sigmoidal thing. So what is the weight that I'm assigning for input X in the input gate into that input XD plus weight associated for the hidden state in the input gate into the previous uh hidden state. Okay. plus the bias that is involved okay sorry the bias that is

**[14:57]** involved in the input gate vi then similarly first we get this CT tilda now for which we use uh tan h activation here what is the weight that is associated with x in this computation of candidate into x3 3 plus what is the weight that is associated with uh uh

**[15:26]** transforming the previous hidden state in the candidate state into the HD minus one plus this BC. Now using this now combining uh the forget gate input gate and the previous contexts we get the current cell state. Now CT is you take this FT go ahead and perform element wise multiplication the hadammad product CT minus one plus it

**[15:57]** into CT tilda see since we are looking at a sigmoidal operation in the forget gate as well as the input gate so you can uh and you can think of sigmoidal operation as some kind of a percentage calculation. Now that is that gives rise to the famous way in which I told forgetting input and all those things. So you you can uh go ahead and look at in a similar fashion. Okay. So now now we got uh FTV we got it we got CTV we got. Now what is the output? Now the

**[16:29]** output computation is uh now OT is equal to some sigmoid into the weight that is associated for the input XT plus the weight that is associated for the hidden state HD minus one plus DO. Okay. and final computation of this HT that you want to send it for the next state now

**[16:59]** which is OT this taking hadamar product of tan h of ct okay so now the way in which we can visualize this now this is uh an uh lsdm unit now you get uh ht minus one and then uh you get uh ct minus one. Okay, this is uh HT,

**[17:31]** this is CT. The input is XT. Normally you can compute YT also when which uh the weight that is associated into HD plus BY. You can go ahead with the linear transformation and then do it. Now this will be the YT. The same set of weights. See now there are lot of weights here that are involved. This is weight. This this this this this now you need to learn there are no

**[18:02]** weights here. Okay. These are the different weights that are involved in one cell. Okay. Now these weights are repeated across different say for example if you want for X1 also you use the same weight for X2 also you use the same weight for X3 also you use the same weight. Okay. So like that you go ahead and use the same weight. Now if you have layers okay if you have multiple layers now then you

**[18:29]** will have different weights for layer layer one will have one weight layer two will have a totally different weight okay so that's how this LSTMs will work okay that's what we are declaring here so now you can see here the input size here is three okay and the hidden state size is eight the HD size is 8 and then the number of layers that I have is one now that means that it's a single cell that I'm having. So, and then batch

**[18:57]** first is true. The first uh uh element that is the shape of zero has to be the batch size. So, now if you have this the input shape, output shape, ped shape and the cell shape. So, you can uh look into it and then exactly in the same way we perform the computation. Now, uh we can perform the thing here. Okay. So now let's look at a another uh uh uh what do you call way in which sequences are

**[19:26]** handled now using what is known as GRUs known as gated recurrent units. Okay, let's look into GRUs uh GRUs, gated recurrent units. Now, these are uh simpler than LSTMs for sure. Okay, but it only uh has one state. So, RNN had only one state. GRU, sorry, RNN had only one state. LSTM had

**[19:56]** two states. GRU has only one state again. Now let's look at how does uh the transformation happens here at GRUs. Okay. Now we have now what is known as the reset gate. These are the famous again similar to what we have in LSTM. These are the terminologies better to look at them as some transformations. So that would be a better way of doing it. So and then we

**[20:24]** have the update gate. This is represented by RT. This update gate is represented by ZT. And then now we have the the final uh outcome which is the hidden state. Now how does the transformation here happens? Now we go ahead with computation of this RT. Now we use the sigmoid the weight that is associated with respect to x for in the reset gate into xt. So now what is the

**[20:56]** input in this case? You have ht minus one and xt. Okay. Now the output will be uh so ht and you can even compute yt similarly. Okay. Fine. So then uh the weight associated with respect to the hidden state in the reset gate into htus 1 plus the bias term br. Now then we go ahead and compute this ZT. Now this is

**[21:26]** again a sigmoid model operation uh w x in the update gate into the input x plus w h z the weight associated with respect to the update gate for the hidden state into htus 1 + bz. Okay. So now then we compute this uh candidate hidden state now which is uh h

**[22:00]** tilda. How do we do it? Now we use uh the tan h operation into w uh xh into xt + w h uh sorry hh okay into htus 1 plus bh oh uh sorry a very small mistake here so

**[22:30]** so whh into this RT which is there taking the hadamad product with HT minus one that is why it is known as the reset gate okay you take the hadamad product of this plus this BH yeah fine and then how do you compute the final hidden state the final hidden state is 1 minus ZT you can as I told you since the ZT is having a sigmoid model activation we can think of it as some kind of a percentage so

**[22:58]** taking the hammer product of HD tilda plus ZT taking the hard part of HT minus one. This is how we compute uh HD and then you can compute YT now which is uh so YT is w uh H Y or Y H. So, so we have

**[23:24]** been using first we represent uh so y h uh that is ht + dy. So the way in which it it looks similar to the ironn unit. So you get ht minus one and then you get ht as an output whenever you have given xt and similarly you get yt here. Now the weights that are involved

**[23:53]** here are this this these are these will be matrices. Now now these are the weights that are involved. We can perform the operation using these. Okay. Now let's uh look at that a similar fashion. So the input and output will be similar to what we had in the RN. Okay. So now let's use this and uh come up with a simple classifier.

**[24:22]** Okay. Now here the input dimension is a hidden dimension and number of classes. See to perform the classification the way you need a fully connected layer. Okay. From so what do you take? You take the hidden state and then using that hidden state you apply a fully connected layer to the uh the whatever is the outcome of that hidden state at the last instant actually at the last instance. Now if you want it in each of the case for sure you can do it at each instance. So and then you can go ahead and

**[24:53]** transfer it into the space of whatever is our number of classes. Okay. So now LSTM input size is input dimension whatever you have given hidden dimension number of layers I have fixed it as one and then the batch first is true. That means that every first dimension has to be a batch. So LSTM you pass it through LSTM. Now you get three output whenever you do it in LSTM. Why? Now because the it has two states. Now one is the hidden state,

**[25:22]** other is the cell state. Okay, you get the output hidden state and then the cell state. Okay, now then what do you do? You take the last hidden value which is there and then pass that last hidden value into the fully connected layer to get the logits. Okay. So here you are defining uh the instance of the classifier. So input dimension is three, hidden dimension is 16 and then the number of classes is two. Okay. So then you're creating X know which is uh eight

**[25:53]** examples are there. Each example have five sequences and each sequences is of uh input dimension or embedding dimension which is there is three. So when you pass this so you get for each of the eight examples in the batch you get two. Number of classes you have declared it as two. So you get a probability vector of size two. You can use just one and then pass it through sigma. That is also perfectly fine. Okay. So it's up to you to uh way in which you handle this. That's perfectly

**[26:22]** fine. Okay. So now let's take a very simple toy data set and then try to work around with this. So now what is that I'm uh doing is now if the sum of the sequence values is greater than zero then I'll say it as class one. Now else I'll say it as cash zero. This is a simple uh data that I'm taking. Now as we discussed earlier now any custom data set that you want to

**[26:50]** come up with now has to be a child of this data set. Okay. And then you need data loader to go ahead and create it. Fine. So then uh the number of samples I'm having 1,000 samples. The sequence length. Okay. Now for each of the sequence I'm there will be 10 there will be 10 sequences. So in one x there will be 10 of them and each of that x1 x2 x3 will be of dimension three. So that is what you are uh in uh generating it in a

**[27:18]** random way. So number of samples sequence length input dimension. So and then you compute the sum of the first and the second dimension. That means you take the second and the third. So if you have x1 x uh okay so now rather than showing it it is uh better to write it now each xt now which is there has three values now what is he doing is now he's taking

**[27:48]** these two if the sum of these two is greater than zero it's greater than zero now then y is equal to 1 else y is equal to0 this is what is being done. Okay. So now then if the sums are greater than zero now then you go ahead and uh convert it. So this is a relational

**[28:18]** operation. So you'll be convert either it will become one or zero depending on the outcome. Okay. And the length and then the uh get item you have to define it. Anyhow, now this is the data set that is being created. So you have created the data set and then you have to create the data loader. Now the batch I have taken it as 32. You can increase the batch size for sure. No problem. So you can see that so each 32 10 3. Okay, this makes sense. So this is

**[28:50]** our input. Now then create a classifier. Now now the input dimension is three. So each uh xt will be of three dimension. Hidden dimension I'm taking it as 32. So number of classes I'm taking taking it as two. The loss I'm taking it as n. Entropy. Okay. So I'm using atom optimizer with 0001 as the learning rate. Okay. And the same good old story. So you're doing it

**[29:18]** for 10 epochs. You take each batch and do it. Same story. Whatever we did any classification task which is there whatever we did for CNN's or images or MLP the same good old story is what uh we are trying to do it here as you can see here over the epoch we can easily see that the loss value is decreasing and then we can see an increase in the evaluation metric which is the accuracy in this case. Okay. Now this is how we work with the uh sequences. Okay. So now

**[29:51]** let's see how we can lo save the model. Okay. See what do we mean by model is the weights that are there. So whenever we mean by a model we actually mean these weights whatever these weights are there. Now this is what we mean by the model. Now these weights will be stored. Now say that you want to infer sometime later. Now rather than at that point of inference starting Aresh from the random

**[30:22]** initialized weight now you can use these weights and then just do the inference or now you have already trained model on one specific task. Now if it is something similar you can take those pre-trained weights which are there and then only fine-tune it later. Now these are the options now which are there. Now therefore saving a model in a sense saving all these weights is very much important. Now let's see how that can be done. Now we use this torch.save. Now we used all these uh parameters which are

**[30:53]** there in this particular path. LSTM classifier. PTH. PTH is the extension that we will be using. Okay. Now you can specify the path or you can just specify the the just the file name. So here what will happen is in the current working directory now it will be saved. If I run this, you can see here in collab that here you can see this uh being saved all the weights being saved. Okay. Now you can then what you can do is you can create an instance of

**[31:23]** uh the model and then you can load the weights. Now this is what you created an instance. Okay. See whenever you are creating an instance now whatever model you have said save saved with the same parameters. For example here, now this classifier was having 32 as a hidden dimension. Okay. Now you create an instance with 35 will not work or input sequence input dimension is five will not work. Okay. Because there have been incompatibility with the weight matrices during uh its transformations. Okay.

**[31:52]** Those things needs to be taken care. So we have created an instance and then you have ported onto the device and then what did you do? So loaded model dot load state dictionary. So what are you loading? Torch.load from this specific path which has been specified. You you look at this LSTM classifier. PTH file which is there and load it and you port whatever has been these weights that are

**[32:20]** there onto the device. So map location is equal to the device. Okay, this is what is being done. And then loaded model. So you're uh putting it in the evaluation mode. Okay. So this is how you load the model. So yes, model loaded successfully. Okay. So now this is how so it is not specific to the sequences. Now you can do it in a similar fashion for CNN and other things. Okay. Now

**[32:49]** later in the next section of the tutorial now we'll be looking now how we can get these kinds of uh pre-loaded models pre-trained models which are there and we can load them. Okay that we will see it in the next tutorial section. Okay. See till now if we look at go back and look at all the code we have been rewriting the training loops and the evaluation loops for the classification task. So most of the

**[33:16]** times it will be same the whole procedure will be same. Now what do you do? You perform the forward pass get the logits. Now once you have the logits now then what do you do? You compute the loss. Now once you compute the loss you clear off the gradients which are there and then compute the gradients newly for that specific instance for that specific batch and then perform the weight updation. This is the standard thing that we do. So now we can uh make it into a reusable training functions. Now

**[33:46]** this is for one epoch. That means that you have to call this function which is there for how many epochs that you want to go ahead and train. So model, data loader, criteria, optimizer and device. Now these are the standard things that are needed and then the remaining code is exactly the the same. So first you have putting it in the training mode. So and then you have a bookmarkers for total loss correct and total so that you can uh compute the accuracy. So and then you iterate over the data loader you get

**[34:13]** for each batch of inputs and targets you port them onto the device. Okay. And then uh you pass this inputs to this through this model and get the output outputs and targets into the loss function. Get the loss. Clear of the previous gradients. Compute the gradients for this specific batch. Update the weights. So as per the gradients computed and then update the bookmarkers total loss, predictions correct and then total update the

**[34:42]** bookmarkers as discussed earlier. Okay. And then average loss and accuracy are there. No, you can return it at the last. Okay. So similarly you can have it for an evaluation also. So evaluation doesn't need uh uh optimizer. Okay. So model, data loader, criteria and device. So model eval so again bookmarkers so you put all these things in uh with torch.nat no grad so that inside

**[35:10]** whatever is happening inside this you don't need to compute the gradients. So then you [clears throat] uh you iterate over the data loader for each batch. You first port them onto the device, obtain the logits logits and then the targets, pass it through the loss function, get the loss, just keep the loss as the bookmarker, get the predictions, update the correct and total and then finally compute for the whole data uh

**[35:40]** loader which is there for all the examples in the data loader. compute the average loss and then the accuracy and then you can return it. Okay. Now this you can reuse it for any kind of uh classification uh tasks that are there. Okay. So now we'll be using it in our next section of tutorials. Now just to give you a recap on what are the things that we just discussed we in these sections of tutorials starting from

**[36:08]** understanding tensors working with CNN's and then in RNN now we have looked tensor creation and then uh how to work with the different uh uh shapes of the tensors tensor indexing and reshaping so moving a tensor to GPU so automatic uh differentiation and working with gradients so working with n dot modules all It's NN.linear, NN dot con2D, NN RN, NN. LSTM, and NN.GU different things working with linear models loss function

**[36:38]** and different optimizers. We looked at MSSE loss. We looked at cross entropy loss and different optimizers like SGD, ADOM is what we have seen. Similarly, we can look at RMS prop, AdamW and things. Now, creating a custom uh data set or loading the data from the uh existing PyTorch library. We saw that implementation of MLP. Now we saw implementation of CNN's. We saw RNN's, GRO and LSTMs implementation. We saw and

**[37:07]** then we looked at reusable codes for training and evaluation loops. And then we looked at how can you save the trained weights and then how can we load them. Okay. So now the students are ready for advanced deep learning topics. Okay. Now with this we conclude uh today's uh tutorial discussion. I hope you enjoyed and then now you are capable of writing your own uh neural networks using MLP, CRN, CNN's and RNN's. Okay.

**[37:38]** In the next section of the tutorial we'll be looking at how can you load the pre-trained models like VGG like ResNet. So con next. So different models we will see their architecture and then we will see how how we can go ahead and uh load those weights and then tailor them as per our need. Now we'll be working with MRI tumor classification problem and then we'll be going ahead with that. I hope it was informative. Thank you all

**[38:06]** for patiently listening. We'll meet in the next section of the tutorials. Thank you. Bye-bye.
