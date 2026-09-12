# Transcript — Tutorial 4 : CNNs using PyTorch

> **Source:** https://www.youtube.com/watch?v=BhnGtsMwUCU  
> **Channel:** NPTEL - Indian Institute of Science, Bengaluru  
> **Duration:** ~39 min  
> **Note:** Auto-captions cleaned lightly. Minor ASR errors possible.

---

**[00:02]** Hello guys, uh welcome to this uh uh segment of tutorials now where in which we'll be looking into the working of CNN and we'll be looking at uh coming up with a network for a classification task using emnest data set. Okay. So earlier in our uh tutorials we have just discussed [snorts] regarding PyTorch its functionalities. Now creation of data sets, creation of simple MLP layers, the

**[00:31]** forward propagation, the back propagation, how it happens, the automatic differentiations, the gradients, different kinds of loss functions, optimizers, how to bring in together all of these ideas we just discussed in our earlier discussions of uh PyTorch. So now we will continue from the same point. Now here in this uh segment now we will be looking into the convolutional neural networks where which we will be mainly interested in

**[01:01]** know how does the internals of CNN's work. Okay. So now whenever we want to work with images the go-to uh lightweight uh networks that are used are CNN. Nowadays we have vit yes we will look at v sometime uh down the line we will for sure look at vit that's the reason why I use the word lightweight okay now in this we'll be looking at uh CNN's now here now I have

**[01:32]** an input which is a random input I have just taken a random input just for the sake of uh uh convenience as of now so I'm having a batch of eight examples and each example is of size uh 32 32 now this is a so This is the batch size. This is the number of channels in each of the image. Now this is the height and width of each of the ones. Now here so you have got 8 32 32. Okay. So now let's look at how does this convolutions work.

**[02:01]** Okay. So now before that now let's go into this a very nice uh gif. Now this is uh from the Stanford uh CS231 website which is there. Now here uh now we can see that now these are the four two filters are there. Now each filter has uh three components. Now why each filter has three components? The three components uh depends on how many in channels are there. Okay. So and then

**[02:30]** the output the number of channels in the output depends on how many filters are there. Okay. So now you have two filters. Now therefore you have uh uh in the output you have two uh channels. Now you have uh three channels in the input. Therefore in each filter you have three components. So and then at a time you are looking at a 3 +3 segment. Okay. Now this is looking at this 3 + 3 segment. This 3 + 3 segment and this 3 + 3 segment is what is being looked into.

**[02:59]** Okay. Now what do you do? You perform element wise multiplication. Add all of them. So you get a scalar from this operation. You get a scalar from this operation. You get a scalar from this operation through three scalers. So you add all three of them plus one uh this scalar. So that will give you this minus one. Okay. Now you can see zeros here. This is uh the padding. You have go gone ahead and went with zero padding. Now you have done a padding of one layer. Therefore you can see now the repetition

**[03:29]** of zeros at once. There are multiple kinds of padding but most preferably the one that is being used is zero padding. Okay. Now before padding the each of the image size is five 5 + 5. Now you have three channels. Now therefore the prepadded image size is 5 + 5 + 3. The post padding it is 7 + 7 + 3. Okay. So now let's uh make a note of that. So now now here now we can see that see the x which is

**[04:01]** uh the input that I have is a 5 cross 5. This is a 5 cross 5 cross 3. This is the height in width in this is d in. Okay. So and then you have uh padding you had done one so one dimension of

**[04:28]** padding is what you have done and then you have stride that means that after each operation you how much you are moving okay you're moving by two now as you can see from this example very clearly now you're here okay in this point once you go ahead you'll be going with this this segment of the image okay now as you can see here you have moved by uh two in x-axis and the same thing will be moved preferably we will use the

**[04:56]** same stride in xaxx and y axis. Okay. So stride is two and then so you have the kernel size. Okay. So the kernel size which is referred by f which is 3 + 3. Now most of the times we use a square kernel. Okay. Uh unless specified we can use rectangular kernels as well. That's perfectly fine. And then how many filters are there? You have two filters.

**[05:25]** Okay. So now now you can think of it like this. Now how do you compute the number of weights in this case? Okay. You have 3 + 3 that is repeated thrice because your input channel has this is uh the kernel size. This is your uh D in this is your kernel size. Okay, this is repeated uh twice. This is

**[05:54]** your uh k which is number of filters that are there. So 3 + 3 cross 3 into two and for each of this this plus two. Okay. Now this is these are the number of weights that are there that needs to be learned in this particular convolutional operation. See there is another way of looking at this. You can look at CNN's as uh a regularized MLPS now where in which you have weight

**[06:23]** sharing. Okay. And then local receptive fields. Okay. Now that is another way of looking at uh CNN you can think of them there is weight sharing. Now that means that for all the chunks you are using. Okay. Now if I have a let's say that I have four nodes here and then have four nodes here. Now you can say that okay now this is the idea of local receptive field. Now this

**[06:52]** particular neuron that is there is only looking at these two locally. This is only looking at these two. Okay. This is the idea of local receptive fields. On the other hand, now if you say that now this the weight which is there here this weight is same as this weight. Okay. And this weight is same as this weight. Now this is what is known as weight sharing. Now this can be looked at as regularizers. Okay. Now we can look at the CNN as a regularized MLP wherein which we have

**[07:22]** the local receptive fields and then weight sharing. Okay. But that apart. Now these are the number of weights that you need to look into. So which are there. Now given this uh input uh dimension and these hyperparameters. These these are hyperparameters. So so in MLP we have how many number of layers should be there. What should be the number of nodes in each of those layers and other things. These are hyperparameters that are there. Now here

**[07:50]** now we can compute the output after this. The output will be so H out cross W out cross D out. Now this is the the output size which is there. Now how do you compute this given this? Now we can compute this. Now this uh H out equals H in minus F + 2 P divided by S + 1. Okay. Now similarly

**[08:23]** W out equals W in min - F + 2 P divided by S + 1 and then uh D out will be K. Now if I substitute this HN is so you see here hn in the sense now you should consider the prepadded things. So this is 5 - 3 + 2. Okay, this is 4 and this is 2. Now therefore this will be three. This

**[08:52]** will also be three. This is two. So therefore the output will be 3 + 3 + 2. Now using this we will be computing the output and the so so the given the input and these parameters that are there now we can compute the output. Now let's verify now whether we have the same yes you can say this 3 + 3 +2 is what you get okay so now that means that whenever you want to specify a convolutional

**[09:20]** operation these are the things that you should specify how many channels are there in the previous thing how many channels how many filters are there which is nothing but how many channels are there in the output now what is the size of each of these filter or kernel and then what is the stride and what is the padding now if you specify this now you have specified the a convolutional operation Okay. So now let's do that. So how do we do it? We use this what is known as nn.com2d. Okay. So in channels is three. How many

**[09:49]** input channels are there? Out channels is 16. Okay. Now these are you can uh go ahead with that. And the kernel size is three. So that means that you have a 3 + 3 kernel. And then since there are uh three channels in the input. So each filter will have 3 + 3 + 3. Each filter will have 27 parameters and then since you have 16 channels say 16 filters like that. Now you'll have 27 into 16 plus 16 for bias terms. These are the number of

**[10:18]** parameters and then the stride is one and the padding is one. So now let's taking this let's compute the output size. So now now the hin that we have so the the input hin into w into din. So this is 32 cross 32 + 3 is what we have and then what is the padding that we have? We have one stride is 1. The filter size is three. K is 16.

**[10:48]** So now what will be the output size? So we want to compute the output size. Now D out is fixed. This is 16. Okay, this is of no problem. So now h out equals H in which is 32 minus F is 3 plus 2. Wait a minute. I think uh straight is one pad is one. Okay.

**[11:27]** By s is 1 + 1. Okay. Now this is 29 + 2 is 31. 31 + 1 is 32. Now the output will be so this is post the you have a black box which is uh con 2D the output will be 32 + 32 + 16 and this will be the the output okay now think of this now once you perform this operation you can check for the shapes you can see that now each of this

**[11:56]** example will be converted the 16 is the number of channels 32 + 32 like that you have in a batch you had eight. Okay. Now always whenever we are performing the operation we'll be considering the output with respect to one example. Okay. Now this this is the first is the batch size. So you have uh to keep it as it is. Now this three we had taken it as the last and you can permute it perfectly fine. Now this is the first con the uh operation of convolutions. So these

**[12:24]** things you have to specify. you have to specify in channels, out channels, kernel size, stride and padding. Okay. And then another uh important operation that we normally use whenever we look into the CNN is the idea of max pooling. See the idea of max pooling is uh uh to reduce the dimension. Okay. And whenever the max pooling is used with max pooling 2D that is you take a uh 2D as an input kernel is two and stride is equal to

**[12:54]** two. Now what happens is whenever you can uh do this case here. Okay. the max pooling with kernel size which is uh in our case which is which we are referring f. Now if kernel size is two and stride is two. Now how do you compute the output? So h

**[13:24]** in min - f + 2 p divided by s + 1. This is how we do it, right? Okay. Now, what happens is this will go off. Now, there is no padding. So, therefore, this will be zero. Okay. Now, this is h minus 2 / 2 + 1. Now, if this is 32, okay, 32 - 2 is 30. 30 divided by 2 is

**[13:54]** 15. 15 + 1 will be 16. So the input which was uh 32 is uh becoming 16. Now similarly for W also. So if you have an input which is 32 + 32 post performing uh max pooling. Now whenever I say max pooling I'm specifically saying 2 + 2 okay where in which the kernel size is 2 +2 and then the stride is two. In this case now what will happen is it will

**[14:21]** become 16 + 16. Now if you have different uh options you can go ahead and use it. Okay. So you can make stride is one and kernel is three. That's perfectly fine. Now accordingly the shapes can be modified. But whenever we use this is the standard configuration which we use it the size will be uh reduced by half. Okay. So now uh here this is how you create an instance of this max pooling. So nn dot maxpool 2D the kernel size is two. The

**[14:51]** stride is two. There are no parameters involved here. Okay. So in con 2D you had parameters involved but in max pooling there are no parameters. It is just reducing the size by in that 2 +2 kernel. It is taking the the maximum uh element which is there. Now which you can uh understand it by this example. Now this is you take a 2 +2 kernel in this case. What is the maximum element? The maximum element is six. You get this six. Here the maximum element is eight. You get this eight. Now here the three

**[15:19]** and four accordingly. So what was 4 + 4 has been reduced into 2 +2. Okay. So now uh so x dot shape what was 16 32 32 and then please remember now it will not do anything for the number of channels. Okay. Now it is just out of sheer coincidence that this 16 you have three 16s. Now if your input was uh some other thing let's say that if input was 40 so

**[15:47]** then it would have been 2020 in this case. Okay. It is sheer coincidence. There is no rule that these things has to be matched. Okay. Fine. So now let's take a CNN example. Now here we are going ahead with the CNN which is uh 32 + 32 and RGB image. Now that means that it has uh uh three channels. Okay. The input is a 32 + 32 and then it has three channels. Okay. Now therefore now let's look at this

**[16:17]** this convolution one. Now the input is uh a three channel that you are converting into a 16 channel the kernel is three padding is one if you haven't specified the stride now by default the stride will be one okay if you haven't specified the padding by default the padding uh will be zero there will be no padding okay and then you have another convolution and then you have a pooling see here in the init method so this is a simple CN it has to be a child of nn dot module the same story that we did for

**[16:45]** MLP holds good perfectly well Okay. So the number of classes you have 10. So now after that you have fully connected layer now which is this is the input. This is the output 28. So this is 32 into 8 into 8 uh nodes to 128 to 128 to number of classes. Now let's compute this. How is this happening? Now the first x now what is the input of this? This.

**[17:14]** So the the x that you have the input is uh it is 3 cross okay so it is 3 cross 32 cross 32 okay this is the so this you are passing it through the com 2d which is uh having parameters 3 16 kernel is three padding is one stride is one. Now this

**[17:46]** is uh in channels out channels. This is K. This is F. Okay. So now then now we can perform the operation. Uh so what will be the output of this? So the output of this will be 16 cross 16. Why 16? Because your K is 16. Therefore the number of channels output will be 16. So this is uh 32 - 3 okay +

**[18:21]** divided by 1 + 1. So this is 32 - 3 is 29. 29 + 2 is 31. 31 + 1 is 32. This is 32 + 32. Okay. This will be the the the output size of this con 2D. this which you have okay see you apply relu see whenever you are applying relu see ru is applied individually so it will not change the dimension so I keep it as it

**[18:48]** is that you are passing it through a max pool so when what is the max pool kernel size is two and stride is two now therefore whenever you pass it through the the max pool the output will be 16 + 16 + 16 okay this is the output now Then then you're passing it through a convolution 2D sorry convol second convolution which is 32 16 31 okay now

**[19:18]** this you are passing it through another uh con 2D now where in which 16 is the input k is 32 f is 3 p is 1 s is 1 so in this case now what it will be So the output will be uh 32 is fixed. Okay. Now this is 16 - 3 + 2 divided by 1 + 1.

**[19:50]** This is come to 16. So therefore this is 32 + 16 + 16. Okay. And then whatever you have this you pass through the the max pool. Now once you pass through the max pool it is 32 + 8 + 8. This is the output that you have got post your operations of CNN. Okay. So now so whenever you perform this two

**[20:18]** convolutions now you would have 32 + 8 + 8. Now what do you do? You convert it into a vector. Now you have to take a decision. Okay. Whenever you want to take a decision, so the output is uh a vector. Okay. Now you have to convert whatever the features that you have extracted into a vector. Now whenever you convert this into a vector, see the size will be the number of nodes here. Now will be 32 + 8 + 8.

**[20:48]** Okay. Now this has to be converted into a we use a fully connected layer. So this is what is known as flattening X do you take this whatever is the size zero of it and now you convert it you convert it into a vector okay now then 3288 to 128. Now that means that here you have 128 nodes and then

**[21:19]** 128 to number of classes. Okay. Now the number of classes I have said as 10. So it will be 10 here. Now you pass through this and then this output that you get here. Now this is uh logits. Now finally you can pass it through the soft max to get probability of y given x is equal to x. Now this is in this is a probability vector which is in

**[21:49]** R10. So where in which the first element says that what is the probability that y=0 given x= x till probability that y = 9 given x= x. Now then what do you do? You take the argmax of this that will be the decision which is nothing but your y hat or uh y prediction. Okay, this is the

**[22:21]** overall full story. So that is what is being done here. Okay, now we can see this shapes. You have created an instance of it with 10 classes. So you have created an example with four as the batch size and this is the three dimensions for each example RGB 32 + 32. So this dummy input you are giving it through the output. Okay. After the convolution the first convolution the output is 164. This four

**[22:49]** is the batch size has to be there as it is. Now after the the first convolution plus pooling. So the first convolution plus pooling. So that means that this is the output that I got. Okay. 16 + 16 + 16. Now four is the bat size. So it is matching. And then the second convolutions and then the pooling which is uh this. Oh sorry no not this [clears throat] which is which is after convolution.

**[23:17]** This is after pooling which is 3288 is what I got. You can see 3288 and then once you flatten it out 3288 will become a vector of 2048. that 2048. Now this is uh 2048. Now whatever is in 2048 dimension you have uh taking it into a 128 dimension and then you're putting it into 10 dimension. So which is your logits in this case. Okay. So which is four all

**[23:49]** the four four examples 10. So how it will be? Now you'll have uh the first example the zeroth example you'll get a probability vector. The first example you'll get another prob. The second example had another probability vector. The oh sorry the third example. The third example you get another probability vector. Okay. Now each one of this will have 10 elements.

**[24:25]** This is a 4 + 10. Okay. Now this this is the same thing without uh what do you call any prints so that we can use it. Okay. So now let's look at how do you get the data set. Now here we are going into the case where in which the data set that we are considering is present in the PyTorch repository itself. So now how do you get it? Now you want to use this uh torch vision. This is the thing. And then whatever input that you

**[24:55]** have got needs to be transformed. Okay. So now what do you mean by transformation? See the input that you get so maybe in the format of numpy or so that needs to be converted into a tensor. Okay. Sometimes you might want to resize it. Sometime you want to introduce couple of other regularization ideas. Now which uh we shall be looking at somewhere down the line. What are the different regularization ideas that uh we can think into. Okay. So now those are the

**[25:24]** transforms. So you compose you bring in all the com transformations together. Now first you are converting it into a 32 + 32. See mnest which is there is 28 + 28. You are just extrapolating it into 32 + 32. Now the reason uh why I'm doing it is see initially I wanted to show it on a three channel image which is C4 okay it took exactly too much of time to download so I just transformed the same data which is there into a do it for

**[25:53]** mnest so mnest is easy for us to download see going on most of our generative uh models now we will be using emnest only for two reasons specifically now one is it is of small size so another is so you'll not burn too many GPUs. Okay. Now this Google collab which we will be using throughout. Okay. Now it allocates certain amount of uh GPU for you. So we actually don't want to uh use it too

**[26:21]** much. Okay. For that specific reason what we shall be doing is we'll be going ahead and using emnest for most of our uh thing. Uh somewhere down the line we'll be using some small uh emojis data set also which we will look at sometime later. Perfectly fine. So once you have resized it that you are converting it to a tensor that's what line number 10 is doing. Okay. Now how do you get the data set from tors vision data sets you're getting the mnest. Now where is this data? So okay now you are storing this

**[26:50]** data in this from the current working directory in data folder. See sometimes you might have the data already ready with you. Okay in that case you don't need to download it. Now else you might need to download it. Okay. So now you have this uh root in this place. Now you are only getting the training data. That is why the train is true. And then download is equal to true. That means that in the specified location if the needed data is not there, you can go

**[27:18]** ahead and download it. That's what is being done here. Okay. And then post that you apply this transformation which is there. Now what is the transformation? You convert it into a 32 + 32 size and then convert it into a tensor. Okay. This is your training data. The same thing with your test data as well. Now you will put it into the same folder. Training is false. Now indicating that we are looking at the testing data and then download is equal to true. That means that in the

**[27:44]** specified location if the data is not there you go ahead and download it. Now you apply the similar transforms. Okay. See most of the times uh so we will have a similar transforms for uh training and uh test. Now one thing that needs to be taken into consideration in both the cases is the size. Now whatever is the size of your input data the same thing has to be the size of your test data as well. Okay. Train and test data should be of the same size. Okay. So then you are creating the data loader.

**[28:13]** Now this is data set is done. Okay. Now you're creating a data loader using this data set. Now when which it's an accumulation of all the data. So now the batch size is 64. Okay. In each batch you will be going ahead with 64 examples and then the shuffle is true. The same thing with the test. See in the test you can have a different batch size that perfectly fine. Okay in the test you don't need to shuffle because you'll be running it only once. Okay to the case. And then how many train data set classes

**[28:42]** now will give us the total number of uh classes that are there. Okay. So and then what is the length of training data? you have 60,000 examples in the test data you have 10,000 examples okay now 012 these are the different classes that are there okay this is regarding the mnest data see we can uh use any other data set as well you can use FM nest you can use cf FM nest is easy to download but cf it will take some time

**[29:10]** even though if it is around 170 MB so somehow it took lot of time okay so now let's visualize some of them okay I'll not go ahead and expl explain this visualization plot. Now these are the images that are there. Okay. Now this see this is coming in color. Now the reason why it is coming in color is I haven't transformed it to act take a gr grayscale image. You can transform it but the input is grayscale. Okay. The

**[29:37]** input is not a threechedled image. Okay. What happens is this mattplot li which is there it will automatically convert any gray grayscale image as an input. it will automatically convert it into a three channel thing. It will just repeat the same thing for the remaining two channels. Okay, that's what it does unless specified. Okay, you have to specify don't do it. Okay, which I haven't done in this case as why the reason as I told you my initial aim was to work with CR10 somehow it did not download. Okay, fine. So now let's go

**[30:08]** into the training aspect. So the model you create this uh uh instance. So the number of classes that you have is 10. So and then you are porting it onto the device. So you're using cross entropy loss. The optimizer that you are using is Adam. So with the learning rate of uh 0001. Okay. The same old story. The training procedure is exactly the same thing. You're doing it for five epochs.

**[30:35]** For each epoch you put it the model in the training mode. Running loss correct and total you compute exactly the same story that we used earlier for binary classification. So but there what you did was all the examples were taken at once. Okay. But here what happens is you don't you cannot take all the examples at once. You have divided them into batches. Now therefore you have to do it batch wise. So therefore you have to iterate over the train loader. Now at

**[31:05]** each time it will give you the images and their respective labels. You convert them onto the uh you port them onto the device. Perform the forward pass. Compute the loss. Optimizer.0 grad. Okay. Clear off the previous gradients. So loss dot backward compute the gradients for the new batch. Optimizer.step. Based on those gradients, go ahead and update the

**[31:33]** weights. So, and then running loss prediction everything the same old story. So see I I just want you to bring it into your uh notice that in one epoch there can be multiple updations of weight. Okay. So for the first batch you take some W and then you update it. For the next batch you'll be using this updated weights.

**[32:00]** Okay. So it's not like earlier examples where in which for each epoch your weights will be updated only once. So in this case for each batch there will be a weight update. So please keep it clear. Okay for each batch there will be a weight update which will be considered. Okay. So so that means that the number of weight updates is now how many batches are there into how many uh what do you call epochs were done. Okay epoch whenever we mean epoch it is run through

**[32:30]** the whole data once. Okay that's what we mean by epoch. You run through the whole data. So whenever I mean whole data the whole of training data the whole of 60,000 samples that we have whenever we mean iteration now we mean that it is updation of weights once okay fine so you just do it so post five epochs you're getting 99% accuracy see now this uh

**[32:58]** emnest is a beaten data okay so you you will find it for every example It's like first see whenever you take a calculator no normally what you do so even though you know the outcome so you just try it whether the calculator is working or not you just put some 2+ 2 or 1 + 1 or 1 plus 0 depending on your favorite number you just put that no so mnest is like that so it's for sanity check if your data is working well on mnest sorry if your model is working

**[33:26]** well on mnest data so that means that it's working okay it's a sanity check if it is working it's working perfectly fine No problem. Okay. Now then you can think of okay extending okay know can I go into another data set that's how all of us work first we try on mnest data now because of high availability of GPUs people have stopped it but the main usage is first you try it on mnest if it is working on mnest okay you can try it on others okay that is a bare minimum it

**[33:55]** should work on mnest fine now this is training aspect now how do you perform the evaluation see now please uh remember this see the evaluation metrics that are there. Now how do you evaluate a model so in the regression tasks that uh we'll be doing the loss function can also be used as an evaluation metric. For example, the loss that you'll be using which is MSSE loss can also be uh

**[34:24]** thought as an evaluation matrix. But in classification kind of tasks that is not true. Okay. So the evaluation metrics that will be used is accuracy. But whenever you have skewed data now we will go into other metrics like precision, recall, F1 score and stuff. So I'm assuming that you're quite aware of all these different uh evaluation metrics. Now uh else you can find ample

**[34:54]** resources available online. Okay. So fine the evaluation metrics since it is the eminous data that we are using which is a balanced data. What do I mean by balance? The each class has same number of examples. Now we will not be going ahead with precision recall. No we'll be just going ahead with the standard accuracy. So to compute accuracy what is accuracy total correctly classified examples divide by total number of examples. So that for the test set which is there you should have a counter you

**[35:22]** know where in which you keep track of all the correct examples and then you go into how many number of total examples are there. So now here we go ahead with evaluation mode. Okay. So earlier we went with the training mode. So this is the evaluation mode. So whenever we are going ahead with evaluation mode, we will not be going ahead with any kind of gradient tracking or gradient computation. You don't need any of them. So you have the fixed weights that are there. You just perform the forward pass, take the decision and then see what is the uh total uh what do you call

**[35:54]** uh uh examples that are correctly classified and report the same. That's the idea. So so you have this bookmarkers correct and total for the same. So with torch.nd No grad. So that means whatever is inside this block, you will not be going ahead with any kind of gradient computation. See again so you don't need to do it for multiple epochs not needed. Okay, evaluation is only done on the test set once. Now that is the reason why while creating the data loaders you did not give shuffle is equal to true. You can give okay we made

**[36:22]** shuffle is equal to false. It's perfectly fine but not necessary. Okay. So images so always first you have to port it onto the device. So perform the forward pass you get the logits. So take the argmax of those logits in dimension one that is with respect to column. Now y now again see here. Now now you want to go ahead and uh find the arg max with respect to this for the elements of this

**[36:51]** column. Right? Therefore you go ahead with dimension is equal to one. So whatever is the arg max so that will so your y prediction will be like some four okay so it will be sorry so let me write it now so it will be this is for the first example second example third example and fourth example so you get like that and then so this is also a row ve prediction labels this is

**[37:21]** also a row vector so you find what are how many examples are there equal and then you take the sum of it convert it into item the standard thing and then uh you go ahead the test accuracy that you have got is uh 98 okay now please remember that the training accuracy and test accuracy uh can vary okay so I'm assuming that all of you are very much aware of the training loss the test loss the training accuracy training metric and then the

**[37:49]** test metrics and other things okay so now this is uh regarding uh CNN's. Now this is where we uh stop uh at this moment. Now to give you a recap, what did we look at in this uh section of tutorials? We looked at the convolutional neural networks, how they work, how the the computation of convolutions happens. How do you given an input size and then the hyperparameters of that specific

**[38:16]** convolutional operation, how do you go ahead and compute the output size? And then we put all of them together. the feature extractor which is convolutional networks and then the classifier head know which is there we put them together to take a decision. Okay. Now this is what we have uh discussed uh till this point. Now in the next section of uh this tutorial sequence now we will be looking at how do you work with sequence. Okay. So there we'll be looking at RNN's, LSTMs and GRUs. And

**[38:48]** then we will be coming up with an RNN and LSTM classifiers. So uh that is where uh uh then the the next batch of tutorials uh will end and then after that we will see how can we load the pre-trained models and then how can you work with the pre-trained uh models and how can you modify. So we will take an example in which we take a pre-trained imagenet example and then we will work on MRI data and we will see that example down the line. Now with this we conclude

**[39:17]** today's uh tutorial. I hope it was helpful. You enjoyed it. So meet you in the next section where in which we'll be looking at sequences. Thank you. Bye.
