# Transcript — Tutorial 3 : PyTorch Basics

> **Source:** https://www.youtube.com/watch?v=SEtu7Eef5ps  
> **Channel:** NPTEL - Indian Institute of Science, Bengaluru  
> **Duration:** ~62 min  
> **Note:** Auto-captions cleaned lightly. Minor ASR errors possible.

---

**[00:04]** Hello guys, uh welcome to the next bunch of tutorials where in which we will be continuing our discussion. So in the earlier uh bunch of tutorials, we looked at two important things. Now one is we understood the basic constructs of Python and then we looked at implementing the ideas of uh basic uh regression and then CNN's and stuff using numpy. Now we shall move into the a very

**[00:35]** important library which is pytorch. These sets of tutorials are uh designed uh in a way. First we will understand the basic functionalities of PyTorch and then we will uh go ahead and uh implement the basic uh models like uh linear models. So and then MLPs and then we will implement CNN's RNN LSTMs and

**[01:07]** stuff and then in the next section we'll be looking into the idea of loading the pre-trained models and how you can modify the networks for your downstream tasks. This is how we have structured uh this whole section of tutorials where in which we will have multiple tutorials. So in that as the first one let's proceed with understanding the basics of pytorch.

**[01:37]** Now couple of uh important libraries that are needed while going ahead and implementing uh pytorch as you can see here. One is the base library which is which is torch. Now this is in line number five which you can see is import torch. Now this is the base library. And then you want to go ahead and uh work with the neural networks. Okay. So you want to go ahead and uh declare or you want to define a CNN or if you want to define a transformer model. So then now

**[02:08]** we go ahead with this NN and then so different kind of optimizers you want to uh use then we need this torch.optim Optim as optim and then if you want to go ahead with all these uh activation functions then uh we will be using this functional torch nn functional so as f. Okay. Now what is the version that we are using? So if you want to look at the version it is the version that we are

**[02:37]** currently using is 2110. This is the version that we are using. And whenever we are using PyTorch, another important thing that needs to be noted is the availability of GPU. Now we all know that the reason why we don't use uh lists or numpy array is they are designed to work on CPU. we

**[03:08]** need uh data structures that are designed to work on GPUs, accelerator devices. So now uh you might have uh GPU cards if not we can use uh freely available cards from multiple sources. Now one such source is Google Collab. Now you can go to this runtime here and then you can look at here and click on this change runtime. Now you have CPU which is the standard uh thing that is available or you can

**[03:36]** use this hardware accelerator T4 GPU or you can use this TPU. Okay, preferably for most of our working we'll be using this T4 GPU which doesn't uh cost us anything but it is uh allocated based on the availability and your usage for sure. So now you can save this if it is CPU you can make it as GPU and then you can save this. Okay. Now how do you check the availability of GPU? So now here torch device. Now CUDA if

**[04:07]** torch.cuda is available. Now this will check whether any accelerator is available. Okay. So else if the accelerator is not available we'll be using the CPU. Now when I print it. Okay. Now we will be able to see the device. Now as of now we have uh uh using the CUDA device. So you can see it here using device CUDA. Okay. We are using Tesla T4 cards. Okay. So you can get the names of it. Okay. So now uh so just because we are coding in uh uh

**[04:36]** Google Collab that doesn't mean that every time the GPU has to be enabled. Okay. Now the way in which normally you can do is you can write every of your code initially uh do a sanity run and then you can enable the CPU. So you can disable the CPU and enable the the GPU and then you can rerun it so that the allocated time uh will be reduced. Okay. So now now similar to what we saw in numpy now let's create some tensors. Now if you have a list now you can go

**[05:07]** ahead with torch.tensor and then you can uh create from a list you can create a tensor torch dot tensor for this. Now this is your y. Similarly you can create a matrix so on and so forth. Okay this is uh from a list. Okay. Yeah. Now what are the attributes? the attributes that are available are now first here I'm creating a torch randen okay uh so here I'm sampling from

**[05:39]** the normal distribution so I'm creating a tensor which is of shape 3 + 4 okay this is what you are specifying here you are specifying the shape of it okay so now then I'm printing the tensor now x dot so that x is the tensor now it has attributes like shape what is the data type and where is it? So whenever any given uh tensor is created now it will be by default will be on the CPU. You have to explicitly port it onto the GPU.

**[06:06]** Now down the line now we will see how to do so. So that is the reason you can see it here whenever I check the attribute X dot device. Now you can see in the output that it is there on the CPU. Okay. Now data type now what is the shape and then the so you can see that now this shape which is there is in you can get what is shape of zero what is shape of one now you can get it and then know what is the the how many rows are

**[06:34]** there how many columns are there you can fetch it like that okay now this is uh a random so sometimes we might need some special tensors okay sometimes you might want all the elements of the tensors to a zero. Sometimes you want them all once. Sometimes you want it from a specific distribution. So let's say that here you want it from a uniform distribution or from a normal distribution or you want your tensor to be an identity matrix. So like this

**[07:02]** depending on the need we might need to create some special tensors and then we have uh uh the necessary ways of doing it. Now if you want all zeros now torch.zeros you specify the shape. Now in what shape you want in 3 + 4 you specify it. So once if you want all the ones you specify 2 + 5 here. Now if it is uniform okay uniform 01 so you can use just dot dot rand 3 + 3 the 3 + 3

**[07:30]** sensor tensor now where in which each of the uh value is sampled from a uniform distribution. Now this is a normal distribution sampled from a normal distribution. Similarly and then you have an identity matrix. So identity matrix of size 4 + 4. Okay, identity matrix is only defined for a square matrix. So therefore specifying one of the shape is more than enough. So once I run this now you can see that all zeros which is of shape 3 + 4 whatever you

**[07:57]** have defined here all ones just 2 + 5. So a 3 + 3 uniform tensor a 3 + 3 normal tensor and the the four uh uh 4 + 4 identity matrix. Okay. Now this is how you can create the tensor of necessary uh details. Now then indexing of a tensor. See indexing is similar to what we looked in the list or in numpy array that we have. Okay. Now if you

**[08:26]** have a tensor x which has been created from this list. Okay. Now you can go ahead with accessing uh the zeroth element or if you want the last element or you can slice it. The same same old things are available. Now you can uh uh look into it. The same thing works with a 2D tensor as well. Now you have a matrix. This is a matrix and then uh you can access a specific element. You can access the first row,

**[08:55]** first column. Okay. And you can take a specific suction of a matrix as we have done this. This example is similar to what we saw in numpy but using tensor. There we used numpy nd array. Here we are going ahead and using a tensor. That's a major difference. Okay. So now now let's look into the a very important uh feature that is needed for us is I have a tensor of some specific uh uh

**[09:24]** shape. Now I want to change the shape of the tensor. Okay. Now you can take of a straightforward example that you have extracted the feature in a CNN and then you want to flatten it out. No for uh going ahead and putting into a classification head or you have uh used an MLP you want to resize it into a matrix. So depending on variety of applications that you have you can go ahead. Now I have a tensor which is torch.range 12. Now this gives

**[09:54]** me from 0 to 11. So this can be used even for if you want to create some kind of indics. So it can be used this functionality. So torch dot arrange of 12. So the original tensor which is there x and then the shape of it is it's a uh it has 12 elements. Now I'm reshaping it to 3 + 4. Now please remember that you cannot reshape it to 3 + 5. You don't have such elements. So the the it has to be compatible. Now that

**[10:21]** compatibility you should be looking into. So then because of that you have reshaped it. So the first four elements 0 1 2 3 will becomes the row. So it goes on filling row wise. Okay. So therefore whatever you have you have uh converted it into a 3 +4 matrix. So what was elements which are arranged you have converted into a 3 +4 matrix. So

**[10:48]** now sometimes uh we might uh want to do it in some other way. So now this is one way. Reshape is one way. Now we have another way which is view. Okay. So normally you can use either of them. The main difference is uh in terms of memory allocation. See view needs uh the memory to be allocated in a continuous manner but reshape needs

**[11:17]** a non-ontinuous memory allocation works well. But going ahead going out of those nuances now both of them perform the same thing. Okay, you have till 12. Either you can give X dot view and reshape it to 3 + 4 or you can use X dot reshape. Both of them perform the similar operation but only uh what do you call the devil lies in the details. Okay. Now you can go ahead with that. Now, now this is you have a uh some

**[11:48]** numbers. So you want to reshape it. Now sometimes you want to add a dimension. Okay, see if you look at here X which is there now it is has 12 elements. So you can see it here from the original shape it is 12. Okay. Now what if I want to convert it into 1 + 12 or 1 + 1 + 12. Now sometimes during uh the shape uh uh uh manipulation we might need these

**[12:20]** kinds of functionalities down the lines whether in CNN's or in RNN's and stuff. So now how do we do it? Now we can use what is known as unsqueeze and squeeze. Okay. Now this unsqueeze functionality which is there, it will add a dimension and squeeze functionality which is there, it will uh remove the dimension. Okay. So now let's say that I have a tensor which is uh having the values 1 2 and three. So the size of this will be

**[12:49]** three. Okay. That is what you can see here the original size here. And then x dot unsqueeze zero. So that means that you insert a dimension at the index zero. So now because of which now if I go ahead with this unsqueezed shape now you can see it here. Now it is 1 + 3. Okay. Now if you want to make it 3 + 1 now that's also fine. So then you want to squeeze it in the dimension one. So 0

**[13:19]** 1 so always counting start with zero. Okay. So it becomes 3 + 1. So now you can take uh either of them and then if you perform uh the squeezing okay so whatever was uh unsqueezed for which you have added dimension that will be removed and then you get the value like this. So therefore depending on your requirement either you can add dimension or you can uh remove dimension. Now this will be very much useful for us going ahead.

**[13:49]** Okay. Now another important functionalities that is needed for us is changing the order of dimension. Okay. See what happens is most of the times uh uh let's if we take an image now preferably we say that it is uh height cross width cross channels but whenever we are going ahead with implementation we might want the channels now which is the third dimension now if you consider the dimension so

**[14:19]** now if you take it like this is 224 224 3 okay now this is of 0 1 2 but sometimes we might need 3 224 224. Now in that case now we need to permute the dimension okay without modifying the content. Okay we need to permute the dimension. Now how do we do it is what we are looking into.

**[14:48]** So now I have just taken a uh random tensor which is of shape 224 224 3. Now I think you remember this 224 224 + 3. This is the standard size that we use uh in most of our imagenet uh images. Okay. So then you have the shape 224 224 + 3. Now you want to permute. Okay. The first will be 220. Uh so the first will be the second uh uh one. Now which is 0 1 2. So

**[15:18]** three will be the first one and then 01. So this 224 this 224. Okay. So now what I did so it's like I want the second index to be then zero and one. Now this is how I'm permuting it. Okay. So now whatever is here is coming here. This coming here and this coming here. Okay. This is the idea of uh permutation of the dimension. Change the

**[15:47]** order of the dimension. Okay. So now let's look into some of the uh very uh basic uh tensor operations. Now the standard addition, subtraction, multiplication, division, exponentation. So which is similar to the numpy operations that we have already seen. Now we can look into it and then so we can look at the matrix multiplication which is a very important operation that is needed. If I have matrix A and B which is of uh in this

**[16:16]** case which is of the same size 2 + 2 2 +2 you can use this at okay similar to what we used in numpy you can use this at and then you can go ahead with the matrix multiplication okay so this is regarding matrix multiplication so then now coming to the important thing as I told earlier any tensor that is created will be there on on the CPU. Okay. Now you want to port it, you want to move

**[16:45]** it, move the tensor onto the GPU so that the operations can happen on the GPU. And one important thing to be remembered that if you are performing any operation, all the necessary things should be there on the GPU. For example, if you are running a model, okay, now you need the forward pass. So you need uh uh X and Y. Now all these needed uh things has to be there on the GPU. Only subset of them being

**[17:15]** there on the GPU is not allowed. Okay. Now how do you do it? Now you can this is a random tensor I'm creating. So X dot device. Now this is an attribute of the tensor. Now initially it is on the CPU in line number nine. What is that I'm doing is X dot to device. Okay. Now this device. So since the GPU is available, we have already enabled it. I'm moving it onto the GPU. Okay. Now that is the reason when I use it post this porting. Now we can see that it is

**[17:44]** there on the CUDA device. Okay. Now always whenever we want to perform any operations all the needed uh things has to be there on the GPU. Okay. And sometimes if you have multiple GPU cards it it has to be there on the same GPU card. Okay. Now consider that you have a system where in which you have four GPU cards. Your model is on one GPU your X is on another GPU or Y is on another GPU. It will not work. Okay, all of them has to be there on the same GPU card unless there are ways of

**[18:12]** doing it. But this is the standard procedure that we use. So now let's uh look at the a very important uh idea which is needed for all of our uh uh models that we'll be using. Uh how does the gradients gets computed? Okay. Now, uh the workhorse that we have is uh gradient descent. So, we should

**[18:42]** know how does uh the gradient actually go ahead and work. So now here I have creating a tensor with only one element which is two and then I'm enabling require grad is equal to true. Okay. So now this will take care of tracking of the gradients. Okay. So now uh torch.tensor tensor I'm creating with value two. So now the x that I have okay the x that I have is having the

**[19:13]** value 2. So now proceed now y I'm computing y x² + 3x + 1. Okay the y that I'm computing is x² + 3x + 1. Okay. So now now if I go ahead and compute the gradient of this y with respect to x. So now we'll be going ahead and

**[19:42]** finding dy by dx at the given specific x= 2. Now this will be 2x + 3. If you substitute the value now we will get seven. Okay. Now let's check that whether it is there. Now we just need to use this y dot backward. So now it will be able to compute the gradient with respect to the parameter which is there here which is x that is

**[20:12]** being used. So therefore you can see that the gradient which is dy by dx with respect to x is equal to 2. You can see that it is a tensor of 7. See again I just want to emphasize another thing. see uh if X and Y know whenever they are the on the same device the gradients know which will be there will also be on the same device now as of now the device that is there is CPU I haven't ported anything onto the GPU okay so now let's look into how does the same

**[20:40]** thing happens now whenever you have a tensor uh so I had a scalar so the same thing we are uh extending it to a vector now I have my x now which is in this case X which is there is uh X1 X2 X3. Now which is uh taking up uh the values which is 1 2 and 3 which is 1 2

**[21:12]** and 3. So now and then then what do I do? I perform y is equal to x². So that is my y is so x1 squared x2 squared x3 squared. Okay. Now in this case this is 1 4 and 9. Now this will be the content of my y. So now then what is the loss? See here

**[21:42]** again I have declared requires grad is equal to true. Now what is the loss? The loss is y dot mean. Okay. So now now this is uh now loss will be 1 / there are three elements. So 1 + 4 + 9 this is uh 14 by 4 okay sorry 14 by 3 4.667 667 around that we'll come. Okay. Now this

**[22:11]** is the loss. Okay. We have got the loss. Now let's go ahead and find now how does the gradient work. Okay. So now now we have the loss. Now what is this uh loss which is there? Now this is uh 1 / 3 of y1 + y2 + y3. Now this is 1 / 3. Now what is y1 x1 squar + x2 squar +

**[22:53]** x3 squared. Now if you go ahead and find the gradient with respect to what you're finding. Now let's say that you're finding the the loss with respect to some x i. Now how it will be? So now this is x i of 1 / 3 whatever x1² + x2 squared + x3 squared. Now if you use any specific x i. Now let's say that I'm going ahead with x1.

**[23:22]** Now then if you are going ahead when computing the loss with respect to x1 so the remaining things will be zero. So now this will be 1 / 3 2 into x1 now which is 1 3 2 into 1 which is uh 2x 3 now which is 0.667. So now now the gradient with the respect to this for

**[23:54]** any thing so it will be having three elements. So it will be okay it's better to write it as the x itself x. So this is 0.667 this is uh 1.333 this is 2. Now this will be the gradient. So since it is we are doing it for the first time I thought we'll go ahead with the full-fledged computation.

**[24:22]** Now we can see the same here 0.67 1.33 and 2. This is the uh gradient computation that will happen. Okay. So now let's go ahead and perform a manual uh linear regression now with autograd so that we can understand how things working. So now my x now I have 1 2 3 4. Now it's like uh see now please remember this is the first example, second

**[24:48]** example, third example and fourth example. Okay, these are different examples that are there. Okay. So now and then my y is 3 5 7 and 9. Okay. So now the w so now what is that we are trying to compute now? Y is equal to wx + b. Okay. So that is what we are trying to look into. Okay. Now y equ= wx + b.

**[25:19]** So we want to okay now to be even more specific I can say that now we are going ahead with yhat. Okay fine. So now the w that I need is a scalar. So I'm going ahead with some random number uh which is requires the gradients. So and then b which is the biasum which is one. the learning rate that I'm going ahead with 0.01. So now you go ahead and perform the first thing is for you you

**[25:49]** just want to iterate for 100 times. So you're iterating over the whole of the data that you have. So as of now you have four elements. So x uh you are getting it as so x into w + b. Okay, that is your prediction. Now here you're obtaining the mean squared error. So the difference you're taking the square of it so that uh this becomes a positive entity and then you're taking the mean of it loss dot backward now you'll be able to compute uh the gradients okay so

**[26:22]** now with respect w and b so now then you're manually you're manually modifying it okay see normally we don't do this so whenever I use this with torch ngrad now that means that whatever is inside this block no which is there now here these four lines for which in these four lines we are not going ahead with any gradient computation. So okay so but now what we do is normally we take care of this automatically. Okay here we only have

**[26:49]** two parameters. So we thought that okay now rather than going ahead with uh uh updating it uh using a call so we want to do it. So w is equal to w minus so whatever is the learning rate which is uh 0.01 into the gradient. So with respect to the W again the same thing with respect to Z uh with respect to B and then what do you do is see this uh pytor has a natural way of accumulating gradients. So before uh going ahead with

**[27:20]** any kind of gradient updates now it is needed for us to clear off the gradient. Now that is what is being done in line number 26 and line number 27. Okay. Now either you do it post uh updation so that for the next iteration you don't have any gradients accumulated or you do it at the beginning. So it's a design it's it's a choice that you can go ahead and make it. Now here I'm making it a after the updation. So then after every 10 epoch

**[27:48]** so I'm going ahead and printing what is the loss and then the finally the learned weights that we have. Okay. So now you we can see here over the things now we see that the loss decreases. So this is a regression problem. So as the loss decre this is the final loss that we have the parameters that were learned w is equal to 2.13 and the bias term is 0.609. Okay. So now this is how we do it manually. So

**[28:17]** now let's do the same thing. Now uh using the needed uh needed packages. Okay, needed packages and module. So now we need to use what is this known as this NN domodule. Okay. Now we need to create a class for the model. Okay. So and then whenever we are creating models in PyTorch, every model that we create has to be a child of this NN dot module.

**[28:46]** That is what you are specifying it here. Now this linear regression model which is there now this is the child of this NN dot module and in the init first you are calling the init of the parent and then so you are creating a linear layer so with nn.linear nn.linear is the the functionality that we are using. Now this is what is the input and what is the output. So input is a scalar and output is also a scalar. See uh now whenever we are declaring

**[29:15]** this we should not think of uh so how many examples are there in batches okay so every model that is being created is agnostic of the batch okay per example what is the size is what needs to be looked into okay now for each example the input the x which is there is a scalar and the output the y is there which is also a scalar so and then unless specified there will always be a bias term so therefore this declaration that you can see in line number 11 is

**[29:45]** exactly same as what uh we have done here. Okay, in line number 9 and 10 that we can see here. Okay, so now this is what we need to do it in the init method and then pause that we have to write this forward. Okay, this forward method. So it takes X. So you just pass this x which is there through this uh linear layer that is there which takes scalar as an input

**[30:14]** and gives scalar as an output. So that is the output. Now this is this is exactly wx + b when x is a uh scalar. Now how do we implement it? Okay. And then you create an instance of the model in line number 17 and then you can print this model. As you can see now here it says that the in feature is one, the out feature is one and the bias is true. Okay. Now I think uh you agreed to my

**[30:42]** previous statement that is there always unless specified the bias term will always be there. Okay. Now how do you go ahead with the training? [snorts] Now you have your X that you have created, Y that you have created. Now you don't need to create W and B as you did earlier in our manual case. No we don't need to do that. Okay. So now you have this model which is there. Now you're creating an instance now of this uh linear regression model. So earlier we computed loss manually. Now we don't

**[31:12]** need to do it. We can uh use the predefined losses. If you want to create some uh user specific losses that also can be done. Okay. Now we will do it uh post in our uh discussions. So here the criteria that is there this NS NN MSE loss mean squared error loss and then the optimizer that is being used is SGD okay you can use SGD or Adam since it's a very straightforward and a small thing so SGD is fairly comfortable so now it

**[31:41]** is this updation which is there you're using this stoastic gradient descent which is there for updating all the parameters that is there in the model so modelparameters will give you the list of all the parameters now now again I just want to bring it to your note is in this case that in a model now you can have multiple subsets of parameters now when which each parameter is updated by a different optimizer altogether okay it is uh possible in the case we will look at it

**[32:10]** when it is needed okay so then the learning rate that we are using at 0.01 01. So first you go ahead with a prediction. Okay. So this is where wx plus b equation will be implemented and then you compute the loss. Now to compute the loss you just need the prediction and then the true value. So since the loss you have already defined in line number 10 here that it will be an MSE loss. So the MSE loss will be computed. Okay. The first is optimizer.0. Now this is what I

**[32:39]** told you. So optimizer has the linkage for the parameters. So what you are doing is for each of the parameter that is coming under this specific optimizer now whatever parameter is there you go ahead and go ahead and make all the gradients as zero. Okay in the previous step there are chances that the gradients can be accumulated. Okay so we are just going and avoiding it and then loss dot backward. So this is where we compute the gradients. So we you have two parameters one is w and b for both of them the gradients will be computed

**[33:07]** with respect to w and b for the loss. And then optimizer.step. Now this is uh where you update the parameters. This is where w is equal to uh w minus learning rate into the uh the gradient which you have computed is implemented for all the parameters that is coming under that specific optimizer. Okay. So now that will be done and then for each uh this is just the line number 23 and 24 are just for the sake of bookkeeping. Okay. Now you can go ahead and uh see that now

**[33:37]** after 90 epoch the loss you have got is 017. Now here now whenever we did it manual so after 90 epoch we got 0.027. Okay. So we are almost more than 10 times lesser. Now we can print these model parameters. Now what is the name? N name of and the parameter model.named parameters. You can just print it. You can see this here that it is 2.03 03 and the bias is.9.

**[34:05]** Okay. Now in our earlier case the ones that we have got is this is 2.13 and this is 0.6. We can see that now using uh uh these automatic differentiation that is uh using all these backward and uh uh optimizer.step. Now we can have the cases in which there is lesser uh loss and then we go near nearer to the actual weights that is there compared to

**[34:33]** doing it manually. Now this is the standard procedure you know always please remember this first you go ahead with the prediction then you compute the loss then you go ahead and make all the gradients as zero. So this line number 19 can be after number 21 also you fix the place where you want to clear off the previous gradients. So and then loss dot backward now you go there and then here is the step in line number 20 now where in which we obtain all the gradients for all the parameters with

**[35:01]** respect to uh all the parameters for this loss and then optimizer.step is where all the weights are updated. Okay. So now yeah we looked at the parameters. Now let's bring in the idea of the batches. Okay. Now how how to go ahead and uh uh work in terms of batches and then you have uh multiple input features. Now previously also you had uh a bunch of examples but the dimension of

**[35:29]** each of the feature was uh a scalar. Okay. So now here now I have uh I'm creating a linear layer n.linear input feature is four and the output feature is three. Now that means that uh so the input the each input that you are expecting is of size four and then that you're converting it into three. So now what is this doing is now we can uh think of this like the w uh matrix which is there is of r 4 + 3. The x that you

**[36:05]** have obtained is r 5 + 4. Okay. So now what do you do is this uh wx plus b this b will be uh having four elements. Okay. So or you can think of it as an augumented uh thing that's perfectly fine. So this is w is uh x w + b just to retain the the shapes.

**[36:44]** Okay. So this is x is 5 + 4 and this w is 4 + 3 + b. You have five examples. Each of the examples were in fourdimensional space. Now after this multiplication what you have done is you have transformed it. Now you have got 5 + 3 you have transformed it into a three-dimensional space. Okay fine. Now that is what this n linear is doing. So you are creating a random vector which is of shape 5 + 4.

**[37:14]** Five examples when which each example is of shape four. Okay. So and then uh so you just pass it through the linear layer you get the output which is 5 cross 3 as the transformation that we have seen. Okay. So now uh this is uh fine. So now let's bring in nonlinearities. Okay. Now again any neural networks without nonlinearity is of no use for us most of them. Okay.

**[37:43]** Yeah. So you have this tensor. I just want you to look into different nonlinearities that are there. We'll be looking at RLU. We'll be looking at sigmoid. We'll be looking at tanh. Now these are the the famous uh ones you there are others also leaky relu all those things. So we will whenever needed we will invoke them at the right times. So and then these activations are applied element wise for each element it is applied. So whenever we go ahead I'm

**[38:12]** assuming that all of you are very much comfortable with the functionalities of relu sigmoid and tanh. Okay. So this is how you get it. F dot relu. This is the functional that we have declared earlier if you remember. So you pass this x torch dots sigmoid. So of this x and torch tanh of this x just to understand that how the outputs are there. Now you can see that very easy to inspect is relu. So you can see that all the negative values are becoming zero. So

**[38:40]** these two values will become zero and this one and two remains as it is as you can see here. Okay. And similarly the sigmoid and tan h will work. Now then how do you go ahead with uh converting the logits which are there into into posteriors. Okay. So now for that we preferably use uh not we not preferably we use uh softmax. Okay. Now we have this uh tensor know which is 21 and 0.1 that you are trying it to

**[39:10]** convert it into a uh softmax. Okay. So now now this logit switch is there torch dot tensor. Okay. So and then f dots softmax for this logit across one. Now uh now what is this one? Along which dimension? along the dimension one. Okay. Now we have discussed what is zeroth dimension, first dimension, second dimension, so on

**[39:39]** and so forth. You're applying uh uh this uh uh soft max along the dimension one. So you get this logits and then the posteriors. So and the pro that's how we get into probabilities but in terms of uh uh any classification task from which I'm looking into. So this is how you get p of y given x for that specific x. So and then it should for sure it will satisfy the thing. So if you take the

**[40:07]** sum of it, it will be one. Okay. F dots softmax is the one that is used. Okay. So now let's look into we already looked into the MSE loss. Now let's look into the cross entropy loss. Okay. And please remember this thing. Uh so the input for the cross entropy loss are logits. Okay. now but it is converted into probabilities inside of it so we don't

**[40:36]** need to worry about it okay so so now you have these two examples now these are the logits this is for the first example now this is for uh the second example okay so and then these are the targets the first example has to be classified as zero the second example has to be classified as one so now you have three classes 012 okay so now let's look at it now then you use NN dot cross entropy loss and then what is the

**[41:04]** criterion so the criterion is this N dot cross entropy loss and [snorts] and we know that how it works it takes the average of this okay so minus log of the the softmax conversion of this and the minus log of soft max conversion of this value which is there of the true class is what is being looped into and then the average is taken that is the loss okay loss loss will always be a scalar so in line number 18. What is happening? Yes. Now

**[41:31]** you can see that loss do item which is there. See now what happens is uh this loss uh which will uh know be most of times on the GPU. So you just want the numerical value of it. It will be a tensor. You just want the numerical value of it. You want to see in terms of the real value what is it. Now in that time we use this loss item to go ahead with that. Okay. So now the next very interesting idea

**[42:01]** that is needed is how do you create the custom data sets. Now here I just want to bring in couple of uh uh ideas. See most of the needed toy data sets that we use will be available uh uh publicly in the PyTorch repository. For example you might have heard of this data set called as MNEST. You might have heard of another data set which is known as Fmnest. You might have heard of the

**[42:29]** data set which is CR10. Okay. Now emnest and feminist are the grayscale data sets. Now CR10 is colored images which is of the smaller dimensional size. Now similarly you have lot of other data sets which is needed for uh you know different downstream tasks. So not only for classification you have data sets which might be needed for other tasks as well. Okay. So that is available for us to work around to understand what is

**[42:58]** happening. But what happens is whenever we are working on realtime applications. Now what happens is the data sets that we need for us to work may not be available publicly in the PyTorch repository. In those cases what do we need is we need to create a way of obtaining the data sets. Okay. We need to have a mechanism. So now what is needed is now we want the data sets so we want to put them into batches and so

**[43:27]** that we can iterate over it with mini batch stoastic gradient descent that's the uh underlying idea that we want to use. [snorts] Okay. So now let's create a very simple data set and then we will see how one is data set which has all the datas ready and then data loader is where we put in them into batches and then go ahead with that. Now let's see how it can be done. See whenever we are creating a data set now similarly whenever you create a

**[43:55]** neural network model it has to be a child of n dot module. Now similarly this also has to be a child of this data set. Okay. Now the simple data set is the name of the class. You can call it whatever whichever thing you want. So this is the data set that you have. Okay. So then now it should have these three methods. Okay. Now one is this init length and get item okay in exact the naming convention has to be there.

**[44:23]** Now init whatever depending on the initializations that are needed you need to go ahead with it. Now length is uh a functionality it's a method that will give you how much of how many examples are there and then to get one specific item and each element is indexed. So which indexed uh item are you going ahead and referencing is a question. Okay. So now here since it's a very simple data set I'm creating some dummy data here. So x I'm creating 100 examples where in which each example has

**[44:54]** uh two features. Okay each example is a row. Okay it's 100 cross two things and then what will be the y? The y is you you take the sum of this uh the first column and then the second column which is there or zeroth column and then the first column which is there. Now if it is greater than zero is what you are checking here. Now what do you mean by greater than uh zero? So it's a relational operation. The outcome of any relational operation which is there now

**[45:22]** will always be a boolean value. And then when you convert that boolean value either into an integer I or into a float or into a long no it will be converted. If the boolean value is true it will be converted into one. Now if it is false it will be converted into zero. So because of which you're creating y you're creating the labels where in which if the sum of both the columns that are there now if it is greater than zero then it is one else it is zero that's the label that you are creating

**[45:50]** okay fine and then you have this length so the length method which is there which will return how many example the length of this okay so normally we use the length of x or the length of y is preferably used because y will be a a vector okay it will be a standard So that the length will be comfortable. So and then get item you will specify which index that you want that specific indexed item you will get it. Okay each each it's it's it runs it it returns one

**[46:18]** element. Okay. So if I say that I want the element at index 20 now the 20th row know will be returned and then here the 20th element will be returned. Okay. This is how you create you create an instance of this data set. Now what is the length of this? Now the number of samples that you have got is 100. Okay. Now you can go ahead and say that okay I want zero element. So whenever you are saying that I want zeroth element. So

**[46:46]** you're you're actually making a call for this method get item which is uh being defined in line number 16 and 17 you're returning the x value and then the y value as you can see which is which is going into this sample x and sample y and then you can print it out. Okay. Now this is creation of data set. Now the next thing is you have got a data set where which you have 100 elements. Now you want to break them into batches. Okay. Now because of uh see these are two dimensional data very easy to

**[47:16]** handle. But think of imageet where in which each image is of size 224 + 224 + 3. Okay. So in that case we may not handle the whole bunch at the same time. No. In that case we want to make the examples into bring them into different batches. Now that is what is known as a data loader. Creation of data loader. Now using this data set. Now what is the batch size? How many examples has to be

**[47:44]** there in each batch? I'm saying that there has to be 16 examples in each batch. Now I have 100 examples. Now I have 16 examples in each batch. Now in the last one now obviously there will be lesser value. Now as I told you any model which is there is agnostic of the batch size. Okay. So now either you give 16 examples or you give 20 examples or you give five examples in one case it's perfectly fine. It's batch size agnostic. Okay. Now then you see a very

**[48:12]** interesting uh uh thing in line number eight which is shuffle is equal to true. Now what does this mean? What do you mean by this shuffle is equal to true? Now here see if you break your data set into batches of 16 16. Okay. Now let's say that you have broken it sequentially. Now what happens is first 16 example now will be there in the first batch. Now then index 30 uh 1 to uh so sorry index 15 to 31 will

**[48:42]** be in the next batch. Now so on if you divide it sequentially every time the same order you will get the data in the same order. Now rather than that what do you do after each epoch you shuffle things okay now so that the example that came in the first batch no may not come in the same batch okay that's what this shuffle is equal to true means okay so now you get data loader batch x and batch y you work with and each batch will have 16 examples because I have specified that the bat size is 16

**[49:12]** okay this how do you do it now this is a standard procedure okay that is uh used now the only data set modifies Now if you are going ahead with MNST now we will be using the data set for MNS. If you are using C4 10 you'll be going ahead with the data for C410. Okay. So now let's go ahead and uh create an MLP for binary classification. Okay. [laughter] Now here I'm keeping this agnostic with respect to the input size and then the number of classes. The only

**[49:42]** thing that I'm doing it here is okay now the network that I'm doing is so now uh I have some x okay that [snorts] I'm multiplying it by w 1 which is a matrix plus b 1 this is one transformation this I am using some

**[50:10]** activation Okay, this I'm multiplying with W2 plus B2 which is there. This I'm passing it through again same or different activation it's perfectly fine. This I'm multiplying it by W3 plus D3. Okay, either in the last time whether you pass it through activation or not it is uh since it's a classification normally we don't uh pass it through any activation in this case and this is what

**[50:38]** I'm doing here. So now now what is being done here? You need to declare uh w1 w1 you have to declare it as uh some specific sized matrix w2 and w3 and b1 b2 b3 you will be depending on the declaration of w1 ww3 it will be there. Now, so if you want to think of it as uh some this is uh the first

**[51:06]** transformation, this is uh applying W1. This is applying W2 and this is applying W3. This is a network structure that you're going ahead. Now here when I put three dots, I don't mean there are three nodes. Okay, just for uh referencing I'm putting it. Okay. So now here now this is the the input dimension.

**[51:34]** This is uh the number of classes. Okay. Now these two are fixed. Now depending on your data your input dimension and the number of classes will be fixed. I'm going ahead with a simple classification task. Okay. So and then you have choices for this. Now this is the hidden dimension one hidden dimension two. Now you can go ahead and uh go ahead with your choices. It's perfectly fine. Okay. Now here the

**[52:02]** network structure that we have is now I'm creating an MLP which is uh a child of this NN dot module that you have. So the init method input dimension hidden dimension and number of classes. So the first I'm going ahead and creating uh FC1 the fully connected FC is fully connected one which is NN.linear linear with input dimension and hidden dimension. This is W1 creation of W1 and then here FC2 which

**[52:31]** is hidden dimension and hidden dimension. So that means that I'm choosing that the dimension of uh the dimension of this and the dimension of this I'm fixing it to be same. Okay, it's a design choice that I'm making not necessary. Okay, so in this case I'm doing it. it can be different and then I have nonlinear hidden dimension to number of classes. Okay. And then how do we go ahead with the

**[52:59]** forward method? Now first you go ahead and pass it through FC1 and then pass it through RLU and then pass it through FC2 then pass it through RLU and then pass it through F FC3. So now here this is exactly the same. First you pass it through this. This is you are passing it through FC1 and this you're fixing it as ReLU and then whatever is there you are uh

**[53:29]** passing it through FC2 again this is ReLU of it and then this whole thing you're uh this whole thing whatever is the output that you're passing it through this is FC C3 okay this is what is being done here fine good so now here I'm saying that the input

**[53:56]** dimension is two the hidden dimension is 32 so the number of classes is two okay now once you have number of classes is equal to two so you can have it as only one node that's fine one node or two node it it's perfectly fine no problem so input dimension is two now this is 32 this is 32 this is two that means that I I have a matrix which is 2 + 32 32 + 32 and then 32 +2. So this is the first matrix which has 64 element. This is 32

**[54:25]** square elements again I have 64 elements. So in total and then now I have 64 + 64 + 32 squared. Now these are the weights that are there to this plus I have 32 + 32 + 2 which are bias term for each each one of the nodes. Now these many these are the number of parameters that are there and then you want to go ahead and find the gradient

**[54:53]** with respect to each of these parameters and then update. Okay fine. So now this is you're creating uh uh the instance of it and immediately you're moving it onto the device. Okay, you're moving it onto the device, the CUDA device that you have and then you can print the model. Now you can get it. Now then go ahead and uh train the MLP the simple data set that we have declared.

**[55:20]** Now you have two columns y is the sum of so if the sum is greater than uh zero then it is uh one else it is zero and then you have created a data loader you have created an instance. So you're using cross entropy loss. It's a binary classification problem. You could have suffied with only one node and then you could have used BC laws also perfectly fine. There's another way of doing it here. Here you're using the Adam optimizer. The optimizer that you're using here is

**[55:48]** Adam. And then you're using the same optimizer for all the parameters. And then the learning rate that you're using 0.001. The number of epochs that I'm going ahead with 20. Okay. So, and then I'm going with 20 epochs. So, model.train. Now here you are specifically saying that there are two modes of evaluation always. Now one is the training mode which is there. The other is the evaluation mode. Now model train will explicitly say that now the

**[56:16]** model is in the training mode. Okay. So the total loss is zero. Correct is equal to zero. Total is equal to zero. Okay. So this correct is how many of them are correctly classified. Now this total is how many examples are there. So just to compute the accuracy number of correctly classified examples divide by total number of examples because of which you can get the accuracy. Fine good. So now you have batch. So this is what I told you. So please remember in my earlier

**[56:45]** cell I have already made the model which is there. The neural network which is there I have ported it on to the uh device which is the CUDA device. Now the X and Y which is there which I'm getting also needs to be ported. Okay. By default, the device that will be there will always be a CPU device. Okay. Now, that is what you're doing in line number 30 and line number 31. You are porting this X and Y onto the device. Okay? And then you perform the forward

**[57:14]** propagation through this. So, you pass the X, you get the logits and then now please remember whenever your CC loss for the uh loss, the input is logits. So, you don't need to explicitly apply softmax. Okay? So you apply the logits and then uh the the actual uh uh vector of labels. Okay, you're not converting it again into unhot encoding. Okay, no need for that. So and then you go ahead

**[57:42]** and uh uh clear off all the previous gradients. Loss dot backward computation of gradients optimizer.step you're updating the parameters. Okay. And then the total loss is whatever the loss that you have computed. So you are just incrementing it for each you are just incrementing it and then so what is the prediction? How do you get the prediction? So now now this output will be a

**[58:11]** two-dimensional uh uh vector. So the y okay I'll just put it here. Okay the y that I have belongs to r2. Okay. Now what it will be? Now this is y0 and y1. Now what is this y0? This is probability that y belongs to class 0 given x is equal to whatever specific x that I have. This y1 is probability that uh y is equal to 1 given x= x and you add you

**[58:44]** should get one. Okay. So now which uh which is the label. Okay. Now this is of index zero. This is of uh uh index one. Okay. And and please remember there will be a transpose of this. This will be a row. Okay. That's why you are using dimension as one. So I normally prefer to write it as a vector. So that that is the normal notation that is being used so that it is easy for you to uh go ahead and comprehend. Okay. So now you take the maximum of it. Now if you take the maximum of it, you get the

**[59:12]** probability. You don't get the decision. Now whatever is the maximum, you want to find the argument pertaining to it. Okay, if y0 is maximum h what is the argument pertaining to it? It is zero. Okay, what is the index? What what do you mean by argument is I I mean index in this case. Okay, so now which is zero. Now similarly for y1. Okay, so you take the maximum uh you you compute the maximum maximum element and then you compute the what do you call the the

**[59:43]** argument of it. Okay. Now that is uh this uh line here in line number 43. Okay, you got your predictions and then how many of them are correct? Now how do you do it? Now you have your uh predictions. You're checking this is prediction will be uh it will be a row vector. Your batch yv is also a row vector. So you're saying that you're comparing these two vectors.

**[60:11]** Okay, are they same? So now since you are going ahead with relational operation the outcome will be a boolean vector. So and then you take the sum of it. Now that means that whenever you take the sum of it uh what will happen is it will be converted into true will be converted to one and false will be converted to zero. So that will give you how many of them are correctly classified. Now whenever you take the sum it will say how many ones are there. Okay. So which is nothing but how many in that specific batch how many of them have been

**[60:39]** correctly classified. Okay. And dot item. So that you are converted it into a value. So you increment the correct and then this is the batch size of the size zero of it. Okay. Now size will give you uh a tuple. Now you take the zeroth element of it and then you compute the total accuracy and then do it. Now as you can see here initially the accuracy was 50/50 and then it increased increased. So you can see that at after around uh 15th epoch now you

**[61:10]** can see that the accuracy is around uh 0 98. Okay. So you can get it like this. Okay. This is how we work with the MLP. Okay. Now uh this is where uh uh we conclude this segment of our uh uh tutorials. In the next segment now we will see how we can work with CNN's in a similar fashion. How we can work with CNN's how we can

**[61:39]** work with RNN and then we will declare uh CNN's and RNN we will make a uh networks out of it and then we will see the propagation that that is uh what we will be doing in the next segment of the tutorial. Okay, this is where we stop. Thank you. Hope you enjoyed the tutorials. meet you in the next segment when which we'll be discussing CNN's and RNNs. Thank you.
