# Transcript — Tutorial 2 : Introduction to Numpy

> **Source:** https://www.youtube.com/watch?v=E79ld44pfGM  
> **Channel:** NPTEL - Indian Institute of Science, Bengaluru  
> **Duration:** ~69 min  
> **Note:** Auto-captions cleaned lightly. Minor ASR errors possible.

---

**[00:03]** Hello all, welcome back to this tutorial. I hope you enjoyed our uh previous interaction. So where in which we looked into couple of basic constructs of Python like uh looping, conditional statements, functions, classes, methods. In addition to that couple of basic ideas like list, tpples and dictionaries, basic data types and other stuff. I hope these ideas are clear and comfortable.

**[00:32]** The next thing that uh we will be going ahead is we'll be looking into one of the very important uh libraries uh that is needed whenever we we are implementing any small scale uh systems. So the library that is needed for numerical computing is numpy numerical python. Okay. So this will give us a basic uh understanding of uh

**[00:59]** uh tensors. Okay. So we'll formally define uh tensors when we go to pytorch. But as of now you can take it as it's a multi-dimensional list. Okay. A list inside a list. Okay. So now we are importing numpy as np here. Okay. So no error implementation sees that the means that so I am able to import it comfortably. Okay. Now here let's uh obtain an numpy array. Now here

**[01:30]** this is a list. Now if I want to convert a list into numpy array the thing that I have to use np array. Now we'll be able to convert uh uh the list which is there into a numpy array. So a is a numpy array on one-dimensional array. Now a two-dimensional uh array a 2D matrix which is there okay which is B so it's a list inside a list implementation okay so please uh look into all these uh

**[01:59]** opening uh uh brackets closing brackets and stuff and then each element is uh separated by a comma okay so this is a 2D array and then uh we can uh look at them okay and a very important thing okay so One more thing to be noted here is what is the type? Okay, how will be the type look like? Okay, so just to print of uh

**[02:30]** a okay now you can uh see that this is numpy nd array and that is the type that you'll be getting. Okay. Now uh now one of the very important uh things that are needed for uh working with deep learning is the the shapes of the matrices that uh we'll be working with. Okay. So this is a is an umpire array. It's an onedimensional

**[02:59]** array and this is a two dimensional array. Now as you can see here the shape will be a tpple. Now you can recall the example why we took the the 3 224 224 as a tpple because so whenever you use this a dot shape we'll be getting a tpple. Now this tpple has only one element four and this is two + three. Okay. Uh two rows and three columns. Okay. Now we will see how important uh uh the

**[03:28]** shapes of all these uh ND arrays are as we go along. Now if you want to create sometimes uh a special arrays. So for example you want to initialize the weights of a model. Okay. Now then uh now one way of doing it is you want to initialize all with zeros. Now what you'll initialize with all with zeros preferably we'll be initializing the biases terms preferably with zeros and then uh the weights preferably

**[03:58]** you'll be initializing them with uh uh samples from a normal distribution. Okay. Now let's try to see a couple of examples of how to do that. So np.0 0 and then in a tpple you specify now what should be the the shape. Now here you want all zeros which is of shape 3 + 4. Okay. So that means that you obtain 12 zeros and shape them in uh 3 + 4 way np1's all ones 2 + 5. Okay. Now these

**[04:34]** you can see let me run this and show the examples then and there. This is all zeros. This is two rows five columns. Okay. And then if you have uh the random values np.trandom.trand. So you are going ahead and saying I want it as 3 + 3. Okay. So you get random values between 0 and 1. Okay. So 3 + 3. If you want it from uh rand n. Okay.

**[05:07]** So which is uh the normal distribution. You just specify the the shape. Okay. And please uh note that the values uh now will be from minus infinity to plus infinity. So it's it's a sampling from the normal distribution. You're able to obtain the values here. This is this is the output of that. Okay. So now now we can uh obtain the random values like this. There are many other ways. So these are some of the ones that we have

**[05:35]** taken here. So and then indexing. So the indexing here and the slicing is similar to the idea of uh the lists that we have seen or for that matter the strings that we have seen. Okay. So this is of index 0 1 2 3 4. So you can go ahead with zero here and minus one from here. The example shows that and then here uh the slicing that we saw exactly works in the

**[06:04]** similar manner. So you're saying that the starting point should be 0 and then uh 0 1 2. So that means that you get uh 10 till 30 here and then 2 till end that is 0 1 2 that is 30 till 50 in this case. So then now this is indexing of uh 1D array. So then how do we going ahead with the indexing of 2D arrays okay that's much more uh important for us. So this is a matrix that is there this is a

**[06:32]** 3 + 3 matrix. Okay. So now so 0 1. Okay. Now always the first index is for the row and the second index is for the column. So 0 1. So here so the row index and uh the column index always starts from zero. So this is 0 0. This is 0 1. So the value will be 20. So first row see this is matrix of 0. This

**[07:00]** is matrix of 1. This is matrix of two. So matrix of zero, you'll be able to extract the the first row which is there. Okay. On the other hand, if you want just the column, so you just put a colon here and put a one. So this will give you the the first sorry the the first indexed column that is this is zeroth column. This is the first column. Uh so the second column which is there. So indexing always

**[07:26]** starts from zero. So and then similarly you can uh go ahead and say that okay now here from zero I want till 2 that is 0 1 and even columns also I want 01 so that is what it is saying this is in the row uh uh dimension this is in the column dimension so that means 0 1 okay this is 0 and this so if you just say col 2 so you'll be getting uh first two

**[07:54]** rows there the first two columns terms. So this is what so 10 20 40 and 50 is the output that you should get. Okay, as you can see here. Now this is how you can index using in in 2D array. Okay. So similarly you can extend the same ideas to index in uh multi-dimensional arrays as well. Okay. In due time we'll be looking at when we sh when we take uh the hyper volumes.

**[08:25]** Okay. So now the reshaping which is there is a very important uh uh thing that is needed for us whenever we are dealing with deep learning. Okay. So one of the very standard reshape need is uh we start with an [snorts] MLP the input if the data that you are taking the input data is MNEST. So MNEST is a matrix okay it's a 28 + 28 matrix. So

**[08:54]** you flatten it up. So again there is a a need of uh uh reshaping there. Okay you are reshaping it or now if you have a vector you want to make it into a matrix. Now when is it needed? See for example for uh you have used uh some kind of a method to using MLPS to sample from emnest from the emnest distribution emnest distribution. Okay. So

**[09:23]** then in that case you will get a vector which is of size 784 that you had to reshape into a 28 + 28 matrix. Now like this there are ample of examples that we'll be seeing as we go along. Now reshaping is one of the very important feature that is there. Okay. So as you can see here so I have a vector know which is uh 1D array which is 1 2 3 4 5 6. Okay. is the original and then I'm sh saying that okay so this is uh you convert it into 2 + 3 okay so as you can

**[09:57]** see here the original shape is this six elements are there now how do you reshape now these the first three elements this and then the next three okay so you go on filling row wise okay that's what happened so you have reshaped it Okay. So, and then as I this is converting a a vector into a matrix. Now, if you have uh a matrix itself and

**[10:26]** then you want to convert it into a vector. So, for example, you have your feature dimension is a matrix and then you want to convert it into a a vector. Okay. So, you have 1 2 3. So, image.flatten is the one that you use. Okay. Uh so, 1 2 3 and then below that will be 4 5 6. So if you run it 1 2 3 4 5 6. Okay. Now this this is coming as a row. So you can take the transpose and then convert it into a column. Okay. Now this is how uh uh we

**[10:56]** perform the the reshaping. Okay. So now going ahead into the element wise operation that can be done using these numpy arrays. Now here uh now I have an array which is having the elements 1 2 3 here uh 10 20 and 30. So you can add individual elements subtract multiply take the power and uh the standard operations that we have seen. Okay. Now this is uh each array which is

**[11:25]** there is of the same shape. So so going ahead with these element wise operation would be a straightforward thing. Okay. So now let's go ahead into now what is known as broadcasting. Okay. See this broadcasting is a very interesting and important idea. So numpy which is there automatically expands smaller array whenever possible. See for example here I have 1 2 3 and then to this I'm adding 10. Now what do you mean

**[11:54]** by that? Now does that mean that to this numpy array you add 10 as an additional element or just only add for this number 1 10? No. See the way in which broadcast happens is expand smaller arrays when possible. Now here the smaller one is 10. Now this is expanded. So 10 10 10 will come and then you add it. So 1 2 3. So while broadcasting it will become 11 12 and 13. Okay. So now whenever you want to go ahead

**[12:23]** with uh the 2D case now again now similar idea the smaller one which is there will be uh went ahead and then uh it will be expanded. Okay. So now here you have 1 2 3 4 5 6 and then here you have 10 20 and 30. Now if you add them one any any operation that you perform. Okay any arithmetic operation that you perform

**[12:50]** here. So now here you can see that when you add them this 10 20 and 30 is added for this 1 2 3 and then the same thing is uh add in this case also. Okay. Now this is how broadcasting work. Now you should always remember that now it will automatically expand smaller arrays whenever it is possible. Okay fine. So now comes uh a very interesting uh uh

**[13:20]** application which is matrix multiplication. Now which is uh the core operation in the neural networks. Now here I have an array and here I have another array. So see I'm I'm just uh uh going ahead and uh telling the standard thing. If you have a matrix A which is of shape uh M cross N and then you have matrix B which is of shape N cross P. Now these

**[13:49]** two things should match the outcome which is there. Now let's say C is A at P. Now this will be of shape m cross b. The standard rules of matrix multiplication needs to be taken into consideration always. Okay fine. So let me erase this. Okay. So now both of them here in this

**[14:18]** case are 2 +2 matrices. So A so the C will also be a 2 +2 matrix. So I'll not be going ahead and uh explaining the rules of matrix multiplication. I'm assuming all of you know that. Okay. So now let's do the interesting uh operation which is a linear layer wx + b or here xw + b. Now your x is this 1 2 and 3. Okay. And then here it

**[14:50]** is uh these values. So now let's uh look at the the shapes of uh each of this. Now x that I have 1 + 3 and w which is there I think it is uh 3 clause 2. Okay. So is this

**[15:18]** yes three rows two columns and then B which is there this B which is 1 + 2. Now when you go ahead and do this now this this will give you 1 + 2 plus again this is 1 + 2 the outcome will be 1 +2 okay now sorry I'm writing it diagonally okay so

**[15:47]** yeah uh I think the contents are clear so this is what we are doing here xw + b okay So this is 1 3 this is 3 2 and this is 2. So the outcome will be 1 2 this will be the outcome. So as I showed here this is how uh uh the operation uh will be performed. Okay the matrix multiply I'm not I'm not going ahead and looking into

**[16:15]** the individual operations. So just just for the sake of uh uh uh what do you call whenever you you perform any operations where in which matrices are involved now it is always needed for us to go ahead and look into the the shape uh what do you call alignment the shapes needs to be properly aligned now otherwise uh it will create troubles okay you troubles you'll get an error that's all okay so this is a manual

**[16:46]** linear layer So now here now then uh let's look at how the batch of inputs through the linear layer works. Now here x know which is there 1 2 3 this is one example this another example this another example this is another example okay and this w which is there is the same and then how does this happen okay now here now this is uh 4 + 2 now so now let's uh look at uh the execution

**[17:16]** of a batch now this is when we have one example okay now I have a a batch of uh input. Now x which is there is of I had four rows and then three columns. Now my weight remains the same 3 + 2 + b 1 + 2. Okay. Now when you go ahead with

**[17:46]** this this is 4 + 2. You're adding that with 1 +2. Now you understood why the broadcasting is needed. Okay. So now you add this the outcome will be a 4 + 2. Okay. So this is how we go ahead and uh process the the batch. So there will be no

**[18:14]** difference. You get a a 4 +2 output. Okay. Now this is a standard [clears throat] uh the linear uh uh propagation. Okay. Now if you want to add an activation okay now here um the example that I'm going ahead with is rail activation. So rail activation the way in which I'm assuming that all of you are familiar with the rail activation. Now this is uh

**[18:42]** this is a straight line. This is uh this is Z. This is Z and this is G of Z. Okay. Now this is zero. Okay. Till zero it is here and then okay. Uh G of Z is equal to max of 0 comma Z. Okay. This is the the

**[19:14]** rail activation which is very predominantly used. Okay. So now look at this. This is I have defined a function to go ahead and uh perform this yellow operation. So this is np do maximum of 0 comma x. Okay. Now this is done individually. Now this is np array. These are the values. Real of x. So each of them this minus 3 will become 0. - 1 will become 0. 0 will become 0. 2 will

**[19:42]** be two. five will be five. Okay, this is the rail activation. And similarly, now we can go ahead with a sigmoidal activation. Okay, I'm assuming here everybody is more comfortable with sigmoid. Okay, so 1 / 1 + e^ minus x which is there. Now for each element, you know, you'll be applying the sigmoidal value. The value will be in between uh so each value will be between 0 and 1. So if you run this so minus3

**[20:12]** will be converted to this minus 1 will be converted to this 0 will be 0.5 one will be this three will be this okay now instead of x now you can have uh this y now which is there now where in which we have just performed the linear uh transformation so you can make use of it okay so now here we are achieving two things one is I'm introducing the ideas of numpy how you can use the functionalities which are there in

**[20:39]** numpy. In the other hand, I'm sensitivizing you towards the operations that are there in the uh neural networks. Okay. So, we are going ahead with two two things. Okay. Fine. So, now another important uh thing that is there is to convert logits now which are there into the probability vector. Now whenever we are dealing with multi-class classification see for example what do I what do I mean by that is okay let me erase this first

**[21:11]** just a minute okay now let's uh go ahead and consider that same uh network that we had. So the input is 784 dimension 784 you are converting it to 128 from 128 you are converting it to

**[21:42]** 10. Now this is the the MLP that you are using the data set in hand is uh the mnest data set. Okay. So now what will be the outcome? So you have this uh MLP you have this outcome this output which is there is uh logits now this logits has to be converted into a into a probability vector. Now where in

**[22:12]** which so we have 10 elements for example there's this you have let's say that these many elements so this specific element is uh the probability that y is equal to zero given the input x okay so you want to convert them into a so when you add them up so summation i = 1 to 10 probability that y = i given

**[22:43]** x is equal to the input x. Now this should be summing to one. So this feature has to be taken into consideration why we want the output to be probabilities. So so now uh uh now the logit doesn't satisfy that we want to pass them through a softmax function so that uh so the logits are uh normalized properly and becomes a probability vector. Okay. So you take uh the XP okay

**[23:17]** and then uh you divide by the total sum. So these are the values and then uh you convert them into probabilities. Okay fine. Now this is uh the standard softmax function which is there. [snorts] Now let's look into the the way in which uh uh an MLP okay manual MLP forward pass now how it works the way in which we have it is a linear layer after that a relu after that another linear

**[23:45]** layer after that we have a soft max okay so now let me erase this Then right side corner. Right side corner.

**[24:14]** Huh. Okay. Thank you. Okay. Here we have linear, relu, linear

**[24:41]** and soft max. Okay. So we have uh linear so followed by the activation relu followed by linear followed by the the soft max. Okay. Now let's try to do this specific operation. Okay. So np.trandom seed. So this is for the sake of reproducibility. So the seed we are fixing it as 42. Now x which is

**[25:09]** there is uh with four features. Okay. So now the X which is there is belongs to R4 or this is of shape 1 + 4. Okay fine. So now now w1. So what is the operation that we are performing is uh first you have x

**[25:36]** into w1 + b to this you apply the relu activation. Okay. So you apply the ReLU activation for this you apply W2 plus another let's say this is B1 this is B2 for this you apply the soft max okay now this is what uh we are

**[26:07]** trying to achieve here okay so now this w1 is uh 4 + 5 okay so this is this Okay, let me use another color for this. This W1 is 4 + 5. Okay, and then the B is

**[26:34]** 5 + 1. Now, this is sorry this is just five. Okay. So just represent it as uh five here. And then this uh w2 that we have is uh 5 + 3.

**[27:11]** This is 5 + 3 and this is three. Now let's see how does this work. Now this x which is there the x is of size 1 + 4. So then this uh xw1 will lead me 1 + 5 to that you add this this five. Okay. And this is multiplied by and then after that you apply the relu of this.

**[27:39]** Okay. So you individually apply the relu and then to this you multiply 5 + 3 the outcome will be 1 + 3 for this you have applied the you have added the bias terms this you have to convert them into you have to apply the soft max that will give you a vector which is 1 2 and three three elements. Now here in this case uh let's call this as class 0 class one and

**[28:07]** class two. This is a probability that y is equal to class 0 given x= to whatever input it is. Similarly, this is y= this is probability that y = 1 given x= x and this is probability that y = 2 given x= x. Okay. So let's put all of them together. So

**[28:38]** here x into w1 + b and then this is activation this you are passing it through the relu and then using this activation you multiply with weight matrix 2 and then adding the second bias terms okay and then you're performing the flattening you are converting it into a vector okay for the sake of consistency and then you have three things here okay so then what do you do now what is a predicted class now see now what you got is

**[29:07]** a vector of uh length three. Okay. Now what is the decision? What is the class? See we use the arg max. Okay. So whichever element is the maximum what is the index of it is what we are obtaining that is what is known as the arg max. So the maximum one is this. So that means that we are classifying this uh x which is there into the zero. So what does this mean? So the model is saying that with uh probability 0.999 now it belongs to class zero. Okay that is what it is

**[29:38]** saying. Okay. Now similarly if you so this is how you perform the the forward pass. Now if you want to go ahead and obtain the mean squared error. Okay. So mean squed error is very famous loss that is used in regression problems. So you have two vectors. So you want to subtract them and take the square of this and the mean of them. So that is why mean square of this this is what is known as error. Okay. So [snorts] I presume that the function is uh quite

**[30:06]** simple and clear. Okay. And similarly you can go ahead with uh the CC loss the categorical cross entropy loss. Okay. You just need what is the probability of the true class and then you take minus log of that. Okay. If the true class is one so then this is 0 1. Okay. So this is minus log of 0.7 so will be the outcome. Okay. So this is how uh we go ahead with that. Okay. So

**[30:35]** now now let's uh go ahead into uh looking uh into the image representation. So the problem that we are looking at here is I have just uh an image which is height cross width that is you just have a 2D matrix that you want to convert it into a 3D matrix okay now let's say that this is a grayscale image that you have the

**[31:03]** each of the values represents the intensities present in those pixels okay so and then the shape of this is 4 + 4 as you can see now if you want to have a three channel okay so then you just need to np.rand random.rand. So this is 3 44 and one thing is whenever you're using this rand it will be between 0 and one. You can consider it as a normalized image. That's perfectly fine. Okay. So now let's uh look into a very interesting

**[31:32]** and important idea which is uh a 2D convolution. Okay. see in this uh the idea of this uh 2D convolution. a 4 + 4 uh image like this. So now first

**[32:04]** what you will do you will perform the convolution operation on this okay and then you have to move whatever is the stride that you say let's say that you have taken the stride to be uh uh one so then what will happen so you'll be going ahead and moving one moment like this and again one more moment like this so each of them you will get an output so what do you do so when you take this uh 2 +2 block which

**[32:33]** is there. Okay, you have a 2 +2 kernel. You perform element wise multiplication. So whenever you perform the element wise multiplication again you get a 2 +2 block. So that means that whatever is here and here you multiply and obtain here here and here. Oh sorry you obtain here here and here you obtain here here and here you perform and then you take the sum of all these elements. Okay sum of them. Now that

**[33:03]** uh will go to this location. Okay, this is the standard convolution operation. Uh I'm assuming that uh all of you are uh quite uh comfortable with uh standard convolution operation which is there. Okay. So we will see a couple of uh uh uh demos on this when we go into CNN. Okay. So now you have this 4 + 4 and then the kernel that you have is a 2 +2 kernel. Okay. So the image height and

**[33:32]** image width, kernel height and kernel width is this. So now output height is image height minus kernel height + one. So I think uh you remember this uh if you have studied convolutions you might remember this uh obtaining uh the output size uh from the the input and the kernel size. Okay. So for so many times so you want to repeat till the output size output height and output width. Okay. So the output here in this case

**[34:01]** will be 3 + 3. So you create uh a matrix you know where in which all of them are zero. Okay. So and then you select a particular region. So you I is equal to let's say that here output height range of this. So this is output height is uh so three and this is three. Okay. So the first I will be zero. Okay. Now let's take this. Now I is zero. J is zero is what we have taken and then

**[34:33]** I till I + kernel height. Now what is that? So the image you obtain I so image so so you obtain image of I till I + kernel height comma J plus J till J + kernel height. See this is zero. 0 till Okay. So kernel height is three. So 0 +

**[35:06]** 3. This is zero. Sorry. Kel height is two. Sorry. Sorry. So so this kernel height is uh two. This is two. Now that means that this is the slicing that we looked at earlier 2D slicing. So now that means that you will be obtaining this part of the image okay you're taking this you're taking that uh

**[35:36]** 2D chunk okay of the image so now uh taking that okay that is a region of interest as of now so now so region into kernel so your kernel which is there is again your kernel is also a 2 +2 so you perform element wise mult multiplication and you go ahead with the sum. That is what the operation is. Okay. Now then so then you change J. So when you change J what will

**[36:05]** happen? So let's say that at the next iteration the J will be one. Okay. So then you take J is 1. So then this will be 1 till 1 + 2 which is 1 till 3. So now that means that your rows will remains the same. you will be taking this chunk. Okay, that means that you have shifted. Now that is the operation that you are manually doing. See whatever is the

**[36:32]** operation that is uh done this is this is the operation that is done inside a convolutional network. Okay. And then you repeat this. So once you are done with uh this inner for loop so you increase the the element. So this will be one and repeat the similar thing. Okay, this is the uh the overall uh the idea of convolutions. Okay, and you repeat this operation. So final output will be a 3 + 3. This is your output.

**[37:01]** Okay, so so I I'm assuming that uh this idea is clear. Okay. So and then max pooling. See max pooling is another idea that is done in the uh uh CNN to reduce the size of the uh input. So what we do is say for example if you have a let's say that you have a a 2 +2 block and then you stride it and

**[37:34]** then you perform the max pooling. You take this block. Now whatever is the maximum element you put it here and you take the next one you put it here then you take this particular m maximum you put it here you take this and the maximum you put it here this is how you perform it's a 2 +2 kernel there is no element wise multiplication you just take the elements and find the max okay in add so either you can take the max pooling so

**[38:02]** or you can take even the average so whenever you have stride is equal to two And then uh uh the kernel size is also 2x2. See what happens is the input size which is 4 + 4 is reduced into 2 +2. Okay. So this is the reduction in uh the in the size of it. Okay. So there is it is not mandatory that you should always take a kernel of uh 2 +2. You can take different kernels and then compute the

**[38:29]** sizes accordingly. Okay. Fine. So then so the output height and output width will be just the half of it since I'm taking a stride of two and then the pooling size of two. Okay. So this is what I'm taking and then perform the similar operation the max in each of the region you perform the similar operation you get this thing.

**[38:58]** Okay this is regarding max pooling. See here see more of the idea is how do you use these np.m domax now how do you use this np.0 zeros. Now these are the ideas that I'm trying to convey taking the examples already known examples of convolutions and pooling. Okay. So now uh let's look into the sequence

**[39:36]** data. Now here RNNs are very famous for processing of sequence. I'm just taking a a simple sequence. I love deep learning. That's why we are here. So word to index. So world to ID. So this is of 0 1 2 and 3. Okay. And this is the sequence. This is how we can access them. Okay. So this is the first step that is uh normally done. Okay. Now this is 0 1 2 and three. So this word is zero word. Second third and fourth. Okay.

**[40:05]** index with 0 1 2 and 3. Okay. So now then uh why is it needed is see we so we need to go ahead and encode these things with one hot vectors before going ahead and obtaining the embeddings. Okay we'll come we'll look at them. So this one hot uh encoding which is there is a very uh interesting thing uh that is there. For example, let's say that I have uh

**[40:38]** four elements. So let's say that I have a multiclass classification. Okay. So the classes are uh so lion, tiger, uh monkey and man. Okay. So let's say that these are the four classes. So now whenever you want to represent uh any uh

**[41:08]** image with class lion, we will represent 1 0 0 0. This is what is known as a one hot encoding. To whichever class it pertain to, this is one. Again please see that this is a probability vector. You can immediately see it. So when you take the sum of them it is going to one. Okay. So if it is tiger no it is 0 1 0 0. And similarly if it is monkey it is 0 0 1 0 and then if it is a man it is 0 0 1. Okay. This is the one hot encoding.

**[41:38]** So now let's see how it is being done. So number of classes I have is four. Now that means that the length that I should get is four. So the label is two. Now that means that uh it's the second element that is 012. So one hot encoding first I'll make everything zeros which is I get four length 0 0 0 and then whatever is the label I make it one. So this is how I obtain the one hot encoding 0 0 1 0. Okay.

**[42:05]** So then uh you want to go ahead with a batch of things. Okay. So let's consider that these are the labels. The first example is of zero. The second example is of two. The third example is of one. And the fourth example is of three. So like this if this is the case. So then you have a matrix. So you get all of them as zeros. And then now you the first one now whatever is the label. Now this is 0 0 will be one. So this will be so if this is the case. So how do we proceed with this is

**[42:36]** so now I'll get 0 0 0 0 0 0 0 0 0 0 Okay, this is the uh matrix that I have got. Okay. So now here you come here I is equal to 0. So here the zeroth one you make it as one. So that means that this zero is replaced by one in this case. Okay. So, and then here the next one is

**[43:04]** two. So, in this case, okay, this is made as one. And then uh the one is the next one is one. This is made as one. and the last one is three. So this is

**[43:32]** made as one. So this is for the uh the first example. So one hot encoding for secondam first example. This is one hot encoding for the second example. We have to look it up row wise. So on and so forth. Okay. This is what uh this code is doing. Okay. So now uh this is regarding uh obtaining

**[44:02]** the one hot encoding and playing with np.0 and how to make the one hot encoding. Now then uh let's look at how do you like we saw how to go ahead with a manual MLP. Now similarly now let's look at how do you go ahead with an RNN cell. See the RNN cell. See the RNN is this is the the current uh input the weight pertaining to that. This is a previous context and the weight pertaining to that and B. Okay. So this is uh

**[44:32]** X into W1 + some H into W2 + B. This normally we take tan H. Okay. This is the one that we normally take. Okay. Okay, that is what is being done here X. Now let's look at this uh the recurrence. Okay, so now here recurrent cells RNN. Okay, recurrent neural networks. So now here now the sequence length that we are

**[45:00]** taking is five. Now that means that my X which is there me go to the next page maybe that is better. So, so my X now which is there initially it will have 1 2 3 4 five words and then you convert each of them into three features. Okay. So now my X will be

**[45:31]** 5 + 3. Okay. This is what my x will be. Fine. And thing is this is the size. But whenever you are uh going ahead, you'll take each one of them and perform the operation. You take the first one. Okay? So you take the first one and perform the operation, then update the hidden state,

**[45:59]** then take the next one, so on and so forth. That is how we go ahead with the operation. So the W which is there. Okay. So this so this W hx which is there. Now this is of uh shape. So input to hidden. Now the input dimension

**[46:27]** is three. So this is 3 + 4. Okay. So this is the shape of it and then whh this is for the context. This is for the input. This is the weight for the context. Now this is the hidden cross hidden. This is 4 + 4. Okay. So and then B is a fourlength vector. This B is just a four-length

**[46:57]** vector. Now this is what we have got. Now let's see how the operations is done here. Now h initially it is uh h which is all zeros of length four. Okay we can consider them as rows easy 0 0 0. Okay this is what my h is now till the sequence length ends. So

**[47:26]** first I take the the first element which is of length three. Okay. Now what is my h? Now my h I'll be updating it as tan h of whatever x that you have into wx plus the previous h into the weight that you have for that context plus b correct. So now what is the size of the h? You have taken only one element. So

**[47:55]** therefore it is 1 + 3. This is uh 3 + 4. This is uh 1 + 4. This is 4 + 4. This is four. So now this will be now this you get 1 + 4. This you get 1 + 4. You add them you get 1 + 4. This is four. My h is 1 + 4 which is uh going ahead and matching. So h is

**[48:24]** updated. So the new h is obtained. Okay. And then you repeat the process for each of the sequence. Okay. At the time stamp zero, this is taking the first element in the sequence. This is what the H was. The second time you updated, third time you updated and the fourth, third time you update for the fourth time you updated. So like that for each of the element in the sequence you have five elements in your sequence. So you take the first element and update the H. Then with that update a hedge obtain the

**[48:53]** context for the next one the third one fourth one and five times you update this. Okay. Now this is how uh partially the you are going ahead and working with the recurrence cell. So the main idea is how to use uh and implement all these ideas of recurrence is the is the outcome of uh this particular manual uh way of implementing an iron cell. Okay. So now what we will do is so we will

**[49:21]** take a very uh small data set. Okay. Creating a small data set and performing a small binary classification problem. Okay. So a very simple data set. We will have a very simple criterion for going ahead and performing the classification. Okay. So nprandom seed. Now we're just going ahead uh with this for uh the uh reproducibility. So and then I'm taking 100 samples each samples which

**[49:50]** are having two features. Okay. So my X is 100 + 2. Okay. So I'm performing a small binary classification problem. Okay. We are performing a small binary classification problem. So I have 100 features or 100 examples maybe. So each example is of two features. Okay. 100

**[50:21]** examples. Each example is of two features. So you have your X is 100 + 200 rows, two columns. Okay. This is what our X is. Okay. And then what is the Y? The Y is a very simple straightforward one. So you take the element of the first column. So how do you take the element of the first column? If this is X, if you want to take the that is zeroth column, you take all the rows, you take

**[50:49]** the zerooth column plus you take the X of the first column, you add them. Now if it is greater than 0.5 is what is the question. So if it is greater than 0.5, so that means so the outcome will be either true or false. Okay. Now if this outcome is true, then the class is one. If the outcome is false then the class will be zero. So this is what is our y. This dot

**[51:20]** you convert into integer. So your y is a vector which is 100 + 1. Okay. So your this is the binary classification that we are dealing. So now let's see how it is being done. Okay. Now we are uh this is the target shape. So 100 this is the first five outputs and top uh first five labels is what we have obtained here. Okay. So now now what is the next thing that we are

**[51:50]** supposed to do? We have to perform a split. Okay. Now how do we perform the split is the question. So how many samples are there? Now whatever is the first dimension of the X. So that means the number of samples will have 100. So indices np.tarrange till number of samples. So that is so you will have 0 to 99 arranged. All the indices are arranged. Now then you shuffle it. Okay. So that means that so the list which is there with 0 1 2 so on until 99 is shuffled.

**[52:21]** Okay. So you randomly shuffle it. np.trandom.shuffle. And then the training size I'm fixing it as 0.8. Now that means that 80% of the data I'm fixing it for training. and 20% of the data which is there I'm fixing it for testing. So the indices which are there till the train size now that means that once post shuffling the first 80% first 80 indices you are considering as a training indices and the remaining you are considering is a test indices. So

**[52:48]** similarly you obtain x of those training indices x of those test indices you obtain the test data and then the training data. So your training data is 80 elements your test data is of 20 elements. Okay. So we use this uh random.shuffer shuffle. That is the main idea of uh uh doing this task. Okay. So you got our random.shuffle. Okay. So I run this previous cell. Okay. So now here

**[53:16]** done. Now let's uh look into the computation of uh one of the evaluation metrics that is used in uh the classification uh task now which is accuracy. Okay. uh now these are uh we we will come to this function this is uh y true uh this is your actual label and this is what the prediction is now let's uh see how to obtain the accuracy okay so this is

**[53:48]** zero Okay. 0. Okay. What was Let me 0 1 1 0 1 0 1 1 0 1. Now this is uh y true. This is the actual label. And then what is the predicted one? So the predicted one is 0

**[54:18]** 1 0 0 1 0 1 0 0 1. Now how do we compare them? Okay. Now the way in which uh we do is this is uh uh unless this is another list. So what we will do is we will just use the the relational operation. So whenever you compare them you compare them element wise. So I get this is true this is true. This is false. The third one is false. The

**[54:46]** fourth one is uh true. The fifth one is true. So you will uh get a boolean list. Okay. So this is what is uh you give for this function accuracy you give y true and as well as y prred. So you'll compare them. So and then that you're saying that you obtain the sum of it. Now whenever you obtain the sum of it so what it will give is it will uh give you

**[55:15]** the total number of true that is there no which is four. Okay. So in this case, now then with this now then what is the total? The total is length of this white true which is five. Now this is 4 over 5. It is8. Okay. So that is what you will return it as accuracy here and you get this. So as we can see here this is accuracy. This

**[55:44]** is how you compute accuracy. Okay. So in most of the normal cases what happens is since we will having in batches we'll be doing in batches. So this is uh how we obtain the accuracy using these numpy functions. So now let's uh put all of them together and perform a very simple logistic regression. Okay. I'm assuming that all of you know the idea of logistic regression. So this is a single neuron binary classifier. So that we are using.

**[56:13]** So now let's build it from scratch. So you have this uh x which is 200 cross 2. Okay. So this x that is there. Now the x that we have is r 200 + 2. So which means that we have uh 200 examples and each example has two features.

**[56:46]** Okay, now this is what uh we have and then uh what is y now the y is obtained you take x which is there you take the first that is zeroth column plus x which is there you take the first column you take the sum of them if it is greater than 0.5 now then know this will yield you either true or false so wherever true is there you put one.

**[57:15]** Wherever false is there, you put a zero. Okay, now this is what our y is. Okay, this is the same thing that we looked at earlier. So, and then the y the shape of the y will be 200. So, this will be a this will be a tpple. You are declaring it as a tpple. So, you want to make it 200 + 1. So, therefore you use uh the reshape. So, using which we obtain a 200 + 1. Okay. So now then

**[57:46]** now now we need this W. So W is 2 + 1. Okay. So now the W that we have belongs to R2. Okay. So now the same thing when you reshape it we call it 2 + 1. Now B belongs to R which is 1. Now this is the data that we have. X

**[58:14]** uh is uh 200 + 2. Now Y is 200 + 1. This is Y true. So now we have to perform the logistic regression and obtain the predictions. Okay. So it's a binary classification problem. Now we are I'm I'm assuming that all of you are comfortable with these terms. learning rate, epochs and things. Okay.

**[58:43]** So now we are using uh the sigmoid. Okay. So now for epoch in the range of epochs. So you'll be iterating over this. First you obtain the logits. Now how do you obtain the logits? So x is multiplied with this w. So x at w + b. This is a simple logistic regression. So x is uh 200

**[59:13]** + 2 and w is 2 + 1 + 1. So you get 200 + 1. Okay. Now each one of the element of this is actually the logit of that specific example. Now then what do you do? If you want the the probabilities, you pass it through sigmoid of this. So you pass it through

**[59:41]** the sigmoid. Okay. Now then you get the the the outcome will be the probability that y is equal to 1 given that specific x. Okay. This will be the output. That is your y pred. This is your predictions. Okay. So prediction probabilities. Okay. We haven't went into decisions. Since it is a binary classification problem, we can use BC

**[60:10]** loss to go ahead and compute the overall loss. Okay, that is what is being done in this. Okay. So, and then to obtain the gradients, you need the error. So, error is Y prediction minus Y and this is uh DW and this is DB. And I'm assuming that these equations make sense. So you just need to go ahead and differentiate with respect to w differentiate with respect to b and then you get the gradients and then this is

**[60:38]** the parameter update. Okay, this is the standard uh gradient descent. So now where in which we have for the parameter w so w minus learning rate times the the gradient. Okay. So this is how we obtain. So see similarly for B also B minus alpha * the the gradient with respect to B. So since we have the gradients we can directly

**[61:07]** obtain them and then for each uh 100 epochs you're just keeping a bookmark of how many predictions are actually greater than 0.5. Okay so that you're converting it to one and then the remaining things you're converting it to zero. Now this is again a conditional check and then you're obtaining the prediction and then you are printing the accuracy. Okay. So once we run this

**[61:35]** now we get uh at the beginning this is the loss this is the accuracy as we go on. So it becomes see here this is the values. Now these are the weights of W. This is and then this is the final bias. Okay. After this we have obtained 100% accuracy. This is a very simple uh binary classification example. Now then how do we go ahead with the prediction?

**[62:02]** These are the new features. Okay. Now already you have the w you go ahead with that you get the logits and you pass it through the sigmoid and then see whether the given probabilities which are there if it is greater than zero sorry greater than 0.5 then it will be one else it is zero. So and then you will see that now what will be the predictions of each one of them. You'll see that these are the possibilities and these are the predictions probabilities and the

**[62:30]** predictions. Okay. This is how we perform a simple logistic regression. Okay. So now one of the very important thing that we should always remember is uh uh looking out for the shape. Okay. whenever we are performing a lot of these matrix multiplications so we should be pretty much aware of this shape okay now we are let's let's go and do an operation so now I have some x

**[63:02]** okay so this so x is there now that x so you're multing it multiplying it by w1 + b and then after that you are passing it through a relu activation. So whatever is the output that you are multiplying it by W2 let's call this B1 and you are adding B2. Okay. Now for this let's

**[63:32]** see how the the shape uh uh happens. Okay. How the shape is transformed. So now X is 32 + 784. Okay. Now X is 32 + 784. That means that you have 32 examples. Each example has 784 features. Now 784 features should directly ring a bell that uh something looking like a 28 + 28 matrix or it has

**[63:59]** 784 features something similar to emnest or fashion mnest some data like that. Okay. W1 is 784 to 128. Now that means that now the W1 which we have this is transforming from a 784 dimensional space to 128 dimensional space. So therefore now the B1 now has to be a 128 uh uh length thing. Okay. Now

**[64:31]** and then after that W2 is uh 7 oh sorry W2 will be 128 and 10 this is 128 10 and B2 is so what is this so we can visualize this network like this you have here 784 dimens ion you have 128

**[65:02]** here you have 10 okay this is the matrix W1 this is the matrix W2 okay so this is a neural network which is having one hidden layer with 128 nodes so now let's perform the operation the the input shape that we get is 32784 now Then zed 1 is you're multiplying x

**[65:31]** and w1. So this is what you're doing. So this is 38 784 this is 784 128. If you multiply that so this will yield us 32 features 32 examples each of which is 128 and then you add this bias. So bias will just increase or decrease the magnitude but it will not change the shape of the matrix. So then you apply the relu. So relu is individually applied. Now again

**[66:00]** relu doesn't change the shape. Now therefore the outcome of this whole thing is 32 128 and then the w2 that you have is 12810. So again therefore the whole output will be 3210. For each of the 32 examples you are getting a 10 length 10 log. Okay. then you pass it through a softmax function to obtain the the posteriors uh is the whole idea. Okay. So then you

**[66:30]** have relu of this. So we are just looking at each of the shape. So once I run it so input shape is 32784. So then after the first linear layer that is multiplying with w1 and adding b1. So you get 32128. Now after relu will not change the shape. So it is 32128 again. And then after the second linear layer it is uh that is multiplying it by b2 sorry multiplying it by w2 and adding b2 so you get 3210. So this is the the final shape that is expected. So now

**[67:00]** whenever we are dealing with any of the operations now we should be pretty much aware of how does the transformation is happening. Okay. So we cannot have each value looked into whether the transformation is happening or not but at least okay so we should be aware of how the shape of the transformation is getting modified. Okay so now giving a full recap of what are the things that we have discussed in this

**[67:28]** part of the tutorial. So considering the earlier tutorial and this tutorial, we have a nice recap on the basic ideas of Python and then we looked at numpy. See all the examples that we have handled in numpy. So we have explicitly taken it to be related to neural networks or those ideas so that you will be able to understand the operations of uh numerical Python package now as well as

**[67:59]** have a nice recap of how things are. Okay, so I hope you enjoyed it. Just a recap. So we looked at Python variables, list, tpples and dictionaries, loops, conditional statements, functions and classes. In numpy arrays, we looked at obtaining the arrays and how do you look at their shapes. So indexing, slicing and reshaping. So we did it for numpy array and matrix multiplication, we looked at it. So manual linear layer activation functions and then we had a

**[68:27]** nice recap of the convolutions and then the pooling. We actually coded it using for loops and then we looked at the RNN hidden layers. Okay, so RNN things and then we uh coded logistic regression from scratch. So and then we looked at how the shape transformation is happening. So these are the basic ideas that we have looked up. So this completes our uh tutorial uh on numpy. Now then now the next thing that

**[68:57]** we shall be going ahead is introducing pytorch. So hope you guys liked these parts of tutorials. See you again while discussing PyTorch. Thank you.
