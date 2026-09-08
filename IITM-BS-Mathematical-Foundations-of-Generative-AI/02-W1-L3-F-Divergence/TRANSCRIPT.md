# Transcript — W1_L3: F-divergence | variational divergence minimization in generative models

> **Source:** https://www.youtube.com/watch?v=rHnrALMCyIQ  
> **Channel:** IIT Madras - B.S. Degree Programme  
> **Duration:** ~28 min  
> **Note:** Auto-captions cleaned lightly. Minor ASR errors possible.

---

**[00:11]** [Music] Hello everyone. Now welcome to this uh second tutorials uh in uh this course DGM degenerative models. Uh in today's course now we will be looking into the basics of PyTorch. uh the whole of the course you'll be learning lot of theory uh how uh the how how can you mathematically formulate a model and then

**[00:40]** in in addition to that you'll be learning how you can implement these uh kind of models on some simple data sets okay uh we will be using Google collab for this specific exercise. Now, Google Collab is a platform which gives us access to a simple GPU uh 12 gig GPU. Now, we can use it. Now, the whole uh uh coding

**[01:13]** exercise throughout the whole course is structured in such a way that we will not be needing more than that. we will be implementing a very simpler versions of those models and then uh we'll be discussing how to extend those models if you have uh compute a resource. Okay. Uh so in the previous uh tutorials that we have uh undergone we looked at the idea of uh forward propagation and backward

**[01:43]** propagation and how does your gradients are computed and then we looked at the overall training procedure. Okay. Now in this specific uh tutorial we'll be looking at PyTorch basic. Now we will be using PyTorch. In addition to that, if someone is interested, they can look out for resources on how to implement similar models using TensorFlow. But the overall discussion is centered around Pyarch. Now, what do I assume as prerequisites

**[02:14]** uh for this specific tutorial? See, I'm assuming that you know the basic structure of MLP and then you know the basic idea of CNN. Uh so basic structure of MLP. Now we actually discussed in uh previous tutorial. Now if you're not very much comfortable with the structure of CNN's or you want to have a revision of the CNN structure uh we you can look at one

**[02:44]** uh resource that is uh the Stanford University CS231 course. you can uh look at uh this resource. Uh so here you can go to some useful notes in which they have curated a very interesting notes and then you can look at the overall neural networks. They have explained each and

**[03:15]** every idea using uh a very nice examples as well as a detailed notes has been uh provided. Now if you are not so much uh uh comfortable with CNN's you can go to CNN nodes and then here they have uh given a detailed overview of the convolutional layers how do they work now what do you mean by the idea of parameter sharing what what are features now given an input how do you calculate

**[03:45]** now this is how the operation of convolution happens what are the essential parameters that are has been uh described in detail in this resource. Now you can look at this resource. Now CS231N Stanford uh course on computer vision. Now they have uh elaborately given a very nice they have cured a very nice notes. uh you can look at these

**[04:14]** notes then uh uh look at the idea of convolutions and the idea of neural networks all together. I'm assuming that these prerequisites are known then we have given the reference material. Okay. So now coming back uh to pytorch we will be using the standard pytor tutorial available in public. We will be just going through that and then explaining you where much more

**[04:42]** elaboration is needed and then we'll be running those codes whenever needed and then at the end of this specific tutorial what we shall be doing is we shall be writing a simple structure to read some of the data sets that are there and work on those data sets. Okay, now that is a overall idea without any further delay. So let's start into it.

**[05:14]** Now how do you get to this page is just uh in your browser of your choice just type pyarch tutorial. Now you will be going to this is the official pytor tutorials or you can go to this link pytor.org tutorials/index.html. You can go to this. This is the basic tutorials. It has tutorials on most of these ideas but we'll be going ahead with a very simple basics. We'll be learning only we'll be

**[05:43]** looking at only these basics and then uh the extensions of it uh I'll you can actually look at it and it has uh on all these ideas. So now let's start with quick start. Okay. Now what it has done is they have given for each of these tutorials they have given either you can run it on collab or you can download the notebook and then you can use uh your own local machine or the visual studio or the

**[06:12]** jupyter notebook that you have you can use it or you can look at uh GitHub but what I shall be doing in this tutorial is we'll be running on the collab so that you are much comfortable with uh the collab environment as well and then you are comfortable with uh the idea of tensor the ide of py doge as well. Okay. Yeah. So this is uh kind of summary of everything. So we don't no need to go there. Now let's go to tensors. Okay. I'll be opening it in uh the collab.

**[06:49]** Okay. So this will open a collab thing. Collab notebook. Okay. So the collab notebook that is there will have these kinds of dialog boxes. Now this is uh having the extension ipy nd which means interactive python notebook. Okay, this is the name of the file. So where this uh so whenever uh you are running any instruction or running a code now it has to physically run on some device right now where is

**[07:17]** the device now the device is not present in this is this is not using your local resources. Now what is happening is in the Google server now it has given you a chunk of a resource and then you'll be running your uh code in that specific chunk. Okay. So now this uh removes all the problem of installing the packages maintaining those necessary packages. All those hassles have been removed. Now we will not

**[07:43]** be worrying about any of them. So we shall be directly uh going ahead and uh learning now what is necessary and then what I assume is you know Python. Okay. Now that is the assumption. Now if you are not so much comfortable with Python now I urge you to go ahead and look at simple course on Python uh in which uh so you can learn the basics of Python. So now now what are tensors? Now

**[08:14]** tensors are the building blocks of this overall PyTorch. Okay. Now, now you can think of them as some kind of a specialized structure. Okay. Now, which are very similar to the arrays or the matrices uh that we have. Now, if you have worked with numpy now it has something called as ND array. So, n dimensional array. Now, this is similar to that. But what happens is now this tensor which is there now it can run on

**[08:43]** GPU and other hardware accelerators. So now GPUs are the one uh which gave a huge push uh to the deep learning. Okay. So you can uh run these sensors or the operations on these tensors on the GPU. Okay. That is the added advantage of these using these libraries pytorch which has a simple data structure called

**[09:12]** as a tensor. Now in this first uh notebook that we have so now let's look at how do you create a tensor. Okay. And how do you do some of the basic operations on these tensors. Okay. Now to do so uh the first couple of libraries that are needed. Now whenever you are working on pytorch the first library that is needed is you is torch. Now therefore you import torch. Okay. And then uh so you need data. Now either

**[09:41]** you can create a list and convert it into a tensor or you create a numpy array and then convert it into a tensor. Now for a list you don't need to import any libraries. Now if you are going ahead and putting it as a numpy array. Now at that time you will be needing a uh numpy so you need to import it. Now let me run it. Now this is how you run it. You just click on the play button. Now whenever you are running it for the first time now it will take some time now because it has to establish a

**[10:09]** connection with the server. So it will take uh some time to run. Okay. Now these text blocks now this is text block and this is code block. Now we don't so in the text block more information will be given. Now code block is where you write your code snippets and run. Okay. Now I assume that you are pretty much comfortable with all these basic ideas. Yeah. So now uh you create a data you create a

**[10:39]** two-dimensional array. This is a two dimensional array or it is famously called as a matrix. Okay this is a two dimensional array or a list. Okay. So now what you are trying to do is now you're trying to convert this list or a two dimensional array that you have into a tensor. Now how do you do it? Now you just use the method tensor torch.tensor. You use this torch.tensor. Now to that

**[11:08]** method you pass this data as an argument. Now that will return uh a tensor type data now with the same values in it. Okay. Now the values will not be modified only the kind of data. Now this will be a list. This data will be a list. Now you can check it by using type type of data. You can check it. Now this will be a list. Now you are

**[11:35]** changing the type of this. Let's run this. And this is how if you have a list if your data is a list you can convert it into a tensor. Now on the other way if you have your data as a numpy array okay now cons so in the first line here I'm converting my data which is a list into numpy array. So now I have a numpy array. Now how do you convert your numpy array into a tensor. Okay. So now you use this uh numpy array torch. you use a

**[12:05]** method to do from numpy. Okay, you pass this converted numpy array. Now you get this uh uh tensor which has been created using numpy array. So now this means that you can create a tensor now using uh a list under your uh thing or you can uh do it using the numpy array under your data. Okay. So now now most of the times

**[12:34]** now what will happen is now you will need to create uh a random tensor okay or a tensor with some specific values like I want all the values as one or I want a random tensor now of specific size okay now we have two ways of creating that okay now one is you directly give the uh data now which is of that size okay or you say Okay, I want of this

**[13:04]** size. If you give the data itself, now internally what it will do is it will calculate the size and then it will create a new pseudo data or a dummy data of that specific size. If not, now if you give the size itself, it will directly do a okay. Now let's look at both of these ways. Now you have this X data. Now I want a tensor where which all the values are one. Now what is this X data? X data is

**[13:33]** a tensor. Uh so it's a 2D tensor. Okay. I want a tensor. I want a 2D tensor again. Now when which all the values are one, all the values are one. Now if this data which is there has some specific some other properties even those properties will be transferred to X uh the new X1's now that is being created. Okay. Now this is in which all the values are ones. Now in addition to that

**[14:02]** now if you want uh that okay now if you want random values uh so now then what will happen is you just pass this data and then you're saying that the data type of this should be float. Now but you can see that the data type here is uh integer. Now what it will do is it will override uh this x data now which is in override the type of this x data and then now you will have a 2D tensor now when which all the

**[14:32]** value will be a floating point values we can print them you can see that all are ones you can see the size of it it is a 2 +2 tensor similarly if I run it again you can see that different values are getting generated because you are uh doing a sampling if you're a random thing. This is how if you pass the data similar to that

**[15:00]** structure you can obtain the tensors. Now in contrary to that if you have the shape of the tensor now what I'll do is now what they have done now what I know it's not me who has curated this this is the standard uh things that we are using now they are saying that okay now the shape of the tensor should be a 2 + 3 okay so now I'm so torchand now if you use this method and pass the shape now

**[15:30]** it will create a random tensor now of size 2 + 3 okay now if If you want all ones now then you can use torch once where in which you just pass the shape to this method and then you create all you create a tensor where in which all the ones are there and then torch do.0 so you pass the shape to this method where in which all the values will be zero. Okay you can see this here a 2 + 3 tensor is being created

**[16:00]** here. Now this is a random tensor. The first one is a random tensor. The second one is a tensor in which all the ones are there. The third one is a tensor when which all the values are zeros. Okay. So now that means that you can create tensors using uh if you want a structure if you want a shape specific shape either you can relate it to the uh the the the data that you have and then get it or you can specify the shape and then you can

**[16:29]** get okay. So now then look at some of the attributes of tensors. Okay. So now one is the shape of the tensor. Another is what is the type of each of those values rather and on what device is it running. Okay. See what happens is uh now whenever in your system if you have a GPU CPU has to be present for sure. Now your tensor can be on the CPU or on the

**[16:56]** GPU. Okay. Now depending on that know I can say that okay on what device it is. Is it on CPU or is it on GPU? Let's run this and see. Now the shape of the tensor is 3 + 4. As we can see that it has generated this block and then you can say that okay this is the shape and this is the type data type of it. D type is data type. Now it is saying that it has float 32 and then uh the device on which this uh these

**[17:25]** values are present are CPUs. I'm not still using my GPU. Okay. Now these are uh some of the uh attributes that are there for Okay. So now now let's try to look at uh how do you actually move your tensor into an accelator if it is available. Now we will come back to this. Now let's not do it. Now this is uh uh slicing and

**[17:54]** indexing. Now we will not be needing this slicing and indexing uh during our course. So I'll not be spending much time on the slicing and indexing. Now if you know the slicing and indexing of numpy now it is exactly similar to that slicing and indexing. Okay. Now you can uh look at it. Okay. Yeah. So here the first one uh we can uh look at it. Now you're creating a torch. You're creating a tensor where which all

**[18:22]** the values are ones. The shape of that is 4 + 4. And then you're doing some kind of slicing and indexing ideas that you're trying to do. Okay. So I shall not be going into that. Now how do you concatenate tensors? Okay. Now you have a tensor. Now that means it's it's a matrix. Now how do you concatenate a matrix? Now when I say how do you concatenate a matrix? The question is now do I need to put one next to another or one below another? Okay. Now this will be the

**[18:51]** standard question that is there now which needs to be answered whenever we say that okay now go ahead and take a tensor and concatenate it. Okay. So now let's look at this thing and then let's see how it is there. Now torch. Now that is uh the method that is used to concatenate. Now when which you provide a list of tenses now here all of them is the same tensor and then you have something called as dimension one. What is this

**[19:19]** dimension one? and then you are printing this. Okay, let's see now how it happens. Now each tensor is a 4 + 4 matrix. Now that is known to us. Let's look at this. Now what happened here? Now you see 1 2 3. Okay. Or maybe I did not run it. So I still have a sorry I still have a 2 + 3. Okay. Now let me run it. Yeah. Okay. Now this is the tensor that I

**[19:47]** have. Now it has 1 2 3 four. Okay. Now it has four rows. And then how many columns are there? 1 2 3 4 5 6 7 8 9 10 11 12. Now yes. Yeah. So it has repeated thric. Okay. Now 4 + 4 next to that 4 + 4. Next to that 4 + 4. So now you have 12

**[20:16]** columns and then you have four rows. Now that means that dimension one now what does it mean is now you're concatenating as columns. Now if you want to concatenate as rows now you just change this dimension to zero. And if you run it now you can see that it has been concatenated as. So now now whenever you want to do concatenation the question that arises is no do you want to concatenate it as columns or do you want to concatenate it

**[20:43]** with rows but those things can be handled with changing the dimension of it. Okay. So now yeah this is one operation that we need. The next operation uh that is needed is the idea of matrix multiplication know which is the femininal idea in most of uh the neural networks. for all practical purposes. Now your whole forward propagation in a neural network can be thought as a matrix multiplication

**[21:11]** actually. Okay. So now how do you perform matrix multiplication is the is the question here. Okay. Uh I assume that you know the basic rules of matrix multiplication. Now what are the kinds of uh uh things? Okay. Let me uh tell you that rules again. Now if you have a matrix, you have a matrix A now which is of

**[21:39]** size M cross N and then you have a matrix B which is of size N cross P and then you can multiply this matrix. Okay. And then you get a matrix C which is uh M cross P. Okay. So now these two things should match. Okay, the columns of the first matrix has the number of columns in the

**[22:07]** first matrix has to match with number of rows in the second matrix. Now then you can perform this matrix multiplication. Okay, that is a standard rule that is used. Okay, so now you are taking a tensor now which is of size 4 + 4. Okay. So, and then you want to multiply. Now, multiply it with what? Okay. Now, you have one matrix which is 4 + 4 that

**[22:35]** you want to multiply. Now, what are the options? Now, you have to have 4 + anything. Okay? Then you can multiply. So, now the tensor now which is a 4 + 4 matrix that is being multiplied with tensor T. Now, what is this T? Now, do T is used to take the transpose operation. That is the transpose operation. Okay. And this at symbol which is there is used for matrix multiplication.

**[23:02]** Tensor now it is a 4 +4 matrix into tensor the transpose of it. Okay. Now tensor is a 4 +4 matrix. So the transpose of which will also be a 4 + 4 matrix. Now you are multiplying a 4 + 4 matrix with another 4 + 4 matrix. The outcome will be a 4 + 4 matrix. Okay. Now you can use at. Now this is matrix one and this is matrix 2. So you can multiply it. This is one method of doing it. Okay. Now then you

**[23:34]** have uh uh another method. Now you can use uh the first matrix dot you can call a method from there from this tensor which is matt null a matrix multiplication which is a soft uh form as matt null and then you pass the argument the second matrix. Okay. So now this is same as multiplying uh the first matrix with the second matrix. Now this

**[24:01]** is one more way of performing the matrix multiplication. The third way of performing matrix multiplication is directly using the library torch. Now it has a method called as matl. Now within which you have to pass both the matrix A and B and not only that in addition to that you should even pass the resultant matrix. Okay. So now that's the reason why you create a resultant matrix just the structure of it torch. Rand like all the random values of Y1. Y1 is this uh

**[24:33]** matrix. Okay, that is y3 and then you are passing it. Now you multiply this tensor and with its transpose and the output will be stored in this y3. Okay, now these are the three ways in which you can multiply uh matrices. Now one is using the at another is using uh first matrix dot matt matmal or you can use the torch domat when which you'll be passing both the matrices as well as

**[25:02]** where do you want to store the resultant as the input. Okay. Now this is matrix multiplication. Now the one more important thing that uh we shall be needing is the idea of element wise uh multiplication. So now what is this uh element wise multiplication? I just take a look at it. Okay. Now let's look at this. Uh now element wise multiplication is

**[25:34]** one of the core operations involved in convolutions. Okay. element wise uh now to perform element wise multiplication. Now A and B should be of same size. Okay. A and B should be of same size. Okay. Now let's take a matrix A.

**[26:03]** Let me put some random values. is I'll just take a simple 2 +2 matrix 1 2 - 2 3 this is one matrix and the second matrix B which should be of the same size again 2 + 2 I'll take 4 2 1 3 okay let me not put three let me put

**[26:32]** minus1 now how do you perform this element wise multiplication ation that is C equ= it will be 1 into 4 2 into 2 -2 into 1 3 into -1 okay you take each and every element and then you perform the multiplication take this element multiply with this take this okay and then you'll be able to get the resultant of this okay this is the

**[27:00]** idea of element wise multiplication okay so now how do you do this element wise multiplication you you have a tensor know which is of size 4 + 4 that we have already created into tensor you use this uh star symbol here okay this is one or you can use tensor do mull not mattl matml is matrix multiplication so I presume that you know now you are pretty much clear with the idea of matrix

**[27:29]** multiplication and how is it different from element wise multiplication okay or element wise product as it is called as okay. Now the first matrix dot null now you call the method there. So and then you pass the second matrix as an argument or as we saw earlier now you pass both of them torch.mull you pass both of them and even you pass the resultant. So okay now these are some of the basic operations

**[27:57]** that you perform on the tenses. Okay let me run it. Yeah, you can look at it comfortably. Okay. So now with this uh I assume that you know the basics of tensors know what are the attributes of the tenses and how do you create a random tensure a tensor with all ones and how do you perform the very basic operations of uh product uh element wise product and the matrix

**[28:25]** multiplication. Now similarly you can do addition subtraction other things and those are trivial tasks that's the reason why they haven't elaborated on that. Okay. So now that completes the first idea which is the tensor.
