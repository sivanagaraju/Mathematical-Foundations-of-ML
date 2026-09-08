# Transcript — W1_T2: Tutorial 2: Introduction to pytorch: tensors

> **Source:** https://www.youtube.com/watch?v=L5n4rNrLZ_8  
> **Channel:** IIT Madras - B.S. Degree Programme  
> **Duration:** ~18 min  
> **Note:** Auto-captions cleaned lightly. Minor ASR errors possible.

---

**[00:12]** [Music] So now uh all our data that we get now will be in the form of a tensor or it has to be you know we have to convert it to the form of the tensor. So now how how where does the data lies and how do you handle it? How do you how do you get this? And then you want to get it in terms of batches. Okay. Now for uh the updation of weights. Now how do you do all of them? Now is the next idea that uh we shall be looking into now in the second uh

**[00:42]** tutorial. Now which is data second tutorial sheet uh sorry second uh tutorial link here which is data set and data loaders. Let me go there. Now here uh now what they are doing is um before that now we need couple of data sets. Now we need to look at this to data

**[01:16]** sets. Okay. Now what are the data sets that are available in Wait, this the one there is a Yes. Yeah. Okay. See these are some of

**[01:53]** the data sets that are available in PyTorch itself. Now when it is available in the PyTorch repository itself now you don't need to worry about much. Okay. Now when the most of the times what happens is data will not be present here at that time we should know how to create a data loader as it is called as. Now these are some of the data sets that are available. Caltech 101 Caltech 205 256 CL AC far 10 so on and so forth. Now we have lot of these image

**[02:22]** classification data okay imageet and stuff okay so and then you have image detection or segmentation data that also you have image pairs video classification video prediction so on and so forth you have enough data now you can play around these things now here now we are using a data called as fashion emnest now fashion emnest is a group of 10 has it's a 10 class

**[02:52]** problem when which you have 10 kinds of cloths. Okay. Now I'm not sure about what are the kinds that are there but for all practical purposes you can say them from 0 to 9. Now 0 is something one is something. Okay. Now you can assume that now let's look into how does it work. Okay. So now now what are the necessary libraries

**[03:20]** that you want to load? The first library is the torch itself. Okay. Now then torch.utils.data. You have to get uh you have to work with get the create a data set. For that you need this data set. Okay. You can see this capital D here. And then to download from the repository you need this package called as data sets. Okay. And then now once you have got the image you need to convert that

**[03:47]** image into tensor for that you need this uh transform. Now that is what is called as transformation. Now you need to transform it to a tensor. Now for that you need two tensor and then you'll be plotting just to see how how the images look. For that you need this pipelot which is matt.lip.piplot as plt. You're aliasing it as plt and then you'll be using it. These are the necessary libraries now for our work. Okay. So now uh as uh you are very much aware any

**[04:17]** data that is there is uh has two parts to it. Now one is the training data now when which you try to understand the relationship between input and output. The other is the test data which is used to evaluate uh the relation. Okay. Similarly this also has two parts of it. One is the training data and test data. They have been made separate. Okay. So now our aim is to take them. Now where is that available? Data sets dot fashion

**[04:46]** emnest. Okay, this is the name of this thing. Okay, fashion emnest. Now uh root equals data. Now what does it mean is now where do this data is present? Now it is present in the folder called as data. That means that in the current working directory it is present in the folder data. Now I'm looking at the training data. Okay, training data and then download is equal to two. If the data is not present in

**[05:14]** this specific folder, what do you do? You download it and then once you have downloaded the image, okay, you apply this transformation. Now, what is a transformation? Now, it's an image. An image for all practical purposes is nothing but a matrix. Now, you convert it into a tensor. Okay. Now, similarly, you do it for the test data also. It is present in uh the root know which is from the working directory which is the data. Now if training is false now that means that

**[05:42]** you are getting the test data download equals to true. Now that means that if the data is not present in this particular route you download it and then once you download the images you convert them into tensors. Okay that is the idea. Now let's run this. It will take some time. Yeah, it will uh take some time and

**[06:09]** meanwhile uh this is used to just show the images. Yeah, but we are not interested in that. So I'll not be going ahead how pipel works how how do you create this access and other stuff I'll not be uh interested in that. So I just wait till this happens. Yeah. Done. So, and then uh it's code

**[06:43]** strip to visualize. Yeah. This is a code. This is a dress. Yeah. These are different uh what do you call labels that are there code dress handle for all practical purposes? No, you can number them. You can say that this is from 0 to 9. You have 10 classes marked from 0 to 9. Okay. Yeah. Now this is a good case. Now why it is a good case? Because your data that you need is present in the PyTorch repository

**[07:12]** itself. But in most of the cases we might not be so lucky. Okay. Sometimes there will be some custom data or some sensitive data customer sensitive data or some confidential data might be there. At that time your data that is there now may not be in the repository of PyTorch. Now at that time now you should create this uh data you should curate this data on your own. You should write wrapper script around it to get

**[07:41]** it. Let's see how to do that. Now that is called as data custom data set. Okay. So now now whenever you have this uh custom data set now it is having uh two parts to it. Now what are those two parts? Now let us uh go there. sets. Now you have this root directory

**[08:21]** that is called as uh image directory. Now this will be an OS path. Okay. and then you'll be having something called as annotation file. Okay. Now this will be preferably a CSV file. Now what it will have is it will have uh the relative path. Now from the image

**[08:52]** directory how do you and the image name. Okay. So real relative path and the image name or uh whatever data we are not pretty much concerned about the modality here. It can be uh sequence also and then it will have the label. Okay. It will be a CSV file with two columns. Okay. Now this will be

**[09:22]** dot CSV. Okay. Now using this now we should be creating a data set. Now this is the folder and then from that folder now what is the image name? Now what is the label of that image that will be there from here. Now let's look at how do you create your own custom data set. Okay. So now for that uh uh you need OS.

**[09:52]** Now why do you need OS? No, because you want to retrieve the file from its path. For that you need this uh library OS and then you need pandas. Now why do you need pandas? Now you'll be reading the CSV file and then you'll be working with that. Now to do that you need a pd data frames. You'll be needing data frames. So that you need pandas and then to read the images you need this torch vision.io you need this read image. Okay. Yeah. So now uh let's uh I'm assuming that you

**[10:21]** are pretty much comfortable with object-oriented programming. So what do I mean by that is you know what is in it? You know what does uh double underscore mean here. Okay, all those things you are comfortable is what I'm assuming. Okay. So now and then you are pretty much comfortable with inheritance uh is what is being looked into here. You're creating a class which is with the name custom image data set. Now which is a child of this data set. Okay. Now what is this data set? Now you can

**[10:49]** look at it here earlier. This is what is used to create a data set. Okay. Which is child of this. So and then your uh in your init method your constructor you have self which refers to the instance itself and the annotation file. Now this annotation file is what I said as a CSV file. know where in which your uh all the image names and then the relative path or the image name and the labels will be there

**[11:17]** and then the image directory the root uh to that and then transform. Now this is do you want to apply some kind of transformations on the input data on your features on your X target transformation do you want to apply any transformation on your Y on your label they they have been uh uh given as none okay so now self dot image labels now uh now how do you get

**[11:47]** it pd dot read csv now you get you read this whole of the annotation pi and then uh you self dot image directory. Now this is a path. Now whatever transforms you have is being initialized in the constructor method and then these two methods should be there. These are the compulsory methods. The whole thing should be almost as it is. Now the length now should it should be able to give uh the length of uh image labels.

**[12:17]** Now image labels is nothing but the CSV file which is there. Now whatever is the length of it now you should be able to give. Now that means that so many items are there that is the the overall length. Now how do you get the item? Now now how do you get the item is via its index. Okay. So now let's try to understand this. Now for this get item method. Now you're passing this index. Now now what is this index? Now in a CSV file now I'm saying

**[12:47]** that now this which example are you looking at? Okay. Now this is what is told here as idx. Okay. Yeah. Which example which row you are looking into. Okay. So and then image path. Image path. OS.path.join. So you're joining two paths. Now what are the two parts? Now

**[13:15]** one is image directory. Now that is your path and then image labels which is nothing but your annotation file dot eyelock. Now that means IO is refers to I need to access rows. Now which row you want to access. Now you want to access the idx. Now the index now which index you want to access. Let's say that I want to access this index of zero. Now that means that the relative path now that means that I'm joining this image directory which is there. I'm joining

**[13:43]** this image directory with this relative path. Now that means that now I have for an image I have a full path that is there. Okay. Now this is your the first thing that is your image path. Now once you have got the image path then what do you do? You read that image. Okay. You read that image that is what? Read image from the image path. Then what do you do? Now you have got the image. Okay fine. And then you want the label right?

**[14:11]** Now where is label? Now in this image labels now which is an annotation file in the same in the annotation file that you have in the index now the first column that you have is the label you get it okay idx1 that means that you get the label now if there is any transform that is uh need to be done self transform you transform the image no using that if

**[14:38]** there is any target transformation you transform the label using this uh target transform which is uh specified and then you return the image and the label. Okay. Now this is a very simple code snippet that is uh used here. Now most of the times we will be needing this custom image data set. Now you'll be needing this specific uh thing and then we'll be giving you custom data sets so that you can work on so that you

**[15:06]** should be very much proficient with how do you uh get this uh data whenever you have uh uh the data which is not present in your pytor repository. Yeah, the same thing has been explained here. Okay, so now you have got uh the data. Okay, good. Now you have curated the data. Now good. So now then what? Now then you need to create something called as data loader. Okay. Now to create the data

**[15:36]** loader now first you take the training data which you have fetched from the source and you are creating a batches of 64. Now that is what I told you. You are creating a batches of 64 and shuffle equals to true. Now this is an interesting uh argument. Shuffle equals to true. Now that means that uh every time you shuffle the batches okay every time now it will not be in the same order. Now every time you shuffle the batches okay now that will

**[16:06]** give you the uh train data loader and similarly test data whatever the batch size you want to specify and then whether you want to give shuffle is true or not. See in test loader it is not mandatory that you should give shuffle equals to true. It makes sense if you give shuffle equals to false. because you'll be evaluating only once. Okay, how so you will not be having multiple epochs over the test data. So now it doesn't mean anything now when you give

**[16:34]** shuffle equals to true in the the test case in the test data. Yeah, you can run this and then you can access uh these using iterator next. Now when you use this you'll be able to access the each block. Okay. This is just plotting it. Yeah. So

**[17:00]** now what has happened is now now your whole data that is there because the outcome of this your data has been uh divided into d1 d2 so on till some dk. Okay. So where in which each di will be xj comma yj it will be a tpple xj y j is equal to 1 to

**[17:31]** 64. Okay. Now you have considered a batch size of 64. Now each of them will be a batch of 64 as per okay. Now this is how now now you have curated the data. Now you have obtained the data and kept it in the form of a tensor. Okay. Now, now what is that we have done? Now data is ready. This is the step that we have

**[18:04]** done now. Okay. So now let's look into the next step. That's all ish from this uh sheet. Okay, let's go into the next one. Let me leave this. I don't need data sets also. Okay.
