# Transcript — Tutorial 13 : Wasserstein GAN (WGAN) Implementation using Gradient Clip

> **Source:** https://www.youtube.com/watch?v=p0wXSsTnmw0  
> **Channel:** NPTEL - Indian Institute of Science, Bengaluru  
> **Duration:** ~32 min  
> **Note:** Auto-captions cleaned lightly. Minor ASR errors possible.

---

**[00:03]** Hello guys, uh welcome to this section of tutorials uh where in which our main aim is uh understanding and implementing the versrain GAN. So in our earlier tutorial discussions now we implemented different variations of GANs. Now we implemented vanilla GAN. Okay. So when I mean vanilla GAN I actually mean that we

**[00:30]** are using uh MLP so multi-layer perceptron for uh the generator and then the discriminator and then we used convolutional blocks to do the same which we state as DC GAN or uh deep convolutional GAN. We implemented them and then uh we implemented conditional GAN using MLP as generator and discriminator and then uh using convolutional blocks as generators and

**[01:00]** discriminators and it was pretty much evident that the uh GAN training is a saddle point problem and then it was evident even from the examples that were generated. So now uh in the uh the theory discussions that uh sir has done now it was uh evident that using a versus stain metric

**[01:29]** would be a better idea and then we got the objective. Now what we will be doing in uh today's discussion or in today's tutorial will be we will be seeing how to implement this versus train GAN. So let's proceed now. Now we remember that the objective of GAN. So we remember the the GAN objective.

**[02:08]** So which is we have two sets of parameters. So we have theta star and war. Now which is arg min with respect to theta and then max with respect to w. We have expectation px log of w of x plus expectation with respect to

**[02:37]** p theta log of 1us dw of x and uh see I want to bring it to your notice that uh now what is the distributions here. Now here it is px. Now therefore these will be the samples from the data that we have. Now this is from p theta. Theta is the generator.

**[03:05]** Therefore this x will be the samples uh which are generated. Okay. So I'm assuming that uh from the notations it is pretty much clear. and then the dw of x. So we want to look at it as a discriminator whose uh value will be between 0 and one. We are having a sigmoid function at the end of the discriminator and know because of which

**[03:32]** uh this will be the case. Okay. So and then this will be approximate to one if the x is from px. Now it will be zero if the x is from p theta. Okay. So we want to make this discriminator fail. So what do I mean by that? It should be giving uh irrespective of uh the given data it should be giving 0.5 as the output. That is when we say that uh we have cracked

**[04:00]** the idea. Okay. So and then we see that this is the minmax problem now which is evident. So now we have the W GAN objective. Okay, we have the W GAN objective. See, now let's uh look at the W GAN objective. The objective. Now here also we have two

**[04:36]** sets of parameters. We have theta star and then we have the war star. We have arg min of theta. So max of we have this function whose norm has to be less than one. And then we have uh expectation px

**[05:10]** dw of x minus expectation p theta tww of x. Now this is what we have. And please uh uh uh remember that the outcome of this is a real number and then uh we have uh no sigmoid function in place here.

**[05:40]** Okay. So now and then we know that this we have to have a lipshit constraint. So we should have uh one lipshit constraint. Now the question is how do we implement this one lipshits constraint. Okay. So now let's look at it. We have this uh Now one way of doing it is what is known

**[06:19]** as weight clipping. So what do we do here is now post train uh updation. Now we will see to it that the weights of the network is uh between some bound it's between some values minus val something and the positive of the same thing. If I take 0.1, it will be the values will be between minus0.1 and then plus 0.1. Those are the only

**[06:47]** allowed values. And the second method is what we known as the the gradient penalty. Okay. So, uh as of now in our theory discussions, we only looked at the weight clipping idea. Okay. So now we will look at the gradient penalty when we implement and then I'll explore more on it. So now let's revisit the objective and then look at what is the

**[07:18]** objective. So we have two sets of parameters. One is for generator and the other we don't call it as a discriminator. We call it as a critique network. Okay that's that's the terminology that is being used. Now what is the objective that uh we are going ahead with that. So now recalling the objective GAN objective we have uh theta star and w star. Now just uh we have arg min with respect to theta and then uh we have max with respect to okay I should

**[07:50]** not say it as w it is t okay which has to satisfy uh certain conditions okay so one okay so then expectation of PX uh TWW of X minus expectation with respect to P

**[08:22]** theta TWW of Xhat just to indicate that these are generated uh things. Okay. So now what is the objective? So we will have uh critique objective. Now what do I mean by that? So now we want to obtain the optimal values for w. Now what is this? This is uh maximize

**[08:50]** expectation of px tww of x minus expectation of p theta tww of xhat. So now but uh now you should remember while implementing our standard packages allows us to minimize. So we want to maximize this and we we know the trick. Okay. So now therefore we take minimize

**[09:19]** expectation of P theta of TWW of Xhat expectation of PX. So now and we know that these are uh obtained using sample averages. So now you obtain the generated images take get the score take the average of the score. So since they are generated and then uh even though we don't want to use it we

**[09:46]** are calling it as fake but anyhow so the fake score the average fake score minus the average real score is the objective. This is the loss for which we have to be updating the the parameters of W. Okay. So, and then how do we update the parameters of uh theta? See the generator uh

**[10:17]** objective. See here the idea is uh we want to look into the theta. So theta is only there uh okay so if we go back now if you look at it the first term that we see here this is uh independent of theta. Okay so here the x hat is there. So know this x hat is some g theta of z. Okay. Now

**[10:50]** therefore we have only the second term. We want to minimize the negative of the second term. Okay. So now now this will be we want to minimize the negative of the the second term. Okay. Therefore you get what do you mean by this? This is fake scores. You take

**[11:18]** the negative of the uh fake scores that we have obtained from the the critic network. Okay. So now now please remember these two objectives. Now we will be looking into the exact uh same objectives implemented. So unlike uh uh the standard discriminators that are used in GAN now where in which we will be obtaining a sigmoidal output. Now here what does the output of the critique is a real number. Okay. So now

**[11:49]** now let's do it let's look at the implementations. uh we'll be using the collab and then uh the link for the collab notebook will be there in the description of this uh video so you can make use of it. the first cell. So uh I think we have reached a situation uh where in which we don't need to explain any of these terms. We have been using them. So standard libraries I'll just for the completeness sake I have just put it or

**[12:17]** uh you can uh uh take it from the earlier implementations that are there and the device. Okay. So CUDA device how do you enable it and other things now here uh by loading the data I want to explicitly say a couple of things. See earlier when we are implementing GAN we used tan h okay I think you remember the whole idea the same uh

**[12:48]** idea now we will be going ahead so we'll be utilizing the same idea now where in which we have used tan h and the value the outcome of a tan h activation is between minus1 and +1 so since we are retaining the same idea now Here also we'll be going ahead with the normalization which we did similar to what we did in GAN. Okay. So I think uh I'm assuming that you remember those

**[13:15]** discussion otherwise please go back to our uh vanilla GAN and DC GAN implementation where in which we have explicitly dealt with it in particular detail and creation of data loaders. So similar uh and then uh we have visualized the data. Okay. Now the similar ideas so I don't want to go into this and the the hyperparameters so the latent dimension the Z that I'm taking is 100 100 dimensional vector the number

**[13:44]** of epochs that I'm training it for is for 30 epochs I'm changing the uh learning rate to five 10 ^ of minus 5 and then the optimizer that I'll be using is RMS prop okay so we have used SGD we have used uh Adam I'm using RMS MS prop I'm leaving it to the uh experiments of the uh viewers of this video. See change the learning rates change the optimizer that is there

**[14:14]** and see whether it has impact on the generated images and then we know how to compute the impact of the generated images. We use the FID scores. Now again I'm assuming that you know how to do it. So or you can uh uh look into our previous uh tutorials uh uh uh in detail where in which we have looked at the computation of FID scores. Okay. So which we will not be going ahead in this case.

**[14:42]** Now we are doing one thing which is uh explicit here. Okay. Now which is earlier we trained GAN with one updation of uh the discriminator and then one updation of the generator. Okay. Now we can have uh uh uh cases where in which we train the generator five times and then we can train the discriminator only once. Now

**[15:11]** the reason is we want discriminator uh uh not to be very strong. Now otherwise the objective can go wrong. Fine. So but here it's the other way around. Now why is it other way around? See the generator gradients. So whenever you are computing the generator gradients see the generator gradients will be strong only when the discriminate the critique network is able to give us good values. Okay. So

**[15:42]** therefore we want critique to be optimized more. Okay. See this is not your sigma output. it will be giving a real value. We want the critique which is there to be updated more times. Okay. Unlike the GAN case where in which we don't want the discriminator to be optimized more, we want the generator to be optimized more. It is the other way around. Why the generator gradient is only useful when the critique gives reasonable values.

**[16:13]** Okay, critique should be trained perfectly. Now therefore now we will be going ahead and training the critique more in this case. Now we have chosen to train the critique five times here. Okay. And then we'll be using the weight clipping. So so which we discussed minus C to C. Now the clipping value that we have considered is 0.01. Okay. So now the generator network. So the input dimension is 784. We are

**[16:40]** dealing with mnest and the latent dimension is 100. So n.c are sequential. So first you have uh 100 100 to 256 and then the layer 2 256 to 512 I'm using ReLU activation 512 to 1024 and the output image 1024 image dimension which is 784 and then we are using the tan h uh as the activation at the last now because we want the output so this

**[17:11]** is the generator the output will be an image so So the pixels of the image should be in the range of minus1 to +1 which is similar to what we had in the images. Okay. Earlier this is our generator network which give which takes the latent value and then it will give you the parameters from p theta. Okay. So then the critique now the critique now will take the emnest image okay the generated or the one. So it

**[17:40]** will 512. Now here the activation function that I'm using is leaky relu with 0.2. Now again I pause it here and then uh I request the viewers of this video to change this particular value to 0.1 or 0.3 or 0.25 or 0.15 values like this and then see whether it has any impact on the generation of images and then uh uh 512 to 256.

**[18:09]** So and then since it's a real number so it is 256 to 1 is what I'm doing. So please remember I'm still using the MLP for the case. I'm not using the convolutional network. Okay, you can change the structure of uh the generator network and the critique network to work with convolutional blocks. And please remember we keep on repeating the neural networks are function approximators. Now depending on what you want so you

**[18:39]** can use CNN's, you can use RNNs. Now if you are uh somewhere down the line we'll be using transformers. Bring it in and use it. Now these are all different architectures as we keep on repeating. Now remember one thing. Now these are all has to be looked into the form of Lego blocks. Okay. The only thing that we should take care is the dimension of input and the dimension of output. You take care and put any function of your choice. Now we are using MLP just for the sake of uh simplicity and uh since

**[19:09]** we are using the GPU which is freely given by Google. So we don't want to go ahead and uh uh make computationally higher ones. Yeah, CNN's are computationally better. I agree to the point but uh uh I have just used MLP. We can you can for sure use CNN's or you can use transformers or any of your network of your choice. Okay, that's perfectly fine. So now uh the critic network will takes X or uh the real or generated image and

**[19:40]** then it will give you the score. Okay. And then you are creating the instances of generator and then the critic network and then porting it onto the device. Okay. So and then uh so we are performing the weight initialization. Okay. Now why why are we doing this? Now please remember that uh so we are looking into uh what do you call uh one lip constraints. So therefore for

**[20:07]** us uh the weights to be in a particular range is quite an important thing. So rather than it initializing a random weights so we prefer to initialize it. How are we doing it now? First for each of we take the module we take the model and then for each of the layers know if it is an instance of n.linear linear since we are using NN dot uh we we are using linear we are using this okay now if you are using convolutional network please modify this

**[20:36]** accordingly to be the instance of nn.com2d okay accordingly so then uh so nnit normal so we are initializing with normal so all the module weights now wherein which mean is uh with zero and then standard deviation uh is very small which is 0.02 02. So therefore the value so the the weight clipping value we have taken it to be 01.

**[21:04]** So therefore no it will be in between the respective weights and then the bias now you are considering it as the zero. Okay. Therefore it at the initial level itself it satisfy the one lip constraint which we have. So you are initializing the weights of generator and the critic network here. Okay. And then the optimizer that you are using is RMS prop root mean square propagation. So I'm assuming that the viewers of this video knows what is uh root mean square

**[21:34]** propagation and how the gradients are obtained. How does the averaging of uh the magnitude of the gradient happens and how does uh Adam better than RMS prop. I'm assuming that the viewers of this video are comfortable with those ideas. Now else you can look into it. So how is SGD? What is the problem of SGD and how does RMS prop solve it? So when and how is uh the idea of momentum comes into picture and why is the idea of

**[22:01]** momentum better and uh in during what scenarios we have to use what kind of optimizers. Now I'm assuming that viewers of this video are quite comfortable with those ideas. Now if not I urge them to look at it. The reason why we are using RMS prop is no actually from the authors of the paper they have used RMS prop just to uh put that we have using it so there is nothing uh uh specific there now that's the reason I initially told that we can you can

**[22:29]** modify this into Adam and use it you already have access to the notebook I urge the viewers of the video to please do it okay fine so then uh the critique locks now remember so this is uh the maximization term Now which is f is the t that we have used we here has been given as f. Now the the real score minus the fake score since we want to minimize it. Now we have taken the negative of it. So you get this. So the fake score

**[22:58]** dot mean minus real score dot mean. Okay. So I know fake scores are not the good one but I I'm assuming that uh because of our long interaction you know what do I mean in this case. Now similarly the generator loss. Now you'll be you want to maximize this value. So and therefore you'll be minimizing the negative of this. Okay. As we discussed earlier okay you can okay you are minimizing this the

**[23:28]** negative of the fake scores. Okay fine. So and then uh let's look at uh the training of the the critique network. Okay. So when I mean by the critique network, so we want this particular objective to be satisfied. So now now let's look at it. So you want the the real images as well as the generated images so to be able to update or to be

**[23:59]** able to get the objective working. Okay. So now we have uh for that you need real images. So generator critique and the optimizer for the critique we need. So first I get the bat size. So okay and then whatever real images that we have we are reshaping them so that the images becomes a vector and then we are obtaining uh the noise

**[24:28]** okay which is sample from the standard Gaussian distribution. Now we want it to be same as the batch size and then uh uh the latent dimension now which is 100. Okay. And then uh now here I I want you to again remember there is no hard and fast rule that the bat size needs to be same. Okay. Now if you go back to our earlier discussion now we can have the different bat sizes perfectly fine but just for the sake of uniformity we are

**[24:54]** doing it. Okay. See these are the choices. See whenever we are implementing all these things see most of the times we we see them as so different from the equations we might miss all these ideas see the reason why I'm reinforcing all these things again and again is that we should not think that the mathematics that goes behind these implementations and then the implementations are disjoint okay so they are almost 90% overlap will be

**[25:25]** there so for some engineer ing things we will be having some kind of diversions here and there which which will not be a obstruction for the the mathematical nuances which are there. Okay for engineering we will be just uh deviating here and there but I want them I want the viewers of the video to be very specific that we are implementing equations. Okay. And then constraints that we have put now we are transforming them into the code. Okay. Okay. So now

**[25:55]** first then we generate the fake images. So we pass the latent values through the generator and then we get it. So and then since we have the we are passing it through the generator. Now it will be because of the computational graph now we will be having the parameters attached to it. Okay. So we don't want to update the generator parameters in this case. So therefore we will detach it. Therefore you have see we just want a number. We don't want the parameters to be attached to this. Okay. Now

**[26:25]** therefore uh you get it. Then you clear off the gradient then you get the real score. So the critique you pass the real images through the the critique network you get the uh scores. See similarly you pass it through the critique again you get the fake scores. You take the mean of them. This will be the loss and then loss dot backward step. Okay. And then the important thing is now now you please

**[26:55]** remember now since you are updating the critique now which is uh the parameters that you are updating is w now you remember you should remember that it should be satisfying one lip shit's constraint. Okay. Now therefore now we have to go ahead and clip the values. So for parameters in uh critique parameters. So now you go ahead for and clip the parameters between minus the clip value

**[27:22]** and then the plus the clip value. So we have kept the click clip value as uh uh 0.01. Now therefore we'll be going ahead with this. So this is enforcing the one lip constraint. Fine. So then uh one generator step. Now for generator now what is that we need? We just need the uh path uh we want the uh noise. So that is what we are getting

**[27:52]** here. And then uh we are obtaining these values. We are clearing of the gradient. Now we have to pass through the critique. Okay. So that we can get the scores. We are passing it through that and getting the score and then we are getting the minus of the fake scores. Please remember there was a minus okay minus of the fake scores. And then now we are performing the back propagation updating uh uh the values of the generator. Okay. Now here we don't need

**[28:19]** to clip now because these are the updation of the values theta the parameters theta. Okay. Uh the constraint was on the parameters of the critique. Now we don't want to do any kind of weight clipping here. Now then the whole of the wand training. So you have all these uh bookmarkers. So the critic laws and other things. So you get the loader, you get the real images and their batch size. So then you

**[28:50]** go through the critique. Okay. So one step through the critique, you get the real images generator critique network and then the optimizer pertaining to that and then it will return the loss. You update the loss and then you obtain the versus uh metric. See uh the objective of the critique and then the versus train metric is a negative uh difference. So I assume that you please look into it. So the real

**[29:18]** score minus f score and then you update it. Now then we update only the generator when uh it is in uh in modulus with the number of times we have we have updated the critique. Okay. See only when the batch size that we have is uh divisible by five now because we have taken this n critique to be five only then we'll be updating the values of the generator. So that means that in one uh what do you

**[29:46]** call one epoch so if we have 10 batches so critique will be trained 10 times but generator will be trained only twice. Okay fine and then we have this average losses. These are all for the sake of bookmarking. So, so we'll be printing all of them. So, we will see the plot. So, this is the versus critique loss.

**[30:15]** It is. So, this is the generator loss. So, it is decreasing and suddenly it is increasing. So, we have to do some some more optimization. and we have to change the learning rate and then we should take care of this and then we will see how is has impact on the images generated. Now then we will generate the new images. Now how do we do it? While generating we don't need the critique network. We'll just sample from the normal distribution and pass it through the generator. And then finally

**[30:42]** you have to resize it. And then now we have to whenever we generate it will be in the the values of each pixel will be in the range of minus11. We'll be converting it to 01. So and then we'll be plotting it and these are the images that we got very bad. Okay. Now why is it I have left it as it is. Now I did not optimize it further. The sole reason why I have left it. So it it looks like

**[31:10]** two. Now why is this happening? Now you should be able to immediately see the reason. Now there are certain things that we can modify and see why this is happening. you know one is change the optimizer change the number of critique step so make it two or three and see what happens will you be able to generate good images will be the the question that we will be answering okay I leave that to the

**[31:41]** implementation of uh the viewers of this video so this is how we implement w GA gan with taking into consideration the weight clipping mechanism. Okay. Now, here is where uh we stop uh this section of the tutorial. In the next section of the tutorial, we will be looking into how does the W GAN know which is there can be implemented

**[32:10]** using what is known as the gradient penalty. Okay. The remaining things will remain as it is. So the only thing is the weights of the critic network. Now rather than using the weight clipping, how do we use the gradient penalty to take care of having the one lip constraint? That is what we'll be looking in the next tutorials. I hope you enjoyed and then resolve the problems uh that has been left to you.

**[32:38]** You have the code, try it out and generate good images. So with this we conclude today's tutorial. So we'll meet in the next tutorial on gradient penalty for vers gap. Thank you.
