# Transcript — Tutorial 15 : VAE and Beta-VAE Implementation

> **Source:** https://www.youtube.com/watch?v=f_X2vwIXVz4  
> **Channel:** NPTEL - Indian Institute of Science, Bengaluru  
> **Duration:** ~41 min  
> **Note:** Auto-captions cleaned lightly. Minor ASR errors possible.

---

**[00:01]** Hello guys, uh welcome to this section of tutorials in which we'll be looking into the implementation of uh latent variable models variational autoenccoders. So uh I'm assuming that all of you are uh gone through the the theory contents that were discussed in our classes and you know the basic idea of vaes and then the problem with ves

**[00:29]** which is the posterior collapse and then how does the regalized version which is the beta ve it solves this problem. See either you implement beta is equal to 1 that will be your vanilla va or you put some beta value so smaller beta value so your reconstructions will be sharper so that choice of beta value is up to the user depending on what is the

**[00:57]** trade-off that he want to do he can go ahead and do it fine so uh let's look at the implementation of ves I'll leave that beta value as a hyperparameter So it's just one additional thing that needs to be put into your code. Apart from that the overall training everything remains the same. Now I'll give you a very brief overview of the overall training procedure of the VAE and then we will uh jump into the implementation of the same. Okay. So now

**[01:28]** uh yeah we are looking at into the VAE the variational autoenccoders. Now we already know that how does it work? You have uh the x space, you have the z space. Okay. So you take from uh x to z. This is what we known as the the encoding process. Now which are parameter base q of z given x and then uh we have the

**[02:00]** decoding process. This is the the decoding process. Now we have uh p theta of x given z. Okay. Now these are the parameters that uh now we'll be looking into. This is the latent variable models. So so now how does the objective the elbow looks like? Now we have this uh J theta of Q that is the expectation with respect

**[02:32]** to the the posterior Q of Z given X log of P theta of X given Z minus a KL term now which is Q of Z given X this is the prior P theta of Z. Okay. Now we can insert some beta value here.

**[03:04]** Now it will become the beta V when beta is equal to 1. It's uh it's vanilla VA. Okay. So choice of beta I leave it. So so just for the sake of completeness I have put this beta here. Now all our uh implementations that we have we'll be taking beta is equal to 1 and doing it and I'll request the viewers of this video to take the code. Now the link of the collab notebook will be there in the

**[03:33]** description of this video. You can take it modify this beta value and then uh see how does it impact the generations. Okay. So now what do we do? Now how does this the whole thing happens? Now you have uh the generator network which we are giving it by QV which takes XI as an input and then it gives the the parameters which is mu of X I

**[04:05]** sigma V of X I and then what do we do? We sample from the normal distribution 0 I and then we get uh Z I which is uh by shifting and scaling. Now we'll be obtaining this. Now why are we doing this reparameterization? Now I'm pretty much sure that all of you know why we

**[04:34]** are doing it. Okay. So if not go back to the the wonderful discussion that we had in our theory sessions regarding reparameterization which is a beautiful in itself. Okay. So we're taking the the hadamard product. Okay. Now here this is not the regular dot product that we have. These are the hadamad product. Now because we are dealing in vector spaces. So these are these are all not scalers. These are

**[05:02]** vectors. So we'll be taking element wise multiplication which is known as the hadamar product. Okay. So then uh you take this zi as an input. Okay. We have this p theta you get the the parameters of p theta of x given z. See the overall ideas that we should

**[05:30]** remember is uh now during this we we are assuming that p theta of Z is normal 0 I this is the one assumption that uh we are making we are assuming that the latent the prior now which is there now we are considering it to be Gaussian. So uh and then uh so to sample from this either you can take this

**[05:59]** itself uh we are assuming that the output p theta is normal x now which is uh the mean is the parameter that it is giving so this if I represent it okay I'll just write it as p theta of x given z Now no whatever is the output okay not the the the the parameter of this comma

**[06:27]** I is what uh we are assuming it here. So either you can have a dist have a normal distribution and sample from it or you can obtain this is the mean that you are getting and then obtain itself as a sample. You can go ahead and do it. It is it is up to the the person who is implementing. Now we'll be imple we'll be considering that the output itself is a sample and then uh we'll be going ahead with it. Okay. So now how does the overall training happens

**[06:57]** as uh seen in the equations? The overall idea is we will be given a sample x i now which belongs to the data d. Now what do we do? We we pass through pass through the generator and then get the parameters mu of

**[07:25]** x i and z sigma v of x i you get these things uh and then what do you do then we sample epsylon 1 epsylon 2 so until some epsylon which are from a normal distribution. Okay. So this zero what do I mean is it's a vector of zeros. Now I is the uh you decide the dimension of this latent

**[07:54]** and then that will be let's say that I have decided that the dimension of the latent is 20. This I will be an identity matrix with 20 + 20. Okay fine. So now then now once you have this now you have access to this mu and sigma. Now we can obtain so by reparameterization obtain z1 z2 zone tillal zm. Now v reparameterization.

**[08:33]** So obtain this via reparameterization. Now and then what do you do? to pass each of them each of these Z's okay pass pass Z1 Z2 till ZM and then compute no pass through what pass through decoder okay pass through decoder and then compute minus 1 / m

**[09:05]** summation j is equal to 1 to m. See when you pass each of these z's you'll get a different p theta of x given z. Okay. So then you take the average of it. Okay. This is what you do. Now then these this is the obtaining the first term. This is the reconstruction term

**[09:33]** now which you are obtaining for the first term. Now then you take the the second term which is now you consider the the KL term can be computed. So what is this? This is half into log of minus k is the dimension plus

**[10:05]** trace of this Okay. Now this will be a scalar. It is evident that this is a scalar. Trace is a scalar. Norm is a scalar. This is the dimension. This will be a scalar. Okay. All of them will be a scalar. So therefore the scale term will be a scalar. Now and then the MSE loss will also be a scalar. So you

**[10:34]** can add them and then you can get the loss. Okay. Now this is how we obtain it. Now there is other way one other way of doing it. Okay. So the other way of doing it is rather than using MSE loss we can have a small trick and then we can actually use BC loss. Okay. So now what we can do is now this trick is very nice. So now we can use

**[11:06]** BCE instead of MSE. Now how is this possible? See we are dealing with uh the black and white the grayscale images. Now therefore each pixel so let's say that you have some pixel uh let's say that okay PA has already taken uh let's say some pixel s so it is

**[11:34]** indicated by i j now this will be a value between 0 and 1 correct okay see now please remember while doing GAN we explicitly modified this to take values between minus1 and + Now if we remove that normal so what the intensity for 8 bit will be between 0 to 255 you normalize them and you get

**[12:02]** between 0 and 1 okay that will be the actual value now this can be thought of as a probability correct now let's say that the value is si okay so now we can assume that this sig is from a barnoli distribution. So this is what is known as a continuous binary distribution. So which will give a value between uh 0 and 1. Now then the P

**[12:39]** this I given Z now this you can say that as Xig J hat this is from the generated value to this. Okay. Uh sorry, sorry, sorry. Uh small mistake. Now SI is the generated value right now SI to the power of XIG. This XI is your okay SIG this is this is uh

**[13:14]** this is generated okay for the same pixel XI. Now this is the actual value. Okay. Now you can take this 1 minus of So we can write it like this. And then

**[13:40]** if you take the log of P of Xig given Z this I can say it as Xig log of SI plus 1 minus sorry there has to be 1 minus here right here

**[14:06]** also there has to be 1 minus yeah this is 1 minus - xig into log of 1 - sig. Does this ring a bell? Okay. Now, if you take the negative of it, is it not the BC loss? So, now what is that? No, no, no. Just

**[14:34]** let's go back to the term that we want. See what do we want? We want the log of B theta of X given Z right. Now you can directly compute it using BC loss. Okay. Now this whole thing can be computed using BCE if pixel values are

**[15:01]** between 0 and 1. Now we can easily compute it using BC laws. Okay. And taking an expectation that's fine. Now we know how to take care of it. Now therefore we can implement this as BC laws rather than going ahead and uh implementing in this particular idea. See each are equivalent. See you have to do this exact step. Now if you are dealing with

**[15:30]** uh three channel images so you don't have any other option over there. But now since we can assume uh the the pixel value to be a probability now we can invoke the known idea and then we can use the BC laws and work with them that is what in this implementation I'm going ahead and doing okay. So what is the network architecture that uh now we have to go ahead and use see the

**[15:59]** we have encoder and decoder. Now we have uh 784 again uh we are considering uh uh MLP here. So that I'm converting it into 400. Now this has to give now uh the mu and then as well as sigma. Okay. So and then we are assuming that sigma is only a diagonal and then it is only giving you the diagonal elements. So now this will give mu

**[16:30]** and then it will give zigma. So I'm assuming that the latent dimension is 20 and then going ahead and then sampling uh so your z then uh your z belongs to r 20. Okay. And then uh you get Z which is 20 that I convert it into a vector of 400 that to 784. Okay. This is the structure. Now this is

**[16:59]** the the encoder structure remember uh now we are representing all these parameters as theta. We are not saying that encoder parameters is uh one another. So we are updating them all together. Okay. In one backward pass unlike GAN. No we don't have uh updation of uh uh two sets of parameters. We are assuming them as one

**[17:28]** set of parameter and then going ahead with this. Okay. So now let's see this. Now let's see its implementation with these much of things in hand. So we are implementing VA is uh okay. uh I'm not even going to utter any of these things. So I'm just uh excluding these loading of data set and other things. The only thing that I want you to explicitly remember is uh no we are keeping it 0 to one. I have commented it well enough so

**[17:56]** that you can read them. Okay. So we are not normalizing them. Okay. So because why are we not we are using Bernoli style reconstruction loss which is BC loss. Okay. So as we discussed earlier so we are using them. Okay. And then we have data loaders. So then some of the images I'm just visualizing them the standard thing. So now let's define the VAE. So X which is 784. So I'm taking input dimension to hidden dimension now which is uh 7804 to 400. See and then

**[18:26]** please remember the output that I'm taking into consideration is log of sigma square. Okay. So now why are we doing it? So it will be evident. Now why are we doing it? Now remember the KL term you have log of your sigma square right here the first term therefore if you want sigma you take the exp of it okay so we will be doing it in some times

**[18:58]** okay so to get the mu you have hidden dimension 400 to 20 to get the log wire it will be 400 to 20 Okay. So then the decoder so latent dimension which is 20 to hidden dimension 400 and then 400 to 784. Now this is the encoding process. So now uh the encoder now will give us mu and log y. Now this will define the

**[19:30]** normal distribution with uh Q QV of Z given X is defined here with normal mu and this will be a diagonal sigma square thing okay so and then you have this so and then you have this log so we are still using the logar we are not converted and we haven't obtained these samples see the input that will be given as mu and this is for reparameterization the input for reparameterization will be given as mu and log square.

**[19:58]** So now what do we need? So we need Z from mu and sigma square. Okay. Now we reparameterize it. So mu + epsylon into z sigma. So this is hard product again. Now since we have log which is log of sigma square. So how do you get the the sigma? Okay. So sigma square is exp of logar. If you want sigma so exp of 0.5 into logar. This will give you the sigma. Now therefore

**[20:27]** see this is just the small step that we are going doing. So this is how we are getting the standard deviation sigma okay torch of exp of 0.5 into logar. So now then uh you can sample from the uh 01 so random like so which is of the same size as the std. So you can either would have given mu also it would have been the same. So because we are assuming that uh the zigma which is there is a diagonal uh uh matrix. So you

**[20:57]** upper triangular and the lower triangular values are zero. Okay. So now then you get the z. Now this is the whole idea of reparameters and the only thing that you should remember is you are getting logar. So and then therefore you want to pass it through the exp to get the sigma. Okay. So you're taking exp of 0.5 into logar for the whole reason that you want sigma not the sigma square. Fine. So now then how do you decode no X? So you want to reconstruct them. So FC2 and then which is 784

**[21:29]** sorry 20 to 400 and you're using ReLU activation and then 400 to uh 784 now which is you're using the sigmoid activation. Now why are you using sigmoid activation here? This whole reason is you want the reconstructed things to be between 0 and 1. Now because why do you want it to be between 0 and one now? because we are using binoli uh uh distribution we are using BC loss and therefore we want it to be like this now how does the whole forward

**[21:56]** process look like first you get the image then you flatten it out and you encode it now you'll get the parameters Q of Z given X which is mu and log Y and then you reparameterize and get this Z and then you pass it to the decoder to get the reconstruction term so uh to get the reconstructed value and then this X which is we know that This is the parameter. Now we can sample it. But as I told you, we are considering that itself as a sample and then going ahead. Okay. You can consider it as the mean of

**[22:25]** the sample wherein which I to be you can take I as the standard deviation and then you can go ahead with sampling from that distribution also. That is perfectly fine. But uh we'll be considering this itself as a sample. Fine. So mean is also a sample from the distribution as we all know. Okay. Fine. So then uh okay what do I mean by mean is also a distribution now I mean that we are assuming uh [laughter] so Gaussian distribution under the case

**[22:53]** okay so no please don't take it into consideration that I'm rolling a dice the expected value is 3.5 how is it from the distribution okay please don't ask that question [laughter] so I'm saying that under the assumption that uh the underlying distribution that I'm considering is a goian distribution which is true in that case okay that's what I mean by that okay fine So then uh you're creating the model. So latent dimension we are considering it as 20. So the hidden dimension we are taking it as 400. And then we construct

**[23:22]** the model. So your uh 784 to 400, 400 to 20 again 400 to 20, 20 to 400, 400 to 784. This is what we have. Fine. Now then let's look at the the loss. So how do we compute the loss? First is the reconstruction loss. I have already told you that we can consider the reconstruction loss to be BCE. So F.Bc. You take the reconstruction and then you have x and the reduction is sum. Okay, you are uh you have an expectation. Now please remember the expectation of log

**[23:51]** of p theta of uh x given z is what we have. Now therefore expectation you want to take the summation. So therefore the reduction is what you're using it to be summation and then you are considering the kl divergence. So between Q of Z given X and then the latent prior know which is 0 I normal Z I then 0.5 into sum of uh the mu² + sigma square plus this okay no no yeah

**[24:21]** okay leave this term you know what is this term okay so this is torch dot sum so you will get a vector you are taking the sum of individual value so therefore the total thing will be the reconstruction loss which we have plus the KL term. Okay. So this is what the total loss is and then we will be having only one optimizer as I told you. So then uh the training function. Now

**[24:48]** here we don't need to worry about the saddle point problem. So first you get the images clear of the gradients pass those images through the model. Now in which the model will encode it reparameterize it and then get the decoded value. You get the reconstructed images. Compute with that reconstructed images and the original images that we have. Compute the the loss terms or the three loss or the two two loss which is reconstruction

**[25:15]** loss and the KL loss and then you add them up using that loss you go ahead and backrop optimizer.step and then uh you take care of remaining things for bookkeeping. So we will not worry about it. And how do we evaluate? Okay. So now the uh way to evaluate it you take the uh test set which is there and then you pass them and then you obtain the loss for that that's how we are evaluating it you to get a score

**[25:45]** okay or you can sample the Z from the zero I and then you can pass them and then get the FID score that is also another way of evaluation that's perfectly fine either one you can use okay now then you're you're training the whole thing for certain number of epochs. We have training it for 20 epochs and then you're taking the you're bookkeeping the the things. Okay, you can see that uh the loss is uh decreasing.

**[26:14]** So and then you plot the loss curves the same old bookkeeping things. So this is the train loss and then the test loss. Now you can see that both of them are well behaved and similar in terms of the structure and how does the reconstruction loss and the K loss are working. So reconstruction loss you can see that it is decreasing and then the K loss you can see that there is a slight increase in it. Okay. Now this is the reason why you had to bring in the uh beta term and then work with it.

**[26:44]** Okay. So now let's look at the original and the reconstructed uh uh images. So now what is that we are trying to do here? You take the test loader. You get these images and pass it through the model. you get the reconstructed images for the same images. Okay? And then you're plotting them together. So you see this for this seven which is given this is the reconstruction for this two which is given this is the reconstruction. And please remember these are reconstructed images. These

**[27:11]** are not generated images. Okay. And you can immediately see here the reconstruction are bit blurry. Okay. That is pretty much evident from the reconstruction itself. Now let's generate. Now how do you generate? Now you can sample because of uh the KL term which is there you can sample from 0 I so and then you can pass it through the decoder of the model and then you change so it will be a 784 dimensional

**[27:41]** vector. You change it in 28 cross 28 and then you can plot them and these are the images that got generated. Okay see once you see these images it is pretty much evident that these are pretty blurry images. I can see that this is some sometimes it is three sometimes it is this three. So the images are quite blurry now this is pretty much evident. So and we know the reason also now why is this change the value of beta and see how does these blurry images gets rectified. Okay. Now putting the value

**[28:12]** of beta you have no uh problem with uh using BC laws or if you want you can implement it using MSE laws also perfectly fine. See no problem. See the MSSE laws will come into picture whenever you are assuming that P theta now which you have is actually a Gaussian distribution. Now rather than assuming it as a Gaussian distribution you're assuming it it is from a continuous Bernoli distribution. Now what we get is this BC loss. Okay

**[28:41]** fine. So now let's uh see how does the latent uh distribution is there. Now what I'm trying to do here is I'm uh trying to get the uh test loader. Okay. I'm taking some five images of them. So then I'm changing this and I'm getting the mu and then zigma and this is what I get. These are the for an given example this is the mu and

**[29:10]** this is the zigma and these are the z which is there which is perfectly fine. So I just want to demonstrate a simple reparameterization trick. See what I want is between the mu which is two and then standard deviation that I want is 0.5. So I'm taking n samples. So I'm just getting the epsylons. So I just perform mu plus z sigma time. So as you can see this is the histogram of the

**[29:38]** reparameterized value. So we are sampling epsylon from 01 and then you are scaling and shifting them because of which it is evident. See uh people who have taken a course in probability theory should feel it very trivial. Okay. So just for the sake of uh that uh I'm showing uh [clears throat] then I we tried to do a very good uh simple exercise just to see how does this

**[30:06]** latent place a value. What did we do here is see latent dimension earlier we took it as 20 we took it as two now because if it is two we can think of it as xy and then see you can modify the values of the latent and then see how does the images gets generated. Okay, for this specific value of latent and this specific value, the first dimension of latent and the second dimension of latent how is it? So we can actually see uh the uh a grid of images. So we just tried it out. So we have

**[30:37]** model 2D now input dimension is 784 latent dimension is 400 uh the hidden dimension is 400 the latent dimension that we are taking it as two. So we have optimized it the optimizer and then we have trained it. So you can see that uh the loss is decreasing. So now then what we did was we wanted to visualize this 2D latent. Okay, how does this look like? See this is what we got. These are the 2D latents.

**[31:06]** Okay, now you can see here that this is zero. Okay, this is zero. You can see zeros are uh here. Okay, six is here. 8 is nearby. So you can see that in the latent dimension the structure is similar. Now you can see the grouping. So these are when which lines are involved is where the clutter is. Now if the lines are not involved even in 2D latents it is pretty much

**[31:36]** what do you call well separated. Okay see this is a simple digits so we can do it. But if you take uh any higher dimensional data know which is some natural images. So maybe we may not be able to see that but at least in 2D data we can see this separation pretty well wherein which uh very very which are the digits for which the cluttering is there which are the digits for which cluttering is not there it's pretty nicely visible from this particular

**[32:07]** diagram okay so this is uh the 2D latents now as per the uh the labels now we have put them into consideration so now then what we tried did was know we have a grid of latents and then uh we obtained this. Now you can see it here. This is so you can uh modify the value of the latent and then you can see the generated images. This is for the first

**[32:36]** dimension of latent is this and then in the y this is in the y and then you can get it in the x. Now as you change it so you fix this and as you change the the latent dimension two you see that the image getting generated. Now earlier you get one then you start uh getting nine and then you start getting three. You fix a some x and then change the y. See nothing is happening. You get the same thing.

**[33:04]** Okay. But fix a y and change the x. You see this? Okay. Now may maybe this is even more nice. You can see it here that how does this uh you start with three and then you change it you get eight is coming here and then post six is coming here and then it is becoming zero. See it is somehow able to take the curvature and then uh the straight

**[33:35]** line into consideration. As you change the x, see the curvature is taken into consideration. As you change the y, some kind of straight line is taken into consideration is what visibly it is saying. Okay. See a becoming two becoming six. Now you can see this. Now these kinds of understanding of latent can be done visually. Okay, the these these are know visual. I'm I'm not saying anything about this is what it

**[34:03]** represents but this is a very nice indication that as you change the latent you can see a change in the generated images. Okay. So now the code is available. So I want uh the viewers of this video to go ahead and look at this and uh make a much more uh analysis of this. And we did another interesting experiment where which we took two

**[34:31]** digits. Okay. So image A and image B. You took them and then we started getting the linear combination of them. So you flattened it up for those two images. You got the mu of them. Okay. So then what did we do? We started interpolating them for alphas between 0 and 1. So between 0 and 1, we got 12 numbers for each of the alpha. What did we do? [snorts] So 1 minus alpha into mu aa

**[34:59]** plus alpha into mu b. This is the linear combination. Let's say that I have so 10 alphas 0.1 0.2 so on and so forth. Now this is 1 - 0.1 this is 0.9 * mu plus uh 0.1 * mu of the b. So that means it is a combination of the mus of these two. Okay. So this is the linear interpolation and then we decoded it and then we are uh plotting this images. So

**[35:29]** now as you can see here now let's [clears throat] say that the images that were t taken were uh 7 and two. Now you can see it here as you change the alpha see how 7 is getting converted into two. Now this is the case where in which this is let's say that this is mu a and this is mu b. Now this is the alpha. Now here is uh zero and here the alpha is one. Now like

**[35:58]** this we can see a traversal of think. Okay. Now you can interpolate and get things. So now uh deterministic reconstruction. So what did we do here is the experiment is quite simple. So this experiment shows we we just get these uh x and y's. So we got this mu and

**[36:27]** the log Y instead of sampling from Z we just kept the Z to be the mu of it. Okay it's possible no so from our earlier this is possible we kept the Z to be the mu of it and then we reconstructed it. Okay and then we plotted it now that what does it eventually means you're giving the the embedding itself. So you can see this this is well mapped. Okay. On the other hand, we did another

**[36:57]** case now where in which you have we we are sampling from this Q of Z given X which is the posterior. Okay. You give the posterior whatever you got. You get the Z posterior you reparameterize it and get it and then you generate image using those posterior and the other thing is you sample from Z and then you pass it. So now and then we are displaying it here. As you can see

**[37:27]** here this is the the posterior uh uh we are getting and this is from the prior. Now what does this indicate? See it is able to reconstruct properly but it is not able to sample properly. Okay which is very much evident from this. Now why? Because of the KL term. Now we have to regularize it is what all these experiments what is that we are saying that we have to regularize it well enough. Okay. So and then so what

**[37:57]** is the latent statistics? So why is it needed? Because we know the distribution of the latent. Now therefore if I take the latent statistics so for each of the test of the image I can get mu and then uh the log and then I can compute the mu of this and other things. So it is seen that this is the the latent uh uh the average of the posterior mean and the the average of the posterior variance. So we want it to be 0 and one. Yes, it is approximately

**[38:25]** okay near to zero I can say but this is not near to one. Okay. Now what does this mean? Now this actually means that there is some kind of a uh still the training needs to be done more. It needs to be more regularized which is evident from all our examples. See like this we should be able to perform much more of experiments to understand now what is each of these latents are doing. So I urge all of you to take this notebook and then try to play with this

**[38:56]** and try to understand how does the latent manifold which is there can be taken into consideration and then rather than plotting all 10 digits simultaneously. Now you can take two two digits and plot it and see whether uh in the latent dimension there is some kind of separability which is there. Okay. Now this I urge all the uh viewers of this video to do and then in addition to that uh it will be we will be very happy know if you can generate these images

**[39:25]** and put it in the comment section of uh uh these videos. I hope VAE was helpful. So we are able to implement it changing the value of beta. So it's it's very simple. So I can just show you where you have to put this. Okay, if you go here. So whenever we are uh doing the training, okay, I should have shown you there itself but anyhow. Okay. So you have this loss. So you're getting

**[39:54]** these losses, right? The train loss reconstruction and KL term. No, now this place know where in which we had this loss. Where is that? So VA loss so okay so one function inside here see here you can see it here that there is a reconstruction loss and then the KL loss you just put one more variable beta and then change the value of beta and then uh you'll be able to get the beta v so you can get the beta v

**[40:25]** like this and perform the similar experiments and then for different values of beta see how the uh values are how the reconstructed images are changing now that will give you a fairly nice idea of what are the parameters to tweak. So whenever you have uh things in hand. So tweaking what will yield what should be able to get it and train for longer uh uh duration. For the sake of demo I have trained it for some 20 30 epochs but for standard trainings. So 20

**[40:54]** 30 epochs is very minimal. So you have to train it for longer amount of times to get good reconstructions. So I hope this VA tutorials was helpful. In the next tutorials uh we will be looking into vector quantized VAQVE and its implementations and then we will proceed to the state-of-the-art diffusion models. Okay, with this I conclude today tutorials. I hope it was helpful and enjoyable. Thank you.
