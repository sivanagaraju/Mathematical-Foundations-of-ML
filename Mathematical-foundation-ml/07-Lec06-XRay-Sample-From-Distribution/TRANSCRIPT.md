# Transcript — Lec 06 Understanding a Chest X-Ray  as Sample from Distribution

> **Source:** https://www.youtube.com/watch?v=bdcvsSNAHIk  
> **Channel:** NPTEL - Indian Institute of Science, Bengaluru  
> **Duration:** ~26 min  
> **Note:** Auto-captions cleaned lightly. Minor ASR errors possible.

---

**[00:03]** Welcome back. So now uh let's try to get that example uh grounded in in these uh these under with these notations and languages. I'll do that and then we will go to the density functions. Okay. Now recall that in the previous uh session I told you that from a probability perspective, probabistic perspective, the entire process of uh uh imaging and

**[00:33]** assigning a label to it can be seen as assigning a random variable. And once we do the measurement the actual measurement that we do okay uh the one that we get in our uh in our computers they can be seen as elements from the range space of this random variable that we are looking at. And there always exists an underlying probability distribution function or the probability measure uh which has been imposed by the random experiment. Remember that right?

**[01:01]** You remember that. Okay. So now let's consider an image. So let's take an example. of dimensions P cross Q okay P where PQ are arbitrary numbers okay let's say that we have this image

**[01:37]** so remember that an image is represented as grid of PQ numbers in a computer. Okay. When you read an image using OpenCV or any any kind of softwares, what you get is a matrix two dimensional grid with PQ numbers and uh there are PQ pixels as each of this uh each of the entry in this PQ grid is called a pixel and there are PQ pixels. Now this uh for our purpose okay is can be seen can be viewed

**[02:12]** as a R If you stack all the rows of this image and make it a long image long long vector this is an R this is a vector in RPQ dimensions and this PQ is what we are calling as D in general anywhere is a Dimensional vector make sense okay

**[02:51]** any data point Okay, data point is a d-dimensional vector. Okay. So now in the probabistic viewpoint, this RD is actually the range space of the underlying random variable. Which means every data point that we

**[03:17]** have or rather every image that we see okay is an element in the range space of the underlying range space of some underlying random variable that we are interested in. Okay. So uh every image or every data point in the

**[03:48]** rain set or range space of a random variable. You you understand now question &gt;&gt; I don't get how it is I mean it is a rectangle &gt;&gt; okay see the question is how is that a

**[04:16]** vector I mean see uh you represent an image as a rectangular grid of PQ pixels but what you can you can you can view that as a vector by stacking all the rows of this matrix one after the other so you RPQ or or rather PQ number of uh pixels. So you you take PQ is is your D. So uh uh an image is converted into a D- dimensional vector.

**[04:43]** &gt;&gt; Mapping it to &gt;&gt; uh not mapping I mean it is I mean you you are viewing that as a PQ dimensional vector. That's it. Yeah. See in fact as we move further in the course I'll show you uh that you know all all the data points that we get in practice right uh an image I already gave you an example right of how a text can be viewed as a d-dimensional vector and we will we will see more examples as we move on but to concretize the idea let us now start with one data type which is an image and

**[05:11]** which is a d- dimensional vector for all practical purposes but what's important is the following right so every data point is an element in the range space of a random variable this is the the take-home message this is viewpoint that I want you to have question. Yeah. &gt;&gt; Can you also explain like how would the event space of that particular random variable? &gt;&gt; Good question. I'll tell you that's the next thing. The question is what would be the corresponding uh event space or rather the sample space of the underlying experiment. Okay. Uh it's

**[05:41]** open for interpretations because I told you that uh the uh the random experiment is something that the user conceptualizes. there is some experiment that's happening and you know you you get uh some outcomes. So these outcomes are observing these images. Okay. Now if you see uh an image as a d-dimensional vector in the range space

**[06:08]** of a random variable what is that random variable doing you know intuitively is that it is measuring the likelihood of getting a particular value for a particular pixel. That is what is it is measuring. So it is telling you that okay let's let's okay for ease let's say let's view this as d scalar valued

**[06:37]** random variables for now because I told you that both of them are equivalent right a d- dimensional vector valued random variable is equivalent to having d scalar valued random variables now let's say that you have one random variable and suppose you have what is called as a single pixel camera where you are only getting one pixel now what is this measurement giving you this measurement is telling you when you do the measurement you the the surface will give you correspond to some part some value of the pixel right. What is this telling you? It is telling you suppose I

**[07:05]** measure I I I take the picture of all of you sitting here. Okay. And suppose there are 100 of you in the class and I get 100 values. Okay. So these are 100 realization of the random variable. Right? So basically I'm getting 100 points from the range space of the underlying random variable. What is it telling me? So this is telling me that when I conduct this random experiment, what is the likelihood of observing this particular

**[07:36]** value from that experiment that I'm conducting? That's what the correspondence is. Make sense? Right? So as I told you, why are we doing this? Remember that we are doing all this jugulary because we don't know the physics of the problem. The whole point is repeated experiments and use statistics to model. Okay. Now this uh random variable uh is giving you the measurement. Okay. Yeah. A little bit of correction in what I

**[08:05]** said. The the probability measure the underlying probability measure or the distribution function is telling you what is the likelihood of obtaining this particular pixel value when you conduct this random experiment. That is what it is saying. &gt;&gt; I may extend this. So all this Q vectors would be stack one by one &gt;&gt; and one of the pixels would be ranged from 0 to 25 and normalizing to 0 to one

**[08:33]** &gt;&gt; doesn't matter yeah let's say that you normalize see the value that it takes doesn't matter actually but let's say that you normalize so what &gt;&gt; I mean without normalizing how would you say that it's a probability &gt;&gt; oh see okay I see the see the values of this vector is not the probability So remember this right? Every data point is an element in the rain space of random variable. The element in the rain space is not a probability.

**[09:05]** Please be very clear about this. &gt;&gt; It's a real number. &gt;&gt; It's a real number. &gt;&gt; It's a real vector. It's a real number. The element of the r because the range space of probability uh sorry range space of random variable is always real numbers. So the actual value of the data point that you see does not correspond to the probability at all. It it has nothing to do with probabilities. What is a data point? A data point is an element from the range space of the random variable. That's it. Is all of you clear about it? This is

**[09:34]** very very important. Okay. But be since there exists an underlying sample space, there's a probability measure that we have defined on the sample space. Always remember that the probability measure is defined on events. Okay. So when you have the random variable, you know, you can talk about the inverse image of this random variable in the

**[10:02]** event space on which the probability is defined. Okay. So the values that the data points take has nothing to do with probabilities at all. They can take any value. It need not be zero because it's it's just RD. It's just real numbers. You can take any value. Is that clear to all of you? It's very important. [snorts] Okay. But yeah, so this is every data point is an element in the range space of random variable. This is this is important. The other important observation is that the the distribution let me call this of a random variable X.

**[10:37]** Okay. The distribution function likelihood

**[11:02]** of under X view X as some observations right so when you observe

**[11:30]** there's a likelihood of this because every observation that you make so know every every single element in the range space of the random variable corresponds to an event or rather I mean that's not uh exa mathematically correct if you have continuous random variables it's the you should take a subset of uh the values in the or you should you should consider a subset in the range space every subset in the range space of uh the random variable corresponds to has an inverse image in the &gt;&gt; sample &gt;&gt; sample space so that has an underlying

**[11:57]** probability measure so what that probability measure talks about is that what is the likelihood of seeing this particular ular event, right? What is an event in this particular case? The event is obtaining that image event is obtaining that image. So what do you mean by obtaining an image having these particular pixel observations? So the distribution function okay ultimately tells you what is the likelihood of obtaining this particular

**[12:26]** image under this random variable. Do you see the point? Okay. So the moral of the story is the following that if you can estimate the distribution function. Okay. Then you know everything about the underlying sample space. See that's why they say that the

**[12:55]** distribution function completely specifies the sample space because if you know the distribution function right you can know everything that that you are supposed to know about the underlying experiment. Okay let's write that. So distribution function So this is the key observation.

**[13:36]** Okay. Distribution function completely specifies the underlying sample space. Do you understand why is this clear? Because this is very important. Take a moment to understand this if you don't ask questions. So now these uh again I'll trace back. So there's an underlying experiment random experiment that's happening. There's a probability measure that we have defined on top of uh the sample space and when we measure we assume that there's an there is a random variable a

**[14:04]** function that will take the elements of sample space and maps to real numbers and that's what we are seeing and that gives rise to that translates the probability measure onto the distribution functions. Okay. Now what does the probability measure tells you is that what is the likelihood of obtaining this particular event? That is what the interpretation of the probability measure is. Now the same interpretation holds true for the distribution function as well. So distribution function gives you tells

**[14:31]** you what is the probability or what is the likelihood. Let me not use the word probability. What is the likelihood of obtaining this particular image if I may say that. Okay. So now if we know what the distribution function is that is generating or rather uh giving rise to this this this particular observation then we know everything about the sample space. Now this is the statement that we'll be

**[15:01]** justifying and defi I mean uh creating algorithms for in this entire course. This is now this entire course is about estimating distribution functions of different forms given observations from random variables because that will tell you everything that is need that that you need to know about the underlying experiment. So now an engineer has to design the scenario in such a way that his or

**[15:30]** her problem fits into this narrative in the sense that you you you tell me what the underlying sample space is. Okay, you identify what the underlying random experiment is and then design your data or get your data accordingly and design your problem accordingly and then you put an algorithm to solve that problem. Okay. Now if you can see okay so let let me tell you you know one other uh point just to drive home the message. Now we said that in this example there

**[16:00]** is an image right there is an image uh which can be seen as a d- dimensional random variable. See when I say an image can be seen as a dimensional random variable. Please remember what am I saying? Okay. I mean that an image is an element in the range space of a function called the random variable which has been defined from sample space to real numbers. That's what I mean. From now on

**[16:30]** I'll just say that any is a random variable. That's a wrong sentence. Yeah. But I will still say that assuming that you understand what I mean. Okay. And if you see in some books or you know some some research articles or somewhere where people are saying that consider a random variable and the data that we have are observations from random variable. This is what they mean. Okay. Right. So now we said that an image is is is is is the image is an

**[16:59]** element from the range space of a random variable. But our problem also had labels. Do you remember because the entire random experiment was that somebody was somebody's x-ray was taken and some label was assigned to it. How do we model this? We model this as joint distributions. So label is considered as another random

**[17:27]** variable that is attached on top of or rather another random variable that is defined on the same sample space. You see the point right? So in this example so distribution function completely specifies the sample space is a true sentence. Right? So suppose you have you know one other uh random variable that is defined on the same sample space. See calling it a label is an interpretation.

**[17:55]** Mathematically it is another random variable that you are defining on the same sample space. So the entire experiment is an image is being taken and and a label is being assigned to it. So that entire thing is now modeled as two random variables. One which corresponds to getting the image the other that corresponds to assigning a label. Make sense? So typically so this is only a typical thing. So that is why I I'll tell you typically

**[18:25]** a label defined

**[18:55]** on the same sample space. This is the typical case. Okay. uh see just uh uh I I told you in the uh in the first class that uh I'm a generalist right I like seeing things from a very very abstract uh manner right so you will see this now you know I I hope that some of you will

**[19:23]** appreciate this see I don't you know I generally define terms like supervised machine learning unsupervised machine learning and so on okay to me everything is a special case of distribution estimations I'll tell you why even the so-called supervised machine learning can be seen as unsupervised machine learning. I mean it's just an it's just nominature I I'll just tell you why they are they are this they are same they they are so but yeah

**[19:51]** so before I come to that so you understood right typically what happens is that you model the label or an additional uh take another random variable and call that a label that's just a semantic nomature okay then what happens is so in this scenario &gt;&gt; so do we like map that to RD as well &gt;&gt; good question I'll tell you in this scenario X is in RD

**[20:36]** and Y is in R K. Okay, this is the most general definition, right? Y can be in RK. So you you want to have binary labels. You make your R to be one, right? And all your values are coming from a subset of R which is between 0 and one or you can even make it discrete,

**[21:03]** right? Or you can make it a set of this thing, you know, close 01, right? Now suppose you know somebody's doing this somebody's taking an image okay and the task is to uh no localize okay a tumor in an image. So there's an X-ray okay and

**[21:33]** there is a there there can be possibility of some tumor in that X-ray. Your task is to identify where the location of the tumor. Now, how do you identify the location of a tumor? The typical way to do it is that you put a box around it. Okay? And you give the coordinates of that box. Every rectangle in an image can be represented as uh represented using

**[22:02]** three four numbers, right? The center, the height and the width. Okay? Now there what happens is there's an image and three numbers corresponding to the the location the height width and the center of the box. So in that case your k is three make sense that's it. So if if you have

**[22:27]** you know two labels 0 and one then your k is one and you have zero and one. Suppose you have instead of having two labels suppose you have five labels then what happens? five categories then your y will become discrete 0 1 2 3 4 5 it's not RK it is it's this set right it is whatever this thing is right x and y

**[23:09]** they are the co-domains or range of the function that we are talking about. So basically what's happening so I'm I'm I'm repeating this 100 times because this is the most important thing that you that that has to get across once this idea is clear you know everything else is only an algorithmic uh extension on top of it but this is the key understanding that I I want you people to latch on to. Okay is this clear? Okay it can just be true value

**[23:36]** &gt;&gt; it can be yeah discrete 01 two or whatever. See these these numbers uh they don't have any significance per se right I mean unless they correspond to some measurement or something typically in a discrete case it's just some categorization right so know belonging to A or B or four five categories and so on is this clear now what happens is so because we have two random variables there's a joint distribution

**[24:06]** okay now the the nomclature is The notation or So uh learning I have not defined what

**[24:37]** machine learning is yet. Okay. I will define that later. Starts with X3 Y3 and so on

**[25:20]** X and Y. So, so you start with a data set. Okay, you have n images and all of these images have some labels. Okay, so I can I should also say this. So x is typically called the the data space or feature space or data space. Y

**[25:52]** is generally called the the label space or action space. label space I I I'll come to that just a second. So you have data data set which is a set of tpples of sort tpples of form x i comm yi and you have n of them. So it is said that this is sampled. So there's a tilda

**[26:22]** that is written. is sampled iid from an underlying distribution joint distribution pxy. Okay, this is the conventional nomature. So I've introduced a new term here iid. Okay, what is this iid? We'll see in the next lecture.
