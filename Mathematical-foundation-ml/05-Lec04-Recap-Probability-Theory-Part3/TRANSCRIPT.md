# Transcript — Lec 04 Recap of Probability Theory - 1, Part 3

> **Source:** https://www.youtube.com/watch?v=0R6Agp4tqSU  
> **Channel:** NPTEL - Indian Institute of Science, Bengaluru  
> **Duration:** ~29 min  
> **Note:** Auto-captions cleaned lightly. Minor ASR errors possible.

---

**[00:02]** Now what happens to the probability measure? So we had a probability measure P. What happens to this? We don't consider any sort of subset. So when we consider sigma algebra, right? So what sort of subset do we consider? Actually there's a what is the symbol? So we consider particular type of subsets which are called B sigma algebbras. &gt;&gt; Okay. So what sort of uh subsets do we consider? &gt;&gt; Measurable. &gt;&gt; Of course, measurable subset. what sort

**[00:29]** of subsets but in particular we consider subsets that are of the form so if you're looking at R if our range space is R we consider subsets that are of this sort if it's if it's in R so this is a subset in R similar thing in RD right this is the kind of subsets that we consider specifically under random variables and these these subsets are called boral sigma algebbras okay Now

**[00:58]** okay so now what happens is what happens to probability measure. So now what we have to talk about is that so okay so there exists a function that operates on sample space elements of sample space and it maps that on to some element in real real numbers. Okay which means that the subsets of the sample space also gets mapped to something

**[01:26]** under this function and those sets are &gt;&gt; polar sigma algebra elements. Now since we have assigned measures on the elements of f there should be something something equivalent here as well. Correct? So what does that mean? So we this it's like this right? So we have we have assigned probability measure on

**[01:55]** the subsets of f and these subsets get gets mapped to some element or some subset in some subsets of some sub subset of R. Right? Which means that there should be some equivalent in this new space that corresponds to the probability measure. What is it called?

**[02:22]** It is called the distribution function or also the probability distribution function or also cumulative distribution function and so on also called a push forward measure.

**[02:54]** It's that's the terminology that is used. Now the idea is that you have uh a measurable space the triplet and you define a function on top of it and a new measure emerges. So it's called a push forward measure. So starting from so let let's define this. So if you take and my notation is this throughout the course if I write script P with a subset X here I'm saying that this is the distribution function under the probability under the random

**[03:22]** variable X. So if I evaluate this at some small X okay for you know for ease of understanding let me do it for uh one-dimensional I let me take D to be one and do it it easy to comprehend that way and we then we will generalize this to the D dimensions. So suppose I evaluate this distribution function okay px at some x

**[03:51]** what what will this give? This is defined as the probability measure. Okay that is assigned to a set because the probability measure is always assigned to a set. What set? [clears throat] In fact, it's it's the subset of the sigma element of f is what the probability measure is assigned by definition. Okay. Now, if I take a small x and evaluate this distribution function at that small x, what am I saying is? I'm just saying

**[04:20]** first consider a set which is Take a set take a subset of R which is this and see what is the inverse image of this set under the random variable X. What does this mean? Remember that

**[04:51]** random variable is a function from the sample space to R. Now if you take a subset in R there should exist a re in reverse image inverse image of that subset in the sample space a subset of sample space you consider that subset and that that subset has a probability measure right that is what the distribution function evaluates to does it make sense right so this is

**[05:17]** probability of any event A which is equal to which is the inverse image right of the subset minus infinity to x under the function called random variable. what is the probability measure of that particular subset of uh the sample space is what this distribution function evaluates to or I can say equal to right it's better

**[05:45]** notation is a such that it is the inverse image of this particular subset this [snorts] thing see if uh one of you may correct me it is it's it's open at one end right X. &gt;&gt; It's open here, right? &gt;&gt; Yes, sir. &gt;&gt; Yeah. Okay. Hold on.

**[06:12]** Let me repeat. Probability distribution function evaluated at a point X. See what is this small X? What's what sort of uh element this is? This is a real number. This is a real number. So if you evaluate &gt;&gt; it's uh closed at x and open at minus

**[06:48]** infinity right. Ah, this is no this is open here, closed here. Thanks for that. I keep saying this, right? This proves that I'm not an AI or rather not because AI hallucinates too. And you know what? These systems have been trained by using an algorithm called RLHF reinforcement

**[07:19]** learning using human feedback. They have been trained in such a way that they are see because humans like to get confirmation biases. You know what a confirmation bias is right? When we talk to people, we don't seek their opinion. We want them to acknowledge what we say. Most this is called confirmation bias you know all most so because humans seek

**[07:47]** confirmation biases and since the labels are coming from humans if you do RL using human feedback these systems tend to mimic confirmation biases do this experiment ask a simple question to an NLM okay tell that that this is wrong and give a wrong answer it will say yeah yeah I was wrong and it'll say this is right So yeah. Anyway, so here is what

**[08:15]** distribution. So I mean I me making you know uh mistakes and correcting them does not prove that I'm not an AI. Yeah. So at the end of this course we should we should find out what makes us humans. Yeah. Okay. Anyway, so the the definition of a distribution function is that that if you evaluate the distribution function at a real number, first we construct these sorts of sets minus infinity to x and see

**[08:47]** what is the reverse inverse image of that particular set under the random variable that we have defined. So that corresponds to some subset of the sample space and that look at what the probability measure that is assigned to that particular subset and that's what the that's what this probability distribution function evaluates to. Does it make sense? Okay. Yeah. Questions on definition any real number. Now if we have any uh

**[09:20]** interval here how do we know that the image of that will correspond to some like there will be some subset uh &gt;&gt; that's how we have that that relates to the question that he asked right the way you define your random variable is such that the function has to have an inverse image given boral sets when you define random variables you define it in such a way that the boral set inverse image of boral sets always exists Okay.

**[09:49]** Any other question? &gt;&gt; No, we have already defined the probability measure. Is your question how is the probability measure estimated in practice? That's the entire course. The entire uh the central question that we take up in this in this course is given elements from the uh the range space of random

**[10:17]** variables estimate the underlying proity measure by distribution functions. How is it assigned? We don't know. Nature does it or some labeler does it depending upon your problem. If your problem is supervised learning then the labeler see the question that you're asking is how would a doctor know whether a given image is diseased or not. See I have worked with doctors they would say oh this you know they see this look at this

**[10:45]** and you know there's some intuition something they can't put it in words. So we assume that the measure is assigned by nature or the problem or whatever right our job as engineers is to estimate this measure given samples from the underlying distribution. That's what I'll formulate it to be. Yeah. Anything else? &gt;&gt; R2

**[11:12]** R2. &gt;&gt; That's why I told you to refresh your probability theory basics. So the question is how does the the B sigma algebra look in R2 RD? Any guesses? This should have been done in STO, right? How would it look? How would it look? &gt;&gt; Yeah. Cartition products. The cartition products. You take one dimension, the other

**[11:39]** dimension, take the cartition product. That's how it looks like. Yeah. See, that is why you have things called joint distributions and vector valued random variables. We'll come to that in a while. See, now it's all we are talking about is that you know we have a sample space and the mapping is to a real number, right? So what if it maps to RD? Then this random uh the range space what if the range space of this random variable is RD right? In fact

**[12:10]** these ideas are very extendable. No see what you see in one dimension in n dimensions or dimensions they're just the addition products. That's it. Okay is this clear? So now from uh this triplet we got another triplet. What triplet did we get? We got R And what made this transformation

**[12:43]** possible? This is not at all a variable. I just can't see that as a variable. It is a function. Okay, a push forward function that would take one measure space to the other measure space. And we deal with this you know in machine learning. We are we work

**[13:11]** with this. See from now on I will not talk about omega FP. I will only talk about RB and BH. Is that okay? Because the moment we measure something or we we have a sensor and you know have some data as vectors we are operating in the the push forward space and we assume

**[13:40]** that there exists a random variable. Okay. [snorts] So now you see you know you appreciate why somebody needs a random variable in the first place. So it is a random variable is not defined randomly. Right? So there is there's a there's there's a lot of thought that has gone behind this. Does does it make sense to all of you? Okay. Right now Okay. So one important idea uh that we

**[14:10]** have to know before we move forward is that the random variables See, seems trivial but pretty important

**[14:42]** to know. See, we said that a random variable is a function that takes uh omega and maps it to R. It need not map it to R. I've already told that to you. It it may map it to RD. Okay. So now uh in general so in general the random variables have RD

**[15:09]** as their range spaces. Okay. So x is a function from omega to r. See these are also often referred to as vectorred value vector valued random variables I should write that

**[15:38]** see it's again it's a sort of misnomer see what do you mean by vector valued random variables all I'm saying is the random variable is a function and the range space of this function are vectors and they're rd that's it that's a better way to say it isn't it Okay. Now what happens is somebody asked what happens to the sigma algebra in this case they are cartitionian products. So we are looking at if it's R2 then we are looking at sets like

**[16:06]** minus infinity to X1 in one dimension right this is one set it's open here closed here cross if it's a two-dimensional uh cartition plane that we are looking at as rain space then we are looking at these sorts of uh sigma zas And I hope that all of you understand this cross, right?

**[16:34]** Cartition product. What's a cartition product? If you take two dimensional space, right? So uh it's uh if if this is x1 and this is x2, then what space is this? Minus infinity to x1 is everything that is below and minus infinity x2 is everything that is to left. So we are looking at this space, right? U this space.

**[17:06]** Okay, that's what it is. And you extend this idea to dimensions. That's a vector valued random variable. Okay. Now what happens to the distribution function here? So now distribution function gets evaluated. Always remember this the argument to distribution function are

**[17:36]** the elements of the range space of the random variable always correct let me repeat the arguments for the distribution function so if you want to call me that I'll say that the input to the distribution function right is always the elements of the range space of the random variable if the random The range space of the random variable is RD. Then the argument for the random variable

**[18:04]** sorry the distribution function would be RD elements of RD. Right? So a vector valued random variable I mean if I have to use that word. So the input to this is a a sum x which is an element in sorry bear with me when I write this x here

**[18:40]** and x here these two are different. This is the random variable X and this is one element from RD. Right? If you want I can &gt;&gt; see as you as we move forward in the course you will see that I will run out of alphabets and notations too many notations right because you know uh so yeah I I will use capital X for vectors right and uh we I won't

**[19:12]** change this just read this as bold x okay if I write a subscript here write read that as bold Okay. And this is an element of uh element in RD. Okay. Now what is this? If I say that I'm evaluating the distribution function at an X which is an element in RD. What do I mean by that? I mean it should again remember that the distribution function always corresponds to the probability measure. Distribution

**[19:43]** function is a valid measure. The density function is not. By the way, if somebody tells you that evaluating the density function gives you probability, they are very wrong. We'll see that in the next class. If you evaluate the density function, probability density function at a point, you won't get probability because probability is a measure. And because you have defined a random variable as a function on the sample space, distribution function is also a valid measure because it's actually evaluating the probability measure, not

**[20:11]** the density function. This is this is only for you students sitting in the class because I have not formally defined what a density function is which I'll do eventually but anyway probability distribution function is a valid probability measure. Okay. So now if I evaluate the distribution function at some X what am I talking about? I'm talking about the probability of an event which happens to be the

**[20:40]** inverse image okay of the cartesian product that is taken under the random variable in the event space that is the probability that is the event whose probability I'm talking about please note that this event right is a subset in the samp sub subset of the sample space and it can be a singleton set By the way, and in the uh in the X-ray example that we took, it

**[21:10]** can be the probability of that X-ray being diseased. It can be that you get it. Okay. So now what is this? This is actually the the probability. See whenever I write probability I mean the measure. It's nothing to do with the colloquial probability. I'm talking about the underlying

**[21:38]** measure. Yeah, it's the probability of the inverse image of that particular event. See, I can't write that entire set. So, it is uh inverse image of cartitionian product. Let me write it there. Huh? Yeah. See, I will

**[22:10]** skip pro I mean see this is not a course on probability theory. This is just to wet your appetite on probability theory. Please uh you should do an entire this that was a prerequisite for this course. So do a formal course on probability theory. So now I will rush through things. Now what I'll do I think at this I will stop the class here and in the next class I will maybe half of the next class I will sort of rush through the other ideas but I want you people to kindly go back and refresh your probability theory

**[22:40]** basics specifically these things. So we talk of something called joint distributions between pairs of random variables. So there need not be one random variable right? There can be multiple random variables. If you have multiple random variables, we are talking about multiple sample spaces because every random variable is a function defined on random on on a sample space. Now imagine that you know

**[23:08]** you are tossing a coin and you are rolling a dieice and there are two random variables and we can talk about interaction between them and there are two random variables and we are talking about the uh uh the interaction between those two and what do you mean by interaction that's too vague right? So in set theory if you have two sets you can talk of you you can talk you can you can take

**[23:36]** elements of those those two sets and you can talk of different kinds of set operations on two of them right exact same thing happens when you talk of random variables that's it because random variable is a push for measure everything that happens on sets can can happen on the random variables now in the x-ray example that we took let's say that somebody getting a somebody you know getting getting a disease or not getting a disease and being imaged is is one

**[24:04]** sample space. Okay. And the process of imaging is is one sample space and process and and the the mechanism of obtaining or not obtaining a disease is another sample space. You can you can view the process of having an image or not having an image as a coin toss and it has nothing to do with the fact that this person went through an X-ray or not.

**[24:34]** Do you see that these two things may be correlated but they can be seen as two different random experiments? Outcome of two different random experiments. Do you see what I'm saying? This is very very important. Right? Somebody getting an image or an X-ray image is one outcome of one random experiment. Somebody getting a disease or not a dis

**[25:03]** not getting a disease is the outcome of another random experiment. And we can talk of interactions between them. And that's the labeling process. Do you get what I'm saying? In fact, somebody developing a disease and uh uh &gt;&gt; getting a disease or rather somebody getting a disease and getting an X-ray can be seen as the outcome of one single random experiment. See as I told you this notion of random

**[25:30]** experiment is completely up to the user. So that way in my mind a vector valued random variable can be seen as union of [clears throat] multiple scalar valued random variables that are correlated &gt;&gt; or depends. Now you see the point right? See when we talk of a gian distribution of a vector or or a vector valued gshian

**[25:59]** distribution that we talk of we we talk of this correlation matrix right. So what is a vector valued gshian random variable? It's actually the union of multiple scalar valued random variables. You can either see that as one random variable okay which has one underlying sample space and the range space is a d- dimensional vector or you can see that as dcaler valued random variables interacting with each other. Do you see the point and this is why

**[26:30]** you have distributions over vector value random variables. I I will I will actually you know make this notion a little more concrete next time because in this entire course we'll be talking about distributions over vector valued random variables. Okay. Now in this world view there is a nice unification between even the discriminative and generative models. To me both of them look the same. So labeling is one is is simply one

**[26:59]** dimension of the vector valued random variable that we are that we are looking at. See that is where we can define conditional distributions right as see since we can define uh uh sets okay which is you know the intersections and unions of the sets we can define conditional distributions that way. Okay. So anyway, so I think this is a good point to stop. Uh please go back to your probability theory basics.

**[27:29]** Specifically look at what does it mean to say that there are multiple random variables. What does it what what what does it mean to say that you know there is conditional distribution and conditioning of random variables in terms of see try to relate everything that you see now to the sample space and the underlying sets. So that that actually makes notion very very clear. Conditional distribution has to do something with the sample space right. So ask yourself what is the what is the

**[28:00]** events that I'm considering when I'm looking at conditional distributions. Okay. So please do that you know and and look at what distributions are and what are uh the discrete random variables and continuous random variables and so on. I will not do all of that because as I said this is not a course on probability theory but half of next class uh I will give you a a quick run through of some of the key ideas and then we will connect or rather formulate the problem of machine learning in terms of

**[28:28]** distribution learning. Okay, this concludes the topics that I wanted to cover in this particular class where we looked at the definitions of sample spaces, probability measures, distribution functions and so on. In the upcoming class, we'll be looking at multiple random variables and the ideas of conditional distributions and so on and connecting them with the problem of function approximation and define the

**[28:57]** core problem of machine learning. Thank you for attending this class.
