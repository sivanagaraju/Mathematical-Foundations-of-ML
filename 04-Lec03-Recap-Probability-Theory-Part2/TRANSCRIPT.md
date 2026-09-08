# Transcript — Lec 03 Recap of Probability Theory - 1, Part 2

> **Source:** https://www.youtube.com/watch?v=DaBw9qBpt2s  
> **Channel:** NPTEL - Indian Institute of Science, Bengaluru  
> **Duration:** ~14 min  
> **Note:** Auto-captions cleaned lightly. Minor ASR errors possible.

---

**[00:01]** So now you have the probability measure. So what do you do with this? Uh you'll have to finally solve that problem of as I say as as he asked where does it fit in the entire the machine learning framework that we are looking at. Proist realized that you just cannot stop at this. So finally what we have is we have this famous triplet which is the sample space the events F [snorts] and the

**[00:28]** probability measure P. This is called trip probability triplet. Right? The probability triplet. In fact, the analogy is you can start from real real numbers, take a subset of real numbers and assign a measure to it. That's another triplet. R subset of R and length measure. That's

**[00:57]** another triplet. Make sense? Okay. Now we can't stop here and you know we need more uh tools to deal with what we are uh what we are up to because remember that I told you that the sample space it's is pretty abstract it need not have numbers correct however we as engineers need numbers

**[01:25]** because we can crunch numbers we can add them we can subtract them you know we can do algebra on top of it and all Unless we have numbers, we cannot deal with things. You know, we we cannot again it's I mean it's it's sort of uh I'm contradicting myself when I say that you need to look at abstraction and then I say that no you can't stop at abstraction. You need to measure something and need something concrete to make something work. So think abstract as they say no think analog act digital

**[01:54]** think abstract act concrete. So your thinking has to be at an abstract level right. So just thinking that the hunger in the world has to go away will not make anything. You'll have to act for it. So now this this transition from the the abstract space of sample space event space and probability measure to something that can be measured. No that can be measured not in the in the see okay so I'm overloading the term measure. See

**[02:23]** measure is a mathematical construct when I say probability P is a measure. It's a function basically right now when I I I'm also using the word measure in the sense of you know the colloquial English word measure. If you want something to be let's not call it measured you know some let's call it u uh what do you do with a sensor actually measure &gt;&gt; detect or observe &gt;&gt; detect or observe or something right yeah so if you want to do that on top of abstraction you need something else I'll

**[02:52]** tell you I'll tell you uh the if you go back to our analogy of uh the x-ray okay somebody got a disease right somebody's lungs developed a nodule and you know they went to I mean that itself is a part of our sample space. Remember somebody developing a lung nodule itself is a part of our sample space and they coming and standing in front of the machine and all that happened and some and something came in the sample space. We don't know what the elements are. However, if you want to work, we need

**[03:20]** something that is concrete or observed at detected. Now, how do we do that? Typically, we use sensors to connect the the abstract world to the real world. We use devices or sensors and we make measurements. Okay, in this case we get an image, we get an X-ray and we store that on a computer and what we get is a vector at the end of the day. Now you work with vectors

**[03:51]** practically you work with vectors but you started everything from an abstract sample space. What is the connect between these two? You understand? What you work with are numbers, vectors, but what you have as an abstraction is something that is it is only a that may only be a a mental construct. So you need something that bridges the gap between these two. And that's exactly what a random variable

**[04:19]** does. A random variable converts the abstraction of sample space to something that is concrete, which is real numbers. Okay. See when we hear this term random variable for the first time, we rather underestimate it and say think of it like something some esoteric thing, right? Oh, they're talking about some variable and all that. It's it's it's needed. I mean, I'm saying that it's not that somebody just came up with this

**[04:47]** idea. It it is it it was defined because you need it. Let's define it formally. So, there is the sample space omega. But what we operate on are real numbers, right? That's the motivation. So what we do is we define a function X.

**[05:16]** Okay. So this is called the random variable. Random variable is a mathematically it's a function. So what function is it? We know what functions are, right? Functions are mapping between two sets. Now, what are the two sets that we are considering here? Okay, define a function X from the sample space omega to real numbers R. So X is a function that starts from

**[05:46]** take elements of the sample space and maps it to real numbers. Why do we need this? I hope that all of you can really appreciate this fact that we need another function on top of sample space to work with. We cannot stop at sample space. Do you see that? Why? Because sample space is very abstract. You know, we cannot do anything with head, head or tails. We need numbers and we need something another function that would map the abstraction of sample

**[06:14]** space onto something that we can deal with which are real numbers and that's the function that we call as random variable. Hold on I I'll take questions. Now random variable is neither random nor a variable. It's a deterministic function. [snorts] I don't know who named it a random variable. It's the you know biggest misnomer of the world. At least one of them should have been

**[06:42]** meaningful. No. So call it random function or or deterministic variable or something. They called it random variable. Right? So I don't know where this name came from anyway. It came from somewhere. &gt;&gt; Why can't the name? &gt;&gt; Yeah. Huh? &gt;&gt; Why can't the name &gt;&gt; too small to do it? There's nothing in the name anyway, right? So it doesn't matter. Call it anything. So all of you I want you people to understand two things. one appreciate why you need another function on top of sample space. Why can't you

**[07:11]** just stop at sample space? That's very very important. It's the key. And also appreciate the fact that the function that we that we uh that we define and use it in practice is what is known as random variable which maps the elements of sample space to the real numbers. Okay, any questions on definition? Is this a common way to say that measure any measure on a given sample space? &gt;&gt; No, I I'm still not talking about

**[07:40]** measure here. This is simply there exists a function from sample space to real numbers. &gt;&gt; Hold on, hold on. We we what happens the question that you should be asking is now that there is a function that we have defined from sample space to real numbers what happens to the measure that we have defined on the sample space the subsets of sample space in this in this new formulation right and we'll come to that it's called a push forward measure we'll come to that in a while it gets transformed and that's exactly what we

**[08:09]** call as a distribution function we'll come to that &gt;&gt; whyange needs to be R &gt;&gt; need not be R in general it can be anything it can be integers as well but generally you take you consider real numbers in fact that's a very good question that you ask the question is why should it be R in most [snorts] practical cases it don't it won't be R it would be RD it

**[08:37]** would be a D- dimensional real space right so because it's like you know you can see uh other sets are subsets of R, right? You generally define that as R. The range space of random variables is typically taken to be R. &gt;&gt; Yeah, we'll trace that trace that back. Yeah, it will be the it's our data basically, right? We'll come to that in a while. Any other question?

**[09:05]** Okay. Now, yeah. So, let's start with R. No, I don't want to overload the I don't want to create cognitive overloads now. So let's say that we are looking at R for now. So sample spaces to R. Um perhaps I should so because we are looking at that image example right. So in the image example what would this this function be? So I told you in the last the last time that every image can be seen as a point in a dimensional real

**[09:33]** space. Right. Right. So in that case so I'll generalize this notion. Right. So a random variable is a function from sample space to the d-dimensional real space. It's better to generalize it. Okay. So now what what is what is an image now? So you can see the image as the element from the range space of the random variable which has been defined on an underlying sample space.

**[10:06]** Do you see this? What is a data point? Now what is an image that we have? An image is an element in RD. that we know that Y is an image is an element in RD that we just did that scanning and it's it's a point in RD right now what is that RD this RD or the Dimensional real space that we are talking about is actually the range space of a function what function are we talking about it's the random variable okay what is this function defined on

**[10:35]** what is the domain of this uh of this function &gt;&gt; the sample space and what this sample space is uh is is is is talking about or rather denoting it's the underlying experiment. So whenever I say that you have an image with you this entire story has to come in your head. You got it right. So every data point that we see which is a dimensional vector. Now in the in the deterministic viewpoint it was a vector in RD. In the

**[11:05]** probabistic viewpoint, it is an element from the range space of the random variable that has been defined on an underlying sample space that we don't have access to. Only images that are stored in a folder which set is that &gt;&gt; every image is this? Every image is one element in RD. You missed the last class. Don't miss classes. &gt;&gt; So I know that an image can be in RD. But &gt;&gt; image is in RD, not can be. An image is always in RD.

**[11:34]** We I want you all of you to see an image as a point in RD always. Okay. Now what is this in in in the last class in the deterministic treatment it's an element in RD. In the probabilistic treatment it is still an element in RD but this RD is the range space of an underlying function and that function is a random variable and that random variable operates on a sample space and that sample space is generated from a random experiment. This is the story.

**[12:02]** All of you are clear so far. Okay. &gt;&gt; Is the random function because every point from the sample space has a &gt;&gt; I leave this as an exercise. Good question. The question is is random variable a bjective function? It's an exercise. This is not a course on probability theory, right? But it's okay. Yeah. Good. It's actually a good question. Think about it. Yeah.

**[12:31]** Anything else? Okay. Now, now that we have defined Okay. See, uh this should this is the difference between knowing something precisely and knowing you know handwaving stuff. See when I say that oh I have an image X I right I have 10 images and all each image is a is a point in is a vector in RD which is true but know to truth has multiple layers. The moment I talk of distribution functions and probability,

**[12:59]** I should know that there exists an underlying sample space and the measurements that I'm doing are actually elements of the range space of the underlying function that is called random variables. Got it? Yeah. Okay. So now that we have uh uh uh we have defined a function on the sample space uh omega, what happens to f? So we have RD. What happens to the subsets of sample

**[13:26]** space under this function? What are those called? &gt;&gt; They are called boron sigma algebra. As I said, no it sounds very complicated. It's not. See what happens is the moment so under the x under x what happens is sigma gets mapped to rd and f gets transformed to the sigma algebra

**[14:05]** which can be roughly again you know uh a mathematician would would scold me for this but you know a sigma algebra is not perfectly a subset but let's say that it's It's roughly corresponding to the subsets of RD. Okay. Now, what happens to the probability measure? So, we had a probability measure P. What happens to this?
