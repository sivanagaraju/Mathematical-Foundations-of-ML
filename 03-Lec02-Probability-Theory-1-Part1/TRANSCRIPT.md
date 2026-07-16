# Transcript — Lec 02 Recap of Probability Theory - 1, Part 1

> **Source:** https://www.youtube.com/watch?v=YLx3hBqt28k  
> **Channel:** NPTEL - Indian Institute of Science, Bengaluru  
> **Instructor:** Prof. Prathosh A P  
> **Note:** Auto-generated English captions cleaned lightly for readability. Minor ASR errors possible.

---

**[00:00]** Hello everyone, welcome to the second class of this course titled mathematical foundations of machine learning. So in this class I will mostly be dealing with the foundations of probability theory and try to connect the deterministic viewpoint of function approximation to the probabilistic viewpoint of

**[00:26]** distribution estimation. While doing it, I will define the foundations of probability theory starting from the sample spaces, sigma algebra, distribution functions and so on.

**[00:41]** Last time we defined the problem of machine learning as that of function approximation.

**[00:48]** So we said that you are given observations from two sets and there exists a mapping between them which we denoted by the letter F and the task was to estimate the underlying f given observations from these two sets. Now why is that of importance? It's of importance because in lot of practical scenarios

**[01:13]** we want to predict. So what do you mean by prediction? Given that there exists a function between two sets and I've given some points for which I don't know what the range is from the domain. I want to predict what the range is going to be.

**[01:28]** So that's of importance in multiple practical scenarios. And one example that we saw the last time was that of having a an image and finding out whether that image corresponds to a a benign case or a manag case. Right? So you can imagine that there exists a function that maps the image to a set that consists of two

**[01:54]** values represented by 0 and one. One corresponding to each label. Now the task is to find out the underlying function. Find that underlying function f that would map an image to the label sets. So that was the uh idea and I also told you how to do it u at an abstract level that you start by guessing what f

**[02:17]** is and then you refine your guess using the observations that you have made. Okay.

**[02:25]** At the end of the last session I told you that this approach is not feasible for all problems because it's very hard to get to the physics of the problem and try to discern the underlying function.

**[02:38]** Okay, the example that was given is that of uh a coin toss where if you want to predict what the outcome of a coin toss is. Given that you have made some observations, what observations one can make, you can make observations uh physical observations such as the weight of the coin and the the angle

**[02:58]** uh at which it it would get tossed and what are the environment conditions etc. having those observations predicting or rather finding out what the coin toss is going to the outcome of the coin toss is going to be is rather difficult because modeling the physics is hard. Now another example that could be

**[03:15]** appreciated is that of the image classification or the X-ray classification that we taught. It's more so difficult because what you are measuring is the amount of light that a surface reflects or rather observes or lets pass through it. And what you want to predict or learn a mapping to is

**[03:36]** an abstract synthetic concept such as diseased or not diseased. So now finding this mapping or relationship is not trivial starting from the physics of the problem. Now what is the solution for this is that you lot of the machine learning engineers resorted the resorted to the statistical way of solving this

**[03:55]** problem where instead of trying to solve the problem from first principles from the physics of the problem try to solve it statistically. So what do you mean by trying to solve it statistically that you make multiple observations and see if you can discern what the underlying function is via the mult via

**[04:12]** multiple observations that you have made. So this is at a very high level broad level of what we would do in the probabistic uh framework. Now we will concretize those ideas today. Okay. What do you mean to say that you make multiple observations and uh what is what is the correspondence between the

**[04:31]** function learning and the the probability probabilistic theory and statistics and so on is what we are going to do today.

**[04:38]** See this understanding is key to understanding the probabilistic uh framework of machine learning because I've seen often times people miss this right. So when they when they see that oh there is a distribution function that is appearing you know it's very hard to relate it. See when I say that you are

**[04:57]** estimating a function you know I say that there is an image there are labels there's a function that relates the image to the label and all you are trying to do is learn that function the understanding is pretty neat so if I say that there exists an underlying distribution that we are trying to

**[05:10]** estimate through divergence minimization it's pretty hard to connect that but that's precisely what's happening so I'll spend some time in trying to bridge that gap between what this function approximation angle that we saw you know which we have been studying from our high school days and

**[05:29]** how that how does that relate to the statistical probabilistic ideas that we have studied in the first course of probability. So that is what I'm going to talk about. Okay. So probability theory right u just give you a 10 15 minutes quick introduction on what probability theory does. Okay. So we

**[05:49]** have to decide under uncertaintities. So deciding under uncertaintities we wanted a framework mathematical framework to see it's it's it's ironic and funny that despite the fact that you are using probability theory to model all this finally the decisions are deterministic.

**[06:11]** See a clinician cannot say to a patient that you are diseased with a probability of 60%.

**[06:20]** It's of no use. So decisions are to be deterministic, right? So however the conditions that we start with are rather nondeterministic okay and we have to take decisions in a under an uncertain uh scenario. So that is why we need a mathematical tooling to handle that. So in fact as I told the

**[06:42]** last time the only thing that is rather not defined very well defined in the entire probabistic uh you know theoretical space is the notion of randomness.

**[06:54]** We start with something called a random experiment. So everything starts from something called a random experiment that is not well defined. So there is we we assume that when we start probability theory we assume that we we have an understanding of what a random experiment is and we start by saying

**[07:10]** that this random experiment results in a few outcomes that are measurable. This is what we start with right in fact it's very funny again to note that this is the farthest extent that we use the word random. So once we say that we there is a random experiment and we have a few outcomes that are

**[07:35]** observed as a result of this random experiment all randomness stops everything else is set theory and measure theory after that okay as I said in the last class there is nothing random about a random variable it's a deterministic function okay so we say that excuse me we say that there exists an experiment which we

**[07:59]** call a random experiment and there are outcomes of that random experiment and that becomes a set. The outcomes collection of all possible outcomes of an experiment becomes a set and we know how to deal with sets.

**[08:13]** Right? Typical examples are so if coin tossing is a random experiment then the possible outcomes are two faces of the coin. In fact, you should know that it is completely up to a designer to define what the underlying random experiment is.

**[08:35]** Right? When you toss a coin, okay, you say that my random experiment is tossing the coins five times and this can be my random experiment.

**[08:46]** So remember that a random experiment is not something that's sacroant. I mean given a problem you can imagine what the underlying random experiment is. The moment you fix your random experiment and your outcomes everything else follows from that.

**[09:01]** Okay. So coin toss is one random experiment and you can see the outcomes head and tail as the set of possible outcomes and uh rolling a die is another random experiment that we talk of right in probability theory and the possible outcomes are different phases and so on.

**[09:16]** Now because we are dealing everything from a probabistic standpoint, everything that you see has to be now related to a random experiment. Remember this.

**[09:29]** So when we do machine learning, our starting point is a random experiment. There exists a random experiment, okay, whose outcome we are dealing with.

**[09:40]** Now I will concretize these ideas using the example that I started with in the last class which is taking an X-ray and assigning labels to it. Now what could be the random experiment there?

**[09:53]** The random experiment there is is the entire process of somebody getting diseased and they going to a a radiology center, somebody measuring them, you know, somebody imaging them using the uh the the X-ray machine and then printing that thing and finally uploading that on a computer and this entire thing, right?

**[10:17]** And a clinician sitting and putting a label on top of it. This is your random experiment.

**[10:22]** Do you understand? Now every data point that you get you know every single input set that you would get right in machine learning should be now seen from this angle that there is an underlying random experiment. Suppose you are dealing with let's say natural language text.

**[10:41]** So let's let's say that your task is that you are given a a paragraph and the task is to find out whether that paragraph has positive sentiment or negative sentiment. So that is one problem that or okay let's let's make it a little more concrete spam filter. You want to say whether a given email is

**[11:00]** spam or not. Now what is the random experiment that's happening here? somebody thinking that they have to draft a spam email and drafting a spam email and you know sending it and somebody labeling it as a spam or not because the idea of spam is very subjective. You see all labels are subjective. You know that

**[11:19]** becomes a little philosophical but we'll come to that in a while. You now see why I mean how everything can be seen as a random experiment. Right? A text is a random experiment. An image is a random or rather rather an outcome of a random experiment and so on. Can all of you see this? Any questions on this is very very

**[11:35]** important. Right? So every data point that we get or let's say that you know you are looking at u um some measurements from the brain right and you you want to uh predict something on that measurement right so that measurement that you get is an outcome of a random experiment so what's the

**[11:54]** random experiment somebody having a brain and you know doing something and you're conducting that experiment and making that measurement you see this now the moment we have a random experiment I said that is where The randomness in in in probability theory stops and set theory begins. The

**[12:11]** outcome of this random experiment is collected in a set and this set is called sample space denoted by this is called sample space which is set of all possible outcomes of a random experiment.

**[12:46]** Now don't ask me what randomness is. I don't know. Okay. We assume that there is a random experiment and some you know it is giving rise to some outcomes and we collect all of those outcomes in the set. That's it.

**[13:01]** Okay. I keep saying this. The biggest irony in computer science is that we have deterministic algorithms to generate random numbers.

**[13:14]** We have random number generators, right? Algorithms for random number generation and they are deterministic algorithms.

**[13:21]** >> Come again. >> Pseudo random numbers. >> Ah yeah that's why you call you say that they are pseudo random numbers but yeah anyway right so we have sample space and all of you understand what a sample space is. Right now once you have a set see finally remember our goal is to do predictions and we our goal is to find

**[13:37]** out whether that X-ray is diseased or not. Okay. Now just by seeing uh uh the process of obtaining an X-ray as a random experiment or the outcome of a random experiment doesn't help. We have to move from here.

**[13:52]** Now how do we do it? The moment we have a set we can't do anything if we do not assign a measure on top of it. we need to measure. So given a particular set we need to assign a measure a mathematical measure that would tell me that that would correspond to the I should not I should not be saying that

**[14:13]** it's the length of the set okay it should it corresponds to some notion okay so I I will not go into the measure theory I assume that that you know people have the idea of what a measure is I can give you examples let's say that we are looking at the real numbers as the set okay if if you're looking at the real.

**[14:30]** You all know that real number is a set, right? A set of real numbers. Now, if I take a subset of real numbers and if I have to compare two subsets of real numbers, how do I compare?

**[14:40]** I cannot compare unless I have the notion of measure or length. See, one measure one of the measures that you that you that all of us know on the set of real numbers is the length measure. It's also called the le measure. So, given a set, see, let's say that we have a real set, right? real

**[14:59]** subset of real numbers which are like this. One is like this. The other is like this. These are two sets. Let's say that this is set A and this is set B. Okay.

**[15:10]** Now, if I ask you a question, which set is bigger? Can you answer? Which set is bigger?

**[15:19]** >> Any answers? >> You would say it's set B. Why? >> Huh? Because we have the notion of length measure right and we ascribe you know we compare sets using measures. This is this is the le length measure that you have right. So so the length of set A is less than that of length of set B is

**[15:43]** what we say under the le measure that we have associated. Now suppose you have sets of sets in R squared right what sort of measures can we talk of?

**[16:00]** >> We talk of area measures right? So any set that we have if we have to compare sets and assign a way to quantify how big or large or something.

**[16:13]** It need not be the the semantics that we associate with a measure is totally subjective and it's it's dependent on the the the person or rather the uh the the setting in which we assign the measure but we have to assign a measure to do anything on top of the set. Okay. Now we know about

**[16:33]** length measure, right? If you go to R2 then we have the area. If you go to R3 then we have volumes and if you go to RD then we have the generalized leic measure right which is the D- dimensional integral volume and so on.

**[16:46]** So now we know how to compare the set. Now suppose we have subsets of sample space consider the subsets of sample space.

**[17:10]** So what do you mean by subsets of sample space? Now suppose you know let's go back to our example of uh the X-ray. So if our sample space what does our sample space contain? Our sample space contain the outcome of uh taking multiple X-rays.

**[17:27]** Right? We some many many people are coming and all of their X-rays are being taken and we enumerate all of that in our uh in our sample space. And please note that this sample space is still not a real number or anything. So it is simply see in the coin toss example the elements of of our

**[17:46]** sample space is simply head or tail it has no numerical notion yet similarly this is very important I'll tell you why this is important in a while so please note in your minds that the sample space is not numeric not be numeric so it's a very abstract idea so the idea of sample space is it contains all

**[18:07]** possible outcomes of an experiment now this outcome is as I said somebody uh having a disease getting getting there getting imaged and this entire notion so how do you quantify that I have not come there yet that's where we need the notion of random variables okay push forward measures as they are

**[18:25]** called we'll come to that in a while so for now you should appreciate the fact that the notion of sample space is independent of anything numeric it's just that some experiment is being carried out and we are enumerating the outcomes of the experiment in a set that's it In the X-ray example, what are the

**[18:42]** elements in the sample space? >> As I said, it is the question is what are the elements of sample space in the X-ray example it is it seees that somebody getting diseased and they are coming they the entire imaging is happening and they are being detected that they have disease. So this is one

**[19:01]** element. Now the question is how to quantify this? We'll come to that in a while. So sample space is rather abstract. you know it can be abstract it need not be measurable so not in the the measurable sense of the measure theory it need not be numeric let me say it that way right so that see as I said the last class I

**[19:20]** said English is very fragile as a language so that's why we need math as a different the other language so now see sets by construction may not be numeric per se right it's just collection of objects even if you study set theory the definition of set is that it's collection of objects and

**[19:35]** if you ask them what are objects we assume that we know what objects are and then we proceed from there right that's how it is h okay so now you if you consider the subsets of sample spaces just like we could consider the subsets of real numbers and assign measures to it the question is can we assign a

**[19:52]** measure to the subsets of sample spaces is the question just like we have the length measure the le measure on subsets of real numbers can we assign some set of a measure the notion uh of the length of the set equivalent to the length of the set in the case of real number, subsets of real numbers. Can we assign a

**[20:12]** measure on top of the subsets of sample space? Does this question make sense? It makes sense, right? It need not be the length measure by the way. It is not actually, right? It is some other measure. So, can we assign so please note that if you have a set, you can construct subsets of that set and each

**[20:31]** of the on each of the subset you can assign a measure on each of those subsets. Okay. So the question so let's denote this by subsets of subspace let denote capital f denote the collection of all possible subsets.

**[21:01]** Okay, question or rather objective is to assign a measure on f. See, I may sound a little hand wavy uh because this is not a course on measure theory. There there's an entire course on measure theory where these notions are made more formal, but I'm just giving you the intuitive idea of what

**[21:30]** this is, right? very very uh roughly uh take a set construct those construct subsets of those sets and try to assign a measure. Okay, there are properties of what a measure constitutes, right? For instance, the measure has to be non- negative and uh uh it has to if if you take two sets, one is subset of the

**[21:47]** other. There are properties just like you have properties of uh subspaces, you have properties of measures, okay, which would when would uh uh uh a function constitute a measure, there are there are a few uh properties that it should satisfy. Now the question is that or rather the objective is to assign a

**[22:04]** measure on the subsets of sample space. Okay. Now one measure that uh we assign which we work on is what is called as the probability measure. Okay, probability measure is one such measure either you can say it probability or probability measure or anything. Okay.

**[22:36]** So it is denoted as P. Measure is a function right. So it takes elements of F. What is F? Set of all subsets of the sample space and assigns it to particularly probability measure.

**[22:52]** Assigns the elements of F to set of 01. This is by construction. This is by definition. Right? So this is how we assign a measure on top of the subsets of sample space.

**[23:10]** Now there are a few properties that we ascribe to this. So what are the properties that we have on this uh probability measure?

**[23:20]** >> So let's let's look at the properties. See this is very equivalent to the legg measure. The length measure. So if you want an analogy always think of this as the length measure on subsets of real numbers. Okay. Properties of P. What are properties of P? Studied all this right.

**[23:42]** So one is I'll not enumerate all of this. So the measure. So if if let's say that if if a b are two elements that are in f.

**[24:00]** First property is probability measure of a is you take any subsets any subset of any element in f. This probability measure always has this measure to be either zero or greater than zero. So strictly non- negative similar to the length measure.

**[24:29]** Okay. But the other thing is this is upper bounded by one unlike the length measure. Okay. And what are the other properties? So this measure the measure that is on the entire sample space is one. The measure that is assigned on the null set is >> zero. And what else?

**[24:50]** If you take if you take two sets A and B such that the intersection is null, then the measures add.

**[25:14]** Thanks. Right. So the these are the properties of this particular measure that we have assigned. Okay. Now till now we have we have not interpreted this measure.

**[25:31]** This is still a mathematical construct. Right? And now you can interpret this as the likelihood of this event.

**[25:44]** Okay? Don't ask me what likelihood is. Just like you don't ask me what length is. You understand what length is, right? So there is a measure that would assign a number to length. So now assuming that we all know what likelihood of likelihood is. So there is so and by the way the elements of f f

**[26:06]** are also called events. elements of f are called events. So every event which is a subset of the sample space is assigned a measure and this measure is called the probability measure and this which has these mathematical properties.

**[26:24]** Now if you want to assign a meaning to it assign a meaning to it. In fact, all the machine learning runs even without you assigning a meaning to this measure.

**[26:38]** Right? The only time when you have to assign a meaning to this is when a doctor is actually you know have to use the algorithm that you have built and finally hand over a diagnosis to the patient saying that you are diseased or not diseased.

**[26:56]** Till that you don't need to assign any meaning to this measure. You get it? See that's the power of math.

**[27:05]** If you abstract out the ideas, right? You can you can work with you can work in the space of abstractions and you can defer the meaning assignment to the very end.

**[27:17]** So I'll tell you one other philosophical uh uh uh observation that I have made over years.

**[27:25]** See if you start seeing the world from a very abstract viewpoint then cross-pollination across ideas is very very easy.

**[27:35]** For instance, it's very easy for a probabilist or a mathematician to pick up machine learning.

**[27:43]** And see that's why they say that you know physics is applied math because precision and formalization and abstraction is very very powerful. So if you start seeing so in this case this this entire course is designed in such a way that you see data as the elements of the range space of

**[28:04]** random variables and if you start seeing the data that way then you don't care if you are working on images if you or you are working on text if you are working on time series if you are working on brain images nothing so I keep saying this always I'm a general machine learning scientist generalist to me anything that

**[28:24]** you give is are just vectors. What are these vectors? These vectors are the elements of the range space of the random variable that I'm talking about. There is an underlying probability measure that I'm trying to estimate using algorithms. That's the way I want to see the problem.

**[28:39]** So that's why I can handle images, I can handle text, I can handle whatever data that you want. You know that's the that's the power of abstraction and generalization. So one one needs to appreciate it. Anyway, so now that we have a measure on the subsets of the sample space, now we can I mean for for

**[28:55]** our understanding we can assign a measure. We can sorry we can assign some meaning. We can say that this gives you the likelihood of the occurrence of particular event.

**[29:05]** Now what do you mean by likelihood? Don't ask me. We assume that we all know what likelihood is because it is bounded between 0 and one. So if probability measure that is constructed or probability measure of a subset of or rather an event is equal to 7 as a mathematician or rather I would stop

**[29:24]** there but if you want to subs assign a meaning to it you say that there is 70% chance that this event would occur.

**[29:32]** >> I think these meanings help us develop better intuition about the underlying. >> No problem. See if if it helps you to uh develop better intuitions I'm fine with it. But all I'm saying is all all the piece of the math will just go through even without assigning meanings. If you

**[29:48]** don't assign meanings, it will be helpful because in one problem it would mean something something in one other problem scenario it may mean something else. You have to assign meaning at the end to make it work.

**[30:00]** Okay, whatever that means. But you don't have to restrict yourself in assigning meaning is all I'm trying to say. See just like you know the length league measure need not correspond to see in fact I'll tell you very interestingly suppose you are constructing the area under some curve

**[30:18]** that may correspond to different things in different context right you see what I'm saying so area area under the curve of acceleration is something else you know area under the curve of some other measurement is something else and so on but the underlying measure is the same so similarly look at probability as a

**[30:35]** measure on subsets of sample space. Now you ascribe whatever meaning you want but generally people see this as the likelihood. See what happens is when people start teaching or studying probability theory they would approach that they start by saying that oh there is some event and there is some

**[30:57]** likelihood of this happening and there is that's what probability is all about. I mean that's a bad way to look at things to me. It's a very neat mathematical construct where there is some set and you have subsets and we have defined a measure on top of it.

**[31:09]** That's it. Make sense? Right? Whatever meaning you want to ascribe, you ascribe it later.

**[31:16]** So as I said, generally probability measure is seen as the one that would give you the likelihood of occurrence of that event. Whatever that means.

**[31:26]** Okay. So this actually, you know, begs a very philosophical question, right? I talked about ignorance modeling in the last class. Now tossing a coin given that you are uh you are an omniscent that you can determine the entire physics of the problem you can determistically say predict what the

**[31:44]** outcome of the coin is because we cannot do it you know we resort to all of this right so there is yeah I mean again this becomes too philosophical right there is nothing random about the world so as I said randomness is even not defined say that there is an experiment some outcome

**[31:59]** is happening enumerate all the outcomes call it a set, define subsets on top of it, assign a measure, that's it.

**[32:06]** Okay, thank you.