# Transcript — Tutorial 7 : Review of Basic Probability 1

> **Source:** https://www.youtube.com/watch?v=owlWCCgYx50  
> **Channel:** NPTEL - Indian Institute of Science, Bengaluru  
> **Duration:** ~50 min  
> **Note:** Auto-captions cleaned lightly. Minor ASR errors possible.

---

**[00:03]** Now hello guys uh welcome to this uh tutorials. This bunch of tutorials that we have is centered around one idea know which is uh the recap of probability theory. Now uh I prefer to emphasize it's a recap. Now we will be going ahead with couple of uh important definitions that are needed and couple of properties that will be needed for us throughout this

**[00:32]** course and then we'll be seeing couple of very interesting examples. In addition to that wherever needed we will be taking use of uh help of the code to understand and reinforce our ideas of probability. So without any further delay so let's jump into that in the sections that were uh taught by the professor he has

**[01:02]** already introduced the ideas of sample space and the needed uh things like uh event space probability measure and stuff. I'll just go ahead with the same recap once for the sake of completeness and then we shall proceed. That's the plan. So uh the first thing is uh now you should be comfortable with the idea of

**[01:30]** random experiment. Okay. Now we perform these random experiments. The outcomes of random experiments we put them and then that is what we are saying it as the sample space. So which is represent by this uh omega which is there and then it has the outcome of the random experiments. Okay. So now and then we define what is known as an event.

**[01:58]** So an event space is a subset of this omega which is there and set of all possible events that are there which is represented by by this scripted f which is there which is known as the event space. Now this will be a subset of the power set of the omega. Okay. So now for all practical purposes so during our discussion now we'll be considering this event space to be the power set. Okay.

**[02:27]** Now this means that every subset of omega which is there is an event that we have. And then we have the idea of probability. See probability is a function okay that assigns a number between 0 and one to each of the event and it has to satisfy certain properties and these are what are famously known as

**[02:55]** the axoms of probability. Now we use this scripted P to represent the probability. Now this probability is a function from the event space which is this f to this real r and it has to be satisfying these uh following rules. Now the one is the non- negativity. Now that means that probability of any event

**[03:23]** which is there will be greater than or equal to zero. Now you cannot have any event whose probability is a negative value. Okay that that's not possible. Okay. So, and this is for all the events as I told you. And then the normalization property, the normalization exam. Now, where in which the probability of the whole sample space should be one and then an important property now which is the the zigma additivity.

**[03:52]** Now if you have events A1 A2 know which belongs to this F okay and then they satisfy the mutual exclusion property now that is that means that there is nothing common between this A and AJ. So that means that the intersection between this a and aj is a null set which is there and for all these events that has been considered. If that is the case now then what we can

**[04:20]** do is the probability of union of all those considered ais which are there is equal to this sum. Okay the sum of the individual probability and then this now these are the three properties that needs to be satisfied. So non- negativity normalization and sigma additivity and then this omega FP now which is known as famously known as the probability triplet which is there or it is known as the probability space. Okay.

**[04:50]** And the next very important definition is the definition of conditional probability. See as I told you know this is a recap of things. So it will be a what do you call a laundry list of uh the definitions and their properties. Okay. And I'll be more emphasizing on what are the conditions that needs to be uh what do you call taken into consideration now while applying any given uh form for formula or any given formulation. Okay. So now let's look at the definition of this conditional

**[05:19]** probability which is quite important for us. Now let B be an event. Okay. Now that means that this B which is there now this belongs to this scripted F which is there and the probability that has been assigned is greater than zero. Okay, if that is the case okay both of them has to be satisfied. B has to be belonging to F and then the probability of this event B has to be greater than

**[05:49]** zero. Now when this is the case we define now what is known as the conditional probability of an event of any event A. Now on B probability of A given B. Okay. So is equal to the probability of A intersection B divided by probability of B. Now to represent this A intersection B we just write it as probability of A divided by probability of B.

**[06:17]** Now, now given B, the conditional probability is a new probability assignment to any event. Okay. Now, we should be pretty much aware of this idea. This conditional probability which is there is a new probability that has been assigned to all the events. Okay. So, the the notation now what we use is kind of an abuse of notation. So, but we will be using it. See once we know that

**[06:46]** it is an abuse of notation so that's fine at least we are aware of it okay so probability of uh b conditioned on b is one and uh uh probability of uh [snorts] this is uh okay sorry there is a this is greater than zero okay now only if probability of ab is greater than zero okay fine

**[07:19]** So now the new probability of each event is determined by what is common that is there with B. Okay. Now you can think of it like based on the knowledge now that we already have that B has already occurred we adjust the probabilities of all the events. Now that is why I prefer to say that not prefer to say that is the reason why this is a the new probability assignment know which is there. Okay. And the same thing can be

**[07:47]** extended for multiple events in a similar manner. Okay. So now let's look into the next uh important definition which is there now which is the the total law of probability which is needed for us. Now let's condition let's consider a partition of uh the sample space. Now what do I mean by that? I have this sample space. Now let's consider that I have this B1, B2, B3 and B4. Okay, let me have this uh uh partition. Now

**[08:18]** whenever I mean partition that between the events that has been considered this is that is between all these BI and BJ for all I not equal to J. Now you don't have any intersection. These are mutually exclusive things. Okay, we can think of it like this. Now for an example you can think of this this is B1 B2 and this is B3 and this is B4. These are the three four partitions that are there. Okay. And each of them are uh uh so they are mutually

**[08:49]** exclusive. Now in that case you consider any event A. So let's let's say that you have uh considered now this specific A event A. Now this one that is there in the blue. Let's consider this is a. Now what is this? Now this is this a this this whole thing is a which is there. Now how do we represent this? Now now this this is union of all these four and each of these uh what do you call a slice of a pie which you can consider.

**[09:17]** Now what is common with A and B1? What is common with B? What is common with A and B2? What is common with A and B3? What is common with A and B4? you consider them and you take the union of them. See uh know whenever we are dealing with the probability we always consider uh pleasant union we normally interchange them and use it. Okay. So we should be able to and anybody who has taken a first course in probability should be very much comfortable with

**[09:43]** these kinds of uh notational uh abuse. Okay. So now you have this uh union of a b1 till a bm. So that means that you have taken each of these section and then you have performed the union. So I can apply probability on both the sides. Now probability of a. Now since now these are uh disjoint these are mutually exclusive. Now I can apply the idea of sigma addivity. Now because of which so the probability of union which is there

**[10:10]** is sum of individual probabilities that will come out and then I can apply the definition of conditional probability which is there. So and then I can write it like this. Okay. Now this becomes the total uh probability rule or total of probability. Okay, this is a very interesting uh thing. So you consider each of the partition. So for whatever event that we are going ahead. So the conditional

**[10:39]** into this the probability of the the the suction of the partition that you are considering. Okay, we can go ahead with like that. Now then comes uh a very important uh rule in probability which is uh the base rule. The probability of uh a given b is equal to probability of a b divided by probability of b and uh by the conditional uh probability rule we can uh say that this ab is this.

**[11:08]** Okay. So probability of b given a into probability of a divided by probability of b. And then using the total law of probability. Now uh now we can write it in this. Now what is the partition that I have done? I have done so a and a complement. See a and a complement are mutually exclusive and then it's a partition on the omega. So it satisfies the uh need for uh applying the total probability law and because of which

**[11:38]** this is how uh uh the denominator will be modified. Now this is the idea of B rule. Okay. Now see uh it's very [snorts] uh interesting that uh now now sometimes now one conditional is easy to compute than another conditional. In that case now you can use B rule and then uh you can interchange and do the uh

**[12:06]** computations. Okay. So now coming to a very important notion now which is independent events which is there. Now we have two events A and B. Now whenever I say that events so now it has to be A and B should belong to this that is taken into consideration. Okay. So they are set to be independent. They're set to be independent if the probability of uh intersection of the event is same as

**[12:37]** the product of individual probabilities. Okay. See here you are dealing with the intersection and here you are dealing with the idea of uh multiplication. See you can observe that uh in the set space see whatever is union and intersection in a way they are getting corresponded with uh the plus and the multiplication uh in the probability space it's it's a nice observation to make it out. Okay.

**[13:05]** So now now this is the definition of independence. Now because of this definition when we use this definition in the conditional probability now what happens is now suppose that probability of uh a and probability of b the probability of events are greater than zero if they are independent so the conditional probability now you have this uh the probability of intersection now since a and b are independent now they can be

**[13:34]** written as the products so and then so this probability of b and probability of b will be cancelling. Now because of which so if A and B are independent the conditional probability and the unconditional probability will be the same. Okay. So now this is a consequence of the definition. So this is because of this definition the outcome now which is the conditional probability and the unconditional probability are being same it's an outcome of the definition and uh

**[14:03]** similarly the same thing can be applied for B given A. So you can get a similar result and it can be easily uh what do you call you can take it as a very small simple exercise that if A and B are independent. Okay. Now so are A and B complement A complement and B and A complement and B complement all these things also will be independent.

**[14:31]** You can take it up as a nice uh exercise and try to prove that if A and B are independent so are A and B complement A complement and B and A complement and B complement. Now the earlier definition we looked at independent of independence of two events. Now let's look at independence of multiple events. Now if I have events A1, A2 till AM. So we define what is known as the total independence.

**[15:01]** Okay, they are set to be total totally independent. Now if you take any integer K which is between 1 and M. So see what is that we are trying to say here is intuitively you take any uh subset of this A1 to AM. Okay. Now they have to satisfy the condition that the probability of intersection of all of them is the product of individual

**[15:30]** probabilities that this has to be satisfied. That is what is being said here. You take any k okay and then uh that k has to satisfy that it has to be between 1 and m. So you if you take k to be m plus2 or something now you have only m events how can you take m plus2 events? So it's not possible. So the condition is quite clear. and uh you take the indices I1 I2 till I K. So what happens is probability of intersection of all these events is same as the

**[15:58]** probability the product of individual probabilities. Okay. Now this is what is called as the total independence. Okay. So in addition to that we have what is known as a pair-wise independence. So see there are different notions of independence. So that I'm trying to uh bring in here. Now there is what is known as a pair-wise independence. See what is being done here is again you have events A1 till E. Now they are set to be pair-wise independent. If you take any two of

**[16:26]** them. So if you take any I and J where I is not equal to J. The probability of intersection of A and AJ is same as the product of individual probabilities. Okay. Now you take two of them. See here we are not going and putting a much more stricter notion of independence here. Okay. So events may be pairwise independent but may not be totally independent. Okay. And I leave it up to you if it is

**[16:54]** totally independent. Is it pairwise independent? Okay. Now now let's look into the next idea of independence which is what is known as the conditional independence. Events A and B are set to be conditionally independent given C. Okay. Now there has to be a conditioning event. Now the conditioning event is C. If the probability of AB given C now you can write it as probability of A given C

**[17:25]** into probability of B given C. Okay. Now when this uh above holds if this is the case now [snorts] I think uh you will be able to easily go ahead and uh get this. Okay. Probability of A given BC. Now, now this how do you compute this using uh the conditional probability? Now you get this. Now that you can split like this and using the definition of conditional independence, this can be

**[17:54]** split to this and BC which is there now can be split as probability of B given C into probability of C because of which this and this will get cancelled. So what remains is only this term. Okay fine. So now events may be conditionally independent but may not be independent conditionally independent. Okay. So one very interesting example that we take is

**[18:24]** so if we perform independent multiple tests for confirming a disease. Okay. Now this is a very interesting example that to ponder upon that events may be conditionally independent but may not be independent. Okay. So now it is also possible that AB are independent but are not conditionally independent given another event C. Okay. So now we

**[18:56]** should be aware of these uh uh what do you call corner cases to take it into consideration. Okay. Now these are a couple of uh interesting properties that uh we should remember. So now let's uh go [snorts] into a a very interesting uh idea which you already been introduced during our lecture sessions. The idea of random variables. Okay. See the need of random variable I'm assume that it is clear. Okay. That means that you have to work

**[19:24]** in the space of real numbers. So you want a kind of mapping to go from our sample space to the real space. Okay. So a random variable on a probability space know which is uh omega fp which is there is a real valued function. Okay. Now you might have already heard the famous uh uh statement. So random variable is neither random nor a variable. It's a deterministic function. So which is from

**[19:54]** omega to r. Okay. Already you might have in the lecture sessions heard this famous statement. Okay, it's a deterministic function which it's a function from omega to r. Now very interesting uh thing now that is uh so let's say that you have header tail like this. So x of the head you're mapping it to one and uh you're mapping tails to zero and this is one. See there are other kinds of things that just just a couple of interesting examples that

**[20:21]** you can think of like uh you go to a hotel uh and then uh know what is the sample space [clears throat] in a hotel. Now what are the dishes that are there in the menu? So now there can be multiple uh mappings. So know one is each item in the menu is mapped into its rate that is one kind of one random variable. So there can be another random variable where in which each item in the menu is mapped to number of calories

**[20:49]** that is there. That is another kind of random variable that is possible. Okay. See what I'm saying is there can be multiple random variables on the same probability space. Okay. That that's perfectly fine. So one is you can think of billing as one kind one random variable and then you can think of the the calories that are there in each food item that you take is another thing. Okay. Fine. So now any random variable uh results in a new probability space

**[21:22]** that is that you have this x so it goes to r b and px. Okay this is the see and then please be very much comfortable about this notation which is there. This px is a new probability measure know which is being uh because of this the random variable x. Okay. Now here R is the new sample space and this scripted B which is there is a subset of two per R there's a new set of events and this PX which is there is a

**[21:50]** probability that you are being assigning so on this uh space B okay so now you take any event now now how do we assign this now how do you assign the uh the probability measure for these is uh elements of this scripted B. Okay. Now this B is belongs let's say that there is a B that belongs to R. Okay. That is a subset sorry that that is a subset of

**[22:18]** R and then it belongs to the event space B and now we need to get the probability of that event B. Okay. Now how do we go ahead with that? The idea is quite simple. Now you you take all the omegas that belong all the w that belongs to this omega. Now which has been mapped to this B. See

**[22:50]** now let's say that you are going ahead and uh computing what is the probability of this particular element. This is in R. Okay. So now what are the W's that formed this B? You take this and since you have this uh omega fp which is there and this probability has been already assigned you take this and then that will be the probability of this b is

**[23:19]** what is being said said said here okay now the example that you can think of is like uh now in the new probability space what is the the px of now x taking the value one. Now if one is here what is one mapped to in the sample space? This is mapped to heads. Now this is same as the this is equal to the probability of we getting the heads. Okay. Now in the hotel example that I

**[23:48]** told you. Now consider that under a very restricted idea that you are allowed to take only one one uh dish over there. Okay. Now what is the probability that your bill will be 120 rupees? Okay. you're allowed to take only one thing. So that means that so here what is the probability of 120? You go back and see what are the uh dishes that has been mapped to this 120 and you take the probability of those dishes. Okay,

**[24:16]** that's how we compute the probability in the new space. Okay, so now this uh we write it like this. We use this x belongs to b. We just uh for the notational convenience we just go ahead and like write this and it is very easy to take it into consideration that okay the new px which is there is a probability measure and it satisfies all the aims of probability and this is quite easy for us to uh intuitively

**[24:44]** understand from here and similarly you can go ahead and prove it out. Okay. Now this is regarding the uh the idea of random variable. See always you should remember that so random variable is a function that has been mapped from omega to r which is there okay you you can take any random experiment that you have in your head okay good so now let's look at couple of important functions that are involved whenever we discuss uh random variables

**[25:12]** the first one being the distribution function of random variables now let x be a random variable on the probability space omega fp now we define this the TDF the cumulative distribution function. See as we go along we just say distribution function. Okay distribution function cumulative distribution function. So we use them interchangeably. The cumulative distribution function of X is PX and please please remember this. So whenever

**[25:43]** I write scripted P now that is what I mean by a probability measure. Whenever I don't write the scripted P and then I just write it as PX like this. Now this is uh the cdf of the random variable x which is there. This is not a probability uh measure that is uh obtained because of the inducing the random variable. No. Okay. For that I write it as p px like this. Okay. This is the the cdf. Okay. Now the cdf is an

**[26:15]** r2r uh function. And how do you define it? Now for you take any x now which is there the the cdf of that x is now you consider all the omegas okay now that are such that this x of omega is less than or equal to this x see the idea is like this okay now let's say that it's an r2r function [snorts]

**[26:43]** now let's say that you have a real line here okay you have got some x here. Okay. Now, each one of these maybe has some mapping back in uh the omega. Okay. Now, it has it might have a map in the sample space. Now, what do you do from minus infinity till this x you add all the probabilities which are there. Okay. This is how you can intuitively that is why it is called as accumulation. Okay, you're accumulating the probabilities.

**[27:11]** So, now here you can see this from this notation. It is very much clear that now this CDF now which is there you can think of it like probability of for all the X which is less than the considered value. So now if I say that if I go into the head and tail example that is there. Now let's say that this is zero and this is one. If someone ask me what is the CDF of 0 minus three. Okay. Now before that there is no uh point for

**[27:42]** which a probability has been measured. So the probability has been assigned. So it is zero. Okay. Now if I if someone asks what is the CDF at uh 0.5. Now till 0.5 there is only one for which I have uh some valid probability that has been assigned. So what that is the cumulative distribution function in that time will be the same as the probability of x taking the value of zero. Okay. Now till so from here. Okay. Now it will be one. Okay. I think that

**[28:13]** will be very easy for you to visualize with this kind of an example. Okay. And this is very easy to see that the cumulative distribution function that you have now is completely specifying the the probability assignment for this random variable X which is there. Okay. Now what do you mean by completely

**[28:41]** specifying? Now that means that if you uh give this CDF for all the X which has been considered now then I can know what is the probability of each of the event. Now it is very easy for us to trace back and obtain. Okay. Now any CDF which is there should satisfy certain properties. Okay. Now what is that? Now this CDF is defined as a probability.

**[29:10]** Now any probability now which is there now for sure will be between 0 and one. Okay. So therefore the CDF which is there has to be between 0 and one for all x uh uh that are considered. Okay. So and then the cdf at minus infinity is zero and the cdf at infinity is one which is an extension of the same idea. And then this uh the CDF

**[29:40]** which is there is nondereasing. Now see uh we are not saying that it is increasing. Now please be aware we are saying that it is non-deereasing. That means that there may be some points which stays uh what do you call as similar to the previous point. See now the example that you can take is for example the CDF at 0.5 and the CDF at 0.6 six will be the same whenever we are considering that uh what do you call the coin tossing example where in which heads is mapped to one and tiles is

**[30:07]** mapped to zero okay now it is uh nondereasing okay but we should not say nondereasing and increasing these are two different properties I just want you to sensitivize to this idea okay now now how do we say it now you take any x1 and x2 now see whenever we are taking x1 and x2 we should know that this x1 and x2 belongs to r okay now we Always remember what is this px? So sorry not done this the this is an r2r function. Okay

**[30:38]** so this x1 and x2 you take two numbers in such a way that x1 is less than or equal to x2. Okay. Now then now what happens is if you consider now all x that is uh less than this x1 now will be uh this subset of this. Now because of which the probability of uh uh the sorry the no not the probability sorry the cdf of uh x1 in the consideration will

**[31:08]** always be less than or equal to the cdf under x2 under consideration. Okay. Now this is a straightforward indication that the cdf which is there is nondereasing. Okay. And another thing is uh this uh cdf which is there is right continuous and has left hand limits. Okay. I at this point I leave it this statement as a what do you call a nice exercise for you guys. Now please go ahead ahead and look it for uh now why is this needed now what will happen if this is not

**[31:37]** there. Okay I leave it to you as a small exercise. Okay please go ahead and do that. Okay. So now now this can be easily seen that if you want to measure uh if you want to know the probability of uh all the x which is in between a and b. Now this can be uh shown that this is uh the what is the cdf at b minus the cdf at a. Okay. Now here is the derivation for that. You can

**[32:08]** look into it. Okay. And then if you are adding this uh equality here now you need to add probability at x= to a needs to be added. Okay. And please see the difference in the notations that I have. Okay. Now please try to understand what is it. Please please be very aware of whether I'm speaking of probability measure or I'm speaking of CDF. So notationally now you should be pretty much clear what is that is being looked

**[32:35]** into. Okay. Fine. So now post uh going ahead with CDF now let's uh look into the idea of different uh kinds of uh random variables that we'll be studying here during our discussion. Okay. The two kinds of things that we'll be that we'll be studying. I'm not saying that there are these are the only two kinds of things. The two kinds of things two kinds of random variables that we shall be studying are discrete random variable and continuous random variable. Okay.

**[33:06]** Now there is one more kind which we shall not be looking at during our uh discussion. Okay. So most of the times now 80% of the times during our course discussion we'll be dealing with the continuous random variables and fairly sometime 20% of the times we'll be dealing with discrete spaces. Okay. Now these are the two kinds of random variables in which we'll be investing our uh almost uh throughout the course. So we'll be looking at only them and and CDF is

**[33:35]** defined for all kinds of random variables. So it's it's not that it is only there for discrete or continuous or some other kind which I haven't defined. CDF is there for all. Okay. Now let's look into the discrete random variable. So now the discrete random variable uh uh the definition goes like this. Now consider a random variable X. Okay. uh on a probability space now is taken into

**[34:04]** consideration from now. So a random variable X is set to be discrete if it takes only countably many discrete values. Now whenever I say countably many now I say that either finite or countably infinite. Okay, I'm assuming that the viewers of uh this course are pretty much uh comfortable with the difference between countably infinite and infinite. Okay.

**[34:33]** Yeah. Okay. So any random variable defined on countable omega now we will be countable sample space is what we are calling it as discrete random variable. Okay. A simple example is uh now let's say say that I'm tossing a coin three times and then uh that is my experiment. Now how is my random variable how is it mapped to the uh r now how many number of heads we have obtained. So whenever I

**[35:02]** toss a coin three times now either I cannot the what are the possible options I don't get any heads that means that when I map it into real space I'll be getting zero I can get one head I can get two heads I can get three heads. So in this case the x can only take 0 1 2 and 3. Okay. Now these are the only four values that I can obtain because of this specific mapping. Okay. Fine. Now as I told you you perform the

**[35:30]** same experiment and then you can have a different random variable which is how many tails are there. Okay. In that case the values now let's say that that random variable you're calling it as y. In that case what are the values that you can take? can just go ahead and do it as a very small exercise. Okay. So [snorts] now let x be a discrete random variable and it takes x1 x2 so on and so forth and then we are assuming that now the x1 is less than x2 so on and so

**[35:58]** forth. Okay if it not you can rearrange and then you can always change changing of names would not uh have any issues. Now let's say that for each of these x i assign this qi as a probability and all these qi now will be greater than zero and then whenever you sum them it will be one. Okay. So now in the case of uh discrete points that we have now probability of uh so not sorry so the cdf at x and cdf at x

**[36:29]** minus will give you the probability at x. Okay. Yeah. So now this is the simple uh how does the CDF for the example the discrete random variable that I have taken I have just given here now [snorts] till zero it will be zero now at zero you will have now that means you have uh at zero you'll have a jump of 1 / 8 now why is what does 0 represents in our random space so in our sample space

**[36:57]** it represents no head okay now that means that you get this So you go here and then at 1 you again have a jump and then uh the jump is 3 or 8. Y3 or 8 I'll leave it to you as a small excise. At two again you will have a jump of 3 or 8. At [snorts] three you will have a jump of 1 / 8 and this will be one. It will continue like this. This is how this is a staircase function or a step function which is there. Okay. Now okay. So now what are the cases? one

**[37:28]** head is uh so head tail tail tail head tail tail tail head. Now these are the three cases which are there. Okay. Now therefore the probability at this the jump here at this point is 3 or 8. Similarly now whenever you have two heads now a similar kind of jump. Okay. Yeah. Now if you consider the accumulation of probability till that point till zero for

**[37:59]** uh x at zero. Now what is the probability at zero? It you have obtained this jump. So therefore it is 1 / 8 at 1. Now that means uh this point you would have accumulated 1 8 + 3 8 which is 48 which is half at 2. Now you would have obtained this much 1 3 3 this is uh 7 / 8 and at 3 you would have

**[38:28]** accumulated the whole one. Okay. So this is so you can see that here the CDF uh uh graph which you have obtained is a staircase function which is there where the jump okay is the the probability of that event happening. Okay, you can see that from this graph very comfortably. Fine. So now let's define uh one more now which is probability mass function

**[38:59]** which I'm writing it as small p and x like this small p x. Now let x [snorts] be a discrete random variable with x1 x2 so on and so forth. Now what is this? Uh the pmf is defined by probability at that point. See CDF is probability till that point. Okay, whatever probability that you have accumulated till that point. So mass function is probability at that point. Now this is only defined for all the x's

**[39:28]** that uh our random variable can take for all the other values. Now it will be zero. Okay. So now yeah this is what it is. Now px is a real valued function of the real variable. So which makes sense. This is a what is being said is this is an R2R function is what is being said here. Okay. And this is a straightforward thing that you can immediately obtain. Fine. So now then

**[39:58]** now what is the relationship between the CDF and the PMF which is there. So now if I go till here this is the probability at that point. This is the probability at two. This is the probability at at one. This is the probability at two. This is the probability at three. So now let's say that you want the probability at you want the CDF at this 2.5. Now how do you

**[40:27]** get it? [snorts] So whatever is the valid points for which the probability is defined, you can just add them, right? Okay, if you add them, that is enough. That is what is being said here. So you take i such that the x i is less than or equal to x. So that means that if you take x is equal to 2.5. Now what are the xis that I should consider? The

**[40:55]** x i can take value 0 1 and 2 and for each of this you add the the mass at that specific point. Okay. That will become the the cdf. Okay. So now fine. So this PMF which is there again this is by definition the PMS is a probability. Now this has to be greater than uh zero for all X and it is zero for all the

**[41:24]** points in R for which the the random variable is not taking any value. Okay. Uh and then whenever you sum over all the points which are there it has to be one. Okay. This is the standard uh uh rules that has to be considered because of the definition. Okay. So [snorts] any function satisfying these two rules that has been stated above is a PMF of some uh discrete random variable. Okay. So now

**[41:53]** any discrete random variable which is there can be completely specified either by giving its CDF or by giving its PMF. So if you want to understand the now what do you mean by under see what do you mean by understanding a random variable? No you we want to understand that what are the elements taking what are what are the probability of each of those elements occurring. Okay in the discrete case.

**[42:21]** So now either you get the CDF so which is nothing but the sum of PMFs or you directly give the PMF. Now given these two things I will be able to reverse engineer and get the probability of each of the events which are there. Okay, that is what is being done here. Okay. So and please make note that the CDF which

**[42:47]** is there is defined for all kinds of random variables and PMF is only defined for discrete random variable. Okay, for continuous random variable, you don't have the idea of uh PMF. Okay, now what is continuous? I haven't defined yet for the people of the audience. So, they might be knowing it. So, the PMF is only defined for a discrete random variable. Okay. Now, let's see some of the the examples

**[43:16]** of uh discrete random variables. The one is the standard Bernoli random variable which is your tossing one one example of Bernoli random variable is a tossing. So the outcomes the X can have two things two events. Now success and failure and then the the PMF of uh the success or whatever the number for which the success has been mapped is P and then the failure is uh 1 minus P. Okay.

**[43:47]** And the value of p has to be between 0 and 1. Now you can think of it like zero as the tail and one as the head. Then you can think of this or you can think of x is uh a test in which is the probability that you perform some uh test and then uh so uh so the person is passing in the test or failing in the test or some disease. Now you you take a one bad thing that all of us remember is

**[44:15]** COVID. you perform a COVID test and what is the probability uh of the success or failure so so on so on so forth okay we can model all these experiments like that okay so now this is the idea of Bernoli random variable see another important variation of Bernoli random variable that we see in most of the cases is what is known as the indicator random variable see the idea of indicator random variable is the indicator random variable takes the

**[44:45]** value one if the event occurs. Okay. So now consider a sample space omega fp which is there you take a b which belongs to this uh event space. The indicator random variable will take the value one if this w belongs to b it takes zero if this w doesn't belongs to b. Now then therefore it can be immediately seen that the probability of indicator random

**[45:14]** variable taking the value one is probability of uh occurrence of B. Okay, it's uh this is what is famously as the indicator random variable. Okay, which which we'll be using very oftenly during our discussions. Okay, now and then we have uh the binomial distribution. Now this binomial distribution uh which is there can be considered as n independent barnoli trials which are there. Now x taking the values between 0

**[45:45]** 1 till n. Now now what is the pmf? How is this pmf defined? Now it is the pmf at the point k. Now what is that you are saying? Now you can think of it like what is the probability that out of n there are k successes. Okay. So now n ck into p to the^ of k 1 minus p to the^ of n minus k. Okay. Now where k can take the value between 0 can take the value 0

**[46:13]** 1 so on and so forth till n. Okay. Now to completely specify uh the barnoli distribution. Now we need n and p. These are the parameters. N will be a positive integer and then p will be a value between 0 and one. So you can think like this. P is the probability of heads and then out of uh N is how many times you want to repeat the experiment. So this is P to the^ of how many this is the P

**[46:42]** is the probability of success. You want the K successes. So whenever you have K success that means that you have N minus K failures. Okay. Yeah. So now consider any independent tosses of coin whose probability of head is P. Okay. Now if X is the number of heads then X has they have a binomial distribution. So this is nothing but n independent Bernoli trials. Now we have another example is uh uh pon distribution.

**[47:11]** Now X here takes the value 01 2 so on and so forth and this is the the PMF. So and this is the parameter. Okay, we specify this uh with the parameter and K has to take a value 0 1 2 so on and so forth and this can be immediately seen that see again any uh PMF which is there now it has to satisfy certain things whenever we define any PMF now you have to immediately check that whether it

**[47:39]** satisfies those conditions or not okay now what are those conditions now each of the PMF at each of the individual individual values that has been considered has to be between 0 and one and when you sum against all the valid values. Now it has to go to one. Now it can be easily seen that now if this is the the PMF if you go ahead and sum it up over all the valid values that it can take now it will go to one. And similarly it can be seen for here and uh in the case of

**[48:09]** indicator random variable and Bernoli trials it's a straightforward. So P + 1 minus P it is one. Okay. Yeah. And then another uh uh uh distribution uh that we will be considering is what is known as the geometric uh distribution. Now where in which x takes the value 1 2 so on and so forth with the PMF. So 1 - p to the^ of

**[48:39]** k minus 1 into p. Now what does this says is now you keep on tossing the coin till you get a head. Now that means it is asking that what is the probability that I get a head at kth instance. Okay. Now if you are getting a head at kth instance that means that all the previous k minus one instance you would have got the tail now which is 1 - p to the^ of k - 1 into p. Okay. This is what is the geometric distribution now which is there. Okay. So now till now this is

**[49:09]** a couple of examples of uh discrete random variables. Now this is where we conclude uh this tutorials. Uh till now what did we see? We looked at uh uh the definition of probability, the conditional probability, the B rule, the idea of independence, multiple variations of independence which are there. And then we looked at random variables. There we specifically looked at discrete random variables. Now in the next set of tutorial now we'll be going

**[49:39]** ahead and looking at uh the continuous random variables different kinds of continuous random variables and then we'll be looking at uh uh things like expectations and then variance uh and so on and so forth. That's the plan for the next tutorial. Okay, thank you all. I hope uh you enjoyed this uh small section of tutorials on review of probability. Thank you. We'll meet in the next tutorial.
