# Transcript — Lec 12 Kullback-Leibler (KL) Divergence

> **Source:** https://www.youtube.com/watch?v=ihkGbIdbbxc  
> **Channel:** NPTEL - Indian Institute of Science, Bengaluru  
> **Duration:** ~16 min  
> **Note:** Auto-captions cleaned lightly. Minor ASR errors possible.

---

**[00:03]** Welcome back. Now remember what our objective was or our objective was to quantify the distance between a pair of distributions. Now this entropy sure will give us the average information contains contained in a particular distribution. Now how is this related to what our objective is? So let us define one other term. Okay. Suppose are

**[00:37]** two distributions. Actually there's a slight abuse of notation here. See here I this this P was probability the probability measure. Here this P is the &gt;&gt; mass function mass function because we are looking at a discrete random variable. Okay. And uh yeah so when I write P and QR two distribution I'm talking about

**[01:04]** mass functions again. So we have to be careful about that. I mean now we are looking at uh discrete random variables because uh for conditional random so for continuous random variables there's something called differential entropy which looks similar mathematically but interpretations are slightly different but anyway suppose p and q are two distributions to compute let's say that we compute let me just be clear about the definition

**[01:32]** I'll tell you why this thing what I'm writing is not symmetric so suppose I look at the expectation of two distributions on the same sample space. Again, right on the same sample space. Let me go to the next page.

**[02:05]** Define negative of log [snorts] of qx. This expectation is computed with respect to px. Okay. What does this give you? So negative of log of Q of X will give you the information or surprisal associated with

**[02:35]** the events under the distribution Q. Correct? Right? Now if you take expectation with respect to PX then what am I saying? I'm saying that my actual samples have come from px but I'm estimating px to be qx. Now if I do that then what is the average surprisal that I get? If I take if I consider that my true distribution is uh

**[03:05]** is qx or rather if I if my true distribution is px while I consider that the samples are coming from qx. Does it make sense? The other way to interpret this is that what is the average information that qx contains with respect to px or about px? Suppose px and qx were the same then what happens? This will become the entropy of px. Now if px and qx are not same they are two distribution two

**[03:34]** distributions then this quantity is telling you what is the average information that one distribution contains about the other. See suppose I take my qx to be a gshian distribution with mean three right and I take and my px is a goshian distribution with mean zero while the samples are actually coming from a goshian distribution with mean zero I'm wrongly thinking that my samples are coming from

**[04:02]** distribution with mean three right I I compute surpris results for all samples and then take an average what am I computing I'm computing what is the information that q contains means with respect to P or rather what is the mistake that I am incurring by wrongly making an assumption that my reality is Q while it was P okay okay so you can interpret it in multiple ways

**[04:29]** and this has a name okay this is actually written as H P comma Q what is this called this is called the cross entropy Okay. And by definition cross entropy is not symmetric. You see this h of p comma q is not equal

**[04:57]** to h of q comma p by definition. Okay. You observe this. Now again going back to our original objective. Our objective was to define distance between distributions. Consider &gt;&gt; small &gt;&gt; uh again this is notation of misnome will be there. Okay. So these are uh mass functions for no discrete random variables and mass functions. Okay.

**[05:24]** Consider take entropy of P and subtract it with the cross entropy between P and Q. let me just get the math right. uh px log px plus

**[06:02]** uh px log And the reason I'm staring at it is it's asymmetric, right? Because it is asymmetric. If I write it the other way, if I write HQ and P, you'll have a different expression altogether. But anyway, it doesn't matter. Okay? Now, what is this expression telling you?

**[06:30]** Let's interpret this, right? So, this will tell you the average information the entropy of P is the average information in P. Right? Cross entropy is the information that Q has about P. If I take the difference between those two, what am I looking at? I'm saying what is the error that I have incurred by wrongly choosing Q over P or wrongly choosing Q while it it should have been

**[07:01]** P or the extra bit of information that is there in Q and not in P. Make sense? What is the difference between the entropy and the cross entropy? It is telling you that if I look at the average information in P and the cross entropy or the information that is in Q with respect to P, subtract them both, I'll get the extra information that is there in Q which is not in P. Make sense? Right? So now that is equal to

**[07:29]** just use the definition of uh logarithms just log uh Correct. This is what it is. Now suppose P and Q are the same then what would this quantity be? It will be zero. Suppose P and Q are very far away from each other. What would this quantity be? It will be high. So take two things as homework. Show that

**[08:02]** this quantity is not symmetric. Okay. So let me write this quantity as by the way this is what is called as the K divergence equal divergence between two

**[08:27]** distributions. So this is this is you know the what is to be shown is that to show that DKL of between P and Q is not the same as KL between Q and P. Okay. and also show that this is non- negative and show that this is

**[08:58]** equal to zero if and only if P matches with Q. It's a two-sided result. Okay. So please take I mean this is use the definitions and use the law of log logarithms and uh the the definitions of density functions you'll get it. That is how you do it. take it as a homework and do it. Now we got our objective, right? We were looking at a measure that would tell us

**[09:27]** how far or close a pair of distributions are and K divergence is one such measure. And also you can show that this is uh not a metric in the sense that this does not obey the law of triangularity. Why? It's obvious, right? If it's not symmetric, if it's not symmetric, if a if a metric is not symmetric, then it'll it cannot satisfy the law of triangularity. Okay, let's show that. I mean, it's

**[09:54]** pretty simple to show. Oh, anyway. So, this is not a metric in the sense in the strictest mathematical sense. Okay, but it gives you a sense of the closeness between pair of distributions. It's called a K divergence. Okay. So a a similar uh definition exists for the case of continuous random variables and as I said uh we assume that most of our uh data is continuous right in most of the time. So I'll also give you a similar definition. So this

**[10:22]** is called this differential entropy and differential cross entropy right so which is x is the definition of k divergence for continuous random variable cases. looks very similar to the discrete case. The only thing is these are density functions now okay and cannot be interpreted as probabilities but yeah of course they are density functions [snorts]

**[10:48]** right P and Q the kale between P and Q here it was between Q and P so it's P log Q by P here it is P log P by Q No, it's the same thing. See, this will stay the same. See, HP minus H Q minus P.

**[11:18]** Sorry, Q P. See, HP minus H Q P. See, only the cross entropy term changes. The entropy term does not change. P with respect to &gt;&gt; Q. Right. Hold on. Let me just uh confirm it once. my my bad. I think the mistake is that

**[12:20]** it is the difference. Okay, I'll just change it this way. The minimal changes difference between the cross entropy and entropy. Okay. So in this case what happens it is uh because of this asymmetry there's always

**[12:45]** a question hold on just give me a second I'll correct &gt;&gt; See uh cross entropy this is P and Q this is correct. So HP and Q is expectation this. Yeah.

**[13:16]** This is So H P is uh P log PX. This is uh the cross entropy there's a there's a minus here and a

**[13:53]** plus here, right? And this will make it P by Q. Okay, correct. &gt;&gt; Yes. Yes. Yes. This is the KL between P and Q. Correct. Thanks for that. Difference between the cross entropy and

**[14:23]** entropy. It's because of the asymmetry, right? So this algebra does not work out. Okay. Thanks for that. So this is not asymmetric and so on. Yeah. So now the definition falls right. So it's P log P by Q. Correct? Okay. No, this is not this is not cross

**[14:52]** entropy. No, this is the difference between the cross entropy and entropy. HPQ is the cross entropy. HP is the entropy. So, we're looking at the difference between the cross entropy and entropy. &gt;&gt; No, test it out. Yeah, it should be the difference between cross entropy and entropy. So, now this I'm also saying one other thing. I'm also saying the cross entropy is always higher than entropy. No

**[15:21]** more information. &gt;&gt; Yeah. So that that should be the case because if if these two match match is always upper bounded by or it is lower bounded by entropy always see that's why you need to take the difference between see when I said difference HP minus HPQ or HPQ minus HP both of them are differences but to ensure that it's non- negative because by definition cross entropy is uh bounded by entropy you'll have to ensure that you take the

**[15:50]** higher and subtract the That's only definition. Okay. Fine. So for the continuous random variable case, the K divergence is defined this way. Okay. Any other questions here? So you would have heard about this. No to erase human. It's becoming more and more prominent these days.

**[16:19]** The more the mistakes that you do, the more it is proven that you are a human being and not an AI &gt;&gt; Huh. So for uh distribution functions I use capital for mass and mass functions. For density functions I use small P and Okay. Uh we will stop here and continue in the next lecture.
