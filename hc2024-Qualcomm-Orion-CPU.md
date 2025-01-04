https://youtu.be/uC0Jpfy5uj8

https://www.servethehome.com/snapdragon-x-elite-qualcomm-oryon-cpu-design-and-architecture-hot-chips-2024-arm/

Gerard Williams III
CEO, President, and Founder at NUVIA Inc

(2:07) 

Hello everyone, it's nice to actually be here today and get a chance to actually talk about the Qualcomm Orion CPU and give you a feel for what we built inside this machine, with that let me jump into the talk. 

so what did we actually do at qualcomm in order to build this CPU, so it's it's actually a full custom CPU, it was built on 4 nm process and it's it's a brand new design used in compute workloads and many other industries which I'll talk a little bit about at the very end.

--------------------------------------------------------------------------------
![Qualcomm-Snapdragon-X-Elite-Hot-Chips-2024_Page_03](https://github.com/user-attachments/assets/1a942326-b7f8-4f76-aa19-f1f70431dd18)

(2:43)

so within the so um which you can see over here on the right hand side, you can see basically the CPU complex, GPU, traditional stuff that you normally find in SoC’s, but the important thing here that I want to highlight is a lot of questions that the industry has been asking what does the CPU complex itself actually look like at a high level.
 
It's comprised of three CPU clusters they are actually the same unbeknownst to some in the audience one is operated in an efficient manner and the other two are fully performant but the underpinnings of the design is the same.

The complexes are comprised of four CPU cores. it has an integrated level two cache and an interface that talks to the system. 

--------------------------------------------------------------------------------
![Qualcomm-Snapdragon-X-Elite-Hot-Chips-2024_Page_04](https://github.com/user-attachments/assets/86a368a7-ad40-4cc8-8af2-b3e13116d08e)

(3:42)

inside of these CPU cores that you'll see is a very classical CPU complex. The efficiency of this starts actually at a high level within the core itself, but the units themselves that make up the core. you can see in the diagram in the center, classical CPU architecture, instruction fetch all the way to memory management, and load store. and the basic premise of this design I'm going to dive into details for you here in a few minutes. 

--------------------------------------------------------------------------------
![Qualcomm-Snapdragon-X-Elite-Hot-Chips-2024_Page_06](https://github.com/user-attachments/assets/56ff7f52-31b0-440a-96f8-34021bab55a6)

(4:14) 

The microarchitecture. so what does each unit comprise of, what we're going to find is that it's again classical design, you see an instruction cache, in this unit and there's a pictorial coming up so you you'll see how it's organized as well but there's a fully coherent instruction cache 192 KB; has an integrated L1 TLB a 256 entry structure; many different predictor structures are built into this engine as well, prefetching concepts are built into this structure as well. 

Some data structure sizes that you're going to see here one important one that we notice is branch mispredict latency of the CPU, it's not what I call industry best but it's balanced with the underpinnings of the design, it is 13 Cycles, and a return stack predictor as well, that's around 50 entries, I'm going to jump to the picture so I don't talk to a bunch of text

--------------------------------------------------------------------------------
![Qualcomm-Snapdragon-X-Elite-Hot-Chips-2024_Page_07](https://github.com/user-attachments/assets/4402cce9-a4ed-4224-91aa-592244fac79b)

(5:19)

This is the pictorial representation of what we've actually designed. 

you can see in the upper half of the diagram that is basically the instruction cache, the predictors and the TLB structures in this design. 

the predictors themselves we didn't talk about in the slides but I'll give you a verbal feel it's built around T architecture that a lot of you have probably seen in Industry papers.
 
They feed the front end of the machine, there is a branch Target buffer there as well to help redirect the machine and to get the next fetch group that the processor is going to need to feed all the downstream decoders that we see at the bottom.

The machine itself basically has eight fundamental decoders in it and they are preparing instructions for the execution units, the load store units, and the vector unit down below. 

The instructions themselves after they're decoded and executed go through and enter into a reorder buffer that you can see along the right hand side of this diagram. It's around 600 entries.  it's a little larger than that but it gives you a feel for how many instructions the machine's going to be managing in flight, and from a retirement point of view, there's eight instructions per cycle that this machine can retire.
 
Now as I said the machine's built around kind of a balanced design. The decoders themselves can handle every instruction class in the processor. 

Typical mapping from I'll call it primary instructions that the ISA comprehends to Micro Ops that the backend of the machine comprehends is very low, it's in the order of about a 2 to 5% penalty that you'll typically see in a machine like this.

--------------------------------------------------------------------------------
![Qualcomm-Snapdragon-X-Elite-Hot-Chips-2024_Page_08](https://github.com/user-attachments/assets/daf91018-6a97-4817-8b59-4d174af69f63)
(7:20)

Moving on to rename dispatch and execution pipes, the register files are physical register files in the machine; both vector and integer are around 400 total entries. This, as I said, feeds the downstream execution pipes and there are a set of reservation stations that you'll find in front of these pipes. kind of on the chart here it shows you and we'll see pictorially each one of these things later on but to give you a feel the integer is Six Wide, the vector unit is four wide and the load store unit is four wide as well.

the integer pipe has a variety and a mixture of instructions and classes which we'll see in the next slide, and the vector unit being four wide each pipe is 128 bits and it can handle basically every type of floating Point format that that we know today so 32 bit multi-add, 64 bit multi-add, and it also supports 16 bit data types and a variety of formats, 8 bit data types for integer operations as well, but with that let's jump to the pictorial side so you get a feel for what's in this machine.

--------------------------------------------------------------------------------
![Qualcomm-Snapdragon-X-Elite-Hot-Chips-2024_Page_09](https://github.com/user-attachments/assets/eeb54376-0630-4971-a528-05b69e831c8c)

(8:46)

You can see the reservation stations at the top of the diagram. There is a general purpose register file that gets read after the instructions come from the reservation stations and start execution. then they hit at the bottom of the diagram, the varying execution units as I said six. and the picture itself gives you a feel for the class of instructions that each pipe executes, you can see alus and shifters are present in all of the execution units, and then there's a variety of multipliers, dividers, branch execution units as well. and the notation that you see in the two block boxes on the lower right (should be lower left? I2V instruction) that's basically what I call general purpose moves, they transfer data to and from the vector register file to the scalar register file that we have here.

--------------------------------------------------------------------------------
![Qualcomm-Snapdragon-X-Elite-Hot-Chips-2024_Page_10](https://github.com/user-attachments/assets/492c4c08-ee49-40c5-a84f-376be3e1bfd0)
(9:47) 

the vector data path is very similar to the instruction one basically, a variety of execution pipes and functional units that you'll see multipliers, adders, compare instructions, very similar layout to the to the scaler design engine, and you'll notice on the right hand side there is data movement coming in from the load store unit, and remember I mentioned there are four pipes those are the four data feeds to and from the load store unit.

--------------------------------------------------------------------------------
![Qualcomm-Snapdragon-X-Elite-Hot-Chips-2024_Page_11](https://github.com/user-attachments/assets/f0141d60-91d0-4eb1-a134-8ec003852bf7)
(10:18) 
Now, load store, what is actually feeding this engine from the outside world. we have a large data cache in this engine, it's 96 kilobytes runs at core speed, it has a four cycle load to use penalty that you see in a traditional design, it's multi-ported using a off-the-shelf standard bit cell that you'd find from the foundry, a classical 64 bit cell design.
 
The translation is very similar to the integer side of the machine. It's a large 224 entry data structure. it's built to kind of balance timing and storage capacity for the physical or virtual to physical translation. We do support in the machine both 4K and 64k translation granules as well.

Now the execution pipe itself, what is load-store actually built of? It's four fundamental engines.  They can handle any combination of load store operations within the pipe. We support full forwarding. The number of inflight load store operations you can see here to the outside world is quite high, 200 plus in flight load store operations in this machine, and as well there is a very tightly integrated L2 cache.

Now not diving into a lot of detail but prefetching is one of the paramount things that you find in this architecture in this machine. there are many many advanced prefetcher algorithms, some proprietary, some that you'll find in Industry standard papers, and if you catch me later on,  I might give you a feel for what some of these are, but fundamentally the algorithms that we have in this are simple line adjacency algorithms next line prefetching or previous line prefetching,  strides, pointers, arrays, pattern prefetchers. there's a multitude of them and they're applied to both caches as well as translation structures.

--------------------------------------------------------------------------------
![Qualcomm-Snapdragon-X-Elite-Hot-Chips-2024_Page_12](https://github.com/user-attachments/assets/f45fee3e-9eb4-4fc4-9365-e26f30699e7b)
(12:35) 

okay, pictorially just like I had for the other parts of the machine, load store you can see at the diagram on the right hand side, we have four reservation stations, and you'll notice that each reservation station has 64 entries, so there's a quite a large number of inflight load store operations. Each data path can handle 16 bytes per cycle into and out of the cache. That is done to match up with the bandwidth of the vector unit and the scaler unit. That interfaces to a 96 kilobyte cache and is all the incoming requests that are being managed, you can see here over 200 entries, and there is an interface to the integrated private L2 cache of four core complex, 64 bytes per cycle into and out of this underlying design, and then to the outside  external world it has a 32 by per cycle interface. 

Now the L2 cache just a few tidbits here, very a little different than what I call large capacity integrated caches. one of the things that you'll see in a lot of the designs that I've done, that cache has a tendency to operate at core frequency with very low latency, and the motivation behind this is the engine has to be balanced for data footprint as well as code footprint as well,  and typically what you're going to see for large caches like this, this data footprint that's going to be sitting inside of this per core is somewhere between three and four megabytes of cache storage will be allocated to that processor, now it can utilize all but a typical footprint will be about that large.

and the cache operates at full core speed, with a latency that's averaging somewhere around 15 to 20 clocks depending upon which part of the cache you're communicating with. 

--------------------------------------------------------------------------------
![Qualcomm-Snapdragon-X-Elite-Hot-Chips-2024_Page_13](https://github.com/user-attachments/assets/7519cc51-c262-437b-8043-ee5debd31e77)
(14:51)
now the memory management how do we actually structure the translation from virtual to physical, as I highlighted in earlier slides there are a set of translation buffers at both the L1 level, but it's backed by a very large unified level two translation buffer, and this is done primarily to handle large data footprint, it's meant to handle all of the virtualized structures, the security layers that are there, but this structure is much larger than 8 kilo entries which is very non-typical and it's meant to try to keep the latency of translation down to an absolute minimum.

This structure as well has prefetching mechanisms built in to try to bring in the translations well in advance of the load and store operations that it's consuming. 

As I said earlier the backing structure supports the 4 and 64 KB translation granules, it does support virtualization and two-stage translation as well as nested virtualization in the design.

and in terms of outstanding table walks and flight in the machine it's not on the slide but it's much larger than it's somewhere between 10 and 20 inflight operations per clock.

--------------------------------------------------------------------------------
![Qualcomm-Snapdragon-X-Elite-Hot-Chips-2024_Page_14](https://github.com/user-attachments/assets/0e6504ff-bba2-4c90-b0ba-f313362f72b2)
(16:23) 

Now how at a high level does this set of CPU cores and processor clusters interface to the SoC in a very simplistic way. the backing system has lpddr5, it supports basically 135 gigabytes a second, the systems which you some of you may have seen out in the courtyard, there are some demo systems actually out there, those systems support up to and the SoC supports up to 64 GB of dam.

and kind of the last piece of information and I know there have been many questions by the audience and on some of the forums, what type of cache is actually out in this SoC and it turns out there's an integrated cache down below, that the CPU complex and the SoC can utilize that (SLC) cache is 6 megabytes in capacity, and it can be used for all of the engines in the SoC.

--------------------------------------------------------------------------------
![Qualcomm-Snapdragon-X-Elite-Hot-Chips-2024_Page_15](https://github.com/user-attachments/assets/757e8699-a2a4-419d-87b8-2686a48a516d)
(17:26)
as well. um with all of the interesting I'll call it attacks that have been occurring in the industry against CPU complexes and SoC. the CPU team has been spending a lot of time on security in order to try to help uh avoid any kind of breach of security within the CPU complex and we've targeted via architecture as well as microarchitecture of a bunch of different elements that have been integrated into the CPU in order to try to avoid these attacks. 

and you can see many of these styles that are listed here, some associated with side channels,  authentication mechanisms. We have our own dedicated random number generator in the CPU, but the premise behind all of this was to avoid or try to minimize any kind of attack in the CPU complex and the SoC.

--------------------------------------------------------------------------------
![Qualcomm-Snapdragon-X-Elite-Hot-Chips-2024_Page_17](https://github.com/user-attachments/assets/0725e2be-b1b4-4e4e-ba4e-c0ebe2027e58)
(18:26)

now um what do we see in a CPU that's built like this, what type of benchmarks and performance is expected out of a machine like this, so what I did here is I pulled together some data across various os's and platforms, and you can see a variety here, there's Windows, there's virtualized version of Linux on Windows, you can see Linux as well.

and kind of two standard two standard benchmarks that the industry uses to measure these CPUs. one is geekbench that you see here, the three tables on the left hand side of this kind of give you a feel for the different os's scheduling properties associated with those os's. but this is a single thread at Peak operating clock so this will be done at Snap Dragon X Elites, top frequency of right now 4.3 gigahertz. 

and so you can see on this left hand side kind of a range of scores, the variation is somewhere between five and I'll call it 12% based on where you look, but as well I wanted to include for the audience, internally measured data on platforms as well for both spec int and spec FP for spec 2017 so you can get a feel for how the processor is operating in classical integer benchmarks. kind of differences in OS stack where you see again windows and Linux, and how they compare and contrast as well as floating point benchmarks that we measure in the industry.
 
And the processor does hold up pretty well given some of the prefetch techniques that we've introduced. 

--------------------------------------------------------------------------------
![Qualcomm-Snapdragon-X-Elite-Hot-Chips-2024_Page_18](https://github.com/user-attachments/assets/451903fb-d92a-436b-a31a-97b04c9c09c7)
(20:17)

What does the latency of this CPU look like um relative to the devices that are out in the industry, you can see on the left hand side there's a transition at 96 kB which corresponds to the L1 cache capacity, this is a four cycle load to use. 

you'll see another transition occurring somewhere around the TLB capacity that's where you see some of these transitions.

And the large transition over here again is the L2 cache capacity at 12 megabytes, you can see standard stride based prefetching schemes, the engine the way it's designed keeps the latency relatively uniform, and it looks like it's hitting in the L1 cache for the majority of the time. 

to the extreme where you see out here the latency is somewhere around I'll call it 90 to 110 nanoseconds based on which aspect you're exploring in the CPU.

--------------------------------------------------------------------------------
![Qualcomm-Snapdragon-X-Elite-Hot-Chips-2024_Page_19](https://github.com/user-attachments/assets/ba0c2621-6022-426c-934c-5212bd7d0064)
(21:17)

the last one is the bandwidth if you remember in the earlier slide I talked about having 136 gigabytes per second of bandwidth available, this chart is showing you the bandwidth of a single thread given two types of load classes that the processor understands, 128 bit Vector loads, and the thing I wanted to highlight here is we see 0 to roughly 130 GBs here, the load store engine is actually able to feed this single core with roughly just shy of 100 gigabytes a second which means a lot to the CPU and its prefetchers that's why you're seeing this offset be so high. Traditionally I've seen machines operate in the 30 to 50 gigabyte per second but we're able to actually push this near to 100.

--------------------------------------------------------------------------------
![Qualcomm-Snapdragon-X-Elite-Hot-Chips-2024_Page_20](https://github.com/user-attachments/assets/07c0a196-2908-402d-85a9-a8832bacc13c)
(22:19)

Lastly, where are we trying to take this CPU complex in the industry? see it today showing up in compute with the CPU Orion technology and there are a variety of segments within Qualcomm that we're driving towards, you can see kind of in the the top we have the compute platform that  a lot of the demonstration devices outside are showing there is mobile, there is auto, and then there is basically headset VR. expect to see a variety of devices appear in these spaces with this CPU complex and its future derivatives and with that we come to the end so I'm going to open the floor up to questions

--------------------------------------------------------------------------------
 [Applause]
23:16
Now let's start with a question from slack. 

Q: Good morning and thanks for your talk Gerard, there's a couple questions around the cache so let me combine these two, one of them was curious about why the data cache is smaller than the instruction cache?

A: It's primarily due to timing, the load store Loop, the four cycle Loop of that cache is quite critical, each cycle is running near its limits, we did study larger cache capacities as well but it was driven by process more than anything.

Q: okay and then also related to that does the L2 concurrently support loads for the L1 
I-Cache and D-Cache? 

A: It does.

Q: thank you George, kosmer from chips and cheese, quick question about the system level cache, do you segment it so that the CPU can only say use two Mega or three Mega and then sort of dynamically allocate that, or could one part of the SoC use all six Mega.

A: It actually can do all of what you described, it's a programmable engine as well configurable element, there are biases built in for certain engines yes but any piece of Ip in the system if so configured can actually access the whole cache including the CPU if it wanted.

Q:  Jen Le at Future way, can I follow up with the SLC question, in terms of the quantity like six megabytes, given that L2 is already 12 megabytes for four cores what's the decision making behind the six megabyte of SlC?

A: It's what you'll discover when you start measuring and looking at it. The SLC itself predominantly minus CPU is using that cache, the CPU can use it but it's biased towards other elements, the video display engines or the GPU engine as well, but the CPU ends up steering away as you rightly asked because of its larger cache. 

Q: hello I'm makono from Lenovo, you mentioned there are three CPU clusters and one works like the efficient, the other work for the performance, is there any hardware difference or it was
programmed by just a firmware.

A: It's actually a physical construction difference only, from a software perspective they look and feel exactly the same.

Q: David Kanter. MLPerf. Congrats to you and the whole team on getting this delivered, really exciting to see. a question for you on the L1 and L2, how many outstanding demand and prefetch requests do you support?

A: just as many entries that are in those queues actually. The prefetcher, there is a bias built in obviously, but there's nothing to prohibit if the load store engine is being fed efficiently that prefetchers can actually utilize the whole structure if it wanted, there's nothing to stop it from doing that, obviously it's not going to do it on its own right so there's going to be a natural balance that you'll find based on the instruction stream itself inserting loads and stores into the queue and the prefetchers, and typically you'll see that water mark if the machines running efficiently sit around 35 to I'll call it 50% where it's normal load store and prefetch requests got it all right.

Q: The question from Robi at Intel is do you expect to customize the core clusters into hybrid architectures with different performance characteristics when you're customizing for the different segments?
A: That's TBD right now but don't be shocked if something like that happens.

Q: it looks like other new architectures have kind of gone back and forth for reservation station granularity and we've seen some people go more towards unified who'd been fine grain and you guys look like you went all in on you know fine grain granular perp Port stations so how early in your design was that kind of foregone conclusion or what led you guys to go with like the very highly parallel fine grain design?

Q: That concept actually was probably there from day one, the motivation behind it is I've gone and studied quite a lot of the other architectures, power PC, Intel, AMD's designs, all of them around the industry. 

What I tried to do fundamentally in architecting a lot of these machines and the CPUs has been trying to do as well is to look at what I call every critical timing loop in the machine and remove them. remove them as barriers, and the unified structure does have it benefits by the way but one of the elements of moving towards a machine architecture like this was to remove that scheduler’s loop scanning ability and finding what you'll hear people in CPU teams talk about finding the first instruction, finding second, third, fourth, fifth, and you have to do things in your machine to either break up statically the structure so that it's easy to find those things or you go all the way to the style of design that I showed here which is it's trivial to find, because what you'll find in these structures everything's find first one, so it breaks the timing loop.

Q: Chester lamb from chips and cheese, you have a load store unit with a very large scheduling capacity like 256 reservation station entries in total, that's actually more than 248 total load store entries so that's the opposite of how it usually is in other architectures and I wanted to understand why 

A: because you're trying to get the next set of operations into the machine. Basically it's a balance, if you go smaller you may find that there will be certain starvations, so in the performance model study that we did on this machine it turns out, and timing wise, it turned out that it was actually more beneficial to grow the capacity. 

and those structures by the way don't handle just load store operations, it does other things as well so that those things will end up taking up capacity, so we're balancing all what I call traditional load store instructions and other activities that they operate on.

Q: This one's from Daniel Deng, it looks like a lot of resources are allocated for conditional and indirect Branch prediction in Orion, what was the reason for so many resources to be focused there and then on top of a little bit of how did you make power performance tradeoffs for adding these predictive structures.

A: Yeah, that's a good one. when what what I didn't show in the slid set here um when you're looking at a machine like this, it shows the potential peak capacity of eight instructions per clock the predictive structures if you think about the nature of programs and you start studying x86 or arm or riscV, any of these architectures and you compile code, there's a rule of thumb that kind of emerges that you'll find when you study code segments and fragments that there's a branch one in every five sometimes, people call it one in every four sometimes, one in every six, but let's just call it one in every five instructions, every fetch group that this machine brings in has at least one branch, okay, at least one. so if you think about feeding this machine continuously with an average IPC of say two three or four because of these branches, you would like it to be eight if it could be, you throw as much resource as you physically can that is both timing capable and as you rightly asked that question power efficient at the same time, you throw as much resource as you can at the algorithms for conditional targets, conditional direction, in order to keep the backend of the machine fed. if you don't do that it's kind of a waste of hardware, you'll build all this backend resource and won't utilize it, so it's a balance, and the method for studying that is actually we've designed a performance model that's pretty detailed and we study capacity associativity banking of the structures. how that's all done we study that in minutia in order to figure out the right trade-off and the right size for those data structures.

Q: right and obviously it's a large code set that you have to look at to figure out which algorithms are going to be 

A: correct, yeah, and that's another interesting point that you make there, from a code set point of view, people have a tendency to bias towards benchmark ABC, you'll see them do that, what I've learned in my career is that every benchmark it doesn't matter where it comes from spec 2000, spec 2006, spec 2017, geekbench, what do you call it, Java virtual machines, large programs, server, data center, large scale machines, it turns out that the properties of these code segments they're useful to study every single one of them because they give you clues about what we've learned in computer science, things like data structures, link lists, pointers, queues, all of these things, they give you all these clues they're in these programs and programs themselves help you study the machine's architecture and how to balance it, so it's not just a bunch of transistors, it does come from computer science, so.

Q: In other words, it hasn't changed in the last 12 years.

A: no, it's the same we keep learning new mechanisms to help advance the processing.


