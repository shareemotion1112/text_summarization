Nowadays deep learning is one of the main topics in almost every field.

It helped to get amazing results in a great number of tasks.

The main problem is that this kind of learning and consequently neural networks, that can be defined deep, are resource intensive.

They need specialized hardware to perform a computation in a reasonable time.

Unfortunately, it is not sufficient to make deep learning "usable" in real life.

Many tasks are mandatory to be as much as possible real-time.

So it is needed to optimize many components such as code, algorithms, numeric accuracy and hardware, to make them "efficient and usable".

All these optimizations can help us to produce incredibly accurate and fast learning models.

Our work focused on two main tasks that have gained significant attention from researchers, that 11 are automated face detection and emotion recognition.

Since these are computationally intensive 12 tasks, not much has been specifically developed or optimized for embedded platforms.

We show how 31 dataset that has 6 classes FIG2 .

To perform reasonable tests an input image of size 100x100x3 has been used.

As shown in Fig. 4 34 we compared results based on computation time for the pipeline with and without accelerations.

Raspberry needs a computation time that is double the time needed by a Movidius, for example the 36 first needs 150ms per frame against the 70ms of the latter.

We conducted several tests and reported 37 the inference time for each task and for the whole pipeline in TAB0 .

<|TLDR|>

@highlight

Embedded architecture for deep learning on optimized devices for face detection and emotion recognition 