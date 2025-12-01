# Probabilistic Brush Compression (PBC)

An unconventional, lossy image compression algorithm I designed, that compresses image data as a series of approved "brush stroke" instructions, carrying many pixels worth of data per stroke while also using less than 1 byte per stroke, effectively saving space over the uncompressed RGB image which would otherwise use 3 bytes per pixel.


### Current version: V2.1. Deployed on [Hugging Face Spaces/PBC_V2.1](https://huggingface.co/spaces/EgeEken/PBC_V2.1)
### Current version compared to JPG at an equal rate of compression: 
<img src="https://github.com/EgeEken/PBC/assets/96302110/c5b012e3-3008-4132-876b-5abdcdec9cd2" alt="Demonstration" width="40%" />

# WORK IN PROGRESS (V2.1 released, currently working on V3.0)
## V1.0

This is the base model, it works by taking in an image, a size list, a multiplier list, and generating random positions to place "brush strokes" on, and then checking every combination of size and multipliers from the lists to find the best attributes for that specific spot, then encoding them as the indexes for the used size and multipliers. 

<img src="https://github.com/EgeEken/PBC/assets/96302110/60513a43-f5ab-43e2-93c3-2011c1b61349" alt="Demonstration" width="40%" />
<img src="https://github.com/EgeEken/PBC/assets/96302110/f582782f-4ae4-4790-95da-9f4c81dac18e" alt="Demonstration" width="40%" />

## V2.0

In this version the biggest difference is in the encoding speed, i achieved a great deal of optimization by using a function to find the best multiplier instead of checking each individual combination. And also by dividing each brush stroke into 4 quadrants, and saving the multipliers of each. This way every seed gets used, so there is no need to signify spaces/strokes like in V1.0, this allows for linear time complexity (O(n), as opposed to the exponential O(n^2) in V1.0).

<img src="https://github.com/EgeEken/PBC/assets/96302110/a230e39b-63d4-49c1-bf73-890c81a15fa4" alt="Demonstration" width="40%" />
<img src="https://github.com/EgeEken/PBC/assets/96302110/0ce1bb10-62c7-4f18-a00f-4c69216587ab" alt="Demonstration" width="40%" />


I also made a gradio demo for the compressor, where it is much easier and faster to test out different parameters before using the dedicated encoder to compress into a file.


<img src="https://github.com/EgeEken/PBC/assets/96302110/79f04588-a7a9-44db-8962-4d924c68a7b7" alt="gradio_demo" width="60%" />

## V2.1

This is essentially the same model as V2.0 but with a good amount of extra options that allow further customization in encoding. Such as the starting color options, cutoff value, and a bunch of new features to the gradio interface, which is also now hosted 24/7 on [Hugging Face Spaces/PBC_V2.1](https://huggingface.co/spaces/EgeEken/PBC_V2.1)

<img src="https://github.com/EgeEken/EgeEken/assets/96302110/da61d3e8-434b-4679-925d-987f19d41771" alt="Demonstration" width="80%" />

Another thing i added later is custom size functions, for this i had help from a friend who studied more maths than i did, but essentially it is a function that allows for gradual transition from linear to sigmoid functions, with only 16 additional bits of info required for a lot of customization. The difference is pretty subtle, but worth the effort in my opinion, the difference is visible at the same rate of compression between a linear size function and a custom finetuned one.

<img src="https://github.com/EgeEken/PBC/assets/96302110/aed58c72-dbb5-475b-befe-8d986b8d2ae0" alt="Demonstration" width="30%" />
<img src="https://github.com/EgeEken/PBC/assets/96302110/2982d7ad-c97e-487c-836b-d3c516dbb315" alt="Demonstration" width="60%" />

## V2.2 (Work in Progress)

For some upgrades to V2.1, i had some ideas, but also wanted to do some analysis on how the algorithm functions as is, to maybe get ideas on how to improve it. Here are some stuff i found:

<img width="859" height="547" alt="image" src="https://github.com/user-attachments/assets/fce40e54-7b0a-4176-a539-5e10d950b9b2" />

As expected, the loss goes down rapidly at the start, where large brush sizes are covering a big part of the image with each stroke, and as the brush size is procedurally lowered, the reduction in loss is more subtle, but this isn't a sign that the algorithm is stagnating, the smaller brush sizes are what allow for higher precision, if brush sizes were kept large, they would end up being useless after a couple thousand strokes.

<img width="813" height="1606" alt="image" src="https://github.com/user-attachments/assets/8e1cb3a3-f431-4021-8226-c47e7b08abae" />



I think this is a very promising idea, i hope i get to improve it further
