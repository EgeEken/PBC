# Probabilistic Brush Compression (PBC)

An unconventional, lossy image compression algorithm I designed, that compresses image data as a series of approved "brush stroke" instructions, the core idea relies on carrying many pixels worth of data per stroke while also using less than 1 byte per stroke, effectively saving space over the uncompressed RGB image which would otherwise use 3 bytes per pixel.


### Current version: V2.2
### Hugging Face Spaces Demo (outdated): V2.1. Deployed on [Hugging Face Spaces/PBC_V2.1](https://huggingface.co/spaces/EgeEken/PBC_V2.1)

---
# Current version (V2.2) Demonstration
---

### (18x compression)
<img width="950" height="522" alt="image" src="https://github.com/user-attachments/assets/7261a093-7990-4b11-ae2f-b634584d29d3" />

https://github.com/user-attachments/assets/23b36875-5c09-4624-8175-dcef8fd52215


### (203x compression)
<img width="881" height="593" alt="image" src="https://github.com/user-attachments/assets/c6af5f7b-b43b-4fc1-927d-6ccdd49a5e67" />

https://github.com/user-attachments/assets/ee95ce52-e2c5-4336-b8ff-bc6eddc848f3

---
# Development / Version History
---

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

### V2.1 compared to JPG at an equal rate of compression: 
<img src="https://github.com/EgeEken/PBC/assets/96302110/c5b012e3-3008-4132-876b-5abdcdec9cd2" alt="Demonstration" width="40%" />

## V2.2

### Huge upgrade to the algorithm, code refactored, features added, quality improved, massively optimized, runtime reduced
Comparison of V2.1 and V2.2 default settings on the same image, same stroke count (40000), same file size / compression rate (17x) as of 06/12/2025

<img src="https://github.com/user-attachments/assets/7f3af6b4-6dca-4163-b80d-811ab887e242" alt="Demonstration" width="40%" />
<img src="https://github.com/user-attachments/assets/3175abfe-1dfc-456c-87bd-efcd746ede39" alt="Demonstration" width="40%" />


For some upgrades to V2.1, i had some ideas, but also wanted to do some analysis on how the algorithm functions as is, to maybe get ideas on how to improve it. Here are some stuff i found:

<img width="990" height="996" alt="image" src="https://github.com/user-attachments/assets/326e4bb3-8734-44b2-ab5b-882c20555d4a" />

<img width="859" height="547" alt="image" src="https://github.com/user-attachments/assets/fce40e54-7b0a-4176-a539-5e10d950b9b2" />

As expected, the loss goes down rapidly at the start, where large brush sizes are covering a big part of the image with each stroke, and as the brush size is procedurally lowered, the reduction in loss is more subtle, but this isn't a sign that the algorithm is stagnating, the smaller brush sizes are what allow for higher precision, if brush sizes were kept large, they would end up being useless after a couple thousand strokes. However, i think there's still a lot to gain from adding a "focus" mechanism that isolates channels, or even parts of the image. I'm working on that right now


## V2.3 (Work in Progress, Just Started)

### (WIP) Massive upgrade to the algorithm quality, potentially lossless compression feature coming 
After V2.2, i realised there is a lot of value to be gained from simple downsample layers before starting the brush strokes process, and made a primitive version of what it could look like to have that incorporated into the system, this version is currently far from completion but the proof of concept is very promising:

<img width="950" height="522" alt="image" src="https://github.com/user-attachments/assets/810dfe9c-5576-47bf-adb0-caf3ea1efb63" />

Just by starting with a 16x downsampled layer of the original image, instead of a single starting color canvas, despite compensating for the added bits from the uncompressed downsampled layer by reducing stroke count, we can halve the MSE loss while maintaining the compression rate. This will be the main idea V2.3 is built on.
