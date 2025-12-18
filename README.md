# Probabilistic Brush Compression (PBC)

An unconventional, lossy image compression algorithm I designed, that compresses image data as a series of approved "brush stroke" instructions, the core idea relies on carrying many pixels worth of data per stroke while also using less than 1 byte per stroke, effectively saving space over the uncompressed RGB image which would otherwise use 3 bytes per pixel.


### Current version: V2.3 (Latest Release)
### Hugging Face Spaces Demo: V2.3. Deployed on [Hugging Face Spaces/PBC_V2.3](https://huggingface.co/spaces/EgeEken/PBC_V2.3)

---
# Current version (V2.3) Demonstration
---

### (332x compression)
<img width=60% alt="image" src="https://github.com/user-attachments/assets/09ac32bf-7b04-4335-96f7-7dac0ef38d9f" />

<video src="https://github.com/user-attachments/assets/271245eb-eb68-4ac0-8227-c72cdb9527a4"> </video>

### (167x compression)
<img width=60% alt="image" src="https://github.com/user-attachments/assets/105912bb-a5f7-4136-a36b-7c997ac7fd95" /> 

<video src="https://github.com/user-attachments/assets/dc0f9080-32b7-4fde-91e1-10febc6fdd81"> </video>

### Comparison to JPEG at equivalent compression rate

<img width="4623" height="1779" alt="image" src="https://github.com/user-attachments/assets/a0373b46-00b7-435f-9090-6540e88de6c6" />
<img width="5370" height="1670" alt="image" src="https://github.com/user-attachments/assets/2867f5fc-d9d5-47c5-ae53-ffe1a4205c91" />

### Hugging Face Spaces Demo

<img width="944" height="548" alt="image" src="https://github.com/user-attachments/assets/16a5ddf7-d833-454c-8142-d05a09982e22" />


---
# Development / Version History
---

## V0.0 (proof of concept)

This was the proof of concept for the idea i had in my mind, i was disappointed to see such terrible results but the very fact that it worked at all was proof enough for me to continue improving it.

<img alt="V0 0 proof of concept" src="https://github.com/user-attachments/assets/543bc587-084e-4241-9272-838225bc9fbb"  width="30%" />

And after tweaking the structure a little, adding some of the base features that would later become the V1.0 model, i was able to make it reach this:

<img alt="V0 1 better proof of concept" src="https://github.com/user-attachments/assets/6ecfbd5b-28db-46ae-8a27-8483407c08d8"  width="30%" />

Still not good at all, but at this point i knew i was onto something, and later the project kept evolving, until it reached its current stage:

<img width="873" height="158" alt="image" src="https://github.com/user-attachments/assets/8075f821-9892-4349-9da6-d740de61bbd8" />




## V1.0

This is the base model, it works by taking in an image, a size list, a multiplier list, and generating random positions to place "brush strokes" on, and then checking every combination of size and multipliers from the lists to find the best attributes for that specific spot, then encoding them as the indexes for the used size and multipliers. 

<img src="https://github.com/EgeEken/PBC/assets/96302110/60513a43-f5ab-43e2-93c3-2011c1b61349" alt="Demonstration" width="40%" />
<img src="https://github.com/EgeEken/PBC/assets/96302110/f582782f-4ae4-4790-95da-9f4c81dac18e" alt="Demonstration" width="40%" />

---

## V2.0

In this version the biggest difference is in the encoding speed, i achieved a great deal of optimization by using a function to find the best multiplier instead of checking each individual combination. And also by dividing each brush stroke into 4 quadrants, and saving the multipliers of each. This way every seed gets used, so there is no need to signify spaces/strokes like in V1.0, this allows for linear time complexity (O(n), as opposed to the exponential O(n^2) in V1.0).

<img src="https://github.com/EgeEken/PBC/assets/96302110/a230e39b-63d4-49c1-bf73-890c81a15fa4" alt="Demonstration" width="40%" />
<img src="https://github.com/EgeEken/PBC/assets/96302110/0ce1bb10-62c7-4f18-a00f-4c69216587ab" alt="Demonstration" width="40%" />


I also made a gradio demo for the compressor, where it is much easier and faster to test out different parameters before using the dedicated encoder to compress into a file.


<img src="https://github.com/EgeEken/PBC/assets/96302110/79f04588-a7a9-44db-8962-4d924c68a7b7" alt="gradio_demo" width="50%" />

---

## V2.1

This is essentially the same model as V2.0 but with a good amount of extra options that allow further customization in encoding. Such as the starting color options, cutoff value, and a bunch of new features to the gradio interface, which is also now hosted 24/7 on [Hugging Face Spaces/PBC_V2.1](https://huggingface.co/spaces/EgeEken/PBC_V2.1)

<img src="https://github.com/EgeEken/EgeEken/assets/96302110/da61d3e8-434b-4679-925d-987f19d41771" alt="Demonstration" width="70%" />

Another thing i added later is custom size functions, for this i had help from a friend who studied more maths than i did, but essentially it is a function that allows for gradual transition from linear to sigmoid functions, with only 16 additional bits of info required for a lot of customization. The difference is pretty subtle, but worth the effort in my opinion, the difference is visible at the same rate of compression between a linear size function and a custom finetuned one.

<img src="https://github.com/EgeEken/PBC/assets/96302110/aed58c72-dbb5-475b-befe-8d986b8d2ae0" alt="Demonstration" width="30%" />
<img src="https://github.com/EgeEken/PBC/assets/96302110/2982d7ad-c97e-487c-836b-d3c516dbb315" alt="Demonstration" width="60%" />

### V2.1 compared to JPG at an equal rate of compression: 
<img src="https://github.com/EgeEken/PBC/assets/96302110/c5b012e3-3008-4132-876b-5abdcdec9cd2" alt="Demonstration" width="40%" />

---

## V2.2

### Huge upgrade to the algorithm, code refactored, features added, quality improved, massively optimized, runtime reduced
Comparison of V2.1 and V2.2 default settings on the same image, same stroke count (40000), same file size / compression rate (17x) as of 06/12/2025

<img src="https://github.com/user-attachments/assets/7f3af6b4-6dca-4163-b80d-811ab887e242" alt="Demonstration" width="40%" />
<img src="https://github.com/user-attachments/assets/3175abfe-1dfc-456c-87bd-efcd746ede39" alt="Demonstration" width="40%" />

For some upgrades to V2.1, i had some ideas, but also wanted to do some analysis on how the algorithm functions as is, to maybe get ideas on how to improve it. I conducted a ton of experiments, finetuning parameters, observing the compression process to see points of weakness, eventually settled on these default parameters which worked pretty well on my experiment set. Also massive optimizations after refactoring the whole code using the Numba library and more appropriate data types.

---

## V2.3

### Big upgrade to the compression quality with very little cost to compression rate
After V2.2, i realised there is a lot of value to be gained from simple downsample layers before starting the brush strokes process. This simple change had a huge effect on output quality.

<img width="950" height="522" alt="image" src="https://github.com/user-attachments/assets/810dfe9c-5576-47bf-adb0-caf3ea1efb63" />

Just by starting with a 16x downsampled layer of the original image, instead of a single starting color canvas, despite compensating for the added bits from the uncompressed downsampled layer by reducing stroke count, we can halve the MSE loss while maintaining the compression rate. This is the main idea V2.3 is built on.

At pre-release V2.3 had already passed a very important milestone, which is that it can achieve better MSE loss at an equal/higher rate of compression compared to JPEG, which is the standard algorithm for lossy image compression:

### JPEG | 171x Compression | 209 MSE Loss 
<img width="1224" height="918" alt="EGE_JPG_MILESTONE171x209-small" src="https://github.com/user-attachments/assets/8af3309f-5185-4e7c-98cf-fa1b97713fd4" />

### PBC V2.3 Preview | 174x Compression | 164 MSE Loss
<img width="1224" height="918" alt="EGE_V2_3_MILESTONE174x164-small" src="https://github.com/user-attachments/assets/b8a62ef9-f611-45a5-96f4-4b84036ee8ff" />

After some more parameter finetuning and stabilization, V2.3 is consistently better than JPEG at the ultra high compression space (below 10% in JPEG's quality setting) in any image over 4 MP in resolution.

<img width="5370" height="1598" alt="image" src="https://github.com/user-attachments/assets/9c0d29b1-e1f0-4e89-9a41-ecb680cb49ac" />

<img width="4623" height="1779" alt="image" src="https://github.com/user-attachments/assets/6eb7a5ed-1503-4454-86ff-4a6591f907e5" />

<img width="4388" height="1779" alt="image" src="https://github.com/user-attachments/assets/25488ff8-63d9-4c4b-983e-1c9b8cfa1a38" />


