# Probabilistic Brush Compression (PBC)

An unconventional lossy image compression algorithm I designed. It compresses image data as a series of approved "brush stroke" instructions. The core idea relies on carrying many pixels worth of data per stroke while also using less than 1 byte per stroke, effectively saving a lot of space over the uncompressed RGB image which would otherwise use 3 bytes per pixel.


### Current latest version: V3.0
### Demo deployed on [Hugging Face Spaces](https://egeeken-pbc.hf.space/)
<img width="1359" height="821" alt="image" src="https://github.com/user-attachments/assets/5582a25f-7b21-4525-bf92-d94c94a94799" />



---
# Current version (V3.0) Demonstration
---

<img width="630" height="514" alt="image" src="https://github.com/user-attachments/assets/f27fc40f-d5d6-41f1-8ec4-f9a0c82744c9" />
<video src="https://github.com/user-attachments/assets/6a4755dd-f81c-4411-b674-6e1823d601de"> </video>

### Comparison to JPEG

Much higher quality with 3 times smaller file size (this is as small as JPEG can go)
<img width="1238" height="536" alt="image" src="https://github.com/user-attachments/assets/e480cef1-1811-40c5-a098-c4a03c6def35" />
Zoomed in (I want to reiterate: The PBC file on the right is 3 times smaller than the JPEG on the left)
<img width="1180" height="390" alt="image" src="https://github.com/user-attachments/assets/766e6f84-19cd-4463-bae6-1992e093e409" />

### Comparison to AVIF (The state of the art in extreme image compression)

Can compress further than AVIF, while almost matching its rate distortion at the lowest file sizes, but AVIF is still better.
<img width="1220" height="476" alt="image" src="https://github.com/user-attachments/assets/8298ad8e-6b31-442b-b936-11dc90de4acd" />
Unique benefit to PBC architecture: The algorithm achieves essentially perfect reconstruction of blurry unfocused backgrounds at a nearly negligible cost. This is a benefit of the iterative residual encoding over the transformative encoding most of the established codecs use, which causes artifacts for these unfocused sections of an image at low quality settings.
<img width="1181" height="389" alt="image" src="https://github.com/user-attachments/assets/c83e6177-407e-4c53-86b6-cec3061874b9" />


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

---

## V2.4

### Optimization and refactoring update, as well as a new fancy demo website. And some hyperparameter tuning tools and research.
The codebase had gotten messy over the course of like a year of adding new features and testing stuff, so i started fromn scratch to rebuild 2.3 and make some optimizations along the way. Quality-wise nothing has changed between V2.3 and V2.4, but encode and decode speeds have been increased around 20-50%.

<img width="630" height="470" alt="image" src="https://github.com/user-attachments/assets/c5a49f78-4ce9-4399-ad8c-f7dee116da33" />

<img width="630" height="470" alt="image" src="https://github.com/user-attachments/assets/af63b41d-306e-4e7e-871f-a00e935f779b" />

---

## V3.0

### Complete algorithm overhaul, masssive gains in quality, compression and speed. 

Changed the guided random placements of PBC2 into a grid-patch system that eliminates residual error one patch at a time. Maximizing quality gained per bit added using a search algorithm. Parts of the search algorithm are then distilled into a supervised tiny neural network, which is then placed in a RL training environment to maximize its efficiency further.

### Quality and Compression

PBC3.0 achieves much higher quality than PBC2.4, at even smaller file sizes. The quality ceiling has been raised massively, although there still is one, lossless encoding is not yet available (Intended feature for 3.1 or maybe 3.2 depending on the rate of progress)

<img width="563" height="310" alt="image" src="https://github.com/user-attachments/assets/b2c6c110-90d4-4d65-b777-23ea55b4683c" />


PBC3.0 now very favorably compares to JPEG's higher compression settings, especially at higher resolutions. Achieving up to 20 times better compression while conserving more quality.

<img width="1220" height="555" alt="image" src="https://github.com/user-attachments/assets/8ee12fce-0718-49ea-83a8-eb7b792743da" />

<img width="1229" height="552" alt="image" src="https://github.com/user-attachments/assets/8ea8ab83-03ad-4488-baa4-6f1029625fcf" />


Rate distortion chart comparing PBC3.0 (red) to JPEG (blue) for 12 MP images. While the quality ceiling is still a problem to tackle, on the extreme compression space PBC completely dominates.

<img width="1201" height="524" alt="image" src="https://github.com/user-attachments/assets/f1464b74-057f-42a9-aad1-5c4bd7e548b0" />


Rate distortion chart comparing PBC3.0 to various state of the art image compression codecs. While the state of the art for lossy compression is still far ahead of PBC, the gap has closed up much more compared to PBC2.4, and more improvements are still on the way.

<img width="1199" height="526" alt="image" src="https://github.com/user-attachments/assets/94f5cec5-a7de-4f41-b812-92a60514af8d" />


### Speed

PBC3.0 architecture is much more efficient than 2.4, achieving much better speed. In it's current state (coded in python, no multi-threading) it's already comparable to the state of the art image codecs, encoding faster than AVIF and JPEG XL.

<img width="740" height="274" alt="image" src="https://github.com/user-attachments/assets/efc850cb-44bd-4c9e-90bd-426915a643cc" />

Outliers (JXL encoding under q3 and JPEG2000 decoding over q70 are *incredibly* slow) removed, PBC3 remains competitive

<img width="1027" height="399" alt="image" src="https://github.com/user-attachments/assets/1c860ae8-5efb-4445-84c0-4a6b2778089b" />


The roadmap for improvements has a bunch of obvious speedups, so expect these results to get much better.

### 
