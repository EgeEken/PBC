# Probabilistic Brush Compression

An unconventional, lossy image compression algorithm I designed, that compresses image data as a series of approved "brush stroke" instructions, carrying many pixels worth of data per stroke using less than 1 byte per stroke.

The paper detailing the process along with the code will be uploaded soon.

# WORK IN PROGRESS
## V1.0

![image](https://github.com/EgeEken/PBC/assets/96302110/60513a43-f5ab-43e2-93c3-2011c1b61349)

![image](https://github.com/EgeEken/PBC/assets/96302110/d4378cac-5da1-4605-920d-87e9ea9adf40)

For now the biggest problem i have to fix is improving the encoding process, both in terms of depth and speed, right now it is way too lossy and also way too slow to be viable, but decoding speed is fast enough (the longest decoding process i've seen so far was 0.9s), and decoding speed could easily be tripled since the R G B channels are calculated independently, a simple multithreading implementation would be enough.

I think this is a very promising idea, i hope i get to improve it further
