# Probabilistic Brush Compression

An unconventional, lossy image compression algorithm I designed, that compresses image data as a series of approved "brush stroke" instructions, carrying many pixels worth of data per stroke using less than 1 byte per stroke.

The paper detailing the process along with the code will be uploaded soon.

## WORK IN PROGRESS, PROOF OF CONCEPT

![image](https://github.com/EgeEken/PBC/assets/96302110/89d18020-ddb4-4add-9312-8d755bf3ec69)

![image](https://github.com/EgeEken/PBC/assets/96302110/6f1b87ac-6bf3-4e7c-b4cf-17a01bf3ae0a)


For now the biggest problem i have to fix is improving the encoding process, both in terms of depth and speed, right now it is way too lossy and also way too slow to be viable, but decoding speed is fast enough (the longest decoding process i've seen so far was 0.9s), and decoding speed could easily be tripled since the R G B channels are calculated independently, a simple multithreading implementation would be enough.

I think this is a very promising idea, i hope i get to improve it further
