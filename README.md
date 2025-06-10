# WWDC25-FaceApp

We are the **FaceApp ML team**, representing a popular photo and video editor https://www.faceapp.com/

We would like to discuss two issues:

On the latest devices, especially with ANE and All Units, our models have fast inference times (~200ms) but extremely slow initialization (up to 10 seconds).
- How can we decrease initialization time?
- Is it possible to cache an already-initialized model to avoid repeated delays?

**2. Slow First Inference on iOS 18+**  
First inference on iOS 18 is slower than on iOS 16, resulting in degraded performance on new devices.
- How can we identify and address the cause?

**3. Slow Inference for Custom Convolution Operations**  
Some architectures use custom convolutions with variable weights (e.g., weights computed on the fly). Even simple operations (like squaring the weight) cause significant slowdowns.
- How can we add logic to model weights without increasing inference time?

We will attach a repository with code samples and a detailed explanation.
Thank you for your help!``