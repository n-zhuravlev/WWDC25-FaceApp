# WWDC25-FaceApp

We are the **FaceApp ML team**, representing a popular photo and video editor https://www.faceapp.com/
We work closely with Apple ML tools to deploy on-device vision models, but have recently encountered several challenges 
we are unable to resolve.

**We would like to discuss several issues**:

## 1. Compute Units and Model Initialization Time
We are facing challenges selecting the optimal `MLComputeUnits` configuration that ensures both performance and 
stability across a range of iPhone models.
### The `.all` Dilemma
Using the default `.all` option for `MLComputeUnits` yields inconsistent behavior. 
While it can deliver excellent performance on some devices, it may cause crashes on others. 
Hardcoding a specific compute unit, such as `.gpu`, is not a scalable solution—it may be more stable, 
but often results in significantly worse performance on newer devices like the iPhone 16 Pro Max, 
which benefit from ANE acceleration.

**Example**: In our sample project (`WWDC25-FaceApp/problem-1`), we observed the following:

- On **iPhone 16 Pro Max**:
    - Inference on `.gpu` is slow
    - Inference on `.all` and `.ane` performs well
- On **iPhone 14**:
    - Inference with `.all` crashes
    - Inference with `.ane` also crashes

As a result, `.all` is not reliable across devices, and targeting specific compute units like `.gpu` underperforms 
on newer hardware.

---

### Model Initialization Bottlenecks
We also encounter prohibitively long initialization times when targeting the Apple Neural Engine (ANE), 
which introduces significant latency before the first inference.

**Performance Impact**: For some models, `MLModel` initialization on ANE is up to **10× slower** than on GPU.

**User Experience**: This delay negatively impacts our product experience. For example, with diffusion-based models, 
ANE initialization can take longer than several inference steps combined—undermining the ANE's performance advantage.

Again referring to our example (`WWDC25-FaceApp/problem-1`), initialization on ANE is significantly slower than on GPU 
(iPhone 16 Pro Max), effectively canceling out the inference speed gains.

---

### Questions for Discussion

- What is the recommended strategy for selecting `MLComputeUnits` to ensure optimal performance and stability 
across the iPhone lineup?
- Are there known techniques or upcoming improvements to reduce model initialization time, especially on ANE?

We will attach a repository with code samples and a detailed explanation.
Thank you for your help!``