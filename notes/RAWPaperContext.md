
Page
4
of 6
Decision Feedback In-Context Symbol Detection
over Block-Fading Channels
Li Fan, Jing Yang, Cong Shen
Charles L. Brown Department of Electrical and Computer Engineering, University of Virginia, USA
E-mails: {lf2by,yangjing,cong}@virginia.edu
Abstract—Pre-trained Transformers, through in-context learn-
ing (ICL), have demonstrated exceptional capabilities to adapt
to new tasks using example prompts without model update.
Transformer-based wireless receivers, where prompts consist of
the pilot data in the form of transmitted and received signal
pairs, have shown high estimation accuracy when pilot data are
abundant. However, pilot information is often costly and limited
in practice. In this work, we propose the DEcision Feedback IN-
ContExt Detection (DEFINED) solution as a new wireless receiver
design, which bypasses channel estimation and directly performs
symbol detection using the (sometimes extremely) limited pilot
data. The key innovation in DEFINED is the proposed decision
feedback mechanism in ICL, where we sequentially incorporate
the detected symbols into the prompts to improve the detec-
tions for subsequent symbols. Extensive experiments across a
broad range of wireless communication settings demonstrate
that DEFINED achieves signiﬁcant performance improvements,
in some cases only needing a single pilot pair.
I. INTRODUCTION
Wireless receiver symbol detection focuses on identifying
the transmitted symbols over fading channels with varying
signal-to-noise ratios (SNRs). Traditional methods typically
follow a two-step process: ﬁrst estimating the channel using,
e.g., the Minimum Mean Square Error (MMSE) estimator, and
then performing symbol detection using the estimated channel.
However, this approach can be computationally intensive and
is impacted by the channel estimation quality. Data-driven
approaches, such as deep learning models [1], [2] that di-
rectly learn channel estimators and symbol detectors, offer an
alternative. However, deep neural networks (DNNs) require a
large amount of data and often perform poorly in the low-data
regime. More importantly, adapting pre-trained DNNs to new
wireless conditions remains a challenge [3].
Advances in Transformer models, particularly decoder-only
architectures like GPT [4], have demonstrated impressive
performance across various ﬁelds. Recent result [5] shows
that pre-trained Transformers can generalize to new tasks
during inference through in-context learning (ICL), without
requiring explicit model updates. Speciﬁcally, given an input
of the form (y1, f (y1), . . . , yn, f (yn), yquery), a pre-trained
Transformer can approximate f (yquery) based on the provided
context, where (y1, . . . , yn, yquery) represents features and f
can represent various classes of functions [6].
LF and CS were partially supported by the US National Science Foundation
(NSF) under Grants AST-2132700, CNS-2002902, and ECCS-2029978. JY
was partially supported by the US NSF under Grants CNS-1956276 and CNS-
2114542.
Wireless symbol detection, which involves estimating trans-
mitted symbols from noisy received signals, aligns well with
the ICL frameworks. [7] introduces Transformers for this
task using ICL, framing it as a regression problem with
Mean Square Error (MSE) loss and achieving near-MMSE
performance. Later works expand this framework: [8] extends
it to MIMO systems, and [9] demonstrates its robustness in
multi-user MIMO environments. Meanwhile, [10] employs
language models to reformulate detection as a linguistic task.
These advances highlight Transformers as a powerful tool for
addressing wireless communication challenges.
Despite these successes, prior studies face limitations. Most
approaches treat detection as a regression task, requiring
MSE-based objectives and post-processing to map continuous
outputs to discrete symbols. Additionally, many require ample
pilot pairs, which may not be possible in practice, and large
models increase inference costs, limiting real-world feasibility.
Inspired by decision feedback in wireless communication
(e.g., decision feedback equalizers over multi-path fading
channels), we propose a new prompt design by incorporating
decision pairs. Speciﬁcally, we combine the current received
signal with the model’s detection as pairs, merging them
with previous prompts to form a new, larger prompt for
subsequent symbol detection. Our DEFINED model uses a
carefully designed mixture training process to achieve high
performance with limited pilots (sometimes only a single pilot)
and maintain accuracy with sufﬁcient pilots. Extensive ex-
periments across modulation schemes validate our approach’s
effectiveness. To summarize, our main contributions include:
• We develop a Transformer model that jointly performs
channel estimation and symbol detection. The key inno-
vation in DEFINED is leveraging decision feedback with
limited pilot data to expand the effective context length
and enhance performance.
• We design a mixed training process for DEFINED that
achieves high performance improvement with limited
pilots and strong accuracy with sufﬁcient pilot data,
making the model adaptable to practical scenarios.
• We validate our approach with experiments across mul-
tiple modulation schemes.
II. SYSTEM MODEL AND CANONICAL METHODS
A. Wireless Model
To more clearly illustrate our design, we consider a canon-
ical receiver symbol detection problem over a standard nar-
2025 IEEE International Conference on Communications (ICC): SAC Machine Learning for Communications and Networking Track
979-8-3315-0521-9/25/$31.00 ©2025 IEEE 3785
ICC 2025 - IEEE International Conference on Communications | 979-8-3315-0521-9/25/$31.00 ©2025 IEEE | DOI: 10.1109/ICC52391.2025.11161684
Authorized licensed use limited to: University of Virginia Libraries. Downloaded on February 03,2026 at 15:19:11 UTC from IEEE Xplore. Restrictions apply.
rowband wireless fading channel. Speciﬁcally, we focus on an
(Nr × Nt) MIMO system, where the channel is represented
by an Nr × Nt complex-valued matrix Ht at time t, following
a distribution PH . We normalize the channel coefﬁcients such
that each entry in Ht has a unit variance. The received signal at
time t is expressed as: yt = Htxt +zt, where the channel noise
zt ∈ CNr is modeled as a complex additive white Gaussian
noise vector with zero mean and covariance matrix σ2I. Each
entry of the input vector xt ∈ CNt is sampled uniformly
at random from a given constellation set X (e.g., QPSK or
16QAM), and this modulation process is independently and
identically distributed (i.i.d.) across both time and space. We
normalize the signal to ensure a unit average total transmit
power, i.e. E[‖xt‖2] = 1. The average signal-to-noise ratio
(SNR) at any receive antenna is given by SNR = 1/σ2.
We focus on the block-fading channel model [11] in this
paper, where the channel Ht remains constant over a coherent
time period of T time slots, and is i.i.d. across different
coherence periods. In other words, Ht = hl, ∀t = (l − 1)T +
1, · · · , lT for the l-th coherence period where hl is drawn i.i.d.
from PH . Correspondingly, the data transmission is organized
into frames, where each frame has a length that is at most T .
The frame structure is designed such that the ﬁrst k transmitted
symbols are known and pre-determined pilot symbols, whose
original purposes include assisting the receiver to perform
channel estimation of hl so that it can perform coherent
symbol detection. In other words, based on the reception of a
few pilot pairs Dk = {(y1, x1), · · · , (yk, xk)}, the goal is to
design a demodulator that accurately recovers the transmitted
symbol xk+1, · · · , xT from the received signal yk+1, · · · , yT
with high probability.
B. Canonical Methods for Symbol Detection
In the traditional approach, the receiver ﬁrst estimates the
channel using pilot signals, then performs symbol detection
on the received signal yt via hypothesis testing for each t =
k + 1, · · · , T . Typically, the (Linear) MMSE estimator is used
for channel estimation, and the MMSE channel estimate ˆH
is given by: ˆHMMSE
k = Y X†(XX† + σ2I)−1, where X is
the pilot matrix and Y is the received signal matrix. With
the estimated channel, the transmitted symbol ˆxt is detected
by projecting yt onto the closest symbol in the modulation
constellation X as: ˆxt = arg minx∈X ‖ ˆHx − yt‖2, ∀t = k +
1, · · · , T.
This two-step process treats channel estimation and symbol
detection as separate tasks. Such decoupling can result in
suboptimal detection, particularly under noisy conditions or
limited pilot data [12]. Optimal estimators like MMSE rely
on precise statistical models of the channel and noise, which
are often hard to obtain. Additionally, these estimators are
computationally intensive due to matrix inversions and poste-
rior probability calculations, making them less appealing for
real-time applications in high-dimensional systems.
To address these challenges, data-driven methods [13] have
been explored for joint channel estimation and symbol de-
tection. Deep learning architectures have shown promise [1],
[2], [14]–[16], but their high data requirements [17] and poor
adaptability to varying channel conditions without retraining
limit their real-world applicability [3].
III. IN-CONTEXT LEARNING-BASED SYMBOL DETECTION
ICL for symbol detection leverages the structure of wireless
communication frames, particularly in block-fading channels
where channel conditions remain stable during the coher-
ent time. Within each frame, pilot signals are followed by
subsequent received signals, which naturally align with the
Transformer architecture’s strength in processing sequence-
based inputs. The Transformer’s ability to model dependencies
among sequential data allows it to capture complex relation-
ships within transmitted signals, making it highly effective for
symbol detection tasks.
In this section, we introduce the ICL-based symbol detection
method. We ﬁrst formulate the symbol detection problem,
and then present the ICL implementation using a popular
Transformer model GPT-2, which also serves as the backbone
of our proposed solution1.
A. ICL-based Problem Formulation
Each ICL detection task τ corresponds to a latent channel
H and a channel noise level σ2, following the unknown joint
distribution Pτ = PH Pσ2 . The ICL-based symbol detection
does not have prior knowledge of the speciﬁc task τ and
is provided only with a prompt Sτ
t = (Dτ
k , yt), consisting
of k target pairs, Dτ
k = {(y1, x1), · · · , (yk, xk)}, which are
sampled from the conditional distribution Px,y|τ and serve
as in-context examples for the current task τ , along with yt,
∀t = k + 1, · · · , T .
As previously mentioned, each context pair (xi, yi) and the
inference pair (yt, xt) are i.i.d. samples given task τ . For
block-fading channels, the context set Dτ
k , also referred to as
pilot signals in wireless communication, enhances the channel
estimation, thereby improving the reliability and accuracy of
data transmission. We note that a signiﬁcant advantage of
the proposed solution is that there is no need to change the
existing frame structure or the design of pilot signals. Rather,
the innovation is entirely at the receiver side where we leverage
the pilot and decoded signals in a different way. This is an
important advantage in practice as it allows for (backward)
compatibility with the existing standard.
The goal of symbol detection is to identify the correspond-
ing input signal xt for the new query signal yt from the
same task. The ICL-based detection makes its decision as
follows: ˆxt = fθ (Sτ
t ), where θ represents the parameters of the
model. The detection for the query is measured by the Symbol
Error Rate (SER), which is the frequency at which transmitted
symbols are incorrectly decoded. The expected SER for the
new query with k contexts, taking the expectation over the
task distribution for ∀k = 1, · · · , T − 1, is deﬁned as:
SERk(θ) = Eτ EDτ
k ,yt|τ [fθ (Dτ
k , yt) 6 = xt] . (1)
1We use GPT-2 with elaborated design choices as a concrete example
throughout the paper. However, the proposed principle can be easily adapted
to other Transformer architectures.
2025 IEEE International Conference on Communications (ICC): SAC Machine Learning for Communications and Networking Track
3786
Authorized licensed use limited to: University of Virginia Libraries. Downloaded on February 03,2026 at 15:19:11 UTC from IEEE Xplore. Restrictions apply.
B. Vanilla In-Context Symbol Detection
Transformer models have emerged as powerful tools for
symbol detection [18], leveraging their ability to capture
long-range dependencies. This approach to wireless symbol
detection was introduced in [7], [8]. The input-output struc-
ture is illustrated in Figure 1. With a causal masked self-
attention mechanism, the model outputs the detection ˆxt at
the corresponding position of yt, relying only on known
preceding contexts and the received signal. During the forward
process, the Transformer solves k + 1 detection problems
for the same task τ , using an increasing number of pilot
data points. Their results in [7], [8] demonstrate that the
Transformer exhibits strong capabilities in symbol estimation
within context, without requiring explicit model updates.
Fig. 1. Decoder-only Transformer architecture for ICL-based symbol detec-
tion with k pilots. Detection output applies to ∀t = k + 1, · · · , T .
IV. DECISION FEEDBACK IN-CONTEXT DETECTION
The vanilla ICL approaches for symbol detection require
sufﬁcient context to achieve accurate estimation, which is often
impractical in real-world scenarios. Pilot signals are costly and
limited, reducing their adaptability for practical applications.
For situations where the number of pilots is small, neither
conventional two-stage (channel estimation then symbol detec-
tion) nor vanilla ICL solutions can achieve good performance.
Furthermore, these approaches generally formulate the symbol
detection task as a regression problem, as in [8], [9], [19],
where a Transformer is trained to minimize the MSE loss.
Although their models achieve performance comparable to the
optimal MMSE estimator for x, an additional projection step
is required to map the output to the appropriate transmitted
symbol, leading to a mismatch and losing optimality.
In contrast, we directly deﬁne the problem as a classiﬁcation
task, enabling the model to jointly learn channel estimation
and symbol detection while directly measuring the SER during
inference. Additionally, we generalize the approach to effec-
tively handle scenarios where pilot information is highly lim-
ited by sequentially feeding back the already decoded symbol
pairs as noisy pilots and incorporating them as part of the
prompt. Our model demonstrates robust performance even in
challenging conditions with only a single pilot, outperforming
previous ICL models that struggle with insufﬁcient pilot data.
At the same time, it maintains high accuracy when sufﬁcient
pilot data is available.
Inspired by the decision-feedback concepts in wireless com-
munication, we propose the DEcision Feedback IN-ContExt
Detection (DEFINED) method for symbol detection, as shown
Fig. 2. DEFINED model architecture with k pilots and (t − k − 1) decision
feedback contexts to detect xt, ∀t = k + 1, · · · , T .
in Figure 2. The DEFINED model extends the prompt by
incorporating the previously received signals and detection
decisions alongside prior prompts to improve subsequent de-
tections. While traditional decision-feedback equalizers (DFE)
focus on inter-symbol interference (ISI), our study addresses
narrowband channels without ISI. Nevertheless, decision feed-
back is effective here due to the latent common channel. Noisy
feedback also provides valuable information, further reﬁning
model detection.
A. Model Parameters
Our speciﬁc Transformer model is designed with an embed-
ding dimension of de = 64, L = 8 layers, and h = 8 attention
heads, resulting in approximately 0.42 million parameters,
which is signiﬁcantly smaller compared to large language
models (LLMs) commonly applied in wireless communication
tasks, such as those discussed by [10], [20]. For instance, even
LLMs like GPT-J 6B contain over 6 billion parameters, making
them approximately 14,000 times larger than our model. This
compact size not only enables deployment on edge devices but
also signiﬁcantly shortens the inference time, enabling low-
latency detection at the receiver.
B. Training Details
In this section, we describe our data generation process
and the training of the DEFINED model. Training includes a
pre-training phase to equip the model with general predictive
abilities and speed up convergence, followed by ﬁne-tuning to
adapt the model to scenarios with limited pilot data.
1) Data Generation: We generate data according to the
wireless communication model described in Section III-A.
Speciﬁcally, we consider both SISO and 2x2 MIMO systems
and explore various modulation schemes, including BPSK,
QPSK, 16QAM, and 64QAM. For each wireless system and
modulation task, we generate prompts consisting of sequences
with T pilot pairs, with the maximum sequence length set
to T = 31. Both systems operate under a Rayleigh fading
channel, where the channel coefﬁcient is sampled as H ∼
PH = CN (0, 1). The channel noise is sampled i.i.d. from a
Gaussian distribution zt ∼ CN (0, σ2I). Within each block,
the noise variance σ2 is constant and shared across all noise
samples. However, for different blocks, σ2 is i.i.d. sampled
from a uniform distribution Pσ2 over the range [σ2
min, σ2
max]
to improves adaptability. This variability applies only during
2025 IEEE International Conference on Communications (ICC): SAC Machine Learning for Communications and Networking Track
3787
Authorized licensed use limited to: University of Virginia Libraries. Downloaded on February 03,2026 at 15:19:11 UTC from IEEE Xplore. Restrictions apply.
training, while evaluation is conducted at ﬁxed SNR levels for
controlled performance assessment.
2) ICL Pre-training: We delve into the details of model
training, which is divided into two phases, as shown in
Figure 3. First, we deﬁne ICL-training and ICL-testing as
operations on ground-truth data, represented by the clean
prompt: SICL
t = {y1, x1, . . . , yt−1, xt−1, yt}, for t =
1, 2, · · · , T. On the other hand, DF-training and
DF-testing use iteratively decoded sequences with
k pilot data and model decision feedback, which
operate on the decision feedback prompt: SDF
t =
{y1, x1, · · · , yk, xk, yk+1, ˆxk+1, . . . , yt−1, ˆxt−1, yt}, for t =
k + 1, . . . , T, where each estimation ˆxt relies on the ﬁrst k
pilot points and prior model decisions.
Fig. 3. The training process includes pre-training on clean data, followed
by ﬁne-tuning on a mixed dataset of clean and decision feedback noisy data.
The model demonstrates strong performance, adapting to both limited and
sufﬁcient pilot scenarios during symbol detection (i.e., inference).
We deﬁne the loss functions for the ICL-training and DF-
training models. The adopted loss function is the cross-entropy
loss between the model’s output and the ground-truth labels:
LICL(θ) = 1
N T
N∑
i=1
T∑
t=1
loss (fθ (SICL
t,i ), xt,i
) , (2)
LDF(θ) = 1
N (T − k)
N∑
i=1
T∑
t=k+1
loss (fθ (SDF
t,i ), xt,i
) . (3)
where θ represents the model parameters, and N is the number
of samples.
DF-training is trained with limited pilot data and utilizes
self-sampled labels, where noisy feedback is combined with
previous contexts to generate new contexts. This process is
time-intensive, as each detection and feedback step requires a
model forward pass. Additionally, noisy data complicates con-
vergence, which can cause the model to struggle to converge
effectively. Training the Transformer with ICL-training and
testing under DF-testing introduces a data mismatch: training
uses clean data, while testing involves noise. Despite this
mismatch, we observed that the Transformer’s performance
demonstrates signiﬁcant robustness to noise during DF-testing
and maintains acceptable performance. ICL-training is also
about ten times faster than DF-training, as it eliminates data
sampling and operates solely on clean data.
Considering all factors, our ﬁnal proposed solution is to per-
form ICL-training ﬁrst, followed by tailored DF-training. Here,
ICL-training serves as pre-training, while tailored DF-training
acts as ﬁne-tuning, similar to the pre-training and ﬁne-tuning
process used in LLMs. Training epochs are carefully structured
into two phases, as shown in Figure 3. In the ﬁrst phase, the
model converges just before reaching a plateau, at which point
we transition to the tailored DF-training method. During this
transition, a spike in the training loss is observed due to the
shift in the training data distribution. As shown later, ICL pre-
training not only accelerates convergence but also improves
recognition of clean data and ICL-testing performance.
3) Decision Feedback Fine-tuning: Instead of vanilla DF
training in the second phase, we employ a carefully designed
ﬁne-tuning process. The loss function is constructed as a linear
combination of the previously deﬁned losses in Equations (2)
and (3), where α controls the balance between them, and the
hyperparameter α is set to 0.7 during model training,
Lﬁne-tuning(θ) = αLDF (θ) + (1 − α)LICL(θ). (4)
As explained in the pre-training phase, after ICL pre-
training, the model is capable of general symbol detection,
performing well on detections with clean data and, to some
extent, on detections using the decision feedback method.
Furthermore, in the ﬁne-tuning process, the training loss
is designed to emphasize decision feedback detection while
retaining the model’s ability to handle clean data.
Training on both clean and noisy data enhances the robust-
ness of the Transformer model by exposing it to a more diverse
dataset. Ultimately, we propose that a single Transformer
model can be trained to perform both ICL-testing and DF-
testing, making our DEFINED model adaptable for practical
wireless systems. For example, in scenarios with sufﬁcient
pilot information, the model can operate in the ICL manner.
However, in challenging situations – common in real-world ap-
plications – where pilots are limited and difﬁcult to acquire, the
model can utilize previous decisions to improve performance
in subsequent symbol detection.
V. EXPERIMENT
We analyze DEFINED’s performance against baselines al-
gorithms under both ICL-testing and DF-testing, represented
by the ”DEFINED-ICL” and ”DEFINED-DF” curves, respec-
tively. Our model not only performs well with sufﬁcient pilot
data but also exhibits signiﬁcant improvements in limited-pilot
scenarios by effectively utilizing noisy feedback. Moreover,
DEFINED demonstrates strong performance in complex mod-
ulation tasks, highlighting the Transformer’s ability to capture
geometric structures within modulation constellations.
A. Baseline Algorithms
We introduce several baseline algorithms, including the ICL
method, the MMSE algorithm, and MMSE-DF, a decision-
feedback variant of MMSE.
1) In-Context Learning: We train a Transformer from
scratch using vanilla ICL-training and evaluate the SER for
both ICL-testing and DF-testing, shown as “ICL-ICL” and
“ICL-DF” curves in the ﬁgures, respectively.
2025 IEEE International Conference on Communications (ICC): SAC Machine Learning for Communications and Networking Track
3788
Authorized licensed use limited to: University of Virginia Libraries. Downloaded on February 03,2026 at 15:19:11 UTC from IEEE Xplore. Restrictions apply.
(a) SISO BPSK at SNR = 5dB with 1 Pilot,
gainDF = 11.2%
(b) SISO QPSK at SNR = 10dB with 1 Pilot,
gainDF = 11.6%
(c) SISO 16QAM at SNR = 20dB with 1
Pilot, gainDF = 43.7%
(d) SISO 64QAM at SNR = 25dB with 1
Pilot, gainDF = 45.6%
(e) MIMO BPSK at SNR = 10dB with 2
Pilots, gainDF = 16.8%
(f) MIMO QPSK at SNR = 15dB with 2
Pilots, gainDF = 38.6%
Fig. 4. SISO performance for BPSK, QPSK, 16QAM, and 64QAM with one pilot, and 2 × 2 MIMO for BPSK and QPSK with two pilots. The X-axis shows
context sequence length, where the t-th point for ∗-ICL uses t ground truth pilots and ∗-DF-Pk uses k clean pilots and (t − k) decision feedback noisy pairs.
2) MMSE Algorithm: This is a coherent detection algo-
rithm, which ﬁrst estimates the channel with the assistance
of pilots and then performs detections for the subsequent
received signals using the estimated channel. In the case of
Rayleigh fading, both the channel and noise follow complex
Gaussian distributions. The MMSE estimator for H is given
by: ˆHMMSE
k = Y X†(XX† + σ2I)−1. With the t-th received
signal yt, the transmitted symbol xt is estimated by projection
onto the closest symbol in X :
ˆxt = arg min
x∈X ‖ ˆHMMSE
k x − yt‖2, ∀t = k + 1, · · · , T. (5)
With k pilot pairs, the mean SER is computed, shown as a
horizontal line labeled ”MMSE-Pk”.
3) MMSE-DF Algorithm: MMSE-DF is an extension of the
previous MMSE solution by sequentially using the decision
feedback data as if they were new pilot pairs. Starting with k
pilots, we compute the MMSE estimator of H and detect ˆxk+1
using yk+1 as in Equation (5). The decision pair (yk+1, ˆxk+1)
merges with the existing dataset, and iteratively used to detect
each signal until ˆxT . We plot the SER against the decision
feedback-extended context sequence length.
B. Experimental Results
During testing, we sample 80,000 prompts to compute the
mean SER. Results for BPSK, QPSK, 16QAM, and 64QAM
in SISO, and BPSK and QPSK in 2 × 2 MIMO, plot SER
against context sequence length, as shown in Figure 4. The t-
th point represents the SER of ˆxt+1 using t contexts, omitting
the 0-th point (random guessing) due to high SER.
To quantify the SER improvement with increasing con-
text length under DF-testing, we deﬁne the gain metric as:
gainDF =
( SERk (θ)−SERT −1(θ)
SERk (θ)
)
× 100%, which represents the
percentage reduction in SER as the context length increases
from k to (T − 1), starting from k known pilots.
1) Comparison with Baseline Algorithms: The MMSE
algorithm is shown alongside horizontal lines representing
MMSE performance with k pilots and with full (30) pilots,
respectively. The ICL-ICL line, for the model trained and
tested with ICL method, shows that with 30 pilots, the
Transformer slightly outperforms the MMSE algorithm during
ICL testing. This improvement arises from model’s ability
to perform channel estimation and symbol detection jointly,
leveraging the synergy between these tasks.
The DEFINED-DF line denotes the performance of our
proposed model during DF-testing, showing a marked SER
reduction as more decision feedback data is incorporated.
For instance, in Figure 4d, for 64QAM with one pilot, the
SER decreases from 0.206 to 0.112 as the feedback context
sequence length increases, achieving a 45.6% SER gain.
This conﬁrms that the Transformer can effectively use noisy
feedback to improve detections with limited pilot data.
Our model also performs well in ICL-testing, as indicated
by the DEFINED-ICL line, which aligns closely with the ICL-
ICL line. This result suggests that ICL pre-training, followed
by carefully designed loss functions during decision feedback
ﬁne-tuning, allows the model to learn effectively from clean
data. Additionally, our DEFINED model adapts well to real-
world symbol detection, excelling with ample pilot data and
performing effectively even with a single pilot.
The ICL-DF line, which represents a Transformer trained
with ICL method but tested under DF conditions, performs
signiﬁcantly worse than our model, nearly coinciding with
the MMSE-DF line. This observation highlights that models
trained solely on clean data struggle with noisy feedback, as
they simply treat all feedback data as clean. This underscores
2025 IEEE International Conference on Communications (ICC): SAC Machine Learning for Communications and Networking Track
3789
Authorized licensed use limited to: University of Virginia Libraries. Downloaded on February 03,2026 at 15:19:11 UTC from IEEE Xplore. Restrictions apply.
the importance of DF ﬁne-tuning for handling noisy feedback.
2) Comparison with Different SNRs and Varying Pilot
Lengths: At high SNR levels, reduced data noise enables more
accurate detection from pilot data, enhancing DEFINED model
performance and accentuating the downward SER trend. How-
ever, at very high SNRs, the already low initial SER limits
further improvement. Our DEFINED model also performs
robustly with minimal pilot data, including the extreme single-
pilot cases. As pilot data increases, all algorithms show
improved performance in DF inference.
3) Comparison with Different Modulation Schemes: We
conduct experiments using BPSK, QPSK, 16QAM, and
64QAM modulations in the SISO system, representing clas-
siﬁcation tasks with 2, 4, 16, and 64 classes, respectively. As
modulation complexity increases, the detection task becomes
more challenging, but we observe greater performance im-
provements with complex schemes, as reﬂected by a more
pronounced SER decrease with additional feedback data in
our DEFINED model.
Analysis of the Transformer’s output logit vector shows that
nearly all of the incorrect detections occur within a small
region around the ground-truth label in the constellation. This
“typical error event” [11] suggests that even incorrect detec-
tions carry valuable information, as the noisy label is often
near the correct one, enhancing the model’s detections. Thus,
as modulation complexity increases, the compact constellation
set allows noisy feedback to provide more useful information,
leading to better SER gains with added feedback data.
These ﬁndings demonstrate that our DEFINED model ef-
fectively captures the constellation set’s geometry. It not only
learns detections but also recognizes relationships between
classes, often assigning higher probabilities to neighboring
labels when errors occur. Due to inherent data noise – e.g.,
channel fading and additive noise – received signals with
nearby latent labels in the constellation may overlap. As a
result, the model sees adjacent labels as close neighbors,
letting its detections retain valuable information, even when
they are not entirely accurate.
4) MIMO Results: The results for MIMO systems with
BPSK and QPSK using two pilots are shown in Figures 4e
and 4f. Compared across different modulation schemes, SNR
levels, our DEFINED model can still perform well for high
dimensional detection problems. In fact, DEFINED exhibits
a more pronounced decrease in SER with the inclusion of
additional decision feedback data, leveraging the underlying
communication system structure more effectively.
VI. CONCLUSION
Inspired by the decision feedback mechanism in wireless
receiver designs, we proposed DEFINED to enhance symbol
detection by incorporating decision pairs into the prompts of
Transformer. Our approach achieved signiﬁcant performance
gains with limited pilot data while maintaining high accu-
racy with sufﬁcient pilot data, demonstrating its adaptability
for practical scenarios. Extensive experiments across various
modulation schemes validated the robustness and ﬂexibility
of our model. These contributions highlighted the potential
of Transformers, underscoring their capabilities for future
wireless communication systems.
REFERENCES
[1] C. Lu, W. Xu, H. Shen, J. Zhu, and K. Wang, “MIMO channel
information feedback using deep recurrent network,” IEEE Commun.
Letter, vol. 23, no. 1, pp. 188–191, 2018.
[2] D. Neumann, T. Wiese, and W. Utschick, “Learning the MMSE channel
estimator,” IEEE Trans. Signal Processing, vol. 66, no. 11, pp. 2905–
2917, 2018.
[3] O. Simeone, S. Park, and J. Kang, “From learning to meta-learning:
Reduced training overhead and complexity for communication systems,”
in 2020 2nd 6G Wireless Summit (6G SUMMIT). IEEE, 2020, pp. 1–5.
[4] A. Radford, J. Wu, R. Child, D. Luan, D. Amodei, I. Sutskever et al.,
“Language models are unsupervised multitask learners,” OpenAI blog,
vol. 1, no. 8, p. 9, 2019.
[5] B. Mann et al., “Language models are few-shot learners,” in Advances
in Neural Information Processing Systems, H. Larochelle, M. Ranzato,
R. Hadsell, M. Balcan, and H. Lin, Eds., vol. 33. Curran Associates,
Inc., 2020, pp. 1877–1901.
[6] S. Garg, D. Tsipras, P. S. Liang, and G. Valiant, “What can transformers
learn in-context? a case study of simple function classes,” Advances in
Neural Information Processing Systems, vol. 35, 2022.
[7] V. Teja Kunde, V. Rajagopalan, C. Shekhara Kaushik Valmeekam,
K. Narayanan, S. Shakkottai, D. Kalathil, and J.-F. Chamberland,
“Transformers are provably optimal in-context estimators for wireless
communications,” arXiv e-prints, pp. arXiv–2311, 2023.
[8] M. Zecchin, K. Yu, and O. Simeone, “In-context learning for MIMO
equalization using transformer-based sequence models,” in 2024 IEEE
International Conference on Communications Workshops (ICC Work-
shops). IEEE, 2024, pp. 1573–1578.
[9] ——, “Cell-free multi-user MIMO equalization via in-context learning,”
in IEEE International Workshop on Signal Processing Advances in
Wireless Communications (SPAWC). IEEE, 2024, pp. 646–650.
[10] M. Abbas, K. Kar, and T. Chen, “Leveraging large language models
for wireless symbol detection via in-context learning,” arXiv preprint
arXiv:2409.00124, 2024.
[11] D. Tse and P. Viswanath, Fundamentals of Wireless Communication.
Cambridge University Press, 2005.
[12] C. Shen and M. P. Fitz, “MIMO-OFDM beamforming for improved
channel estimation,” IEEE J. Select. Areas Commun., vol. 26, no. 6, pp.
948–959, August 2008.
[13] O. Simeone, “A very brief introduction to machine learning with
applications to communication systems,” IEEE Trans. Cogn. Commun.
Netw., vol. 4, no. 4, pp. 648–664, 2018.
[14] F. Liang, C. Shen, and F. Wu, “An iterative BP-CNN architecture for
channel decoding,” IEEE J. Sel. Topics Signal Process., vol. 12, no. 1,
pp. 144–159, February 2018.
[15] F. A. Aoudia and J. Hoydis, “End-to-end learning for ofdm: From neural
receivers to pilotless communication,” IEEE Trans. Wireless Commun.,
vol. 21, no. 2, pp. 1049–1063, 2021.
[16] S. Park, H. Jang, O. Simeone, and J. Kang, “Learning to demodulate
from few pilots via ofﬂine and online meta-learning,” IEEE Trans. Signal
Processing, vol. 69, pp. 226–239, 2020.
[17] H. Kim, S. Kim, H. Lee, C. Jang, Y. Choi, and J. Choi, “Massive MIMO
channel prediction: Kalman ﬁltering vs. machine learning,” IEEE Trans.
Commun., vol. 69, no. 1, pp. 518–528, 2020.
[18] W. Shen, R. Zhou, J. Yang, and C. Shen, “On the training con-
vergence of transformers for in-context classiﬁcation,” arXiv preprint
arXiv:2410.11778, 2024.
[19] A. Ravent´os, M. Paul, F. Chen, and S. Ganguli, “Pretraining task
diversity and the emergence of non-bayesian in-context learning for
regression,” Advances in Neural Information Processing Systems, 2024.
[20] J. Shao, J. Tong, Q. Wu, W. Guo, Z. Li, Z. Lin, and J. Zhang,
“WirelessLLM: Empowering large language models towards wireless
intelligence,” arXiv preprint arXiv:2405.17053, 2024.
2025 IEEE International Conference on Communications (ICC): SAC Machine Learning for Communications and Networking Track
3790
Authorized licensed use limited to: University of Virginia Libraries. Downloaded on February 03,2026 at 15:19:11 UTC from IEEE Xplore. Restrictions apply.
