### Acknowledgments 

We would like to thank the anonymous reviewers and (S)ACs of ICML 2024 for their constructive comments and dedicated service to the community. We also would like to thank all reviewers’ engagement during the author-reviewer discussion period.

----

### Meta Review by Area Chair

This work represents a step towards cross-problem generalization in neural combinatorial optimization and thus the AC recommends acceptance. This is the first work to apply mixture-of-experts models to vehicle routing problems. The reviewers agree that the paper is well-written and the experiments are extensive. A weakness of this work is that mixture-of-experts yields marginal improvement upon a multi-task learning baseline (gap improvement of 0.1-0.5%). Furthermore, the proposed hierarchical gating mechanism does not appear to be advantageous beyond the default node-level input-choice gating. There is also concern about the limited scalability of the methods and the lack of discussion about other types of generalization in VRP, which was partially addressed by the authors' rebuttal.

----

### Official Review by Reviewer YE8Q

**Summary:**

In this paper, the authors propose a multi-task vehicle routing solver as a unified neural solver to solve a wide range of vehicle routing problem (VRP) variants simultaneously. This solver is based on Mixture-of-Experts models which help scale up the model capacity without a proportional increase in computation. Additionally, they also introduce a hierarchical gating mechanism in the MoE layer, which illustrates the tradeoff between empirical performance and computational complexity. Finally, they study the effects of MoE configuration on solving VRPs.

**Strengths And Weaknesses:**

**Strengths**

1. This is the first work to leverage mixture of experts models in solving vehicle routing problems.
2. The paper is well written and well organized.

**Weaknesses**

1. Although using mixture-of-experts model in constructing a solver for VRPs, the scale of number of parameters is not significant. 
2. A theoretical analysis on the effects of hierarchical gating mechanism would help strengthen the paper.

**Questions:**

1. Are there any intuitions to choose the best number of expert $m$?
2. In equation (2), if we replace the Top-K sparse softmax gate by another kind of sparse gate, namely, temperature softmax gate [1], would it help increase the performance?
3. Does the scaling law in [2] apply to this work?

**References**

[1] X. Nie, X. Miao, S. Cao, L. Ma, Q. Liu, J. Xue, Y. Miao, Y. Liu, Z. Yang, and B. Cui. Evomoe: An evolutional mixture-of-experts training framework via dense-to-sparse gate, 2022. Arxiv preprint.

[2] Jakub Krajewski, Jan Ludziejewski, Kamil Adamczewski, Maciej Pióro, Michał Krutul, Szymon Antoniak, Kamil Ciebiera, Krystian Król, Tomasz Odrzygóźdź, Piotr Sankowski, Marek Cygan, Sebastian Jaszczur. Scaling Laws for Fine-Grained Mixture of Experts. Arxiv preprint.

**Post Rebuttal:** 

```
Dear the Authors,

Thanks for your reponses, which partially addresses my concens about the paper. Regarding the theoretical analysis, I suggest that the authors should take into account a recent line of works on the MoE theory [1, 2, 3, 4], and include a discussion about those papers to explain for the challenges of understanding the hierarchical gating mechanism theoretically. Given what we discussed, I think my positive rating 5 is reasonable, so I decide to keep it unchanged contingent upon the inclusion of our discussion in the paper revision. 

[1] Z. Chen, Y. Deng, Y. Wu, Q. Gu, and Y. Li. Towards understanding the mixture-of-experts layer in deep learning. Advances in NeurIPS, 2022.

[2] Nguyen, H., Nguyen, T., and Ho, N. Demystifying softmax gating function in Gaussian mixture of experts. Advances in NeurIPS, 2023.

[3] Nguyen, H., Akbarian, P., Yan, F., and Ho, N. Statistical perspective of top-k sparse softmax gating mixture of experts. In ICLR, 2024.

[4] Nguyen, H., Akbarian, P., and Ho, N. Is Temperature Sample Efficient for Softmax Gaussian Mixture of Experts? arXiv preprint.

Best,

Reviewer YE8Q
```

**Limitations:**

See Weaknesses section.

**Ethics Flag:** No

**Soundness:** 3: good

**Presentation:** 3: good

**Contribution:** 2: fair

**Rating:** 5: Borderline accept: Technically solid paper where reasons to accept outweigh reasons to reject, e.g., limited evaluation. Please use sparingly.

**Confidence:** 2: You are willing to defend your assessment, but it is quite likely that you did not understand the central parts of the submission or that you are unfamiliar with some pieces of related work. Math/other details were not carefully checked.

**Code Of Conduct:** Yes

----

### Official Review by Reviewer BaVm

**Summary:**

This paper introduces MVMoE, a Multi-Task Vehicle Routing Solver with Mixture-of-Experts. Unlike most constructive solvers for VRPs, the authors consider the setting in which a single model can be applied to VRPs of different variants via attribute composition and thus generalize to new problem variants. To improve performance, MVMoE uses a mixture of experts, by which the model can better generalize by specializing model parameters to particular settings and introduces a hierarchical gating mechanism for it. The model is tested on 16 different VRP variants, where training is done on 6 VRPs and 10 new problems are used for testing zero-shot generalization. The model demonstrates improvements compared to a previous multi-task learning (MTL) baseline.

**Strengths And Weaknesses:**

**Strengths**

The paper is well-written and easy to understand. The paper improves on previous models by combining the attribute composition idea from [1] and a mixture of experts applied on top of the POMO model. Experimental settings and ablation studies, including trying (very) recent methods such as the new Soft mixture of experts from ICLR 2024, are extensive. The paper holds significance in the Neural Combinatorial Optimization community as some of the first steps towards cross-problem generalization.

**Weaknesses**

The major weaknesses of the paper, in my opinion, are the rather incremental results and ideas. MVMoE is mostly a pretty straightforward combination of the approach in [1] with a Mixture of experts. Moreover, the model does not seem to actually be “general”. Generalization for VRPs can be broadly divided, as the authors explain, into (1) cross-size, (2) cross-distribution, and (3) cross-problem generalization. However, the proposed approach can basically only tackle (3). The results are incremental compared to the POMO-MTL approach from [1]. Also, it does not outperform POMO trained for a single problem. Most importantly, results for generalizations (1) and (2) do not appear to be promising (for instances of size <300 nodes, >5% average gap in CVRP, and >18% for CVRPTW. Finally, even though the proposed approach is solved with RL, it would be good to show performance against other more scalable approaches such as [2] or [3]. No code was provided at the time of submission.

------

**References**

[1] Liu, Fei, et al. "Multi-Task Learning for Routing Problem with Zero-Shot Generalization." (2023).

[2] Drakulic, Darko, et al. "BQ-NCO: Bisimulation Quotienting for Efficient Neural Combinatorial Optimization." Advances in Neural Information Processing Systems 36 (2024).

[3] Luo, Fu, et al. "Neural combinatorial optimization with heavy decoder: Toward large scale generalization." Advances in Neural Information Processing Systems 36 (2024).

**Questions:**

1. Why did you choose reinforcement learning to solve the problem and not supervised learning instead?
2. Why did you consider the open route as a constraint, and not just for reward (=cost) calculation? In [1], authors did not consider this for the masking procedure. This might be a reason why you needed to include OVRPTW in the training set.
3. Could the mixture of experts help in cross-size and cross-distribution generalization? I.e., one could train the model on a range of sizes, instead of only one as $n=100$.
4. How would the model perform on larger-scale settings, such as $n=1000$?

**Post Rebuttal:** 

```
I would like to thank the authors for their responses which clarified my concerns. I will raise my score to 7 and recommend acceptance for your paper.
```

**Limitations:**

See above weaknesses. No negative societal impact is foreseen for the problem at hand.

**Ethics Flag:** No

**Soundness:** 4: excellent

**Presentation:** 4: excellent

**Contribution:** 2: fair

**Rating:** 6 -> 7: Accept: Technically solid paper, with high impact on at least one sub-area, or moderate-to-high impact on more than one areas, with good-to-excellent evaluation, resources, reproducibility, and no unaddressed ethical considerations.

**Confidence:** 4: You are confident in your assessment, but not absolutely certain. It is unlikely, but not impossible, that you did not understand some parts of the submission or that you are unfamiliar with some pieces of related work.

**Code Of Conduct:** Yes

----

### Official Review by Reviewer Uupk

**Summary:**

This paper develops a multi-task vehicle routing solver with mixture-of-experts (MVMoE). It proposes a hierarchical gating mechanism to balance the computation complexity and the empirical performance. The paper compares the method with several baselines such as classical solver (Or-tools and LKH3), single-task neural solver (POMO), and multi-task neural solver (POMO-MTL). Extensive experiments on 16 VRP variants show the the MVMoE outperforms POMO-MTL on both seen and unseen tasks. Although the results are still much worse than classical solver Ortools, I think the paper is interesting and it is a nice try to solve multi-task VRP problems with more parameters.

**Strengths And Weaknesses:**

The topic of solving multi-task VRPs, especially the generalization on unseen tasks is interesting. I think the paper is a nice try to introduce the MoE to NCO.

The authors experiment on 16 VRP variants and conduct the ablation on the Position of MoEs, Number of Experts, and Gating Mechanism to show how the MOE could benefit the CO fields. 

I think the paper is a nice application paper, however, the methodology contribution is a bit marginal, except for the hierarchical gating mechanism.

The results show the method could outperform POMO-MTL. However, the results are still much worse than classical solver ORtools. It still has a lot of room to improve considering the inferior performance or ORtools in OR fields.

The paper is mostly well-written and easy to read. However, some related work from TOP conferences is missing such as NeuralLKH, TAM, etc. I suggest the authors to conduct more comprehensive reviews.

If the code is released, it would benefit the reproducibility of the paper and the community.

**Questions:**

The paper only experiments on small-scale VRPs, it would be interesting to show the performance of the scalability of the method, such as VRP with over 1000 nodes, which is conducted in Learn-to-delegate and TAM, etc.

Given the computation burden of the large model, how much time does it take to solve large-scale VRPs?

Why the MoE are not used in the FFN of Encoder?

How to improve the MVMoE to outperform Classic solvers?

The inference time of 4E and 4E-L model seems very similar in the results. More experiments should be done to compare the inference time of the two models.

**Post Rebuttal:** 

```
Thanks for the detailed responses and new experiments. Most of my concerns have been addressed. I have raised my score.
```

**Limitations:**

It will be good to mention the limitations of MVMoE in the main part.

**Ethics Flag:** No

**Soundness:** 3: good

**Presentation:** 3: good

**Contribution:** 2: fair

**Rating:** 5 -> 7: Accept: Technically solid paper, with high impact on at least one sub-area, or moderate-to-high impact on more than one areas, with good-to-excellent evaluation, resources, reproducibility, and no unaddressed ethical considerations.

**Confidence:** 4: You are confident in your assessment, but not absolutely certain. It is unlikely, but not impossible, that you did not understand some parts of the submission or that you are unfamiliar with some pieces of related work.

**Code Of Conduct:** Yes

----

### Official Review by Reviewer w7KH

**Summary:**

This paper presents a new method for the multi-task vehicle routing problem, using a mixture of experts. The method is designed to achieve zero-shot generalizability across different vehicle routing tasks, including those not seen during training, as long as the problem can be described using both dynamic and static features. Experimental results indicate that the method provides adequate zero-shot generalization and some degree of few-shot generalization across various vehicle routing tasks.

**Strengths And Weaknesses:**

Strengths:

- The application of a mixture of experts to the multi-task Vehicle Routing Problem (VRP) is both innovative and appears to be effective.
- The paper is clearly written and accessible.
- The experimental design is well-organized.

Weaknesses:

- The scope of the experiments is somewhat limited, with sample sizes of only 50 and 100.
- The justification for addressing the multi-task VRP needs to be more compellingly articulated.

**Questions:**

**The Necessity of Multi-Task VRP**: One might argue that individual tasks within a Vehicle Routing Problem (VRP) can be mastered within three days, allowing for the development of fast models capable of zero-shot inference. This suggests that, in practical terms, we could train models for each specific task and deploy the most appropriate one as needed. However, the question arises: why is there a need for a multi-task VRP approach?

**Extending Zero-Shot Generalization to Novel Problems**: Is there a strategy to broaden the scope of zero-shot generalization to encompass problems that cannot be adequately described through a mix of static and dynamic contexts? This inquiry probes the potential for models to adapt and perform in situations beyond their initial training environments.

------

**Discussion**

**3. Meta-Learning Approaches and Scale Shift**: Could you engage in a discussion about the application of meta-learning strategies, particularly those designed to address scale shifts in single-task scenarios [1,2], as illustrated in the referenced works?

[1] Son, Jiwoo, et al. "Meta-SAGE: scale meta-learning scheduled adaptation with guided exploration for mitigating scale shift on combinatorial optimization." International Conference on Machine Learning. PMLR, 2023.

[2] Qiu, Ruizhong, Zhiqing Sun, and Yiming Yang. "Dimes: A differentiable meta solver for combinatorial optimization problems." Advances in Neural Information Processing Systems 35 (2022): 25531-25546.

**3-1. Mixture of Experts for Improved Zero-Shot Generalization:**

Regarding scale shift, is it feasible to employ a proposed mixture of experts to enhance zero-shot generalization capabilities across different scales, potentially surpassing the approaches in the mentioned studies?

**3-2. Meta-Learning Extensions for VRP Tasks:**

Can the proposed ideas be extended to facilitate meta-learning across various VRP tasks, suggesting a more adaptive and comprehensive strategy for tackling diverse VRP challenges?

**Post Rebuttal:** 

```
The rebuttal was quite effective and addressed most of my concerns. Consequently, I have updated my score to a 7.
```

**Limitations:**

Limited scalability and limited formulation of task.

**Ethics Flag:** No

**Soundness:** 4: excellent

**Presentation:** 3: good

**Contribution:** 3: good

**Rating:** 6 -> 7: Accept: Technically solid paper, with high impact on at least one sub-area, or moderate-to-high impact on more than one areas, with good-to-excellent evaluation, resources, reproducibility, and no unaddressed ethical considerations.

**Confidence:** 4: You are confident in your assessment, but not absolutely certain. It is unlikely, but not impossible, that you did not understand some parts of the submission or that you are unfamiliar with some pieces of related work.

**Code Of Conduct:** Yes