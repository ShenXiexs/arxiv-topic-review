# T2I Jailbreak 相关论文（CCF A/B）

说明：
- 选取标准：CCF 2022 列表中 A/B 会刊；主题聚焦 T2I jailbreak 攻击、红队、评测，以及与 jailbreak 强相关的防御（安全对齐、概念抹除、水印、检测、投毒等）。
- 输出结构：每篇包含框架图解读（以论文核心框架图为线索）、总结与分析。

## 1. Red-Teaming Text-to-Image Systems by Rule-based Preference Modeling
Cao, Y., Miao, Y., Gao, X. S., & Dong, Y. (2025). NeurIPS 2025 (CCF A).  
PDF: [arXiv](https://arxiv.org/pdf/2505.21074)

[Image]

**Stage 1: Prompt Modification and Query（提示词修改与查询）**  
- LLM Agent 从原始提示词生成多种改写版本。  
- 黑盒 T2I 系统返回图像或拒绝结果。  

**Stage 2: Rule-based Preference Modeling（偏好建模）**  
- 基于 NSFW / SFW / Reject 构造偏好顺序。  
- 规则化评分细化粗标签反馈。  

**Stage 3: DPO / LLM Fine-tuning（直接偏好优化）**  
- 以偏好对微调 LLM。  
- 形成多轮闭环提升越狱成功率。  

**Summary（总结）**  
- 提出 RPG-RT 红队框架，通过规则化偏好学习适配未知防御。  
- 在多种黑盒系统上自动发现绕过提示。  

**Analysis（分析）**  
- 优势是“自适应黑盒红队”，适合评估商业 API 安全边界。  
- 防御侧可用其做持续红队测试与漏洞发现。  

## 2. Fuzz-Testing Meets LLM-Based Agents: An Automated and Efficient Framework for Jailbreaking Text-To-Image Generation Models
Dong, Y., Meng, X., Yu, N., Li, Z., & Guo, S. (2025). IEEE S&P 2025 (CCF A).  
PDF: [arXiv](https://arxiv.org/pdf/2408.00523)

[Image]

**Stage 1: Seed Pool（种子池初始化）**  
- 收集原始与已越狱提示词作为种子。  

**Stage 2: Guided Mutation（LLM 引导变异）**  
- LLM Agent 进行语义一致的变异生成。  

**Stage 3: Oracle Evaluation（判定与反馈）**  
- 通过黑盒查询与判定器筛选有效样本。  

**Summary（总结）**  
- 将 fuzzing 与 LLM Agent 结合，实现高效 jailbreak 搜索。  
- 强调自然语义与高覆盖提示空间。  

**Analysis（分析）**  
- 适合大规模自动化红队评测。  
- 依赖查询预算与判定器准确度。  

## 3. SneakyPrompt: Jailbreaking Text-to-image Generative Models
Yang, Y., Hui, B., Yuan, H., Gong, N., & Cao, Y. (2024). IEEE S&P 2024 (CCF A).  
PDF: [arXiv](https://arxiv.org/pdf/2305.12082)

[Image]

**Stage 1: Blocked Prompt（被拦截提示词）**  
- 以触发安全过滤的提示词为输入。  

**Stage 2: RL-based Perturbation（强化学习扰动）**  
- RL 策略迭代扰动 token 并查询系统。  

**Stage 3: Success Feedback（成功反馈更新）**  
- 根据是否生成/拒绝更新策略。  

**Summary（总结）**  
- 首个自动化 T2I jailbreak 攻击框架。  
- 在 DALL·E 2 和 SD + 安全过滤器上验证有效。  

**Analysis（分析）**  
- 揭示“文本过滤不足以保证安全”。  
- 多次查询成本高，适合离线评测。  

## 4. PLA: Prompt Learning Attack against Text-to-Image Generative Models
Lyu, X., Liu, Y., Li, Y., & Xiao, B. (2025). ICCV 2025 (CCF A).  
PDF: [ICCV](https://openaccess.thecvf.com/content/ICCV2025/papers/Lyu_PLA_Prompt_Learning_Attack_against_Text-to-Image_Generative_Models_ICCV_2025_paper.pdf)

[Image]

**Stage 1: Learnable Prompt Setup（可学习提示初始化）**  
- 将提示词表示为可优化参数。  

**Stage 2: Multimodal Similarity Proxy（相似度代理梯度）**  
- 用多模态相似度近似梯度信号。  

**Stage 3: Black-box Optimization（黑盒优化）**  
- 优化提示以绕过过滤与安全检查。  

**Summary（总结）**  
- 提出黑盒条件下的提示学习攻击框架。  
- 性能优于传统词替换攻击。  

**Analysis（分析）**  
- 攻击更系统化，提示搜索更强。  
- 防御需考虑对抗式提示学习。  

## 5. JailbreakDiffBench: A Comprehensive Benchmark for Jailbreaking Diffusion Models
Jin, X., Weng, Z., Guo, H., Yin, C., Cheng, S., Shen, G., & Zhang, X. (2025). ICCV 2025 (CCF A).  
PDF: [ICCV](https://openaccess.thecvf.com/content/ICCV2025/papers/Jin_JailbreakDiffBench_A_Comprehensive_Benchmark_for_Jailbreaking_Diffusion_Models_ICCV_2025_paper.pdf)

[Image]

**Stage 1: Dataset Construction（数据集构建）**  
- 人工标注多样化越狱提示与图像。  

**Stage 2: Evaluation Protocol（评测协议）**  
- 衡量过滤与防御的有效性。  

**Stage 3: Attack Assessment Module（攻击评估模块）**  
- 统一评估多种 jailbreak 策略。  

**Summary（总结）**  
- 提供标准化越狱评测基准。  
- 支持 T2I 与 T2V 模型对比。  

**Analysis（分析）**  
- 有助于公平对比不同安全机制。  
- 基准需持续更新以避免过拟合。  

## 6. Efficient Input-level Backdoor Defense on Text-to-Image Synthesis via Neuron Activation Variation (NaviDet)
Zhai, S., Li, J., Liu, Y., Chen, H., Tian, Z., Qu, W., Shen, Q., Jia, R., Dong, Y., & Zhang, J. (2025). ICCV 2025 (CCF A).  
PDF: [ICCV](https://openaccess.thecvf.com/content/ICCV2025/papers/Zhai_Efficient_Input-level_Backdoor_Defense_on_Text-to-Image_Synthesis_via_Neuron_Activation_ICCV_2025_paper.pdf)

[Image]

**Stage 1: Early-step Activation Monitoring（早期激活监测）**  
- 观察扩散早期神经激活变化。  

**Stage 2: Trigger Detection（触发检测）**  
- 识别触发词导致的异常变化。  

**Stage 3: Mitigation（拦截/防御）**  
- 阻断或过滤疑似后门输入。  

**Summary（总结）**  
- 提出通用输入级后门检测框架。  
- 适用于多种后门目标与模型结构。  

**Analysis（分析）**  
- 适合部署在推理入口的安全网关。  
- 对自适应攻击的鲁棒性仍关键。  

## 7. Prompting4Debugging: Red-Teaming Text-to-Image Diffusion Models by Finding Problematic Prompts
Chin, Z.-Y., Jiang, C.-M., Huang, C.-C., Chen, P.-Y., & Chiu, W.-C. (2024). ICML 2024 (CCF A).  
PDF: [PMLR](https://proceedings.mlr.press/v235/chin24a/chin24a.pdf)

[Image]

**Stage 1: Target Model with Safety（带安全机制的 T2I 模型）**  
- 以部署中的安全机制为目标。  

**Stage 2: Problematic Prompt Search（问题提示搜索）**  
- 自动搜索可绕过安全的提示。  

**Stage 3: Debugging & Reporting（评估与报告）**  
- 量化安全机制脆弱性。  

**Summary（总结）**  
- 自动发现安全机制的盲点。  
- 证明“安全基准测试”易低估风险。  

**Analysis（分析）**  
- 适合作为持续红队工具。  
- 强调安全评估需覆盖多样提示。  

## 8. Multimodal Pragmatic Jailbreak on Text-to-image Models
Liu, T., Lai, Z., Wang, J., Zhang, G., Chen, S., Torr, P., Demberg, V., Tresp, V., & Gu, J. (2025). ACL 2025 (CCF A).  
PDF: [ACL](https://aclanthology.org/2025.acl-long.234.pdf)

[Image]

**Stage 1: Safe-in-isolation Pairing（安全单独组合）**  
- 构造图像与文字分别安全的组合。  

**Stage 2: Model & Filter Evaluation（模型与过滤评测）**  
- 测试多种 T2I 模型与过滤器。  

**Stage 3: Causal Analysis（成因分析）**  
- 追踪文本渲染与数据分布因素。  

**Summary（总结）**  
- 提出语用级 jailbreak，揭示跨模态风险。  
- 提供数据集与系统性评测。  

**Analysis（分析）**  
- 强调“跨模态联合理解”对安全的重要性。  
- 对商用系统过滤策略有直接警示。  

## 9. Latent Guard: a Safety Framework for Text-to-image Generation
Liu, R., Khakzar, A., Gu, J., Chen, Q., Torr, P., & Pizzati, F. (2024). ECCV 2024 (CCF B).  
PDF: [ECCV](https://www.ecva.net/papers/eccv_2024/papers_ECCV/papers/03726.pdf)

[Image]

**Stage 1: Data Generation（数据生成）**  
- 使用 LLM 构造有害/无害对比数据。  

**Stage 2: Latent Space Learning（潜空间学习）**  
- 训练文本编码器上方安全潜空间。  

**Stage 3: Prompt Screening（提示拦截）**  
- 在潜空间检测黑名单概念并阻断。  

**Summary（总结）**  
- 提出更灵活的安全检测框架。  
- 可扩展黑名单概念集合。  

**Analysis（分析）**  
- 提供高效文本侧安全门。  
- 需持续更新以应对语义绕过。  

## 10. Safeguard Text-to-Image Diffusion Models with Human Feedback Inversion (HFI)
Kim, S., Jung, S., Kim, B., Choi, M., Shin, J., & Lee, J. (2024). ECCV 2024 (CCF B).  
PDF: [ECCV](https://www.ecva.net/papers/eccv_2024/papers_ECCV/papers/08393.pdf)

[Image]

**Stage 1: Human Feedback Collection（反馈收集）**  
- 收集人类对生成图像的安全反馈。  

**Stage 2: Feedback Inversion（反馈反演）**  
- 将反馈压缩为文本 token。  

**Stage 3: Concept Mitigation（概念移除）**  
- 用 token 引导模型对齐与抑制。  

**Summary（总结）**  
- 将人类判断转化为可训练信号。  
- 提升安全移除的语义一致性。  

**Analysis（分析）**  
- 有助于缓解“概念定义不准”。  
- 反馈质量与覆盖范围决定效果。  

## 11. Geom-Erasing: Implicit Concept Removal of Diffusion Models
Liu, Z., Chen, K., Zhang, Y., Han, J., Hong, L., Xu, H., Li, Z., Yeung, D.-Y., & Kwok, J. (2024). ECCV 2024 (CCF B).  
PDF: [ECCV](https://www.ecva.net/papers/eccv_2024/papers_ECCV/papers/03200.pdf)

[Image]

**Stage 1: Geometry Extraction（几何信息提取）**  
- 由检测器获取隐式概念的空间位置。  

**Stage 2: Prompt Injection（几何提示注入）**  
- 将位置与概念信息转为提示条件。  

**Stage 3: Negative Sampling（负提示采样）**  
- 在生成中抑制隐式概念。  

**Summary（总结）**  
- 解决水印/文字等隐式概念移除问题。  
- 强调几何信息的重要性。  

**Analysis（分析）**  
- 对不可显式命名的概念很有效。  
- 依赖检测器的稳定性与准确度。  

## 12. R.A.C.E.: Robust Adversarial Concept Erasure for Secure Text-to-Image Diffusion Model
Kim, C., Min, K., & Yang, Y. (2024). ECCV 2024 (CCF B).  
PDF: [ECCV](https://www.ecva.net/papers/eccv_2024/papers_ECCV/papers/11117.pdf)

[Image]

**Stage 1: Adversarial Prompt Generation（对抗提示生成）**  
- 构造对抗文本嵌入。  

**Stage 2: Robust Erasure Training（鲁棒擦除训练）**  
- 通过对抗训练提升概念移除鲁棒性。  

**Stage 3: Robustness Evaluation（鲁棒评估）**  
- 对白盒/黑盒攻击进行验证。  

**Summary（总结）**  
- 显著降低攻击成功率。  
- 强调防御在对抗场景下的稳健性。  

**Analysis（分析）**  
- 直接面向 jailbreak 风险。  
- 训练成本与复杂度较高。  

## 13. Reliable and Efficient Concept Erasure (RECE)
Gong, C., Chen, K., Wei, Z., Chen, J., & Jiang, Y.-G. (2024). ECCV 2024 (CCF B).  
PDF: [ECCV](https://www.ecva.net/papers/eccv_2024/papers_ECCV/papers/06950.pdf)

[Image]

**Stage 1: Closed-form Editing（闭式模型编辑）**  
- 通过闭式解修改交叉注意力。  

**Stage 2: Embedding Derivation（嵌入推导）**  
- 推导并对齐新嵌入表示。  

**Stage 3: Iterative Loop（迭代擦除）**  
- 编辑与推导循环迭代，最小化副作用。  

**Summary（总结）**  
- 3 秒级别高效概念移除。  
- 对抗红队工具表现更稳健。  

**Analysis（分析）**  
- 适合快速补丁式安全修复。  
- 对结构假设较敏感。  

## 14. Receler: Reliable Concept Erasing of Text-to-Image Diffusion Models via Lightweight Erasers
Huang, C.-P., Chang, K.-P., Tsai, C.-T., Lai, Y.-H., Yang, F., & Wang, Y.-C. F. (2024). ECCV 2024 (CCF B).  
PDF: [ECCV](https://www.ecva.net/papers/eccv_2024/papers_ECCV/papers/05685.pdf)

[Image]

**Stage 1: Lightweight Eraser Training（轻量擦除器训练）**  
- 引入小型模块执行擦除。  

**Stage 2: Localized Regularization（局部正则）**  
- 保护非目标概念。  

**Stage 3: Adversarial Prompt Learning（对抗提示学习）**  
- 提升鲁棒性与泛化。  

**Summary（总结）**  
- 兼顾鲁棒性与局部性。  
- 支持多概念场景。  

**Analysis（分析）**  
- 轻量模块利于工程部署。  
- 仍需对复杂概念评估。  

## 15. Detect-and-Guide: Self-regulation of Diffusion Models for Safe Text-to-Image Generation via Guideline Token Optimization
Li, F., Zhang, M., Sun, Y., & Yang, M. (2025). CVPR 2025 (CCF A).  
PDF: [CVPR](https://openaccess.thecvf.com/content/CVPR2025/papers/Li_Detect-and-Guide_Self-regulation_of_Diffusion_Models_for_Safe_Text-to-Image_Generation_via_CVPR_2025_paper.pdf)

[Image]

**Stage 1: Harm Detection（有害概念检测）**  
- 优化 guideline tokens，基于注意力图定位有害概念。  

**Stage 2: Safety Guidance（安全引导）**  
- 自适应强度与区域进行抑制。  

**Stage 3: Safe Generation（安全生成）**  
- 无需微调即可生成安全图像。  

**Summary（总结）**  
- 提出自诊断与自调节安全框架。  
- 平衡安全性与文本对齐。  

**Analysis（分析）**  
- 推理期防御，部署灵活。  
- 依赖小规模标注数据。  

## 16. Implicit Bias Injection Attacks against Text-to-Image Diffusion Models
Huang, H., Jin, X., Miao, J., & Wu, Y. (2025). CVPR 2025 (CCF A).  
PDF: [CVPR](https://openaccess.thecvf.com/content/CVPR2025/papers/Huang_Implicit_Bias_Injection_Attacks_against_Text-to-Image_Diffusion_Models_CVPR_2025_paper.pdf)

[Image]

**Stage 1: Bias Direction Precompute（偏置方向计算）**  
- 在提示嵌入空间估计偏置向量。  

**Stage 2: Adaptive Adjustment（自适应调整）**  
- 根据输入动态调节偏置。  

**Stage 3: Plug-in Injection（插件式注入）**  
- 无需改用户输入即可植入隐式偏置。  

**Summary（总结）**  
- 提出“隐式偏置注入”攻击。  
- 具有隐蔽性与迁移性。  

**Analysis（分析）**  
- 风险来自模型内部操控。  
- 防御需参数审计与对抗检测。  

## 17. OpenSDI: Spotting Diffusion-Generated Images in the Open World
Wang, Y., Huang, Z., & Hong, X. (2025). CVPR 2025 (CCF A).  
PDF: [CVPR](https://openaccess.thecvf.com/content/CVPR2025/papers/Wang_OpenSDI_Spotting_Diffusion-Generated_Images_in_the_Open_World_CVPR_2025_paper.pdf)

[Image]

**Stage 1: OpenSDID Dataset（开放世界数据集）**  
- 构建多样化操控场景。  

**Stage 2: SPM Scheme（混合预训练模型）**  
- 协同多模型提升泛化。  

**Stage 3: MaskCLIP Detection（检测与定位）**  
- CLIP + MAE 对齐进行检测。  

**Summary（总结）**  
- 解决开放世界扩散图像检测难题。  
- 同时支持检测与定位。  

**Analysis（分析）**  
- 适合越狱后内容治理。  
- 需考虑成本与泛化。  

## 18. SleeperMark: Towards Robust Watermark against Fine-Tuning Text-to-image Diffusion Models
Wang, Z., Guo, J., Zhu, J., Li, Y., Huang, H., Chen, M., & Tu, Z. (2025). CVPR 2025 (CCF A).  
PDF: [CVPR](https://openaccess.thecvf.com/content/CVPR2025/papers/Wang_SleeperMark_Towards_Robust_Watermark_against_Fine-Tuning_Text-to-image_Diffusion_Models_CVPR_2025_paper.pdf)

[Image]

**Stage 1: Watermark Embedding（嵌入水印）**  
- 训练期植入水印。  

**Stage 2: Disentanglement（语义解耦）**  
- 将水印与语义分离以抗微调。  

**Stage 3: Verification（验证）**  
- 生成图像中检测水印。  

**Summary（总结）**  
- 针对模型被微调盗用的鲁棒水印。  
- 支持多种扩散模型。  

**Analysis（分析）**  
- 有利于模型 IP 保护。  
- 需对抗去水印攻击。  

## 19. Black-Box Forgery Attacks on Semantic Watermarks for Diffusion Models
Müller, A., Lukovnikov, D., Thietke, J., Fischer, A., & Quiring, E. (2025). CVPR 2025 (CCF A).  
PDF: [CVPR](https://openaccess.thecvf.com/content/CVPR2025/papers/Muller_Black-Box_Forgery_Attacks_on_Semantic_Watermarks_for_Diffusion_Models_CVPR_2025_paper.pdf)

[Image]

**Stage 1: Latent Manipulation（潜空间操控）**  
- 使用不相关模型逼近水印潜表示。  

**Stage 2: Forgery & Removal（伪造与移除）**  
- 伪造目标水印或移除水印。  

**Stage 3: Evaluation（安全性评估）**  
- 评估语义水印的脆弱性。  

**Summary（总结）**  
- 证明语义水印可被黑盒伪造。  
- 对水印安全提出挑战。  

**Analysis（分析）**  
- 需要更强鲁棒水印机制。  
- 建议多重验证策略。  

## 20. Silent Branding Attack: Trigger-free Data Poisoning Attack on Text-to-Image Diffusion Models
Jang, S., Choi, J. S., Jo, J., Lee, K., & Hwang, S. J. (2025). CVPR 2025 (CCF A).  
PDF: [CVPR](https://openaccess.thecvf.com/content/CVPR2025/papers/Jang_Silent_Branding_Attack_Trigger-free_Data_Poisoning_Attack_on_Text-to-Image_Diffusion_CVPR_2025_paper.pdf)

[Image]

**Stage 1: Data Poisoning（数据投毒）**  
- 在图片中隐蔽插入品牌标识。  

**Stage 2: Model Training（模型训练/微调）**  
- 使用投毒数据训练扩散模型。  

**Stage 3: Trigger-free Output（无触发输出）**  
- 模型在正常提示下生成含 logo 图像。  

**Summary（总结）**  
- 说明数据供应链是关键安全面。  
- 投毒具备高隐蔽性与高成功率。  

**Analysis（分析）**  
- 防御需数据审计与投毒检测。  
- 提醒开源数据共享风险。  

## 21. Nearly Zero-Cost Protection Against Mimicry by Personalized Diffusion Models
Ahn, N., Yoo, K., Ahn, W., Kim, D., & Nam, S.-H. (2025). CVPR 2025 (CCF A).  
PDF: [CVPR](https://openaccess.thecvf.com/content/CVPR2025/papers/Ahn_Nearly_Zero-Cost_Protection_Against_Mimicry_by_Personalized_Diffusion_Models_CVPR_2025_paper.pdf)

[Image]

**Stage 1: Perturbation Pre-training（扰动预训练）**  
- 降低保护策略推理开销。  

**Stage 2: Mixture-of-Perturbations（多扰动混合）**  
- 自适应匹配输入特性。  

**Stage 3: Targeted Protection（定向保护）**  
- 抑制个性化模型模仿。  

**Summary（总结）**  
- 提出低成本防模仿保护方案。  
- 保持隐蔽性与质量。  

**Analysis（分析）**  
- 适合实际部署的前置保护。  
- 需评估对创作自由的影响。  

## 22. ConceptGuard: Continual Personalized Text-to-Image Generation with Forgetting and Confusion Mitigation
Guo, Z., & Jin, T. (2025). CVPR 2025 (CCF A).  
PDF: [CVPR](https://openaccess.thecvf.com/content/CVPR2025/papers/Guo_ConceptGuard_Continual_Personalized_Text-to-Image_Generation_with_Forgetting_and_Confusion_Mitigation_CVPR_2025_paper.pdf)

[Image]

**Stage 1: Shift Embedding & Binding Prompts（嵌入偏移与绑定）**  
- 加强概念可分性。  

**Stage 2: Memory Preservation（记忆正则）**  
- 防止旧概念遗忘。  

**Stage 3: Priority Queue（优先队列更新）**  
- 管理概念重要性与顺序。  

**Summary（总结）**  
- 解决持续个性化的遗忘/混淆问题。  
- 提升多概念长期稳定性。  

**Analysis（分析）**  
- 对安全概念持续保留有价值。  
- 规模化场景仍需验证。  

## 23. ACE: Anti-Editing Concept Erasure in Text-to-Image Models
Wang, Z., Wei, Y., Li, F., Pei, R., Xu, H., & Zuo, W. (2025). CVPR 2025 (CCF A).  
PDF: [CVPR](https://openaccess.thecvf.com/content/CVPR2025/papers/Wang_ACE_Anti-Editing_Concept_Erasure_in_Text-to-Image_Models_CVPR_2025_paper.pdf)

[Image]

**Stage 1: Dual-path Guidance（双路径擦除指导）**  
- 同时作用于条件与无条件预测。  

**Stage 2: Stochastic Correction（随机校正）**  
- 降低非目标概念侵蚀。  

**Stage 3: Generation + Editing（生成与编辑）**  
- 同时保证生成和编辑场景安全。  

**Summary（总结）**  
- 解决编辑场景的擦除失效问题。  
- 适用于版权与敏感概念移除。  

**Analysis（分析）**  
- 更贴合实际编辑工作流。  
- 对编辑操作依赖较强。  

## 24. Localized Concept Erasure for Text-to-Image Diffusion Models Using Training-Free Gated Low-Rank Adaptation (GLoCE)
Lee, B. H., Lim, S., & Chun, S. Y. (2025). CVPR 2025 (CCF A).  
PDF: [CVPR](https://openaccess.thecvf.com/content/CVPR2025/papers/Lee_Localized_Concept_Erasure_for_Text-to-Image_Diffusion_Models_Using_Training-Free_Gated_CVPR_2025_paper.pdf)

[Image]

**Stage 1: Gate Injection（门控模块注入）**  
- 低秩模块插入扩散模型。  

**Stage 2: Training-free Gate Selection（无训练门控）**  
- 通过少量生成步骤确定门控。  

**Stage 3: Local Erasure（局部擦除）**  
- 仅擦除目标区域，保留其他区域。  

**Summary（总结）**  
- 训练自由、局部概念擦除。  
- 提升保真度与鲁棒性。  

**Analysis（分析）**  
- 对区域敏感概念更友好。  
- 依赖门控定位准确性。  

## 25. Fine-Grained Erasure in Text-to-Image Diffusion-based Foundation Models (FADE)
Thakral, K., Glaser, T., Hassner, T., Vatsa, M., & Singh, R. (2025). CVPR 2025 (CCF A).  
PDF: [CVPR](https://openaccess.thecvf.com/content/CVPR2025/papers/Thakral_Fine-Grained_Erasure_in_Text-to-Image_Diffusion-based_Foundation_Models_CVPR_2025_paper.pdf)

[Image]

**Stage 1: Concept Neighborhood（概念邻域）**  
- 识别邻接相关概念集合。  

**Stage 2: Mesh Modules（网格模块）**  
- 结合多损失结构化训练。  

**Stage 3: Adjacency-aware Erasure（邻接感知擦除）**  
- 在保留相关概念的同时移除目标。  

**Summary（总结）**  
- 面向邻接概念的精细擦除。  
- 显著提升保留性能。  

**Analysis（分析）**  
- 缓解“过度抹除”问题。  
- 概念邻域构建是关键。  

## 26. Six-CD: Benchmarking Concept Removals for Text-to-image Diffusion Models
Ren, J., Chen, K., Cui, Y., Zeng, S., Liu, H., Xing, Y., Tang, J., & Lyu, L. (2025). CVPR 2025 (CCF A).  
PDF: [CVPR](https://openaccess.thecvf.com/content/CVPR2025/papers/Ren_Six-CD_Benchmarking_Concept_Removals_for_Text-to-image_Diffusion_Models_CVPR_2025_paper.pdf)

[Image]

**Stage 1: Benchmark Design（基准设计）**  
- 提出概念移除数据集与指标。  

**Stage 2: Method Evaluation（方法评测）**  
- 对多种移除方法系统测试。  

**Stage 3: Failure Analysis（失效分析）**  
- 分析提示有效性与良性生成能力。  

**Summary（总结）**  
- 建立概念移除评测标准。  
- 揭示现有方法不足。  

**Analysis（分析）**  
- 对比较研究非常有价值。  
- 需持续更新场景覆盖。  

## 27. MACE: Mass Concept Erasure in Diffusion Models
Lu, S., Wang, Z., Li, L., Liu, Y., & Kong, A. W.-K. (2024). CVPR 2024 (CCF A).  
PDF: [CVPR](https://openaccess.thecvf.com/content/CVPR2024/papers/Lu_MACE_Mass_Concept_Erasure_in_Diffusion_Models_CVPR_2024_paper.pdf)

[Image]

**Stage 1: Cross-attention Refinement（交叉注意力修正）**  
- 闭式修正目标概念表示。  

**Stage 2: LoRA Finetuning（LoRA 微调）**  
- 进行大规模概念擦除。  

**Stage 3: Multi-LoRA Integration（多 LoRA 集成）**  
- 多概念互不干扰融合。  

**Summary（总结）**  
- 支持 100 概念级批量移除。  
- 在多任务场景下表现突出。  

**Analysis（分析）**  
- 适合安全策略的大规模清理。  
- 需要评估对模型多样性的影响。  

## 28. Gaussian Shading: Provable Performance-Lossless Image Watermarking for Diffusion Models
Yang, Z., Zeng, K., Chen, K., Fang, H., Zhang, W., & Yu, N. (2024). CVPR 2024 (CCF A).  
PDF: [CVPR](https://openaccess.thecvf.com/content/CVPR2024/papers/Yang_Gaussian_Shading_Provable_Performance-Lossless_Image_Watermarking_for_Diffusion_Models_CVPR_2024_paper.pdf)

[Image]

**Stage 1: Latent Watermark Mapping（潜空间水印映射）**  
- 将水印映射为高斯分布表示。  

**Stage 2: Plug-and-play Embedding（无损嵌入）**  
- 不改模型参数即可嵌入。  

**Stage 3: Extraction via Inversion（反演提取）**  
- 通过 DDIM 反演获取水印。  

**Summary（总结）**  
- 无损、无需训练的水印方案。  
- 具备较强抗压缩/抗擦除能力。  

**Analysis（分析）**  
- 适合版权追踪与溯源。  
- 需结合防伪造策略。  

## 29. Safe Latent Diffusion: Mitigating Inappropriate Degeneration in Diffusion Models
Schramowski, P., Brack, M., Deiseroth, B., & Kersting, K. (2023). CVPR 2023 (CCF A).  
PDF: [CVPR](https://openaccess.thecvf.com/content/CVPR2023/papers/Schramowski_Safe_Latent_Diffusion_Mitigating_Inappropriate_Degeneration_in_Diffusion_Models_CVPR_2023_paper.pdf)

[Image]

**Stage 1: I2P Benchmark（不当提示基准）**  
- 构建不当提示数据集。  

**Stage 2: Safe Sampling（安全采样）**  
- 在采样阶段抑制不当内容。  

**Stage 3: Quality Preservation（质量保持）**  
- 保持文本对齐与图像质量。  

**Summary（总结）**  
- 推理期安全机制无需训练。  
- 建立 I2P 基准推动评测。  

**Analysis（分析）**  
- 易于部署，但对隐式风险可能不足。  
- 可与概念擦除方法互补。  

## 30. Tree-Rings Watermarks: Invisible Fingerprints for Diffusion Images
Wen, Y., Kirchenbauer, J., Geiping, J., & Goldstein, T. (2023). NeurIPS 2023 (CCF A).  
PDF: [arXiv](https://arxiv.org/pdf/2305.20030)

[Image]

**Stage 1: Noise Fingerprint（噪声指纹）**  
- 将水印嵌入初始噪声。  

**Stage 2: Fourier-space Robustness（傅里叶稳健性）**  
- 设计对变换稳健的噪声结构。  

**Stage 3: Inversion Detection（反演检测）**  
- 通过扩散反演提取水印。  

**Summary（总结）**  
- 生成过程内生水印，隐蔽性强。  
- 对常见图像变换更鲁棒。  

**Analysis（分析）**  
- 对溯源与治理有直接价值。  
- 需结合防伪造策略。

## 31. MMA-Diffusion: MultiModal Attack on Diffusion Models
Yijun Yang, Ruiyuan Gao, Xiaosen Wang, Tsung-Yi Ho, Nan Xu, & Qiang Xu. (2024). CVPR 2024 (CCF A).  
PDF: [CVPR](https://openaccess.thecvf.com/content/CVPR2024/papers/Yang_MMA-Diffusion_MultiModal_Attack_on_Diffusion_Models_CVPR_2024_paper.pdf)

[Image]

**Stage 1: Multimodal Seed Construction（多模态攻击种子构造）**  
- 设计含攻击意图的文本与图像组合输入。  
- 目标同时影响提示过滤与安全检测模块。  

**Stage 2: Multimodal Adversarial Optimization（多模态对抗优化）**  
- 联合优化文本提示与图像扰动以绕过防护。  
- 在不同防护组合下搜索高成功率路径。  

**Stage 3: Black-box Query & Evaluation（黑盒查询与评估）**  
- 在开源与商用系统上测试攻击成功率。  
- 衡量语义一致性与图像质量。  

**Summary（总结）**  
- 提出 MMA-Diffusion，将文本与图像协同用于越狱攻击。  
- 证明多模态攻击能显著削弱现有安全机制。  

**Analysis（分析）**  
- 说明单一文本过滤无法覆盖多模态攻击面。  
- 防御应考虑跨模态联合检测与约束。  

## 32. Unsafe Diffusion: On the Generation of Unsafe Images and Hateful Memes From Text-To-Image Models
Yiting Qu, Xinyue Shen, Xinlei He, Michael Backes, Savvas Zannettou, & Yang Zhang. (2023). ACM CCS 2023 (CCF A).  
PDF: [arXiv](https://arxiv.org/pdf/2305.13873.pdf)

[Image]

**Stage 1: Unsafe Taxonomy & Prompt Sets（不安全类别与提示集合）**  
- 构建五类不安全内容类型与多源提示集。  
- 覆盖性、暴力、仇恨、政治等场景。  

**Stage 2: Multi-Model Generation & Measurement（多模型生成与测量）**  
- 用多种 T2I 模型生成图像并统计不安全比例。  
- 比较不同模型的风险水平。  

**Stage 3: Hateful Meme Variant Generation（仇恨梗图变体生成）**  
- 结合 DreamBooth、Textual Inversion、SDEdit 生成变体。  
- 衡量对特定个体/群体攻击的可行性。  

**Summary（总结）**  
- 系统量化 T2I 模型产生不安全内容的风险。  
- 证明编辑方法可用于生成高相似仇恨梗图。  

**Analysis（分析）**  
- 为越狱风险评估提供基线与量化框架。  
- 提醒防御需覆盖“编辑型”攻击链路。  

## 33. T2IShield: Defending Against Backdoors on Text-to-Image Diffusion Models
Zhongqi Wang, Jie Zhang, Shiguang Shan, & Xilin Chen. (2024). ECCV 2024 (CCF B).  
PDF: [arXiv](https://arxiv.org/pdf/2407.04215)

[Image]

**Stage 1: Assimilation Phenomenon Discovery（同化现象发现）**  
- 观察后门触发导致的交叉注意力异常。  
- 作为检测与定位的关键线索。  

**Stage 2: Backdoor Detection（后门检测）**  
- 提出 Frobenius Norm Threshold Truncation 与 CDA。  
- 低成本检测疑似后门样本。  

**Stage 3: Trigger Localization & Mitigation（触发定位与缓解）**  
- 二分搜索定位触发区域。  
- 结合概念编辑方法削弱后门效果。  

**Summary（总结）**  
- 首个面向 T2I 后门的系统性防御框架。  
- 同时覆盖检测、定位与缓解流程。  

**Analysis（分析）**  
- 适合部署在模型上线前的安全审计。  
- 对新型后门触发仍需持续更新。  

## 34. Training-Free Safe Text Embedding Guidance for Text-to-Image Diffusion Models
Byeonghu Na, Mina Kang, Jiseok Kwak, Minsang Park, Jiwoo Shin, SeJoon Jun, Gayoung Lee, Jin-Hwa Kim, & Il-chul Moon. (2025). NeurIPS 2025 (CCF A).  
PDF: [arXiv](https://arxiv.org/pdf/2510.24012)

[Image]

**Stage 1: Safety Function Design（安全函数设计）**  
- 定义评估潜在输出安全性的函数。  
- 作为采样过程的约束信号。  

**Stage 2: Text Embedding Guidance（文本嵌入引导）**  
- 在采样过程中对文本嵌入进行安全引导更新。  
- 无需额外训练或微调。  

**Stage 3: Safe Sampling（安全采样）**  
- 生成更安全的图像同时保持语义一致性。  
- 在多类风险场景上评估效果。  

**Summary（总结）**  
- 提出 STG，在推理阶段直接约束文本嵌入。  
- 兼顾安全性与生成质量。  

**Analysis（分析）**  
- 适合快速部署在已有模型上。  
- 对安全函数设计与覆盖范围依赖较强。  

## 35. Erasing Undesirable Influence in Diffusion Models (EraseDiff)
Jing Wu, Trung Le, Munawar Hayat, & Mehrtash Harandi. (2025). CVPR 2025 (CCF A).  
PDF: [CVPR](https://openaccess.thecvf.com/content/CVPR2025/papers/Wu_Erasing_Undesirable_Influence_in_Diffusion_Models_CVPR_2025_paper.pdf)

[Image]

**Stage 1: Forget Set Definition（遗忘目标定义）**  
- 指定需被移除的概念/数据影响。  
- 兼顾保留数据的生成效用。  

**Stage 2: Constrained Optimization（约束优化）**  
- 以价值函数构建约束优化问题。  
- 通过偏离真实去噪轨迹实现擦除。  

**Stage 3: Utility-Preserving Update（效用保留更新）**  
- 控制擦除强度，减少性能损失。  
- 与多种遗忘方法对比评估。  

**Summary（总结）**  
- 提出 EraseDiff 平衡遗忘与效用保留。  
- 针对 NSFW 等不良概念实现可控擦除。  

**Analysis（分析）**  
- 适合合规场景的“可遗忘”修复。  
- 擦除范围与生成质量仍需权衡。  

## 36. Self-Discovering Interpretable Diffusion Latent Directions for Responsible Text-to-Image Generation
Hang Li, Chengzhi Shen, Philip H. S. Torr, Volker Tresp, & Jindong Gu. (2024). CVPR 2024 (CCF A).  
PDF: [CVPR](https://openaccess.thecvf.com/content/CVPR2024/papers/Li_Self-Discovering_Interpretable_Diffusion_Latent_Directions_for_Responsible_Text-to-Image_Generation_CVPR_2024_paper.pdf)

[Image]

**Stage 1: Self-Supervised Direction Discovery（自监督方向发现）**  
- 在潜空间自动发现与目标概念相关方向。  
- 支持不良或偏见概念的定位。  

**Stage 2: Interpretable Direction Modeling（可解释方向建模）**  
- 将方向视为可控语义因素。  
- 连接文本条件与生成行为。  

**Stage 3: Responsible Generation Control（责任生成控制）**  
- 沿方向施加约束缓解不当生成。  
- 在公平与安全场景评测。  

**Summary（总结）**  
- 提供可解释的潜空间安全控制途径。  
- 兼顾生成质量与安全性。  

**Analysis（分析）**  
- 对解释性与安全治理具有直接价值。  
- 需依赖方向发现的稳定性。  

## 37. Harnessing Frequency Spectrum Insights for Image Copyright Protection Against Diffusion Models (CoprGuard)
Zhenguang Liu, Chao Shuai, Shaojing Fan, Ziping Dong, Jinwu Hu, Zhongjie Ba, & Kui Ren. (2025). CVPR 2025 (CCF A).  
PDF: [CVPR](https://openaccess.thecvf.com/content/CVPR2025/papers/Liu_Harnessing_Frequency_Spectrum_Insights_for_Image_Copyright_Protection_Against_Diffusion_CVPR_2025_paper.pdf)

[Image]

**Stage 1: Spectral Watermark Embedding（频域水印嵌入）**  
- 在训练图像的频域植入水印信号。  
- 对不同数据来源具备可识别性。  

**Stage 2: Model Training/Fine-tuning（模型训练/微调）**  
- 水印样本占比低时仍保留可检测特征。  
- 适配多类扩散与 T2I 模型。  

**Stage 3: Unauthorized Use Detection（侵权使用检测）**  
- 通过生成图像的频谱统计识别侵权训练。  
- 模型无关、黑盒可用。  

**Summary（总结）**  
- 提出 CoprGuard，通过频谱统计保护版权。  
- 在小比例水印数据下仍可检测。  

**Analysis（分析）**  
- 适合真实数据混合场景的版权追踪。  
- 需要关注对抗性去水印风险。  

## 38. Your Text Encoder Can Be An Object-Level Watermarking Controller
Naresh Kumar Devulapally, Mingzhen Huang, Vishal Asnani, Shruti Agarwal, Siwei Lyu, & Vishnu Suresh Lokhande. (2025). ICCV 2025 (CCF A).  
PDF: [ICCV](https://openaccess.thecvf.com/content/ICCV2025/papers/Devulapally_Your_Text_Encoder_Can_Be_An_Object-Level_Watermarking_Controller_ICCV_2025_paper.pdf)

[Image]

**Stage 1: Token Embedding Tuning（文本嵌入调优）**  
- 仅微调文本 token 嵌入作为水印载体。  
- 减少对模型参数改动。  

**Stage 2: Object-Level Watermarking（对象级水印）**  
- 将水印绑定到特定对象或区域。  
- 支持跨 LDM 的即插即用。  

**Stage 3: Robust Extraction（鲁棒提取）**  
- 早期嵌入提升抗攻击能力。  
- 达到高比特准确率。  

**Summary（总结）**  
- 提供对象级可控水印方案。  
- 参数开销极低，适合工程部署。  

**Analysis（分析）**  
- 适用于溯源与版权保护。  
- 需评估去水印与微调攻击。  

## 39. PlugMark: A Plug-in Zero-Watermarking Framework for Diffusion Models
Pengzhen Chen, Yanwei Liu, Xiaoyan Gu, Enci Liu, Zhuoyi Shang, Xiangyang Ji, & Wu Liu. (2025). ICCV 2025 (CCF A).  
PDF: [ICCV](https://openaccess.thecvf.com/content/ICCV2025/papers/Chen_PlugMark_A_Plug-in_Zero-Watermarking_Framework_for_Diffusion_Models_ICCV_2025_paper.pdf)

[Image]

**Stage 1: Knowledge Extraction（知识提取）**  
- 插件式提取器从模型中提取知识表示。  
- 输出分类结果作为边界基础。  

**Stage 2: Boundary-Based Zero-Watermark（边界零水印）**  
- 将决策边界表示为零失真水印。  
- 不改动原模型生成质量。  

**Stage 3: Verification & Robustness（验证与鲁棒性）**  
- 对微调与后处理保持稳定识别。  
- 区分原模型与衍生模型。  

**Summary（总结）**  
- 提出零失真水印框架保护扩散模型 IP。  
- 训练成本低，部署友好。  

**Analysis（分析）**  
- 适合开源模型溯源与鉴权。  
- 需要防御模型级别的对抗伪造。  

## 40. Who Controls the Authorization? Invertible Networks for Copyright Protection in Text-to-Image Synthesis
Baoyue Hu, Yang Wei, Junhao Xiao, Wendong Huang, Xiuli Bi, & Bin Xiao. (2025). ICCV 2025 (CCF A).  
PDF: [ICCV](https://openaccess.thecvf.com/content/ICCV2025/papers/Hu_Who_Controls_the_Authorization_Invertible_Networks_for_Copyright_Protection_in_ICCV_2025_paper.pdf)

[Image]

**Stage 1: Invertible Embedding（可逆嵌入）**  
- 通过可逆耦合模块嵌入版权水印。  
- 支持近似无损恢复。  

**Stage 2: Authorization & Traceability（授权与可追踪）**  
- 对合法个性化提供授权通道。  
- 未授权生成将产生显著伪迹。  

**Stage 3: Robustness Evaluation（鲁棒性评估）**  
- 对净化与提示修改保持可追踪性。  
- 兼顾隐私保护与可验证性。  

**Summary（总结）**  
- 提供可追踪且可授权的版权保护框架。  
- 支持合法个性化与非法滥用区分。  

**Analysis（分析）**  
- 适用于个性化模型的版权治理。  
- 实际部署需关注用户体验影响。  

## 41. USD: NSFW Content Detection for Text-to-Image Models via Scene Graph
Yuyang Zhang, Kangjie Chen, Xudong Jiang, Jiahui Wen, Yihui Jin, Ziyou Liang, Yihao Huang, Run Wang, & Lina Wang. (2025). USENIX Security 2025 (CCF A).  
PDF: [USENIX](https://www.usenix.org/system/files/usenixsecurity25-zhang-yuyang.pdf)

[Image]

**Stage 1: Scene Graph Generation（场景图生成）**  
- 从生成图像中抽取实体与关系。  
- 将抽象有害概念落到关系层面。  

**Stage 2: Unsafe Relation Classification（不安全关系判定）**  
- 基于实体关系识别不安全场景。  
- 兼顾语义与结构信息。  

**Stage 3: Localization & Filtering（定位与过滤）**  
- 定位不安全区域并进行过滤。  
- 保持剩余内容一致性。  

**Summary（总结）**  
- 提出场景图驱动的 NSFW 检测框架。  
- 提升不安全场景识别与定位能力。  

**Analysis（分析）**  
- 可作为 T2I 生成后的安全网关。  
- 依赖场景图生成质量与覆盖。  

## 42. Exposing the Guardrails: Reverse-Engineering and Jailbreaking Safety Filters in DALL·E Text-to-Image Pipelines
Corban Villa, Shujaat Mirza, & Christina Pöpper. (2025). USENIX Security 2025 (CCF A).  
PDF: [USENIX](https://www.usenix.org/system/files/usenixsecurity25-villa.pdf)

[Image]

**Stage 1: Timing Side-Channel Analysis（时间侧信道分析）**  
- 通过响应时间差推断安全机制结构。  
- 识别多级过滤链路。  

**Stage 2: Guardrail Reverse Engineering（防护机制逆向）**  
- 对比 DALL·E 2/3 的过滤方式差异。  
- 发现 LLM 改写与 CLIP 对齐不一致。  

**Stage 3: Jailbreak Construction（越狱构造）**  
- 设计否定式与低资源语言攻击。  
- 提出多类防护改进方向。  

**Summary（总结）**  
- 首次系统性逆向 DALL·E 安全管线。  
- 给出可复现的越狱攻击范式。  

**Analysis（分析）**  
- 暴露多组件安全边界不一致问题。  
- 推动更统一的安全管线设计。  

## 43. On the Proactive Generation of Unsafe Images From Text-To-Image Models Using Benign Prompts
Yixin Wu, Ning Yu, Michael Backes, Yun Shen, & Yang Zhang. (2025). USENIX Security 2025 (CCF A).  
PDF: [USENIX](https://www.usenix.org/system/files/usenixsecurity25-wu-yixin-generation.pdf)

[Image]

**Stage 1: Benign Prompt Targeting（良性提示攻击目标）**  
- 选择常见 benign prompts 作为触发入口。  
- 目标是输出不安全内容。  

**Stage 2: Poisoning Feasibility Study（投毒可行性分析）**  
- 验证投毒可诱导不安全输出。  
- 发现副作用会波及非目标提示。  

**Stage 3: Stealthy Poisoning Design（隐蔽投毒设计）**  
- 基于概念相似性分析削弱副作用。  
- 平衡隐蔽性与攻击效果。  

**Summary（总结）**  
- 将“良性提示”转化为攻击入口。  
- 揭示投毒攻击的隐蔽性风险。  

**Analysis（分析）**  
- 对供应链安全与模型审核提出警示。  
- 防御需结合数据审计与分布检测。  

## 44. Backdooring Bias (B^2) into Stable Diffusion Models
Ali Naseh, Jaechul Roh, Eugene Bagdasarian, & Amir Houmansadr. (2025). USENIX Security 2025 (CCF A).  
PDF: [USENIX](https://www.usenix.org/system/files/usenixsecurity25-naseh.pdf)

[Image]

**Stage 1: Trigger Selection（触发器选择）**  
- 选取自然文本触发短语。  
- 目标是引入可控偏置。  

**Stage 2: Low-cost Backdooring（低成本后门注入）**  
- 通过少量恶意样本注入偏置。  
- 保持常规生成效用。  

**Stage 3: Bias Activation & Detection（偏置激活与可检测性）**  
- 评估触发后偏置输出与对齐程度。  
- 分析检测难度与成本。  

**Summary（总结）**  
- 展示低成本可扩散的偏置后门攻击。  
- 强调偏置检测与防御的困难性。  

**Analysis（分析）**  
- 对内容安全与社会风险提出挑战。  
- 需更强的训练数据与模型审计。  

## 45. EvilEdit: Backdooring Text-to-Image Diffusion Models in One Second
Hao Wang, Shangwei Guo, Jialing He, Kangjie Chen, Shudong Zhang, Tianwei Zhang, & Xiang Tao. (2024). ACM Multimedia 2024 (CCF A).  
PDF: [ACM](https://dl.acm.org/doi/pdf/10.1145/3664647.3680689)

[Image]

**Stage 1: Editing Objective Setup（编辑目标设置）**  
- 定义触发词与目标输出。  
- 约束生成质量不下降。  

**Stage 2: One-Second Model Editing（秒级模型编辑）**  
- 通过快速权重编辑植入后门。  
- 无需完整再训练。  

**Stage 3: Attack Validation（攻击验证）**  
- 在 Stable Diffusion 上验证触发效果。  
- 对比传统投毒的时间成本。  

**Summary（总结）**  
- 提出秒级后门注入方法。  
- 显著降低攻击门槛。  

**Analysis（分析）**  
- 强调模型编辑工具的安全风险。  
- 防御需监控权重修改与分发链路。  

## 46. Text-to-Image Diffusion Models can be Easily Backdoored through Multimodal Data Poisoning (BadT2I)
Shengfang Zhai, Yinpeng Dong, Qingni Shen, Shi Pu, Yuejian Fang, & Hang Su. (2023). ACM Multimedia 2023 (CCF A).  
PDF: [arXiv](https://arxiv.org/pdf/2305.04175)

[Image]

**Stage 1: Multimodal Poisoning Design（多模态投毒设计）**  
- 构造文本与图像联合触发样本。  
- 适配不同语义层级。  

**Stage 2: Backdoor Injection（后门注入）**  
- 在像素、对象、风格三层注入后门。  
- 保持正常提示下性能。  

**Stage 3: Robustness & Persistence（鲁棒性与持久性）**  
- 测试触发词变体与持续训练影响。  
- 验证后门可长期保留。  

**Summary（总结）**  
- 系统化展示 T2I 扩散模型的投毒脆弱性。  
- 提供多层级后门范式。  

**Analysis（分析）**  
- 对训练数据治理提出高要求。  
- 防御需结合训练过程的异常检测。  

## 47. Rickrolling the Artist: Injecting Backdoors into Text Encoders for Text-to-Image Synthesis
Lukas Struppek, Dominik Hintersdorf, & Kristian Kersting. (2023). ICCV 2023 (CCF A).  
PDF: [ICCV](https://openaccess.thecvf.com/content/ICCV2023/papers/Struppek_Rickrolling_the_Artist_Injecting_Backdoors_into_Text_Encoders_for_Text-to-Image_ICCV_2023_paper.pdf)

[Image]

**Stage 1: Trigger Token Selection（触发 Token 设计）**  
- 使用同形字符/表情作为隐蔽触发。  
- 兼容常规提示结构。  

**Stage 2: Encoder Backdoor Injection（编码器后门注入）**  
- 通过教师-学生训练注入后门。  
- 维持正常提示生成质量。  

**Stage 3: Targeted Generation（目标生成）**  
- 触发后生成指定属性/目标图像。  
- 评估攻击有效性与隐蔽性。  

**Summary（总结）**  
- 指出文本编码器是关键后门入口。  
- 触发形式隐蔽且易传播。  

**Analysis（分析）**  
- 模型下载与共享链路风险高。  
- 防御需校验编码器完整性。  

## 48. Membership Inference on Text-to-Image Diffusion Models via Conditional Likelihood Discrepancy
Shengfang Zhai, Huanran Chen, Yinpeng Dong, Jiajun Li, Qingni Shen, Yansong Gao, Hang Su, & Yang Liu. (2024). NeurIPS 2024 (CCF A).  
PDF: [NeurIPS](https://proceedings.neurips.cc/paper_files/paper/2024/file/874411a224a1934b80d499068384808b-Paper-Conference.pdf)

[Image]

**Stage 1: Conditional Overfitting Discovery（条件过拟合发现）**  
- 识别 T2I 模型更易过拟合条件分布。  
- 作为隐私泄露信号。  

**Stage 2: CLiD Indicator Derivation（指标推导）**  
- 构建 Conditional Likelihood Discrepancy 作为判别量。  
- 降低随机性带来的误差。  

**Stage 3: Membership Inference（成员推断）**  
- 以 CLiD 判定样本是否在训练集。  
- 在多数据规模上验证性能。  

**Summary（总结）**  
- 提供面向 T2I 的隐私审计方法。  
- 明确训练数据泄露风险。  

**Analysis（分析）**  
- 对合规与版权审计有直接价值。  
- 需要配套隐私保护训练策略。  

## 49. Cross-Modal Prompt Inversion: Unifying Threats to Text and Image Generative AI Models
Dayong Ye, Tianqing Zhu, Feng He, Bo Liu, Minhui Xue, & Wanlei Zhou. (2025). USENIX Security 2025 (CCF A).  
PDF: [USENIX](https://www.usenix.org/system/files/usenixsecurity25-ye-inversion.pdf)

[Image]

**Stage 1: Inversion Model Training（反演模型训练）**  
- 从模型输出反推提示词近似。  
- 跨文本与图像模型统一框架。  

**Stage 2: RL Refinement（强化学习优化）**  
- 使用 RL 提升提示还原精度。  
- 减少与真实提示的偏差。  

**Stage 3: Cross-Modal Evaluation（跨模态评估）**  
- 在文本与图像生成模型上验证。  
- 扩展到更多生成模态。  

**Summary（总结）**  
- 提出统一的提示词窃取框架。  
- 覆盖文本与图像生成模型。  

**Analysis（分析）**  
- 对商用模型提示词安全提出挑战。  
- 需考虑提示保护与输出去敏。  

## 50. Bridging the Gap in Vision Language Models in Identifying Unsafe Concepts Across Modalities
Yiting Qu, Michael Backes, & Yang Zhang. (2025). USENIX Security 2025 (CCF A).  
PDF: [USENIX](https://www.usenix.org/system/files/usenixsecurity25-qu-yiting.pdf)

[Image]

**Stage 1: UnsafeConcepts Dataset（不安全概念数据集）**  
- 构建跨模态不安全概念与图像集合。  
- 覆盖多类风险场景。  

**Stage 2: VLM Evaluation（VLM 评测）**  
- 测试感知能力与伦理对齐能力。  
- 发现“模态差距”现象。  

**Stage 3: Alignment via PPO（PPO 对齐）**  
- 使用 PPO 强化对图像不安全概念的识别。  
- 保持通用能力不下降。  

**Summary（总结）**  
- 系统揭示 VLM 在不同模态下的安全差距。  
- 提供 RL 对齐方法缩小差距。  

**Analysis（分析）**  
- 可作为 T2I 安全过滤器的评测基础。  
- 对安全对齐与数据建设均有指导价值。

## 51. Mind the Trojan Horse: Image Prompt Adapter Enabling Scalable and Deceptive Jailbreaking
Chen, J., Dong, J., & Xie, X. (2025). CVPR 2025 (CCF A).  
PDF: [CVPR](https://openaccess.thecvf.com/content/CVPR2025/papers/Chen_Mind_the_Trojan_Horse_Image_Prompt_Adapter_Enabling_Scalable_and_CVPR_2025_paper.pdf)

[Image]

**Stage 1: IP-Adapter Threat Model（IP-Adapter 威胁模型）**  
- 目标是集成 IP-Adapter 的 T2I 服务（IGS）。  
- 关键依赖是开放的图像编码器。  

**Stage 2: Image-space AEs（图像空间对抗样本）**  
- 构造几乎不可见的图像扰动。  
- 让编码器输出对齐到不安全语义。  

**Stage 3: Hijacking & Defense（劫持与防御）**  
- 通过上传 AEs 劫持大规模 benign 用户的生成结果。  
- 评估并讨论可行的防御组合。  

**Summary（总结）**  
- 提出基于 IP-Adapter 的“劫持式”越狱。  
- 展示黑盒 IGS 场景下的规模化攻击风险。  

**Analysis（分析）**  
- 攻击面来自通用图像编码器与图像输入通道。  
- 防御需从编码器鲁棒性和输入过滤两端协同。  

## 52. Perception-guided Jailbreak against Text-to-Image Models
Huang, Y., Liang, L., Li, T., Jia, X., Wang, R., Miao, W., Pu, G., & Liu, Y. (2025). AAAI 2025 (CCF A).  
PDF: [arXiv](https://arxiv.org/pdf/2408.10848)

[Image]

**Stage 1: Perception Gap Mining（感知差异挖掘）**  
- 发现“语义不同但人类感知接近”的词组。  
- 由 LLM 生成安全替代短语。  

**Stage 2: Prompt Substitution（替换式越狱）**  
- 用安全替代短语替换不安全词。  
- 保持视觉感知接近目标。  

**Stage 3: Black-box Evaluation（黑盒评测）**  
- 在开源与商业服务上测试成功率。  
- 统计越狱与语义保持效果。  

**Summary（总结）**  
- PGJ 是显式黑盒越狱方法。  
- 利用“感知一致、语义不一致”实现绕过。  

**Analysis（分析）**  
- 对基于关键词/语义匹配的过滤器威胁大。  
- 需要引入感知级的安全检测。  

## 53. Prompt Stealing Attacks Against Text-to-Image Generation Models
Shen, X., Qu, Y., Backes, M., & Zhang, Y. (2024). USENIX Security 2024 (CCF A).  
PDF: [USENIX](https://www.usenix.org/system/files/usenixsecurity24-shen-xinyue.pdf)

[Image]

**Stage 1: Image Observation（图像观察）**  
- 仅利用生成图像作为输入。  
- 目标是恢复高价值提示词。  

**Stage 2: PromptStealer Modules（PromptStealer 组件）**  
- Subject Generator 推断主体。  
- Modifier Detector 识别修饰词。  

**Stage 3: Prompt Reconstruction & Defense（重建与防御）**  
- 组合主体与修饰词重建提示。  
- 评估并探讨初步防护。  

**Summary（总结）**  
- 首次系统研究 T2I 提示词窃取。  
- 在无模型访问下恢复高价值提示。  

**Analysis（分析）**  
- 直接冲击“提示词资产化”生态。  
- 需要图像级或提示级水印/混淆保护。  

## 54. Nightshade: Prompt-Specific Poisoning Attacks on Text-to-Image Generative Models
Shan, S., Ding, W., Passananti, J., Wu, S., Zheng, H., & Zhao, B. Y. (2024). IEEE S&P 2024 (CCF A).  
PDF: [UChicago](https://people.cs.uchicago.edu/~ravenben/publications/abstracts/nightshade-oakland24.html)

[Image]

**Stage 1: Target Concept Selection（目标概念选择）**  
- 针对特定概念/提示进行攻击设计。  

**Stage 2: Prompt-specific Poisoning（定向投毒）**  
- 用少量优化样本污染训练数据。  
- 诱导模型在该概念上产生错误输出。  

**Stage 3: Effect & Robustness（效果与鲁棒性）**  
- 验证模型对特定概念的系统性偏移。  
- 讨论潜在防御与对抗。  

**Summary（总结）**  
- 证明 T2I 模型对“概念级小比例投毒”敏感。  
- 攻击不依赖模型内部细节。  

**Analysis（分析）**  
- 黑盒训练数据链路是关键风险面。  
- 需要更强的数据审计与去噪机制。  

## 55. Glaze: Protecting Artists from Style Mimicry by Text-to-Image Models
Shan, S., Cryan, J., Wenger, E., Zheng, H., Hanocka, R., & Zhao, B. Y. (2023). USENIX Security 2023 (CCF A).  
PDF: [USENIX](https://www.usenix.org/system/files/usenixsecurity23-shan.pdf)

[Image]

**Stage 1: Style Cloak Generation（风格披风生成）**  
- 对艺术作品添加微小扰动。  
- 保持人眼不可察觉。  

**Stage 2: Training-time Misleading（训练期误导）**  
- 使用披风图像训练/微调模型。  
- 误导模型学习风格特征。  

**Stage 3: Robustness Evaluation（鲁棒评测）**  
- 评估对抗自适应去噪/反制手段。  
- 验证对风格模仿的抑制效果。  

**Summary（总结）**  
- 提出艺术家可用的黑盒保护工具。  
- 显著降低被模仿的风险。  

**Analysis（分析）**  
- 适合内容发布前的前置防护。  
- 与 Nightshade 可形成组合策略。  

## 56. T2ISafety: Benchmark for Assessing Fairness, Toxicity, and Privacy in Image Generation
Li, L., Shi, Z., Hu, X., Dong, B., Qin, Y., Liu, X., Sheng, L., & Shao, J. (2025). CVPR 2025 (CCF A).  
PDF: [CVPR](https://openaccess.thecvf.com/content/CVPR2025/papers/Li_T2ISafety_Benchmark_for_Assessing_Fairness_Toxicity_and_Privacy_in_Image_CVPR_2025_paper.pdf)

[Image]

**Stage 1: Taxonomy & Prompts（风险分类与提示构建）**  
- 构建公平、毒性、隐私三大域层级体系。  
- 采集大规模风险提示。  

**Stage 2: Data & Evaluator（数据与评测器）**  
- 生成并标注大规模图像。  
- 训练安全评测器用于检测风险。  

**Stage 3: Black-box Benchmarking（黑盒评测）**  
- 评测多款开源与闭源模型。  
- 揭示安全短板与偏差。  

**Summary（总结）**  
- 提供系统化 T2I 安全基准。  
- 评测覆盖毒性、公平与隐私。  

**Analysis（分析）**  
- 适用于持续红队与安全回归测试。  
- 有助于衡量防御策略的真实收益。  

## 57. SurrogatePrompt: Bypassing the Safety Filter of Text-to-Image Models via Substitution
Ba, Z., Zhong, J., Lei, J., Cheng, P., Wang, Q., Qin, Z., Wang, Z., & Ren, K. (2024). CCS 2024 (CCF A).  
PDF: [arXiv](https://arxiv.org/pdf/2309.14122)

[Image]

**Stage 1: Risky Segment Identification（风险片段定位）**  
- 从高风险提示中定位敏感片段。  

**Stage 2: Surrogate Substitution（替换式生成）**  
- 用代理短语替换敏感片段。  
- 组合 LLM / image-to-text / image-to-image 模块生成攻击提示。  

**Stage 3: Closed-source Bypass（闭源绕过验证）**  
- 在 Midjourney 等闭源系统上验证绕过。  
- 量化成功率与可感知性。  

**Summary（总结）**  
- 提出“替换式”提示词越狱框架。  
- 成功绕过闭源安全过滤器。  

**Analysis（分析）**  
- 黑盒系统的关键弱点在于替换空间巨大。  
- 需要引入多模态一致性与语义风险检测。  

## 58. On the Feasibility of Poisoning Text-to-Image AI Models via Adversarial Mislabeling
Wu, S., Bhaskar, R., Ha, A. Y. J., Shan, S., Zheng, H., & Zhao, B. Y. (2025). CCS 2025 (CCF A).  
PDF: [UChicago](https://people.cs.uchicago.edu/~ravenben/publications/abstracts/amp-ccs25.html)

[Image]

**Stage 1: VLM Mislabeling（VLM 误标注）**  
- 对图像施加微扰，使 VLM 生成错误描述。  

**Stage 2: Poisoned Caption Pipeline（污染训练管线）**  
- 将误标注样本注入 T2I 训练数据。  
- 形成“脏标签”投毒。  

**Stage 3: Black-box Validation（黑盒验证）**  
- 在商业 VLM 场景验证攻击有效性。  
- 量化投毒成功率与影响范围。  

**Summary（总结）**  
- 展示“误标注投毒”可破坏 T2I 训练管线。  
- 现实世界黑盒场景同样可行。  

**Analysis（分析）**  
- 数据生产链路成为关键攻击面。  
- 需加强 VLM 标注鲁棒性与数据校验。  

## 59. Towards Reliable Verification of Unauthorized Data Usage in Personalized Text-to-Image Diffusion Models
Li, B., Wei, Y., Fu, Y., Wang, Z., Li, Y., Zhang, J., Wang, R., & Zhang, T. (2025). IEEE S&P 2025 (CCF A).  
PDF: [IEEE S&P](https://doi.org/10.1109/SP61157.2025.00073)

[Image]

**Stage 1: Learnable Coating Design（可学习涂层设计）**  
- 设计能被模型学习的隐蔽涂层。  
- 提升个性化训练中的可识别性。  

**Stage 2: Personalization & Learning（个性化训练）**  
- 观察模型在个性化任务中的学习行为。  
- 使涂层成为可检测特征。  

**Stage 3: Black-box Verification（黑盒验证）**  
- 通过输出与统计检验判断是否非法使用数据。  
- 评估多模型、多场景鲁棒性。  

**Summary（总结）**  
- 提出面向黑盒个性化模型的数据使用验证方法。  
- 增强涂层可学习性与检测可靠性。  

**Analysis（分析）**  
- 适合版权与数据合规审计。  
- 可与水印/溯源体系互补。  

## 60. Identifying Provenance of Generative Text-to-Image Models
Ding, W., Wu, S., Shan, S., Zheng, H., & Zhao, B. Y. (2026). USENIX Security 2026 (CCF A).  
PDF: [UChicago](https://annaha.net/publication/model-provenance/)

[Image]

**Stage 1: Black-box Querying（黑盒查询）**  
- 用丰富提示查询目标模型。  
- 收集生成图像作为观测。  

**Stage 2: Feature Distribution Comparison（特征分布对比）**  
- 提取视觉特征并与基准模型比对。  
- 估计是否由某基座模型微调而来。  

**Stage 3: Provenance Decision（溯源判定）**  
- 统计检验判断模型谱系。  
- 在真实环境与对抗场景验证。  

**Summary（总结）**  
- 仅需黑盒访问即可进行模型谱系归因。  
- 有助于模型合规与责任追踪。  

**Analysis（分析）**  
- 适合平台对模型来源的合规审查。  
- 需关注对抗性规避与后处理扰动。


---

# 汇总表

| Title | PDF | Model/Target | Access | Defense | Venue | CCF | Year |
| --- | --- | --- | --- | --- | --- | --- | --- |
| Red-Teaming Text-to-Image Systems by Rule-based Preference Modeling | [PDF](https://arxiv.org/pdf/2505.21074) | 黑盒T2I系统（商用+开源） | 黑盒 | 无（红队/攻击） | NeurIPS | A | 2025 |
| Fuzz-Testing Meets LLM-Based Agents (JailFuzzer) | [PDF](https://arxiv.org/pdf/2408.00523) | T2I系统 | 黑盒 | 无（攻击） | S&P | A | 2025 |
| SneakyPrompt: Jailbreaking Text-to-image Generative Models | [PDF](https://arxiv.org/pdf/2305.12082) | DALL·E 2 + SD | 黑盒 | 无（攻击） | S&P | A | 2024 |
| PLA: Prompt Learning Attack against Text-to-Image Generative Models | [PDF](https://openaccess.thecvf.com/content/ICCV2025/papers/Lyu_PLA_Prompt_Learning_Attack_against_Text-to-Image_Generative_Models_ICCV_2025_paper.pdf) | 黑盒T2I模型 | 黑盒 | 无（攻击） | ICCV | A | 2025 |
| JailbreakDiffBench: A Comprehensive Benchmark for Jailbreaking Diffusion Models | [PDF](https://openaccess.thecvf.com/content/ICCV2025/papers/Jin_JailbreakDiffBench_A_Comprehensive_Benchmark_for_Jailbreaking_Diffusion_Models_ICCV_2025_paper.pdf) | T2I/T2V系统 | 评测 | 无（基准） | ICCV | A | 2025 |
| Efficient Input-level Backdoor Defense on Text-to-Image Synthesis via Neuron Activation Variation (NaviDet) | [PDF](https://openaccess.thecvf.com/content/ICCV2025/papers/Zhai_Efficient_Input-level_Backdoor_Defense_on_Text-to-Image_Synthesis_via_Neuron_Activation_ICCV_2025_paper.pdf) | T2I扩散模型 | 白盒 | 防御 | ICCV | A | 2025 |
| Prompting4Debugging: Red-Teaming Text-to-Image Diffusion Models by Finding Problematic Prompts | [PDF](https://proceedings.mlr.press/v235/chin24a/chin24a.pdf) | T2I扩散模型 | 黑盒 | 无（红队） | ICML | A | 2024 |
| Multimodal Pragmatic Jailbreak on Text-to-image Models | [PDF](https://aclanthology.org/2025.acl-long.234.pdf) | 多T2I模型 | 黑盒 | 无（评测） | ACL | A | 2025 |
| Latent Guard: a Safety Framework for Text-to-image Generation | [PDF](https://www.ecva.net/papers/eccv_2024/papers_ECCV/papers/03726.pdf) | T2I系统 | 白盒 | 防御 | ECCV | B | 2024 |
| Safeguard Text-to-Image Diffusion Models with Human Feedback Inversion (HFI) | [PDF](https://www.ecva.net/papers/eccv_2024/papers_ECCV/papers/08393.pdf) | T2I扩散模型 | 白盒 | 防御 | ECCV | B | 2024 |
| Geom-Erasing: Implicit Concept Removal of Diffusion Models | [PDF](https://www.ecva.net/papers/eccv_2024/papers_ECCV/papers/03200.pdf) | T2I扩散模型 | 白盒 | 防御 | ECCV | B | 2024 |
| R.A.C.E.: Robust Adversarial Concept Erasure for Secure T2I Diffusion Model | [PDF](https://www.ecva.net/papers/eccv_2024/papers_ECCV/papers/11117.pdf) | T2I扩散模型 | 白盒 | 防御 | ECCV | B | 2024 |
| Reliable and Efficient Concept Erasure (RECE) | [PDF](https://www.ecva.net/papers/eccv_2024/papers_ECCV/papers/06950.pdf) | T2I扩散模型 | 白盒 | 防御 | ECCV | B | 2024 |
| Receler: Reliable Concept Erasing via Lightweight Erasers | [PDF](https://www.ecva.net/papers/eccv_2024/papers_ECCV/papers/05685.pdf) | T2I扩散模型 | 白盒 | 防御 | ECCV | B | 2024 |
| Detect-and-Guide: Self-regulation of Diffusion Models for Safe T2I Generation | [PDF](https://openaccess.thecvf.com/content/CVPR2025/papers/Li_Detect-and-Guide_Self-regulation_of_Diffusion_Models_for_Safe_Text-to-Image_Generation_via_CVPR_2025_paper.pdf) | T2I扩散模型 | 白盒 | 防御 | CVPR | A | 2025 |
| Implicit Bias Injection Attacks against T2I Diffusion Models | [PDF](https://openaccess.thecvf.com/content/CVPR2025/papers/Huang_Implicit_Bias_Injection_Attacks_against_Text-to-Image_Diffusion_Models_CVPR_2025_paper.pdf) | T2I扩散模型 | 白盒 | 攻击 | CVPR | A | 2025 |
| OpenSDI: Spotting Diffusion-Generated Images in the Open World | [PDF](https://openaccess.thecvf.com/content/CVPR2025/papers/Wang_OpenSDI_Spotting_Diffusion-Generated_Images_in_the_Open_World_CVPR_2025_paper.pdf) | 扩散图像检测 | 评测 | 检测 | CVPR | A | 2025 |
| SleeperMark: Robust Watermark against Fine-Tuning T2I Diffusion Models | [PDF](https://openaccess.thecvf.com/content/CVPR2025/papers/Wang_SleeperMark_Towards_Robust_Watermark_against_Fine-Tuning_Text-to-image_Diffusion_Models_CVPR_2025_paper.pdf) | 扩散模型 | 白盒 | 水印/防护 | CVPR | A | 2025 |
| Black-Box Forgery Attacks on Semantic Watermarks | [PDF](https://openaccess.thecvf.com/content/CVPR2025/papers/Muller_Black-Box_Forgery_Attacks_on_Semantic_Watermarks_for_Diffusion_Models_CVPR_2025_paper.pdf) | 语义水印 | 黑盒 | 攻击 | CVPR | A | 2025 |
| Silent Branding Attack: Trigger-free Data Poisoning Attack on T2I Diffusion Models | [PDF](https://openaccess.thecvf.com/content/CVPR2025/papers/Jang_Silent_Branding_Attack_Trigger-free_Data_Poisoning_Attack_on_Text-to-Image_Diffusion_CVPR_2025_paper.pdf) | 扩散模型 | 白盒 | 攻击（投毒） | CVPR | A | 2025 |
| Nearly Zero-Cost Protection Against Mimicry | [PDF](https://openaccess.thecvf.com/content/CVPR2025/papers/Ahn_Nearly_Zero-Cost_Protection_Against_Mimicry_by_Personalized_Diffusion_Models_CVPR_2025_paper.pdf) | 个性化扩散模型 | 白盒 | 防护 | CVPR | A | 2025 |
| ConceptGuard: Continual Personalized T2I Generation | [PDF](https://openaccess.thecvf.com/content/CVPR2025/papers/Guo_ConceptGuard_Continual_Personalized_Text-to-Image_Generation_with_Forgetting_and_Confusion_Mitigation_CVPR_2025_paper.pdf) | 个性化扩散模型 | 白盒 | 防御 | CVPR | A | 2025 |
| ACE: Anti-Editing Concept Erasure in T2I Models | [PDF](https://openaccess.thecvf.com/content/CVPR2025/papers/Wang_ACE_Anti-Editing_Concept_Erasure_in_Text-to-Image_Models_CVPR_2025_paper.pdf) | 扩散模型 | 白盒 | 防御 | CVPR | A | 2025 |
| GLoCE: Localized Concept Erasure (Training-Free) | [PDF](https://openaccess.thecvf.com/content/CVPR2025/papers/Lee_Localized_Concept_Erasure_for_Text-to-Image_Diffusion_Models_Using_Training-Free_Gated_CVPR_2025_paper.pdf) | 扩散模型 | 白盒 | 防御 | CVPR | A | 2025 |
| FADE: Fine-Grained Erasure in T2I Diffusion Models | [PDF](https://openaccess.thecvf.com/content/CVPR2025/papers/Thakral_Fine-Grained_Erasure_in_Text-to-Image_Diffusion-based_Foundation_Models_CVPR_2025_paper.pdf) | 扩散模型 | 白盒 | 防御 | CVPR | A | 2025 |
| Six-CD: Benchmarking Concept Removals | [PDF](https://openaccess.thecvf.com/content/CVPR2025/papers/Ren_Six-CD_Benchmarking_Concept_Removals_for_Text-to-image_Diffusion_Models_CVPR_2025_paper.pdf) | 扩散模型 | 评测 | 无（基准） | CVPR | A | 2025 |
| MACE: Mass Concept Erasure in Diffusion Models | [PDF](https://openaccess.thecvf.com/content/CVPR2024/papers/Lu_MACE_Mass_Concept_Erasure_in_Diffusion_Models_CVPR_2024_paper.pdf) | 扩散模型 | 白盒 | 防御 | CVPR | A | 2024 |
| Gaussian Shading: Performance-Lossless Watermarking | [PDF](https://openaccess.thecvf.com/content/CVPR2024/papers/Yang_Gaussian_Shading_Provable_Performance-Lossless_Image_Watermarking_for_Diffusion_Models_CVPR_2024_paper.pdf) | 扩散模型水印 | 白盒 | 水印/防护 | CVPR | A | 2024 |
| Safe Latent Diffusion | [PDF](https://openaccess.thecvf.com/content/CVPR2023/papers/Schramowski_Safe_Latent_Diffusion_Mitigating_Inappropriate_Degeneration_in_Diffusion_Models_CVPR_2023_paper.pdf) | 扩散模型 | 白盒 | 防御 | CVPR | A | 2023 |
| Tree-Rings Watermarks | [PDF](https://arxiv.org/pdf/2305.20030) | 扩散模型水印 | 白盒 | 水印/防护 | NeurIPS | A | 2023 |
| MMA-Diffusion: MultiModal Attack on Diffusion Models | [PDF](https://openaccess.thecvf.com/content/CVPR2024/papers/Yang_MMA-Diffusion_MultiModal_Attack_on_Diffusion_Models_CVPR_2024_paper.pdf) | T2I扩散模型（开源+商用） | 黑盒 | 攻击（越狱） | CVPR | A | 2024 |
| Unsafe Diffusion: On the Generation of Unsafe Images and Hateful Memes From Text-To-Image Models | [PDF](https://arxiv.org/pdf/2305.13873.pdf) | T2I模型安全评测 | 评测 | 无（风险评估） | CCS | A | 2023 |
| T2IShield: Defending Against Backdoors on Text-to-Image Diffusion Models | [PDF](https://arxiv.org/pdf/2407.04215) | T2I扩散模型 | 白盒 | 防御 | ECCV | B | 2024 |
| Training-Free Safe Text Embedding Guidance (STG) | [PDF](https://arxiv.org/pdf/2510.24012) | T2I扩散模型 | 白盒 | 防御 | NeurIPS | A | 2025 |
| Erasing Undesirable Influence in Diffusion Models (EraseDiff) | [PDF](https://openaccess.thecvf.com/content/CVPR2025/papers/Wu_Erasing_Undesirable_Influence_in_Diffusion_Models_CVPR_2025_paper.pdf) | 扩散模型 | 白盒 | 防御/遗忘 | CVPR | A | 2025 |
| Self-Discovering Interpretable Diffusion Latent Directions | [PDF](https://openaccess.thecvf.com/content/CVPR2024/papers/Li_Self-Discovering_Interpretable_Diffusion_Latent_Directions_for_Responsible_Text-to-Image_Generation_CVPR_2024_paper.pdf) | T2I扩散模型 | 白盒 | 防御/责任生成 | CVPR | A | 2024 |
| CoprGuard: Harnessing Frequency Spectrum Insights for Image Copyright Protection | [PDF](https://openaccess.thecvf.com/content/CVPR2025/papers/Liu_Harnessing_Frequency_Spectrum_Insights_for_Image_Copyright_Protection_Against_Diffusion_CVPR_2025_paper.pdf) | 训练数据版权保护 | 黑盒 | 水印/版权 | CVPR | A | 2025 |
| Your Text Encoder Can Be An Object-Level Watermarking Controller | [PDF](https://openaccess.thecvf.com/content/ICCV2025/papers/Devulapally_Your_Text_Encoder_Can_Be_An_Object-Level_Watermarking_Controller_ICCV_2025_paper.pdf) | T2I LDM | 白盒 | 水印 | ICCV | A | 2025 |
| PlugMark: A Plug-in Zero-Watermarking Framework for Diffusion Models | [PDF](https://openaccess.thecvf.com/content/ICCV2025/papers/Chen_PlugMark_A_Plug-in_Zero-Watermarking_Framework_for_Diffusion_Models_ICCV_2025_paper.pdf) | 扩散模型IP保护 | 白盒 | 水印 | ICCV | A | 2025 |
| Who Controls the Authorization? Invertible Networks for Copyright Protection in Text-to-Image Synthesis | [PDF](https://openaccess.thecvf.com/content/ICCV2025/papers/Hu_Who_Controls_the_Authorization_Invertible_Networks_for_Copyright_Protection_in_ICCV_2025_paper.pdf) | 个性化T2I | 白盒 | 版权保护 | ICCV | A | 2025 |
| USD: NSFW Content Detection for Text-to-Image Models via Scene Graph | [PDF](https://www.usenix.org/system/files/usenixsecurity25-zhang-yuyang.pdf) | T2I生成图像检测 | 评测 | 检测 | USENIX Security | A | 2025 |
| Exposing the Guardrails: Reverse-Engineering and Jailbreaking Safety Filters in DALL·E T2I Pipelines | [PDF](https://www.usenix.org/system/files/usenixsecurity25-villa.pdf) | DALL·E管线 | 黑盒 | 攻击（越狱） | USENIX Security | A | 2025 |
| On the Proactive Generation of Unsafe Images From T2I Models Using Benign Prompts | [PDF](https://www.usenix.org/system/files/usenixsecurity25-wu-yixin-generation.pdf) | T2I模型（被投毒） | 白盒 | 攻击（投毒） | USENIX Security | A | 2025 |
| Backdooring Bias (B^2) into Stable Diffusion Models | [PDF](https://www.usenix.org/system/files/usenixsecurity25-naseh.pdf) | Stable Diffusion | 白盒 | 攻击（后门/偏置） | USENIX Security | A | 2025 |
| EvilEdit: Backdooring Text-to-Image Diffusion Models in One Second | [PDF](https://dl.acm.org/doi/pdf/10.1145/3664647.3680689) | Stable Diffusion | 白盒 | 攻击（后门） | ACM MM | A | 2024 |
| Text-to-Image Diffusion Models can be Easily Backdoored through Multimodal Data Poisoning (BadT2I) | [PDF](https://arxiv.org/pdf/2305.04175) | Stable Diffusion | 白盒 | 攻击（后门/投毒） | ACM MM | A | 2023 |
| Rickrolling the Artist: Injecting Backdoors into Text Encoders for Text-to-Image Synthesis | [PDF](https://openaccess.thecvf.com/content/ICCV2023/papers/Struppek_Rickrolling_the_Artist_Injecting_Backdoors_into_Text_Encoders_for_Text-to-Image_ICCV_2023_paper.pdf) | 文本编码器 | 白盒 | 攻击（后门） | ICCV | A | 2023 |
| Membership Inference on Text-to-Image Diffusion Models via Conditional Likelihood Discrepancy | [PDF](https://proceedings.neurips.cc/paper_files/paper/2024/file/874411a224a1934b80d499068384808b-Paper-Conference.pdf) | T2I扩散模型 | 白盒 | 攻击（隐私推断） | NeurIPS | A | 2024 |
| Cross-Modal Prompt Inversion: Unifying Threats to Text and Image Generative AI Models | [PDF](https://www.usenix.org/system/files/usenixsecurity25-ye-inversion.pdf) | 文本/图像生成模型 | 黑盒 | 攻击（提示词窃取） | USENIX Security | A | 2025 |
| Bridging the Gap in VLMs in Identifying Unsafe Concepts Across Modalities | [PDF](https://www.usenix.org/system/files/usenixsecurity25-qu-yiting.pdf) | VLM安全检测 | 评测 | 检测/对齐 | USENIX Security | A | 2025 |
| Mind the Trojan Horse: Image Prompt Adapter Enabling Scalable and Deceptive Jailbreaking | [PDF](https://openaccess.thecvf.com/content/CVPR2025/papers/Chen_Mind_the_Trojan_Horse_Image_Prompt_Adapter_Enabling_Scalable_and_CVPR_2025_paper.pdf) | T2I-IP-DMs / IGS | 黑盒 | 攻击（越狱） | CVPR | A | 2025 |
| Perception-guided Jailbreak against Text-to-Image Models | [PDF](https://arxiv.org/pdf/2408.10848) | 多种T2I模型 | 黑盒 | 攻击（越狱） | AAAI | A | 2025 |
| Prompt Stealing Attacks Against Text-to-Image Generation Models | [PDF](https://www.usenix.org/system/files/usenixsecurity24-shen-xinyue.pdf) | 生成图像 -> 提示词 | 黑盒 | 攻击（提示词窃取） | USENIX Security | A | 2024 |
| Nightshade: Prompt-Specific Poisoning Attacks on Text-to-Image Generative Models | [PDF](https://people.cs.uchicago.edu/~ravenben/publications/abstracts/nightshade-oakland24.html) | T2I训练数据链路 | 黑盒 | 攻击（投毒） | S&P | A | 2024 |
| Glaze: Protecting Artists from Style Mimicry by Text-to-Image Models | [PDF](https://www.usenix.org/system/files/usenixsecurity23-shan.pdf) | 艺术风格保护 | 黑盒 | 防护 | USENIX Security | A | 2023 |
| T2ISafety: Benchmark for Assessing Fairness, Toxicity, and Privacy in Image Generation | [PDF](https://openaccess.thecvf.com/content/CVPR2025/papers/Li_T2ISafety_Benchmark_for_Assessing_Fairness_Toxicity_and_Privacy_in_Image_CVPR_2025_paper.pdf) | T2I安全评测 | 黑盒 | 评测（安全基准） | CVPR | A | 2025 |
| SurrogatePrompt: Bypassing the Safety Filter of Text-to-Image Models via Substitution | [PDF](https://arxiv.org/pdf/2309.14122) | Midjourney 等闭源T2I | 黑盒 | 攻击（越狱） | CCS | A | 2024 |
| On the Feasibility of Poisoning Text-to-Image AI Models via Adversarial Mislabeling | [PDF](https://people.cs.uchicago.edu/~ravenben/publications/abstracts/amp-ccs25.html) | T2I训练管线 | 黑盒 | 攻击（投毒） | CCS | A | 2025 |
| Towards Reliable Verification of Unauthorized Data Usage in Personalized Text-to-Image Diffusion Models | [PDF](https://doi.org/10.1109/SP61157.2025.00073) | 个性化T2I模型 | 黑盒 | 溯源/验证 | S&P | A | 2025 |
| Identifying Provenance of Generative Text-to-Image Models | [PDF](https://annaha.net/publication/model-provenance/) | T2I模型谱系 | 黑盒 | 溯源/归因 | USENIX Security | A | 2026 |
