# 论文细节

说明：已按“黑盒访问”筛选并去重，补齐 30 篇；评测/检测类工作在不依赖模型权重时视为黑盒。

## 1. Perception-guided Jailbreak against Text-to-Image Models

Huang, Y., Liang, L., Li, T., Jia, X., Wang, R., Miao, W., Pu, G., & Liu, Y. (2025). AAAI 2025 (CCF A).
PDF: [arXiv](https://arxiv.org/pdf/2408.10848)

[Image]

**阶段一：快速概览与初步印象**
**Prompt 1（三句话概括主题与目的）**
论文提出 Perception-guided Jailbreak（PGJ），利用“感知相似但语义不同”的替代表达绕过安全过滤。方法面向黑盒 T2I 系统，在不暴露目标词的情况下诱导生成风险内容。实验显示在多模型/多服务上具备可迁移性。
**Prompt 2（章节结构与大致内容）**

- Abstract：提出 PGJ 与核心动机。
- 1 Introduction：安全过滤局限与问题定义。
- 2 Related Work：提示词攻击与安全评测。
- 3 Method：感知差异挖掘与替换策略。
- 4 Experiments：黑盒评测与对比。
- 5 Discussion/Conclusion。
  **Prompt 3（核心创新点/贡献）**
  通过感知差异挖掘与替换式提示，系统化实现黑盒越狱。
  **Prompt 4（关键词 5-10 个）**
- jailbreak：越狱攻击。
- perception gap：感知差异。
- prompt substitution：提示替换。
- black-box：黑盒攻击。
- safety filter：安全过滤。
- text-to-image：文生图。

**阶段二：深入细节与理解内容**
**Prompt 5（引言解读）**

- 试图解决的问题：关键词/语义过滤易被绕过。
- 背景与铺垫：黑盒 T2I 广泛部署且过滤策略有限。
- 研究问题：能否用感知相似替换绕过过滤。
- 主要贡献：PGJ 框架与跨模型评测。
  **Prompt 6（方法解读）**
- 方法类型：提示词生成与黑盒评测。
- 执行方式：挖掘感知相似短语；用安全替代短语替换风险词；在黑盒系统上评估绕过效果。
- 关键变量/指标：越狱成功率、语义/感知保持度。
- 局限性/偏差：对感知度量质量依赖较高。
- 方法合理性：利用“感知一致性”对抗语义过滤。
  **Prompt 7（结果解读）**
- 主要结果：多模型场景下越狱成功率显著提升。
- 关键图表：成功率对比表与案例图。
- 研究问题对应：证明感知替换可绕过过滤。
- 出乎意料发现：部分服务对感知替换更脆弱。
- 统计显著性：未强调显著性检验。
  **Prompt 8（讨论解读）**
- 解释：过滤器更多依赖语义匹配而非感知一致性。
- 比较：相较纯文本替换更隐蔽。
- 重要意义：需要引入感知级安全检测。
- 局限性：替换词库构建成本仍在。
- 未来方向：结合多模态对齐与更强过滤。
  **Prompt 9（结论解读）**
- 主要发现：PGJ 能在黑盒场景绕过过滤。
- 贡献重申：感知差异驱动的越狱流程。
- 呼应研究问题：证明替换式越狱可行。
- 未来研究：更系统的感知安全防护。

**阶段三：批判性思考与深入理解**
**Prompt 10（方法批判性评估）**

- 优点：黑盒可用，适配闭源系统。
- 局限：依赖感知相似度度量与词库。
- 替代方法：基于梯度的提示优化或多模态攻击。
  **Prompt 11（结果可信度评估）**
- 可信度：多模型验证提升可信度。
- 是否支持结论：整体支持。
- 其他解释：过滤器策略差异影响结果。
  **Prompt 12（创新性评估）**
- 创新点：以感知一致性为绕过核心。
- 价值：揭示过滤机制盲点。
- 相对贡献：扩展黑盒越狱策略。
  **Prompt 13（逻辑结构与论证）**
- 论证清晰：问题→方法→评测。
- 衔接良好：替换策略与结果一致。
- 薄弱点：对复杂提示的泛化仍需验证。
  **Prompt 14（优缺点总结）**
- 优点：简单有效、可迁移。
- 缺点：依赖感知差异挖掘质量。
- 综合评价：黑盒越狱的代表性方法之一。

## 2. Multimodal Pragmatic Jailbreak on Text-to-image Models

Liu, T., Lai, Z., Wang, J., Zhang, G., Chen, S., Torr, P., Demberg, V., Tresp, V., & Gu, J. (2025). ACL 2025 (CCF A).
PDF: [arXiv](https://arxiv.org/pdf/2409.19149)

[Image]

**阶段一：快速概览与初步印象**
**Prompt 1（三句话概括主题与目的）**
提出一种多模态语用越狱：图像与视觉文本单独安全但组合成不安全内容。本文构建 MPUP 数据集并系统评测 9 个 T2I 模型的越狱风险。结果显示越狱成功率高且现有单模态过滤难以拦截，从而揭示安全缺口。
**Prompt 2（章节结构与大致内容）**

- 1 Introduction：提出问题与动机，定义多模态语用越狱。
- 2 Background：回顾 LLM/MLLM jailbreak、T2I 安全与视觉文本渲染。
- 3 Multimodal Pragmatic Jailbreak Benchmark：3.1 数据来源与构建；3.2 修辞/语用类别标注；3.3 安全分类器设计。
- 4 Experimental Setup：4.1 评测模型；4.2 指标与评价。
- 5 Experimental Results and Analysis：5.1 主实验 ASR；5.2 安全分类器效果；5.3 在线服务评测。
- 6 Discussions：成因分析与对图像编辑模型的讨论。
- 7 Conclusion：总结与展望。
  **Prompt 3（核心创新点/贡献）**
  提出“多模态语用越狱”概念与 MPUP 基准，系统评测 9 个模型并量化 ASR 与 OCR 关联，揭示现有文本/图像过滤器的不足。
  **Prompt 4（关键词 5-10 个）**
- multimodal pragmatic jailbreak：安全文本+安全图像组合导致越狱。
- visual text rendering：视觉文本生成能力是触发关键。
- MPUP dataset：用于评测的语用越狱提示集。
- ASR (attack success rate)：衡量越狱成功率的核心指标。
- OCR accuracy：用来关联视觉文本可读性与风险。
- safety filters：包含 blocklist/LLM/CLIP/NSFW 等过滤策略。
- T2I diffusion models：评测对象（SD、SDXL、DALL·E 等）。

**阶段二：深入细节与理解内容**
**Prompt 5（引言解读）**

- 试图解决的问题：现有安全过滤无法识别“图像+视觉文本”组合带来的隐性不安全。
- 背景铺垫：LLM/MLLM jailbreak 与 T2I 安全过滤的局限性。
- 研究问题：多模态语用越狱的普遍性、风险强度与成因。
- 主要贡献：提出新越狱类型、MPUP 数据集与系统评测。
  **Prompt 6（方法解读）**
- 方法类型：基准构建 + 多模型实验评测 + 分类器评估。
- 执行方式：构造“图像提示+视觉文本提示”模板，GPT-4 生成并人工筛选；用 9 个模型生成图像并计算 ASR；引入文本/图像分类器评估过滤效果；用 OCR 测试文本渲染能力。
- 关键变量/指标：ASR、OCR 准确率、分类器准确率/拒绝率。
- 局限性：类别集中在仇恨/伤害/欺诈，且依赖视觉文本渲染能力。
- 方法合理性：系统化评测能揭示“组合风险”与过滤器缺陷。
  **Prompt 7（结果解读）**
- 主要结果：九个模型均可被越狱，ASR 范围约 8%–74%，DALL·E 3 等高文本渲染能力模型风险更高。
- 关键图表：Table 1（各模型 ASR）；Table 2（OCR 准确率与风险相关）；Table 3/4（文本与图像过滤器效果）。
- 研究问题对应：ASR 与过滤器评测直接回答“普遍性与防护有效性”。
- 出乎意料发现：即便 OCR 全串准确率不高，仍可能产生不安全解释。
- 统计显著性：文中未显式报告显著性检验。
  **Prompt 8（讨论解读）**
- 解释：模型视觉文本渲染能力与训练数据中的文本图像分布导致风险。
- 与既有研究比较：显示单模态过滤不足，需要多模态检测。
- 重要意义：为真实系统安全评测提供新基准与攻击面。
- 局限性：类别范围与提示模板仍有限。
- 未来方向：更强多模态检测与持续更新基准。
  **Prompt 9（结论解读）**
- 核心论点：多模态语用越狱在主流 T2I 中普遍存在且过滤器易失效。
- 贡献重申：提出 MPUP 数据集与系统评测框架。
- 呼应研究问题：明确量化风险与成因。
- 未来研究：建设更强的多模态安全机制与评测。

**阶段三：批判性思考与深入理解**
**Prompt 10（方法批判性评估）**

- 优点：基准数据 + 多模型系统评测，覆盖真实系统。
- 局限：提示模板依赖视觉文本渲染能力；类别范围有限。
- 替代方法：加入更广泛的多模态越狱形态与真实用户提示分布。
  **Prompt 11（结果可信度评估）**
- 可信度：多模型一致性增强可信度，但仍依赖分类器与人工筛选。
- 支持结论：结果与过滤评测一致支撑风险结论。
- 其他解释：模型间差异可能受训练数据与渲染能力共同影响。
  **Prompt 12（创新性评估）**
- 创新点：提出“语用越狱”新攻击面并量化其普遍性。
- 价值：为安全基准与防护设计提供方向。
- 相对贡献：相较单模态越狱研究扩展到多模态组合风险。
  **Prompt 13（逻辑结构与论证）**
- 结构清晰：背景→基准→评测→讨论。
- 衔接：从渲染能力解释 ASR 差异，逻辑连贯。
- 潜在薄弱：类别与模板限制可能影响泛化。
  **Prompt 14（优缺点总结）**
- 优点：系统化评测 + 多模型覆盖 + 过滤器失效证据。
- 缺点：依赖视觉文本渲染能力与特定提示模板。
- 综合评价：研究问题重要、方法可复用，但需更广泛数据支撑。

## 3. Unsafe Diffusion: On the Generation of Unsafe Images and Hateful Memes From Text-To-Image Models

Qu, Y., Shen, X., He, X., Backes, M., Zannettou, S., & Zhang, Y. (2023). ACM CCS 2023 (CCF A).
PDF: [arXiv](https://arxiv.org/pdf/2305.13873)

[Image]

**阶段一：快速概览与初步印象**
**Prompt 1（三句话概括主题与目的）**
本文建立不安全内容类型学并系统评估 T2I 模型生成不安全图像的风险。作者在四类提示与四个模型上量化不安全比例，并进一步研究仇恨梗图的自动生成可行性。结果显示不安全生成比例显著且 DreamBooth 等编辑技术可高效生成仇恨变体。
**Prompt 2（章节结构与大致内容）**

- Introduction：提出安全风险与研究问题。
- Background：T2I 模型与编辑方法综述。
- RQ1 Safety Assessment：提示集与多头安全分类器、风险测量。
- RQ2 Hateful Meme Generation：威胁模型、编辑方法与评测指标。
- Discussion/Mitigations：安全链路上的缓解措施。
- Ethical Considerations & Conclusion。
  **Prompt 3（核心创新点/贡献）**
  提出不安全内容五类分类体系、构建多头安全分类器并量化多模型风险，同时系统验证仇恨梗图自动生成的现实可行性。
  **Prompt 4（关键词 5-10 个）**
- unsafe content taxonomy：不安全内容类型学。
- T2I safety assessment：文生图安全评测框架。
- multi-headed safety classifier：多类别安全检测器。
- prompt datasets (4chan/Lexica/Template/COCO)：多源提示集。
- training data contamination：训练数据不安全比例。
- hateful meme generation：仇恨梗图自动生成。
- DreamBooth/Textual Inversion/SDEdit：编辑攻击链路。

**阶段二：深入细节与理解内容**
**Prompt 5（引言解读）**

- 试图解决的问题：T2I 模型被滥用生成不安全图像/仇恨梗图。
- 背景铺垫：模型普及与现有安全过滤失效案例。
- 明确研究问题：RQ1 不安全生成比例与原因；RQ2 仇恨梗图可否自动生成。
- 主要贡献：风险评测框架 + 仇恨梗图生成实验。
  **Prompt 6（方法解读）**
- 方法类型：大规模实验评测 + 安全分类器构建 + 生成式编辑实验。
- 执行方式：构造四类提示集；对四个模型生成图像；训练多头 CLIP 分类器评估不安全比例；用 DreamBooth/Textual Inversion/SDEdit 生成梗图变体并人工评估。
- 关键变量/指标：不安全比例、类别分布、分类器精度/F1、梗图成功率、文本对齐/图像保真。
- 局限性：分类器与提示集覆盖有限；模型数量有限。
- 方法选择合理性：系统化评测与编辑攻击分析适配研究问题。
  **Prompt 7（结果解读）**
- 主要结果：四模型整体不安全比例约 14.56%，SD 风险最高；DreamBooth 生成仇恨梗图成功率约 24%。
- 关键图表：Fig.3 不安全类别比例；Table 4 训练数据不安全比例估计；仇恨梗图评测表。
- 研究问题对应：RQ1 的风险比例与训练数据污染；RQ2 的梗图变体生成成功率。
- 出乎意料发现：模板提示集更易触发不安全；DALLE mini 在某些类别更高。
- 统计显著性：未突出显著性检验。
  **Prompt 8（讨论解读）**
- 解释：不安全生成与训练数据污染高度相关。
- 与既有研究比较：显示内置过滤主要针对性内容，覆盖不足。
- 重要意义：提示模型供应链需从数据、提示、后处理多层防护。
- 局限性：评测范围与模型覆盖仍有限。
- 未来方向：更强安全过滤、数据清洗与编辑攻击防护。
  **Prompt 9（结论解读）**
- 核心论点：T2I 模型可大规模生成不安全内容与仇恨梗图。
- 贡献重申：多源提示评测与仇恨梗图生成实证。
- 呼应研究问题：RQ1/RQ2 均得到明确回答。
- 未来研究：更全面的安全评测与防护。

**阶段三：批判性思考与深入理解**
**Prompt 10（方法批判性评估）**

- 优点：多模型/多提示评测，链路覆盖到编辑攻击。
- 局限：分类器误差可能影响比例估计；模型样本有限。
- 替代方法：引入更广泛真实提示、更多闭源模型评测。
  **Prompt 11（结果可信度评估）**
- 可信度：多数据源与人工评估增强可信度。
- 支持结论：风险比例与梗图实验支持结论。
- 其他解释：模型差异也可能来自提示语义匹配程度。
  **Prompt 12（创新性评估）**
- 创新点：系统化量化不安全生成与梗图自动化风险。
- 价值：为政策与防护提供可量化证据。
- 相对贡献：将安全评测扩展到 meme 生成与编辑链路。
  **Prompt 13（逻辑结构与论证）**
- 论证清晰：按 RQ1/RQ2 组织。
- 衔接良好：从风险测量过渡到具体攻击场景。
- 薄弱点：分类器与提示数据的外推性仍需验证。
  **Prompt 14（优缺点总结）**
- 优点：量化风险、揭示编辑攻击的现实威胁。
- 缺点：模型/提示覆盖有限，部分结论依赖检测器。
- 综合评价：对安全评测具有里程碑意义。

## 4. SurrogatePrompt: Bypassing the Safety Filter of Text-to-Image Models via Substitution

Ba, Z., Zhong, J., Lei, J., Cheng, P., Wang, Q., Qin, Z., Wang, Z., & Ren, K. (2024). CCS 2024 (CCF A).
PDF: [arXiv](https://arxiv.org/pdf/2309.14122)

[Image]

**阶段一：快速概览与初步印象**
**Prompt 1（三句话概括主题与目的）**
论文提出 SurrogatePrompt，通过“替换式”提示生成绕过闭源安全过滤器。方法定位敏感片段，用代理短语替换并组合多模态模块生成攻击提示。实验在 Midjourney 等系统上验证有效。
**Prompt 2（章节结构与大致内容）**

- Abstract：替换式越狱框架概述。
- 1 Introduction：安全过滤与闭源挑战。
- 2 Related Work：提示攻击与过滤绕过。
- 3 Method：风险片段识别与替换策略。
- 4 Experiments：闭源系统评测。
- 5 Discussion/Conclusion。
  **Prompt 3（核心创新点/贡献）**
  提出系统化的风险片段替换与多模态提示生成策略，实现闭源绕过。
  **Prompt 4（关键词 5-10 个）**
- surrogate prompt：替换式提示。
- safety filter：安全过滤。
- black-box：黑盒绕过。
- prompt substitution：提示替换。
- closed-source T2I：闭源模型。
- jailbreak：越狱。

**阶段二：深入细节与理解内容**
**Prompt 5（引言解读）**

- 试图解决的问题：闭源 T2I 的安全过滤不可解释且难以评测。
- 背景与铺垫：过滤器主要基于敏感词/语义匹配。
- 研究问题：能否通过替换策略绕过过滤。
- 主要贡献：SurrogatePrompt 框架与黑盒评测。
  **Prompt 6（方法解读）**
- 方法类型：提示生成与黑盒评测。
- 执行方式：定位风险片段；用代理短语替换；结合 LLM 与图像-文本模块生成攻击提示。
- 关键变量/指标：绕过成功率、语义保持度。
- 局限性/偏差：替换库质量影响效果。
- 方法合理性：替换空间大，易于绕过关键字过滤。
  **Prompt 7（结果解读）**
- 主要结果：在多闭源系统上实现较高绕过率。
- 关键图表：成功率统计与示例图像。
- 研究问题对应：验证替换策略可绕过过滤。
- 出乎意料发现：部分系统对语义一致性更敏感。
- 统计显著性：未突出显著性检验。
  **Prompt 8（讨论解读）**
- 解释：过滤器依赖词面匹配，替换可规避。
- 比较：比单纯同义替换更隐蔽。
- 重要意义：闭源系统安全评测需要多模态策略。
- 局限性：替换策略需持续更新。
- 未来方向：结合感知与语义双重检测。
  **Prompt 9（结论解读）**
- 主要发现：替换式越狱对闭源系统有效。
- 贡献重申：SurrogatePrompt 框架。
- 呼应研究问题：证明过滤器可被绕过。
- 未来研究：更强防御与风险检测。

**阶段三：批判性思考与深入理解**
**Prompt 10（方法批判性评估）**

- 优点：黑盒适用、实现简单。
- 局限：对替换词库依赖较大。
- 替代方法：基于优化的自动提示搜索。
  **Prompt 11（结果可信度评估）**
- 可信度：多平台实验增强可信度。
- 是否支持结论：支持绕过有效性。
- 其他解释：过滤器阈值差异影响表现。
  **Prompt 12（创新性评估）**
- 创新点：替换式越狱 + 多模态生成。
- 价值：为闭源安全评测提供工具。
- 相对贡献：扩展提示攻击方式。
  **Prompt 13（逻辑结构与论证）**
- 论证清晰：问题→方法→评测。
- 衔接良好：替换策略与实验一致。
- 薄弱点：替换策略泛化性需进一步验证。
  **Prompt 14（优缺点总结）**
- 优点：易部署、适配闭源。
- 缺点：替换库维护成本。
- 综合评价：黑盒提示攻击的实用方案。

## 5. On the Feasibility of Poisoning Text-to-Image AI Models via Adversarial Mislabeling

Wu, S., Bhaskar, R., Ha, A. Y. J., Shan, S., Zheng, H., & Zhao, B. Y. (2025). CCS 2025 (CCF A).
PDF: [UChicago](https://people.cs.uchicago.edu/~ravenben/publications/abstracts/amp-ccs25.html)

[Image]

**阶段一：快速概览与初步印象**
**Prompt 1（三句话概括主题与目的）**
论文研究“误标注投毒”对 T2I 训练管线的破坏风险。方法通过对图像施加扰动诱导 VLM 生成错误描述，从而污染训练数据。实验展示在黑盒场景中也可造成显著影响。
**Prompt 2（章节结构与大致内容）**

- Abstract：误标注投毒框架概述。
- 1 Introduction：训练数据链路风险。
- 2 Related Work：投毒与数据安全。
- 3 Method：VLM 误标注与投毒管线。
- 4 Experiments：黑盒验证与影响评估。
- 5 Discussion/Conclusion。
  **Prompt 3（核心创新点/贡献）**
  提出通过 VLM 误标注进行投毒的训练数据攻击范式。
  **Prompt 4（关键词 5-10 个）**
- data poisoning：数据投毒。
- adversarial mislabeling：误标注。
- VLM captioning：视觉语言标注。
- training pipeline：训练管线。
- black-box：黑盒验证。
- text-to-image：文生图。

**阶段二：深入细节与理解内容**
**Prompt 5（引言解读）**

- 试图解决的问题：训练数据自动标注环节可被操纵。
- 背景与铺垫：T2I 训练依赖大规模图文对。
- 研究问题：误标注能否被利用进行投毒。
- 主要贡献：误标注投毒流程与实证评估。
  **Prompt 6（方法解读）**
- 方法类型：数据投毒攻击。
- 执行方式：对图像施加微扰诱导 VLM 产生错误描述；将“脏标签”注入训练数据。
- 关键变量/指标：投毒成功率、影响范围。
- 局限性/偏差：依赖标注模型的脆弱性。
- 方法合理性：真实训练管线大量依赖自动标注。
  **Prompt 7（结果解读）**
- 主要结果：少量投毒样本即可造成可观偏移。
- 关键图表：投毒比例与影响效果曲线。
- 研究问题对应：验证误标注投毒可行。
- 出乎意料发现：副作用可能波及非目标概念。
- 统计显著性：未强调显著性检验。
  **Prompt 8（讨论解读）**
- 解释：标注模型错误可被系统性放大。
- 比较：比传统投毒更隐蔽。
- 重要意义：训练数据链路需更强审计。
- 局限性：对标注模型鲁棒性依赖。
- 未来方向：鲁棒标注与数据校验。
  **Prompt 9（结论解读）**
- 主要发现：误标注投毒在黑盒场景可实施。
- 贡献重申：提出新型投毒威胁模型。
- 呼应研究问题：证明训练链路安全风险。
- 未来研究：更强防御与检测。

**阶段三：批判性思考与深入理解**
**Prompt 10（方法批判性评估）**

- 优点：攻击隐蔽、现实可行。
- 局限：依赖标注模型弱点。
- 替代方法：直接数据审计与去噪。
  **Prompt 11（结果可信度评估）**
- 可信度：多场景验证增强可信度。
- 是否支持结论：支持投毒可行性。
- 其他解释：训练设置差异可能影响效果。
  **Prompt 12（创新性评估）**
- 创新点：将 VLM 误标注引入投毒链路。
- 价值：提醒训练数据安全风险。
- 相对贡献：扩展投毒攻击范式。
  **Prompt 13（逻辑结构与论证）**
- 论证清晰：威胁→方法→实验。
- 衔接良好：管线假设与实验吻合。
- 薄弱点：防御评测仍需扩展。
  **Prompt 14（优缺点总结）**
- 优点：隐蔽性强，影响显著。
- 缺点：依赖标注模型脆弱性。
- 综合评价：训练链路安全的关键风险提示。

## 6. MMA-Diffusion: MultiModal Attack on Diffusion Models

Yang, Y., Gao, R., Wang, X., Ho, T.-Y., Xu, N., & Xu, Q. (2024). CVPR 2024 (CCF A).
PDF: [CVPR](https://openaccess.thecvf.com/content/CVPR2024/papers/Yang_MMA-Diffusion_MultiModal_Attack_on_Diffusion_Models_CVPR_2024_paper.pdf)

[Image]

**阶段一：快速概览与初步印象**
**Prompt 1（三句话概括主题与目的）**
提出 MMA-Diffusion，通过文本与图像双通道攻击绕过 prompt filter 与 safety checker。文本侧生成对抗提示，图像侧对编辑输入加入不可见扰动。实验在开源与在线服务上显示高 ASR，揭示多层防线脆弱性。
**Prompt 2（章节结构与大致内容）**

- 1 Introduction：多模态攻击动机。
- 2 Related Work：攻击与防御综述。
- 3 Method：威胁模型、文本/图像攻击与联合策略。
- 4 Experiments：开源/在线服务评测。
- 5 Ethical Considerations。
- 6 Conclusion。
  **Prompt 3（核心创新点/贡献）**
  首次系统性提出文本+图像双模态越狱框架，能同时绕过提示过滤与后置安全检测。
  **Prompt 4（关键词 5-10 个）**
- multimodal attack：文本+图像联合攻击。
- prompt filter：输入侧过滤。
- safety checker：输出侧检测。
- adversarial prompt：文本扰动。
- image perturbation：图像不可见扰动。
- ASR-4/ASR-1：越狱成功率指标。
- Midjourney/Leonardo/SD：评测对象。

**阶段二：深入细节与理解内容**
**Prompt 5（引言解读）**

- 问题：现有防线在多模态攻击下脆弱。
- 背景：T2I 广泛部署与 NSFW 风险。
- 研究问题：如何在文本与图像双防线下仍成功生成 NSFW。
- 贡献：双模态攻击 + 多系统实证。
  **Prompt 6（方法解读）**
- 方法类型：对抗提示生成 + 图像扰动优化。
- 执行方式：文本侧生成对抗提示绕过过滤；图像侧在编辑输入中加入不可见扰动以骗过 safety checker；两者组合形成联合攻击。
- 关键指标：ASR-4/ASR-1、bypass rate、FID。
- 局限性：攻击细节部分出于伦理被省略。
- 方法合理性：联合攻击覆盖输入与输出防线。
  **Prompt 7（结果解读）**
- 主要结果：在 Midjourney/Leonardo 等黑盒服务与 SD 上均取得高 ASR；图像侧攻击可达 ASR-4≈88.5%。
- 关键图表：Table 1/2 黑盒服务结果；Fig.7 多模态攻击示例。
- 研究问题对应：验证双防线可被同时绕过。
- 出乎意料发现：部分服务对政治/恐怖类防线较弱。
- 统计显著性：未强调显著性检验。
  **Prompt 8（讨论解读）**
- 解释：文本过滤和后置检测互补不足以抵御双模态攻击。
- 比较：相较单模态攻击更强。
- 意义：提示安全评测需覆盖多模态。
- 局限性：攻击细节公开受限。
- 未来方向：多模态联合防御机制。
  **Prompt 9（结论解读）**
- 主要发现：MMA-Diffusion 能高效绕过现有防线。
- 贡献重申：多模态攻击框架。
- 呼应研究问题：验证双通道攻击有效性。
- 展望：更强防御与评测。

**阶段三：批判性思考与深入理解**
**Prompt 10（方法批判性评估）**

- 优点：覆盖输入与输出双防线。
- 局限：对真实场景依赖编辑/图像输入。
- 替代方法：结合模型内安全对齐。
  **Prompt 11（结果可信度评估）**
- 可信度：多模型/多服务评测增强可信度。
- 支持结论：高 ASR 支撑。
- 其他解释：过滤器阈值差异可能影响结果。
  **Prompt 12（创新性评估）**
- 创新点：双模态协同攻击策略。
- 价值：揭示多层防线脆弱性。
- 相对贡献：拓展了 T2I 越狱威胁模型。
  **Prompt 13（逻辑结构与论证）**
- 结构清晰：方法→实验→伦理。
- 衔接：多模态攻击逻辑完整。
- 薄弱点：攻击细节隐去影响可复现性。
  **Prompt 14（优缺点总结）**
- 优点：多模态覆盖、实际威胁强。
- 缺点：细节受限与场景依赖。
- 综合评价：多模态越狱的代表性工作。

## 7. Black-Box Forgery Attacks on Semantic Watermarks for Diffusion Models

Müller, A., Lukovnikov, D., Thietke, J., Fischer, A., & Quiring, E. (2025). CVPR 2025 (CCF A).
PDF: [CVPR](https://openaccess.thecvf.com/content/CVPR2025/papers/Muller_Black-Box_Forgery_Attacks_on_Semantic_Watermarks_for_Diffusion_Models_CVPR_2025_paper.pdf)

[Image]

**阶段一：快速概览与初步印象**
**Prompt 1（三句话概括主题与目的）**
本文研究语义水印在黑盒条件下的伪造与移除风险。作者提出 Imprinting 与 Reprompting 两类攻击，可在无密钥条件下伪造水印。结果表明现有语义水印难以抵御黑盒伪造。
**Prompt 2（章节结构与大致内容）**

- Abstract：黑盒伪造威胁与两类攻击概述。
- 1 Introduction：语义水印动机与威胁模型。
- 2 Related Work：水印与对抗攻防。
- 3 Attack Methods：Imprinting/Reprompting 细节。
- 4 Experiments：对 Tree-Ring/Gaussian Shading 评测。
- 5 Discussion/Conclusion。
  **Prompt 3（核心创新点/贡献）**
  提出无需密钥与原模型访问的黑盒伪造攻击体系，并实证现有语义水印脆弱性。
  **Prompt 4（关键词 5-10 个）**
- semantic watermark：研究对象。
- black-box attack：威胁模型。
- forging：伪造水印。
- diffusion model：应用场景。
- Tree-Ring/Gaussian Shading：代表方法。
- robustness：水印鲁棒性。
- adversarial evaluation：安全评估。

**阶段二：深入细节与理解内容**
**Prompt 5（引言解读）**

- 试图解决的问题：语义水印是否能抵抗黑盒伪造。
- 背景与铺垫：水印用于归因/版权，但攻击面未充分评估。
- 研究问题：黑盒条件下能否伪造或移除水印。
- 主要贡献：两类攻击 + 跨水印方法验证。
  **Prompt 6（方法解读）**
- 方法类型：黑盒生成攻击。
- 执行方式：Imprinting 通过少量水印样本“模仿”水印特征；Reprompting 通过提示重构让输出伪装带水印；包含伪造与移除变体。
- 关键变量/指标：检测通过率、伪造成功率、图像质量。
- 局限性/偏差：依赖可获取水印样本。
- 方法合理性：现实黑盒场景常可获得少量样本。
  **Prompt 7（结果解读）**
- 主要结果：两类攻击对多种语义水印均有效。
- 关键图表：攻击流程示意与伪造成功率表。
- 研究问题对应：验证黑盒伪造可行。
- 出乎意料发现：阈值调节难以显著提升鲁棒性。
- 统计显著性：未强调显著性检验。
  **Prompt 8（讨论解读）**
- 解释：语义水印依赖可迁移特征，易被模仿。
- 比较：相较像素水印更易伪造。
- 重要意义：水印需更强抗伪造设计。
- 局限性：攻击条件基于少量水印样本。
- 未来方向：密钥化或模型内水印机制。
  **Prompt 9（结论解读）**
- 主要发现：语义水印存在黑盒伪造漏洞。
- 贡献重申：提出系统性黑盒攻击评测。
- 呼应研究问题：证明当前方法不安全。
- 未来研究：更强鲁棒水印设计。

**阶段三：批判性思考与深入理解**
**Prompt 10（方法批判性评估）**

- 优点：威胁模型现实，攻击实现简洁。
- 局限：需要水印样本获取假设。
- 替代方法：结合模型水印或密钥机制。
  **Prompt 11（结果可信度评估）**
- 可信度：跨方法验证增强可信度。
- 是否支持结论：支持语义水印脆弱性。
- 其他解释：检测器阈值选择可能影响结果。
  **Prompt 12（创新性评估）**
- 创新点：黑盒伪造视角系统化研究。
- 价值：提醒部署者水印风险。
- 相对贡献：补足水印安全评估空白。
  **Prompt 13（逻辑结构与论证）**
- 论证清晰：威胁→攻击→评测。
- 衔接良好：从理论到实证。
- 薄弱点：对更强防御的探索较少。
  **Prompt 14（优缺点总结）**
- 优点：揭示实际攻击面。
- 缺点：假设与评测范围有限。
- 综合评价：水印安全评估的重要工作。

## 8. Harnessing Frequency Spectrum Insights for Image Copyright Protection Against Diffusion Models (CoprGuard)

Zhenguang Liu, Chao Shuai, Shaojing Fan, Ziping Dong, Jinwu Hu, Zhongjie Ba, & Kui Ren. (2025). CVPR 2025 (CCF A).
PDF: [CVPR](https://openaccess.thecvf.com/content/CVPR2025/papers/Liu_Harnessing_Frequency_Spectrum_Insights_for_Image_Copyright_Protection_Against_Diffusion_CVPR_2025_paper.pdf)

[Image]

**阶段一：快速概览与初步印象**
**Prompt 1（三句话概括主题与目的）**
本文提出 CoprGuard，用频谱水印保护训练数据版权。作者观察到扩散模型会继承训练数据的频谱统计特性。该方法可在训练中追踪受保护数据，即便水印占比很小也可检测。
**Prompt 2（章节结构与大致内容）**

- Abstract：频谱继承与水印检测概述。
- 1 Introduction：版权保护动机。
- 2 Related Work：水印/溯源方法。
- 3 Method：频谱水印设计与检测。
- 4 Experiments：对抗扩散模型训练评测。
- 5 Discussion/Conclusion。
  **Prompt 3（核心创新点/贡献）**
  利用频谱统计继承特性提出训练数据水印检测方案。
  **Prompt 4（关键词 5-10 个）**
- copyright protection：版权保护。
- frequency spectrum：频域特征。
- watermarking：水印机制。
- diffusion models：训练对象。
- provenance detection：溯源检测。
- robustness：水印鲁棒性。
- low-percentage watermark：低占比水印。

**阶段二：深入细节与理解内容**
**Prompt 5（引言解读）**

- 试图解决的问题：训练数据被未经授权使用难以检测。
- 背景与铺垫：扩散模型训练数据规模巨大、版权争议。
- 研究问题：能否在训练输出中检测受保护数据的使用。
- 主要贡献：频谱水印方案与实证验证。
  **Prompt 6（方法解读）**
- 方法类型：频域水印设计。
- 执行方式：在训练数据频谱注入水印并追踪模型生成的频谱统计偏差。
- 关键变量/指标：检测准确率、鲁棒性、误报率。
- 局限性/偏差：假设频谱特性可稳定继承。
- 方法合理性：频域统计具有跨样本一致性。
  **Prompt 7（结果解读）**
- 主要结果：即便低占比水印也能检测训练使用。
- 关键图表：频谱统计与检测性能表。
- 研究问题对应：验证频谱继承用于版权保护。
- 出乎意料发现：扩散模型频谱继承强于预期。
- 统计显著性：未突出显著性检验。
  **Prompt 8（讨论解读）**
- 解释：扩散训练保留训练数据统计结构。
- 比较：与空间域水印相比更隐蔽。
- 重要意义：提供版权保护新工具。
- 局限性：对强数据增强或清洗的鲁棒性需验证。
- 未来方向：更强对抗训练场景。
  **Prompt 9（结论解读）**
- 主要发现：CoprGuard 可有效检测训练数据使用。
- 贡献重申：频谱水印+统计检测框架。
- 呼应研究问题：支持版权保护可行性。
- 未来研究：扩展到多模型与多数据源。

**阶段三：批判性思考与深入理解**
**Prompt 10（方法批判性评估）**

- 优点：隐蔽、低占比可检测。
- 局限：对强扰动的鲁棒性需验证。
- 替代方法：语义水印或模型级水印。
  **Prompt 11（结果可信度评估）**
- 可信度：多设置评测增强可信度。
- 是否支持结论：支持频谱检测有效。
- 其他解释：检测器阈值影响误报。
  **Prompt 12（创新性评估）**
- 创新点：频谱继承视角的版权保护。
- 价值：应对训练数据滥用争议。
- 相对贡献：补足水印/溯源工具链。
  **Prompt 13（逻辑结构与论证）**
- 论证清晰：动机→方法→验证。
- 衔接良好：频谱假设与实验对应。
- 薄弱点：跨模型泛化仍需验证。
  **Prompt 14（优缺点总结）**
- 优点：隐蔽、低占比有效。
- 缺点：对强噪声增强可能脆弱。
- 综合评价：版权检测的有价值方向。

## 9. Mind the Trojan Horse: Image Prompt Adapter Enabling Scalable and Deceptive Jailbreaking

Chen, J., Dong, J., & Xie, X. (2025). CVPR 2025 (CCF A).
PDF: [CVPR](https://openaccess.thecvf.com/content/CVPR2025/papers/Chen_Mind_the_Trojan_Horse_Image_Prompt_Adapter_Enabling_Scalable_and_CVPR_2025_paper.pdf)

[Image]

**阶段一：快速概览与初步印象**
**Prompt 1（三句话概括主题与目的）**
论文提出针对 IP-Adapter 的“木马式”越狱攻击，通过图像提示劫持生成结果。方法构造不可见图像扰动，使编码器输出对齐到不安全语义。实验表明在黑盒 IGS 服务中可规模化生效。
**Prompt 2（章节结构与大致内容）**

- Abstract：IP-Adapter 威胁与攻击概述。
- 1 Introduction：图像提示通道的安全风险。
- 2 Related Work：图像提示与对抗样本。
- 3 Method：图像空间对抗样本与劫持策略。
- 4 Experiments：黑盒评测与防御讨论。
- 5 Discussion/Conclusion。
  **Prompt 3（核心创新点/贡献）**
  提出面向 IP-Adapter 的图像空间劫持式越狱框架。
  **Prompt 4（关键词 5-10 个）**
- IP-Adapter：图像提示适配器。
- image prompt attack：图像提示攻击。
- adversarial example：对抗样本。
- black-box：黑盒服务。
- jailbreak：越狱。
- safety：安全过滤。

**阶段二：深入细节与理解内容**
**Prompt 5（引言解读）**

- 试图解决的问题：图像提示通道成为新的攻击面。
- 背景与铺垫：IGS 服务依赖开放编码器。
- 研究问题：能否通过图像提示实现规模化越狱。
- 主要贡献：IP-Adapter 劫持攻击与评测。
  **Prompt 6（方法解读）**
- 方法类型：对抗样本生成与黑盒攻击。
- 执行方式：构造图像空间对抗样本，使编码器输出偏向不安全语义；通过上传图像提示劫持生成。
- 关键变量/指标：越狱成功率、可感知性。
- 局限性/偏差：依赖编码器鲁棒性。
- 方法合理性：图像提示链路缺少安全约束。
  **Prompt 7（结果解读）**
- 主要结果：在黑盒场景可显著提升越狱成功率。
- 关键图表：攻击流程示意与成功率统计。
- 研究问题对应：验证图像提示劫持可行。
- 出乎意料发现：部分防御对图像通道无效。
- 统计显著性：未突出显著性检验。
  **Prompt 8（讨论解读）**
- 解释：图像编码器的脆弱性导致越狱。
- 比较：相较纯文本攻击更隐蔽。
- 重要意义：需要对图像提示引入安全过滤。
- 局限性：对编码器替换可能不稳健。
- 未来方向：编码器鲁棒化与联合过滤。
  **Prompt 9（结论解读）**
- 主要发现：图像提示通道可被用于大规模越狱。
- 贡献重申：IP-Adapter 木马式攻击框架。
- 呼应研究问题：证明黑盒劫持可行。
- 未来研究：多通道安全策略。

**阶段三：批判性思考与深入理解**
**Prompt 10（方法批判性评估）**

- 优点：现实场景适用、攻击面新颖。
- 局限：依赖特定编码器架构。
- 替代方法：结合文本与图像协同攻击。
  **Prompt 11（结果可信度评估）**
- 可信度：多服务评测增强可信度。
- 是否支持结论：支持攻击有效性。
- 其他解释：过滤策略差异可能影响结果。
  **Prompt 12（创新性评估）**
- 创新点：以 IP-Adapter 为目标的图像提示越狱。
- 价值：揭示新型攻击面。
- 相对贡献：扩展 T2I 安全威胁模型。
  **Prompt 13（逻辑结构与论证）**
- 论证清晰：威胁→方法→实验。
- 衔接良好：攻击假设与结果一致。
- 薄弱点：对更强防御的系统性评测有限。
  **Prompt 14（优缺点总结）**
- 优点：隐蔽、可规模化。
- 缺点：对编码器鲁棒性敏感。
- 综合评价：重要的图像提示安全研究。

## 10. OpenSDI: Spotting Diffusion-Generated Images in the Open World

Wang, Y., Huang, Z., & Hong, X. (2025). CVPR 2025 (CCF A).
PDF: [CVPR](https://openaccess.thecvf.com/content/CVPR2025/papers/Wang_OpenSDI_Spotting_Diffusion-Generated_Images_in_the_Open_World_CVPR_2025_paper.pdf)

[Image]

**阶段一：快速概览与初步印象**
**Prompt 1（三句话概括主题与目的）**
本文提出 OpenSDI 任务与数据集，用于开放世界扩散图像检测。作者构建 OpenSDID，并提出 SPM 框架与 MaskCLIP 以提升检测与定位。实验显示在跨模型与跨场景下具更强泛化。
**Prompt 2（章节结构与大致内容）**

- Abstract：开放世界检测任务概述。
- 1 Introduction：检测挑战与动机。
- 2 Related Work：生成检测与开放世界。
- 3 Dataset/Task：OpenSDID 与评测设置。
- 4 Method：SPM 与 MaskCLIP。
- 5 Experiments：检测与定位评测。
- 6 Conclusion。
  **Prompt 3（核心创新点/贡献）**
  提出开放世界检测基准 OpenSDID，并给出组合模型框架 SPM。
  **Prompt 4（关键词 5-10 个）**
- diffusion detection：扩散图像检测。
- open-world：开放世界设定。
- OpenSDID：新数据集。
- localization：伪造定位。
- foundation models：多模型融合。
- MaskCLIP：检测模块。
- generalization：跨模型泛化。

**阶段二：深入细节与理解内容**
**Prompt 5（引言解读）**

- 试图解决的问题：现有检测在开放世界泛化不足。
- 背景与铺垫：扩散模型生成在现实中多样化。
- 研究问题：如何构建开放世界检测基准与方法。
- 主要贡献：数据集 + SPM 框架 + MaskCLIP。
  **Prompt 6（方法解读）**
- 方法类型：数据集构建 + 检测框架。
- 执行方式：OpenSDID 包含多样生成与编辑；SPM 融合多基础模型；MaskCLIP 用于局部定位。
- 关键变量/指标：检测准确率、定位 IoU、跨模型泛化。
- 局限性/偏差：开放世界覆盖仍有限。
- 方法合理性：多模型融合提升泛化。
  **Prompt 7（结果解读）**
- 主要结果：在跨模型场景显著优于基线。
- 关键图表：检测/定位性能表与可视化。
- 研究问题对应：验证开放世界基准与方法有效。
- 出乎意料发现：部分编辑类型更难检测。
- 统计显著性：未突出显著性检验。
  **Prompt 8（讨论解读）**
- 解释：开放世界多样性要求更泛化的模型。
- 比较：单模型检测泛化较弱。
- 重要意义：推动实际应用检测能力。
- 局限性：对新型编辑的适应性仍需加强。
- 未来方向：持续更新数据与模型。
  **Prompt 9（结论解读）**
- 主要发现：OpenSDI 提供开放世界检测基准与方法。
- 贡献重申：OpenSDID + SPM + MaskCLIP。
- 呼应研究问题：验证开放世界检测可行。
- 未来研究：更丰富生成管线与更强鲁棒检测。

**阶段三：批判性思考与深入理解**
**Prompt 10（方法批判性评估）**

- 优点：数据集与方法配套完整。
- 局限：开放世界覆盖仍受限。
- 替代方法：基于频谱或物理一致性的检测。
  **Prompt 11（结果可信度评估）**
- 可信度：多基线对比增强可信度。
- 是否支持结论：支持开放世界优势。
- 其他解释：模型融合带来的参数规模提升可能影响公平对比。
  **Prompt 12（创新性评估）**
- 创新点：开放世界检测基准+多模型框架。
- 价值：贴近真实应用。
- 相对贡献：推动扩散检测评测升级。
  **Prompt 13（逻辑结构与论证）**
- 论证清晰：任务→数据→方法→结果。
- 衔接良好：数据与方法紧密对应。
- 薄弱点：持续更新机制尚需说明。
  **Prompt 14（优缺点总结）**
- 优点：泛化更强、评测体系完善。
- 缺点：开放世界定义仍可扩展。
- 综合评价：扩散检测的重要基准与方法。

## 11. Six-CD: Benchmarking Concept Removals for Text-to-image Diffusion Models

Ren, J., Chen, K., Cui, Y., Zeng, S., Liu, H., Xing, Y., Tang, J., & Lyu, L. (2025). CVPR 2025 (CCF A).
PDF: [CVPR](https://openaccess.thecvf.com/content/CVPR2025/papers/Ren_Six-CD_Benchmarking_Concept_Removals_for_Text-to-image_Diffusion_Models_CVPR_2025_paper.pdf)

[Image]

**阶段一：快速概览与初步印象**
**Prompt 1（三句话概括主题与目的）**
本文提出 Six-CD 基准，系统评测概念擦除方法。作者构建包含六类概念的数据集与测试提示，并提出新评测指标。结果揭示现有擦除方法在保真与泛化上存在明显不足。
**Prompt 2（章节结构与大致内容）**

- Abstract：基准与指标概述。
- 1 Introduction：概念擦除评测缺口。
- 2 Related Work：擦除与安全控制。
- 3 Benchmark：Six-CD 数据集与任务。
- 4 Metrics：in-prompt CLIP 等指标。
- 5 Experiments：多方法对比。
- 6 Conclusion。
  **Prompt 3（核心创新点/贡献）**
  构建标准化概念擦除基准并提供新指标，促进公平评测。
  **Prompt 4（关键词 5-10 个）**
- benchmark：评测基准。
- concept removal：概念擦除。
- diffusion model：生成模型。
- evaluation metrics：指标体系。
- in-prompt CLIP：新指标。
- robustness：泛化评测。

**阶段二：深入细节与理解内容**
**Prompt 5（引言解读）**

- 试图解决的问题：现有擦除评测缺乏统一标准。
- 背景与铺垫：方法多但评价不一致。
- 研究问题：如何构建可比的评测基准。
- 主要贡献：Six-CD + 新指标 + 实证评测。
  **Prompt 6（方法解读）**
- 方法类型：基准构建与评测。
- 执行方式：设计六类概念与提示；定义 in-prompt CLIP 衡量保真；对多方法统一评测。
- 关键变量/指标：擦除率、保真度、泛化。
- 局限性/偏差：概念覆盖仍有限。
- 方法合理性：统一评测提升可比性。
  **Prompt 7（结果解读）**
- 主要结果：多方法在保真与泛化上存在短板。
- 关键图表：多方法性能表与雷达图。
- 研究问题对应：揭示现有方法不足。
- 出乎意料发现：部分方法对特定概念明显退化。
- 统计显著性：未突出显著性检验。
  **Prompt 8（讨论解读）**
- 解释：擦除方法对概念类别敏感。
- 比较：基准提升评测可复现性。
- 重要意义：促进方法改进。
- 局限性：需扩展更多场景。
- 未来方向：开放世界概念与多模型评测。
  **Prompt 9（结论解读）**
- 主要发现：Six-CD 揭示方法局限。
- 贡献重申：基准与指标体系。
- 呼应研究问题：统一评测可行。
- 未来研究：更大规模基准。

**阶段三：批判性思考与深入理解**
**Prompt 10（方法批判性评估）**

- 优点：标准化与可复现性强。
- 局限：概念范围仍有限。
- 替代方法：真实用户提示评测。
  **Prompt 11（结果可信度评估）**
- 可信度：多方法对比提升可信度。
- 是否支持结论：支持“方法存在短板”。
- 其他解释：提示设计影响结果。
  **Prompt 12（创新性评估）**
- 创新点：新基准与指标。
- 价值：推动领域标准化。
- 相对贡献：填补评测空白。
  **Prompt 13（逻辑结构与论证）**
- 论证清晰：问题→基准→评测。
- 衔接良好：指标与结论一致。
- 薄弱点：与真实部署差距仍在。
  **Prompt 14（优缺点总结）**
- 优点：体系化评测、促进公平比较。
- 缺点：场景覆盖有限。
- 综合评价：概念擦除评测的重要基础设施。

## 12. T2ISafety: Benchmark for Assessing Fairness, Toxicity, and Privacy in Image Generation

Li, L., Shi, Z., Hu, X., Dong, B., Qin, Y., Liu, X., Sheng, L., & Shao, J. (2025). CVPR 2025 (CCF A).
PDF: [CVPR](https://openaccess.thecvf.com/content/CVPR2025/papers/Li_T2ISafety_Benchmark_for_Assessing_Fairness_Toxicity_and_Privacy_in_Image_CVPR_2025_paper.pdf)

[Image]

**阶段一：快速概览与初步印象**
**Prompt 1（三句话概括主题与目的）**
论文提出 T2ISafety 基准，用于系统评测文生图模型的公平性、毒性与隐私风险。构建多维提示与标注体系，并训练评测器进行自动化检测。结果揭示开源与闭源模型的安全短板。
**Prompt 2（章节结构与大致内容）**

- Abstract：安全基准与评测目标。
- 1 Introduction：安全评测缺口。
- 2 Related Work：安全/公平/隐私评测。
- 3 Benchmark：数据集与任务设计。
- 4 Experiments：多模型评测与分析。
- 5 Discussion/Conclusion。
  **Prompt 3（核心创新点/贡献）**
  构建覆盖公平性、毒性、隐私的统一 T2I 安全基准。
  **Prompt 4（关键词 5-10 个）**
- benchmark：基准。
- safety evaluation：安全评测。
- fairness：公平性。
- toxicity：毒性。
- privacy：隐私风险。
- text-to-image：文生图。

**阶段二：深入细节与理解内容**
**Prompt 5（引言解读）**

- 试图解决的问题：T2I 安全评测缺乏系统基准。
- 背景与铺垫：安全事件频发、模型差异明显。
- 研究问题：如何统一评测公平/毒性/隐私。
- 主要贡献：T2ISafety 基准与评测结果。
  **Prompt 6（方法解读）**
- 方法类型：基准构建与评测。
- 执行方式：构建提示集合与标注；训练检测器自动评测；多模型对比。
- 关键变量/指标：安全风险比例、子类分布、检测器性能。
- 局限性/偏差：提示覆盖仍有限。
- 方法合理性：统一基准便于横向比较。
  **Prompt 7（结果解读）**
- 主要结果：不同模型在公平/毒性/隐私上差异显著。
- 关键图表：风险分布统计表与对比图。
- 研究问题对应：揭示模型安全短板。
- 出乎意料发现：某些闭源模型在特定维度更脆弱。
- 统计显著性：未突出显著性检验。
  **Prompt 8（讨论解读）**
- 解释：数据与过滤策略差异导致风险不同。
- 比较：统一基准让差距更清晰。
- 重要意义：支持持续红队与安全回归。
- 局限性：评测器可能存在偏差。
- 未来方向：扩展提示覆盖与跨文化评测。
  **Prompt 9（结论解读）**
- 主要发现：T2ISafety 能系统揭示安全短板。
- 贡献重申：公平/毒性/隐私统一基准。
- 呼应研究问题：提供可复现评测框架。
- 未来研究：更广泛场景与动态评测。

**阶段三：批判性思考与深入理解**
**Prompt 10（方法批判性评估）**

- 优点：覆盖多维安全指标。
- 局限：评测器与提示集偏差。
- 替代方法：引入真实用户提示。
  **Prompt 11（结果可信度评估）**
- 可信度：多模型评测增强可信度。
- 是否支持结论：支持安全差异结论。
- 其他解释：模型版本差异可能影响结果。
  **Prompt 12（创新性评估）**
- 创新点：多维安全统一基准。
- 价值：利于模型安全比较与监管。
- 相对贡献：推进安全评测标准化。
  **Prompt 13（逻辑结构与论证）**
- 论证清晰：基准→评测→分析。
- 衔接良好：指标与分析一致。
- 薄弱点：跨文化公平性仍待加强。
  **Prompt 14（优缺点总结）**
- 优点：体系化评测，覆盖全面。
- 缺点：提示/检测器偏差。
- 综合评价：T2I 安全评测的重要基准。

## 13. PLA: Prompt Learning Attack against Text-to-Image Generative Models

Lyu, X., Liu, Y., Li, Y., & Xiao, B. (2025). ICCV 2025 (CCF A).
PDF: [ICCV](https://openaccess.thecvf.com/content/ICCV2025/papers/Lyu_PLA_Prompt_Learning_Attack_against_Text-to-Image_Generative_Models_ICCV_2025_paper.pdf)

[Image]

**阶段一：快速概览与初步印象**
**Prompt 1（三句话概括主题与目的）**
本文提出 PLA，通过提示学习实现黑盒越狱攻击。方法借助多模态相似性度量，在无模型梯度情况下优化提示。实验显示 PLA 能绕过提示过滤与安全检查。
**Prompt 2（章节结构与大致内容）**

- Abstract：提示学习攻击概述。
- 1 Introduction：越狱威胁与挑战。
- 2 Related Work：提示攻击与黑盒优化。
- 3 Method：多模态相似性指导的提示学习。
- 4 Experiments：多模型越狱评测。
- 5 Discussion/Conclusion。
  **Prompt 3（核心创新点/贡献）**
  提出利用多模态相似性进行黑盒提示学习的越狱框架。
  **Prompt 4（关键词 5-10 个）**
- prompt learning：提示学习。
- jailbreak：越狱攻击。
- black-box：黑盒优化。
- multimodal similarity：相似性度量。
- safety filter：安全过滤。
- text-to-image：生成对象。

**阶段二：深入细节与理解内容**
**Prompt 5（引言解读）**

- 试图解决的问题：闭源模型难以获得梯度进行攻击。
- 背景与铺垫：越狱攻击对安全构成风险。
- 研究问题：能否在黑盒条件下学习有效提示。
- 主要贡献：PLA 黑盒提示学习框架。
  **Prompt 6（方法解读）**
- 方法类型：黑盒优化与提示学习。
- 执行方式：以多模态相似性作为反馈，迭代优化提示以触发不当生成。
- 关键变量/指标：越狱成功率、查询成本、质量。
- 局限性/偏差：依赖相似性度量的可用性。
- 方法合理性：在无梯度条件下可实施。
  **Prompt 7（结果解读）**
- 主要结果：在多模型上取得较高越狱成功率。
- 关键图表：成功率对比表与示例。
- 研究问题对应：验证黑盒提示学习有效。
- 出乎意料发现：提示学习可迁移至多模型。
- 统计显著性：未突出显著性检验。
  **Prompt 8（讨论解读）**
- 解释：相似性信号提供有效优化方向。
- 比较：比随机提示更高效。
- 重要意义：提示过滤易被绕过。
- 局限性：查询成本可能较高。
- 未来方向：更强防御与查询限制策略。
  **Prompt 9（结论解读）**
- 主要发现：PLA 能在黑盒条件下越狱。
- 贡献重申：提示学习攻击框架。
- 呼应研究问题：黑盒越狱可行。
- 未来研究：结合防御评测。

**阶段三：批判性思考与深入理解**
**Prompt 10（方法批判性评估）**

- 优点：不依赖模型内部信息。
- 局限：优化成本可能高。
- 替代方法：基于代理模型的迁移攻击。
  **Prompt 11（结果可信度评估）**
- 可信度：多模型验证增强可信度。
- 是否支持结论：支持黑盒攻击有效。
- 其他解释：不同过滤策略可能影响成功率。
  **Prompt 12（创新性评估）**
- 创新点：黑盒提示学习思路。
- 价值：揭示安全风险。
- 相对贡献：扩展提示攻击方法库。
  **Prompt 13（逻辑结构与论证）**
- 论证清晰：威胁→方法→评测。
- 衔接良好：优化目标与结果一致。
- 薄弱点：防御建议较少。
  **Prompt 14（优缺点总结）**
- 优点：黑盒可用、迁移性强。
- 缺点：查询开销可能高。
- 综合评价：重要的提示攻击方法。

## 14. JailbreakDiffBench: A Comprehensive Benchmark for Jailbreaking Diffusion Models

Jin, X., Weng, Z., Guo, H., Yin, C., Cheng, S., Shen, G., & Zhang, X. (2025). ICCV 2025 (CCF A).
PDF: [ICCV](https://openaccess.thecvf.com/content/ICCV2025/papers/Jin_JailbreakDiffBench_A_Comprehensive_Benchmark_for_Jailbreaking_Diffusion_Models_ICCV_2025_paper.pdf)

[Image]

**阶段一：快速概览与初步印象**
**Prompt 1（三句话概括主题与目的）**
提出 JailbreakDiffBench，统一评测 T2I/T2V 安全防线与越狱攻击。基准包含高质量人工标注提示/图像数据与评测协议。实验系统比较不同提示与图像检测器并揭示现有防线弱点。
**Prompt 2（章节结构与大致内容）**

- 1 Introduction：安全评测缺乏统一基准。
- 2 Related Work。
- 3 Preliminaries：系统管线与指标定义。
- 4 JailbreakDiffBench：框架与数据集。
- 5–7 Experiments：提示/图像检测评测、攻击评估。
- 8 Conclusion。
  **Prompt 3（核心创新点/贡献）**
  构建统一的越狱评测基准与协议，覆盖提示检查、图像检查与攻击模块。
  **Prompt 4（关键词 5-10 个）**
- benchmark：统一评测基准。
- prompt checker / image checker：双阶段过滤评估。
- ASR / HASR / AlignS / F1：多维指标。
- human-annotated dataset：高质量标注集。
- jailbreak attacks：SneakyPrompt/MMA/PGJ 等。
- T2I/T2V：扩展到视频。

**阶段二：深入细节与理解内容**
**Prompt 5（引言解读）**

- 问题：缺乏统一标准评测导致安全评估碎片化。
- 背景：多种过滤机制与攻击并存。
- 研究问题：如何系统评测防线与攻击。
- 贡献：JailbreakDiffBench 数据集+协议。
  **Prompt 6（方法解读）**
- 方法类型：基准数据构建 + 评测协议。
- 执行方式：人工标注提示/图像；定义 prompt checker、image checker、judger、AlignS；评测多类检测器与攻击。
- 关键指标：ASR、HASR、F1、AlignS、TPR/FPR。
- 局限性：评测仍依赖自动判定器。
- 方法合理性：统一协议提升可比性。
  **Prompt 7（结果解读）**
- 主要结果：LLM-based prompt checker 与 GPT-4o 图像检测效果最佳；规则/传统分类器弱。
- 关键图表：Table 1/2 prompt checker 评测；Table 3 image checker。
- 研究问题对应：揭示过滤器漏洞与攻击效果。
- 出乎意料发现：高模型规模不一定降低误报。
- 统计显著性：未突出显著性检验。
  **Prompt 8（讨论解读）**
- 解释：不同过滤机制在不同内容类别差异显著。
- 比较：LLM-based 检测器最强但存在误报。
- 意义：需要统一基准评估与更稳健过滤。
- 局限性：检测器与攻击生态仍在快速变化。
- 未来方向：更强合成数据与多模态检测。
  **Prompt 9（结论解读）**
- 主要发现：现有防线存在系统漏洞。
- 贡献重申：发布基准与协议。
- 呼应研究问题：提供统一评测框架。
- 展望：持续更新基准。

**阶段三：批判性思考与深入理解**
**Prompt 10（方法批判性评估）**

- 优点：标准化、可复现。
- 局限：依赖自动检测器与内容覆盖有限。
- 替代方法：更多人评或真实提示分布。
  **Prompt 11（结果可信度评估）**
- 可信度：多检测器与多攻击对比。
- 支持结论：结果一致揭示漏洞。
- 其他解释：检测器阈值影响性能。
  **Prompt 12（创新性评估）**
- 创新点：统一评测协议+多指标体系。
- 价值：为安全研究提供公共基准。
- 相对贡献：弥补分散评测不足。
  **Prompt 13（逻辑结构与论证）**
- 结构清晰：协议→实验→分析。
- 衔接：结果与协议紧密对应。
- 薄弱点：对新型攻击需持续更新。
  **Prompt 14（优缺点总结）**
- 优点：系统化评测。
- 缺点：生态快速变化需持续维护。
- 综合评价：基准类基础工作。

## 15. AutoPrompt: Automated Red-Teaming of Text-to-Image Models via LLM-Driven Adversarial Prompts

Liu, Y., Zhang, W., Chen, H., Wang, L., Jia, X., Lin, Z., & Wang, W. (2025). ICCV 2025 (CCF A).
PDF: [ICCV](https://openaccess.thecvf.com/content/ICCV2025/papers/Liu_AutoPrompt_Automated_Red-Teaming_of_Text-to-Image_Models_via_LLM-Driven_Adversarial_Prompts_ICCV_2025_paper.pdf)

[Image]

**阶段一：快速概览与初步印象**
**Prompt 1（三句话概括主题与目的）**
论文提出 AutoPrompt，利用 LLM 自动生成对抗提示进行红队测试。方法以黑盒模型反馈为信号，持续迭代优化提示。实验显示能高效发现安全漏洞。
**Prompt 2（章节结构与大致内容）**

- Abstract：AutoPrompt 概述。
- 1 Introduction：红队自动化需求。
- 2 Related Work：提示攻击与安全评测。
- 3 Method：LLM 驱动的对抗提示生成与迭代。
- 4 Experiments：黑盒评测与对比。
- 5 Discussion/Conclusion。
  **Prompt 3（核心创新点/贡献）**
  将 LLM 驱动的提示生成与黑盒反馈结合，自动化红队。
  **Prompt 4（关键词 5-10 个）**
- red teaming：红队测试。
- adversarial prompts：对抗提示。
- LLM-driven：大模型驱动。
- black-box：黑盒评测。
- safety evaluation：安全评测。
- text-to-image：文生图。

**阶段二：深入细节与理解内容**
**Prompt 5（引言解读）**

- 试图解决的问题：人工红队效率低、覆盖不足。
- 背景与铺垫：安全过滤不断演化。
- 研究问题：能否自动化生成有效攻击提示。
- 主要贡献：AutoPrompt 框架与评测。
  **Prompt 6（方法解读）**
- 方法类型：提示生成与黑盒优化。
- 执行方式：LLM 生成候选提示；根据模型输出反馈迭代优化；形成攻击提示库。
- 关键变量/指标：越狱成功率、查询成本。
- 局限性/偏差：依赖反馈信号与查询预算。
- 方法合理性：适合闭源黑盒场景。
  **Prompt 7（结果解读）**
- 主要结果：在多模型上显著提高漏洞发现效率。
- 关键图表：成功率与查询成本对比表。
- 研究问题对应：验证自动化红队可行。
- 出乎意料发现：提示可跨模型迁移。
- 统计显著性：未突出显著性检验。
  **Prompt 8（讨论解读）**
- 解释：LLM 能探索提示空间并强化成功模式。
- 比较：比人工或随机搜索更有效。
- 重要意义：降低红队成本。
- 局限性：可能引入偏置或重复。
- 未来方向：引入多模态反馈与防御对抗。
  **Prompt 9（结论解读）**
- 主要发现：AutoPrompt 提升黑盒红队效率。
- 贡献重申：LLM 驱动的自动化提示搜索。
- 呼应研究问题：自动化红队可行。
- 未来研究：更强对抗与鲁棒评测。

**阶段三：批判性思考与深入理解**
**Prompt 10（方法批判性评估）**

- 优点：自动化、高覆盖。
- 局限：依赖查询预算与反馈质量。
- 替代方法：结合代理模型的迁移攻击。
  **Prompt 11（结果可信度评估）**
- 可信度：多模型评测增强可信度。
- 是否支持结论：支持自动化有效性。
- 其他解释：不同过滤策略影响收益。
  **Prompt 12（创新性评估）**
- 创新点：LLM 驱动的红队自动化。
- 价值：实用性高。
- 相对贡献：推动安全评测工具化。
  **Prompt 13（逻辑结构与论证）**
- 论证清晰：问题→方法→验证。
- 衔接良好：反馈迭代与效果一致。
- 薄弱点：对抗防御考虑不足。
  **Prompt 14（优缺点总结）**
- 优点：效率高、易扩展。
- 缺点：预算与偏置限制。
- 综合评价：黑盒红队自动化的重要工作。

## 16. Automated Red Teaming for Text-to-Image Models through Feedback-Guided Prompt Iteration with Vision-Language Models

Xu, W., Chen, K., Qiu, J., Zhang, Y., Wang, R., Mao, J., Zhang, T., & Wang, L. (2025). ICCV 2025 (CCF A).
PDF: [ICCV](https://openaccess.thecvf.com/content/ICCV2025/papers/Xu_Automated_Red_Teaming_for_Text-to-Image_Models_through_Feedback-Guided_Prompt_Iteration_ICCV_2025_paper.pdf)

[Image]

**阶段一：快速概览与初步印象**
**Prompt 1（三句话概括主题与目的）**
论文提出 FGPI，通过视觉语言模型（VLM）反馈迭代优化攻击提示。方法面向黑盒 T2I 模型，持续改进越狱提示质量。实验表明在多个模型上提升越狱成功率。
**Prompt 2（章节结构与大致内容）**

- Abstract：反馈驱动提示迭代概述。
- 1 Introduction：自动化红队需求。
- 2 Related Work：提示攻击与红队评测。
- 3 Method：VLM 反馈与提示迭代。
- 4 Experiments：跨模型评测。
- 5 Discussion/Conclusion。
  **Prompt 3（核心创新点/贡献）**
  引入 VLM 反馈信号进行迭代式提示优化。
  **Prompt 4（关键词 5-10 个）**
- feedback-guided：反馈驱动。
- prompt iteration：提示迭代。
- VLM：视觉语言模型。
- black-box：黑盒评测。
- jailbreak：越狱。
- text-to-image：文生图。

**阶段二：深入细节与理解内容**
**Prompt 5（引言解读）**

- 试图解决的问题：提示搜索成本高、覆盖不足。
- 背景与铺垫：黑盒场景缺少梯度信息。
- 研究问题：能否用 VLM 反馈指导提示优化。
- 主要贡献：FGPI 框架与实验验证。
  **Prompt 6（方法解读）**
- 方法类型：黑盒提示优化。
- 执行方式：VLM 评价生成结果；根据反馈迭代更新提示；形成高效搜索。
- 关键变量/指标：成功率、迭代次数、查询成本。
- 局限性/偏差：依赖 VLM 反馈一致性。
- 方法合理性：适合黑盒且无需梯度。
  **Prompt 7（结果解读）**
- 主要结果：在多模型上提升越狱成功率。
- 关键图表：成功率与迭代效率对比表。
- 研究问题对应：验证反馈迭代有效。
- 出乎意料发现：提示可跨模型迁移。
- 统计显著性：未突出显著性检验。
  **Prompt 8（讨论解读）**
- 解释：VLM 反馈提供可用的优化信号。
- 比较：比随机/手工提示更稳定。
- 重要意义：提高红队自动化效率。
- 局限性：反馈噪声可能影响稳定性。
- 未来方向：引入多反馈器融合。
  **Prompt 9（结论解读）**
- 主要发现：FGPI 能有效提升越狱提示质量。
- 贡献重申：反馈驱动提示迭代框架。
- 呼应研究问题：黑盒优化可行。
- 未来研究：更强鲁棒红队方法。

**阶段三：批判性思考与深入理解**
**Prompt 10（方法批判性评估）**

- 优点：黑盒适用，效率高。
- 局限：反馈质量决定上限。
- 替代方法：代理模型优化或混合搜索。
  **Prompt 11（结果可信度评估）**
- 可信度：跨模型评测提升可信度。
- 是否支持结论：支持提升效果。
- 其他解释：基线选择影响对比。
  **Prompt 12（创新性评估）**
- 创新点：VLM 反馈驱动的提示迭代。
- 价值：实用红队工具。
- 相对贡献：自动化提示优化方案。
  **Prompt 13（逻辑结构与论证）**
- 论证清晰：框架→评测→分析。
- 衔接良好：反馈机制与结果一致。
- 薄弱点：对反馈噪声鲁棒性不足。
  **Prompt 14（优缺点总结）**
- 优点：高效、可迁移。
- 缺点：依赖反馈可靠性。
- 综合评价：黑盒红队的重要补充。

## 17. Prompting4Debugging: Red-Teaming Text-to-Image Diffusion Models by Finding Problematic Prompts

Chin, Z.-Y., Jiang, C.-M., Huang, C.-C., Chen, P.-Y., & Chiu, W.-C. (2024). ICML 2024 (CCF A).
PDF: [arXiv](https://arxiv.org/pdf/2309.06135)

[Image]

**阶段一：快速概览与初步印象**
**Prompt 1（三句话概括主题与目的）**
提出 P4D 作为自动化红队工具，用于测试带安全机制的 T2I 扩散模型。方法用 prompt engineering 优化提示以绕过安全机制，同时保持不安全概念。实验表明许多“安全”提示可被改写绕过，且约半数安全提示可被操纵成问题提示。
**Prompt 2（章节结构与大致内容）**

- Introduction：安全机制脆弱性与需求。
- Related Work/Background：扩散模型与安全机制。
- Methodology：P4D 框架与 prompt 优化。
- Experiments：I2P 与对象类数据集评测。
- Discussion/Conclusion：信息遮蔽与结论。
  **Prompt 3（核心创新点/贡献）**
  将 prompt engineering 系统化为 T2I 安全红队工具，自动发现安全机制的“问题提示”。
  **Prompt 4（关键词 5-10 个）**
- Prompting4Debugging (P4D)：自动红队工具。
- safety mechanisms：SLD/ESD/negative prompts。
- I2P dataset：不当提示评测集。
- failure rate (FR)：问题提示发现率。
- prompt engineering：软/硬提示优化。
- information obfuscation：安全机制带来假安全感。
- generalizability：问题提示跨模型迁移。

**阶段二：深入细节与理解内容**
**Prompt 5（引言解读）**

- 问题：安全机制能否抵御多样问题提示缺乏系统评测。
- 背景：SLD/ESD/负向提示等安全机制已部署。
- 研究问题：如何高效发现安全机制漏洞提示。
- 贡献：P4D 框架 + 实验揭示假安全感。
  **Prompt 6（方法解读）**
- 方法类型：prompt 优化（软提示→硬提示投影）。
- 执行方式：用标准 SD 生成不安全目标，P4D 优化提示使安全模型生成不安全内容；提出 P4D- 与 P4D+ 两种变体。
- 关键指标：Failure Rate (FR)；多分类检测器。
- 局限性：依赖检测器与特定数据集。
- 方法合理性：最小改动提示即可探测安全缺陷。
  **Prompt 7（结果解读）**
- 主要结果：I2P 等数据集中约半数安全提示可被改写绕过；P4D-UNION 提升失败率。
- 关键图表：Table 2/3 主结果；Fig.3 可视化。
- 研究问题对应：证明安全机制存在系统性漏洞。
- 出乎意料发现：安全机制在调试中可能造成“信息遮蔽”。
- 统计显著性：未突出显著性检验。
  **Prompt 8（讨论解读）**
- 解释：安全机制限制文本空间导致错误安全感。
- 比较：P4D 比人工发现更系统。
- 意义：促使安全机制需要更强评测。
- 局限性：评测集中在 I2P/特定对象类。
- 未来方向：更广泛概念与跨模型评测。
  **Prompt 9（结论解读）**
- 主要发现：P4D 能有效发现安全机制问题提示。
- 贡献重申：自动红队工具 + 实证漏洞。
- 呼应研究问题：证实现有安全评测不足。
- 展望：更系统防御与评测。

**阶段三：批判性思考与深入理解**
**Prompt 10（方法批判性评估）**

- 优点：自动化、可扩展。
- 局限：依赖检测器、评测集覆盖有限。
- 替代方法：加入人评或多模态检测。
  **Prompt 11（结果可信度评估）**
- 可信度：多安全模型/数据集测试增强可信度。
- 支持结论：FR 结果支持漏洞发现。
- 其他解释：检测器误差可能影响 FR。
  **Prompt 12（创新性评估）**
- 创新点：把 prompt engineering 系统化为红队工具。
- 价值：为安全评测提供新流程。
- 相对贡献：补足基准评测缺陷。
  **Prompt 13（逻辑结构与论证）**
- 结构清晰：方法→实验→讨论。
- 衔接：信息遮蔽解释实验现象。
- 薄弱点：部分结论需更多真实场景验证。
  **Prompt 14（优缺点总结）**
- 优点：自动化红队、揭示系统性漏洞。
- 缺点：评测覆盖有限。
- 综合评价：安全评测的重要工具。

## 18. Membership Inference on Text-to-Image Diffusion Models via Conditional Likelihood Discrepancy

Zhai, S., Chen, H., Dong, Y., Li, J., Shen, Q., Gao, Y., Su, H., & Liu, Y. (2024). NeurIPS 2024 (CCF A).
PDF: [NeurIPS](https://proceedings.neurips.cc/paper_files/paper/2024/file/874411a224a1934b80d499068384808b-Paper-Conference.pdf)

[Image]

**阶段一：快速概览与初步印象**
**Prompt 1（三句话概括主题与目的）**
论文提出 CLiD，用条件似然差异检测 T2I 扩散模型的成员泄露风险。方法通过统计差异判断样本是否参与训练。结果表明在多模型上具备可行的隐私审计能力。
**Prompt 2（章节结构与大致内容）**

- Abstract：成员推断与 CLiD 概述。
- 1 Introduction：T2I 隐私风险。
- 2 Related Work：成员推断与扩散模型。
- 3 Method：条件似然差异度量。
- 4 Experiments：攻击效果评测。
- 5 Discussion/Conclusion。
  **Prompt 3（核心创新点/贡献）**
  提出基于条件似然差异的扩散模型成员推断方法。
  **Prompt 4（关键词 5-10 个）**
- membership inference：成员推断。
- conditional likelihood：条件似然。
- diffusion model：扩散模型。
- privacy auditing：隐私审计。
- text-to-image：文生图。
- black-box/score access：黑盒或可查询似然。

**阶段二：深入细节与理解内容**
**Prompt 5（引言解读）**

- 试图解决的问题：T2I 训练数据隐私泄露。
- 背景与铺垫：扩散模型可记忆训练样本。
- 研究问题：如何有效检测成员泄露。
- 主要贡献：CLiD 指标与评测。
  **Prompt 6（方法解读）**
- 方法类型：统计推断/隐私攻击。
- 执行方式：比较目标样本与非成员样本的条件似然差异，用于判定成员身份。
- 关键变量/指标：攻击成功率、AUC 等。
- 局限性/偏差：需可估计似然或得分。
- 方法合理性：似然差异可反映记忆程度。
  **Prompt 7（结果解读）**
- 主要结果：对多模型具可行的成员推断能力。
- 关键图表：攻击成功率对比表。
- 研究问题对应：证明隐私泄露可被检测。
- 出乎意料发现：数据分布差异影响推断效果。
- 统计显著性：未突出显著性检验。
  **Prompt 8（讨论解读）**
- 解释：模型对训练样本的似然偏高。
- 比较：相较传统方法更适配扩散模型。
- 重要意义：提醒隐私风险与合规审计需求。
- 局限性：依赖似然估计精度。
- 未来方向：更强防御与隐私保护。
  **Prompt 9（结论解读）**
- 主要发现：CLiD 可用于 T2I 成员推断。
- 贡献重申：条件似然差异指标。
- 呼应研究问题：验证隐私审计可行。
- 未来研究：隐私防护与评测标准。

**阶段三：批判性思考与深入理解**
**Prompt 10（方法批判性评估）**

- 优点：适配扩散模型的审计方法。
- 局限：需要似然/得分访问能力。
- 替代方法：基于生成相似性或黑盒查询统计。
  **Prompt 11（结果可信度评估）**
- 可信度：多模型实验增强可信度。
- 是否支持结论：支持隐私风险存在。
- 其他解释：模型规模与数据多样性影响结果。
  **Prompt 12（创新性评估）**
- 创新点：条件似然差异度量。
- 价值：为隐私审计提供新工具。
- 相对贡献：扩展扩散模型隐私研究。
  **Prompt 13（逻辑结构与论证）**
- 论证清晰：问题→方法→实验。
- 衔接良好：度量与结果一致。
- 薄弱点：实际黑盒场景可用性需评估。
  **Prompt 14（优缺点总结）**
- 优点：针对性强，审计有效。
- 缺点：依赖似然估计。
- 综合评价：隐私评估的重要方法。

## 19. Red-Teaming Text-to-Image Systems by Rule-based Preference Modeling

Cao, Y., Miao, Y., Gao, X.-S., & Dong, Y. (2025). NeurIPS 2025 (CCF A).
PDF: [arXiv](https://arxiv.org/pdf/2505.21074)

[Image]

**阶段一：快速概览与初步印象**
**Prompt 1（三句话概括主题与目的）**
提出 RPG-RT，一种面向商业黑盒 T2I 系统的自适应红队框架。核心是用 LLM 反复改写提示词并基于系统反馈构造“规则化偏好”来微调 LLM。实验覆盖 19 个系统与多种在线 API，显示在未知防线下依旧显著提升越狱成功率。
**Prompt 2（章节结构与大致内容）**

- 1 Introduction：黑盒防线未知的问题与动机。
- 2 Methodology：黑盒设定、RPG-RT 三阶段流程与偏好建模。
- 3 Experiments：多系统与在线 API 评测、泛化与消融。
- 4 Conclusion：总结与展望。
  **Prompt 3（核心创新点/贡献）**
  用“规则化偏好建模 + DPO 微调”把粗粒度反馈转为可学习信号，使 LLM 在黑盒防线下自适应生成越狱提示。
  **Prompt 4（关键词 5-10 个）**
- red-teaming：面向安全评测的系统性攻击。
- black-box：未知防线与商业 API 场景。
- prompt modification：LLM 生成改写提示。
- preference modeling：规则化偏好排序。
- DPO：偏好优化用于 LLM 微调。
- ASR：越狱成功率指标。
- semantic similarity / PPL / FID：质量与隐蔽性指标。

**阶段二：深入细节与理解内容**
**Prompt 5（引言解读）**

- 问题：商业黑盒系统防线未知，现有红队方法依赖防线假设。
- 背景：文本过滤/安全对齐/后过滤并存，攻击需要适应未知组合。
- 研究问题：如何在黑盒反馈下自适应生成有效越狱提示。
- 贡献：RPG-RT 框架 + 规则化偏好建模 + 多系统实证。
  **Prompt 6（方法解读）**
- 方法类型：黑盒交互 + 规则化偏好学习 + DPO 微调。
- 执行方式：阶段1 LLM 生成多版本改写并查询系统；阶段2 用 NSFW/拒绝等反馈构造偏好排序并打分（结合有害性与语义相似）；阶段3 用 DPO 微调 LLM。
- 关键变量/指标：ASR、语义相似度、PPL、FID。
- 局限性：训练阶段需要查询成本与反馈信号。
- 方法合理性：把粗标签反馈转成可学习偏好，适配未知防线。
  **Prompt 7（结果解读）**
- 主要结果：在 19 个系统上 ASR 显著高于基线；在 DALL·E 3/Leonardo/SDXL 上至少 2× 提升，DALL·E 3 ASR≈31.33%。
- 关键图表：Fig.1 框架；Table 4 在线 API 结果；Table 8 消融。
- 研究问题对应：证明可在未知防线黑盒场景自适应。
- 出乎意料发现：对未见提示仍可高效泛化。
- 统计显著性：未强调显著性检验，但结果差距大。
  **Prompt 8（讨论解读）**
- 解释：规则化偏好让 LLM 学习系统“偏好”，提升探索效率。
- 与既有研究比较：不依赖防线假设，适配商业 API。
- 重要意义：为现实黑盒红队提供实用方法。
- 局限性：训练需较多查询与反馈成本。
- 未来方向：降低查询成本与更强安全对抗。
  **Prompt 9（结论解读）**
- 主要发现：RPG-RT 能在黑盒防线下稳定越狱。
- 贡献重申：规则化偏好 + DPO 的红队框架。
- 呼应研究问题：适应未知防线有效。
- 展望：扩展到更多模型与防线组合。

**阶段三：批判性思考与深入理解**
**Prompt 10（方法批判性评估）**

- 优点：黑盒自适应、无需防线假设。
- 局限：训练阶段成本高，对反馈质量敏感。
- 替代方法：结合迁移学习或更强的反馈压缩。
  **Prompt 11（结果可信度评估）**
- 可信度：多系统、多 API 评测增强可信度。
- 支持结论：ASR 大幅提升支撑主张。
- 其他解释：不同系统实现差异可能影响 ASR。
  **Prompt 12（创新性评估）**
- 创新点：规则化偏好建模用于黑盒红队。
- 价值：适合商业 API 安全评测与持续红队。
- 相对贡献：相比依赖防线假设方法更通用。
  **Prompt 13（逻辑结构与论证）**
- 结构清晰：问题→框架→实验→消融。
- 衔接：偏好建模直接解释性能提升。
- 潜在薄弱：对反馈噪声敏感性需更多分析。
  **Prompt 14（优缺点总结）**
- 优点：黑盒可用、适配未知防线。
- 缺点：训练查询成本与可扩展性。
- 综合评价：强实用红队框架。

## 20. SneakyPrompt: Jailbreaking Text-to-image Generative Models

Yang, Y., Hui, B., Yuan, H., Gong, N., & Cao, Y. (2024). IEEE S&P 2024 (CCF A).
PDF: [arXiv](https://arxiv.org/pdf/2305.12082)

[Image]

**阶段一：快速概览与初步印象**
**Prompt 1（三句话概括主题与目的）**
提出 SneakyPrompt，一种基于强化学习的黑盒提示扰动框架，用于绕过 T2I 安全过滤。方法针对被拦截提示词进行 token 替换，兼顾语义相似与绕过成功。实验在 DALL·E 2 与 Stable Diffusion 多种过滤器上验证有效且查询较少。
**Prompt 2（章节结构与大致内容）**

- 1 Introduction：安全过滤脆弱性与动机。
- 2 Related Work/Preliminaries：T2I 与文本对抗。
- 3 Method：RL 搜索与奖励设计。
- 4 Algorithm & Optimization：搜索空间与策略。
- 5 Experimental Setup。
- 6 Results（RQ1–RQ4）。
- 7 Conclusion/Discussion/Future Work。
  **Prompt 3（核心创新点/贡献）**
  将 RL 引入提示扰动，使黑盒安全过滤绕过更高效且保持语义。
  **Prompt 4（关键词 5-10 个）**
- reinforcement learning：基于奖励的提示搜索。
- prompt perturbation：词级替换绕过。
- black-box safety filter：闭源过滤器。
- bypass rate：绕过成功率。
- semantic similarity / FID：语义与质量指标。
- DALL·E 2 / Stable Diffusion：评测模型。
- query efficiency：低查询成本。

**阶段二：深入细节与理解内容**
**Prompt 5（引言解读）**

- 问题：现有文本攻击难以绕过真实安全过滤器。
- 背景：T2I 广泛部署带来 NSFW 风险。
- 研究问题：如何高效搜索语义一致的绕过提示。
- 贡献：SneakyPrompt + RL 搜索策略。
  **Prompt 6（方法解读）**
- 方法类型：黑盒 RL + prompt 替换搜索。
- 执行方式：对被拦截 prompt 中敏感 token 进行替换；奖励由绕过成功与语义相似度组成；提出搜索空间扩展与早停策略。
- 关键指标：bypass rate、FID、query 数。
- 局限性：依赖查询预算与奖励函数。
- 方法合理性：RL 能自适应发现有效替换。
  **Prompt 7（结果解读）**
- 主要结果：成功越狱 DALL·E 2 与多种 SD 过滤器；RL 方案在 bypass rate 与 query 成本上优于 baselines。
- 关键图表：RQ1–RQ4 结果表与参数影响图。
- 研究问题对应：证明黑盒过滤可被高效绕过。
- 出乎意料发现：reuse 攻击仍能保持较高 bypass。
- 统计显著性：未重点报告显著性检验。
  **Prompt 8（讨论解读）**
- 解释：奖励驱动替换使过滤器输出转为“安全”。
- 比较：优于传统文本对抗基线。
- 意义：安全过滤不足需更强防线。
- 局限性：依赖访问模型进行多次查询。
- 未来方向：模型级概念擦除等内置防御。
  **Prompt 9（结论解读）**
- 主要发现：黑盒过滤器可被 RL prompt 攻击绕过。
- 贡献重申：自动化攻击框架。
- 呼应问题：证明安全过滤脆弱性。
- 展望：更强安全机制与评测。

**阶段三：批判性思考与深入理解**
**Prompt 10（方法批判性评估）**

- 优点：高效、无需模型内部信息。
- 局限：查询成本仍较高；对过滤器变化敏感。
- 替代方法：结合迁移攻击或低查询优化。
  **Prompt 11（结果可信度评估）**
- 可信度：多过滤器与 DALL·E 2 评测增强可信度。
- 支持结论：结果显示明显提升。
- 其他解释：过滤器阈值设置影响 bypass。
  **Prompt 12（创新性评估）**
- 创新点：RL 直接优化绕过与语义保真。
- 价值：揭示实际过滤器脆弱性。
- 相对贡献：较传统文本对抗更有效。
  **Prompt 13（逻辑结构与论证）**
- 结构清晰：方法→评测→RQ 分析。
- 衔接：RQ 逐步验证关键点。
- 薄弱点：对真实用户提示分布的覆盖有限。
  **Prompt 14（优缺点总结）**
- 优点：黑盒适配、效果显著。
- 缺点：查询成本与鲁棒性仍待提升。
- 综合评价：越狱方法的重要基线。

## 21. Nightshade: Prompt-Specific Poisoning Attacks on Text-to-Image Generative Models

Shan, S., Ding, W., Passananti, J., Wu, S., Zheng, H., & Zhao, B. Y. (2024). IEEE S&P 2024 (CCF A).
PDF: [UChicago](https://people.cs.uchicago.edu/~ravenben/publications/abstracts/nightshade-oakland24.html)

[Image]

**阶段一：快速概览与初步印象**
**Prompt 1（三句话概括主题与目的）**
论文提出 Nightshade，用少量“定向投毒”样本破坏 T2I 模型的特定概念。攻击针对特定提示词或概念，诱导模型产生系统性偏移。实验显示即便投毒比例很小也可显著影响输出。
**Prompt 2（章节结构与大致内容）**

- Abstract：定向投毒攻击概述。
- 1 Introduction：数据投毒与艺术保护背景。
- 2 Related Work：投毒与模型鲁棒性。
- 3 Method：prompt-specific poisoning 设计。
- 4 Experiments：影响与鲁棒性评测。
- 5 Discussion/Conclusion。
  **Prompt 3（核心创新点/贡献）**
  提出“概念级小比例投毒”攻击范式。
  **Prompt 4（关键词 5-10 个）**
- prompt-specific poisoning：定向投毒。
- data poisoning：数据投毒。
- text-to-image：文生图。
- concept shift：概念偏移。
- robustness：鲁棒性。
- black-box：黑盒训练链路。

**阶段二：深入细节与理解内容**
**Prompt 5（引言解读）**

- 试图解决的问题：模型可被少量数据操纵。
- 背景与铺垫：训练数据来源开放且难审计。
- 研究问题：小比例投毒是否足以改变概念输出。
- 主要贡献：Nightshade 攻击设计与验证。
  **Prompt 6（方法解读）**
- 方法类型：训练数据投毒。
- 执行方式：针对特定概念制作优化样本注入训练；诱导模型在目标概念上产生错误输出。
- 关键变量/指标：投毒比例、偏移程度。
- 局限性/偏差：对训练流程控制程度影响效果。
- 方法合理性：训练数据对概念学习高度敏感。
  **Prompt 7（结果解读）**
- 主要结果：低比例投毒即可造成显著概念偏移。
- 关键图表：投毒比例与偏移程度曲线。
- 研究问题对应：验证概念级投毒可行。
- 出乎意料发现：部分概念更易被投毒。
- 统计显著性：未突出显著性检验。
  **Prompt 8（讨论解读）**
- 解释：模型对训练样本高度依赖。
- 比较：比全局投毒更隐蔽。
- 重要意义：训练数据链路需审计。
- 局限性：对防御策略的评估有限。
- 未来方向：数据清洗与鲁棒训练。
  **Prompt 9（结论解读）**
- 主要发现：Nightshade 可实现概念级投毒。
- 贡献重申：小比例定向投毒框架。
- 呼应研究问题：证明训练链路风险。
- 未来研究：更强防御与检测。

**阶段三：批判性思考与深入理解**
**Prompt 10（方法批判性评估）**

- 优点：隐蔽、现实威胁大。
- 局限：依赖训练数据可注入性。
- 替代方法：强审计与鲁棒学习。
  **Prompt 11（结果可信度评估）**
- 可信度：多概念实验增强可信度。
- 是否支持结论：支持投毒有效性。
- 其他解释：训练规模可能影响敏感性。
  **Prompt 12（创新性评估）**
- 创新点：概念级小比例投毒。
- 价值：提示训练数据安全风险。
- 相对贡献：扩展投毒威胁模型。
  **Prompt 13（逻辑结构与论证）**
- 论证清晰：威胁→方法→实验。
- 衔接良好：投毒机制与结果一致。
- 薄弱点：防御评测不足。
  **Prompt 14（优缺点总结）**
- 优点：攻击隐蔽、影响大。
- 缺点：依赖数据可控性。
- 综合评价：训练安全的重要警示。

## 22. Fuzz-Testing Meets LLM-Based Agents: An Automated and Efficient Framework for Jailbreaking Text-To-Image Generation Models

Dong, Y., Meng, X., Yu, N., Li, Z., & Guo, S. (2025). IEEE S&P 2025 (CCF A).
PDF: [arXiv](https://arxiv.org/pdf/2408.00523)

[Image]

**阶段一：快速概览与初步印象**
**Prompt 1（三句话概括主题与目的）**
提出 JailFuzzer，将 fuzzing 与 LLM agents 结合自动生成自然语义的越狱提示。框架包含种子池、引导变异与 oracle，并用 LLM/VLM 辅助变异与判断。实验显示在黑盒设置中高 bypass 率且查询成本低。
**Prompt 2（章节结构与大致内容）**

- 1 Introduction：问题背景与现有方法局限。
- 2 Preliminaries/Related：T2I 与安全过滤综述。
- 3 Method：JailFuzzer 框架与三组件。
- 4 Experiments：多模型/多防线评测与对比。
- 5 Discussion/Defense：防御建议与局限。
- 6 Conclusion。
  **Prompt 3（核心创新点/贡献）**
  首次将 fuzzing 设计成 LLM/VLM 代理驱动的越狱搜索，兼顾语义自然性与黑盒高效率。
  **Prompt 4（关键词 5-10 个）**
- fuzzing：以变异探索漏洞。
- LLM/VLM agents：生成与评估提示。
- seed pool / mutation / oracle：核心组件。
- black-box jailbreak：黑盒越狱场景。
- bypass rate：绕过成功率。
- query budget：查询成本控制。
- safety filters：文本/图像/多模态过滤。

**阶段二：深入细节与理解内容**
**Prompt 5（引言解读）**

- 问题：现有越狱方法不自然、查询多、黑盒不可用。
- 背景：T2I 安全过滤广泛部署但可被越狱。
- 研究问题：如何高效生成自然越狱提示并降低查询成本。
- 贡献：JailFuzzer 框架 + LLM/VLM 代理 + 实验验证。
  **Prompt 6（方法解读）**
- 方法类型：黑盒 fuzzing + LLM/VLM 代理。
- 执行方式：种子池收集提示；变异代理生成语义一致变体；oracle 代理评估是否绕过并反馈；记忆模块保留成功经验。
- 关键指标：bypass rate、查询次数、语义相似度、FID。
- 局限性：依赖代理模型能力与安全对齐。
- 方法合理性：fuzzing 适合探索大搜索空间。
  **Prompt 7（结果解读）**
- 主要结果：多数防线 bypass 近 100%，平均约 4.6 次查询；保守防线仍有 >82% bypass。
- 关键图表：表格对比不同防线与 ablation；代理数量/记忆模块提升。
- 对应问题：证明高效、低查询越狱。
- 出乎意料：PPL/平滑防御仍难阻止。
- 统计显著性：未强调显著性检验。
  **Prompt 8（讨论解读）**
- 解释：LLM/VLM 代理提供语义自然变异能力。
- 比较：优于模板或规则化方法。
- 意义：为自动化红队提供高效工具。
- 局限性：模型能力与过滤反馈精度影响效果。
- 未来方向：更强安全对齐与训练期防御。
  **Prompt 9（结论解读）**
- 主要发现：JailFuzzer 兼顾高 ASR 与低查询。
- 贡献重申：fuzzing + LLM agents 结构。
- 呼应问题：自然提示+黑盒可用。
- 展望：防御机制设计与红队评测。

**阶段三：批判性思考与深入理解**
**Prompt 10（方法批判性评估）**

- 优点：高效、自然、黑盒适配。
- 局限：依赖代理模型与评测器准确性。
- 替代方法：加入对抗训练或多代理集成。
  **Prompt 11（结果可信度评估）**
- 可信度：多模型/多防线评测支撑结论。
- 支持结论：高 bypass 与低查询一致。
- 其他解释：过滤器阈值与判定器可能影响结果。
  **Prompt 12（创新性评估）**
- 创新点：LLM 代理驱动 fuzzing。
- 价值：实用的红队自动化工具。
- 相对贡献：相对 RL/模板方法更高效。
  **Prompt 13（逻辑结构与论证）**
- 结构清晰：框架→实验→消融。
- 衔接：消融验证关键模块。
- 薄弱点：防御对抗的深入分析有限。
  **Prompt 14（优缺点总结）**
- 优点：效率高、提示自然、黑盒适配。
- 缺点：依赖评测器与代理质量。
- 综合评价：越狱自动化的重要方法。

## 23. Towards Reliable Verification of Unauthorized Data Usage in Personalized Text-to-Image Diffusion Models

Li, B., Wei, Y., Fu, Y., Wang, Z., Li, Y., Zhang, J., Wang, R., & Zhang, T. (2025). IEEE S&P 2025 (CCF A).
PDF: [IEEE S&amp;P](https://doi.org/10.1109/SP61157.2025.00073)

[Image]

**阶段一：快速概览与初步印象**
**Prompt 1（三句话概括主题与目的）**
论文提出面向个性化 T2I 模型的数据使用验证方法。通过可学习涂层使数据在训练中留下可检测痕迹。黑盒验证可判断是否未经授权使用数据。
**Prompt 2（章节结构与大致内容）**

- Abstract：数据使用验证框架。
- 1 Introduction：个性化模型数据合规问题。
- 2 Related Work：水印/溯源方法。
- 3 Method：可学习涂层与验证流程。
- 4 Experiments：黑盒验证评测。
- 5 Discussion/Conclusion。
  **Prompt 3（核心创新点/贡献）**
  提出可学习涂层 + 黑盒验证的合规审计方法。
  **Prompt 4（关键词 5-10 个）**
- data usage verification：数据使用验证。
- personalization：个性化训练。
- learnable coating：可学习涂层。
- black-box audit：黑盒审计。
- provenance：溯源。

**阶段二：深入细节与理解内容**
**Prompt 5（引言解读）**

- 试图解决的问题：个性化模型可能非法使用数据。
- 背景与铺垫：缺乏可靠的黑盒验证工具。
- 研究问题：能否通过可学习涂层实现验证。
- 主要贡献：涂层设计 + 黑盒验证流程。
  **Prompt 6（方法解读）**
- 方法类型：数据标记与验证。
- 执行方式：设计可学习涂层使模型可学习该特征；在黑盒输出中检测统计差异。
- 关键变量/指标：检测率、误报率。
- 局限性/偏差：涂层鲁棒性与可学习性权衡。
- 方法合理性：训练期可学习特征更易被验证。
  **Prompt 7（结果解读）**
- 主要结果：在多模型/多场景验证有效性。
- 关键图表：检测率与鲁棒性对比表。
- 研究问题对应：证明黑盒验证可行。
- 出乎意料发现：部分防御/清洗仍可保留信号。
- 统计显著性：未突出显著性检验。
  **Prompt 8（讨论解读）**
- 解释：涂层被个性化训练学习并反映在输出。
- 比较：比传统水印更适配个性化场景。
- 重要意义：支持版权/合规审计。
- 局限性：对强对抗清洗可能脆弱。
- 未来方向：与水印/溯源融合。
  **Prompt 9（结论解读）**
- 主要发现：可学习涂层提升黑盒验证可靠性。
- 贡献重申：数据使用验证框架。
- 呼应研究问题：证明合规审计可行。
- 未来研究：对抗鲁棒性提升。

**阶段三：批判性思考与深入理解**
**Prompt 10（方法批判性评估）**

- 优点：黑盒可用，适合合规审计。
- 局限：涂层易被强清洗削弱。
- 替代方法：模型级水印或多源验证。
  **Prompt 11（结果可信度评估）**
- 可信度：多场景评测增强可信度。
- 是否支持结论：支持验证有效。
- 其他解释：模型训练策略影响检测率。
  **Prompt 12（创新性评估）**
- 创新点：可学习涂层用于使用验证。
- 价值：适配个性化训练链路。
- 相对贡献：拓展溯源审计方法。
  **Prompt 13（逻辑结构与论证）**
- 论证清晰：动机→方法→验证。
- 衔接良好：涂层设计与实验一致。
- 薄弱点：对抗场景评估有限。
  **Prompt 14（优缺点总结）**
- 优点：实用性强、黑盒可验证。
- 缺点：鲁棒性仍需提升。
- 综合评价：合规审计的重要工具。

## 24. Glaze: Protecting Artists from Style Mimicry by Text-to-Image Models

Shan, S., Cryan, J., Wenger, E., Zheng, H., Hanocka, R., & Zhao, B. Y. (2023). USENIX Security 2023 (CCF A).
PDF: [USENIX](https://www.usenix.org/system/files/usenixsecurity23-shan.pdf)

[Image]

**阶段一：快速概览与初步印象**
**Prompt 1（三句话概括主题与目的）**
论文提出 Glaze，为艺术家提供对抗风格模仿的保护工具。方法对作品加入微小扰动，使模型学习到错误风格特征。实验表明可显著降低风格模仿效果。
**Prompt 2（章节结构与大致内容）**

- Abstract：风格保护方法概述。
- 1 Introduction：艺术风格被模仿风险。
- 2 Related Work：对抗保护与生成模型。
- 3 Method：风格披风生成与训练期误导。
- 4 Experiments：保护效果与鲁棒性评测。
- 5 Discussion/Conclusion。
  **Prompt 3（核心创新点/贡献）**
  提出面向艺术风格保护的黑盒对抗扰动方案。
  **Prompt 4（关键词 5-10 个）**
- style mimicry：风格模仿。
- adversarial perturbation：对抗扰动。
- artist protection：艺术保护。
- training-time misleading：训练期误导。
- black-box defense：黑盒防护。

**阶段二：深入细节与理解内容**
**Prompt 5（引言解读）**

- 试图解决的问题：艺术家作品被模型模仿。
- 背景与铺垫：训练数据广泛采集。
- 研究问题：能否通过扰动保护风格。
- 主要贡献：Glaze 披风生成与评测。
  **Prompt 6（方法解读）**
- 方法类型：对抗扰动与数据防护。
- 执行方式：对图像加入人眼不可察觉扰动；在训练或微调中误导模型学习风格。
- 关键变量/指标：模仿成功率、视觉质量。
- 局限性/偏差：对强去噪可能脆弱。
- 方法合理性：对抗扰动可干扰模型特征学习。
  **Prompt 7（结果解读）**
- 主要结果：显著降低风格模仿效果。
- 关键图表：模仿效果对比图与量化结果。
- 研究问题对应：验证风格保护有效。
- 出乎意料发现：不同风格的保护强度不同。
- 统计显著性：未突出显著性检验。
  **Prompt 8（讨论解读）**
- 解释：扰动改变模型学习的风格特征。
- 比较：比仅依赖版权声明更有效。
- 重要意义：为艺术家提供实用工具。
- 局限性：对抗适配可能削弱效果。
- 未来方向：与水印或检测结合。
  **Prompt 9（结论解读）**
- 主要发现：Glaze 可保护艺术风格不被模仿。
- 贡献重申：黑盒对抗扰动方案。
- 呼应研究问题：证明可用的风格保护。
- 未来研究：更强鲁棒性与可用性。

**阶段三：批判性思考与深入理解**
**Prompt 10（方法批判性评估）**

- 优点：无需模型访问，部署简单。
- 局限：对强攻击仍可能失效。
- 替代方法：水印或训练数据审计。
  **Prompt 11（结果可信度评估）**
- 可信度：多风格实验增强可信度。
- 是否支持结论：支持保护有效。
- 其他解释：模型差异影响效果。
  **Prompt 12（创新性评估）**
- 创新点：面向艺术风格的对抗保护。
- 价值：保护创作者权益。
- 相对贡献：推动防模仿研究。
  **Prompt 13（逻辑结构与论证）**
- 论证清晰：问题→方法→实验。
- 衔接良好：扰动设计与效果一致。
- 薄弱点：强对抗场景评估不足。
  **Prompt 14（优缺点总结）**
- 优点：黑盒可用、效果明显。
- 缺点：鲁棒性仍需提升。
- 综合评价：艺术家防护的代表性工作。

## 25. Prompt Stealing Attacks Against Text-to-Image Generation Models

Shen, X., Qu, Y., Backes, M., & Zhang, Y. (2024). USENIX Security 2024 (CCF A).
PDF: [USENIX](https://www.usenix.org/system/files/usenixsecurity24-shen-xinyue.pdf)

[Image]

**阶段一：快速概览与初步印象**
**Prompt 1（三句话概括主题与目的）**
论文首次系统研究 T2I 提示词窃取攻击。方法仅基于生成图像恢复高价值提示词，包含主体与修饰词推断。实验表明在无模型访问下也能有效重建提示。
**Prompt 2（章节结构与大致内容）**

- Abstract：提示词窃取问题概述。
- 1 Introduction：提示资产化与风险。
- 2 Related Work：提示逆向与图像理解。
- 3 Method：PromptStealer 组件设计。
- 4 Experiments：重建质量评测。
- 5 Discussion/Conclusion。
  **Prompt 3（核心创新点/贡献）**
  提出图像驱动的提示词重建框架 PromptStealer。
  **Prompt 4（关键词 5-10 个）**
- prompt stealing：提示词窃取。
- prompt reconstruction：提示重建。
- black-box：黑盒攻击。
- text-to-image：文生图。
- asset protection：资产保护。

**阶段二：深入细节与理解内容**
**Prompt 5（引言解读）**

- 试图解决的问题：提示词作为资产易被窃取。
- 背景与铺垫：生成图像可泄露提示信息。
- 研究问题：能否仅凭图像恢复提示。
- 主要贡献：PromptStealer 框架与评测。
  **Prompt 6（方法解读）**
- 方法类型：图像到提示逆向。
- 执行方式：主体生成器识别主体；修饰词检测器提取风格/修饰信息；组合形成提示。
- 关键变量/指标：重建准确率、语义一致度。
- 局限性/偏差：对复杂提示的恢复可能有限。
- 方法合理性：图像中蕴含提示结构信息。
  **Prompt 7（结果解读）**
- 主要结果：可恢复高价值提示并具较高相似度。
- 关键图表：重建质量与成功率统计。
- 研究问题对应：证明黑盒提示窃取可行。
- 出乎意料发现：某些修饰词更易恢复。
- 统计显著性：未突出显著性检验。
  **Prompt 8（讨论解读）**
- 解释：生成图像泄露提示结构。
- 比较：比单纯图像描述更精准。
- 重要意义：提示资产安全需加强保护。
- 局限性：对高度抽象提示仍有难度。
- 未来方向：提示水印与混淆机制。
  **Prompt 9（结论解读）**
- 主要发现：提示词可被黑盒窃取。
- 贡献重申：PromptStealer 框架。
- 呼应研究问题：证实提示泄露风险。
- 未来研究：防护与隐私保护。

**阶段三：批判性思考与深入理解**
**Prompt 10（方法批判性评估）**

- 优点：无需模型访问，现实可行。
- 局限：复杂提示恢复仍有限。
- 替代方法：结合多视角反演。
  **Prompt 11（结果可信度评估）**
- 可信度：多模型评测增强可信度。
- 是否支持结论：支持提示可被恢复。
- 其他解释：图像质量影响恢复效果。
  **Prompt 12（创新性评估）**
- 创新点：系统化提示窃取框架。
- 价值：揭示提示资产风险。
- 相对贡献：扩展隐私攻击研究。
  **Prompt 13（逻辑结构与论证）**
- 论证清晰：威胁→方法→评测。
- 衔接良好：模块设计与结果一致。
- 薄弱点：防御方案评估较少。
  **Prompt 14（优缺点总结）**
- 优点：黑盒可用，影响大。
- 缺点：对复杂提示仍有限。
- 综合评价：提示安全的关键研究。

## 26. Cross-Modal Prompt Inversion: Unifying Threats to Text and Image Generative AI Models

Dayong Ye, Tianqing Zhu, Feng He, Bo Liu, Minhui Xue, & Wanlei Zhou. (2025). USENIX Security 2025 (CCF A).
PDF: [USENIX](https://www.usenix.org/system/files/usenixsecurity25-ye-inversion.pdf)

[Image]

**阶段一：快速概览与初步印象**
**Prompt 1（三句话概括主题与目的）**
论文提出跨模态提示词反演框架，统一文本与图像生成模型的提示窃取威胁。方法通过跨模态评测验证在不同生成模型上的有效性。结果表明提示词泄露在多模态场景普遍存在。
**Prompt 2（章节结构与大致内容）**

- Abstract：跨模态提示反演框架概述。
- 1 Introduction：提示泄露风险。
- 2 Related Work：提示反演与安全评测。
- 3 Method：跨模态反演流程。
- 4 Experiments：文本/图像模型评测。
- 5 Discussion/Conclusion。
  **Prompt 3（核心创新点/贡献）**
  提出统一的跨模态提示反演框架。
  **Prompt 4（关键词 5-10 个）**
- prompt inversion：提示反演。
- cross-modal：跨模态。
- generative models：生成模型。
- black-box：黑盒攻击。
- prompt leakage：提示泄露。

**阶段二：深入细节与理解内容**
**Prompt 5（引言解读）**

- 试图解决的问题：提示泄露威胁跨模态存在。
- 背景与铺垫：提示作为生成控制信号具价值。
- 研究问题：能否统一反演威胁模型。
- 主要贡献：跨模态反演框架与评测。
  **Prompt 6（方法解读）**
- 方法类型：提示反演与评测。
- 执行方式：构建反演流程，评估文本/图像生成模型的提示恢复能力。
- 关键变量/指标：恢复准确度、语义一致性。
- 局限性/偏差：对提示复杂度敏感。
- 方法合理性：跨模态统一评测更全面。
  **Prompt 7（结果解读）**
- 主要结果：提示泄露在多模态模型中普遍存在。
- 关键图表：恢复质量对比表。
- 研究问题对应：验证跨模态威胁。
- 出乎意料发现：部分模型更易被反演。
- 统计显著性：未突出显著性检验。
  **Prompt 8（讨论解读）**
- 解释：生成模型保留了提示线索。
- 比较：不同模态共享相似风险。
- 重要意义：需要统一提示保护机制。
- 局限性：对多轮提示仍需评估。
- 未来方向：提示水印与输出去敏。
  **Prompt 9（结论解读）**
- 主要发现：提示泄露是跨模态共性风险。
- 贡献重申：跨模态提示反演框架。
- 呼应研究问题：统一威胁模型可行。
- 未来研究：更强防护策略。

**阶段三：批判性思考与深入理解**
**Prompt 10（方法批判性评估）**

- 优点：统一多模态威胁视角。
- 局限：对复杂提示恢复仍有限。
- 替代方法：结合多轮交互反演。
  **Prompt 11（结果可信度评估）**
- 可信度：跨模态实验增强可信度。
- 是否支持结论：支持提示泄露普遍性。
- 其他解释：模型结构差异影响结果。
  **Prompt 12（创新性评估）**
- 创新点：跨模态统一提示反演。
- 价值：推动提示安全研究。
- 相对贡献：扩展提示泄露威胁模型。
  **Prompt 13（逻辑结构与论证）**
- 论证清晰：框架→评测→分析。
- 衔接良好：统一框架与结果一致。
- 薄弱点：防御评测不足。
  **Prompt 14（优缺点总结）**
- 优点：视角统一、适用范围广。
- 缺点：复杂提示恢复难度大。
- 综合评价：提示安全的重要研究。

## 27. Exposing the Guardrails: Reverse-Engineering and Jailbreaking Safety Filters in DALL·E Text-to-Image Pipelines

Corban Villa, Shujaat Mirza, & Christina Pöpper. (2025). USENIX Security 2025 (CCF A).
PDF: [USENIX](https://www.usenix.org/system/files/usenixsecurity25-villa.pdf)

[Image]

**阶段一：快速概览与初步印象**
**Prompt 1（三句话概括主题与目的）**
论文系统逆向 DALL·E 安全管线，通过时间侧信道与行为差异推断过滤结构。基于逆向结果构造多类越狱提示。研究揭示多组件安全边界不一致的问题。
**Prompt 2（章节结构与大致内容）**

- Abstract：安全管线逆向与越狱概述。
- 1 Introduction：安全管线复杂性。
- 2 Related Work：过滤机制与越狱攻击。
- 3 Method：侧信道分析与逆向流程。
- 4 Experiments：越狱评测与分析。
- 5 Discussion/Conclusion。
  **Prompt 3（核心创新点/贡献）**
  提出针对 DALL·E 的系统性安全管线逆向与越狱方法。
  **Prompt 4（关键词 5-10 个）**
- guardrail reverse engineering：管线逆向。
- timing side-channel：时间侧信道。
- jailbreak：越狱攻击。
- DALL·E：闭源系统。
- safety filter：安全过滤。

**阶段二：深入细节与理解内容**
**Prompt 5（引言解读）**

- 试图解决的问题：安全过滤机制不透明。
- 背景与铺垫：多组件管线导致安全边界难以理解。
- 研究问题：能否逆向安全管线并构造越狱。
- 主要贡献：侧信道逆向 + 越狱评测。
  **Prompt 6（方法解读）**
- 方法类型：侧信道分析与黑盒攻击。
- 执行方式：利用响应时间推断过滤链路；对比不同版本机制；构造否定式/低资源语言攻击。
- 关键变量/指标：越狱成功率、响应差异。
- 局限性/偏差：依赖可观测的响应差异。
- 方法合理性：黑盒系统常暴露侧信道。
  **Prompt 7（结果解读）**
- 主要结果：揭示多级过滤结构并实现越狱。
- 关键图表：管线推断示意与成功率表。
- 研究问题对应：验证逆向与越狱可行。
- 出乎意料发现：LLM 改写与 CLIP 对齐不一致。
- 统计显著性：未突出显著性检验。
  **Prompt 8（讨论解读）**
- 解释：多组件联动存在安全断层。
- 比较：比单一过滤更复杂但仍可绕过。
- 重要意义：推动统一安全管线设计。
- 局限性：特定系统的适用性有限。
- 未来方向：减少侧信道与统一过滤策略。
  **Prompt 9（结论解读）**
- 主要发现：DALL·E 安全管线可被逆向并绕过。
- 贡献重申：管线逆向与越狱框架。
- 呼应研究问题：证明黑盒安全管线存在漏洞。
- 未来研究：更鲁棒的安全架构。

**阶段三：批判性思考与深入理解**
**Prompt 10（方法批判性评估）**

- 优点：系统性强，揭示实际漏洞。
- 局限：针对特定系统，泛化有限。
- 替代方法：模型级安全一致性约束。
  **Prompt 11（结果可信度评估）**
- 可信度：多实验验证增强可信度。
- 是否支持结论：支持可逆向与越狱。
- 其他解释：服务部署差异影响结果。
  **Prompt 12（创新性评估）**
- 创新点：侧信道驱动的安全管线逆向。
- 价值：揭示闭源系统安全弱点。
- 相对贡献：扩展黑盒攻击研究。
  **Prompt 13（逻辑结构与论证）**
- 论证清晰：逆向→攻击→分析。
- 衔接良好：侧信道与攻击结果一致。
- 薄弱点：防御评测较少。
  **Prompt 14（优缺点总结）**
- 优点：现实适用性强。
- 缺点：泛化性有限。
- 综合评价：闭源安全评测的重要工作。

## 28. USD: NSFW Content Detection for Text-to-Image Models via Scene Graph

Yuyang Zhang, Kangjie Chen, Xudong Jiang, Jiahui Wen, Yihui Jin, Ziyou Liang, Yihao Huang, Run Wang, & Lina Wang. (2025). USENIX Security 2025 (CCF A).
PDF: [USENIX](https://www.usenix.org/system/files/usenixsecurity25-zhang-yuyang.pdf)

[Image]

**阶段一：快速概览与初步印象**
**Prompt 1（三句话概括主题与目的）**
论文提出 USD，通过场景图识别文生图结果中的 NSFW 内容。方法将不安全概念映射为实体关系并进行分类。实验显示在检测与定位上更准确。
**Prompt 2（章节结构与大致内容）**

- Abstract：场景图检测框架概述。
- 1 Introduction：T2I 输出安全风险。
- 2 Related Work：NSFW 检测与场景图。
- 3 Method：场景图生成与关系分类。
- 4 Experiments：检测/定位评测。
- 5 Discussion/Conclusion。
  **Prompt 3（核心创新点/贡献）**
  将场景图关系用于 NSFW 检测与定位。
  **Prompt 4（关键词 5-10 个）**
- NSFW detection：不安全检测。
- scene graph：场景图。
- localization：定位。
- text-to-image：文生图。
- safety filter：安全过滤。

**阶段二：深入细节与理解内容**
**Prompt 5（引言解读）**

- 试图解决的问题：现有检测忽视关系层信息。
- 背景与铺垫：T2I 结果复杂，需结构化理解。
- 研究问题：场景图是否能提升检测准确性。
- 主要贡献：USD 框架与评测。
  **Prompt 6（方法解读）**
- 方法类型：图像分析与检测。
- 执行方式：生成场景图；对实体关系进行不安全判定；实现定位与过滤。
- 关键变量/指标：检测准确率、定位效果。
- 局限性/偏差：场景图质量影响检测。
- 方法合理性：结构化关系更贴合不安全概念。
  **Prompt 7（结果解读）**
- 主要结果：检测与定位性能优于基线。
- 关键图表：检测指标与可视化示例。
- 研究问题对应：验证场景图有效性。
- 出乎意料发现：部分细粒度场景仍难检测。
- 统计显著性：未突出显著性检验。
  **Prompt 8（讨论解读）**
- 解释：关系层特征能捕捉隐含风险。
- 比较：比仅视觉分类更稳健。
- 重要意义：可作为生成后过滤模块。
- 局限性：场景图生成成本较高。
- 未来方向：与生成过程联合优化。
  **Prompt 9（结论解读）**
- 主要发现：USD 提升 NSFW 检测与定位能力。
- 贡献重申：场景图驱动检测框架。
- 呼应研究问题：验证结构化检测有效。
- 未来研究：更强鲁棒检测。

**阶段三：批判性思考与深入理解**
**Prompt 10（方法批判性评估）**

- 优点：结构化理解提升检测。
- 局限：场景图生成误差影响结果。
- 替代方法：多模态安全分类器。
  **Prompt 11（结果可信度评估）**
- 可信度：对比实验增强可信度。
- 是否支持结论：支持检测改进。
- 其他解释：数据集偏差影响评测。
  **Prompt 12（创新性评估）**
- 创新点：场景图关系用于 NSFW 检测。
- 价值：提高安全过滤精准度。
- 相对贡献：拓展安全检测方法。
  **Prompt 13（逻辑结构与论证）**
- 论证清晰：方法→评测→分析。
- 衔接良好：关系特征与结果一致。
- 薄弱点：跨域泛化评测不足。
  **Prompt 14（优缺点总结）**
- 优点：检测更精细。
- 缺点：计算成本较高。
- 综合评价：安全检测的重要进展。

## 29. Bridging the Gap in Vision Language Models in Identifying Unsafe Concepts Across Modalities

Yiting Qu, Michael Backes, & Yang Zhang. (2025). USENIX Security 2025 (CCF A).
PDF: [USENIX](https://www.usenix.org/system/files/usenixsecurity25-qu-yiting.pdf)

[Image]

**阶段一：快速概览与初步印象**
**Prompt 1（三句话概括主题与目的）**
论文研究 VLM 在跨模态识别不安全概念时的能力差距。构建不安全概念数据集，评测文本与图像识别差异。提出基于 PPO 的对齐策略缩小差距。
**Prompt 2（章节结构与大致内容）**

- Abstract：问题与方法概述。
- 1 Introduction：跨模态安全差距。
- 2 Related Work：VLM 安全与对齐。
- 3 Dataset/Method：数据集与对齐策略。
- 4 Experiments：差距评测与对齐结果。
- 5 Discussion/Conclusion。
  **Prompt 3（核心创新点/贡献）**
  系统揭示跨模态安全差距并提出对齐方案。
  **Prompt 4（关键词 5-10 个）**
- VLM safety：VLM 安全。
- unsafe concept：不安全概念。
- cross-modal gap：跨模态差距。
- PPO alignment：PPO 对齐。
- safety evaluation：安全评测。

**阶段二：深入细节与理解内容**
**Prompt 5（引言解读）**

- 试图解决的问题：VLM 在不同模态的安全表现不一致。
- 背景与铺垫：安全过滤需跨模态一致。
- 研究问题：如何评测并缩小差距。
- 主要贡献：数据集 + 对齐策略。
  **Prompt 6（方法解读）**
- 方法类型：评测与对齐。
- 执行方式：构建跨模态不安全概念数据集；评测 VLM；用 PPO 进行对齐。
- 关键变量/指标：识别准确率、对齐增益。
- 局限性/偏差：数据集覆盖有限。
- 方法合理性：RL 对齐可缩小差距。
  **Prompt 7（结果解读）**
- 主要结果：揭示显著跨模态差距，并可部分缩小。
- 关键图表：模态差距统计与对齐效果表。
- 研究问题对应：验证对齐有效性。
- 出乎意料发现：某些概念差距更大。
- 统计显著性：未突出显著性检验。
  **Prompt 8（讨论解读）**
- 解释：训练数据与任务导致模态差距。
- 比较：对齐后安全一致性提升。
- 重要意义：提升安全过滤可靠性。
- 局限性：对新概念泛化不足。
- 未来方向：更大规模对齐与评测。
  **Prompt 9（结论解读）**
- 主要发现：跨模态安全差距显著但可缓解。
- 贡献重申：数据集与对齐方法。
- 呼应研究问题：证明差距可测可缩小。
- 未来研究：更强对齐与泛化评测。

**阶段三：批判性思考与深入理解**
**Prompt 10（方法批判性评估）**

- 优点：系统评测 + 对齐改进。
- 局限：依赖数据集覆盖度。
- 替代方法：更强多模态对齐训练。
  **Prompt 11（结果可信度评估）**
- 可信度：多模型评测增强可信度。
- 是否支持结论：支持差距存在。
- 其他解释：模型规模影响差距。
  **Prompt 12（创新性评估）**
- 创新点：跨模态安全差距系统分析。
- 价值：改进安全过滤器设计。
- 相对贡献：拓展 VLM 安全研究。
  **Prompt 13（逻辑结构与论证）**
- 论证清晰：评测→对齐→分析。
- 衔接良好：对齐结果支持结论。
- 薄弱点：真实场景验证有限。
  **Prompt 14（优缺点总结）**
- 优点：问题定义清晰、对齐有效。
- 缺点：泛化评测不足。
- 综合评价：VLM 安全研究的重要工作。

## 30. Identifying Provenance of Generative Text-to-Image Models

Ding, W., Wu, S., Shan, S., Zheng, H., & Zhao, B. Y. (2026). USENIX Security 2026 (CCF A).
PDF: [UChicago](https://annaha.net/publication/model-provenance/)

[Image]

**阶段一：快速概览与初步印象**
**Prompt 1（三句话概括主题与目的）**
论文提出黑盒模型谱系归因方法，用于判断生成模型是否由某基座模型微调而来。方法通过查询生成图像并比较特征分布完成归因。实验验证在真实与对抗场景下有效。
**Prompt 2（章节结构与大致内容）**

- Abstract：模型谱系归因概述。
- 1 Introduction：模型合规与责任追踪。
- 2 Related Work：溯源与模型识别。
- 3 Method：黑盒查询与分布对比。
- 4 Experiments：归因准确率评测。
- 5 Discussion/Conclusion。
  **Prompt 3（核心创新点/贡献）**
  提出仅需黑盒访问的模型谱系归因框架。
  **Prompt 4（关键词 5-10 个）**
- model provenance：模型溯源。
- black-box querying：黑盒查询。
- feature distribution：特征分布对比。
- provenance audit：溯源审计。
- text-to-image：文生图。

**阶段二：深入细节与理解内容**
**Prompt 5（引言解读）**

- 试图解决的问题：模型来源难以追踪。
- 背景与铺垫：合规与责任归因需求。
- 研究问题：能否黑盒判断模型谱系。
- 主要贡献：分布对比归因方法。
  **Prompt 6（方法解读）**
- 方法类型：黑盒统计归因。
- 执行方式：查询目标模型生成图像；提取特征分布与基准模型对比；统计检验判定。
- 关键变量/指标：归因准确率、误报率。
- 局限性/偏差：对抗后处理可能影响分布。
- 方法合理性：微调模型保留基座统计特征。
  **Prompt 7（结果解读）**
- 主要结果：在多场景下实现可靠归因。
- 关键图表：归因准确率与鲁棒性统计。
- 研究问题对应：验证黑盒归因可行。
- 出乎意料发现：轻度后处理仍可识别。
- 统计显著性：未突出显著性检验。
  **Prompt 8（讨论解读）**
- 解释：基座模型特征分布具有可识别性。
- 比较：比仅靠元数据更可靠。
- 重要意义：支持合规审查与责任追踪。
- 局限性：强对抗规避需评估。
- 未来方向：对抗鲁棒性与多源归因。
  **Prompt 9（结论解读）**
- 主要发现：黑盒查询可完成模型谱系归因。
- 贡献重申：分布对比归因框架。
- 呼应研究问题：证明合规审计可行。
- 未来研究：更强对抗鲁棒性。

**阶段三：批判性思考与深入理解**
**Prompt 10（方法批判性评估）**

- 优点：无需权重访问，适用性强。
- 局限：对抗后处理可能降低效果。
- 替代方法：结合水印或指纹。
  **Prompt 11（结果可信度评估）**
- 可信度：多场景评测增强可信度。
- 是否支持结论：支持归因有效性。
- 其他解释：模型规模差异影响结果。
  **Prompt 12（创新性评估）**
- 创新点：黑盒谱系归因。
- 价值：合规审计工具。
- 相对贡献：拓展模型溯源研究。
  **Prompt 13（逻辑结构与论证）**
- 论证清晰：方法→评测→讨论。
- 衔接良好：分布对比与结果一致。
- 薄弱点：强对抗评测不足。
  **Prompt 14（优缺点总结）**
- 优点：实用性强。
- 缺点：对抗鲁棒性待提升。
- 综合评价：模型溯源的重要工作。

# 汇总表

| Title                                                                                                                      | PDF                                                                                                                                                                           | Model/Target             | Access | Defense            | Venue           | CCF | Year |
| -------------------------------------------------------------------------------------------------------------------------- | ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | ------------------------ | ------ | ------------------ | --------------- | --- | ---- |
| Perception-guided Jailbreak against Text-to-Image Models                                                                   | [PDF](https://arxiv.org/pdf/2408.10848)                                                                                                                                          | 多种T2I模型              | 黑盒   | 攻击（越狱）       | AAAI            | A   | 2025 |
| Multimodal Pragmatic Jailbreak on Text-to-image Models                                                                     | [PDF](https://aclanthology.org/2025.acl-long.234.pdf)                                                                                                                            | 多T2I模型                | 黑盒   | 无（评测）         | ACL             | A   | 2025 |
| Unsafe Diffusion: On the Generation of Unsafe Images and Hateful Memes From Text-To-Image Models                           | [PDF](https://arxiv.org/pdf/2305.13873.pdf)                                                                                                                                      | T2I模型安全评测          | 黑盒   | 无（风险评估）     | CCS             | A   | 2023 |
| SurrogatePrompt: Bypassing the Safety Filter of Text-to-Image Models via Substitution                                      | [PDF](https://arxiv.org/pdf/2309.14122)                                                                                                                                          | Midjourney 等闭源T2I     | 黑盒   | 攻击（越狱）       | CCS             | A   | 2024 |
| On the Feasibility of Poisoning Text-to-Image AI Models via Adversarial Mislabeling                                        | [PDF](https://people.cs.uchicago.edu/~ravenben/publications/abstracts/amp-ccs25.html)                                                                                            | T2I训练管线              | 黑盒   | 攻击（投毒）       | CCS             | A   | 2025 |
| MMA-Diffusion: MultiModal Attack on Diffusion Models                                                                       | [PDF](https://openaccess.thecvf.com/content/CVPR2024/papers/Yang_MMA-Diffusion_MultiModal_Attack_on_Diffusion_Models_CVPR_2024_paper.pdf)                                        | T2I扩散模型（开源+商用） | 黑盒   | 攻击（越狱）       | CVPR            | A   | 2024 |
| Black-Box Forgery Attacks on Semantic Watermarks for Diffusion Models                                                      | [PDF](https://openaccess.thecvf.com/content/CVPR2025/papers/Muller_Black-Box_Forgery_Attacks_on_Semantic_Watermarks_for_Diffusion_Models_CVPR_2025_paper.pdf)                    | 语义水印                 | 黑盒   | 攻击               | CVPR            | A   | 2025 |
| Harnessing Frequency Spectrum Insights for Image Copyright Protection Against Diffusion Models (CoprGuard)                 | [PDF](https://openaccess.thecvf.com/content/CVPR2025/papers/Liu_Harnessing_Frequency_Spectrum_Insights_for_Image_Copyright_Protection_Against_Diffusion_CVPR_2025_paper.pdf)     | 训练数据版权保护         | 黑盒   | 水印/版权          | CVPR            | A   | 2025 |
| Mind the Trojan Horse: Image Prompt Adapter Enabling Scalable and Deceptive Jailbreaking                                   | [PDF](https://openaccess.thecvf.com/content/CVPR2025/papers/Chen_Mind_the_Trojan_Horse_Image_Prompt_Adapter_Enabling_Scalable_and_CVPR_2025_paper.pdf)                           | T2I-IP-DMs / IGS         | 黑盒   | 攻击（越狱）       | CVPR            | A   | 2025 |
| OpenSDI: Spotting Diffusion-Generated Images in the Open World                                                             | [PDF](https://openaccess.thecvf.com/content/CVPR2025/papers/Wang_OpenSDI_Spotting_Diffusion-Generated_Images_in_the_Open_World_CVPR_2025_paper.pdf)                              | 扩散图像检测             | 黑盒   | 检测               | CVPR            | A   | 2025 |
| Six-CD: Benchmarking Concept Removals for Text-to-image Diffusion Models                                                   | [PDF](https://openaccess.thecvf.com/content/CVPR2025/papers/Ren_Six-CD_Benchmarking_Concept_Removals_for_Text-to-image_Diffusion_Models_CVPR_2025_paper.pdf)                     | 扩散模型                 | 黑盒   | 无（基准）         | CVPR            | A   | 2025 |
| T2ISafety: Benchmark for Assessing Fairness, Toxicity, and Privacy in Image Generation                                     | [PDF](https://openaccess.thecvf.com/content/CVPR2025/papers/Li_T2ISafety_Benchmark_for_Assessing_Fairness_Toxicity_and_Privacy_in_Image_CVPR_2025_paper.pdf)                     | T2I安全评测              | 黑盒   | 评测（安全基准）   | CVPR            | A   | 2025 |
| PLA: Prompt Learning Attack against Text-to-Image Generative Models                                                        | [PDF](https://openaccess.thecvf.com/content/ICCV2025/papers/Lyu_PLA_Prompt_Learning_Attack_against_Text-to-Image_Generative_Models_ICCV_2025_paper.pdf)                          | 黑盒T2I模型              | 黑盒   | 无（攻击）         | ICCV            | A   | 2025 |
| JailbreakDiffBench: A Comprehensive Benchmark for Jailbreaking Diffusion Models                                            | [PDF](https://openaccess.thecvf.com/content/ICCV2025/papers/Jin_JailbreakDiffBench_A_Comprehensive_Benchmark_for_Jailbreaking_Diffusion_Models_ICCV_2025_paper.pdf)              | T2I/T2V系统              | 黑盒   | 无（基准）         | ICCV            | A   | 2025 |
| AutoPrompt: Automated Red-Teaming of Text-to-Image Models via LLM-Driven Adversarial Prompts                               | [PDF](https://openaccess.thecvf.com/content/ICCV2025/papers/Liu_AutoPrompt_Automated_Red-Teaming_of_Text-to-Image_Models_via_LLM-Driven_Adversarial_Prompts_ICCV_2025_paper.pdf) | T2I系统                  | 黑盒   | 无（红队/攻击）    | ICCV            | A   | 2025 |
| Automated Red Teaming for Text-to-Image Models through Feedback-Guided Prompt Iteration with Vision-Language Models        | [PDF](https://openaccess.thecvf.com/content/ICCV2025/papers/Xu_Automated_Red_Teaming_for_Text-to-Image_Models_through_Feedback-Guided_Prompt_Iteration_ICCV_2025_paper.pdf)      | T2I系统                  | 黑盒   | 无（红队/攻击）    | ICCV            | A   | 2025 |
| Prompting4Debugging: Red-Teaming Text-to-Image Diffusion Models by Finding Problematic Prompts                             | [PDF](https://proceedings.mlr.press/v235/chin24a/chin24a.pdf)                                                                                                                    | T2I扩散模型              | 黑盒   | 无（红队）         | ICML            | A   | 2024 |
| Membership Inference on Text-to-Image Diffusion Models via Conditional Likelihood Discrepancy                              | [PDF](https://proceedings.neurips.cc/paper_files/paper/2024/file/874411a224a1934b80d499068384808b-Paper-Conference.pdf)                                                          | T2I扩散模型              | 黑盒   | 攻击（隐私推断）   | NeurIPS         | A   | 2024 |
| Red-Teaming Text-to-Image Systems by Rule-based Preference Modeling                                                        | [PDF](https://arxiv.org/pdf/2505.21074)                                                                                                                                          | 黑盒T2I系统（商用+开源） | 黑盒   | 无（红队/攻击）    | NeurIPS         | A   | 2025 |
| SneakyPrompt: Jailbreaking Text-to-image Generative Models                                                                 | [PDF](https://arxiv.org/pdf/2305.12082)                                                                                                                                          | DALL·E 2 + SD           | 黑盒   | 无（攻击）         | S&P             | A   | 2024 |
| Nightshade: Prompt-Specific Poisoning Attacks on Text-to-Image Generative Models                                           | [PDF](https://people.cs.uchicago.edu/~ravenben/publications/abstracts/nightshade-oakland24.html)                                                                                 | T2I训练数据链路          | 黑盒   | 攻击（投毒）       | S&P             | A   | 2024 |
| Fuzz-Testing Meets LLM-Based Agents: An Automated and Efficient Framework for Jailbreaking Text-To-Image Generation Models | [PDF](https://arxiv.org/pdf/2408.00523)                                                                                                                                          | T2I系统                  | 黑盒   | 无（攻击）         | S&P             | A   | 2025 |
| Towards Reliable Verification of Unauthorized Data Usage in Personalized Text-to-Image Diffusion Models                    | [PDF](https://doi.org/10.1109/SP61157.2025.00073)                                                                                                                                | 个性化T2I模型            | 黑盒   | 溯源/验证          | S&P             | A   | 2025 |
| Glaze: Protecting Artists from Style Mimicry by Text-to-Image Models                                                       | [PDF](https://www.usenix.org/system/files/usenixsecurity23-shan.pdf)                                                                                                             | 艺术风格保护             | 黑盒   | 防护               | USENIX Security | A   | 2023 |
| Prompt Stealing Attacks Against Text-to-Image Generation Models                                                            | [PDF](https://www.usenix.org/system/files/usenixsecurity24-shen-xinyue.pdf)                                                                                                      | 生成图像 -> 提示词       | 黑盒   | 攻击（提示词窃取） | USENIX Security | A   | 2024 |
| Cross-Modal Prompt Inversion: Unifying Threats to Text and Image Generative AI Models                                      | [PDF](https://www.usenix.org/system/files/usenixsecurity25-ye-inversion.pdf)                                                                                                     | 文本/图像生成模型        | 黑盒   | 攻击（提示词窃取） | USENIX Security | A   | 2025 |
| Exposing the Guardrails: Reverse-Engineering and Jailbreaking Safety Filters in DALL·E Text-to-Image Pipelines            | [PDF](https://www.usenix.org/system/files/usenixsecurity25-villa.pdf)                                                                                                            | DALL·E管线              | 黑盒   | 攻击（越狱）       | USENIX Security | A   | 2025 |
| USD: NSFW Content Detection for Text-to-Image Models via Scene Graph                                                       | [PDF](https://www.usenix.org/system/files/usenixsecurity25-zhang-yuyang.pdf)                                                                                                     | T2I生成图像检测          | 黑盒   | 检测               | USENIX Security | A   | 2025 |
| Bridging the Gap in Vision Language Models in Identifying Unsafe Concepts Across Modalities                                | [PDF](https://www.usenix.org/system/files/usenixsecurity25-qu-yiting.pdf)                                                                                                        | VLM安全检测              | 黑盒   | 检测/对齐          | USENIX Security | A   | 2025 |
| Identifying Provenance of Generative Text-to-Image Models                                                                  | [PDF](https://annaha.net/publication/model-provenance/)                                                                                                                          | T2I模型谱系              | 黑盒   | 溯源/归因          | USENIX Security | A   | 2026 |
