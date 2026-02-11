# 黑盒 T2I 攻击论文总结（7 篇，结合攻击类别图）

说明：
- 这些论文均为黑盒场景（或以黑盒为主），覆盖越狱、提示词窃取与水印攻击。
- 下表复刻你给的“攻击类别”图的结构，并把 7 篇论文映射进去。

## 攻击类别映射（基于图示）

| 攻击类别 | 代表方法/思路 | 原理简述 | 本次对应论文 | 适用场景 |
| --- | --- | --- | --- | --- |
| 语义伪装 (Black-box) | PromptTune, AutoDAN-T2I | 用 LLM 改写 Prompt，使用隐喻/同义词替换敏感词。 | Multimodal Pragmatic Jailbreak; Perception-guided Jailbreak | API 调用，普通用户 |
| 搜索与优化 (Gray-box) | SneakyPrompt, RIATIG | 利用 RL 或遗传算法搜索能绕过检测的特殊词汇组合。 | SneakyPrompt; PLA; Cross-Modal Prompt Inversion; Black-Box Forgery Attacks (watermark) | API 调用，研究人员 |
| 梯度攻击 (White-box) | PEZ, MMA-Diffusion | 计算梯度，生成能触发特定输出的扰动。 | 无（本 7 篇均非白盒） | 开源模型 (SD, FLUX) |
| 视觉注入 (Multi-modal) | AdvI2I | 在参考图中注入攻击噪声，通过视觉通道越狱。 | Mind the Trojan Horse (Image Prompt Adapter) | 图生图、ControlNet |

## 逐篇总结（按类别）

### 语义伪装 (Black-box)

**Multimodal Pragmatic Jailbreak on Text-to-image Models (ACL 2025)**
- 场景：多 T2I 模型黑盒评测，关注“语用层面”绕过。
- 方法：将不安全语义拆解为“安全图像内容 + 安全文字”，用固定模板提示词让模型生成含可读文字的图像。
- 方法：依赖“跨模态组合后的语用含义”触发不安全效果，而非显式敏感词或梯度优化。
- 贡献：揭示跨模态语用理解的结构性风险，提供系统化评测与数据集，并发现文字渲染越强越脆弱。

**Perception-guided Jailbreak against Text-to-Image Models (AAAI 2025)**
- 场景：黑盒 T2I 越狱，目标是绕过关键词/语义过滤。
- 方法：遵循 PSTSI 原则（感知相似、语义不一致）做最小词替换，让文本看似安全但视觉意图保持。
- 方法：LLM 自动定位高风险词并替换，跨模型可迁移，几乎不依赖目标系统。
- 贡献：提出感知导向越狱范式，证明纯文本过滤与关键词黑名单存在结构性缺口。

### 搜索与优化 (Gray-box)

**SneakyPrompt: Jailbreaking Text-to-image Generative Models (S&P 2024)**
- 场景：黑盒 T2I，基于系统拒绝/成功信号的越狱搜索。
- 方法：用影子文本编码器锚定目标语义，RL 采样替换 token 形成对抗提示。
- 方法：黑盒查询后以“是否拦截 + 语义相似度”作为双重反馈，满足阈值即成功。
- 贡献：首个自动化 T2I jailbreak 框架，明确区分“绕过过滤”与“保持语义”两步目标。

**PLA: Prompt Learning Attack against Text-to-Image Generative Models (ICCV 2025)**
- 场景：黑盒 T2I 模型，提示词可学习化。
- 方法：从目标提示中提取敏感语义嵌入（SKE），注入到可学习的连续 prompt 表示。
- 方法：用多模态相似度损失 + ZOO 黑盒优化迭代生成自然语言对抗提示。
- 贡献：把越狱从离散词替换提升为连续表示学习问题，攻击更稳定可控。

**Cross-Modal Prompt Inversion: Unifying Threats to Text and Image Generative AI Models (USENIX Security 2025)**
- 场景：黑盒生成模型（文本+图像），目标是提示词窃取。
- 方法：两阶段反演：先监督学习直接反演，再用 RL 微调提高关键词/细节恢复率。
- 方法：优化目标是“语义等价、输出一致”，而非逐词复现。
- 贡献：提出统一的跨模态提示词窃取框架，证明黑盒输出中存在可学习的提示痕迹。

**Black-Box Forgery Attacks on Semantic Watermarks for Diffusion Models (CVPR 2025)**
- 场景：黑盒水印防护体系，攻击目标是伪造或移除语义水印。
- 方法：对带水印图像做扩散反演，得到水印潜噪声作为“语义锚点”。
- 方法：通过 reprompting 或对抗优化伪造/移除水印，实现错误归因。
- 贡献：证明语义水印在黑盒下可被伪造，动摇“可追责溯源”的安全假设。

### 视觉注入 (Multi-modal)

**Mind the Trojan Horse: Image Prompt Adapter Enabling Scalable and Deceptive Jailbreaking (CVPR 2025)**
- 场景：黑盒 IGS / 图像条件 T2I，利用图像输入通道进行攻击。
- 方法：AEO 仅攻击图像编码器，在良性图像上加微扰使特征对齐 NSFW 语义。
- 方法：通过 IP-Adapter 把被“劫持”的图像语义注入扩散模型，实现隐蔽越狱。
- 贡献：揭示图像通道的规模化攻击面，强调编码器鲁棒性与图像输入过滤协同防御。

## 小结（研究定位）
- 这 7 篇覆盖了“语义伪装 + 搜索优化 + 视觉注入”的黑盒攻击主干。
- 反馈信号是主流驱动（RL/偏好/判定器），而 CoT 在黑盒攻击中基本未出现。
