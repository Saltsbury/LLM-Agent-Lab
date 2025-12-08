# 大语言模型（LLM）上机课讲义

## 课程基本信息
- **课程名称**：大数据与商务智能 - 大语言模型应用开发实践
- **授课对象**：中山大学工商管理非全专硕MBA学生
- **课时**：3学时（上机课）
- **前置知识**：基础Python编程

## 学习目标
完成本课程后，您将能够：
1. 独立搭建本地大语言模型运行环境
2. 使用Python脚本调用在线和本地LLM服务
3. 通过Cherry Studio可视化工具开发LLM应用
4. 设计结构化系统提示词提升AI响应质量
5. 开发简单的LLM应用解决实际业务问题

## 1.1 LLM服务准备

### 1.1.1 在线LLM API获取

#### 1.1.1.1 获取在线API

1. Deepseek（深度求索）
    
    Deepseek提供多语言大模型服务，适合代码生成和专业领域问答场景。官方文档：https://api-docs.deepseek.com/zh-cn/
    
    1. **注册Deepseek账号**  
        访问Deepseek官网：https://platform.deepseek.com
    
    2. **开通API服务**  
        - 登录后进入"控制台"→"API服务"
        - 点击"开通服务"，阅读并同意服务协议
    
    3. **创建API密钥**  
        - 在控制台左侧菜单选择"API密钥"
        - 点击"新建密钥"，输入密钥名称（如"llm-course"）
        - 保存生成的API Key（注意：仅显示一次，需及时复制）
    
    4. **获取API端点**  
        Deepseek通用对话API地址：`https://api.deepseek.com/v1/chat/completions`

2. 通义千问（阿里云）

    通义千问是阿里云提供的大语言模型服务，通过阿里云百炼平台提供API调用方式，适合中文场景应用开发。官方文档：https://bailian.console.aliyun.com/?tab=doc#/doc/?type=model&url=2840915

    1. **注册阿里云账号**  
    访问阿里云官网：https://www.aliyun.com/

    2. **开通通义千问服务**  
        - 访问阿里云百炼平台：https://bailian.aliyun.com/
        - 点击"立即开通"，完成服务开通流程

    3. **创建API密钥**  
        - 登录后点击右上角"创建我的API-KEY"
        - 选择默认业务空间（或创建新空间）
        - 点击"确定"生成API密钥
        - 点击"查看"并复制生成的API密钥

    4. **获取API端点**  
        通义千问API地址：`https://dashscope.aliyuncs.com/api/v1/services/aigc/text-generation/generation`

3. KIMI（月之暗面）
    
    KIMI是月之暗面公司推出的大语言模型，以长文本处理能力见长。官方文档：https://platform.moonshot.cn/docs/overview
    
    1. **注册KIMI账号**  
        访问KIMI控制台：https://platform.moonshot.cn/console
    
    2. **开通API权限**  
        - 登录后进入"产品中心"→"KIMI API"
        - 点击"立即开通"，完成基础配置
    
    3. **创建API密钥**  
        - 进入"控制台"→"API密钥"
        - 点击"创建密钥"，选择对应的项目空间
        - 复制生成的API Key和Secret（部分接口可能需要）
    
    4. **获取API端点**  
        KIMI对话API地址：`https://api.moonshot.cn/v1/chat/completions`

4. 火山方舟（字节跳动）

    火山方舟提供火山方舟大模型服务平台，支持多种模型如豆包等调用。官方文档：https://www.volcengine.com/docs/82379/1399008

    1. **注册火山方舟账号**  
        访问火山方舟官网：https://exp.volcengine.com/ark

    2. **开通大模型服务**  
        - 登录后进入"控制台"→"火山方舟"
        - 选择需要的模型（如"字节跳动-豆包"），点击"开通服务"
    
    3. **创建API密钥**  
        - 进入"访问控制"→"密钥管理"
        - 点击"新建密钥"，生成Access Key ID和Secret Access Key
        - 记录密钥信息（需妥善保管）
    
    4. **获取API端点**  
        火山方舟通用API地址：`https://ark.cn-beijing.volces.com/api/v3/chat/completions`

#### 1.1.1.2 环境变量设置

把API Key配置到环境变量，从而避免在代码里显式地配置API Key，降低泄漏风险：

1. 在Windows系统桌面中按Win+Q键，在搜索框中搜索编辑系统环境变量，单击打开系统属性界面。

2. 在系统属性窗口，单击环境变量，进入环境变量配置页面。

    ![p894015.png](attachment:p894015.png)

3. 在系统变量区域分别新建以下环境变量：
    - 深度求索：变量名`DEEPSEEK_API_KEY`，变量值填入Deepseek API Key
    - 通义千问：变量名`DASHSCOPE_API_KEY`，变量值填入DashScope API Key
    - KIMI：变量名`KIMI_API_KEY`，变量值填入KIMI API Key
    - 火山方舟：变量名`VOLCENGINE_ACCESS_KEY`和`VOLCENGINE_SECRET_KEY`，分别填入对应的密钥

4. 依次单击三个窗口的确定，关闭系统属性配置页面，完成环境变量配置。

#### 1.1.1.3 验证API配置

1. 使用命令提示符（CMD），运行以下命令验证各环境变量：
    ```bash
    # 验证深度求索
    echo %DEEPSEEK_API_KEY%
    # 验证通义千问
    echo %DASHSCOPE_API_KEY%
    # 验证KIMI
    echo %KIMI_API_KEY%
    # 验证火山方舟
    echo %VOLCENGINE_ACCESS_KEY%
    echo %VOLCENGINE_SECRET_KEY%
    ```
    如果正确显示对应API Key，说明环境变量配置成功。

2. 运行以下Python代码，验证各API配置是否成功：
    ```python
    import requests
    import os
    from dotenv import load_dotenv

    load_dotenv()

    def verify_deepseek_api():
        """验证深度求索API配置是否成功"""
        url = "https://api.deepseek.com/v1/chat/completions"
        
        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {os.getenv('DEEPSEEK_API_KEY')}"
        }
        
        payload = {
            "model": "deepseek-chat",
            "messages": [{"role": "user", "content": "你好，请返回'API配置成功'"}]
        }
        
        try:
            response = requests.post(url, headers=headers, json=payload)
            response.raise_for_status()
            return "API配置成功" in response.json()["choices"][0]["message"]["content"]
        except Exception as e:
            print(f"深度求索验证失败: {str(e)}")
            return False

    def verify_kimi_api():
        """验证KIMI API配置是否成功"""
        url = "https://api.moonshot.cn/v1/chat/completions"
        
        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {os.getenv('KIMI_API_KEY')}"
        }
        
        payload = {
            "model": "moonshot-v1-8k",
            "messages": [{"role": "user", "content": "你好，请返回'API配置成功'"}]
        }
        
        try:
            response = requests.post(url, headers=headers, json=payload)
            response.raise_for_status()
            return "API配置成功" in response.json()["choices"][0]["message"]["content"]
        except Exception as e:
            print(f"KIMI验证失败: {str(e)}")
            return False

    def verify_volcengine_api():
        """验证火山方舟API配置是否成功"""
        url = "https://ark.cn-beijing.volces.com/api/v3/chat/completions"
        
        headers = {
            "Content-Type": "application/json",
            "X-Volc-AccessKey": os.getenv('VOLCENGINE_ACCESS_KEY'),
            "X-Volc-SecretKey": os.getenv('VOLCENGINE_SECRET_KEY')
        }
        
        payload = {
            "model": "doubao-pro",
            "messages": [{"role": "user", "content": "你好，请返回'API配置成功'"}]
        }
        
        try:
            response = requests.post(url, headers=headers, json=payload)
            response.raise_for_status()
            return "API配置成功" in response.json()["choices"][0]["message"]["content"]
        except Exception as e:
            print(f"火山方舟验证失败: {str(e)}")
            return False

    def verify_tongyi_api():
        """验证通义千问API配置是否成功"""
        url = "https://dashscope.aliyuncs.com/api/v1/services/aigc/text-generation/generation"
        
        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {os.getenv('TONGYI_API_KEY')}"
        }
        
        payload = {
            "model": "qwen-plus",
            "input": {"prompt": "你好，请返回'API配置成功'"},
            "parameters": {"max_tokens": 50}
        }
        
        try:
            response = requests.post(url, headers=headers, json=payload)
            response.raise_for_status()
            return "API配置成功" in response.json()["output"]["text"]
        except Exception as e:
            print(f"验证失败: {str(e)}")
            return False

    if __name__ == "__main__":
        if verify_deepseek_api():
            print("深度求索API配置成功")
        else:
            print("深度求索API配置失败")
            
        if verify_kimi_api():
            print("KIMI API配置成功")
        else:
            print("KIMI API配置失败")
            
        if verify_volcengine_api():
            print("火山方舟API配置成功")
        else:
            print("火山方舟API配置失败")
            
        if verify_tongyi_api():
            print("通义千问API配置成功")
        else:
            print("通义千问API配置失败")
    ```

#### 1.1.1.4 LLM API计费规则（2025年11月更新）

在线LLM API的计费核心是**Token计量**，但不同服务商的Token计算逻辑、计费模式存在差异。以下结合主流服务商（深度求索、KIMI、火山引擎、通义千问）的规则展开说明：

1. Token概念
    - **定义**：在自然语言处理NLP中，Token是模型处理文本的基本单位，可理解为“语义片段”。例如：
        - 英文中，1个Token约等于4个字符（如“hello”为1个Token）；
        - 中文中，1个Token通常对应1个词语或一个汉字，如“中国”可能计为1个Token，而单个汉字"夔"可能会被分解为若干 Token 的组合。大致来说，对于一段通常的中文文本，1 个 Token 大约相当于 1.5-2 个汉字。
    - **影响范围**：Token数量直接决定费用（按Token数计费）和模型处理能力（受上下文窗口限制，如“8k模型”指最大支持8192个Token的输入+输出）。

2. 主流服务商的Token计算与计费差异

    | 服务商       | Token计算规则                                                                 | 计费模式                                  | 特殊政策                                                                 |
    |--------------|-----------------------------------------------------------------------------|-----------------------------------------|----------------------------------------------------------------------|
    | 深度求索Deepseek [计费页](https://api-docs.deepseek.com/zh-cn/quick_start/pricing) | 中文按词语计（1词语=1 Token），英文按分词计（1单词≈1 Token）；系统提示词和历史对话均计入Token。 | 输入/输出Token分别计费，不同模型单价不同（如`deepseek-chat`输入0.3元/千Token，输出0.6元/千Token）。 | 新用户赠送一定额度免费Token（约50万-100万），过期未使用自动清零。                       |
    | KIMI（月之暗面）[计费页](https://platform.moonshot.cn/docs/pricing/chat) | Chat Completion 接口收费：我们对 Input 和 Output 均实行按量计费，中文按词语计（1词语=1 Token），英文按分词计（1单词≈1 Token）。如果您上传并抽取文档内容，并将抽取的文档内容作为 Input 传输给模型，那么文档内容也将按量计费。文件相关接口（文件内容抽取/文件存储）接口限时免费，即您只上传并抽取文档，这个API本身不会产生费用。| 输入/输出合并计费，`moonshot-v1-8k`模型统一0.3元/千Token，`moonshot-v1-32k`长文本模型单价略高。 | 免费额度每日限量（约10万Token/天），超出部分按阶梯价计费（用量越大单价越低）。            |
    | 火山方舟/豆包（字节跳动） [计费页](https://www.volcengine.com/docs/82379/1544681)| 中文1词语=1 Token，英文1单词≈1 Token；上下文窗口内的所有内容（含系统提示、历史消息）均计入。 | 分模型计费，`doubao-pro`输入0.2元/千Token，输出0.4元/千Token；`doubao-lite`单价约为前者的1/3。 | 企业用户可购买套餐包（如100万Token套餐约150元），有效期1年，比按次计费便宜30%。         |
    | 通义千问 [计费页](https://help.aliyun.com/zh/model-studio/billing-for-model-studio?spm=5176.29597918.J_etFdgJIDidF3OAYTCTJFS.2.518c7b08zG1O5P)      | 中文1词语=1 Token，英文按BPE分词（1000字符≈300-400 Token）；输入/输出分别计量。         | 按模型档位收费，`qwen-plus`输入0.15元/千Token，输出0.3元/千Token；`qwen-max`单价约为其3倍。 | 阿里云账号实名认证后赠送100万免费Token，有效期3个月，仅支持基础模型。                   |


3. 计费常见问题
    1. **Token数量查询**  
        - 多数服务商提供API响应头（如`X-Token-Usage`）返回输入/输出Token数；  
        - 可通过官方工具预估（如KIMI的[Token计算器](https://platform.moonshot.cn/tools/token-calculator)）。

    2. **上下文窗口与超额计费**  
        - 若输入+输出Token超过模型最大窗口（如8k），会被截断或拒绝服务，需提前控制对话长度；  
        - 部分服务（如火山引擎）支持动态调整窗口，超额部分按溢价计费（约为基础价的1.5倍）。

    3. **免费额度限制**  
        - 免费Token通常仅支持基础模型，且有调用频率限制（如Deepseek每秒最多5次调用）；  
        - 商用场景需提前升级账号，避免因额度耗尽导致服务中断。

#### 1.1.1.5 四家服务商在线API比较

1. Deepseek（深度求索）
    - 在线API模型
        | 模型名称               | 上下文窗口 | 核心定位                  | 适用场景                          |
        |------------------------|------------|---------------------------|-----------------------------------|
        | deepseek-chat（基础）  | 8k/16k     | 通用对话、日常交互        | 客服、问答机器人、轻量交互        |
        | deepseek-coder-v2（代码）| 32k/64k   | 代码生成/调试/重构        | 程序员开发、自动化脚本、代码审核  |
        | deepseek-moe-7B/13B（进阶）| 64k     | 专业领域推理（数学/科研） | 学术论文撰写、数据分析、公式推导  |
        | deepseek-vl（多模态）  | 16k        | 图文理解+对话             | 图片解析、图文问答、视觉推理      |
    - 定价（2025年11月）
        | 模型类型       | 输入单价（元/千Token） | 输出单价（元/千Token） | 套餐包优惠（100万Token） | 免费额度          |
        |----------------|------------------------|------------------------|--------------------------|-------------------|
        | 通用对话（8k）  | 0.25                   | 0.50                   | 120元（省30%）           | 新用户50万Token（30天） |
        | 代码模型（32k） | 0.40                   | 0.80                   | 200元（省25%）           | 新用户30万Token（30天） |
        | MOE进阶模型    | 0.80                   | 1.60                   | 400元（省20%）           | 无免费额度        |
        | 多模态模型    | 0.60（文本）+ 0.1元/张图 | 1.20                   | 300元（省20%）           | 新用户10万Token+100张图 |
    - 模型特色
        - **写代码优势**：deepseek-coder-v2支持20+编程语言，代码生成准确率在HumanEval评测中达82.3%，超过GPT-4o-mini（79.8%），支持64k长代码上下文（如大型项目重构）。
        - **专业场景优化**：针对数学、物理、科研领域优化，MOE模型在MMLU（综合能力评测）中得分83.5，擅长复杂公式推导和学术写作。
        - **低延迟优势**：通用对话API响应时间≤300ms，适合实时交互场景（如直播弹幕回复、在线客服）。

2. KIMI（月之暗面）
    - 在线API模型
        | 模型名称               | 上下文窗口 | 核心定位                  | 适用场景                          |
        |------------------------|------------|---------------------------|-----------------------------------|
        | moonshot-v2-pro（闭源） | 256k       | 长文本+高精度对话         | 电子书解析、法律文书处理、万字报告生成 |
        | moonshot-v2-max（闭源） | 512k       | 超长篇文本+复杂推理       | 百万字小说分析、企业年报解读、多文档对比 |
        | moonshot-v2-open-70B（开源）| 128k    | 开源高精度模型            | 企业私有化部署（在线API支持调用）、成本敏感场景 |
        | moonshot-mini-1.8B（开源）| 32k      | 轻量开源模型              | 边缘设备、高并发低预算场景        |

    - 定价（2025年11月）
        | 模型类型       | 输入单价（元/千Token） | 输出单价（元/千Token） | 套餐包优惠（100万Token） | 免费额度          |
        |----------------|------------------------|------------------------|--------------------------|-------------------|
        | v2-pro（256k）  | 0.35                   | 0.70                   | 175元（省30%）           | 每日10万Token（不限期） |
        | v2-max（512k）  | 0.70                   | 1.40                   | 350元（省30%）           | 无免费额度        |
        | v2-open-70B（开源）| 0.20               | 0.40                   | 100元（省33%）           | 新用户80万Token（60天） |
        | mini-1.8B（开源） | 0.05                   | 0.10                   | 25元（省33%）            | 每日20万Token（不限期） |

    - 模型特色
        - **长文本处理优势**：512k上下文窗口支持“一次性解析百万字文本”，无需分段处理，在法律合同、学术论文、小说分析场景中效率远超同类模型（多数竞品最大窗口为128k）。
        - **开源模型性能突破**：
            - **moonshot-v2-open-70B**：在2025年10月Hugging Face Open LLM Leaderboard中，MMLU得分85.7（闭源模型GPT-4o-mini为88.2，豆包pro为86.3），C-Eval（中文综合能力）得分91.2（超过GPT-4o-mini的89.5），推理、逻辑、中文理解能力已逼近主流闭源模型。
            - **核心优势**：支持128k长上下文，在线API调用成本仅为闭源模型的57%（v2-open-70B输入0.2元/千Token vs v2-pro 0.35元/千Token），且开源可商用（协议宽松，支持企业二次开发）。
        - **多轮对话一致性强**：针对长对话优化，连续50+轮交互不丢失上下文，适合智能助手、虚拟客服等场景。

3. 火山方舟（字节跳动）
    - 在线API模型
        | 模型名称               | 上下文窗口 | 核心定位                  | 适用场景                          |
        |------------------------|------------|---------------------------|-----------------------------------|
        | doubao-lite（基础）    | 8k         | 轻量通用对话              | 小程序、APP内嵌交互、低预算场景    |
        | doubao-pro-2（进阶）   | 64k        | 通用+多模态（图文/语音）  | 企业客服、内容生成、语音转文字交互 |
        | doubao-max-2（旗舰）   | 128k       | 复杂推理+工具调用         | 企业级数据分析、智能办公、API联动 |
        | doubao-vision（多模态）| 32k        | 图文生成+理解             | 海报设计、图片编辑、视觉问答      |

    - 定价（2025年11月）
        | 模型类型       | 输入单价（元/千Token） | 输出单价（元/千Token） | 套餐包优惠（100万Token） | 免费额度          |
        |----------------|------------------------|------------------------|--------------------------|-------------------|
        | doubao-lite    | 0.10                   | 0.20                   | 50元（省33%）            | 每日30万Token（不限期） |
        | doubao-pro-2   | 0.25                   | 0.50                   | 125元（省30%）           | 新用户60万Token（45天） |
        | doubao-max-2   | 0.60                   | 1.20                   | 300元（省25%）           | 新用户20万Token（30天） |
        | 多模态（图文）  | 0.30（文本）+ 0.15元/张图 | 0.60                   | 180元（省25%）           | 新用户10万Token+200张图 |

    - 模型特色
        - **多模态能力全面**：支持图文理解、语音交互、图片生成一体化，doubao-vision生成图片的分辨率可达4K，且支持“文本描述+参考图”混合生成，适合内容创作场景。
        - **企业级服务优势**：提供私有化部署、数据隔离、定制训练服务，适配金融、政务等敏感行业，支持SLA保障（可用性99.99%）。
        - **生态整合强**：无缝对接火山引擎的云服务器、存储、CDN等产品，适合需要全链路技术支持的企业用户。

4. 通义千问（阿里云）
    - 核心在线API模型矩阵
        | 模型名称               | 上下文窗口 | 核心定位                  | 适用场景                          |
        |------------------------|------------|---------------------------|-----------------------------------|
        | qwen-lite（基础）      | 8k         | 轻量通用对话              | 高并发场景、低成本交互            |
        | qwen-plus（进阶）      | 64k        | 通用高性能对话            | 内容生成、智能客服、知识问答      |
        | qwen-max-turbo（旗舰） | 128k       | 复杂推理+工具调用         | 企业级决策支持、数据分析、代码生成 |
        | qwen-vl-plus（多模态） | 32k        | 图文理解+生成             | 电商商品图生成、文档OCR、视觉推理 |

    - 定价（2025年11月）
        | 模型类型       | 输入单价（元/千Token） | 输出单价（元/千Token） | 套餐包优惠（100万Token） | 免费额度          |
        |----------------|------------------------|------------------------|--------------------------|-------------------|
        | qwen-lite      | 0.08                   | 0.16                   | 40元（省33%）            | 每日20万Token（不限期） |
        | qwen-plus      | 0.20                   | 0.40                   | 100元（省33%）           | 实名认证后100万Token（90天） |
        | qwen-max-turbo | 0.80                   | 1.60                   | 400元（省20%）           | 新用户30万Token（30天） |
        | 多模态（图文）  | 0.25（文本）+ 0.12元/张图 | 0.50                   | 150元（省30%）           | 新用户10万Token+150张图 |

    - 模型特色
        - **阿里云生态联动**：无缝对接阿里云OSS、RDS、函数计算等产品，适合在阿里云体系内开发的企业，可降低跨平台集成成本。
        - **通用性能强劲**：qwen-max-turbo在MMLU得分89.1，接近GPT-4o（92.3），支持复杂逻辑推理、多文档汇总、代码生成等全场景任务。
        - **中文优化极致**：针对中文语义、文化语境深度优化，在方言理解、古文解析、中文内容生成场景中准确率高于多数竞品。


四家服务商在线API横向对比

| 维度                | Deepseek（深度求索）       | KIMI（月之暗面）           | 火山方舟（字节跳动）           | 通义千问（阿里）           |
|---------------------|----------------------------|----------------------------|----------------------------|----------------------------|
| 核心优势            | 代码生成、专业领域推理     | 长文本处理、开源模型逼近闭源 | 多模态、企业级服务         | 阿里云生态、中文通用性能   |
| 最大上下文窗口      | 64k（代码模型）            | 512k（v2-max）             | 128k（doubao-max-2）       | 128k（qwen-max-turbo）     |
| 开源模型支持（在线API） | 无                        | 有（v2-open-70B/mini-1.8B） | 无                        | 无                        |
| 多模态能力          | 图文理解                   | 文本为主（支持长文本OCR）   | 图文/语音/生成一体化       | 图文理解+生成             |
| 免费额度            | 新用户50万Token（30天）    | 每日10万Token（不限期）    | 每日30万Token（lite模型）  | 实名认证100万Token（90天） |
| 适用场景            | 程序员、科研人员           | 长文本处理、成本敏感用户   | 企业级多模态、敏感行业     | 阿里云生态用户、通用场景   |

### 1.1.2 本地部署LLM

#### 1.1.2.1 本地部署的优劣势

本地部署LLM的核心优势集中在**隐私安全性**：所有数据（输入文本、对话历史、业务数据等）均在本地服务器或终端设备内处理，无需通过网络传输至第三方API服务器，从根源上避免了数据泄露、隐私泄露的风险，尤其适配金融、政务、医疗等对数据保密性要求极高的场景，或涉及个人敏感信息、企业核心数据的使用场景。

但对应的核心劣势是**高性能硬件依赖与部署成本**：本地模型的运行速度、推理效果直接取决于硬件配置，哪怕是轻量小模型，也需要至少4GB以上显存的GPU（或高性能CPU+16GB以上内存）才能流畅运行；若需部署中大型模型（13B及以上参数），则需搭载24GB以上显存的专业显卡（如NVIDIA A10、3090等），硬件采购成本较高。此外，本地部署还需具备一定技术门槛，需处理环境配置、依赖安装、模型优化（如量化压缩）、版本更新等问题，后续维护也需要持续投入精力，对非技术用户不够友好。

#### 1.2.2.2 适合本地部署的小模型

本地部署优先选择**参数规模小（1B-7B参数）、量化版本成熟、部署工具友好**的模型，以下为适配本地场景的主流小模型（更新于2025年11月）：

| 模型名称                | 核心特点                                                                 | 最低硬件要求                          | 部署工具/方式                          |
|-------------------------|--------------------------------------------------------------------------|---------------------------------------|---------------------------------------|
| 通义千问 qwen2-1.5B/7B-Instruct（2025最新版） | 中文优化极致，支持8k上下文，推理速度快；支持多轮对话、内容生成、简单工具调用；量化版本（4-bit/8-bit）成熟 | 1.5B：CPU 8GB+ / GPU 4GB+（显存）；7B：GPU 8GB+（显存） | 1. Ollama一键部署（`ollama run qwen2:1.5b-instruct`）；2. Hugging Face Transformers+PyTorch；3. 阿里云百炼本地部署工具 |
| 深度求索 Deepseek-chat-1.3B/6.7B | 保留Deepseek核心优势，轻量版仍支持基础代码生成、数学计算；中文理解能力强，4-bit量化后体积小，推理延迟低 | 1.3B：CPU 6GB+ / GPU 3GB+（显存）；6.7B：GPU 6GB+（显存） | 1. 官方提供量化模型（https://github.com/deepseek-ai/DeepSeek-Chat）；2. LMDeploy快速部署；3. Transformers+accelerate框架 |
| KIMI K2 moonshot-k2-3B/7B-Instruct | 长文本处理优化（轻量版支持32k上下文），开源可商用；对话一致性强，支持长文档本地解析；量化后资源占用低 | 3B：CPU 8GB+ / GPU 4GB+（显存）；7B：GPU 8GB+（显存） | 1. 月之暗面开源仓库（https://github.com/moonshot-ai/moonshot-k2）；2. Ollama（需手动导入模型）；3. vLLM提速部署 |

部署建议：
1. 优先使用量化模型（4-bit/8-bit），可将显存占用降低50%以上（如7B模型8-bit量化后仅需8GB显存）；
2. 无独立GPU时，可选择1.5B/3B级模型+CPU部署，但推理速度较慢（单轮响应约3-5秒），适合低并发场景；
3. 部署工具推荐：Ollama（零代码、一键启动）、LMDeploy（支持GPU提速）、vLLM（高并发场景首选）。

#### 1.1.2.3 使用Ollama本地部署小模型

1. Ollama工具安装（全系统通用流程）
    1. 访问Ollama官方网站（https://ollama.com/），下载安装包
    2. 安装验证：
       打开命令提示符，输入命令 `ollama --version`，执行后若显示版本信息则说明安装成功。

2. 模型部署
    1. 打开命令提示符/终端，输入部署部署指令：
         - 通义千问 qwen2-1.5B-Instruct：`ollama run qwen2:1.5b-instruct`
         - 深度求索 Deepseek-chat-1.3B：`ollama run deepseek-chat:1.3b`
         - KIMI K2 moonshot-k2-3B-Instruct：`ollama run moonshot-k2:3b-instruct`
    2. 模型下载：系统自动触发模型文件下载，终端将显示进度条（下载时长取决于网络带宽，通常1-10分钟）；
    3. 部署完成：终端显示“>>>”交互提示符，即完成本地部署。

3. 部署验证
    1. 功能验证：在终端“>>>”提示符后输入测试指令（如“请返回‘本地模型部署成功’”），按下回车；
    2. 结果确认：若模型在3-10秒内返回预期响应（如“本地模型部署成功”），则说明部署有效且性能良好，可正常使用；
    3. 退出操作：输入 `/exit` 命令，可关闭模型运行，释放系统资源。

4. 注意事项
    1. 模型选择：优先选择1.5B-3B参数轻量模型，平衡运行速度与硬件占用，避免因参数过大导致卡顿；
    2. 网络保障：模型下载阶段需保持网络稳定，若下载中断，重新执行部署命令即可继续下载；
    3. 资源监控：运行期间可通过系统任务管理器查看CPU/GPU占用情况，若占用过高，可关闭其他程序或切换更小参数模型。

## 1.2. 使用Python脚本调用LLM

### 1.2.1 调用在线API

#### 1.2.1.1 Deepseek（深度求索）API

```python
import requests
import os
from dotenv import load_dotenv

load_dotenv()  # 加载环境变量

def call_deepseek(prompt: str, model: str = "deepseek-chat") -> str:
    """
    调用Deepseek在线API
    :param prompt: 用户查询文本
    :param model: 模型名称（可选：deepseek-chat、deepseek-coder-v2等）
    :return: 模型响应文本
    """
    url = "https://api.deepseek.com/v1/chat/completions"
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {os.getenv('DEEPSEEK_API_KEY')}"  # 读取环境变量
    }
    # 请求参数（支持多轮对话，可扩展messages列表）
    payload = {
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
        "temperature": 0.7,  # 随机性：0-1，越低越严谨
        "max_tokens": 1024  # 最大输出Token数
    }
    try:
        response = requests.post(url, headers=headers, json=payload, timeout=30)
        response.raise_for_status()  # 抛出HTTP错误
        result = response.json()["choices"][0]["message"]["content"]
        return result
    except Exception as e:
        return f"Deepseek调用失败：{str(e)}"

# 测试调用
if __name__ == "__main__":
    print(call_deepseek("用Python写一个快速排序算法"))
```

#### 1.2.1.2 KIMI（月之暗面）API调用
```python
import requests
import os
from dotenv import load_dotenv

load_dotenv()

def call_kimi(prompt: str, model: str = "moonshot-v1-8k") -> str:
    """
    调用KIMI在线API（支持长文本模型）
    :param prompt: 用户查询文本（长文本可直接传入，无需分段）
    :param model: 模型名称（可选：moonshot-v1-8k、moonshot-v2-pro、moonshot-v2-open-70B等）
    :return: 模型响应文本
    """
    url = "https://api.moonshot.cn/v1/chat/completions"
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {os.getenv('KIMI_API_KEY')}"
    }
    payload = {
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
        "temperature": 0.6,
        "max_tokens": 2048  # 长文本模型可设更高（如v2-max支持512k窗口）
    }
    try:
        response = requests.post(url, headers=headers, json=payload, timeout=60)
        response.raise_for_status()
        return response.json()["choices"][0]["message"]["content"]
    except Exception as e:
        return f"KIMI调用失败：{str(e)}"

# 测试调用（长文本处理示例）
if __name__ == "__main__":
    long_text = "请总结以下文档核心观点：[此处粘贴万字文档内容]"
    print(call_kimi(long_text, model="moonshot-v2-pro"))
```

#### 1.2.1.3 火山方舟（字节跳动）API调用
```python
import requests
import os
from dotenv import load_dotenv

load_dotenv()

def call_volcengine(prompt: str, model: str = "doubao-pro") -> str:
    """
    调用火山方舟在线API（支持豆包系列模型）
    :param prompt: 用户查询文本
    :param model: 模型名称（可选：doubao-lite、doubao-pro-2、doubao-max-2等）
    :return: 模型响应文本
    """
    url = "https://ark.cn-beijing.volces.com/api/v3/chat/completions"
    headers = {
        "Content-Type": "application/json",
        "X-Volc-AccessKey": os.getenv("VOLCENGINE_ACCESS_KEY"),
        "X-Volc-SecretKey": os.getenv("VOLCENGINE_SECRET_KEY")
    }
    payload = {
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
        "temperature": 0.5,
        "max_tokens": 1024
    }
    try:
        response = requests.post(url, headers=headers, json=payload, timeout=30)
        response.raise_for_status()
        return response.json()["choices"][0]["message"]["content"]
    except Exception as e:
        return f"火山方舟调用失败：{str(e)}"

# 测试调用（多模态示例：图文问答，需传入图片URL）
if __name__ == "__main__":
    # 文本调用
    print(call_volcengine("解释什么是大语言模型"))
    # 多模态调用（需使用doubao-vision模型）
    # print(call_volcengine("分析这张图片的内容：https://example.com/image.jpg", model="doubao-vision"))
```

#### 1.2.1.4 通义千问（阿里）API调用
```python
import requests
import os
from dotenv import load_dotenv

load_dotenv()

def call_tongyi(prompt: str, model: str = "qwen-plus") -> str:
    """
    调用通义千问在线API（阿里云生态适配）
    :param prompt: 用户查询文本
    :param model: 模型名称（可选：qwen-lite、qwen-plus、qwen-max-turbo等）
    :return: 模型响应文本
    """
    url = "https://dashscope.aliyuncs.com/api/v1/services/aigc/text-generation/generation"
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {os.getenv('DASHSCOPE_API_KEY')}"  # 统一环境变量名
    }
    payload = {
        "model": model,
        "input": {"prompt": prompt},
        "parameters": {
            "temperature": 0.7,
            "max_tokens": 1024,
            "result_format": "text"  # 响应格式：text/json
        }
    }
    try:
        response = requests.post(url, headers=headers, json=payload, timeout=30)
        response.raise_for_status()
        return response.json()["output"]["text"]
    except Exception as e:
        return f"通义千问调用失败：{str(e)}"

# 测试调用
if __name__ == "__main__":
    print(call_tongyi("写一篇500字的企业数字化转型方案摘要"))
```

### 1.2.2 本地Ollama模型调用（支持qwen2、deepseek-chat、moonshot-k2）
**前置依赖**：安装Ollama Python客户端（首次使用需执行）
```bash

```
> 注意：调用前需确保本地Ollama服务已启动（部署模型后未关闭终端，或重新启动Ollama应用）

```python
import ollama

def call_ollama_local(prompt: str, model: str) -> str:
    """
    调用本地Ollama部署的模型
    :param prompt: 用户查询文本
    :param model: 模型名称（需与部署时一致）
    - 通义千问：qwen2:1.5b-instruct
    - 深度求索：deepseek-chat:1.3b
    - KIMI K2：moonshot-k2:3b-instruct
    :return: 模型响应文本
    """
    try:
        # 调用本地模型，支持多轮对话（可扩展messages参数）
        response = ollama.chat(
            model=model,
            messages=[{"role": "user", "content": prompt}],
            options={"temperature": 0.6, "num_ctx": 8192}  # num_ctx=上下文窗口大小
        )
        return response["message"]["content"]
    except Exception as e:
        return f"本地Ollama模型调用失败：{str(e)}（请检查模型是否已部署、Ollama服务是否启动）"

# 分模型测试调用
if __name__ == "__main__":
    # 1. 调用本地通义千问qwen2
    print("=== 本地qwen2响应 ===")
    print(call_ollama_local("介绍自己", model="qwen2:1.5b-instruct"))
    
    # 2. 调用本地深度求索deepseek-chat
    print("\n=== 本地deepseek-chat响应 ===")
    print(call_ollama_local("解一元二次方程：x²-5x+6=0", model="deepseek-chat:1.3b"))
    
    # 3. 调用本地KIMI K2
    print("\n=== 本地moonshot-k2响应 ===")
    print(call_ollama_local("总结‘人工智能发展趋势’的核心要点", model="moonshot-k2:3b-instruct"))
```

## 1.3 使用Cherry Studio调用LLM

### 1.3.1 Cherry Studio简介

Cherry Studio是一款低代码开发平台，专为集成大语言模型（LLM）设计，支持用户通过可视化界面快速构建AI应用，无需编写复杂代码。平台支持多种主流LLM模型的接入，包括本地部署模型和在线API模型，方便用户根据需求灵活选择。

工具特点：
- 可视化界面设计，无需编程
- 支持多种LLM模型集成
- 内置提示词管理和测试功能

### 1.3.2 下载与安装
1. 访问Cherry Studio官网：https://www.cherryai.com/
2. 下载并安装对应操作系统版本
3. 完成注册并登录

### 1.3.3 在线API模型配置与调用
**前置准备**：已完成1.1.1.2节环境变量配置，或准备好各服务商API密钥/端点信息。

1. **工具启动**：打开Cherry Studio，进入「模型管理」→「添加模型」→「在线API」。

2. **配置服务商**：
   1. Deepseek（深度求索）配置
    - 模型名称：自定义（如“Deepseek-chat”）
    - API类型：Chat Completions（OpenAI兼容）
    - 接口地址：`https://api.deepseek.com/v1/chat/completions`
    - 认证方式：Bearer Token
    - Token：粘贴环境变量中的`DEEPSEEK_API_KEY`
    - 默认模型：选择`deepseek-chat`（或其他需使用的模型）
    - 点击「测试连接」，显示“连接成功”即可保存。

   2. KIMI（月之暗面）配置
    - 模型名称：自定义（如“KIMI-v2-pro”）
    - API类型：Chat Completions（OpenAI兼容）
    - 接口地址：`https://api.moonshot.cn/v1/chat/completions`
    - 认证方式：Bearer Token
    - Token：粘贴环境变量中的`KIMI_API_KEY`
    - 默认模型：选择`moonshot-v1-8k`（或`moonshot-v2-pro`）
    - 点击「测试连接」，保存配置。

   3. 火山方舟（字节跳动）配置
    - 模型名称：自定义（如“火山豆包-pro”）
    - API类型：Chat Completions（火山方舟专属）
    - 接口地址：`https://ark.cn-beijing.volces.com/api/v3/chat/completions`
    - 认证方式：AccessKey + SecretKey
    - AccessKey：粘贴`VOLCENGINE_ACCESS_KEY`
    - SecretKey：粘贴`VOLCENGINE_SECRET_KEY`
    - 默认模型：选择`doubao-pro`（或`doubao-max-2`）
    - 点击「测试连接」，保存配置。

   4. 通义千问（阿里）配置
    - 模型名称：自定义（如“通义千问-plus”）
    - API类型：通义千问专属
    - 接口地址：`https://dashscope.aliyuncs.com/api/v1/services/aigc/text-generation/generation`
    - 认证方式：Bearer Token
    - Token：粘贴环境变量中的`DASHSCOPE_API_KEY`
    - 默认模型：选择`qwen-plus`（或`qwen-lite`）
    - 点击「测试连接」，保存配置。

3. **调用操作**：
   - 进入「对话界面」，右上角选择已配置的在线模型（如“KIMI-v2-pro”）；
   - 输入查询文本（如“分析一份法律合同的风险点”），点击「发送」；
   - 等待响应结果，支持多轮对话、导出响应内容。

### 1.3.4 本地Ollama模型连接与调用

**前置准备**：本地Ollama服务已启动（模型已部署，终端显示“>>>”或Ollama应用运行中）。

1. **模型连接**：
   - 打开Cherry Studio，进入「模型管理」→「添加模型」→「本地模型」→「Ollama」；
   - 服务地址：默认`http://localhost:11434`（Ollama默认端口，无需修改）；
   - 点击「刷新模型列表」，Cherry Studio会自动识别本地已部署的Ollama模型（如`qwen2:1.5b-instruct`、`deepseek-chat:1.3b`、`moonshot-k2:3b-instruct`）；
   - 选择需添加的模型，自定义模型名称（如“本地KIMI K2”），点击「保存」。

2. **调用操作**：
   - 进入「对话界面」，右上角选择已添加的本地模型（如“本地deepseek-chat”）；
   - 输入查询文本（如“写一个简单的Python脚本”），点击「发送」；
   - 响应无需网络传输，速度更快，支持查看CPU/GPU占用情况（「系统监控」面板）。

关键注意事项
1. 在线API调用需确保网络通畅，免费额度内避免高频大量调用；
2. 本地Ollama模型调用时，若响应卡顿，可在Cherry Studio「模型配置」中降低“上下文窗口大小”（如设为4096）；
3. 多轮对话时，Cherry Studio会自动保留历史上下文，无需手动传入历史消息；
4. 支持导出对话记录（PDF/Markdown格式），适合汇报或文档留存。

## 1.4 系统提示词设计

### 1.4.1 基础结构
```markdown
# 系统角色
你是[专业角色]，拥有[经验背景]。

## 任务要求
[详细描述任务内容和目标]

## 输出格式
[指定回答的结构和格式]
```

### 1.4.2 脑筋急转弯生成示例
```markdown
# 系统角色
你是专业谜语创作者，擅长设计中文脑筋急转弯。

## 难度要求
- 简单：适合儿童，答案直接
- 困难：包含双关语或多步推理

## 输出格式
**题目**：[脑筋急转弯题目]
**答案**：[简洁回答]
**解析**：[解释幽默原理]
```

## 1.5 课程作业：定制脑筋急转弯LLM

### 1.5.1 作业目标
使用Cherry Studio开发脑筋急转弯生成器，支持在线/本地模型切换，能生成不同难度的题目并解释答案。

### 1.5.2 具体要求
1. **功能要求**：
   - 支持简单/困难难度选择
   - 可输入自定义主题
   - 显示题目、答案和解析

2. **实现步骤**：
   - 配置至少一种模型（本地Ollama或在线API）
   - 设计系统提示词定义生成规则
   - 创建包含难度选择的交互界面
   - 测试并优化生成效果

3. **提交内容**：
   - 3个生成示例（含解析）
   - 应用界面截图
   - 开发总结（说明模型选择理由）