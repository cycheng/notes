# Contents
  * Kubernetes (k8s)
  * ASAP - real2sim2real
  * DeepSeek
    * v2
  * high-flyer

## Kubernetes (k8s)
* Kubernetes is a portable, extensible, open source platform for managing containerized workloads and services, that facilitates both declarative configuration and automation.
* Virtualized deployment era:
  * Each VM is a full machine running all the components, including its own operating system, on top of the virtualized hardware.
* Container deployment era:
  * Containers are similar to VMs, but they have relaxed isolation properties to share the Operating System (OS) among the applications.
    Therefore, containers are considered lightweight.
  * 与传统的虚拟机不同，容器共享主机操作系统的内核，因此更加高效和便携。 

## ASAP - real2sim2real
https://hao.cnyes.com/post/133793

## DeepSeek
### DeepSeek-VL2 - 2024 Jun 19
* https://github.com/deepseek-ai/DeepSeek-VL2
* a strong Mixture-of-Experts (MoE) language model
* 236B total parameters, 21B are activated for each token,  context length of 128K tokens
* innovative architectures
  * Multi-head Latent Attention (MLA)
    * MLA guarantees efficient inference through significantly compressing the Key-Value (KV) cache into a latent vector
  * DeepSeekMoE
    * enables training strong models at an economical cost through sparse computation
  * v.s. DeepSeek 67B:
    * significantly stronger performance
    * saves 42.5% of training costs
    * reduces the KV cache by 93.3%
    * boosts the maximum generation throughput to 5.76 times
  * pretrain: on a high-quality and multi-source corpus consisting of 8.1T tokens
  * Supervised Fine-Tuning (SFT) and Reinforcement Learning (RL)
![image](https://github.com/user-attachments/assets/bde50e68-af54-4627-8dd2-239be143dbe6)

1. Introduction
* we introduce MLA, an attention mechanism equipped with low-rank key-value joint compression.
  Empirically, MLA achieves superior performance compared with MHA, significantly reduces the KV cache during inference, thus boosting the inference efficiency.
* For Feed-Forward Networks (FFNs), we follow the DeepSeekMoE architecture (Dai et al., 2024), which adopts
  fine-grained expert segmentation and shared expert isolation for higher potential in expert specialization 
* We construct a high-quality and multi-source pre-training corpus consisting of 8.1T tokens.
* DeepSeek-V2 Chat (SFT): we collect 1.5M conversational sessions, which encompass various domains such as math, code, 
  writing, reasoning, safety, and more, to perform Supervised Fine-Tuning (SFT)  
* DeepSeek-V2 Chat (RL): Finally, we employ Group Relative Policy Optimization (GRPO) to further align the model with
  human preference
![image](https://github.com/user-attachments/assets/44477fe4-c33f-4cce-9cc4-40c9f1257ed8)
* We evaluate DeepSeek-V2 on a wide range of benchmarks in English and Chinese
  * even with only 21B activated parameters, V2 is the strongest open-source MoE language model
  * compared with DeepSeek 67B, V2 saves 42.5% of training costs, reduces the KV cache by 93.3%, and boosts the
    maximum generation throughput to 5.76 times
  * V2 Chat (RL) achieves
    - 38.9 length-controlled win rate on AlpacaEval 2.0 (Dubois et al., 2024),
    - 8.97 overall score on MT-Bench (Zheng et al., 2023), and
    - 7.91 overall score on AlignBench (Liu et al., 2023).
  * DeepSeek-V2-Lite
    - 15.7B parameters
    - 2.4B are activated for each token.

2. Architecture
  * For attention, we design MLA, which utilizes low-rank key-value joint compression to
    eliminate the bottleneck of inference-time key-value cache, thus supporting efficient inference.
  * For Feed-Forward Networks (FFNs), we adopt the DeepSeekMoE architecture (Dai et al., 2024), a
    high-performance MoE architecture that enables training strong models at an economical cost.

2.1. Multi-Head Latent Attention: Boosting Inference Efficiency

2.1.1. Preliminaries: Standard Multi-Head Attention

#### Multi-Head Attention (MHA) 解釋 by DeepSeek V3:
* https://chat.deepseek.com/a/chat/s/e579d7f5-60ad-4fa2-87c3-88b37221b0a2

Multi-Head Attention (MHA) 是 Transformer 模型中的核心機制，用於捕捉輸入序列中不同位置的關係。以下是其關鍵點：
* Self-Attention 基礎
  * 通過計算輸入序列中每個元素與其他元素的相關性，生成加權表示。
  * 輸入序列 $$X = (x_1, x_2, \dots, x_n)$$\
    通過線性變換得到查詢（Query）、鍵（Key）和值（Value）向量：\
    $$Q = XW_Q, \quad K = XW_K, \quad V = XW_V$$
  * 注意力得分通過點積計算：\
    $$\text{Attention}(Q, K, V) = \text{softmax}\left(\frac{QK^T}{\sqrt{d_k}}\right)V$$\
    其中 $$\( d_k \)$$ 是鍵向量的維度。

* Multi-Head Attention
  - 通過並行計算多個 Self-Attention 頭，捕捉不同的特徵表示。
  - 每個頭獨立計算注意力：\
    $head_i = \text{Attention}(QW_{Q_i}, KW_{K_i}, VW_{V_i})$

  - 所有頭的輸出拼接後通過線性變換：\
    $\text{MHA}(Q, K, V) = \text{Concat}(\text{head}_1, \text{head}_2, \dots, \text{head}_h)W_O$\
    $其中 \ h \ 是頭數，\ W_O \ 是輸出權重矩陣。$

* 優點
  - **並行計算**：多個頭可以同時處理不同特徵。
  - **多樣化特徵捕捉**：每個頭關注不同的子空間，增強模型表達能力。

* 應用
  - **Transformer**：MHA 是其編碼器和解碼器的核心。
  - **BERT、GPT**：這些預訓練模型也依賴 MHA 進行上下文建模。

* 總結
  - **Multi-Head Attention** 通過並行計算多個 Self-Attention 頭，增強了模型捕捉多樣化特徵的能力，是 Transformer 模型成功的關鍵。

#### Multi-Head Attention (MHA) 解釋 by This paper 2.1.1:

$\text{Let } d \text{ be the embedding dimension, }$\
$n_h \text{ be the number of attention heads, }$\
$d_h \text{ be the dimension per head, and }$\
$h_t \in \mathbb{R}^d \text{ be the attention input of the 𝑡-th token at an attention layer.}$\
Standard MHA first produces $q_t, k_t, v_t \in \mathbb{R}^{d_h n_h}$ through three matrices $W^Q, W^K, W^V \in \mathbb{R}^{d_h n_h \times d}$
![image](https://github.com/user-attachments/assets/812a3f6e-4157-41e6-a204-f28eec731fdf)

![image](https://github.com/user-attachments/assets/8868b2d6-7093-4bcf-a353-211a75412d8d)
Figure 3 | Simplified illustration of Multi-Head Attention (MHA), Grouped-Query Attention (GQA), Multi-Query Attention (MQA), and Multi-head Latent Attention (MLA). Through
jointly compressing the keys and values into a latent vector, MLA significantly reduces the KV
cache during inference.

Then, $q_t, k_t, v_t$ will be sliced into $n_h$ heads for the multi-head attention computation:
![image](https://github.com/user-attachments/assets/37c9eb32-0668-4a36-9d21-b9e9730ba823)

* where $q_{t,i}, k_{t,i}, v_{t,i} \in \mathbb{R}^{d_h}$ denote the query, key, and value of the 𝑖-th attention head, respectively;
* $W^o \in \mathbb{R}^{d \times d_h n_h}$ denotes the output projection matrix.
* During inference, all keys and values need to be cached to accelerate inference, so MHA needs to cache $2n_h d_h l$ elements for each token. 
* In model deployment, this heavy KV cache is a large bottleneck that limits the maximum batch size and sequence length.

2.1.2. Low-Rank Key-Value Joint Compression
* The core of MLA is the low-rank joint compression for keys and values to reduce KV cache:
![image](https://github.com/user-attachments/assets/19d3de7f-50ba-4d63-ad23-1e652a221c28)


3. 

## high-flyer
* https://www.high-flyer.cn/blog/llama2-1/
* https://github.com/deepseek-ai/DeepSeek-VL2


