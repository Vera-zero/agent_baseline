# MRAG 方法各阶段实现分析

> **MRAG** = **M**etriever + **RAG**（Modular Temporal-Aware Retrieval-Augmented Generation）
> 核心思路：针对时间敏感型问答，设计了 **Metriever** 四模块渐进式重排管道——从粗粒度段落级到细粒度句子级，再结合**时间样条系数**对语义相似度加权，最终通过 LLM 阅读器生成答案。
> 支持数据集：TempRAGEval（TimeQA + SituatedQA）

---

## 整体架构

```
┌──────────────────────────────────────────────────────────────────────────┐
│                        MRAG 整体流水线                                    │
│                                                                          │
│  离线阶段：Stage 1 检索（Contriever / BM25）→ Top 1000 段落               │
│                          │                                               │
│  ┌───────────────────────▼──────────────────────────────────────────┐   │
│  │                   Metriever 四模块重排管道                          │   │
│  │                                                                  │   │
│  │  问题预处理 → 关键词提取(LLM) → ctx_keyword_rank(Top 1000→100)    │   │
│  │          → ctx_semantic_rank(Top 100) → QFS摘要 + snt_keyword    │   │
│  │          → snt_hybrid_rank(时间样条 × 语义分) → Top 句子集合       │   │
│  └──────────────────────────────────────────────────────────────────┘   │
│                          │                                               │
│  ┌───────────────────────▼──────────────────────────────────────────┐   │
│  │                    LLM 阅读器（Reader）                            │   │
│  │  Concat 模式：Top-3 段落 → Llama/GPT 生成答案                     │   │
│  │  Fusion 模式：Checker 过滤 → 并行摘要 → 时间排序 → CombinedReader  │   │
│  └──────────────────────────────────────────────────────────────────┘   │
│                          │                                               │
│                    评估：Acc / F1                                        │
└──────────────────────────────────────────────────────────────────────────┘
```

---

## 贯穿全文的运行例子

| 字段 | 值 |
|------|-----|
| **数据集** | TempRAGEval (TimeQA) |
| **问题** | *Who was the spouse of Donald Trump between 2010 and 2014?* |
| **time_relation** | `between` |
| **标准答案** | `Melania Trump` / `Melania Knauss` |
| **gold_evidence** | `Donald Trump married Slovenian model Melania Knauss in 2005.` |
| **exact** | 0（扰动日期，非 key date） |

---

## 阶段一：Stage 1 初始检索（离线预计算）

**代码位置**：`metriever.py: main()` 加载部分 + `pre_contriever.py`、`bm25/retrieve_bm25.py`

### 输入

| 字段 | 描述 |
|------|------|
| 问题列表 | TempRAGEval 数据集（TimeQA + SituatedQA） |
| 知识库 | enwiki-dec2021 段落集（`psgs_w100.json`，每段约 100 词） |

### 处理过程

支持三种 Stage 1 检索模式：

| 模式 | 方法 | 权重 |
|------|------|------|
| `contriever` | Facebook Contriever 向量检索 | — |
| `bm25` | BM25 关键词检索 | — |
| `hybrid` | 两者融合（Reciprocal Rank Fusion） | 语义 0.8 + BM25 0.2 |

**Hybrid 融合公式**：
```python
score = 0.8 × (1 / rank_contriever) + 0.2 × (1 / rank_bm25)
```
合并后再按融合分降序排列，取前 1000 个段落。

### 运行例子

**检索问题**：*Who was the spouse of Donald Trump between 2010 and 2014?*

Contriever Top-5（示意）：
```
1. Donald Trump | Donald John Trump is an American politician ...
2. Melania Trump | Melania Trump née Knauss, born 26 April 1970 ...
3. Ivana Trump  | Ivana Trump, born Ivana Marie Zelníčková ...
4. Marla Maples | Marla Ann Maples is an American actress ...
5. Donald Trump Jr. | Donald John Trump Jr. is an American businessman ...
```

### 输出

`TempRAGEval/contriever_output/TempRAGEval.json`：每条样本包含 `ctxs`（Top 1000 段落列表，每段有 `id`, `title`, `text`, `score`）

---

## 阶段二：问题时间预处理（Question Temporal Preprocessing）

**代码位置**：`metriever.py: main()` 第 397–468 行

### 输入

| 字段 | 描述 |
|------|------|
| `question`（str） | 原始问题，如 *"...between 2010 and 2014?"* |
| `time_relation`（str） | 数据集预标注的时间关系词，如 `"between"` |

### 处理过程

```
1. 以 time_relation 为分隔符切割问题：
   parts = question.split("between")
   no_time_question = parts[0]          # "Who was the spouse of Donald Trump "
   date_part = parts[-1]                # " 2010 and 2014?"

2. year_identifier(date_part) 提取年份：
   years = [2010, 2014]

3. 根据 time_relation 和 years 数量确定类型：
   len(years) > 1  → time_relation_type = "between"

4. 提取月份信息：
   months = [0, 0]  （无月份信息）

5. remove_implicit_condition(no_time_question)：
   检测 "first"/"last" 关键词 → implicit_condition
   "Who was the spouse of Donald Trump " → implicit_condition = None

6. 清理标点符号：
   normalized_question = "Who was the spouse of Donald Trump"
```

四种 `time_relation_type`：

| 类型 | 触发词 | 含义 |
|------|--------|------|
| `before` | before / as of / by / until | 在某时间点之前 |
| `after` | from / since / after | 在某时间点之后 |
| `between` | years > 1 | 在两时间点之间 |
| `other` | in / on / around / during | 精确时间点 |

### 运行例子

```python
time_relation      = "between"
time_relation_type = "between"
years              = [2010, 2014]
months             = [0, 0]
normalized_question = "Who was the spouse of Donald Trump"
implicit_condition  = None
```

### 输出

每条样本新增字段：`normalized_question`, `time_relation_type`, `years`, `months`, `implicit_condition`

---

## 阶段三：关键词提取（Keyword Extraction via LLM）

**代码位置**：`metriever.py` 第 478–516 行、`prompts.py: get_keyword_prompt()`、`utils.py: expand_keywords()`

### 输入

| 字段 | 描述 |
|------|------|
| `normalized_question`（str） | 去除时间表达后的问题 |
| LLM | Llama-3.1-8B-Instruct（vLLM 推理） |

### 处理过程

**Step 3.1：构建关键词提取 Prompt**

```
Your task is to extract keywords from the question. Response by a list of keyword strings.
Do not include pronouns, prepositions, articles.

<Question>
Who was the spouse of Donald Trump
</Question>
<Keywords>
```

**Step 3.2：过滤与验证**

```python
EXCL = ['time', 'years', 'for', 'new', 'recent', 'current', 'whom', 'who', 'out', 'place', 'not']
for kw in raw_keywords:
    if kw in EXCL: continue
    # 验证关键词必须在原问题中出现
    while kw.lower() not in question.lower():
        kw = ' '.join(kw.split()[:-1])   # 逐步缩短到能匹配的最长子串
    if kw != '': revised.append(kw)
revised = list(set(revised))
```

**Step 3.3：关键词扩展（`expand_keywords`）**

```python
# 1. 词形变化（WordNetLemmatizer）
"spouse" → "spouses", "spousal"

# 2. 动词形态变化（pattern.en.lexeme）
"married" → "marry", "marries", "marriage"

# 3. 数字同义词（number_map）
"1" → "one", "first"

# 4. 保留各词的 POS tag 作为 keyword_type_list：
# NNP（专有名词）权重 > NN（普通名词）> VB（动词）
```

### 运行例子

LLM 输出：`["spouse", "Donald Trump"]`

过滤后：`["spouse", "Donald Trump"]`

扩展后：
```python
expanded_keyword_list = ["spouse", "spouses", "Donald Trump", "Trump"]
keyword_type_list     = ["NN", "NN", "NNP", "NNP"]
```

### 输出

`question_keyword_map[normalized_question] = (expanded_keyword_list, keyword_type_list)`

---

## 阶段四：ctx_keyword_rank（段落关键词排序）

**代码位置**：`metriever.py` 第 555–572 行、`utils.py: count_keyword_scores()`

### 输入

| 字段 | 描述 |
|------|------|
| Stage 1 Top 1000 段落 | `ctxs` 列表（来自 Contriever/BM25） |
| `expanded_keyword_list` | 扩展关键词列表 |
| `keyword_type_list` | 每个关键词的 POS 类型（NNP > NN > VB） |

### 处理过程

```python
def count_keyword_scores(text, keywords, keyword_types):
    score = 0
    for kw, kw_type in zip(keywords, keyword_types):
        if kw.lower() in text.lower():
            weight = 3 if kw_type == 'NNP' else (2 if kw_type in ['NN','NNS'] else 1)
            score += weight
    return score

for ctx in ctxs[:1000]:
    text = ctx['title'] + ' ' + ctx['text']
    score = count_keyword_scores(text, expanded_keyword_list, keyword_type_list)
    ctx_kw_scores.append((ctx, score))

ctx_kw_scores.sort(key=lambda x: x[1], reverse=True)
latest_ctxs = [tp[0] for tp in ctx_kw_scores][:100]  # 保留 Top 100
```

### 运行例子

| 排名 | 标题 | 关键词命中 | 得分 |
|------|------|----------|------|
| 1 | Donald Trump | spouse(2) + Donald Trump(3×2) | 8 |
| 2 | Melania Trump | spouse(2) + Trump(3) | 5 |
| 3 | Ivana Trump | Trump(3) | 3 |
| 900 | Random Sports Article | — | 0 |

### 输出

`ex['ctx_keyword_rank']`：Top 100 段落，按关键词分降序排列

---

## 阶段五：ctx_semantic_rank（段落语义重排）

**代码位置**：`metriever.py` 第 575–617 行

### 输入

| 字段 | 描述 |
|------|------|
| `ctx_keyword_rank`（Top 100） | 阶段四筛选出的 100 个段落 |
| `normalized_question` | 去时间表达的查询文本 |
| 语义编码模型 | `nvidia/NV-Embed-v2`（或其他跨编码器） |

### 处理过程

**NV-Embed-v2（双塔向量相似度）**：
```python
query_prefix = "Instruct: Given a question, retrieve passages that answer the question\nQuery: "
query_embeddings  = model.encode([normalized_question], instruction=query_prefix)
passage_embeddings = model.encode([title+' '+text for ctx in latest_ctxs])
# 余弦相似度
query_embeddings  = F.normalize(query_embeddings, p=2, dim=1)
passage_embeddings = F.normalize(passage_embeddings, p=2, dim=1)
scores = (query_embeddings @ passage_embeddings.T).view(-1)
```

注意：使用 `normalized_question`（去掉时间表达）而非原始问题进行语义匹配，避免时间词干扰实体语义检索。

按分数降序重排，保存为 `ctx_semantic_rank`。

### 运行例子

| 排名 | 标题 | 语义得分 |
|------|------|---------|
| 1 | Melania Trump | 0.91（spouse + Trump 高度相关） |
| 2 | Donald Trump | 0.88 |
| 3 | Ivana Trump | 0.72 |
| 4 | Marla Maples | 0.65 |
| … | … | … |

### 输出

`ex['ctx_semantic_rank']`：100 个段落按语义相似度降序排列

---

## 阶段六：QFS 摘要生成 + snt_keyword_rank（句子关键词排序）

**代码位置**：`metriever.py` 第 629–694 行、`prompts.py: LLMGenerations()`

### 输入

| 字段 | 描述 |
|------|------|
| Top `QFS_topk=5` 段落 | 语义排名最高的 5 个段落 |
| `normalized_question` | 查询问题 |
| LLM | Llama-3.1-8B-Instruct |

### 处理过程

**Step 6.1：对 Top-5 段落生成 QFS 摘要**

```python
def LLMGenerations(document, question):
    prompt = f"""Generate a short and concise summary from the given document that is relevant to the following question.
Document: {document}
Question: {question}
Summary:"""
    return prompt
```

LLM 生成聚焦于问题的简洁摘要（≤50 词），替代原文段落作为补充句子。

**Step 6.2：段落拆分为句子，QFS 摘要作为额外句子加入**

```python
for idx, ctx in enumerate(latest_ctxs):
    snts = sent_tokenize(ctx['text'])
    snts = [ctx['title'] + ' ' + snt for snt in snts]  # 拼接标题
    if idx < QFS_topk:
        summary = summary_responses[idx]   # QFS 摘要
        if 'None' not in summary:
            snts.append(summary)
    # 对每个句子打关键词分
    for snt in snts:
        snt_kw_score = count_keyword_scores(ctx['title']+' '+snt, ...)
        sentence_tuples.append((ctx_id, snt, snt_kw_score))
```

**Step 6.3：按句子关键词分重排，段落顺序由其最佳句子排名决定**

```python
sentence_tuples.sort(key=lambda x: x[2], reverse=True)
# 以首次出现顺序去重，重建段落排名
for ctx_id, snt, score in sentence_tuples:
    if ctx_id not in id_included:
        id_included.append(ctx_id)
        latest_ctxs.append(get_ctx_by_id[ctx_id])
```

### 运行例子

**Top-1 段落（Melania Trump）**的 QFS 摘要（LLM 生成）：
```
Melania Trump married Donald Trump in 2005. She served as the First Lady of the United States from 2017 to 2021.
```

段落内所有句子（含 QFS）的关键词分：

| 句子 | 关键词命中 | 分值 |
|------|----------|------|
| `Melania Trump married Donald Trump in 2005.` | spouse类(2)+Trump(3+3) | 8 |
| QFS: `Melania Trump married Donald Trump in 2005...` | 同上 | 8 |
| `Donald Trump married Marla Maples in 1993...` | Trump(3)+spouse(2) | 5 |
| `She was born in Novo Mesto, Slovenia.` | — | 0 |

得分最高句子所在段落（Melania Trump）排名第 1。

### 输出

`ex['snt_keyword_rank']`：段落列表，由其包含的最高分句子决定排名

---

## 阶段七：snt_hybrid_rank（时间-语义混合排序）

**代码位置**：`metriever.py` 第 703–778 行、`get_spline_function()`、`get_temporal_coeffs()`

> 这是 MRAG 最核心的创新模块——用**时间样条系数**对语义分加权。

### 输入

| 字段 | 描述 |
|------|------|
| Top `snt_topk=200` 个句子 | 阶段六的句子元组列表 |
| `years`, `time_relation_type`, `implicit_condition` | 阶段二提取的时间信息 |
| 语义编码模型 | NV-Embed-v2 |

### 处理过程

**Step 7.1：构建时间样条函数（`get_spline_function`）**

根据时间类型确定 `[x_start, x_end]` 和 `[y_start, y_end]`：

```python
low = 0.6
if time_relation_type == "between":
    start, end = 2010, 2014
elif time_relation_type == "after":
    start, end = years[0], years[0]+50
elif time_relation_type == "before":
    start, end = years[0]-50, years[0]

# implicit_condition 决定斜率方向：
if implicit_condition == 'first':
    y_points = [1, low]    # 越早越好
else:
    y_points = [low, 1]    # 越晚越好（或无偏好时默认）

# 线性插值
spline = interp1d([start, end], y_points, kind='linear')
# 时间区间外：coeff = 0.5（中性）
```

对于例子（between 2010 and 2014，无 implicit_condition）：
```
x: [2010 ─────────── 2014]
y: [0.6  ─────────── 1.0 ]   # 越接近2014越好（默认"last"）
区间外：coeff = 0.5
```

**Step 7.2：为每个句子计算时间系数（`get_temporal_coeffs`）**

```python
for snt in top_200_sentences:
    snt_years = year_identifier(snt)   # 提取句子中的年份数字
    # 在 [start, end] 区间内找最近/最远年份
    relevant_years = [y for y in snt_years if start <= y <= end]
    if implicit_condition == 'first':
        closest_year = min(relevant_years)
    else:
        closest_year = max(relevant_years)   # 默认取最晚的
    coeff = spline(closest_year)           # 查表得系数
    # 若无相关年份：coeff = 0.5
```

**Step 7.3：计算混合分数**

```python
# 先用 normalized_question（去时间）对句子做语义打分
semantic_scores = model.encode([normalized_question], instruction=query_prefix)

# 时间-语义混合
final_score = hybrid_base * semantic_score + (1-hybrid_base) * semantic_score * temporal_coeff
#   hybrid_base = 0 时：final_score = semantic_score × temporal_coeff
#   时间系数直接放大/缩小语义分
```

按 `final_score` 降序排列句子 → 重建段落排名 → `snt_hybrid_rank`

### 运行例子

句子得分计算过程：

| 句子 | snt_years | coeff（between 2010-2014） | semantic_score | final_score |
|------|----------|--------------------------|---------------|-------------|
| "Melania Trump married Trump in 2005." | [2005] | 0.5（区间外） | 0.88 | 0.88×0.5 = **0.44** |
| "First Lady of US from 2017 to 2021." | [2017, 2021] | 0.5（区间外） | 0.76 | 0.76×0.5 = **0.38** |
| "Melania and Donald were together in 2013." | [2013] | 0.88（spline(2013)≈0.88） | 0.85 | 0.85×0.88 = **0.75** |
| QFS: "Melania married Trump in 2005..." | [2005] | 0.5 | 0.90 | 0.90×0.5 = **0.45** |

→ 包含 2013 的句子排名最高，其所在段落（Melania Trump）排名第一

### 输出

- `ex['snt_hybrid_rank']`：段落列表（按最高分句子排名）
- `ex['top_snts']`：Top 20 句子拼接文本（`\n\n` 分隔）

---

## 阶段八：LLM 阅读器（Reader）

**代码位置**：`reader.py: main()`

支持两种范式：

---

### 范式 A：Concat 模式（直接拼接）

**输入**：Top-3 段落文本（`snt_hybrid_rank[:3]`）

```python
text = '\n\n'.join([ctx['title'] + ' | ' + ctx['text'].strip() for ctx in top3])
prompt = c_prompt(question, text)
```

`c_prompt` 为 `CombinedReader`（含 Few-shot ICL 示例），要求模型在 `<Thought>` 推理后输出 `<Answer>`。

---

### 范式 B：Fusion 模式（Checker → 并行摘要 → CombinedReader）

**Step 8.1：Checker 筛选**

对每个段落调用 `GradeDocuments(context, normalized_question)`，LLM 返回 Yes/No：

```
<Context>
Melania Trump | Melania Trump née Knauss, born 26 April 1970, is a Slovenian-American ...
She married Donald Trump on January 22, 2005 ...
</Context>
<Question>
Who was the spouse of Donald Trump?
</Question>
<Thought>
The context mentions Melania Trump married Donald Trump, which answers the question.
</Thought>
<Response>
Yes
</Response>
```

**Step 8.2：并行摘要生成（LLMGenerations）**

对所有 Checker 通过的段落，LLM 生成简短摘要（≤50 词）：

```
"Melania Trump married Donald Trump on January 22, 2005. She served as the First Lady from 2017 to 2021."
"Donald Trump married Ivana in 1977 (divorced 1991), Marla Maples in 1993 (divorced 1999), Melania in 2005."
```

**Step 8.3：按年份排序摘要**

```python
years = [year_identifier(s) for s in summarizations]
sorted_data = sorted(zip(summarizations, min_years), key=lambda x: x[1])
```

**Step 8.4：CombinedReader**

```python
combined_prompt = CombinedReader('\n\n'.join(sorted_summarizations), question)
```

Few-shot ICL 示例（含时间推理）：
```
<Context>
Sam Nujoma: president of Namibia (1990-2005)
Hifikepunye Pohamba was the president of Namibia between 2005 and 2015.
Hage Geingob is the incumbent president of Namibia from 2015.
</Context>
<Question>
Who was the last Namibia's president from 29 January 2002 to January 2016?
</Question>
<Thought>
...Hage Geingob is the last... Therefore, the answer is Hage Geingob.
</Thought>
<Answer>
Hage Geingob
</Answer>
```

**降级回退策略**：

```python
if check_no_knowledge(rag_pred):      # 无知识 → Concat 模式
    rag_pred = concat_reader(top3_ctx)
if check_no_knowledge(rag_pred):      # 仍无知识 → 零上下文 LLM
    rag_pred = zero_context_cot(question)
```

### 运行例子

**Concat 模式 LLM 响应**：
```
<Thought>
According to the context, Donald Trump married Melania Knauss in 2005.
2010-2014 is after 2005 and Melania was still his spouse.
Therefore, the answer is Melania Trump.
</Thought>
<Answer>
Melania Trump
</Answer>
```

**输出**：`rag_pred = "Melania Trump"`

---

## 阶段九：零上下文参数预测（Parametric Prediction）

**代码位置**：`reader.py` 第 340–344 行

### 输入 / 处理

与 RAG 预测**并行**进行，直接用 LLM 的参数记忆回答问题（不提供任何检索上下文）：

```python
prompts = [zc_prompt(ex['question']) for ex in examples]
# 或 CoT 版本：
prompts = [zc_cot_prompt(ex['question']) for ex in examples]
param_preds = call_pipeline(args, prompts, max_tokens=400)
```

用于对比**RAG 预测 vs 参数预测**的性能差距，分析检索的实际贡献。

### 输出

`ex['param_pred']`、`ex['param_acc']`、`ex['param_f1']`

---

## 阶段十：评估

**代码位置**：`reader.py: eval_reader()`、`temp_eval.py: normalize()`

### 输入

| 字段 | 描述 |
|------|------|
| `rag_pred` | LLM 生成的 RAG 答案 |
| `answers` | 标准答案列表 |
| `source` | `timeqa` 或 `situatedqa` |

### 处理过程

**规范化**：
```python
def normalize(s):
    s = s.lower().strip()
    s = re.sub(r'\b(a|an|the)\b', ' ', s)
    s = re.sub(r'[^\w\s]', '', s)   # 去标点
    return ' '.join(s.split())

rag_acc = int(normalize(rag_pred) in [normalize(ans) for ans in answers])
```

**Token F1**：
```python
def max_token_f1(gold_answers, pred):
    pred_tokens = pred.split()
    scores = []
    for gold in gold_answers:
        gold_tokens = gold.split()
        common = Counter(pred_tokens) & Counter(gold_tokens)
        num_same = sum(common.values())
        f1 = 2*num_same / (len(pred_tokens)+len(gold_tokens))
        scores.append(f1)
    return max(scores)
```

**分组输出**（按子集 + 时间类型）：

```
--- TimeQA ---
w/o date:       acc=x.xx, f1=x.xx
w/ key date:    acc=x.xx, f1=x.xx
w/ perturb date: acc=x.xx, f1=x.xx    ← MRAG 重点改进的子集

--- SituatedQA ---
...
```

同时输出检索召回率（`eval_recall`）对比各中间阶段：
```
ctx_keyword_rank:  Recall@5=xx%
ctx_semantic_rank: Recall@5=xx%
snt_keyword_rank:  Recall@5=xx%
snt_hybrid_rank:   Recall@5=xx%   ← 最高
```

### 运行例子

```
predictions  = "Melania Trump"
answers      = ["Melania Trump", "Melania Knauss"]

normalize("Melania Trump") = "melania trump"
normalize("Melania Trump") = "melania trump"

rag_acc = 1
rag_f1  = 1.0

param_pred = "Melania Trump"  （LLM 参数记忆也知道）
param_acc  = 1
```

---

## 完整数据流总结

```
TempRAGEval (TimeQA + SituatedQA)
    每条: {question, answers, gold_evidences, time_relation, exact, source}
                │
                ▼ 阶段一（离线）：Contriever + BM25 检索
    ctxs: Top 1000 Wikipedia 段落
                │
                ▼ 阶段二：时间预处理
    normalized_question = "Who was the spouse of Donald Trump"
    time_relation_type = "between", years = [2010, 2014]
    implicit_condition = None
                │
                ▼ 阶段三：关键词提取（LLM）
    expanded_keywords = ["spouse", "spouses", "Donald Trump", "Trump"]
                │
                ▼ 阶段四：ctx_keyword_rank（关键词过滤）
    Top 100 段落（按关键词命中数排序）
                │
                ▼ 阶段五：ctx_semantic_rank（NV-Embed-v2 重排）
    Top 100 段落（按语义相似度重排）
                │
                ▼ 阶段六：QFS 摘要 + snt_keyword_rank（句子级关键词排序）
    所有句子 + LLM 生成的 QFS 摘要 → 按关键词分重排段落
                │
                ▼ 阶段七：snt_hybrid_rank（时间样条 × 语义分）
    时间系数 coeff = spline(closest_year_in_sentence)
    final_score = semantic_score × coeff
    → Top 20 句子 + 段落重排
                │
                ▼ 阶段八A：Concat Reader（Llama-3.1-8B-Instruct）
    context = Top-3 段落文本拼接
    → CombinedReader Prompt → rag_pred = "Melania Trump"
                │
              （并行）
                ▼ 阶段九：零上下文参数预测
    zc_prompt(question) → param_pred = "Melania Trump"
                │
                ▼ 阶段十：评估
    rag_acc=1, rag_f1=1.0, param_acc=1, param_f1=1.0
    保存 CSV: answered/xxx_results.csv
```

---

## 各阶段 I/O 速查表

| 阶段 | 名称 | 输入 | 输出 | 关键代码 |
|------|------|------|------|----------|
| 1 | Stage 1 检索 | 问题 + Wikipedia 语料 | Top 1000 段落 `ctxs` | Contriever / BM25 / 混合 |
| 2 | 时间预处理 | 问题 + time_relation | normalized_question + years + time_type | `year_identifier()` + `remove_implicit_condition()` |
| 3 | 关键词提取 | normalized_question + LLM | expanded_keyword_list + keyword_type_list | `get_keyword_prompt()` + `expand_keywords()` |
| 4 | ctx_keyword_rank | Top 1000 + 关键词 | Top 100 段落 | `count_keyword_scores()` |
| 5 | ctx_semantic_rank | Top 100 + NV-Embed-v2 | Top 100 按语义重排 | NV-Embed-v2 余弦相似度 |
| 6 | QFS + snt_keyword | Top 100 + LLM + 关键词 | 段落+QFS摘要 + 按句子关键词重排 | `LLMGenerations()` + `sent_tokenize()` |
| 7 | snt_hybrid_rank | Top 200 句子 + 时间信息 | 时间×语义混合排序的段落+句子 | `get_spline_function()` + `get_temporal_coeffs()` |
| 8 | Reader | Top-3 段落 + 问题 | rag_pred + rag_acc + rag_f1 | `CombinedReader()` / `GradeDocuments()` |
| 9 | 参数预测 | 问题（无上下文） | param_pred + param_acc | `zc_prompt()` |
| 10 | 评估 | rag_pred + gold answers | Acc / F1（分子集分时间类型） | `normalize()` + `max_token_f1()` |

---

## 关键设计亮点

### 1. 时间样条系数（Temporal Spline Coefficient）
MRAG 最核心的创新：
```
final_score = semantic_score × spline(year_in_sentence)
```
样条函数将时间约束（before/after/between）编码为 `[0.5, 1.0]` 的连续系数，对语义分进行**时间加权放大**，而非简单过滤——使模型更关注时间上相关的句子。

### 2. 四级渐进式过滤（Coarse-to-Fine）

```
Top 1000 (Stage1)
  → Top 100 (ctx_keyword)
    → Top 100 (ctx_semantic, 细粒度排名更新)
      → 句子级 (snt_keyword + QFS)
        → Top 200 句子 (snt_hybrid, 时间×语义)
```

每级从不同维度过滤，粗粒度用计算廉价的关键词匹配，细粒度用计算昂贵的语义模型，平衡效率与质量。

### 3. QFS 摘要作为额外句子注入
对 Top-5 段落用 LLM 生成 Query-Focused Summary，将其**作为额外句子注入句子池**，弥补原文中可能分散在多段的时间信息。

### 4. 样本分类评估（no_time / exact / perturb）
将测试集分为三类进行分析：
- `no_time`：无时间约束的问题（基线对比）
- `exact`：问题中的日期与黄金文档完全匹配
- `not_exact`（扰动）：日期表述与黄金文档不同（MRAG 重点改进场景）

---

## 与其他方法的对比

| 维度 | MRAG | QAAP | TG-LLM | NeSTR | RTQA |
|------|------|------|--------|-------|------|
| **时间处理核心** | 时间样条系数×语义分 | Python datetime IoU | 结构化时间图 | 谓词逻辑 | 占位符引用 |
| **检索方式** | Contriever/BM25 + 四模块重排 | Wikipedia API实时检索 | 无（上下文给定） | 上下文给定 | BGE-M3+FAISS |
| **句子级处理** | 是（snt_keyword + hybrid） | 否（段落级分片） | 否 | 否 | 否 |
| **QFS 摘要** | 是（LLM生成，注入句子池） | 否 | 否 | 否 | 否 |
| **LLM 训练** | 无 | 无 | LoRA SFT | 无 | 无 |
| **支持问题类型** | before/after/between/other | 时间区间匹配 | 时间段查询 | 单一时间点 | 复杂多跳 |
| **模型** | Llama-3.1-8B + NV-Embed-v2 | GPT-3.5 | Llama-2-13b | GPT/Qwen | BGE-M3 + GPT |
