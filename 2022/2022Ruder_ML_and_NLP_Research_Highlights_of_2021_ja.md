---
original: 
author: [Sebastian Ruder](https://ruder.io)
date: 2022_0124
layout: default
---

# 機械学習 ML と自然言語処理 NLP 研究 2021 年ハイライト 
<!-- # ML and NLP Research Highlights of 2021 -->

### [Sebastian Ruder](https://ruder.io)

この記事では 2021 年の ML と NLP における複数のインパクトのある分野での進捗をまとめる。
<!-- This post summarizes progress across multiple impactful areas in ML and NLP in 2021. -->

24 Jan 2022 [31 min read]

2021 年は機械学習 (ML) と自然言語処理 (NLP) において，多くのエキサイティングな進歩が見られた。
本記事では，私が最も感銘を受けた論文と研究分野を取り上げる。 
私が知っている論文を取り上げるようにしたが，関連する多くの論文を見逃している可能性がある。
そのような論文や，あなたが感銘を受けた論文をコメント欄で自由に紹介して欲しい。
以下は，そのハイライトである。
<!-- 2021 saw many exciting advances in machine learning (ML) and natural language processing (NLP).  
In this post, I will cover the papers and research areas that I found most inspiring.  
I tried to cover the papers that I was aware of but likely missed many relevant ones. 
Feel free to highlight them as well as ones that you found inspiring in the comments.
I discuss the following highlights: -->

1. [普遍モデル Universal Models](#1universalmodels)
2. [大規模マルチタスク学習 Massive Multi-task Learning](#2massivemultitasklearning)
3. [トランスフォーマーを越えて Beyond the Transformer](#3beyondthetransformer)
4. [プロンプト Prompting](#4prompting)
5. [効率的方法 Efficient Methods](#5efficientmethods)
6. [ベンチマーク Benchmarking](#6benchmarking)
7. [条件付き画像生成 Conditional Image Generation](#7conditionalimagegeneration)
8. [機械学習の科学応用 ML for Science](#8mlforscience)
9. [プログラム合成 Program Synthesis](#9programsynthesis)
10. [バイアス Bias](#10bias)
11. [検索拡張 Retrieval Augmentation](#11retrievalaugmentation)
12. [トークンに依存しないモデル Token-free Models](#12tokenfreemodels)
13. [時間的適応 Temporal Adaptation](#13temporaladaptation)
14. [データの重要性 The Importance of Data](#14theimportanceofdata)
15. [メタ学習 Meta-learning](#15metalearning)

## 1. 普遍モデル
<!-- # 1. Universal Models-->

<center>
<img src="https://ruder.io/content/images/2022/01/xslr.png" width="94%"><br/>
<!-- Self-supervised speech representation learning")<br/> -->
<!-- ![](https://ruder.io/content/images/2022/01/xslr.png "Self-supervised speech representation learning")<br/> -->
<div style="width:94%;text-align:left;background-color:cornsilk">

XLS-R による音声の教師あり異言語表現学習。
このモデルは自己教師付き wav2vec 2.0 スタイルの損失を用いて，多様な多言語音声データで事前に学習される。
学習したモデルは，様々な音声課題で微調整することができる [(Babu et al., 2021)](https://arxiv.org/abs/2111.09296)。
<!-- ![Self-supervised cross-lingual representation learning on speech using XLS-R. 
The model is pre-trained on diverse multilingual speech data using a self-supervised wav2vec 2.0-style loss. 
The trained model can then be fine-tuned on different speech tasks [(Babu et al., 2021)](https://arxiv.org/abs/2111.09296).](https://ruder.io/content/images/2022/01/xslr.png "Self-supervised speech representation learning") -->
</div>
</center>

**何が起こった?** 
2021 年大規模な事前学習済みモデルの開発が続けられた。
事前学習済みモデルは多くの異なるドメインで適用され ML 研究にとって重要であると考えられ始めた[1]。
コンピュータビジョン分野では Vision Transformer[2] のような教師付き事前学習モデルがスケールアップされ[3]，自己教師付き事前学習モデルがその性能に匹敵し始めた[4]。
後者は ImageNet の制御された環境を超えて，画像のランダムなコレクションにスケールアップされている[5]。
音声では W2v-BERT[7] などの wav2vec 2.0[6] に基づく新しいモデルや，XLS-R[8] などより強力な多言語モデルが構築されている。
同時にビデオと言語[9] や音声と言語[10] など，これまで研究が進んでいなかったモダリティ対に対する新しい統一的な事前学習済みモデルを見ることもできた。
視覚と言語では，制御された研究がこのようなマルチモーダルモデルの重要な構成要素に新しい光を当てている[11], [12]。
言語モデリングのパラダイムで異なる課題を構成することにより，モデルは強化学習[13] やタンパク質構造予測[14] など他のドメインでも大きな成功を収めた。
これらのモデルの多くでスケーリング動作が観測されていることから，異なるパラメータサイズでの性能を報告することが一般的になってきている。
しかし，事前学習による性能の向上は必ずしも下流の設定に反映されない[15], [16]。
<!-- **What happened?**   2021 saw the continuation of the development of ever larger pre-trained models. 
Pre-trained models were applied in many different domains and started to be considered critical for ML research [1](index.html#fn1). 
In computer vision, supervised pre-trained models such as Vision Transformer [2](index.html#fn2) have been scaled up [3](index.html#fn3) and self-supervised pre-trained models have started to match their performance [4](index.html#fn4). 
The latter have been scaled beyond the controlled environment of ImageNet to random collections of images [5](index.html#fn5). 
In speech, new models have been built based on wav2vec 2.0 [6](index.html#fn6) such as W2v-BERT [7](index.html#fn7) as well as more powerful multilingual models such as XLS-R [8](index.html#fn8). 
At the same time, we saw new unified pre-trained models for previously under-researched modality pairs such as for videos and language [9](index.html#fn9) as well as speech and language [10](index.html#fn10). 
In vision and language, controlled studies shed new light on important components of such multi-modal models [11](index.html#fn11)[12](index.html#fn12).
By framing different tasks in the paradigm of language modelling, models have had great success also in other domains such as reinforcement learning [13](index.html#fn13) and protein structure prediction [14](index.html#fn14). 
Given the observed scaling behaviour of many of these models, it has become common to report performance at different parameter sizes. However, increases in pre-training performance do not necessarily translate to downstream settings [15](index.html#fn15), [16](index.html#fn16). -->

**なぜそれが重要なのか？**
また，少数ショットとロバストな学習が可能である。
このように，事前学習済みモデルは研究の進歩のための貴重なビルディングブロックであり，新しい実用的な応用を可能する。
<!-- **Why is it important?**   Pre-trained models have been shown to generalize well to new tasks in a given domain or modality. 
They demonstrate strong few-shot learning behaviour and robust learning capabilities. 
As such, they are a valuable building block for research advances and enable new practical applications. -->

**次は何か?**
今後より多くの，より大規模な事前学習済みモデルが開発されることは間違いないでしょう。
同時に個々のモデルがより多くの課題を同時に実行できるようになることも期待されます。
言語分野では既にそうなっており，テキストからテキストへの共通の枠組みによって，モデルは多くの課題を実行することができます。
同様に画像や音声のモデルも 1 つのモデルで多くの共通課題を実行できるようになる可能性があります。
最後に複数のモダリティに対応するモデルを学習させる研究がさらに進むでしょう。
<!-- **What's next?**   We will undoubtedly see more and even larger pre-trained models developed in the future. 
At the same time, we should expect individual models to perform more tasks at the same time. 
This is already the case in language where models can perform many tasks by framing them in a common text-to-text format. 
Similarly, we will likely see image and speech models that can perform many common tasks in a single model. 
Finally, we will see more work that trains models for multiple modalities. -->

## 2. 大規模マルチタスク学習 
<!-- # 2) Massive Multi-task Learning  -->

<center>
<img src="https://ruder.io/content/images/2022/01/ext5.png" width="88%"><br/>
<div style="text-align:left; width:94%;background-color:cornsilk">

ExT5 による大規模なマルチタスク学習 [Massive multi-task learning (Aribandi et al.,2021)](https://arxiv.org/abs/2111.10952)。
事前学習ではテキスト-テキスト形式の多様な課題の入力 (左) に対してモデルを学習させ，対応する出力 (右) を生成する。
課題にはマスク化言語モデル，要約，意味解析，閉架式質問応答，スタイル変換，対話モデリング，自然言語推論，Winograd-schema 形式の共参照解決 (上から下)，などがある。
<!-- ![Massive multi-task learning with ExT5. 
During pre-training, the model is trained on the inputs (left) of a diverse set of different tasks in a text-to-text format to produce the corresponding outputs (right). 
The tasks include masked language modeling, summarization, semantic parsing, closed-book question answering, style transfer, dialogue modeling, natural language inference, Winograd-schema style coreference resolution (top to bottom), among others [(Aribandi et al.,2021)](https://arxiv.org/abs/2111.10952).](https://ruder.io/content/images/2022/01/ext5.png "Massive multi-task learning") -->
</div>
</center>
<!-- ![[(Aribandi et al.,2021)](https://arxiv.org/abs/2111.10952).](https://ruder.io/content/images/2022/01/ext5.png "Massive multi-task learning")<br/> -->

**何が起こったのか？**  
前節で紹介したほとんどの事前学習済みモデルは自己教師付きだ。
これらは一般に、明示的な監視を必要としない目的によって大量のラベルなしデータから学習する。
しかし，多くのドメインでは既に大量のラベル付きデータが存在し，それを利用することでより良い表現を学習することができる。
これまで，T0 [17],  FLAN [18],  ExT5 [19] などのマルチタスクモデルは，主に言語に関する100 程度の課題について事前学習が行われてきた。
このような大規模なマルチタスク学習は，メタ学習と密接な関係がある。
多様な課題分布[20]にアクセスすることで，モデルは文脈内学習[21] の方法など，異なるタイプの振る舞いを学習するように **学習**することができまる。
<!-- **What happened?**  
Most pre-trained models in the previous section are self-supervised. 
They generally learn from large amounts of unlabelled data via an objective that does not require explicit supervision. 
However, for many domains large amounts of labelled data are already available, which can be used to learn better representations. 
So far, multi-task models such as T0 [17], FLAN [18], and ExT5 [19] have been pre-trained on around 100 tasks mainly for language. 
Such massive multi-task learning is closely related to meta-learning. 
Given access to a diverse task distribution [20], models can *learn* to learn different types of behaviour such as how to do in-context learning [21]. -->

**なぜそれが重要なのか？**
T5 や GPT-3 などの最近のモデルは text-to-text 形式を採用しているため，大規模なマルチタスク学習が可能である。
このため，複数課題にまたがって効率的に学習する。そのため，課題固有の損失関数や課題固有の層を手作業で設計する必要がなくなる。
このように，最近のアプローチは，自己教師付き事前学習と教師付きマルチタスク学習を組み合わせることの利点を強調し，両者の組み合わせがより一般的なモデルにつながることを実証している。
<!-- **Why is it important?** 
Massive multi-task learning is possible due to the fact that many recent models such as T5 and GPT-3 use a text-to-text format. 
Models thus no longer require hand-engineered task-specific loss functions or task-specific layers in order to effectively learn across multiple tasks. 
Such recent approaches highlight the benefit of combining self-supervised pre-training with supervised multi-task learning and demonstrate that a combination of both leads to models that are more general. -->

**次に何か？**
データセットが統一された形式で入手可能でオープンソースであることを考えると，新しく作成された高品質のデータセットを使って，ますます多様化する課題コレクションでより強力なモデルを訓練し，それをインザループで使ってより難しいデータセットを作成するという好循環が想像できる。
<!-- **What's next?** 
Given the availability and open-source nature of datasets in a unified format, we can imagine a virtuous cycle where newly created high-quality datasets are used to train more powerful models on increasingly diverse task collections, which could then be used in-the-loop to create more challenging datasets. -->

## 3. トランスフォーマーを越えて
<!-- ## 3. Beyond the Transformer  -->

<center>
<img src="https://ruder.io/content/images/2022/01/perceiver.png" width="94%"><br/>
<div style="text-align:left; width:94%;background-color:cornsilk">

Perceiver (Jaegle et al.,2021) は高次元の入力バイト配列を交差注意によって固定次元の潜在配列に投影し，それを変換自己注意ブロックによって処理する。
その後，交差注意と自己注意のブロックが交互に適用される
[(Jaegle et al.,2021)](https://arxiv.org/abs/2103.03206).
<!-- ![The Perceiver projects a high-dimensional input byte array via cross-attention to a fixed-dimensional latent array, which it processes with a transformer self-attention block. 
Cross-attention and self-attention blocks are then applied alternatingly [(Jaegle et al.,2021)](https://arxiv.org/abs/2103.03206).](https://ruder.io/content/images/2022/01/perceiver.png "Perceiver") -->
</div>
</center>

**何が起こったのか？** <!-- **What happened?** -->
前のセクションで説明したほとんどの事前学習済みモデルは transformer アーキテクチャ[22](index.html#fn22)をベースに構築されている。
2021 年には transformer に代わるモデルアーキテクチャが開発された。
Perceiver [23](index.html#fn23) はトランスフォーマーに似たアーキテクチャで，基本表現として固定次元の潜在的配列を用い，これを交差注意によって入力に条件付けることにより，非常に高次元の入力に拡張するものである。
Perceiver IO [24](index.html#fn24) はこのアーキテクチャを拡張して，構造化された出力空間も扱えるようにしたものである。
他のモデルは MLP-Mixer [25](index.html#fn25) や gMLP [26](index.html#fn26) のような多層パーセプトロン (MLP) を用いて，どこにでもある自己注意層の置き換えを試みている。
また FNet[27](index.html#fn27) では，自己注意の代わりに 1 次元フーリエ変換を用いて，トークン単位で情報を混合している。
一般にアーキテクチャを事前学習戦略から切り離したものとして考えることは有用である。
もし CNN が変換モデルと同じように事前学習されれば，多くの NLP 課題で競争力のある性能を達成することができる[28](index.html#fn28)。
同様に ELECTRA スタイルの事前学習 [29](index.html#fn29) のような代替の事前学習目的を用いることで，利益が得られるかもしれない [30](index.html#fn30)。
<!-- Most pre-trained models discussed in the previous sections build on the transformer architecture [22](index.html#fn22). 
2021 saw the development of alternative model architectures that are viable alternatives to the transformer. 
The Perceiver [23](index.html#fn23) is a transformer-like architecture that scales to very high-dimensional inputs by using a latent array of a fixed dimensionality as its base representation and conditioning this on the input via cross-attention. 
Perceiver IO [24](index.html#fn24) extended the architecture to also deal with structured output spaces. Other models have tried to replace the ubiquituous self-attention layer, most notably using multilayer perceptrons (MLPs) such as in the MLP-Mixer [25](index.html#fn25) and gMLP [26](index.html#fn26). 
Alternatively, FNet[27](index.html#fn27) uses 1D Fourier Transforms instead of self-attention to mix information at the token level. 
In general, it is useful to think of an architecture as decoupled from the pre-training strategy. 
If CNNs are pre-trained the same way as transformer models, they achieve competitive performance on many NLP tasks [28](index.html#fn28). 
Similarly, using alternative pre-training objectives such as ELECTRA-style pre-training [29](index.html#fn29) may lead to gains [30](index.html#fn30). -->

**なぜそれが重要なのか？** <!-- **Why is it important?** -->
研究は，多くの補完的または直交する方向性を同時に探求することで進展する。
もし，ほとんどの研究が単一のアーキテクチャに焦点を当てている場合，これは必然的に偏り，盲点，機会を逃すことにつながる。
新しいモデルは，注意の計算複雑性，ブラックボックス的性質，秩序-無宗教性など，トランスフォーマーの限界のいくつかに対処できるかもしれない。
例えば一般化加法モデルの神経拡張は，現在のモデルと比較してはるかに優れた解釈性を提供する[31](index.html#fn31)。
<!-- Research progresses by exploring many complementary or orthogonal directions at the same time. 
If most research focuses on a single architecture, this will inevitably lead to bias, blind spots, and missed opportunities. 
New models may address some of the transformers\' limitations such as the computational complexity of attention, its black-box nature, and order-agnosticity. 
For instance, neural extensions of generalized additive models offer much better interpretability compared to current models [31](index.html#fn31). -->

**次に来るものは？** <!-- **What's next?** -->
しかし，長距離依存関係や高次元入力のモデル化，あるいは解釈や説明のしやすさが要求される場合など，現在のモデルでは不十分な場面では，特に代替アーキテクチャの登場が期待される。
<!-- While pre-trained transformers will likely continue to be deployed as standard baselines for many tasks, we should expect to see alternative architectures particularly in settings where current models fail short, such as modeling long-range dependencies and high-dimensional inputs or where interpretability and explainability are required. -->

## 4. プロンプト
<!-- # 4) Prompting  -->

<center>
<img src="https://ruder.io/content/images/2022/01/p3.png" width="94%"><br/>
<!-- ![[(Sanh et al.,2021)](https://arxiv.org/abs/2110.08207).](https://ruder.io/content/images/2022/01/p3.png "Prompting")<br/> -->
<div style="text-align:left; width:94%;background-color:cornsilk">

P3 プロンプト集に収録されているプロンプトテンプレート[Sanh et al.,2021](https://arxiv.org/abs/2110.08207)。
各課題には複数のテンプレートが用意されている。
各テンプレートは入力パターンとターゲットの言語化で構成される。
言い換えの場合 **選択肢** は 1 つ目と 2 つ目のテンプレートでそれぞれ {Not duplicates, Duplicates} と {Yes, No} からなる。
<!-- ![Prompt templates from the P3 prompt collection. Multiple templates are possible for each task. 
Each template consists of an input pattern and a target verbalizer. For paraphrasing, *Choices* consist of {Not duplicates, Duplicates} and {Yes, No} in the first and second template respectively [(Sanh et al.,2021)](https://arxiv.org/abs/2110.08207).](https://ruder.io/content/images/2022/01/p3.png "Prompting") -->
</div>
</center>

**何が起こったか?** GPT-3 [32](index.html#fn32) によって普及したプロンプトは NLP モデルのための実行可能な代替入力形式として出現した。
プロンプトは通常，モデルに特定の予測をするように要求する **パターン** と，予測をクラスラベルに変換する **バーバライザー** を含んでいる。
PET [33](index.html#fn33) iPET [34](index.html#fn34) や AdaPET [35](index.html#fn35) などのいくつかのアプローチはプロンプトを利用して少数ショット学習をしている。
しかし，プロンプトは 銀の弾丸ではない。
プロンプトは決して特効薬ではなく，プロンプトの種類によってモデルの性能は大きく異なり，最適なプロンプトを見つけるにはラベル付けされた例が必要である[36]。
このため，少数ショット設定でモデルを比較するための新しい評価方法が開発されている[37]。
多くのプロンプトが [public pool of prompts (P3)](https://github.com/bigscience-workshop/promptsource) の一部として利用可能であり，プロンプトの最適な利用方法を探ることが可能です。
このサーベイ[38] は，一般的な研究分野の優れた概観を提供する。
<!-- **What happened?**   Popularized by GPT-3 [32](index.html#fn32), prompting has emerged as a viable alternative input format for NLP models. 
Prompts typically include a *pattern* that asks the model to make a certain prediction and a *verbalizer* that converts the prediction to a class label. 
Several approaches such as PET, [33](index.html#fn33) iPET [34](index.html#fn34), and AdaPET [35](index.html#fn35) leverage prompts for few-shot learning. 
Prompts are not a silver bullet, however. Models\' performance varies drastically depending on the prompt and finding the best prompt still requires labeled examples [36](index.html#fn36). 
In order to compare models reliably in a few-shot setting, new evaluation procedures have been developed [37](index.html#fn37). 
A large number of prompts are available as part of the [public pool of prompts (P3)](https://github.com/bigscience-workshop/promptsource), enabling exploration of the best way to use prompts. 
This survey [38](index.html#fn38) provides an excellent overview of the general research area. -->

**なぜそれが重要なのか？**
プロンプトは課題固有の情報を符号化するために用いられ，課題によっては最大 3,500 のラベル付けされた例に相当することができる[39]。
このようにプロンプトは，手作業によるラベル付けやラベル付け関数の定義を超えて，専門家の情報をモデル学習に取り込む新しい方法を可能にする[40]。
<!-- **Why is it important?** 
A prompt can be used to encode task-specific information, which can be worth up to 3,500 labeled examples, depending on the task [39]. 
Prompts thus an enable a new way to incorporate expert information into model training, beyond manually labeling examples or defining labeling functions [40]. -->

**次は何か？**
我々はプロンプトを利用してモデル学習を向上させるための表面を削ったに過ぎない。
また，プロンプトはより精巧になり，例えば，より長い指示 [18:1] や正例・負例 [41]，一般的なヒューリスティックを含むようになるだろう。
また，プロンプトは自然言語による説明[42] をモデル学習に取り入れる，より自然な方法となるかもしれない。
<!-- **What's next?** 
We have only scratched the surface of using prompts to improve model learning. 
Prompts will become more elaborate, for instance including longer instructions [18:1] as well as positive and negative examples [41] and general heuristics. 
Prompts may also be a more natural way to incorporate natural language explanations [42] into model training. -->

## 5. 効率的方法 
<!-- # 5) Efficient Methods  -->

<center>
<img src="https://ruder.io/content/images/2022/01/magma.png" width="88%"><br/>
<div style="text-align:left;width:94%;background-color:cornsilk">

MAGMA によるマルチモーダル適応[(Eichenberg et al.,2021)](https://arxiv.org/abs/2112.05253)。
凍結した事前学習済み言語モデルを画像エンコーダを介して学習した視覚プレフィックスと視覚に特化したアダプタ層を用いてマルチモーダルな課題に適応させる。
<!-- ![Multi-modal adaptation with MAGMA. A frozen pre-trained language model is adapted to multi-modal tasks using a visual prefix learned via an image encoder as well as vision-specific adpater layers [(Eichenberg et al.,2021)](https://arxiv.org/abs/2112.05253).](https://ruder.io/content/images/2022/01/magma.png "MAGMA") -->
</div>
</center>

**何が起こったのか？**
一般的に，事前に学習させたモデルは非常に大きく，実際に使用するには非効率的であることが多いという欠点がある。
2021 年には，より効率的なアーキテクチャと，より効率的な微調整方法の両方が進歩しました。
モデリング面では，より効率的な自己注意のバージョン [43], [44] がある。
この総説論文[45] では 2021 年以前のモデルの概要が紹介されている。
現在の事前学習済みモデルは非常に強力で，少数のパラメータを更新するだけで効果的に条件付けができるため，連続プロンプトに基づくより効率的な微調整アプローチ [46,47] やアダプター [48,49,50] 等が発展してきている。
これら能力により，適切な接頭辞[51] や適切な変換[52, 53] を学習して新しい様式への適応も可能である。
また，より効率的な最適化器を作るための量子化[54] やスパース性といった手法も用いられてきた。
<!-- **What happened?**   A downside of pre-trained models is that they are generally very large and often inefficient to use in practice. 
2021 brought advances both in more efficient architectures as well as in more efficient fine-tuning methods. 
On the modeling side, we saw several more efficient versions of self-attention [43](index.html#fn43), [44](index.html#fn44).
This survey [45](index.html#fn45) provides an overview of pre-2021 models. 
Current pre-trained models are so powerful that they can be effectively conditioned by only updating few parameters, which has led to the development of more efficient fine-tuning approaches based on continuous prompts [46](index.html#fn46),[47](index.html#fn47) and adapters [48](index.html#fn48),[49](index.html#fn49), [50](index.html#fn50), among others. 
This capability also enables adaptation to new modalities by learning an appropriate prefix [51](index.html#fn51) or suitable transformations [52](index.html#fn52),[53](index.html#fn53). 
Other methods such as quantization for creating more efficient optimizers [54](index.html#fn54) as well as sparsity have also been used. -->

**なぜそれが重要なのか?** 
標準的なハードウェアで実行することが不可能であったり，法外に高価である場合，モデルは有用ではない。
効率性の向上は，モデルが大きくなる一方で，実務家にとって有益で利用しやすいものになることを保証する。
<!-- **Why is it important?**   Models are not useful if they are infeasible or prohibitively expensive to run on standard hardware. 
Advances in efficiency will ensure that while models are growing larger, they will be benefical and accessible to practicioners. -->

**次は何か?** 
効率的なモデルや訓練方法は，より使いやすく，よりアクセスしやすくなるはずである。
同時に，コミュニティは，大規模なモデルとのインターフェースや，ゼロから新しいモデルを事前訓練することなく，それらを効率的に適応，結合，修正するための，より効果的な方法を開発することになるだろう。
<!-- **What's next?**   Efficient models and training methods should become easier to use and more accessible. 
At the same time, the community will develop more effective ways to interface with large models and to efficiently adapt, combine or modify them without having to pre-train a new model from scratch. -->

## 6. ベンチマーク
<!-- ## 6. Benchmarking  -->

<center>
<img src="https://ruder.io/content/images/2022/01/dataset_concentration.png" width="94%"><br/>
<div style="text-align:left;width:94%;background-color:cornsilk">

一般的な ML ベンチマークの飽和状態[(Koch et al.,2021)](https://openreview.net/forum?id=zNQBIBKJRkd)。
機関やデータセットへのデータセット利用集中が時間の経過とともに増加。 
機関ごとのデータセット利用状況地図(左)。
50% 以上のデータセット利用が 12 の機関に起因している。
ジニ係数で測定した機関および特定のデータセットへのデータセット利用集中は，近年増加している (右)。
<!-- Benchmark saturation of popular ML benchmarks
Increases in concentration of dataset usage on institutions and datasets over time. 
Map of dataset usages per institution (left). 
Over 50% of dataset usages can be attributed to 12 institutions. 
The concentration of dataset usage on institutions and specific datasets as measured by the Gini coefficient has increased in recent years (right) -->
</div>
</center>
<!-- 
![Increases in concentration of dataset usage on institutions and datasets over time. 
Map of dataset usages per institution (left). 
Over 50% of dataset usages can be attributed to 12 institutions. 
The concentration of dataset usage on institutions and specific datasets as measured by the Gini coefficient has increased in recent years (right) [(Koch et al.,2021)](https://openreview.net/forum?id=zNQBIBKJRkd).](https://ruder.io/content/images/2022/01/dataset_concentration.png "Benchmark saturation of popular ML benchmarks") -->

**何が起きたのか？** 
最近の ML や NLP モデルの性能は急速に向上し，多くのベンチマークがそれらを測定する能力を上回った。
同時にコミュニティは，少数のエリート機関から発信されるベンチマークで評価することは少なくなっている[55]。
その結果 2021 年には，ベストプラクティスと，今後このようなモデルを確実に評価する方法について多くの議論がなされた。
この議論は [このブログの記事](https://ruder.io/nlp-benchmarking/) で取り上げている。
2021 年に NLP コミュニティで登場した注目すべきリーダーボードパラダイムは，動的敵対的評価[56]，コミュニティメンバーが共同で評価データセットを作成するコミュニティ駆動型評価 [BIG-bench](https://github.com/google/BIG-bench)，異なるエラータイプにわたる対話的での詳細評価[57]，単一性能尺度でのモデル評価を超えた多次元評価 [58] である。
さらに少数点評価[59,60] やクロスドメイン汎化[61] などの有力な設定に対応した新しいベンチマークも提案された。
また，音声のような特定のモダリティ[62] や，インドネシア語やルーマニア語のような特定の言語に対する，汎用の事前学習済みモデルの評価に焦点を当てた新しいベンチマークも見受けられる[63,64]。
また，モダリティ間[65]，多言語環境[66] での評価に焦点を当てた新しいベンチマークも見られる。
加えて，評価指標にももっと注意を払うべきであろう。
機械翻訳 (MT) のメタ評価[67] では，過去 10 年間の 769 の MT 論文のうち，108 の代替指標 (多くの場合より良い人間翻訳との相関を持つ) が提案されているにもかかわらず 74.3% が BLEU しか使用していないことが明らかになった。
GEM[68] や 双次元リーダーボード (bidimensional leaderboards)[69] などの最近の取り組みでは，モデルと手法を合同で評価することが提案されている。
<!-- **What happened?**  
The rapidly improving capabilities of recent ML and NLP models have outpaced the ability of many benchmarks to measure them. 
At the same time, communities evaluate on fewer and fewer benchmarks, which originate from a small number of elite institutions[55]. 
Consequently, 2021 saw much discussion of best practices and ways in which we can reliably evaluate such models going forward, which I cover in [this blog post](https://ruder.io/nlp-benchmarking/). 
Notable leaderboard paradigms that emerged in 2021 in the NLP community are dynamic adversarial evaluation[56], community-driven evaluation where community members collaborate on creating evaluation datasets such as [BIG-bench](https://github.com/google/BIG-bench), interactive fine-grained evaluation across different error types [57], and multi-dimensional evaluation that goes beyond evaluating models on a single performance metric [58]. 
In addition, new benchmarks were proposed for influential settings such as few-shot evaluation [59,60] and cross-domain generalization [61].
We also saw new benchmarks focused on evaluating general-purpose pre-trained models, for specific modalities such as speech [62] and specific languages, for instance, Indonesian and Romanian 
[63,64], as well as across modalities [65] and in a multilingual setting [66]. 
We also should pay more attention to evaluation metrics. 
A machine translation (MT) meta-evaluation [67] revealed that among 769 MT papers of the last decade, 74.3% only used BLEU, despite 108 alternative metrics---often with better human correlation---having been proposed. 
Recent efforts such as GEM [68] and bidimensional leaderboards[69] thus propose to evaluate models and methods jointly. -->

**なぜそれが重要なのか？
ベンチマークと評価は，機械学習と自然言語処理における科学的進歩の要である。
正確で信頼できるベンチマークがなければ，我々は本当に進歩しているのか，それとも凝り固まったデータセットや測定基準に過剰に適合しているのかを判断することはできない。
<!-- **Why is it important?** 
Benchmarking and evaluation are the linchpins of scientific progress in machine learning and NLP. 
Without accurate and reliable benchmarks, it is not possible to tell whether we are making genuine progress or overfitting to entrenched datasets and metrics.-->

**次は何か？**
ベンチマークの問題に対する認識が高まれば，新しいデータセットをより慎重に設計することにつながるはずである。
また，新しいモデルの評価は，単一の性能指標に焦点を当てるのではなく，モデルの公平性，効率性，堅牢性などの複数の側面を考慮に入れる必要がある。
<!-- **What's next?** 
Increased awareness around issues with benchmarking should lead to a more thoughful design of new datasets. 
Evaluation of new models should also focus less on a single performance metric but take multiple dimensions into account, such as a model\'s fairness, efficiency, and robustness. -->

## 7. 条件付き画像生成 
<!-- # 7) Conditional image generation  -->

<center>
<img src="https://ruder.io/content/images/2022/01/clip_generation.gif" width="94%"><br/>
<!-- <a href="https://ml.berkeley.edu/blog/posts/clip-art"><img src="https://ruder.io/content/images/2022/01/clip_generation.gif">Charlie Snell</a><br/> -->
<div style="text-align:left;width:94%;background-color:cornsilk">

生成モデルを用いて潜在ベクトルから画像を生成。
生成された画像とテキスト説明文の CLIP による埋め込みの類似度に基づいて，潜在ベクトルを更新する。
この処理を収束するまで繰り返す。
(画像出典: [Charlie Snell](https://ml.berkeley.edu/blog/posts/clip-art/)).
<!-- A generative model generates an image based on a latent vector. 
The latent vector is then updated based on the similarity of CLIP's embeddings of the generated image and the text description. 
This process is repeated until convergence  -->
</div>
</center>

<!-- ![How CLIP Generates Art. 
A generative model generates an image based on a latent vector. 
The latent vector is then updated based on the similarity of CLIP's embeddings of the generated image and the text description. 
This process is repeated until convergence 
(Credit: [Charlie Snell](https://ml.berkeley.edu/blog/posts/clip-art/)).](https://ruder.io/content/images/2022/01/clip_generation.gif "CLIP art generation") -->

**何が起きたのか？** <!-- **What happened?**  -->
条件付き画像生成，つまり，テキストの記述に基づいて画像を生成することは，2021 年に印象的な結果を見た。
最新世代の生成モデルの周辺にはアートシーンが出現した (概要については [このブログ記事 clip-art](https://ml.berkeley.edu/blog/posts/clip-art/) 参照)。
DALL-E モデル [70](index.html#fn70) のようにテキスト入力に基づいて直接画像を生成するのではなく，最近のアプローチは CLIP [72](index.html#fn72) のような画像とテキストの結合埋込モデルを用いて VQ-GAN [71](index.html#fn71) など強力な生成モデルの出力を操縦している。
信号から徐々にノイズを除去する尤度ベースの拡散モデルは GAN を凌駕する強力な新しい生成モデルとして登場した [73 Diffusion Beat GAN]。
またテキスト入力に基づき出力を誘導することで，最近のモデルは実際の写真のようなな画像品質に近づきつつある [74 GLIDE]。
また，このようなモデルは特にインペインティングを得意とし，記述に基づいて画像の領域を修正することができる。
<!-- Conditional image generation, i.e., generating images based on a text description, saw impressive results in 2021. 
An art scene emerged around the most recent generation of generative models (see [this blog post](https://ml.berkeley.edu/blog/posts/clip-art/) for an overview). 
Rather than generating an image directly based on a text input as in the DALL-E model [70](index.html#fn70), recent approaches steer the output of a powerful generative model such as VQ-GAN [71](index.html#fn71) using a joint image-and-text embedding model such as CLIP [72](index.html#fn72). 
Likelihood-based diffusion models, which gradually remove noise from a signal have emerged as powerful new generative models that can outperform GANs [73](index.html#fn73). 
By guiding their outputs based on text inputs, recent models are approaching photorealistic image quality [74](index.html#fn74). 
Such models are also particularly good at inpainting and can modify regions of an image based on a description.-->

**なぜそれが重要なのか？** <!-- **Why is it important?**  -->
ユーザーがガイドできる高品質な画像の自動生成は，視覚資産の自動設計，モデル支援型プロトタイピングやデザイン，パーソナライゼーションなど，芸術的・商業的応用の幅を広げる。
<!-- Automatic generation of high quality images that can be guided by users opens a wide range of artistic and commercial applications, from the automatic design of visual assets, model-assisted prototyping and design, personalization, etc.  -->

**次はどうなる？** <!-- **What's next?**  -->
最近の拡散モデルのサンプリングは GAN 型モデルに比べて非常に遅い。
このようなモデルを実世界で活用するためには効率の改善が必要である。
また，この分野では人間とコンピュータの相互作用に関する研究をさらに進め，このようなモデルが人間を支援するための最適な方法や応用を特定する必要がある。
<!-- Sampling from recent diffusion-based models is much slower compared to their GAN-based counterparts. 
These models require improvements in efficiency to make them useful for real-world applications. 
This area also requires more research in human-computer interaction, to identify the best ways and applications where such models can assist humans. -->

## 8. 機械学習の科学応用
<!-- ## 8. ML for Science  -->

<center>
<img src="https://ruder.io/content/images/2022/01/alphafold_2.0.png" width="94%"><br/>
<div style="text-align:left;width:94%;background-color:cornsilk">

AlphaFold 2.0 のアーキテクチャ: アミノ酸残基対だけでなく，進化的に関連したタンパク質配列に着目し，両表現間で反復的に情報を受け渡す。
(画像出典: [DeepMind](https://deepmind.com/blog/article/alphafold-a-solution-to-a-50-year-old-grand-challenge-in-biology))。
<!-- The architecture of AlphaFold 2.0. 
The model attends over evolutionarily related protein sequences as well as amino acid residue pairs and iteratively passes information between both representations -->
</div>
</center>

<!-- ![The architecture of AlphaFold 2.0. The model attends over evolutionarily related protein sequences as well as amino acid residue pairs and iteratively passes information between both representations (Credit: [DeepMind](https://deepmind.com/blog/article/alphafold-a-solution-to-a-50-year-old-grand-challenge-in-biology)).](https://ruder.io/content/images/2022/01/alphafold_2.0.png "AlphaFold 2.0") -->

**何が起こったのか？** 
2021 年には自然科学の発展のために ML を応用したいくつかのブレークスルーがあった。
気象学では降水ナウキャスティングと予測[75,76] の進歩により，予測精度が大幅に改善された。
いずれの場合も，モデルは最先端の物理ベースの予測モデルを凌駕した。 
生物学では AlphaFold 2.0 が類似の構造が知られていない場合でさえ，前例のない精度でタンパク質の構造を予測することに成功した[14:1]。
数学では ML は新しい接続やアルゴリズムを発見するために数学者の直観を導くことができることが示された[77]。
またトランスフォーマーモデルは，十分な量のデータで学習した場合，局所安定性のような微分システムの数学的特性を学習できることが示されている[78]。
<!-- **What happened?**  
2021 saw several breakthroughs in ML applied to advance the natural sciences. 
In meteorology, advances in precipitation nowcasting and forecasting [75,76] led to substantial improvements in forecast accuracy. 
In both cases, models outperformed state-of-the-art physics-based forecast models.  
In biology, AlphaFold 2.0 managed to predict the structure of proteins with unprecedented accuracy, even in cases where no similar structure is known [14:1]. 
In mathematics, ML was shown to be able to guide the intuition of mathematicians in order to discover new connections and algorithms[77]. 
Transformer models have also been shown to be capable of learning mathematical properties of differential systems such as local stability when trained on sufficient amounts of data[78].-->

**なぜ重要なのか？** 
ML を自然科学分野での理解や応用を進めるために利用することは，最もインパクトのある応用の一つである。
強力な ML 手法を用いることで，新しい応用を可能にし，また，創薬デザインのような既存の応用を大幅にスピードアップさせることができる。
<!-- **Why is it important?**  
Using ML for advancing our understanding and applications in natural sciences is one of its most impactful applications. 
Using powerful ML methods enables both new applications and can greatly speed up existing ones such as drug design. -->

**次に何か？** 
新しい進歩の発見と開発において研究者を支援するためにモデルを in-the-loop で使用することは，特に説得力のある方向性である。
そのためには，強力なモデルの開発だけでなく，対話的な機械学習や人間とコンピュータの相互作用に関する研究が必要である。
<!-- **What's next?**  
Using models in-the-loop to assist researchers in the discovery and development of new advances is a particularly compelling direction. 
It requires both the development of powerful models as well as work on interactive machine learning and human-computer interaction. -->

## 9. プログラム合成
<!-- ## 9. Program synthesis -->

<center>
<img src="https://ruder.io/content/images/2022/01/dobf.png" width="94%"><br/>
<div style="text-align:left;width:94%;background-color:cornsilk">

コードのモデリングを行うためのマスク化言語モデル (MLM) と難読化解除 (DOBF) の事前学習目的関数の比較 ([Roziere et al., (2021)](https://openreview.net/forum?id=3ez9BSHTNT)<!-- Comparison of masked language modeling (ML) and deobfuscation (DOBF) pre-training objectives for modeling code. -->
MLM は主にプログラミング言語の文法に関連するランダムにマスクされたトークンを予測する。
DOBF は関数名や変数名を難読化する必要があり，より困難である。<!-- MLM predicts randomly masked tokens, which mainly relate to a programming language\'s syntax. 
DOBF requires deobfuscating the names of functions and variables, which is much more challenging--> ([Roziere et al., 2021](https://openreview.net/forum?id=3ez9BSHTNT)).
</div>
</center>

**何が起こったのか？**
2021 年，大規模言語モデルの最も注目すべきアプリケーションの一つはコード生成であり，Codex [79] が[GitHub Copilot](https://copilot.github.com/) の一部として初めて主要製品に統合された。
その他，事前学習モデルの進歩は，より優れた事前学習目的関数 [80,81] からスケーリング実験 [82,82,83] まで多岐に渡った。
しかし，複雑で長大なプログラムを生成することは，現在のモデルにとってまだ課題である。
また，プログラムの実行やモデル化の学習は，多段階の計算を行い，その途中の計算をスクラッチパッドに記録することで改善することができる。
<!-- **What happened?** 
One of the most notable applications of large language models this year was code generation, which saw with Codex [79] its first integration into a major product as part of [GitHub Copilot](https://copilot.github.com/). 
Other advances in pre-training models ranged from better pre-training objectives [80,81] to scaling experiments [82,82,83]. 
Generating complex and long-form programs is still a challenge for current models, however. 
An interesting related direction is learning to execute or model programs, which can be improved by performing multi-step computation where intermediate computation steps are recorded in a \"scratchpad\" [84].-->

**なぜ重要なのか？**
複雑なプログラムを自動的に合成できることはソフトウェア技術者の支援など，様々な用途に役立つ。
<!-- **Why is it important?** 
Being able to automatically synthesize complex programs is useful for a wide variety of applications such as supporting software engineers. -->

**次は何か？**
コード生成モデルが実際にソフトウェア技術者のワークフローをどの程度改善するかは，まだ未解決の問題である[85]。
真に役立つためには，このようなモデル---対話モデルと同様—は，新しい情報に基づいて予測を更新できる必要があり，局所的および大域的な文脈を考慮する必要がある。
<!-- **What's next?** 
It is still an open question how much code generation models improve the workflow of software engineers in practice [85]. 
In order to be truly helpful, such models---similarly to dialogue models---need to be able to update their predictions based on new information and need to take the local and global context into account. -->

## 10. バイアス
<!-- ## 10. Bias  -->

<center>
<img src="https://ruder.io/content/images/2022/01/toxicity_reduction.png" width="99%"><br/>
<div style="text-align:left;width:94%;background-color:cornsilk">

自動的な毒性緩和の意図しない副作用 (Side-effects of toxicity mitigation)
疎外されたグループに関するテキストを過剰にフィルタリングすると，言語モデルが当該グループに関する (ポジティブな) テキストを生成する能力が低下する ([Welbl et al.,2021](https://aclanthology.org/2021.findings-emnlp.210/))
<!-- Unintended side-effects of automatic toxicity mitigation.
Over-filtering of text about marginalized groups reduces the ability of language models to produce (even positive) text about said groups ([Welbl et al.,2021](https://aclanthology.org/2021.findings-emnlp.210/)) "Side-effects of toxicity mitigation" -->
<!-- ![Unintended side-effects of automatic toxicity mitigation.
Over-filtering of text about marginalized groups reduces the ability of language models to produce (even positive) text about said groups ([Welbl et al.,2021](https://aclanthology.org/2021.findings-emnlp.210/)).](https://ruder.io/content/images/2022/01/toxicity_reduction.png "Side-effects of toxicity mitigation") -->
</div></center>

**何が起こったのか？**<!-- **What happened?** -->
事前に学習された大規模なモデルの潜在的な影響を考えると，有害なバイアスを含まず，有害なコンテンツを生成するために誤用されず，持続可能な方法で使用されることが極めて重要である。
いくつかのレビュー [1:1](index.html#fn1), [86](index.html#fn86), [87](index.html#fn87) は，このようなモデルの潜在的なリスクに焦点を当てている。
性別，特定の民族，政治的傾向などの保護された属性に関するバイアスが調査されている [88](index.html#fn88), [89](index.html#fn89)。
しかし，毒性などのモデルからバイアスを取り除くことは，トレードオフを伴い，社会から疎外されたグループに関するテキストや著者のテキストに対するカバレッジを減少させることにつながる可能性がある [90](index.html#fn90)。
<!-- Given the potential impact of large pre-trained models, it is crucial that they do not contain harmful biases, are not misused to generate harmful content, and are used in a sustainable manner. 
Several reviews [1:1](index.html#fn1),[86](index.html#fn86),[87](index.html#fn87) highlight the potential risks of such models. 
Bias has been investigated with regard to protected attributes such as gender, particular ethnic groups, and political leaning [88](index.html#fn88),[89](index.html#fn89).
Removing bias from models such as toxicity, however, comes with trade-offs and can lead to reduced coverage for texts about and authored by marginalized groups [90](index.html#fn90). -->

**なぜそれが重要なのか？** <!-- **Why is it important?**   -->
実世界のアプリケーションでモデルを使用するためには，有害なバイアスを示さず，どのグループに対しても差別的であってはならない。
そのため，現在のモデルの偏りを理解し，それを取り除く方法を開発することは，安全で責任ある ML モデルの展開を可能にするために重要である。
<!-- In order to use models in real-world applications, they should not exhibit any harmful bias and not discriminate against any group. 
Developing a better understanding of the biases of current models and how to remove them is thus crucial for enabling safe and responsible deployment of ML models.  -->

**次はどうなるのか？** <!-- **What's next?** -->
これまでのところ，バイアスは英語，事前学習済みモデル，特定のテキスト生成や分類の応用で主に調査されてきた。
このようなモデルのライフサイクルを考えると，多言語環境，異なるモダリティの組み合わせ，事前学習済みモデルの異なる使用段階 (事前学習後，微調整後，テスト時) において，バイアスを特定し緩和することも目指すべきであろう。
<!-- Bias has so far been mostly explored in English and in pre-trained models and for specific text generation or classification applications. 
Given the intended use and lifecycle of such models, we should also aim to identify and mitigate bias in a multilingual setting, with regard to the combination of different modalities, and at different stages of a pre-trained model's usage---after pre-training, after fine-tuning, and at test time. -->

## 11. 検索拡張
<!-- # 11) Retrieval Augmentation  -->

<center>
<img src="https://ruder.io/content/images/2022/01/retro.png" width="88%"><br/>
<div style="text-align:left;width:88%;background-color:cornsilk">

RETRO のアーキテクチャの概要。
入力系列は複数のチャンクに分割される (左)。
各入力チャンクに対して BERT の埋め込み類似度に基づく近似最近傍探索を用いて最近傍チャンクが検索される。
標準的な変換層に挟まれたチャンク型交差注意 (右) により近傍チャンクに注意される。([Borgeaud et al.,2021](https://arxiv.org/abs/2112.04426)).
<!-- Overview of the RETRO architecture. An input sequence is split into multiple chunks (left). 
For each input chunk, nearest neighbor chunks are retrieved using approximate nearest neighbor search based on BERT embedding similarity. 
The model attends to the nearest neighbors using chunked cross-attention (right) interleaved with standard transformer layers -->
</div>
</center>

<!-- ![Overview of the RETRO architecture. An input sequence is split into multiple chunks (left). 
For each input chunk, nearest neighbor chunks are retrieved using approximate nearest neighbor search based on BERT embedding similarity. 
The model attends to the nearest neighbors using chunked cross-attention (right) interleaved with standard transformer layers ([Borgeaud et al.,2021](https://arxiv.org/abs/2112.04426)).](https://ruder.io/content/images/2022/01/retro.png "RETRO") -->

**何が起こったのか？** <!-- **What happened?**  -->
検索を事前学習と下流での利用に統合した 検索拡張済言語モデルは，すでに私の [2020 年のハイライト](https://ruder.io/research-highlights-2020/#2-retrieval-augmentation) で紹介した。
2021 年には検索コーパスが 1 兆トークンまでスケールアップし[91](index.html#fn91)，モデルには質問に答えるためにウェブを照会する機能が搭載された[92](index.html#fn92),[93](index.html#fn93)。
また事前に学習された言語モデルに検索を統合する新しい方法も見られる[94](index.html#fn94),[95](index.html#fn95)。
<!-- Retrieval-augmented language models, which integrate retrieval into pre-training and downstream usage, have already featured in my [highlights of 2020](https://ruder.io/research-highlights-2020/#2-retrieval-augmentation).
In 2021, retrieval corpora have been scaled up to a trillion tokens [91](index.html#fn91) and models have been equipped with the ability to query the web for answering questions [92](index.html#fn92),[93](index.html#fn93).
We have also seen new ways to integrate retrieval into pre-trained language models [94](index.html#fn94),[95](index.html#fn95). -->

**なぜ重要なのか？** <!-- **Why is it important?** -->
検索拡張はモデルのパラメータに格納する知識を減らし，代わりにそれを検索することができるため，モデルのパラメータ効率を大幅に向上させることができる。
また，検索に使用するデータを更新するだけで，効果的なドメイン適応が可能になる[96](index.html#fn96)。
<!-- Retrieval augmentation enables models to be much more parameter-efficient as they need to store less knowledge in their parameters and can instead retrieve it. It also enables effective domain adaptation by simply updating the data used for retrieval [96](index.html#fn96). -->

**次はどうなるのか？** <!-- **What's next?** -->
常識的な知識，事実関係，言語情報など，さまざまな種類の情報を活用するために，異なる形式の検索が見られるかもしれない。
また，検索拡張は，知識ベースの集団やオープンな情報抽出からの手法など，より構造化された形の知識検索と組み合わされる可能性もある。
<!-- We might see different forms of retrieval to leverage different kinds of information such as common sense knowledge, factual relations, linguistic information, etc. 
Retrieval augmentation could also be combined with more structured forms of knowledge retrieval, such as methods from knowledge base population and open information extraction. -->

## 12. トークンフリーモデル 
<!-- ## 12. Token-free Models  -->

<center>
<img src="https://ruder.io/content/images/2022/01/charformer.png" width="88%"><br/>
<div style="text-align:left;width:88%;background-color:cornsilk">

"Charformer" における部分単語ブロックの形成と得点化。
部分単語は連続した n-gram 列に基づいて形成され(a)，別の得点化ネットワークによって得点化される。
ブロックの得点は元の位置に複製される(b)。
最後に，各位置の部分単語を合計し，ブロック得点に基づいて重み付けし，潜在的な部分単語を形成する ([Tay et al.,2021](https://arxiv.org/abs/2106.12672))。<!-- Subword block formation and scoring in Charformer. 
<!-- Subwords are formed based on contiguous n-gram sequences (a), which are scored by a separate scoring network. 
Block scores are then replicated over their original positions (b). 
Finally, subwords at each position are summed, weighted based on their block scores to form latent subwords ([Tay et al.,2021](https://arxiv.org/abs/2106.12672)).] -->
</div>
</center>

<!-- ![Subword block formation and scoring in Charformer. 
Subwords are formed based on contiguous n-gram sequences (a), which are scored by a separate scoring network. 
Block scores are then replicated over their original positions (b). 
Finally, subwords at each position are summed, weighted based on their block scores to form latent subwords ([Tay et al.,2021](https://arxiv.org/abs/2106.12672)).](https://ruder.io/content/images/2022/01/charformer.png "Charformer") -->

**何が起こったのか？** <!-- **What happened?** -->
2021 年には文字列を直接消費する新しいトークンフリー手法が登場した[97](index.html#fn97), [98](index.html#fn98), [99](index.html#fn99)。
これらのモデルは多言語モデルを凌駕し，特に非標準言語に対して優れた性能を発揮することが実証されている。
このため，部分単語に基づく変換モデル (これらの「文字戦争」についての報道は [本ニュースレター](https://newsletter.ruder.io/issues/iclr-2021-outstanding-papers-char-wars-speech-first-nlp-virtual-conference-ideas-483703) 参照) に代わる有望なモデルである。
<!-- 2021 saw the emergence of new token-free methods that directly consume a sequence of characters [97](index.html#fn97),[98](index.html#fn98),[99](index.html#fn99). 
These models have been demonstrated to outperform multilingual models and perform particularly well on non-standard language. 
They are thus a promising alternative to the entrenched subword-based transformer models (see [this newsletter](https://newsletter.ruder.io/issues/iclr-2021-outstanding-papers-char-wars-speech-first-nlp-virtual-conference-ideas-483703) for a coverage of these 'Char Wars'). -->

**なぜそれが重要なのか？** <!-- **Why is it important?** -->
BERT のような事前に学習された言語モデル以来，トークン化された部分単語からなるテキストが自然言語処理における標準的な入力形式となっている。
しかし部分単語トークン化は，ソーシャルメディアによくあるタイプミスやスペルのバリエーション，ある種の形態素など，雑音の多い入力では性能が低いことが示されている。
また，トークン化に依存するため，モデルを新しいデータに適応させる際にミスマッチを起こす可能性がある。
<!-- Since pre-trained language models like BERT, a text consisting of tokenized subwords has become the standard input format in NLP. 
However, subword tokenization has been shown to perform poorly on noisy input, such as on typos or spelling variations common on social media, and on certain types of morphology. 
In addition, it imposes a dependence on the tokenization, which can lead to a mismatch when adapting a model to new data. -->

**次は何？** <!-- **What's next?** -->
トークン・フリー・モデルは柔軟性が高いため，形態素をモデル化するのに適しており，新しい単語や言語変化に対してより良く一般化できる可能性がある。
しかし，形態素や単語形成の異なるタイプの処理において，部分単語に基づく手法と比較してどうなのか，またこれらのモデルがどのようなトレードオフをするのかはまだ不明である。
<!-- Due to their increased flexibility, token-free models are better able to model morphology and may generalize better to new words and language change. 
It is still unclear, however, how they fare compared to subword-based methods on different types of morphological or word formation processes and what trade-offs these models make. -->

## 13. 時間的適応 
<!-- ## 13. Temporal Adaptation  -->

<center>
<img src="https://ruder.io/content/images/2022/01/temporal_strategies.png" width="88%"><br/>
<div style="text-align:left; width:88%; background-color:cornsilk">

"Temporal戦略" T5 を用いた時間適応のための様々な学習戦略。
Uniform model (左) は明示的な時間情報を持たずに全てのデータに対して学習を行う。
Yearly setup (中央) は各年に対して個別のモデルを学習し Temporal model (右) は各例に時間の接頭辞を付ける ([Dhingra et al.,2021](https://arxiv.org/abs/2106.15110)). 
<!-- Different training strategies for temporal adaptation with T5. 
The Uniform model (left) trains on all data without explicit time information. 
The Yearly setup (middle) trains a separate model for each year while the Temporal model (right) prepends a time prefix to each example ([Dhingra et al.,2021](https://arxiv.org/abs/2106.15110)). "Temporal strategies" -->
</div>
</center>

<!-- ![Different training strategies for temporal adaptation with T5. 
The Uniform model (left) trains on all data without explicit time information. 
The Yearly setup (middle) trains a separate model for each year while the Temporal model (right) prepends a time prefix to each example ([Dhingra et al.,2021](https://arxiv.org/abs/2106.15110)).](https://ruder.io/content/images/2022/01/temporal_strategies.png "Temporal strategies") -->

**何が起こったのか？**<!-- **What happened?** -->
モデルは学習させたデータに基づいて様々なバイアスがかかっている。
2021 年に注目されるようになったこれらのバイアスの 1 つはモデルが学習したデータの時間枠に関するバイアスである。
言語は絶えず進化し，新しい用語が言説に入り込むことを考えると，古いデータで訓練されたモデルは比較的汎化が悪いことが示されている [100](index.html#fn100)。
しかし，どのような場合に時間適応が有効かは，下流の課題に依存する可能性がある。
例えば，言語使用におけるイベントドリブンな変化が課題成績に関係しない課題では，時間適応はあまり役に立たないかもしれない [101](index.html#fn101)。
<!-- Models are biased in many ways based on the data that they are trained on. 
One of these biases that has received increasing attention in 2021 is a bias regarding the timeframe of the data the models have been trained on. 
Given that language continuously evolves and new terms enter the discourse, models that are trained on outdated data have been shown to generalize comparatively poorly [100](index.html#fn100). 
When temporal adaptation is useful, however, may depend on the downstream task. 
For instance, it may be less helpful for tasks where event-driven changes in language use are not relevant for task performance [101](index.html#fn101). -->

**なぜ重要なのか？**<!-- **Why is it important?**-->
時間適応は質問がいつなされたかによって答えが変わるような質問応答において特に重要である[102](index.html#fn102)、[103](index.html#fn103).
<!-- Temporal adaptation is particularly important for question answering where answers to a question may change depending on when the question was asked [102](index.html#fn102),[103](index.html#fn103).-->

**次のステップは？** <!-- **What's next?** -->
新しい時間枠に適応できる手法を開発するには，静的な 事前訓練--微調整の設定から脱却し，事前に学習したモデルの知識を更新する効率的な方法が必要である。
[効率的な方法](index.html#5efficientmethods) や [検索拡張](index.html#11retrievalaugmentation) は、この点で有用である。
また，入力が真空中に存在するのではなく，言語外の文脈や実世界に根ざしたモデルを開発する必要がある。
このトピックに関する詳しい作業は EMNLP 2022 の EvoNLP ワークショップを参照。
<!-- Developing methods that can adapt to new timeframes requires moving away from the static pre-train--fine-tune setting and requires efficient ways to update the knowledge of pre-trained models. 
Both [efficient methods](index.html#5efficientmethods) as well as [retrieval augmentation](index.html#11retrievalaugmentation) are useful in this regard. 
It also requires developing models for which the input does not exist in a vacuum but is grounded to extra-linguistic context and the real world. 
For more work on this topic, check out the EvoNLP workshop at EMNLP 2022. -->

## 14. データの重要性 
<!-- ## 14. The Importance of Data  -->

<center>
<img src="https://ruder.io/content/images/2022/01/marvl.png" width="88%"><br/>
<div style="text-align:left; width:88%; background-color:cornsilk">

MaRVL の例ではスワヒリ語の概念 **leso** (ハンカチ) に関するものでキャプションの記述が真か偽かを識別することをモデルに求めている。
キャプション (スワヒリ語) は以下の通り。
Picha moja ina watu kadhaa waliovaa leso na picha nyingine ina leso bila watu.
(**ある絵にはハンカチをつけている人が何人かいて，別の絵には人のいないハンカチがある**)。
ラベルは偽である([Liu et al.,2021](https://aclanthology.org/2021.emnlp-main.818/))。
<!-- An example from MaRVL related to the Swahili concept *leso* ("handkerchief"), which requires models to identify whether the description in the caption is true or false. 
The caption (in Swahili) is: *Picha moja ina watu kadhaa waliovaa leso na picha nyingine ina leso bila watu.* ("One picture contains several people wearing handkerchiefs and another picture has a handkerchief without people."). 
The label is false ([Liu et al.,2021](https://aclanthology.org/2021.emnlp-main.818/)). -->
</div>
</center>

<!-- ![An example from MaRVL related to the Swahili concept *leso* ("handkerchief"), which requires models to identify whether the description in the caption is true or false. 
The caption (in Swahili) is: *Picha moja ina watu kadhaa waliovaa leso na picha nyingine ina leso bila watu.* ("One picture contains several people wearing handkerchiefs and another picture has a handkerchief without people."). 
The label is false ([Liu et al.,2021](https://aclanthology.org/2021.emnlp-main.818/)).](https://ruder.io/content/images/2022/01/marvl.png "MaRVL") -->

**何が起こったのか？** <!-- **What happened?** -->
 データは長い間 ML にとって重要な要素であったが，一般的にモデリングの進歩によって影が薄くなっている。
しかし，モデルのスケールアップのためのデータの重要性を考えると，モデル中心からデータ中心のアプローチへと徐々に関心が移行しつつある。
重要なトピックとしては，新しいデータセットをいかに効率的に構築・維持するか，データの品質をいかに確保するかなどがある (概要については NeurIPS 2021 の [Data-centric AI workshop](https://datacentricai.org/) を参照)。
特に マルチモーダルデータセット[104](index.html#fn104) や，英語・多言語テキストコーパス[105](index.html#fn105),[106](index.html#fn106) など，事前学習したモデルで使用する大規模データセットが今年精査の対象となった。
このような分析はマルチモーダル推論のための MaRVL [107](index.html#fn107) のようなより代表的なリソースの設計に情報を提供することができる。
<!-- Data has long been a critical ingredient for ML but is typically overshadowed by advances in modelling. 
Given the importance of data for scaling up models, however, attention is slowly shifting from model-centric to data-centric approaches. 
Important topics include how to build and maintain new datasets efficiently and how to ensure data quality (see the [Data-centric AI workshop](https://datacentricai.org/) at NeurIPS 2021 for an overview). 
In particular, large-scale datasets used by pre-trained models came under scrutiny this year including multi-modal datasets [104](index.html#fn104) as well as English and multilingual text corpora [105](index.html#fn105),[106](index.html#fn106).
Such an analysis can inform the design of more representative resources such as MaRVL [107](index.html#fn107) for multi-modal reasoning. -->

**なぜそれが重要なのか？** <!-- **Why is it important?** -->
データは大規模な ML モデルを学習させるために非常に重要であり，モデルが新しい情報を獲得するための重要な要因である。
モデルの規模が大きくなるにつれて，規模に応じたデータ品質の確保がより困難になる。

**次はどうするか？** <!-- **What's next?**  -->
現在，様々な課題のためのデータセットを効率的に構築する方法，データ品質を確実に保証する方法などに関するベストプラクティスや原則的な方法が不足している。
また，データがモデルの学習とどのように相互作用し，データがモデルのバイアスをどのように形成するかについても，まだ十分に理解されていない。
例えば，学習データのフィルタリングは，言語モデルの疎外されたグループのカバー率にマイナスの影響を与える可能性がある[90:1](index.html#fn90)。
<!-- Data is critically important for training large-scale ML models and a key factor in how models acquire new information. 
As models are scaled up, ensuring data quality at scale becomes more challenging.
We currently lack best practices and principled methods regarding how to efficiently build datasets for different tasks, reliably ensure data quality, etc. 
It is also still poorly understood how data interacts with a model's learning and how the data shapes a model's biases. 
For instance, training data filtering may have negative effects on a language model\'s coverage of marginalized groups [90:1](index.html#fn90). -->

## 15. メタ学習
<!-- ## 15. Meta-learning  -->

<center>
<img src="https://ruder.io/content/images/2022/01/universal_template.png" width="94%"><br/>
<div style="text-align:left;width:94%;background-color:cornsilk">

ユニバーサルテンプレートモデルの学習とテストの設定。
共有畳み込み重みとデータセット固有の FiLM 層からなるモデルをマルチタスクで学習する(左)。
テストエピソードの FiLM パラメータ値は学習データで学習した FiLM パラメータセットの凸組み合わせに基づいて初期化される (右)。
その後，サポートセットに対する勾配降下法を用いて更新され，出力層には最近接セントロイド分類器を用いている
([Triantafillou et al.,2021](https://arxiv.org/abs/2105.07029))
<!-- The training and test setup of the universal template model. 
A model consisting of shared convolutional weights and dataset-specific FiLM layers is trained in a multi-task setting (left). 
FiLM parameter values for a test episode are initialized based on a convex combination of the trained sets of FiLM parameters, learned on the training data (right). 
They are then updated using gradient descent on the support set, with a nearest-centroid classifier as the output layer ([Triantafillou et al.,2021](https://arxiv.org/abs/2105.07029) -->
</div></center>

<!-- ![The training and test setup of the universal template model. 
A model consisting of shared convolutional weights and dataset-specific FiLM layers is trained in a multi-task setting (left). 
FiLM parameter values for a test episode are initialized based on a convex combination of the trained sets of FiLM parameters, learned on the training data (right). 
They are then updated using gradient descent on the support set, with a nearest-centroid classifier as the output layer ([Triantafillou et al.,2021](https://arxiv.org/abs/2105.07029)).](https://ruder.io/content/images/2022/01/universal_template.png "Universal template") -->

**何が起こったのか？** 
メタ学習と転移学習は，少数点学習という共通の目標があるにもかかわらず， ほとんどが異なるコミュニティで研究されてきた。
新しいベンチマーク[108]では，大規模な転移学習がメタ学習に基づくアプローチを凌駕している。
有望な方向性は，メタ学習法のスケールアップであり，よりメモリ効率の良い学習法と組み合わせることで，実世界のベンチマークにおけるメタ学習モデルの性能を向上させることができる[109]。
また，メタ学習法は FiLM層[110] などの[効率的適応法]と組み合わせることで，一般的なモデルを新しいデータセットに効率的に適応させることができる[111]。
<!-- **What happened?** 
Meta-learning and transfer learning, despite sharing the common goal of few-shot learning, have been studied mostly in distinct communitites. 
On a new benchmark [108], large-scale transfer learning methods outperform meta-learning-based approaches. 
A promising direction is to scale up meta-learning methods, which, combined with more memory-efficient training methods, can improve the performance of meta-learning models on real-world benchmarks [109]. 
Meta-learning methods can also be combined with [efficient adaptation methods] such as FiLM layers [110] to adapt a general model effectively to new datasets [111].-->

**なぜそれが重要なのか？** 
メタ学習は重要なパラダイムであるが，メタ学習システムを念頭に置いて設計されていない標準的なベンチマークでは，最先端の結果を得るには至っていない。
メタ学習と転移学習のコミュニティを近づけることで，実世界の応用に役立つ，より実用的なメタ学習法が生まれるかもしれない。
<!-- **Why is it important?**  
Meta-learning is an important paradigm but has fallen short of yielding state-of-the-art results on standard benchmarks that are not designed with meta-learning systems in mind.
Bringing meta-learning and transfer learning communities closer together may lead to more practical meta-learnig methods that are useful in real-world applications. -->

**次に来るものは何か？**
メタ学習は[大規模マルチタスク学習](index.html#2massivemultitasklearning)で利用できる大量の自然課題と組み合わせると特に有用である。
また，メタ学習は，利用可能な多数のプロンプトに基づいてプロンプトを設計または使用する方法を学習することにより，プロンプトの改善に役立つ場合がある。
<!-- **What's next?** 
Meta-learning can be particularly useful when combined with the large number of natural tasks available for [massive multi-task learning](index.html#2massivemultitasklearning).
Meta-learning can also help improve prompting by learning how to design or use prompts based on the large number of available prompts. -->

## 引用方法

学術的な文献や書籍に引用する場合には，この著作物を以下のように引用して欲しい:
<!-- For attribution in academic contexts or books, please cite this work as: -->
    Sebastian Ruder, "ML and NLP Research Highlights of 2021". http://ruder.io/ml-highlights-2021/, 2022.

BibTeX 用書誌情報:
<!-- BibTeX citation: -->
    @misc{ruder2022mlhighlights,
    author = {Ruder, Sebastian},
    title = {{ML and NLP Research Highlights of 2021}},
    year = {2022},
    howpublished = {\url{http://ruder.io/ml-highlights-2021/}},
    }

## 謝辞
<!-- ## Credits -->

Eleni Triantafillou と Dani Yogatama には，感想やご意見をいただいた。
<!-- Thanks to Eleni Triantafillou and Dani Yogatama for thoughts and suggestions. -->

---

## 文献

1. Bommasani, R., Hudson, D. A., Adeli, E., Altman, R., Arora, S., von Arx, S., ... Liang, P. (2021). On the Opportunities and Risks of Foundation Models. [http://arxiv.org/abs/2108.07258](https://arxiv.org/abs/2108.07258)
2. Dosovitskiy, A., Beyer, L., Kolesnikov, A., Weissenborn, D., Zhai, X., Unterthiner, T., ... Houlsby, N. (2021). An Image is Worth 16x16 Words: Transformers for Image Recognition at Scale. In Proceedings of ICLR 2021. 
3. Zhai, X., Kolesnikov, A., Houlsby, N., & Beyer, L. (2021). Scaling Vision Transformers.  [http://arxiv.org/abs/2106.04560](https://arxiv.org/abs/2106.04560)
4. He, K., Chen, X., Xie, S., Li, Y., Dollár, P., & Girshick, R. (2021). Masked Autoencoders Are Scalable Vision Learners. 
    [http://arxiv.org/abs/2111.06377](https://arxiv.org/abs/2111.06377)
5. Goyal, P., Caron, M., Lefaudeux, B., Xu, M., Wang, P., Pai, V., ... Bojanowski, P. (2021). Self-supervised Pretraining of Visual Features in the Wild.
[http://arxiv.org/abs/2103.01988](https://arxiv.org/abs/2103.01988)
6. Baevski, A., Zhou, H., Mohamed, A., & Auli, M. (2020). wav2vec 2.0: A framework for self-supervised learning of speech representations. Advances in Neural Information Processing Systems, 2020.
7. Chung, Y.-A., Zhang, Y., Han, W., Chiu, C.-C., Qin, J., Pang, R., & Wu, Y. (2021). W2v-BERT: Combining Contrastive Learning and Masked Language Modeling for Self-Supervised Speech Pre-Training.
    [http://arxiv.org/abs/2108.06209](https://arxiv.org/abs/2108.06209)
8. Babu, A., Wang, C., Tjandra, A., Lakhotia, K., Xu, Q., Goyal, N.,
    ... Auli, M. (2021). XLS-R: Self-supervised Cross-lingual Speech
    Representation Learning at Scale.
    [http://arxiv.org/abs/2111.09296](https://arxiv.org/abs/2111.09296)
9. Fu, T.-J., Li, L., Gan, Z., Lin, K., Wang, W. Y., Wang, L., &
    Liu, Z. (2021). VIOLET: End-to-End Video-Language Transformers with
    Masked Visual-token Modeling.
    [http://arxiv.org/abs/2111.12681](https://arxiv.org/abs/2111.12681)
10. Bapna, A., Chung, Y., Wu, N., Gulati, A., Jia, Y., Clark, J. H., ...
    Zhang, Y. (2021). SLAM: A Unified Encoder for Speech and Language
    Modeling via Speech-Text Joint Pre-Training.
    [http://arxiv.org/abs/2110.10329](https://arxiv.org/abs/2110.10329)
11. Bugliarello, E., Cotterell, R., Okazaki, N., & Elliott, D. (2021). 
    Multimodal pretraining unmasked: A meta-analysis and a unified framework of vision-and-language berts. Transactions of the Association for Computational Linguistics, 9, 978--994.
    <https://doi.org/10.1162/tacl_a_00408>
12. Hendricks, L. A., Mellor, J., Schneider, R., Alayrac, J. B., & Nematzadeh, A. (2021). Decoupling the role of data, attention, and losses in multimodal transformers. 
Transactions of the Association for Computational Linguistics, 9, 570--585. <https://doi.org/10.1162/tacl_a_00385>
13. Chen, L., Lu, K., Rajeswaran, A., Lee, K., Grover, A., Laskin, M.,... Mordatch, I. (2021). Decision Transformer: Reinforcement Learning via Sequence Modeling. 
    [http://arxiv.org/abs/2106.01345](https://arxiv.org/abs/2106.01345)
14. Jumper, J., Evans, R., Pritzel, A., Green, T., Figurnov, M., Ronneberger, O., \... & Hassabis, D. (2021). Highly accurate protein structure prediction with AlphaFold. Nature, 596(7873), 583-589.
15. Abnar, S., Dehghani, M., Neyshabur, B., & Sedghi, H. (2021). Exploring the Limits of Large Scale Pre-training. [http://arxiv.org/abs/2110.02095](https://arxiv.org/abs/2110.02095)
16. Tay, Y., Dehghani, M., Rao, J., Fedus, W., Abnar, S., Chung, H. W., ... Metzler, D. (2021). Scale Efficiently: Insights from Pre-training and Fine-tuning Transformers.
    [http://arxiv.org/abs/2109.10686](https://arxiv.org/abs/2109.10686)
17. Sanh, V., Webson, A., Raffel, C., Bach, S. H., Sutawika, L., Alyafeai, Z., ... Rush, A. M. (2021). Multitask Prompted Training Enables Zero-Shot Task Generalization.
    [http://arxiv.org/abs/2110.08207](https://arxiv.org/abs/2110.08207)
18. Wei, J., Bosma, M., Zhao, V. Y., Guu, K., Yu, A. W., Lester, B., ...Le, Q. V. (2021). Finetuned Language Models Are Zero-Shot Learners. [http://arxiv.org/abs/2109.01652](https://arxiv.org/abs/2109.01652)
19.  Aribandi, V., Tay, Y., Schuster, T., Rao, J., Zheng, H. S., Mehta, S. V., ... Metzler, D. (2021). ExT5: Towards Extreme Multi-Task Scaling for Transfer Learning.
    [http://arxiv.org/abs/2111.10952](https://arxiv.org/abs/2111.10952)
20. Bansal, T., Gunasekaran, K., Wang, T., Munkhdalai, T., & McCallum, A. (2021). Diverse Distributions of Self-Supervised Tasks for Meta-Learning in NLP. In Proceedings of EMNLP 2021 (pp.    5812--5824). <https://doi.org/10.18653/v1/2021.emnlp-main.469>
21. Min, S., Lewis, M., Zettlemoyer, L., & Hajishirzi, H. (2021). MetaICL: Learning to Learn In Context. [http://arxiv.org/abs/2110.15943](https://arxiv.org/abs/2110.15943)
22. Vaswani, A., Shazeer, N., Parmar, N., Uszkoreit, J., Jones, L., Gomez, A. N., ... Polosukhin, I. (2017). Attention Is All You Need.  In Proceedings of NIPS 2017.
23. Jaegle, A., Gimeno, F., Brock, A., Zisserman, A., Vinyals, O., &  Carreira, J. (2021). Perceiver: General Perception with Iterative Attention. In Proceedings of ICML 2021.
    [http://arxiv.org/abs/2103.03206](https://arxiv.org/abs/2103.03206)
24. Jaegle, A., Borgeaud, S., Alayrac, J.-B., Doersch, C., Ionescu, C.,
    Ding, D., ... Carreira, J. (2021). Perceiver IO: A General
    Architecture for Structured Inputs & Outputs.
    [http://arxiv.org/abs/2107.14795](https://arxiv.org/abs/2107.14795)
25. Tolstikhin, I., Houlsby, N., Kolesnikov, A., Beyer, L., Zhai, X.,
    Unterthiner, T., ... Dosovitskiy, A. (2021). MLP-Mixer: An all-MLP
    Architecture for Vision.
    [http://arxiv.org/abs/2105.01601](https://arxiv.org/abs/2105.01601)
26. Liu, H., Dai, Z., So, D. R., & Le, Q. V. (2021). Pay Attention to
    MLPs, (Mlm). Retrieved from
    [http://arxiv.org/abs/2105.08050](https://arxiv.org/abs/2105.08050)
27. Lee-Thorp, J., Ainslie, J., Eckstein, I., & Ontanon, S. (2021).
    FNet: Mixing Tokens with Fourier Transforms.
    [http://arxiv.org/abs/2105.03824](https://arxiv.org/abs/2105.03824)
28. Tay, Y., Dehghani, M., Gupta, J., Bahri, D., Aribandi, V., Qin, Z.,
    & Metzler, D. (2021). Are Pre-trained Convolutions Better than
    Pre-trained Transformers? In Proceedings of ACL 2021. Retrieved from
    [http://arxiv.org/abs/2105.03322](https://arxiv.org/abs/2105.03322)
29. Clark, K., Luong, M.-T., Le, Q. V., & Manning, C. D. (2020).
    ELECTRA: Pre-training Text Encoders as Discriminators Rather Than
    Generators. In Proceedings of ICLR 2020.
30. He, P., Gao, J., & Chen, W. (2021). DeBERTaV3: Improving DeBERTa using ELECTRA-Style Pre-Training with Gradient-Disentangled Embedding Sharing.
    [http://arxiv.org/abs/2111.09543](https://arxiv.org/abs/2111.09543)
31. Agarwal, R., Melnick, L., Frosst, N., Zhang, X., Lengerich, B.,
    Caruana, R., & Hinton, G. (2021). Neural Additive Models:
    Interpretable Machine Learning with Neural Nets. In Proceedings of
    NeurIPS 2021.
    [http://arxiv.org/abs/2004.13912](https://arxiv.org/abs/2004.13912)
32. Brown, T. B., Mann, B., Ryder, N., Subbiah, M., Kaplan, J.,
    Dhariwal, P., ... Amodei, D. (2020). Language Models are Few-Shot
    Learners. In Proceedings of NeurIPS 2020.
    [http://arxiv.org/abs/2005.14165](https://arxiv.org/abs/2005.14165)
33. Schick, T., & Schütze, H. (2021). Exploiting cloze questions for few
    shot text classification and natural language inference. In
    Proceedings of EACL 2021 (pp. 255--269).
34. Schick, T., & Schütze, H. (2021). It's Not Just Size That Matters:
    Small Language Models Are Also Few-Shot Learners. In Proceedings of
    NAACL 2021. [http://arxiv.org/abs/2009.07118](https://arxiv.org/abs/2009.07118)
35. Tam, D., Menon, R. R., Bansal, M., Srivastava, S., & Raffel, C. (2021). Improving and Simplifying Pattern Exploiting Training. [http://arxiv.org/abs/2103.11955](https://arxiv.org/abs/2103.11955)
36. Perez, E., Kiela, D., & Cho, K. (2021). True Few-Shot Learning with Language Models. In Proceedings of NeurIPS 2021. [http://arxiv.org/abs/2105.11447](https://arxiv.org/abs/2105.11447)
37. Zheng, Y., Zhou, J., Qian, Y., Ding, M., Li, J., Salakhutdinov, R., ... Yang, Z. (2021). FewNLU: Benchmarking State-of-the-Art Methods for Few-Shot Natural Language Understanding.
    [http://arxiv.org/abs/2109.12742](https://arxiv.org/abs/2109.12742)
38. Liu, P., Yuan, W., Fu, J., Jiang, Z., Hayashi, H., & Neubig, G. (2021). Pre-train, Prompt, and Predict: A Systematic Survey of Prompting Methods in Natural Language Processing. [http://arxiv.org/abs/2107.13586](https://arxiv.org/abs/2107.13586)
39. Scao, T. Le, & Rush, A. M. (2021). How Many Data Points is a Prompt Worth? In Proceedings of NAACL 2021. [http://arxiv.org/abs/2103.08493](https://arxiv.org/abs/2103.08493)
40. Ratner, A., De Sa, C., Wu, S., Selsam, D., & Ré, C. (2016). Data Programming: Creating Large Training Sets, Quickly. In Advances in Neural Information Processing Systems 29 (NIPS 2016).
    [http://arxiv.org/abs/1605.07723](https://arxiv.org/abs/1605.07723)
41. Mishra, S., Khashabi, D., Baral, C., & Hajishirzi, H. (2021). 
    Cross-Task Generalization via Natural Language Crowdsourcing Instructions.
    [http://arxiv.org/abs/2104.08773](https://arxiv.org/abs/2104.08773)
42. Wiegreffe, S., & Marasović, A. (2021). Teach Me to Explain: A Review of Datasets for Explainable Natural Language Processing. In 35th
    Conference on Neural Information Processing Systems (NeurIPS 2021)
    Track on Datasets and Benchmarks.
    [http://arxiv.org/abs/2102.12060](https://arxiv.org/abs/2102.12060)
43. Ma, X., Kong, X., Wang, S., Zhou, C., May, J., Ma, H., & Zettlemoyer, L. (2021). Luna: Linear Unified Nested Attention. In Proceedings of NeurIPS 2021.
    [http://arxiv.org/abs/2106.01540](https://arxiv.org/abs/2106.01540)
44. Peng, H., Pappas, N., Yogatama, D., Schwartz, R., Smith, N. A., & Lingpeng Kong. (2021). Random Feature Attention. In Proceedings of ICLR 2021. 
45.  Tay, Y., Dehghani, M., Bahri, D., & Metzler, D. (2020). Efficient Transformers: A Survey. ArXiv Preprint ArXiv:2009.06732. Retrieved from [http://arxiv.org/abs/2009.06732](https://arxiv.org/abs/2009.06732)
46.  Lester, B., Al-Rfou, R., & Constant, N. (2021). The Power of Scale for Parameter-Efficient Prompt Tuning. In Proceedings of EMNLP 2021. [http://arxiv.org/abs/2104.08691](https://arxiv.org/abs/2104.08691)
47.  Liu, X., Zheng, Y., Du, Z., Ding, M., Qian, Y., Yang, Z., & Tang, J. (2021). GPT Understands, Too. [http://arxiv.org/abs/2103.10385](https://arxiv.org/abs/2103.10385)
48.  
    Mao, Y., Mathias, L., Hou, R., Almahairi, A., Ma, H., Han, J., ...
    Khabsa, M. (2021). UniPELT: A Unified Framework for
    Parameter-Efficient Language Model Tuning.
    [http://arxiv.org/abs/2110.07577](https://arxiv.org/abs/2110.07577)
49.  
    He, J., Zhou, C., Ma, X., Berg-Kirkpatrick, T., & Neubig, G. (2021).
    Towards a Unified View of Parameter-Efficient Transfer Learning.
    [http://arxiv.org/abs/2110.04366](https://arxiv.org/abs/2110.04366)
50.  
    Mahabadi, R. K., Henderson, J., & Ruder, S. (2021). Compacter:
    Efficient Low-Rank Hypercomplex Adapter Layers. In Proceedings of
    NeurIPS 2021.
    [http://arxiv.org/abs/2106.04647](https://arxiv.org/abs/2106.04647)
51. Tsimpoukelli, M., Menick, J., Cabi, S., Eslami, S. M. A., Vinyals,
    O., & Hill, F. (2021). Multimodal Few-Shot Learning with Frozen
    Language Models. In Proceedings of NeurIPS 2021.
    [http://arxiv.org/abs/2106.13884](https://arxiv.org/abs/2106.13884)
52. Pfeiffer, J., Vulić, I., Gurevych, I., & Ruder, S. (2020). MAD-X: An
    Adapter-based Framework for Multi-task Cross-lingual Transfer. In
    Proceedings of EMNLP 2020.
53. Eichenberg, C., Black, S., Weinbach, S., Parcalabescu, L., & Frank, A. (2021). MAGMA\--Multimodal Augmentation of Generative Models through Adapter-based Finetuning.
    <https://arxiv.org/abs/2112.05253>
54. Dettmers, T., Lewis, M., Shleifer, S., & Zettlemoyer, L. (2021).
    8-bit Optimizers via Block-wise Quantization.
    [http://arxiv.org/abs/2110.02861](https://arxiv.org/abs/2110.02861)
55. Koch, B., Denton, E., Hanna, A., & Foster, J. G. (2021). Reduced, Reused and Recycled: The Life of a Dataset in Machine Learning Research. In 35th Conference on Neural Information Processing
    Systems (NeurIPS 2021) Track on Datasets and Benchmarks. [http://arxiv.org/abs/2112.01716](https://arxiv.org/abs/2112.01716)
56. Kiela, D., Bartolo, M., Nie, Y., Kaushik, D., Geiger, A., Wu, Z.,... Williams, A. (2021). Dynabench: Rethinking Benchmarking in NLP. 
    In Proceedings of NAACL 2021 (pp. 4110--4124). <https://doi.org/10.18653/v1/2021.naacl-main.324>
57. Liu, P., Fu, J., Xiao, Y., Yuan, W., Chang, S., Dai, J., ...Neubig, G. (2021). ExplainaBoard: An Explainable Leaderboard for NLP. In Proceedings of ACL 2021: System demonstrations (pp.
    280--289). 
58. Ma, Z., Ethayarajh, K., Thrush, T., Jain, S., Wu, L., Jia, R., ...
    Kiela, D. (2021). Dynaboard: An Evaluation-As-A-Service Platform for
    Holistic Next-Generation Benchmarking.
    [http://arxiv.org/abs/2106.06052](https://arxiv.org/abs/2106.06052)
59. Bragg, J., Cohan, A., Lo, K., & Beltagy, I. (2021). FLEX: Unifying
    Evaluation for Few-Shot NLP. In Proceedings of NeurIPS 2021.
    Retrieved from
    [http://arxiv.org/abs/2107.07170](https://arxiv.org/abs/2107.07170)
60. Ye, Q., Lin, B. Y., & Ren, X. (2021). CrossFit: A Few-shot Learning
    Challenge for Cross-task Generalization in NLP. In Proceedings of
    EMNLP 2021. 
61. Koh, P. W., Sagawa, S., Marklund, H., Xie, S. M., Zhang, M.,
    Balsubramani, A., ... Liang, P. (2021). WILDS: A Benchmark of
    in-the-Wild Distribution Shifts. In Proceedings of ICML 2021.
    [http://arxiv.org/abs/2012.07421](https://arxiv.org/abs/2012.07421)
62. Yang, S., Chi, P.-H., Chuang, Y.-S., Lai, C.-I. J., Lakhotia, K.,
    Lin, Y. Y., ... Lee, H. (2021). SUPERB: Speech processing Universal
    PERformance Benchmark. In Proceedings of Interspeech 2021.
    [http://arxiv.org/abs/2105.01051](https://arxiv.org/abs/2105.01051)
63. Cahyawijaya, S., Winata, G. I., Wilie, B., Vincentio, K., Li, X.,
    Kuncoro, A., ... Fung, P. (2021). IndoNLG: Benchmark and Resources for Evaluating Indonesian Natural Language Generation. In Proceedings of EMNLP 2021 (pp. 8875--8898). <https://doi.org/10.18653/v1/2021.emnlp-main.699>
64. Dumitrescu, S., Rebeja, P., Rosia, L., Marchidan, G., Yogatama, D.,
    Avram, A., ... Morogan, L. (2021). LiRo: Benchmark and leaderboard
    for Romanian language tasks. In 35th Conference on Neural Information Processing Systems (NeurIPS 2021) Track on Datasets and Benchmarks. 
65.  Tamkin, A., Liu, V., Lu, R., Fein, D., Schultz, C., & Goodman, N. (2021). DABS: A Domain-Agnostic Benchmark for Self-Supervised
    Learning. In 35th Conference on Neural Information Processing Systems (NeurIPS 2021) Track on Datasets and Benchmarks.
    [http://arxiv.org/abs/2111.12062](https://arxiv.org/abs/2111.12062)
66. Ruder, S., Constant, N., Botha, J., Siddhant, A., Firat, O., Fu, J.,... Johnson, M. (2021). XTREME-R: Towards More Challenging and Nuanced Multilingual Evaluation. 
In Proceedings of EMNLP 2021. [http://arxiv.org/abs/2104.07412](https://arxiv.org/abs/2104.07412)
67.  Marie, B., Fujita, A., & Rubino, R. (2021). Scientific Credibility of Machine Translation Research: A Meta-Evaluation of 769 Papers. In Proceedings of ACL 2021 (pp. 7297--7306).
    <https://doi.org/10.18653/v1/2021.acl-long.566>
68.  Gehrmann, S., Adewumi, T., Aggarwal, K., Ammanamanchi, P. S., Anuoluwapo, A., Bosselut, A., ... Zhou, J. (2021). The GEM Benchmark: Natural Language Generation, its Evaluation and Metrics.
    [http://arxiv.org/abs/2102.01672](https://arxiv.org/abs/2102.01672)
69.  Kasai, J., Sakaguchi, K., Bras, R. Le, Dunagan, L., Morrison, J., Fabbri, A. R., ... Smith, N. A. (2021). Bidimensional Leaderboards: Generate and Evaluate Language Hand in Hand.
    [http://arxiv.org/abs/2112.04139](https://arxiv.org/abs/2112.04139)
70.  Ramesh, A., Pavlov, M., Goh, G., Gray, S., Voss, C., Radford, A.,... Sutskever, I. (2021). Zero-Shot Text-to-Image Generation.  <https://arxiv.org/abs/2102.12092>
71.  Esser, P., Rombach, R., & Ommer, B. (2020). Taming Transformers for High-Resolution Image Synthesis. <https://arxiv.org/abs/2012.09841>
72.  Radford, A., Kim, J. W., Hallacy, C., Ramesh, A., Goh, G., Agarwal, S., ... Sutskever, I. (2021). Learning Transferable Visual Models From Natural Language Supervision.
    [http://arxiv.org/abs/2103.00020](https://arxiv.org/abs/2103.00020)
73.  
    Dhariwal, P., & Nichol, A. (2021). Diffusion Models Beat GANs on
    Image Synthesis. <https://arxiv.org/abs/2105.05233>
74.  Nichol, A., Dhariwal, P., Ramesh, A., Shyam, P., Mishkin, P., McGrew, B., ... Chen, M. (2021). GLIDE: Towards Photorealistic Image Generation and Editing with Text-Guided Diffusion Models. [http://arxiv.org/abs/2112.10741](https://arxiv.org/abs/2112.10741)
75.  
    Ravuri, S., Lenc, K., Willson, M., Kangin, D., Lam, R., Mirowski,
    P., ... & Mohamed, S. (2021). Skillful Precipitation Nowcasting
    using Deep Generative Models of Radar. Nature, 597.
    <https://www.nature.com/articles/s41586-021-03854-z>
76.  
    Espeholt, L., Agrawal, S., Sønderby, C., Kumar, M., Heek, J.,
    Bromberg, C., ... Kalchbrenner, N. (2021). Skillful Twelve Hour
    Precipitation Forecasts using Large Context Neural Networks, 1--34.
    Retrieved from
    [http://arxiv.org/abs/2111.07470](https://arxiv.org/abs/2111.07470)
77.  
    Davies, A., Veličković, P., Buesing, L., Blackwell, S., Zheng, D.,
    Tomašev, N., ... Kohli, P. (2021). Advancing mathematics by guiding
    human intuition with AI. Nature, 600(7887), 70--74.
    <https://doi.org/10.1038/s41586-021-04086-x>
78.  
    Charton, F., Hayat, A., & Lample, G. (2021). Deep Differential
    System Stability Learning advanced computations from examples. In
    Proceedings of ICLR 2021. 
79.  
    Chen, M., Tworek, J., Jun, H., Yuan, Q., Ponde, H., Kaplan, J., ...
    Zaremba, W. (2021). Evaluating Large Language Models Trained on
    Code. Retrieved from
    [http://arxiv.org/abs/2107.03374](https://arxiv.org/abs/2107.03374)
80.  
    Roziere, B., Marc, M. L., & Guillaume, S. (2021). DOBF: A
    Deobfuscation Pre-Training Objective for Programming Languages. In
    Proceedings of NeurIPS 2021.
81.  
    Jain, P., Jain, A., Zhang, T., Abbeel, P., Gonzalez, J. E., &
    Stoica, I. (2021). Contrastive Code Representation Learning. In
    Proceedings of EMNLP 2021.
82.  
    Elnaggar, A., Gibbs, T., & Matthes, F. (2021). CodeTrans: Towards
    Cracking the Language of Silicone's Code Through Self-Supervised
    Deep Learning and High Performance Computing.
83.  
    Austin, J., Odena, A., Nye, M., Bosma, M., Michalewski, H., Dohan,
    D., ... Sutton, C. (2021). Program Synthesis with Large Language
    Models, 1--34.
    [http://arxiv.org/abs/2108.07732](https://arxiv.org/abs/2108.07732)
84.  
    Nye, M., Andreassen, A. J., Gur-Ari, G., Michalewski, H., Austin,
    J., Bieber, D., ... Odena, A. (2021). Show Your Work: Scratchpads
    for Intermediate Computation with Language Models, 1--16.
    [http://arxiv.org/abs/2112.00114](https://arxiv.org/abs/2112.00114)
85.  
    Xu, F. F., Vasilescu, B., & Neubig, G. (2021). In-IDE Code
    Generation from Natural Language: Promise and Challenges. ACM
    Transactions on Software Engineering and Methodology.
86.  
    Bender, E., Gebru, T., McMillan-Major, A., & Shmitchell, S. (2021).
    On the dangers of stochastic parrots: can language models be too
    big? In FAccT 2021 - Proceedings of the 2021 ACM Conference on
    Fairness, Accountability, and Transparency. Association for
    Computing Machinery. <https://doi.org/10.1145/3442188.3445922>
87.  
    Weidinger, L., Mellor, J., Rauh, M., Griffin, C., Uesato, J., Huang,
    P.-S., ... Gabriel, I. (2021). Ethical and social risks of harm from
    Language Models. 
88.  
    Liu, R., Jia, C., Wei, J., Xu, G., Wang, L., & Vosoughi, S. (2021).
    Mitigating Political Bias in Language Models Through Reinforced
    Calibration. In Proceedings of AAAI 2021.
89.  
    Ahn, J., & Oh, A. (2021). Mitigating Language-Dependent Ethnic Bias
    in BERT. In Proceedings of EMNLP 2021.
    <https://doi.org/10.18653/v1/2021.emnlp-main.42>
90.  
    Welbl, J., Glaese, A., Uesato, J., Dathathri, S., Mellor, J.,
    Hendricks, L. A., ... Huang, P.-S. (2021). Challenges in Detoxifying
    Language Models. In Findings of EMNLP 2021 (pp. 2447--2469).
    [http://arxiv.org/abs/2109.07445](https://arxiv.org/abs/2109.07445)
91.  
    Borgeaud, S., Mensch, A., Hoffmann, J., Cai, T., Rutherford, E.,
    Millican, K., ... Sifre, L. (2021). Improving language models by
    retrieving from trillions of tokens.
    [http://arxiv.org/abs/2112.04426](https://arxiv.org/abs/2112.04426)
92.  
    Komeili, M., Shuster, K., & Weston, J. (2021). Internet-Augmented
    Dialogue Generation. 
93.  
    Nakano, R., Hilton, J., Balaji, S., Wu, J., Ouyang, L., Kim, C., ...
    Schulman, J. (2021). WebGPT: Browser-assisted question-answering
    with human feedback.
    [http://arxiv.org/abs/2112.09332](https://arxiv.org/abs/2112.09332)
94.  
    Sachan, D. S., Reddy, S., Hamilton, W., Dyer, C., & Yogatama, D.
    (2021). End-to-End Training of Multi-Document Reader and Retriever
    for Open-Domain Question Answering. In Proceedings of NeurIPS 2021.
    [http://arxiv.org/abs/2106.05346](https://arxiv.org/abs/2106.05346)
95.  
    Yogatama, D., D'autume, C. de M., & Kong, L. (2021). Adaptive
    semiparametric language models. Transactions of the Association for
    Computational Linguistics, 9, 362--373.
    <https://doi.org/10.1162/tacl_a_00371>
96.  
    Khandelwal, U., Levy, O., Jurafsky, D., Zettlemoyer, L., & Lewis, M.
    (2020). Generalization through Memorization: Nearest Neighbor
    Language Models. In Proceedings of ICLR 2020.
    [http://arxiv.org/abs/1911.00172](https://arxiv.org/abs/1911.00172)
97.  
    Xue, L., Barua, A., Constant, N., Al-Rfou, R., Narang, S., Kale, M.,
    ... Raffel, C. (2021). ByT5: Towards a token-free future with
    pre-trained byte-to-byte models. ArXiv Preprint ArXiv:2105.13626.
    ttp://arxiv.org/abs/2105.13626
98.  
    Clark, J. H., Garrette, D., Turc, I., & Wieting, J. (2021). Canine:
    Pre-training an Efficient Tokenization-Free Encoder for Language
    Representation. <https://arxiv.org/abs/2103.06874>
99.  
    Tay, Y., Tran, V. Q., Ruder, S., Gupta, J., Chung, H. W., Bahri, D.,
    ... Metzler, D. (2021). Charformer: Fast Character Transformers via
    Gradient-based Subword Tokenization.
    [http://arxiv.org/abs/2106.12672](https://arxiv.org/abs/2106.12672)
100.  
     Lazaridou, A., Kuncoro, A., & Gribovskaya, E. (2021). Mind the Gap
     : Assessing Temporal Generalization in Neural Language Models. In
     Proceedings of NeurIPS 2021.
101.  
     Röttger, P., & Pierrehumbert, J. B. (2021). Temporal Adaptation of
     BERT and Performance on Downstream Document Classification:
     Insights from Social Media. In Findings of EMNLP 2021 (pp.
     2400--2412). <https://doi.org/10.18653/v1/2021.findings-emnlp.206>
102.  
     Dhingra, B., Cole, J. R., Eisenschlos, J. M., Gillick, D.,
     Eisenstein, J., & Cohen, W. W. (2021). Time-Aware Language Models
     as Temporal Knowledge Bases.
     [http://arxiv.org/abs/2106.15110](https://arxiv.org/abs/2106.15110)
103.  
     Zhang, M. J. Q., & Choi, E. (2021). SituatedQA: Incorporating
     Extra-Linguistic Contexts into QA. In Proceedings of EMNLP 2021.
     <https://doi.org/10.18653/v1/2021.emnlp-main.586>
104.  
     Birhane, A., Prabhu, V. U., & Kahembwe, E. (2021). Multimodal
     datasets: misogyny, pornography, and malignant stereotypes.
     [http://arxiv.org/abs/2110.01963](https://arxiv.org/abs/2110.01963)
105.  
     Dodge, J., Sap, M., Marasović, A., Agnew, W., Ilharco, G.,
     Groeneveld, D., & Gardner, M. (2021). Documenting the English
     Colossal Clean Crawled Corpus. In Proceedings of EMNLP 2021.
106.  
     Kreutzer, J., Caswell, I., Wang, L., Wahab, A., Esch, D. van,
     Ulzii-Orshikh, N., ... Adeyemi, M. (2021). Quality at a Glance: An
     Audit of Web-Crawled Multilingual Datasets. In Transactions of the
     ACL 2021. 
107.  
     Liu, F., Bugliarello, E., Ponti, E. M., Reddy, S., Collier, N., &
     Elliott, D. (2021). Visually Grounded Reasoning across Languages
     and Cultures. In Proceedings of EMNLP 2021.
     <https://arxiv.org/abs/2109.13238>
108.  
     Dumoulin, V., Houlsby, N., Evci, U., Zhai, X., Goroshin, R., Gelly,
     S., & Larochelle, H. (2021). Comparing Transfer and Meta Learning
     Approaches on a Unified Few-Shot Classification Benchmark. In 35th
     Conference on Neural Information Processing Systems (NeurIPS 2021)
     Track on Datasets and Benchmarks.
     [http://arxiv.org/abs/2104.02638](https://arxiv.org/abs/2104.02638)
109.  
     Bronskill, J., Massiceti, D., Patacchiola, M., Hofmann, K.,
     Nowozin, S., & Turner, R. E. (2021). Memory Efficient Meta-Learning
     with Large Images, (NeurIPS).
     [http://arxiv.org/abs/2107.01105](https://arxiv.org/abs/2107.01105)
110.  
     Perez, E., Strub, F., De Vries, H., Dumoulin, V., & Courville, A.
     (2018). FiLM: Visual reasoning with a general conditioning layer.
     In 32nd AAAI Conference on Artificial Intelligence, AAAI 2018 (pp.
     3942--3951). 
111. 
     Triantafillou, E., Larochelle, H., Zemel, R., & Dumoulin, V.
     (2021). Learning a Universal Template for Few-shot Dataset
     Generalization. In Proceedings of ICML 2021.
     [http://arxiv.org/abs/2105.07029](https://arxiv.org/abs/2105.07029)
