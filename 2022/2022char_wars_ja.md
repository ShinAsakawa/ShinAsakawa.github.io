---
title: "文字戦争 Char Wars"
author: 浅川伸一
layout: default
---

## [Char Wars](https://newsletter.ruder.io/issues/iclr-2021-outstanding-papers-char-wars-speech-first-nlp-virtual-conference-ideas-483703)

> 部分単語の脅威 (The Subword Menace) (inspired by a tweet by [Sasha Rush](https://twitter.com/srush_nlp/status/1377605631780261888?utm_campaign=NLP%20News&utm_medium=email&utm_source=Revue%20newsletter))

少し前に、あなたの近くのプレプリントサーバーで...。<!-- Not long ago on a preprint server near you…-->

バイト単位の戦いの時代 <!-- It is a period of byte-based battles. -->

[最近の論文](https://arxiv.org/abs/2103.06874?utm_campaign=NLP%20News&utm_medium=email&utm_source=Revue%20newsletter) で提案された純粋な文字レベルモデルは，偏った部分単語の秩序に初勝利を収めた。
<!-- Pure character-level models, proposed in [a recent paper](https://arxiv.org/abs/2103.06874?utm_campaign=NLP%20News&utm_medium=email&utm_source=Revue%20newsletter), have won their first victory against the biased Subword Order. -->

**部分単語トークン化** <!--**Subword tokenization**. -->
Transformer のアーキテクチャの他に，自然言語処理における最先端のモデルの特徴として，部分単語型トークン化がある。
部分単語トークン化は，他の型トークン化と同様，ある種のデータには他の型よりも適しているという前提で行われる。
具体的には，文字列を頻度で分割することに依存する。
これは標準的な英語のテキストではうまくいきますが、サブワード トークン化を使用するモデルは，自然なノイズ (タイプミス，ソーシャルメディアにおけるスペルのばらつきなど: Sun et al.2020) と合成ノイズ (敵対的事例: Pruthi et al.2019) の両方で苦労している。
<!-- Besides the Transformer architecture, the other hegemonic feature of state-of-the-art models in NLP is subword tokenization. 
Subword tokenization—like any type of tokenization really—makes assumptions that are more suitable for some type of data than others. 
Specifically, it relies on splitting strings by frequency. 
While this works well on standard English text, models using subword tokenization struggle with noise, both natural (typos, spelling variations in social media, etc.; Sun et al., 2020) and synthetic (adversarial examples; Pruthi et al., 2019).  -->

**非連鎖式形態素** <!-- **Non-concatenative morphology**. -->
部分単語トークン化は，形態素が連続的に連なっていない単語のモデル化が苦手なことでも知られている (これは非連鎖性形態素として知られている)。
英語では foot → feet のように不規則名詞の複数形で見られることがある，ヘブライ語やアラビア語のような他の言語ではより一般的である。
<!--
Subword tokenization is also notoriously bad at modelling words that don’t consist of morphemes strung together sequentially—which is known as non-concatenative morphology. 
It can be seen occasionally in English with the plural of irregular nouns, such as foot → feet, but is much more common in other languages such as Hebrew and Arabic.  -->

**部分単語トークン化の改善** 
これらの課題に対処する方法の 1 つは，部分単語トークン化をより堅牢にすることである。
部分単語正規化 ([Kudo et al., 2018](https://arxiv.org/abs/1804.10959)) は，入力に対して異なる単語分割をサンプリングすることでこれを実現するもので，単語分割に対するドロップアウトと見なすことができる。
最近の [NAACL 2021 の論文](https://arxiv.org/abs/2103.08490) では，これを一貫性正則化目的関数 (半教師付き学習からのアイデアに触発された) と組み合わせ，事前学習された多言語モデルをより堅牢にしている。
このマルチビュー部分単語正則化は，微調整の際にのみ適用することができ，他の言語に移行する際に一貫して性能を向上させることができるのが良い点である。
<!-- **Improving subword tokenization**. 
One way to deal with these challenges is to make subword tokenization more robust. 
Subword regularization (Kudo et al., 2018) achieves this by sampling different segmentations for the input—it can be seen as dropout over segmentations. 
In a recent NAACL 2021 paper, we combine this with a consistency regularization objective (inspired by ideas from semi-supervised learning) to make pre-trained multilingual models more robust. 
The nice thing is that this multi-view subword regularization can be applied only during fine-tuning and improves performance consistently when transferring to other languages. -->

**文字ベースモデル**. <!-- **Character-based models**. -->
純粋な文字ベースのモデルは，一般的に単語レベルの対応するモデルを下回っている。
その代わりに，モデルは一般的に単語の文字に CNN を使用して文字を考慮した表現 ([Kim et al.,2016](https://arxiv.org/abs/1508.06615)) を取得し，これは ELMo ([Peters et al.,2018](https://www.aclweb.org/anthology/N18-1202/)) にも使用されている。
この方法は BERT にも適用されているが ([Boukkouri et al.,2020](https://arxiv.org/abs/2010.10392))，一般に効率が悪く，部分単語トークン化に基づく Transformer に負けている。
最近では，文字認識と部分単語ベースの情報を組み合わせて，スペルミスに対する頑健性を向上させることも行われている ([Ma et al.,2020](https://arxiv.org/abs/2011.01513))。
<!-- Pure character-based models have generally underperformed their word-level counterparts. 
Instead, models typically obtain a character-aware representation (Kim et al.,2016) using a CNN over the characters of a word, which has also been used in ELMo (Peters et al.,2018).
While this method has been applied to BERT (Boukkouri et al.,2020), it is generally less efficient and outperformed by subword tokenization-based Transformers. 
Recently, character-aware and subword-based information has also been combined, improving robustness to spelling errors (Ma et al.,2020). -->

<img src="https://s3.amazonaws.com/revue/items/images/008/639/789/mail/canine.png?1617553016">

**CANINE**. 
CANINE ([Clark et al., 2021](https://arxiv.org/abs/2103.06874)) は最近の Transformer モデルで，トークン化不要のため純粋な文字ベースモデルの伝統を受け継ぎ，入力として文字系列を直接使用する。
ダウンサンプリングとアップサンプリングの巧みな組み合わせにより，他の文字レベルモデルと比較してより効率的である (上記参照)。
局所的な自己注意を持つ Transformer は，文脈に応じた文字埋め込みを生成し，それをストライド畳み込みによってダウンサンプリングする;
次に (BERTのような) 標準的な深層 Transformer がこの系列に適用される。
最後に 2 つの Transformerの 表現が連結され，アップサンプリングされる。
事前学習では (空白の境界に基づいて選択された) 文字スパンがランダムにマスクされ，予測される。
このモデルは，多言語オープンドメイン質問応答データセット TyDi QA において mBERT を上回る性能を示した。
<!-- CANINE (Clark et al., 2021) is a recent Transformer model that follows in the tradition of pure character-based models by being tokenization-free—it directly consumes a sequence of characters as the input. 
It is more efficient compared to other character-level models by means of a clever combination of down and up-sampling (see above): 
A Transformer with local self-attention produces contexualized character embeddings, which are then down-sampled via strided convolutions; a standard deep Transformer (as in BERT) is then applied to this sequence; 
finally, the representations of the two Transformers are concatenated and up-sampled. 
For pre-training, character spans (chosen based on whitespace boundaries) are randomly masked and predicted. 
The model outperforms mBERT on the multilingual open-domain question answering dataset TyDi QA. -->

CANINE は，部分単語トークン化から一歩進んで，より柔軟で入力データ変動に適したモデルへと進化している。
このようなモデルは，他の言語だけでなく，新しい単語や言語の変化に対応したモデルの一般化を可能にする可能性がある ([前回のニュースレター](https://newsletter.ruder.io/issues/ie-how-did-we-get-here-large-lms-the-human-side-of-ml-292310)参照)。
しかし，部分単語分割は，そのシンプルさと使いやすさから，依然として標準的な手法であり続けるだろう。
したがって，最終的に誰がこの戦争に勝利するかはまだわからない...
<!-- CANINE is a step beyond subword tokenization and towards models that are more flexible and better suited to handle variations in the input data. 
Such models hold promise not only for other languages but may also enable models to generalize better to new words and language change (see the last newsletter). 
However, subword segmentation may still stay the standard due to its simplicity and ease of use; so it remains to be seen who will ultimately win this war… -->

部分単語の不吉なセンテンスピースに追われながら，新しい単語分割手法の研究は，我々のモデルをトークン化の重荷から解放し，世界の言語に正当な単語分割を回復することができる帰納バイアスの管理者を求めて前進している...
<!-- Pursued by the Subword’s sinister sentence pieces, research on new segmentation methods races forward, in search of the custodian of the inductive biases that can free our models from the burdens of their tokenization and restore the rightful segmentation to the world’s languages… -->

