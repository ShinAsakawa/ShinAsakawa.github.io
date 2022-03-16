---
title: Hugging Face's notebooks
layout: default
---

# HuggingFace's notebooks 

このページは，Huggeingface の提供している [Documentation notebooks](https://huggingface.co/docs/transformers/master/en/notebooks) を一部，日本語化したものです。

本文書のどのページも colab ノートブックとして開くことができますが (当該ページに直接ボタンがあります)，必要に応じてここにリストアップされています。
<!-- You can open any page of the documentation as a notebook in colab (there is a button directly on said pages) but they are also listed here if you need to:-->

## Huggingface's notebooks

文書ノートブック

* **クイックツアー Quicktour of the library**,  Transformers 各種 API の紹介 <!-- A presentation of the various APIs in Transformers -->, [クイックツアー <img src="assets/colab_icon.svg">](https://colab.research.google.com/github/ShinAsakawa/ShinAsakawa.github.io/blob/master/2022notebooks/2022_0112quicktour.ipynb), [オリジナル `quicktour.ipynb` Open In Colab](https://colab.research.google.com/github/huggingface/notebooks/blob/master/transformers_doc/quicktour.ipynb)
* **課題のまとめ Summary of the tasks**, 課題ごとの Transfomers モデルの実行方法 <!-- How to run the models of the Transformers library task by task -->, [課題のまとめ <img src="assets/colab_icon.svg">](https://colab.research.google.com/github/ShinAsakawa/ShinAsakawa.github.io/blob/master/2022notebooks/2022_0112task_summary.ipynb), 
[オリジナル `task_summary.ipynb` Open In Colab](https://colab.research.google.com/github/huggingface/notebooks/blob/master/transformers_doc/task_summary.ipynb)
* **データの前処理 Preprocessing data**, トークン化器を使ったデータの前処理方法<!-- How to use a tokenizer to preprocess your data -->, [データ前処理  <img src="assets/colab_icon.svg">](https://colab.research.google.com/github/ShinAsakawa/ShinAsakawa.github.io/blob/master/2022notebooks/2022_0112preprocessing.ipynb), 
[オリジナル  `preprocessing.ipynb` Open In Colab](https://colab.research.google.com/github/huggingface/notebooks/blob/master/transformers_doc/preprocessing.ipynb)
* **訓練済モデルの微調整 Fine-tuning a pretrained model**, Trainer を使って訓練済モデルを微調整する方法<!-- How to use the Trainer to fine-tune a pretrained model -->, [訓練済みモデルを微調整 `train.ipynb` <img src="assets/colab_icon.svg">](https://colab.research.google.com/github/ShinAsakawa/ShinAsakawa.github.io/blob/master/2022notebooks/2022_0316huggingface_tutorial_training.ipynb), [`training.ipynb` Open In Colab](https://colab.research.google.com/github/huggingface/notebooks/blob/master/transformers_doc/training.ipynb)
* **トークン化器のまとめ Summary of the tokenizers**, トークン化器のアルゴリズムの違い<!-- The differences between the tokenizers algorithm -->, [トークナイザーのまとめ <img src="assets/colab_icon.svg">](https://colab.research.google.com/github/ShinAsakawa/ShinAsakawa.github.io/blob/master/2022notebooks/2022_0316Huggingface_tutorial_tokenizer_summary.ipynb) [`tokenizer_summary.ipynb` Open In Colab](https://colab.research.google.com/github/huggingface/notebooks/blob/master/transformers_doc/tokenizer_summary.ipynb)
* **多言語モデル Multilingual models**, How to use the multilingual models of the library, [推論のための多言語モデル](https://colab.research.google.com/github/ShinAsakawa/ShinAsakawa.github.io/blob/master/2022notebooks/2022_0316Huggingface_tutorial_multilingual.ipynb#scrollTo=HxzwFeTknHi6) [`multilingual.ipynb` Open In Colab <img src="assets/colab_icon.svg">](https://colab.research.google.com/github/huggingface/notebooks/blob/master/transformers_doc/multilingual.ipynb)
* **自前のデータセットを用いた微調整 Fine-tuning with custom datasets**, How to fine-tune a pretrained model on various tasks, [一般的な下流作業のためにモデルを微調整する方法 <img src="assets/colab_icon.svg">](https://colab.research.google.com/github/ShinAsakawa/ShinAsakawa.github.io/blob/master/2022notebooks/2022_0316Huggingface_tutorial_custom_datasets.ipynb) [`custom_datasets.ipynb` Open In Colab](https://colab.research.google.com/github/huggingface/notebooks/blob/master/transformers_doc/custom_datasets.ipynb)


#### PyTorch notebooks:

* **トークン化器の訓練** 自作トークン化器を用いた訓練方法 [ゼロから独自トークン化器を訓練 <img src="assets/colab_icon.svg">](https://colab.research.google.com/github/ShinAsakawa/ShinAsakawa.github.io/blob/master/2022notebooks/2022_0316Huggingface_tutorial_Train_your_tokenizer.ipynb) , [`tokenizer_training.ipynb` Open In Colab](https://colab.research.google.com/github/huggingface/notebooks/blob/master/examples/tokenizer_training.ipynb)
* **自作言語モデルの訓練** トランスフォーマを簡単に使い始める方法 [言語モデルの訓練 train a language model <img src="assets/colab_icon.svg">](https://colab.research.google.com/github/ShinAsakawa/ShinAsakawa.github.io/blob/master/2022notebooks/2022_0112Train_a_language_model.ipynb)
[オリジナル `language_modeling_from_scratch.ipynb` Open In Colab](https://colab.research.google.com/github/ShinAsakawa/ShinAsakawa.github.io/blob/master/2022notebooks/2022_0112Train_a_language_model.ipynb)，[オリジナル](https://colab.research.google.com/github/huggingface/notebooks/blob/master/examples/language_modeling_from_scratch.ipynb)

* **文書分類課題における微調整の方法** データを前処理し，全 GLUE 課題で事前訓練済モデルを微調整する方法を紹介 [`text_classification.ipynb` Open in Colab](https://colab.research.google.com/github/huggingface/notebooks/blob/master/examples/text_classification.ipynb)
* **言語モデルの微調整方法** データを前処理し，因果関係のある，またはマスク化言語モデル課題で事前訓練済モデルを微調整する方法 [`language_modeling.ipynb` Open in Colab](https://colab.research.google.com/github/huggingface/notebooks/blob/master/examples/language_modeling.ipynb)
* **トークン分類課題の微調整方法** トークン分類タスク(NER, PoS) において、データの前処理と事前学習したモデルの微調整を行う方法 [`token_classification.ipynb` Open in Colab](https://colab.research.google.com/github/huggingface/notebooks/blob/master/examples/token_classification.ipynb)
* **Q A 課題での微調整方法** SQUADでのデータの前処理と事前学習済みモデルの微調整方法 [`multiple_choice.ipynb` Open in Colab](https://colab.research.google.com/github/huggingface/notebooks/blob/master/examples/multiple_choice.ipynb)
* **多分類課題での微調整方法** SWAGでのデータの前処理と事前学習モデルの微調整の方法を示す [`multiple_choice.ipynb` Open in Colab](https://colab.research.google.com/github/huggingface/notebooks/blob/master/examples/multiple_choice.ipynb)
* **翻訳課題での微調整方法** データの前処理とWMTでの事前学習済みモデルの微調整方法 [`translation.ipynb` Open in Colab](https://colab.research.google.com/github/huggingface/notebooks/blob/master/examples/translation.ipynb)
* **要約課題での微調整方法** データの前処理とXSUMでの事前学習済みモデルの微調整方法 [`summarization.ipynb` Open in Colab](https://colab.research.google.com/github/huggingface/notebooks/blob/master/examples/summarization.ipynb)
* **英語音声認識課題での微調整方法** データを前処理し、TIMIT 上で事前訓練された音声モデルを微調整方法 [`speech_recognition.ipynb` <img src="assets/colab_icon.svg">](https://colab.research.google.com/github/ShinAsakawa/ShinAsakawa.github.io/blob/master/2022notebooks/2022_0111Fine_Tune_Speech_Recognition_Model_with_Transformers_using_CTC.ipynb), [オリジナル](https://colab.research.google.com/github/huggingface/notebooks/blob/master/examples/speech_recognition.ipynb)
* **How to fine-tune a speech recognition model in any language** Common Voice [`multi_lingual_speech_recognition.ipynb` Open in Colab](https://colab.research.google.com/github/huggingface/notebooks/blob/master/examples/multi_lingual_speech_recognition.ipynb)上でデータを前処理し、複数言語の事前学習済み音声モデルを微調整する方法を示す
* **音声分類課題での微調整方法** データの前処理と、Keyword Spottingでの前処理済み音声モデルの微調整方法 [`audio_classification.ipynb` Open in Colab](https://colab.research.google.com/github/huggingface/notebooks/blob/master/examples/audio_classification.ipynb)
* **ゼロからの言語モデルの訓練方法** カスタムデータで Transformer モデルを効果的に訓練するためのすべてのステップをハイライト [`01_how_to_train.ipynb` Open in Colab](https://colab.research.google.com/github/huggingface/blog/blob/master/notebooks/01_how_to_train.ipynb)
* **テキスト生成方法** トランスフォーマーで言語を生成するために、さまざまな復号化方法を使用する方法 [`02_how_to_generate.ipynb` Open in Colab](https://colab.research.google.com/github/huggingface/blog/blob/master/notebooks/02_how_to_generate.ipynb)
* **モデルを ONNX へ輸出する方法** ONNXで推論ワークロードをエクスポートして実行する方法を紹介します  
* **ベンチマークの方法** How to benchmark models with transformers [`benchmark.ipynb` Open in Colab](https://colab.research.google.com/github/huggingface/notebooks/blob/master/examples/benchmark.ipynb)
* **Reformer** How Reformer pushes the limit of language modeling [`03_reformer.ipynb` Open in Colab](https://colab.research.google.com/github/patrickvonplaten/blog/blob/master/notebooks/03_reformer.ipynb)

<!-- 
* **Train your tokenizer** How to train and use your very own tokenizer  
[Open In Colab](https://colab.research.google.com/github/huggingface/notebooks/blob/master/examples/tokenizer_training.ipynb)
* **Train your language model**  How to easily start using transformers  
[Open In Colab](https://colab.research.google.com/github/huggingface/notebooks/blob/master/examples/language_modeling_from_scratch.ipynb)
* **How to fine-tune a model on text classification** Show how to preprocess the data and fine-tune a pretrained model on any GLUE task. 
[Open in Colab](https://colab.research.google.com/github/huggingface/notebooks/blob/master/examples/text_classification.ipynb)
* **How to fine-tune a model on language modeling** Show how to preprocess the data and fine-tune a pretrained model on a causal or masked LM task. 
[Open in Colab](https://colab.research.google.com/github/huggingface/notebooks/blob/master/examples/language_modeling.ipynb)
* **How to fine-tune a model on token classification** Show how to preprocess the data and fine-tune a pretrained model on a token classification task (NER, PoS). 
[Open in Colab](https://colab.research.google.com/github/huggingface/notebooks/blob/master/examples/token_classification.ipynb)
* **How to fine-tune a model on question answering**  Show how to preprocess the data and fine-tune a pretrained model on SQUAD.  
[Open in Colab](https://colab.research.google.com/github/huggingface/notebooks/blob/master/examples/multiple_choice.ipynb)
* **How to fine-tune a model on multiple choice**  Show how to preprocess the data and fine-tune a pretrained model on SWAG. 
[Open in Colab](https://colab.research.google.com/github/huggingface/notebooks/blob/master/examples/multiple_choice.ipynb)
* **How to fine-tune a model on translation** Show how to preprocess the data and fine-tune a pretrained model on WMT. 
[Open in Colab](https://colab.research.google.com/github/huggingface/notebooks/blob/master/examples/translation.ipynb)
* **How to fine-tune a model on summarization**   Show how to preprocess the data and fine-tune a pretrained model on XSUM. 
[Open in Colab](https://colab.research.google.com/github/huggingface/notebooks/blob/master/examples/summarization.ipynb)
* **How to fine-tune a speech recognition model in English**  Show how to preprocess the data and fine-tune a pretrained Speech model on TIMIT 
[Open in Colab](https://colab.research.google.com/github/huggingface/notebooks/blob/master/examples/speech_recognition.ipynb)
* **How to fine-tune a speech recognition model in any language** Show how to preprocess the data and fine-tune a multi-lingually pretrained speech model on Common Voice 
[Open in Colab](https://colab.research.google.com/github/huggingface/notebooks/blob/master/examples/multi_lingual_speech_recognition.ipynb)
* **How to fine-tune a model on audio classification**  Show how to preprocess the data and fine-tune a pretrained Speech model on Keyword Spotting 
[Open in Colab](https://colab.research.google.com/github/huggingface/notebooks/blob/master/examples/audio_classification.ipynb)
* **How to train a language model from scratch**  Highlight all the steps to effectively train Transformer model on custom data 
[Open in Colab](https://colab.research.google.com/github/huggingface/blog/blob/master/notebooks/01_how_to_train.ipynb)
* **How to generate text** How to use different decoding methods for language generation with transformers  
[Open in Colab](https://colab.research.google.com/github/huggingface/blog/blob/master/notebooks/02_how_to_generate.ipynb)
* **How to export model to ONNX**   Highlight how to export and run inference workloads through ONNX    
* **How to use Benchmarks**  How to benchmark models with transformers  
[Open in Colab](https://colab.research.google.com/github/huggingface/notebooks/blob/master/examples/benchmark.ipynb)
* **Reformer** How Reformer pushes the limits of language modeling 
[Open in Colab](https://colab.research.google.com/github/patrickvonplaten/blog/blob/master/notebooks/03_reformer.ipynb)
-->

