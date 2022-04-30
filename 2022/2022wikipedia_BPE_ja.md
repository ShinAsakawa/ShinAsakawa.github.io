---
title: BPE(Bypte Pair Encoding) from wikipedia
author: 浅川伸一
layout: default
---
## [Byte Pair Encoding (wikipedia)](https://en.wikipedia.org/wiki/Byte_pair_encoding)

バイトペアエンコーディング[1][2]またはダイグラムコーディング[3]は，データ圧縮の単純な形態で，データの連続するバイトの最も一般的な対を，そのデータ内に発生しないバイトに置き換えるものである。
元のデータを復元するためには，置換されたバイトの表が必要となる。
このアルゴリズムは 1994 年 2 月の C Users Journalの記事 A New Algorithm for Data Compression で Philip Gage が初めて公に説明した[4]。
<!-- 
Byte pair encoding[1][2] or digram coding[3] is a simple form of data compression in which the most common pair of consecutive bytes of data is replaced with a byte that does not occur within that data. 
A table of the replacements is required to rebuild the original data. 
The algorithm was first described publicly by Philip Gage in a February 1994 article "A New Algorithm for Data Compression" in the C Users Journal.[4]-->


この手法は Google の SentencePiece[5] や OpenAI の GPT-3[6] など，いくつかの自然言語処理 (NLP) アプリケーションで有用であることが示されている。
<!-- A variant of the technique has shown to be useful in several natural language processing (NLP) applications, such as Google's SentencePiece,[5] and OpenAI's GPT-3.[6] -->

### BPE の例
<!-- ### Byte pair encoding example -->

以下のデータを符号化することを考える
<!-- Suppose the data to be encoded is -->
```
aaabdaaabac
```

バイト対 "aa" は最も出現頻度が高いので，データで使われていないバイト "Z" に置き換わることになる。
これにより，次のようなデータと置換表ができる:
<!-- The byte pair "aa" occurs most often, so it will be replaced by a byte that is not used in the data, "Z". 
Now there is the following data and replacement table:-->
```
ZabdZabac
Z=aa
```

次にバイト対 "ab” を Y に置き換えて，この処理を繰り返す:
<!-- Then the process is repeated with byte pair "ab", replacing it with Y: -->

```
ZYdZYac
Y=ab
Z=aa
```

リテラルなバイト対は 1 回しか発生しないので，ここでエンコードを止めてもよい。
あるいは "ZY" を "X" に置き換えて，再帰的にバイト対符号化を続けることもできる。
<!-- The only literal byte pair left occurs only once, and the encoding might stop here. 
Or the process could continue with recursive byte pair encoding, replacing "ZY" with "X":-->

```
XdXac
X=ZY
Y=ab
Z=aa
```

このデータは 2 回以上出現するバイト対が存在しないため，バイトペアエンコーディングではこれ以上圧縮できない。
<!-- This data cannot be compressed further by byte pair encoding because there are no pairs of bytes that occur more than once.-->

このデータを伸長するには，単純に逆順に置換を行う。
<!-- To decompress the data, simply perform the replacements in the reverse order. -->


#### 文献

1. Gage, Philip (1994). [A New Algorithm for Data Compression](http://www.pennelynn.com/Documents/CUJ/HTML/94HTML/19940045.HTM). The C User Journal.
2. [A New Algorithm for Data Compression](http://www.drdobbs.com/article/print?articleId=184402829). Dr. Dobb's Journal. 1 February 1994. Retrieved 10 August 2020.
3. Witten, Ian H.; Moffat, Alistair; Bell, Timothy C. (1994). Managing Gigabytes. New York: Van Nostrand Reinhold. ISBN [978-0-442-01863-4](https://en.wikipedia.org/wiki/Special:BookSources/978-0-442-01863-4).
4. [Byte Pair Encoding](https://web.archive.org/web/20160326130908/http://www.csse.monash.edu.au/cluster/RJK/Compress/problem.html). Archived from [the original](http://www.csse.monash.edu.au/cluster/RJK/Compress/problem.html) on 2016-03-26.
5. [google/sentencepiece](https://github.com/google/sentencepiece). Google. 2021-03-02. Retrieved 2021-03-02.
6. Brown, Tom B.; Mann, Benjamin; Ryder, Nick; Subbiah, Melanie; Kaplan, Jared; Dhariwal, Prafulla; Neelakantan, Arvind; Shyam, Pranav; Sastry, Girish; Askell, Amanda; Agarwal, Sandhini (2020-06-04). "Language Models are Few-Shot Learners". arXiv:[2005.14165](https://arxiv.org/abs/2005.14165) [cs.CL].


