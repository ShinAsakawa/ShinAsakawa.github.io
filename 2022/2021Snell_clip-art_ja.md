---
source: https://ml.berkeley.edu/blog/posts/clip-art/
author: Charlie Snell
date: 2021_0630
---

# 宇宙人の夢: アートシーンの出現
<!-- # Alien Dreams: An Emerging Art Scene -->

**By Charlie Snell**

ここ数ヶ月 AI によるアートシーンはちょっとした爆発的な盛り上がりを見せています。
<!-- In recent months there has been a bit of an explosion in the AI generated art scene.-->

OpenAI が CLIP モデルの訓練済み重みとコードを公開して以来，さまざまなハッカー，アーティスト，研究者，ディープラーニング愛好家が CLIP をさまざまな生成モデルの効果的な「自然言語のハンドル」として活用する方法を考え出しました。
これにより，アーティストは，テキスト (キャプション，詩，歌詞，単語) をこれらのモデルの 1 つに入力するだけで，あらゆる種類の興味深いビジュアルアートを作成することができます。
<!-- Ever since OpenAI released the weights and code for their CLIP model, various hackers, artists, researchers, and deep learning enthusiasts have figured out how to utilize CLIP as a an effective "natural language steering wheel" for various generative models, allowing artists to create all sorts of interesting visual art merely by inputting some text -- a caption, a poem, a lyric, a word -- to one of these models. -->

例えば「夜の街並み」と入力すると，抽象的でクールな街明かりの描写が表示されます。
<!-- For instance inputting "a cityscape at night" produces this cool, abstract-looking depiction of some city lights: -->

<center>

<img src="https://i.imgur.com/eSQk5DK.png" width="66%"><br/>
(<a href="https://twitter.com/RiversHaveWings/status/1382402580379148290?s=20" target="_blank" rel="nofollow noopener noreferrer">
source</a>: <a href="https://twitter.com/RiversHaveWings" target="_blank" rel="nofollow noopener noreferrer">@
RiversHaveWings</a> on Twitter)
</center>

夕日の画像を求めると，この面白いミニマリストのようなものが返ってきたり。
<!-- Or asking for an image of the sunset returns this interesting minimalist thing: -->
<center>
<img src="https://i.imgur.com/1MtKFMt.jpg" width="66%"><br/>
(<a href="https://twitter.com/advadnoun/status/1369403672006963206?s=20" target="_blank" rel="nofollow noopener noreferrer">source</a>: <a href="https://twitter.com/advadnoun" target="_blank" rel="nofollow noopener noreferrer">@Advadnoun</a> on Twitter)
</center>

「小さなお城に支配された惑星の抽象画」をお願いしたところ，このような満足感のあるトリッピーな作品が出来上がりました。
<!-- Asking for "an abstract painting of a planet ruled by little castles" results in this satisfying and trippy piece: -->
<center>
<img src="https://i.imgur.com/wGhQW0t.png" width="66%"><br/>
(<a href="https://twitter.com/RiversHaveWings/status/1386755619164622848?s=20" target="_blank" rel="nofollow noopener noreferrer">source</a>: <a href="https://twitter.com/RiversHaveWings" target="_blank" rel="nofollow noopener noreferrer">@RiversHaveWings</a> on Twitter)
</center>

T.S.エリオットの詩 [荒地](https://www.poetryfoundation.org/poems/47311/the-waste-land) の一部をシステムに送り込むと，この崇高で落ち着きのある作品が出来上がります。
<!-- Feed the system a portion of the poem ["The Wasteland" by T.S. Eliot](https://www.poetryfoundation.org/poems/47311/the-waste-land) and you get this sublime, calming work: -->
<center>
<img src="https://i.imgur.com/tvYNz8T.png" width="66%"><br/>
(<a href="https://twitter.com/advadnoun/status/1367375268629770245" target="_blank" rel="nofollow noopener noreferrer">source</a>: <a href="https://twitter.com/advadnoun" target="_blank" rel="nofollow noopener noreferrer">@Advadnoun</a> on Twitter)
</center>

また，特定の文化的背景を提示することも可能で，その場合，ほぼ正確な結果が得られます。
「スタジオジブリの風景」と入力すると，それなりに説得力のある結果が得られます。
<!-- You can even mention specific cultural references and it'll usually come up with something sort of accurate. 
Querying the model for a "studio ghibli landscape" produces a reasonably convincing result: -->
<center>
<img src="https://i.imgur.com/bKu7Wce.jpg" width="66%"><br/>
(<a href="https://twitter.com/ak92501/status/1402134906931720195?s=20" target="_blank" rel="nofollow noopener noreferrer">source</a>: <a href="https://twitter.com/ak92501" target="_blank" rel="nofollow noopener noreferrer">@ak92501</a> on Twitter)
</center>

これと同じ方法で，ちょっとしたアニメーションを作ることもできます。
試しに「星降る夜」をリクエストしてみたら，こんな素敵な GIF が出来上がりました。
<!-- You can create little animations with this same method too. 
In my own experimentation, I tried asking for "Starry Night" and ended up with this pretty cool looking gif: -->
<center>
<img src="https://i.imgur.com/3LlMl9X.gif" width="66%">
</center>

言葉を入力すると，その言葉をシステムが精一杯，抽象的なスタイルで表現してくれるのです。
本当に楽しくて，驚きの連続です。
何が出てくるかわからない，トリッピーな擬似写実主義の風景かもしれないし，もっと抽象的でミニマルなものかもしれません。
<!-- These models have so much creative power: just input some words and the system does its best to render them in its own uncanny, abstract style. 
It's really fun and surprising to play with: I never really know what's going to come out; it might be a trippy pseudo-realistic landscape or something more abstract and minimal.-->

また，実際に画像を生成する作業の大半をモデルが行っているにもかかわらず，モデルとの作業ではクリエイティブな気分，つまりアーティストのような気分でいられます。
モデルに何を指示するかということに，クリエイティブな要素があるのです。
自然言語入力は，完全にオープンなサンドボックスであり，モデルの好みに合わせて言葉を操ることができれば，ほとんど何でも作ることができるのです。
<!-- And despite the fact that the model does most of the work in actually generating the image, I still feel creative -- I feel like an artist -- when working with these models. 
There's a real element of creativity to figuring out what to prompt the model for. 
The natural language input is a total open sandbox, and if you can weild words to the model's liking, you can create almost anything. -->

コンセプトとしては，テキストの記述から画像を生成するこのアイデアは，オープン AI の DALL-E モデルに驚くほど似ています (私の [以前のブログ](https://ml.berkeley.edu/blog/posts/vq-vae/)，[投稿](https://ml.berkeley.edu/blog/posts/dalle2/) をご覧になった方は DALL-E の技術的内情と哲学的アイデアの両方を詳細にカバーしています)。
しかし，実は，この方法は全く違います。
DALL-E は言語から直接高品質な画像を生成することだけを目的に，エンドツーエンドで訓練されていますが，この CLIP 法は，既存の無条件画像生成モデルを操作するために言語を使うための，美しくハックされたトリックのようなものなのです。
<!-- In concept, this idea of generating images from a text description is incredibly similar to Open-AI's DALL-E model (if you've seen my [previous blog](https://ml.berkeley.edu/blog/posts/vq-vae/) [posts](https://ml.berkeley.edu/blog/posts/dalle2/), I covered both the technical inner workings and philosophical ideas behind DALL-E in great detail). 
But in fact, the method here is quite different. DALL-E is trained end-to-end for the sole purpose of producing high quality images directly from language, whereas this CLIP method is more like a beautifully hacked together trick for using language to steer existing unconditional image generating models. -->

<center>
<img src="https://i.imgur.com/TmUpiHN.png" width="88%"><br/>
<div style="text-align:left;width:66%;background-color:cornsilk">

- DALL-E のエンドツーエンドのテキストから画像への変換がどのように行われるかをハイレベルに描写しています。<br/>
<!-- ## A high-level depiction of how DALL-E's end-to-end text-to-image generation works. -->
訳注: DALL-E の画像生成方法: 「アボガドの形をした座椅子」といったテキスト文を入力すると，一枚の画像をエンドツーエンドで出力するように訓練されている。
</div>
</center>

<center>
<img src="https://i.imgur.com/X1tqraa.gif" width="94%"><br/>
<div style="text-align:left;width:66%;background-color:cornsilk">

CLIP がどのようにアートを生成するのかを高いレベルで描いています。
<!-- ###### A high level depiction of how CLIP can be used to generate art. -->
</div>
</center>

DALL-E のウェイトはまだ公開されていませんので，この CLIP 作品は DALL-E の約束事を再現するためのハッカー的な試みと見ることができます。
<!-- The weights for DALL-E haven't even been publicly released yet, so you can see this CLIP work as somewhat of a hacker's attempt at reproducing the promise of DALL-E.-->

CLIP を使ったアプローチは，もう少しハチャメチャなので，DALL-E で実証されたような高品質で正確な出力は得られません。
その代わり，これらのシステムから生み出される映像は，奇妙で，トリッピーで，抽象的です。
出力は確かに私たちの世界に根ざしているのですが，まるでちょっと違うものを見ている宇宙人が作ったような感じなのです。
<!-- Since the CLIP based approach is a little more hacky, the outputs are not quite as high quality and precise as what's been demonstrated with DALL-E. 
Instead, the images produced by these systems are weird, trippy, and abstract. The outputs are grounded in our world for sure, but it's like they were produced by an alien that sees things a little bit differently. -->

その奇妙さこそが CLIP を使った作品を独特の芸術的な美しさにしているのだと思います。
見慣れたものを異星人の視点で見るというのは，何か特別な感じがします。
<!-- It's exactly the weirdness that makes these CLIP based works so uniquely artistic and beautiful to me. 
There's something special about seeing an alien perspective on something familiar. -->

**(注: 厳密には DALL-E は CLIP を使って出力を再ランク付けしていますが，ここで CLIP を使った手法と言った場合 DALL-E のことではありません)**
<!-- *(Note: technically DALL-E makes use of CLIP to re-rank its outputs, but when I say CLIP based methods here, I'm not talking about DALL-E.)* -->

<center>
<img src="https://i.imgur.com/WNLOg1R.jpg" width="88%">
</center>

ここ数ヶ月，私の Twitter のタイムラインは，この CLIP で生成されたアートで占拠されています。
アーティスト，研究者，ハッカーのコミュニティは，これらのモデルで実験し，その成果を共有しています。
また，生成された画像の品質や芸術的なスタイルを変更するためのコードやさまざまなトリック/方法も共有されています。
これは，まるで新興のアートシーンのようです。
<!-- Over the last few months, my Twitter timeline has been taken over by this CLIP generated art. A growing community of artists, researchers, and hackers have been experimenting with these models and sharing their outputs. People have also been sharing code and various tricks/methods for modifying the quality or artistic style of the images produced. 
It all feels a bit like an emerging art scene. -->

このアートシーンが 1 年の間に発展し，進化していくのを見るのはとても楽しかったのです。
私にとってはとてもクールなことなので，ブログ記事を書こうと思いました。
<!-- I've had a lot of fun watching as this art scene has developed and evolved over the course of the year, so I figured I'd write a blog post about it because it's just so cool to me.-->

このシステムがどのようにアートを生成するのか，技術的な詳細について深く説明するつもりはありません。
その代わり，このアートシーンの思いがけない起源と進化を記録し，その過程で私自身の考えやクールなアートワークも紹介するつもりです。
<!-- I'm not going to go in-depth on the technical details of how this system generates art. 
Instead, I'm going to document the unexpected origins and evolution of this art scene, and along the way I'll also present some of my own thoughts and some cool artwork.-->

もちろん，一回のブログでこのアートシーンのあらゆる側面をカバーすることはできません。
もし，私が見逃しているかもしれない重要なことがあれば，下のコメント欄や [ツィート](https://twitter.com/sea_snell) でお気軽にお知らせください。
<!-- Of course I am not able to cover every aspect of this art scene in a single blog post. 
But I think this blog hits most of the big points and big ideas, and if there's anything important that you think I might have missed, feel free to comment below or [tweet at me](https://twitter.com/sea_snell). -->

## CLIP: 思わぬ起源の物語
<!-- ## CLIP: An Unexpected Origin Story-->

2021 年 1 月 5 日，OpenAI は [CLIP](https://openai.com/blog/clip/) のモデル重みとコードを公開しました。
このモデルは，与えられた画像に最も合う脚注を脚注の集合の中から決めるために学習させたモデルです。
この方法で何億もの画像から学習した後 CLIP は与えられた画像に最適な脚注を選ぶことに非常に熟達しただけでなく，視覚に関する驚くほど抽象的で一般的な表現も学習しました ([Goh et al. on Distill](https://distill.pub/2021/multimodal-neurons/) のマルチモーダルニューロンを参照)。
<!-- On January 5th 2021, OpenAI released the model-weights and code for [CLIP](https://openai.com/blog/clip/): a model trained to determine which caption from a set of captions best fits with a given image. 
After learning from hundreds of millions of images in this way, CLIP not only became quite proficient at picking out the best caption for a given image, but it also learned some surprisingly abstract and general representations for vision (see [multimodal neuron work from Goh et al. on Distill](https://distill.pub/2021/multimodal-neurons/)). -->

例えば CLIP はスパイダーマンに関する画像や概念に特異的に活性化するニューロンを表現するように学習しました。
他にも，感情や地理的な場所，あるいは有名な人物に関連するイメージに対して活性化するニューロンもある (これらのニューロンの活性化は [OpenAIの顕微鏡ツール](https://microscope.openai.com/models) で自分で調べることができます)。
<!-- For instance, CLIP learned to represent a neuron that activates specifically for images and concepts relating to Spider-Man. 
There are also other neurons that activate for images relating to emotions, geographic locations, or even famous individuals (you can explore these neuron activations yourself with [OpenAI's microscope tool](https://microscope.openai.com/models).-->

このような抽象度の高い画像表現は，この種のものとしては，何やら初めてのものでした。
さらに，このモデルは，これまでのどの研究よりも優れた分類の頑健性も実証していました。
<!-- Image representations at this level of abstraction were somewhat of a first of their kind. 
And in addition to all of this, the model also demonstrated a greater classification robustness than any prior work. -->

ですから，研究の観点からは CLIP は非常にエキサイティングで強力なモデルでした。
しかし，それがアートを生み出すのに役立つということを明確に示唆するものはここにはありません。
<!-- So from a research perspective, CLIP was an incredibly exciting and powerful model. 
But nothing here clearly suggests that it would be helpful with generating art -- let alone spawning the art scene that it did.-->

それでも，様々なハッカー，研究者，アーティスト (特に Twitter の [@advadnoun](https://twitter.com/advadnoun) と [@quasimondo](https://twitter.com/quasimondo))) が CLIP を使えば既存の画像生成モデル (
GAN, 自己符号化器,  SIREN のような 非明示的なニューラルネットワークによる画像表象 (Implicit Neural Representation) が与えられた脚注に合ったオリジナルの画像を生成できることを理解するのにほんの 1 日しかかからなかったのです。
<!-- Nonetheless, it only took a day for various hackers, researchers, and artists (most notably [@advadnoun](https://twitter.com/advadnoun) and [@quasimondo](https://twitter.com/quasimondo) on Twitter) to figure out that with a simple trick CLIP can actually be used to guide existing image generating models (like GANs, Autoencoders, or Implicit Neural Representations like SIREN) to produce original images that fit with a given caption. -->

この方法では CLIP は生成モデルに対する「自然言語のハンドル」のような役割を果たします。
CLIP は基本的に，与えられた生成モデルの潜在空間を検索して，与えられた語句の列に合う画像に対応する潜在を見つけるようガイドします。
<!-- In this method, CLIP acts as something like a "natural language steering wheel" for generative models. 
CLIP essentially guides a search through the latent space of a given generative model to find latents that map to images which fit with a given sequence of words. -->

この技術を使った初期の結果は奇妙なものでしたが，それでも驚きと期待が持てました。
<!-- Early results using this technique were weird but nonetheless surprising and promising: -->

<!-- <center>
<img src="https://i.imgur.com/szNPfXI.jpg" width="66%">
</center>

![](data:image/svg+xml;base64,PHN2ZyBhcmlhLWhpZGRlbj0idHJ1ZSIgZm9jdXNhYmxlPSJmYWxzZSIgaGVpZ2h0PSIxNiIgdmVyc2lvbj0iMS4xIiB2aWV3Ym94PSIwIDAgMTYgMTYiIHdpZHRoPSIxNiI+PHBhdGggZmlsbC1ydWxlPSJldmVub2RkIiBkPSJNNCA5aDF2MUg0Yy0xLjUgMC0zLTEuNjktMy0zLjVTMi41NSAzIDQgM2g0YzEuNDUgMCAzIDEuNjkgMyAzLjUgMCAxLjQxLS45MSAyLjcyLTIgMy4yNVY4LjU5Yy41OC0uNDUgMS0xLjI3IDEtMi4wOUMxMCA1LjIyIDguOTggNCA4IDRINGMtLjk4IDAtMiAxLjIyLTIgMi41UzMgOSA0IDl6bTktM2gtMXYxaDFjMSAwIDIgMS4yMiAyIDIuNVMxMy45OCAxMiAxMyAxMkg5Yy0uOTggMC0yLTEuMjItMi0yLjUgMC0uODMuNDItMS42NCAxLTIuMDlWNi4yNWMtMS4wOS41My0yIDEuODQtMiAzLjI1QzYgMTEuMzEgNy41NSAxMyA5IDEzaDRjMS40NSAwIDMtMS42OSAzLTMuNVMxNC41IDYgMTMgNnoiPjwvcGF0aD48L3N2Zz4=)
left -- ([source](https://twitter.com/quasimondo/status/1347956102898606081?s=20): [@quasimondo](https://twitter.com/quasimondo) on Twitter); right -- ([source](https://twitter.com/advadnoun/status/1346767585266679808?s=20): [@advadnoun](https://twitter.com/advadnoun) on Twitter) {#left--source-quasimondo-on-twitter-right--source-advadnoun-on-twitter} -->

<center>
<img src="https://i.imgur.com/szNPfXI.jpg" width="66%"><br/>
(<a href="http://twitter.com/quasimondo/status/1347956102898606081?s=20" target="_blank" rel="nofollow noopener noreferrer"> source</a>: <a href="https://twitter.com/quasimondo" target="_blank" rel="nofollow noopener noreferrer">@quasimondo</a> on Twitter); right – (<a href="https://twitter.com/advadnoun/status/1346767585266679808?s=20" targe="_blank" rel="nofollow noopener noreferrer">source</a>: <a href="https://twitter.com/advadnoun" target="_blank" rel="nofollow noopener noreferrer">@advadnoun</a> on Twitter)
</center>

## ビッグスリープ 謙虚な始まり
<!-- ## The Big Sleep: Humble Beginnings-->

ほんの 2 週間ほどで，画期的なことが起こりました。
ビッグスリープのコードは [Big GAN](https://arxiv.org/abs/1809.11096) を生成モデルとして使用した CLIP ベースのテキストから画像への変換技術として公開されました。
<!-- In just a couple weeks, there was a breakthrough.
[@advadnoun](https://twitter.com/advadnoun) released code for The Big Sleep: a CLIP based text-to-image technique, which used [Big GAN](https://arxiv.org/abs/1809.11096) as the generative model. -->

<center>
<img src="https://i.imgur.com/IJjWOIH.jpg" width="66%"><br/>
(<a href="https://twitter.com/advadnoun/status/1351038053033406468" target="_blank" rel="nofollow noopener noreferrer">source</a>: <a href="https://twitter.com/advadnoun" target="_blank" rel="nofollow noopener noreferrer">@advadnoun</a> on Twitter)
</center>

ビッグスリープは，独自の方法で，テキストから画像への変換の期待に応えました。
言葉にできるものなら，何でもおおよそレンダリングできる。
「夕焼け」「M.C.エッシャーの絵のような顔」「風が吹くとき」「3D のグランドキャニオン」。
<!-- In its own unique way, the Big Sleep roughly met the promise of text-to-image. 
It can approximately render just about anything you can put into words: "a sunset", "a face like an M.C. Escher drawing", "when the wind blows", "the grand canyon in 3d".-->

もちろん「ビッグスリープ」からの出力は，誰もが好むものではないかもしれません。
奇妙で抽象的で，通常はグローバルに首尾一貫しているのだが，時にはあまり意味をなさないこともあります。
ビッグスリープの作品には，確かに独特のスタイルがあり，個人的には美的感覚に優れていると感じています。
<!-- Of course, the outputs from The Big Sleep are maybe not everyone's cup of tea. They're weird and abstract, and while they are usually globally coherent, sometimes they don't make much sense. 
There is definitely a unique style to artworks produced by The Big Sleep, and I personally find it to be aesthetically pleasing. -->

<center>
<img src="https://i.imgur.com/ZTUsHOk.png" width="66%"><br/>
「日没」ビッグスリープによる “a sunset” according to The Big Sleep <a href="https://twitter.com/advadnoun/status/1378226110995984386?s=20" target="_blank" rel="nofollow noopener 
noreferrer">source</a>: <a href="https://twitter.com/advadnoun" target="_blank" rel="nofollow noopener noreferrer">@advadnoun</a> on Twitter)
</center>

<center>
<img src="https://i.imgur.com/EF0IlTh.png" width="66%"><br/>

「エッシャーが描いたような顔」ビッグスリープ "a face like an M.C. Escher drawing" from The Big Sleep ([source](https://twitter.com/advadnoun/status/1359723192890269696): [@advadnoun](https://twitter.com/advadnoun) on Twitter) 
</center>

<center>
<img src="https://i.imgur.com/Un5tSbt.png" width="66%"><br/>

「風が吹くとき」ビッグスリープ “when the wind blows” from ThBig Sleep (<a href="https://twitter.com/advadnoun/status/1361205540970319875?s=20" target="_blank" rel="nofollow noopener noreferrer">source</a>: <a href="https://twitter.com/advadnoun" target="_blank" rel="nofollow noopener noreferrer">@advadnoun</a> on Twitter)
</center>

しかし，私が「ビッグスリープ」から受ける主な驚きと魅力は，必ずしもその美学から来るものではなく，むしろもう少しメタ的なものです。
ビッグスリープが画像を生成する際の最適化の目的は GAN 潜在空間において CLIP の下で与えられた単語の列に最大に対応する点を見つけることです。
つまり ビッグスリープの出力を見ると，文字通り CLIP が言葉をどう解釈し，それが私たちの視覚世界にどう対応していると「考えて」いるかがわかります。
<!-- But the main wonder and enchantment that I get from The Big Sleep does not necessarily come from its aesthetics, rather it's a bit more meta. 
The Big Sleep's optimization objective when generating images is to find a point in GAN latent space that maximally corresponds to a given sequence of words under CLIP. 
So when looking at outputs from The Big Sleep, we are literally seeing how CLIP interprets words and how it "thinks" they correspond to our visual world. -->

これを本当に理解するためには CLIP を統計的なものと考えるか，あるいは宇宙人のようなものと考えるかです。
私は後者の方が好きです。
CLIP は「ビッグスリープ」のような技術を使って，私たちが鍵を開けて覗き込むことができる宇宙人の脳のようなものだと考えたいのです。
ニューラルネットワークは人間の脳とは大きく異なるので CLIP を宇宙人の脳のようなものと考えても，実はそれほどおかしくはありません。
もちろん CLIP が本当に「知的」なわけでありませんが，それでも「別の」ものの見方を見せてくれているわけで，その考え方はとても魅力的だと思います。
<!-- To really appreciate this, you can think of CLIP as being either statistical or alien. 
I prefer the latter. I like to think of CLIP as something like an alien brain that we're able to unlock and peer into with the help of techniques like The Big Sleep. 
Neural networks are very different from human brains, so thinking of CLIP as some kind of alien brain is not actually that crazy. 
Of course CLIP is not truly "intelligent", bit it's still showing us a *different* view of things, and I find that idea quite enchanting.-->

CLIP の別の視点・哲学は，もう少し統計的で冷徹なものです。
CLIP の出力は，インターネット上に存在する言語と視覚の相関関係を計算した結果であり，単なる統計的平均値の産物であると考えることができます。
このように考えると CLIP の出力は，時代の流れ  (少なくとも CLIP の学習データがかき集められた時点の時代の流れ) を覗き見して「インターネットの統計的平均」のようなものを見ているようなものです (もちろん，これはデータの真の分布に対する近似誤差が最小であることを前提としており，おそらく無理な仮定です)。
<!-- The alternative perspective/philosophy on CLIP is a little more statistical and cold. You could think of CLIP's outputs as the product of mere statistical averages: the result of computing the correlations between language and vision as they exist on the internet. And so with this perspective, the outputs from CLIP are more akin to peering into the zeitgeist (at least the zeitgeist at the time that CLIP's training data was scraped) and seeing things as something like a "statistical average of the internet" (of course this assumes minimal approximation error with respect to the true distribution of data, which is probably an unreasonable assumption). -->

CLIP の出力はとても奇妙なので，宇宙人の視点の方がずっと理にかなっていると思います。
統計的ザイジストの視点は GPT-3 のように近似誤差がかなり小さいと推測される場合に適用されるのでしょう。
<!-- Since CLIP's outputs are so weird, the alien viewpoint makes a lot more sense to me. I think the statistical zeigeist perspective applies more to situations like GPT-3, where the approximation error is presumably quite low. -->

<center>
<img src="https://i.imgur.com/WYkcxdF.png" width="66%"><br/>

「すべての終わり，崩れ落ちる建物と空を貫く武器と」ビッグスリープより “At the end of everything, crumbling buildings and a weapon to pierce the sky” fr The Big Sleep 
(<a href="https://twitter.com/advadnoun/status/1358304614337052673?s=20" target="_blank" rel="nofollow noopener noreferrer">source</a>: 
<a href="https://twitter.com/advadnoun" target="_blank" rel="nofollow noopener noreferrer">@advadnoun</a> on Twitter)
</center>


<video style="width:66%" src='https://i.imgur.com/SYNaqDl.mp4' controls ></video>
<br/>

3 次元グランドキャニオン "the grand canyon in 3d" according to The Big Sleep


<!-- <center> ![](https://i.imgur.com/SYNaqDl.mp4)<br/> </center> -->

振り返ってみるとビッグスリープは，ニューラルネットワークの「心」を覗き込むような不思議な感覚を捉えた最初の AI アート手法ではありませんが，その感覚は，これまでのどの手法よりも間違いなくよく捉えています。
<!-- Looking back, The Big Sleep is not the first AI art technique to capture this magical feeling of peering into the "mind" of a neural network, but it does capture that feeling arguably better than any technique that has come before.-->

だからといって，旧来の AI アートの手法が無関係であるとか，面白みがないと言っているわけではありません。
実際 ビッグスリープは，ある意味で過ぎ去った時代の最も人気のあるニューラルネットワークのアートテクニックの 1 つから影響を受けているようです。ディープドリーム(DeepDream) です。
<!-- That's not to say that older AI art techniques are irrelevant or uninteresting. In fact, it seems that The Big Sleep was in some ways influenced by one of the most popular neural network art techniques from a foregone era: DeepDream. -->

ビッグスリープの作者である [@advadnoun](https://twitter.com/advadnoun) によれば: 
<!-- Per [@advadnoun](https://twitter.com/advadnoun) (The Big Sleep's creator):  -->

> ビッグスリープの名前は ディープドリームとシュールレアリスムのフィルムノワール The Big Sleep からの引用です。2 つ目の言及は，その奇妙で夢のような質感によるものです。([引用元](https://rynmurdock.github.io/2021/02/26/Aleph2Image.html)).

<!-- > The Big Sleep's name is "an allusion to DeepDream and the surrealist film noir, The Big Sleep. 
The second reference is due to its strange, dreamlike quality"  ([source](https://rynmurdock.github.io/2021/02/26/Aleph2Image.html).-->

今思えば ディープドリーム (DeepDream) と ビッグスリープ (The Big Sleep) は精神的なつながりがあるので ディープドリームに因んでビッグスリープと名付けたのは面白いですね。
<!-- It's interesting that [@advadnoun](https://twitter.com/advadnoun) partly named The Big Sleep after DeepDream because looking back now, they are spiritually sort of related. -->

[DeepDream](https://ai.googleblog.com/2015/07/deepdream-code-example-for-visualizing.html) は，一世代前 (2015年) に非常に人気のあった AI アートの手法でした。
この技法は基本的に，画像を取り込み，その画像が画像を分類するために訓練されたニューラルネットワークの特定のニューロンを最大限に活性化するように，わずかに (あるいは劇的に) 修正するものです。
その結果は通常，下の画像のように非常にサイケデリックでトリッピーなものになります。
<!-- [DeepDream](https://ai.googleblog.com/2015/07/deepdream-code-example-for-visualizing.html) was an incredibly popular AI art technique from a previous generation (2015). 
The technique essentially takes in an image and modifies it slightly (or dramatically) such that the image maximally activates certain neurons in a neural network  trained to classify images. 
The results are usually very psychedelic and trippy, like the image below. -->

<img src="https://i.imgur.com/fLgqd2a.jpg" width="99%"><br/>
ディープドリームによって生成された画像例 ([source](https://kaptein.me/blog/the-beauty-of-computer-dreams/)).
<!-- ![](data:image/svg+xml;base64,PHN2ZyBhcmlhLWhpZGRlbj0idHJ1ZSIgZm9jdXNhYmxlPSJmYWxzZSIgaGVpZ2h0PSIxNiIgdmVyc2lvbj0iMS4xIiB2aWV3Ym94PSIwIDAgMTYgMTYiIHdpZHRoPSIxNiI+PHBhdGggZmlsbC1ydWxlPSJldmVub2RkIiBkPSJNNCA5aDF2MUg0Yy0xLjUgMC0zLTEuNjktMy0zLjVTMi41NSAzIDQgM2g0YzEuNDUgMCAzIDEuNjkgMyAzLjUgMCAxLjQxLS45MSAyLjcyLTIgMy4yNVY4LjU5Yy41OC0uNDUgMS0xLjI3IDEtMi4wOUMxMCA1LjIyIDguOTggNCA4IDRINGMtLjk4IDAtMiAxLjIyLTIgMi41UzMgOSA0IDl6bTktM2gtMXYxaDFjMSAwIDIgMS4yMiAyIDIuNVMxMy45OCAxMiAxMyAxMkg5Yy0uOTggMC0yLTEuMjItMi0yLjUgMC0uODMuNDItMS42NCAxLTIuMDlWNi4yNWMtMS4wOS41My0yIDEuODQtMiAzLjI1QzYgMTEuMzEgNy41NSAxMyA5IDEzaDRjMS40NSAwIDMtMS42OSAzLTMuNVMxNC41IDYgMTMgNnoiPjwvcGF0aD48L3N2Zz4=) -->


<!-- ###### ![](data:image/svg+xml;base64,PHN2ZyBhcmlhLWhpZGRlbj0idHJ1ZSIgZm9jdXNhYmxlPSJmYWxzZSIgaGVpZ2h0PSIxNiIgdmVyc2lvbj0iMS4xIiB2aWV3Ym94PSIwIDAgMTYgMTYiIHdpZHRoPSIxNiI+PHBhdGggZmlsbC1ydWxlPSJldmVub2RkIiBkPSJNNCA5aDF2MUg0Yy0xLjUgMC0zLTEuNjktMy0zLjVTMi41NSAzIDQgM2g0YzEuNDUgMCAzIDEuNjkgMyAzLjUgMCAxLjQxLS45MSAyLjcyLTIgMy4yNVY4LjU5Yy41OC0uNDUgMS0xLjI3IDEtMi4wOUMxMCA1LjIyIDguOTggNCA4IDRINGMtLjk4IDAtMiAxLjIyLTIgMi41UzMgOSA0IDl6bTktM2gtMXYxaDFjMSAwIDIgMS4yMiAyIDIuNVMxMy45OCAxMiAxMyAxMkg5Yy0uOTggMC0yLTEuMjItMi0yLjUgMC0uODMuNDItMS42NCAxLTIuMDlWNi4yNWMtMS4wOS41My0yIDEuODQtMiAzLjI1QzYgMTEuMzEgNy41NSAxMyA5IDEzaDRjMS40NSAwIDMtMS42OSAzLTMuNVMxNC41IDYgMTMgNnoiPjwvcGF0aD48L3N2Zz4=)
(#an-image-produced-by-deepdream-source)an image produced by DeepDream ([source](https://kaptein.me/blog/the-beauty-of-computer-dreams/)). {#an-image-produced-by-deepdream-source} -->

ディープドリームとビッグスリープは美学的に全く異なりますが，これらの技術はどちらも同じようなビジョンを持っています: どちらも，必ずしも芸術を生み出すことを意図していないニューラルネットワークから芸術を引き出すことを目的としています。
両者ともネットワークの内部に潜り込み，美しい画像を引き出します。
これらのアート技術は，ディープラーニングの解釈可能性ツールが，途中で偶然にアートを生成したように感じられます。
<!-- Although aesthetically DeepDream is quite different from The Big Sleep, both of these techniques share a similar vision: they both aim to extract art from neural networks that were not necessarily meant to generate art. 
They dive inside the network and pull out beautiful images. 
These art techniques feel like deep learning interpretability tools that accidentally produced art along the way.-->

ですから，ある意味ビッグスリープはディープドリームの続編のようなものです。
しかし，この場合，続編は間違いなくオリジナルよりも優れています。
ディープドリームが生成する宇宙観は，それ自体が時代を超えたものですが，CLIP の知識を自然言語で探っていけるというのは，本当にパワフルなことだと思います。
言葉にすれば何でも，エイリアンドリームのようなレンズを通して表現してくれます。
これは，とても魅力的な作品づくりです。
<!-- So in a way, The Big Sleep is sort of like a sequel to DeepDream. 
But in this case the sequel is arguably better than the original. 
The alien views generated by DeepDream will always be timeless in their own respect, but there's something really powerful about being able to probe CLIP's knowledge by prompting it with natural language. 
Anything you can put into words will be rendered through this alien dream-like lense. 
It's just such an enchanting way to make art. -->

## VQ-GAN: 新しい生成のスーパーパワー
<!-- ## VQ-GAN: New Generative Superpowers-->

2020年12月17日，ハイデルベルク大学の研究者 Esserらが 論文 [高解像度な画像生成を行うトランスフォーマーを使いこなす (Taming Transformers for High-Resolution Image Synthesis)](https://arxiv.org/pdf/2012.09841.pdf) を Arxiv に投稿しました。
彼らは VQ-GAN と呼ばれる新しい GAN アーキテクチャを発表しました。
これは畳み込みネットワークの局所的な推論バイアスとトランスフォーマーの大域的な注意の両方を最適に利用する方法で，畳み込みニューラルネットワークとトランスフォーマーを組み合わせ，特に強い生成モデルを実現するものです。
<!-- On December 17 2020, researchers (Esser et al.) from Heidelberg University, posted their paper ["Taming Transformers for High-Resolution Image Synthesis"](https://arxiv.org/pdf/2012.09841.pdf) on Arxiv. 
They presented a novel GAN architecture called VQ-GAN which combines conv-nets with transformers in a way that optimally takes advantage of both the local inductive biases of conv-nets and the global attention in transformers, making for a particularly strong generative model. -->

<!-- Around early April [@advadnoun](https://twitter.com/advadnoun) and [@RiversHaveWings](https://twitter.com/RiversHaveWings) started doing some 4月上旬頃から、[@advadnoun](https://twitter.com/advadnoun)と[@RiversHaveWings](https://twitter.com/RiversHaveWings) とが VQ-GAN と CLIP を使ってテキストから画像を生成する実験を始めています。
高いレベルでは，彼らが使った方法はビッグ・スリープとほとんど同じです。
主な違いは，生成モデルとして Big-GAN を使う代わりに，このシステムは VQ-GAN を使ったということだけです。
experiments combining VQ-GAN and CLIP to generate images from a text prompt. 
On a high level, the method they used is mostly identical to The Big Sleep. 
The main difference is really just that instead of using Big-GAN as the generative model, this system used VQ-GAN. -->

その結果、スタイルが大きく変化しました。
<!-- The results were a huge stylistic shift: -->

<img src="https://i.imgur.com/UVWMu6B.jpg" width="88%"><br/>
「チューブのつながり」"A Series Of Tubes" from VQ-GAN+CLIP ([出典](https://twitter.com/RiversHaveWings/status/1384486778447568899): [@RiversHaveWings](https://twitter.com/RiversHaveWings) on Twitter)

<img src="https://i.imgur.com/DlhpIVZ.jpg" width="88%"><br/>
「窓ガラスに銃口をこすりつける黄色い煙」"The Yellow Smoke That Rubs Its Muzzle On The Window-Panes"<br/> from VQ-GAN+CLIP ([出典](https://twitter.com/RiversHaveWings/status/1386103970934886403?s=20): [@RiversHaveWings](https://twitter.com/RiversHaveWings) on Twitter)

<img src="https://i.imgur.com/aFRcs4S.jpg" width="88%"><br/>
「遊星都市 C」"Planetary City C" from VQ-GAN+CLIP ([出典](https://twitter.com/RiversHaveWings/status/1386456030217785349?s=20): [@RiversHaveWings](https://twitter.com/RiversHaveWings) on Twitter)

<img src="https://i.imgur.com/z4l9QJV.jpg" width="88%"><br/>
「月光舞踏」"Dancing in the moonlight" from VQ-GAN+CLIP ([出典](https://twitter.com/advadnoun/status/1376103552959934464?s=20): [@advadnoun](https://twitter.com/advadnoun) on Twitter)

<img src="https://i.imgur.com/1wIrlLG.jpg" width="88%"><br/>
「メカニック願望」"Mechanic Desire" from VQ-GAN+CLIP ([出典](https://twitter.com/RiversHaveWings/status/1388266038031233027?s=20): [@RiversHaveWings](https://twitter.com/RiversHaveWings) on Twitter)

<img src="https://i.imgur.com/MxlXnID.jpg" width="88%"><br/>
「メカニック願望」"Mechanic Desire" from VQ-GAN+CLIP ([source](https://twitter.com/RiversHaveWings/status/1388266038031233027?s=20): [@RiversHaveWings](https://twitter.com/RiversHaveWings) on Twitter)

<img src="https://i.imgur.com/W64VJlk.jpg" width="88%"><br/>
「武器化した木」"a tree with weaping branches" from VQ-GAN+CLIP ([source](https://twitter.com/advadnoun/status/1399896134420615170?s=20): [@advadnoun](https://twitter.com/advadnoun) on Twitter)

VQ-GAN+CLIP の出力はビッグスリープよりもペイントが少なく，彫刻のように見える傾向があります。
抽象的で現実的でないイメージであっても，そこに写っているものが手仕事で作られたものであるかのような質感があります。
ニューラルネットワークを覗き込んで，その視点で物事を見ているようなオーラは，この作品でも健在です。
<!-- The outputs from VQ-GAN+CLIP tend to look less painted than The Big Sleep and more like a sculpture. 
Even when the images are too abstract to be real, there's a certain material quality to them that makes it seem as if the objects in the images could have been crafted by hand. 
At the same time, there's still an alien weirdness to it all, and the aura of peering into a neural network and seeing things from its viewpoint is most definitely not lost here.-->

生成モデルを Big-GAN から VQ-GAN に置き換えるだけで，独自のスタイルと視点を持った新しいアーティストを得たようなもので，CLIP の目を通して世界を見るための新しいレンズを手に入れたようなものです。
このことは，CLIP ベースのシステムの汎用性を浮き彫りにしています。
新しい潜在生成モデルが出たときに，それを CLIP に差し込めば，突然新しいスタイルやフォルムのアートを生成することができるのです。
DALL-E の dVAE 重みが公開されてから 8 時間も経たないうちに [@advadnoun](https://twitter.com/advadnoun) が dVAE+CLIP で作ったアートをツイートしています。
<!-- Just swapping out the generative model from Big-GAN to VQ-GAN was almost like gaining a whole new artist with their own unique style and viewpoint: a new lens for seeing the world through CLIP's eyes. 
This highlights the generality of this CLIP based system. 
Anytime a new latent-generative model is released, it can usually be plugged into CLIP without too much trouble, and then suddenly we can generate art with a new style and form. 
In fact, this has already happened at least once: less than 8 hours after DALL-E's dVAE weights were publically released, [@advadnoun](https://twitter.com/advadnoun) was already Tweeting out art made with dVAE+CLIP. -->

## プロンプト・プログラミングの楽しみ。アンリアルエンジンの魔法 Unreal Engine Trick
<!-- ## The Joys of Prompt Programming: The Unreal Engine Trick-->

生成モデルを切り替えることで CLIP の出力のスタイルをそれほど苦労せずに劇的に変更できることを見てきましたが，これを行うにはもっと簡単なトリックがあることがわかりました。
<!-- We've seen how switching generative models can dramatically modify the style of CLIP's outputs without too much effort, but it turns out that there's an even simpler trick for doing this. -->

プロンプトに，希望する画像のスタイルについて何かを示す特定のキーワードを追加するだけで CLIP はその出力を「理解」してそれに応じて変更するために最善を尽くします。
たとえばプロンプトに「マインクラフト風」とか「マンガ風」とか「ディープドリーム風」とかを追加すれば，ほとんどの場合 CLIP は実際にそのスタイルにほぼ一致するものを出力してくれます。
<!-- All you need to do is add some specific key-words to your prompt that indicate something about the style of your desired image and CLIP will do its best to "understand" and modify its output accordingly. 
For example you could append "in the style of Minecraft" or "in the style of a Cartoon" or even "in the style of DeepDream" to your prompt and most of the time CLIP will actually output something that roughly matches the style described. -->

実際，ある特定のプロンプトの魔法がかなり人気を集めています。
それは「アンリアルエンジンの魔法 unreal engine trick」として知られるようになりました。
<!--In fact, one specific prompting trick has gained quite a bit of traction. 
It has become known as the "unreal engine trick". -->

<img src="https://i.imgur.com/AefuKTa.jpg" style="width:88%;padding-top:10pt"><br/>

[@arankomatsuzaki](https://twitter.com/arankomatsuzaki) on Twitter)
<div style="background-color:cornsilk;text-align:left;width:88%">

訳注: スクリーンショット中のツィートには以下のように書かれている:<br/>
VQGAN+CLIP を使って画像を生成するときプロンプトに「アンリアルエンジン」と入力するだけで劇的に画質が改善する。<br/>
これは「非現実エンジンの魔法」と呼ばれるようになった(笑)<br/>
例: 『空気の天使。アンリアルエンジン "the angel of air.unreal engine"』
</div>

ほんの数週間前に，ツィッターアカウント @jbustter [EleutherAI](https://www.eleuther.ai) の Discord が発見したことですが，
「アンリアルエンジンで塗りつぶせ "rendered in unreaedl engine"」とプロンプトに入力するだけで出力画像はより写実的になります。
<!-- It was discovered by @jbustter in [EleutherAI](https://www.eleuther.ai)'s Discord just a few weeks ago that if you add "rendered in unreal engine" to your prompt, the outputs look much more realistic.-->

<center>
<img src="https://i.imgur.com/kq3tuOj.png" style="width:49%;padding-top:5pt"><br/>

(出典: the #art channel in [EleutherAI](https://www.eleuther.ai)'s Discord)
<div style="background-color:cornsilk;text-align:left;width:66%">

訳注: スクリーンショット中のツィートには以下のように書かれている: 
"3d render" がうまくいくなら，非現実エンジン "unreal engine" ってヤバイ
</div>
</center>

[アンリアルエンジン Unreal Engine](https://www.unrealengine.com/en-US/) とは Epic Games 社制の人気の 3D ビデオゲームエンジンです。
CLIP は (訳注:訓練中に) 「Unreal Engine でレンダリング」という脚注が付いたビデオゲームの画像をたくさん目にしたことでしょう。
そこで，このプロンプトを追加することで，これらのアンリアルエンジン Unreal Engine の画像の外観を再現するよう，モデルに効果的に働きかけているのです。
<!-- [Unreal Engine](https://www.unrealengine.com/en-US/) is a popular 3D video game engine created by Epic Games. 
CLIP likely saw lots of images from video games that were tagged with the caption "rendered in Unreal Engine". 
So by adding this to our prompt, we're effectively incentivizing the model to replicate the look of those Unreal Engine images.-->

これは非常に効果的です。下の例を観てください。
<!-- And it works pretty well, just look at some of these examples: -->


![](https://i.giphy.com/media/XksXQ5KbCc9px1i2oD/giphy.webp)

[![](data:image/svg+xml;base64,PHN2ZyBhcmlhLWhpZGRlbj0idHJ1ZSIgZm9jdXNhYmxlPSJmYWxzZSIgaGVpZ2h0PSIxNiIgdmVyc2lvbj0iMS4xIiB2aWV3Ym94PSIwIDAgMTYgMTYiIHdpZHRoPSIxNiI+PHBhdGggZmlsbC1ydWxlPSJldmVub2RkIiBkPSJNNCA5aDF2MUg0Yy0xLjUgMC0zLTEuNjktMy0zLjVTMi41NSAzIDQgM2g0YzEuNDUgMCAzIDEuNjkgMyAzLjUgMCAxLjQxLS45MSAyLjcyLTIgMy4yNVY4LjU5Yy41OC0uNDUgMS0xLjI3IDEtMi4wOUMxMCA1LjIyIDguOTggNCA4IDRINGMtLjk4IDAtMiAxLjIyLTIgMi41UzMgOSA0IDl6bTktM2gtMXYxaDFjMSAwIDIgMS4yMiAyIDIuNVMxMy45OCAxMiAxMyAxMkg5Yy0uOTggMC0yLTEuMjItMi0yLjUgMC0uODMuNDItMS42NCAxLTIuMDlWNi4yNWMtMS4wOS41My0yIDEuODQtMiAzLjI1QzYgMTEuMzEgNy41NSAxMyA5IDEzaDRjMS40NSAwIDMtMS42OSAzLTMuNVMxNC41IDYgMTMgNnoiPjwvcGF0aD48L3N2Zz4=)](#a-magic-fairy-house-unreal-engine-from-vq-ganclip-source-arankomatsuzaki-on-twitter)魔法の妖精の家，アンリアルエンジン "a magic fairy house, unreal engine"<br/> from VQ-GAN+CLIP([出典](https://twitter.com/arankomatsuzaki/status/1400162046252138496): [@arankomatsuzaki](https://twitter.com/arankomatsuzaki) on Twitter)


![](https://i.giphy.com/media/iGDczdOFJ3qemx10f1/giphy.webp)

[![](data:image/svg+xml;base64,PHN2ZyBhcmlhLWhpZGRlbj0idHJ1ZSIgZm9jdXNhYmxlPSJmYWxzZSIgaGVpZ2h0PSIxNiIgdmVyc2lvbj0iMS4xIiB2aWV3Ym94PSIwIDAgMTYgMTYiIHdpZHRoPSIxNiI+PHBhdGggZmlsbC1ydWxlPSJldmVub2RkIiBkPSJNNCA5aDF2MUg0Yy0xLjUgMC0zLTEuNjktMy0zLjVTMi41NSAzIDQgM2g0YzEuNDUgMCAzIDEuNjkgMyAzLjUgMCAxLjQxLS45MSAyLjcyLTIgMy4yNVY4LjU5Yy41OC0uNDUgMS0xLjI3IDEtMi4wOUMxMCA1LjIyIDguOTggNCA4IDRINGMtLjk4IDAtMiAxLjIyLTIgMi41UzMgOSA0IDl6bTktM2gtMXYxaDFjMSAwIDIgMS4yMiAyIDIuNVMxMy45OCAxMiAxMyAxMkg5Yy0uOTggMC0yLTEuMjItMi0yLjUgMC0uODMuNDItMS42NCAxLTIuMDlWNi4yNWMtMS4wOS41My0yIDEuODQtMiAzLjI1QzYgMTEuMzEgNy41NSAxMyA5IDEzaDRjMS40NSAwIDMtMS42OSAzLTMuNVMxNC41IDYgMTMgNnoiPjwvcGF0aD48L3N2Zz4=)](#a-void-dimension-rendered-in-unreal-engine-from-vq-ganclip-source-arankomatsuzaki-on-twitter)虚無次元，アンリアルエンジンで塗りつぶせ "A Void Dimension Rendered in Unreal Engine"<br/> from VQ-GAN+CLIP ([出典](https://twitter.com/arankomatsuzaki/status/1397698220755537922): [@arankomatsuzaki](https://twitter.com/arankomatsuzaki) on Twitter)


<!-- ![](https://i.giphy.com/media/AtfknAgZtA3EkxvFlV/giphy.mp4) -->


<video style="width:66%" src='https://i.giphy.com/media/AtfknAgZtA3EkxvFlV/giphy.mp4' controls ></video>


[![](data:image/svg+xml;base64,PHN2ZyBhcmlhLWhpZGRlbj0idHJ1ZSIgZm9jdXNhYmxlPSJmYWxzZSIgaGVpZ2h0PSIxNiIgdmVyc2lvbj0iMS4xIiB2aWV3Ym94PSIwIDAgMTYgMTYiIHdpZHRoPSIxNiI+PHBhdGggZmlsbC1ydWxlPSJldmVub2RkIiBkPSJNNCA5aDF2MUg0Yy0xLjUgMC0zLTEuNjktMy0zLjVTMi41NSAzIDQgM2g0YzEuNDUgMCAzIDEuNjkgMyAzLjUgMCAxLjQxLS45MSAyLjcyLTIgMy4yNVY4LjU5Yy41OC0uNDUgMS0xLjI3IDEtMi4wOUMxMCA1LjIyIDguOTggNCA4IDRINGMtLjk4IDAtMiAxLjIyLTIgMi41UzMgOSA0IDl6bTktM2gtMXYxaDFjMSAwIDIgMS4yMiAyIDIuNVMxMy45OCAxMiAxMyAxMkg5Yy0uOTggMC0yLTEuMjItMi0yLjUgMC0uODMuNDItMS42NCAxLTIuMDlWNi4yNWMtMS4wOS41My0yIDEuODQtMiAzLjI1QzYgMTEuMzEgNy41NSAxMyA5IDEzaDRjMS40NSAwIDMtMS42OSAzLTMuNVMxNC41IDYgMTMgNnoiPjwvcGF0aD48L3N2Zz4=)](#a-lucid-nightmare-rendered-in-unreal-engine-from-vq-ganclip-source-arankomatsuzaki-on-twitter)"アンリアルエンジンで描かれた明晰な悪夢 "A Lucid Nightmare Rendered in Unreal Engine" from VQ-GAN+CLIP ([source](https://twitter.com/arankomatsuzaki/status/1397698397608452099): [@arankomatsuzaki](https://twitter.com/arankomatsuzaki) on Twitter)


<!-- [![](data:image/svg+xml;base64,PHN2ZyBhcmlhLWhpZGRlbj0idHJ1ZSIgZm9jdXNhYmxlPSJmYWxzZSIgaGVpZ2h0PSIxNiIgdmVyc2lvbj0iMS4xIiB2aWV3Ym94PSIwIDAgMTYgMTYiIHdpZHRoPSIxNiI+PHBhdGggZmlsbC1ydWxlPSJldmVub2RkIiBkPSJNNCA5aDF2MUg0Yy0xLjUgMC0zLTEuNjktMy0zLjVTMi41NSAzIDQgM2g0YzEuNDUgMCAzIDEuNjkgMyAzLjUgMCAxLjQxLS45MSAyLjcyLTIgMy4yNVY4LjU5Yy41OC0uNDUgMS0xLjI3IDEtMi4wOUMxMCA1LjIyIDguOTggNCA4IDRINGMtLjk4IDAtMiAxLjIyLTIgMi41UzMgOSA0IDl6bTktM2gtMXYxaDFjMSAwIDIgMS4yMiAyIDIuNVMxMy45OCAxMiAxMyAxMkg5Yy0uOTggMC0yLTEuMjItMi0yLjUgMC0uODMuNDItMS42NCAxLTIuMDlWNi4yNWMtMS4wOS41My0yIDEuODQtMiAzLjI1QzYgMTEuMzEgNy41NSAxMyA5IDEzaDRjMS40NSAwIDMtMS42OSAzLTMuNVMxNC41IDYgMTMgNnoiPjwvcGF0aD48L3N2Zz4=)](#a-lucid-nightmare-rendered-in-unreal-engine-from-vq-ganclip-source-arankomatsuzaki-on-twitter)"アンリアルエンジンで描かれた明晰(悪)夢 "A Lucid Nightmare Rendered in Unreal Engine" from VQ-GAN+CLIP ([source](https://twitter.com/arankomatsuzaki/status/1397698397608452099): [@arankomatsuzaki](https://twitter.com/arankomatsuzaki) on Twitter) -->

CLIP でモデルから望ましい振る舞いを引き出すためには，プロンプトにそのことを入力するだけでよいという，十分に一般的な表現を学習しました。
もちろん，最適な出力を得るために適切な言葉を見つけることは非常に困難です。
結局，アンリアルエンジンの魔法を発見するのに数ヶ月を要しました。
<!-- CLIP learned general enough representations that in order to induce desired behavior from the model, all we need to do is to ask for it in the prompt. 
Of course, finding the right words to get the best outputs can be quite a challenge; after all, it did take several months to discover the unreal engine trick.-->

ある意味，アンリアル・エンジンの魔法はブレイクスルーでした。
プロンプトにキーワードを追加することがいかに効果的であるかを人々に認識させたのです。
そしてここ数週間 CLIP から最高品質の出力を引き出すことを目的とした，複雑なプロンプトが使われるようになってきています。
<!-- In a way, the unreal engine trick was a breakthrough. 
It made people realize just how effective adding keywords to the prompt can be. 
And in the last couple weeks, I've seen increasingly complicated prompts being used that are aimed at extracting the highest quality outputs possible from CLIP. -->

例えば VQ-GAN+CLIP で「山頂付近の吹雪の中の小さな小屋，夕暮れ時に明かりが 1 つ灯る|アンリアルエンジン” とプロンプに入れて出力させると，次のように超リアルな出力が得られます。
<!-- For example, asking VQ-GAN+CLIP for "a small hut in a blizzard near the top of a mountain with one light turn on at dusk trending on artstation 
\| unreal engine" produces this hyper-realistic looking output: -->

<img src="https://i.imgur.com/yl4uU3q.jpg" width="66%">
<!-- ![](https://i.imgur.com/yl4uU3q.jpg) -->

[![](data:image/svg+xml;base64,PHN2ZyBhcmlhLWhpZGRlbj0idHJ1ZSIgZm9jdXNhYmxlPSJmYWxzZSIgaGVpZ2h0PSIxNiIgdmVyc2lvbj0iMS4xIiB2aWV3Ym94PSIwIDAgMTYgMTYiIHdpZHRoPSIxNiI+PHBhdGggZmlsbC1ydWxlPSJldmVub2RkIiBkPSJNNCA5aDF2MUg0Yy0xLjUgMC0zLTEuNjktMy0zLjVTMi41NSAzIDQgM2g0YzEuNDUgMCAzIDEuNjkgMyAzLjUgMCAxLjQxLS45MSAyLjcyLTIgMy4yNVY4LjU5Yy41OC0uNDUgMS0xLjI3IDEtMi4wOUMxMCA1LjIyIDguOTggNCA4IDRINGMtLjk4IDAtMiAxLjIyLTIgMi41UzMgOSA0IDl6bTktM2gtMXYxaDFjMSAwIDIgMS4yMiAyIDIuNVMxMy45OCAxMiAxMyAxMkg5Yy0uOTggMC0yLTEuMjItMi0yLjUgMC0uODMuNDItMS42NCAxLTIuMDlWNi4yNWMtMS4wOS41My0yIDEuODQtMiAzLjI1QzYgMTEuMzEgNy41NSAxMyA5IDEzaDRjMS40NSAwIDMtMS42OSAzLTMuNVMxNC41IDYgMTMgNnoiPjwvcGF0aD48L3N2Zz4=)](#source-ak92501-on-twitter-1)([出典](https://twitter.com/ak92501/status/1406678318288621572): [@ak92501](https://twitter.com/ak92501) on Twitter)


また「山頂からの眺め。眼下には夜の集落が見える|アートステーション|vray "view from on the top of mountain that can see a village below with lights on landscape painting trending on artstation | vray"」とモデルに問い合わせると，このような感動的な景色が表示されるのです。
<!-- Or querying the model with "view from on top of a mountain where you can see a village below at night with the lights on landscape painting trending on artstation \| vray" gives this awe-inspiring view: -->

<img src="https://i.imgur.com/RxzU1WC.jpg" width="66%">
<!-- ![](https://i.imgur.com/RxzU1WC.jpg) -->

[![](data:image/svg+xml;base64,PHN2ZyBhcmlhLWhpZGRlbj0idHJ1ZSIgZm9jdXNhYmxlPSJmYWxzZSIgaGVpZ2h0PSIxNiIgdmVyc2lvbj0iMS4xIiB2aWV3Ym94PSIwIDAgMTYgMTYiIHdpZHRoPSIxNiI+PHBhdGggZmlsbC1ydWxlPSJldmVub2RkIiBkPSJNNCA5aDF2MUg0Yy0xLjUgMC0zLTEuNjktMy0zLjVTMi41NSAzIDQgM2g0YzEuNDUgMCAzIDEuNjkgMyAzLjUgMCAxLjQxLS45MSAyLjcyLTIgMy4yNVY4LjU5Yy41OC0uNDUgMS0xLjI3IDEtMi4wOUMxMCA1LjIyIDguOTggNCA4IDRINGMtLjk4IDAtMiAxLjIyLTIgMi41UzMgOSA0IDl6bTktM2gtMXYxaDFjMSAwIDIgMS4yMiAyIDIuNVMxMy45OCAxMiAxMyAxMkg5Yy0uOTggMC0yLTEuMjItMi0yLjUgMC0uODMuNDItMS42NCAxLTIuMDlWNi4yNWMtMS4wOS41My0yIDEuODQtMiAzLjI1QzYgMTEuMzEgNy41NSAxMyA5IDEzaDRjMS40NSAwIDMtMS42OSAzLTMuNVMxNC41IDYgMTMgNnoiPjwvcGF0aD48L3N2Zz4=)](#source-ak92501-on-twitter-2)([出典](https://twitter.com/ak92501/status/1405296306223042566?s=20): [@ak92501](https://twitter.com/ak92501) on Twitter)


あるいは「真夜中の丘の上にある小さな蛍が飛び交う家をスタジオジブリ風に描いたマットペイント｜アートステーション｜アンリアルエンジン」といったものです。
<!-- Or "matte painting of a house on a hilltop at midnight with small fireflies flying around in the style of studio ghibli \| artstation \| unreal engine": -->

<img src="https://i.imgur.com/Zp1Nd9u.jpg" width="66%">

[![](data:image/svg+xml;base64,PHN2ZyBhcmlhLWhpZGRlbj0idHJ1ZSIgZm9jdXNhYmxlPSJmYWxzZSIgaGVpZ2h0PSIxNiIgdmVyc2lvbj0iMS4xIiB2aWV3Ym94PSIwIDAgMTYgMTYiIHdpZHRoPSIxNiI+PHBhdGggZmlsbC1ydWxlPSJldmVub2RkIiBkPSJNNCA5aDF2MUg0Yy0xLjUgMC0zLTEuNjktMy0zLjVTMi41NSAzIDQgM2g0YzEuNDUgMCAzIDEuNjkgMyAzLjUgMCAxLjQxLS45MSAyLjcyLTIgMy4yNVY4LjU5Yy41OC0uNDUgMS0xLjI3IDEtMi4wOUMxMCA1LjIyIDguOTggNCA4IDRINGMtLjk4IDAtMiAxLjIyLTIgMi41UzMgOSA0IDl6bTktM2gtMXYxaDFjMSAwIDIgMS4yMiAyIDIuNVMxMy45OCAxMiAxMyAxMkg5Yy0uOTggMC0yLTEuMjItMi0yLjUgMC0uODMuNDItMS42NCAxLTIuMDlWNi4yNWMtMS4wOS41My0yIDEuODQtMiAzLjI1QzYgMTEuMzEgNy41NSAxMyA5IDEzaDRjMS40NSAwIDMtMS42OSAzLTMuNVMxNC41IDYgMTMgNnoiPjwvcGF0aD48L3N2Zz4=)](#source-ak92501-on-twitter-3)([出典](https://twitter.com/ak92501/status/1407477121174364160): [@ak92501](https://twitter.com/ak92501) on Twitter)


どの画像も前節で見た VQ-GAN+CLIP のアートとは似ても似つかないものでした。
出力はまだシュールな感じで，部分的にまとまりがないところもあります。
ですが，全体的にポップで，編集された写真やゲームのシーンのような印象です。
「アートステーション artstation のトレンド」「アンリアル Unreal Engine」「vray」といったキーワードが、これらの作品のユニークなスタイルを決定付ける上で重要な役割を担っているのでしょう。
<!-- Each of these images looks nothing like the VQ-GAN+CLIP art we saw in the previous section. 
The outputs still have a certain surreal quality to them and maybe the coherence breaks down at a few points, but overall the images just pop like nothing else we've seen so far; they look more like edited photographs or scenes from a video game. 
So it seems that each of these keywords -- "trending on artstation", "unreal engine", "vray" -- play a crucial role in defining the unique style of these outputs.-->

このように，モデルに対して望ましい振る舞いを促す一般的なパラダイムは「プロンプト・プログラミング」と呼ばれるようになりました。ですが，これは非常に高度な技術なのです。
どのようなプロンプトが効果的なのか直感的に理解するためには，モデルがどのように「考え」，学習中にどのような種類のデータを「見た」のかについて，何らかの手がかりが必要です。
そうでなければ，プロンプトは運任せになってしまいます。
しかし，将来的にモデルがより大きく，より強力になれば，このようなことも少しは容易になることでしょう。
<!-- This general paradigm of prompting models for desired behavior is becoming known as "prompt programming", and it is really quite an art. 
In order to have any intuition as to what prompts might be effective, you need some clue as to how the model "thinks" and what types of data the model "saw" during training. Otherwise, prompting can be a little bit like dumb luck. Although hopefully, in the future, as models get even larger and more powerful, this will become a little bit easier. -->

## これは始まりに過ぎない
<!-- ## This is Just The Beginning-->

このブログ記事では CLIP ベースの生成芸術の進化における初期のマイルストーンをいくつか紹介しました。
しかし，これは決して CLIP を使って人々が作ることができたアートを広範囲にカバーしたものではありません。
[超クールな](https://twitter.com/RiversHaveWings/status/1380194080055984129?s=20) [StyleGAN+CLIPで行われた仕事](https://twitter.com/RinonGal/status/1407995418349391872?s=20) や，本当に面白い[CLIPDraw](https://arxiv.org/pdf/2106.14843.pdf) [作品](https://twitter.com/RiversHaveWings/status/1410020043178446848?s=20)，さらには[実験のサガ](https://twitter.com/advadnoun/status/1364765781758599169?s=20) [DALL-E の dVAE+CLIP で行った](https://twitter.com/advadnoun/status/1364822183751471109?s=20)[の話まではしていないんですよ](https://twitter.com/advadnoun/status/1364738521441837056?s=20)[https://twitter.com/RiversHaveWings/status/1409600293172432899?s=20]。
このように CLIP を使った新しい作品制作の方法は，毎週増えています。
まだまだ改良の余地がありそうだし，創造的な発見もたくさんありそうです。
<!-- In this blog post I've described some of the early milestones in the evolution of CLIP-based generative art. 
But by no means was this an extensive coverage of the art that people have been able to create with CLIP. 
I didn't even get around to talking about the [super cool](https://twitter.com/RiversHaveWings/status/1380194080055984129?s=20) [work that's been done with StyleGAN+CLIP](https://twitter.com/RinonGal/status/1407995418349391872?s=20) or the really interesting [CLIPDraw](https://arxiv.org/pdf/2106.14843.pdf) [work](https://twitter.com/RiversHaveWings/status/1410020043178446848?s=20) or even [the saga](https://twitter.com/advadnoun/status/1364765781758599169?s=20) [of experiments](https://twitter.com/advadnoun/status/1364822183751471109?s=20) [done with](https://twitter.com/advadnoun/status/1364738521441837056?s=20) [DALL-E's dVAE+CLIP](https://twitter.com/RiversHaveWings/status/1409600293172432899?s=20). 
I could go on and on, and the list of new methods for creating art with CLIP is expanding each week. 
In fact, it really feels like this is just the beginning; there is likely so much to improve and build upon and so many creative discoveries yet to be made. -->

もし，あなたがこのようなことに興味を持ち，CLIP ベースのアートシステムの仕組みについてもっと知りたいなら，あるいは，この分野で最も革新的なアーティストたちの動向を知りたいなら，あるいは，あなた自身がアートを生み出すことに挑戦したいのなら，ぜひ，以下のリソースをチェックしてみてください。
<!-- So if this stuff is interesting to you, and you'd like to learn more about how these CLIP based art systems work, or even if you just want to keep up with some of the most innovative artists in this space, or if you want to try your own hand at generating some art, be sure to checkout the resources below. -->

## 参考文献，ノートブック，関連 Twitter アカウント
<!-- ## References, Notebooks, and Relevant Twitter Accounts-->

### 参考文献
<!-- ### References -->

(参考文献は各作品の下にある脚注を参照，参考文献のない画像はすべて私が作成した作品です)
<!-- *(see the captions below each piece of artwork for its corresponding reference; all images without references are works that I created)* -->

- [CLIP blog post](https://openai.com/blog/clip/)
- [CLIP paper](https://arxiv.org/abs/2103.00020)
- [Big-GAN paper](https://arxiv.org/abs/1809.11096)
- [VQ-GAN paper](https://arxiv.org/abs/2012.09841)
- [The Big Sleep blog post](https://rynmurdock.github.io/2021/02/26/Aleph2Image.html)
- [DeepDream blog post](https://ai.googleblog.com/2015/07/deepdream-code-example-for-visualizing.html)
- [DALL-E blog post](https://openai.com/blog/dall-e/)
- [Multimodal Neurons Distill](https://distill.pub/2021/multimodal-neurons/)

### Colab ノート

以下の Colab ノートブックは，プロンプトを入力するだけで CLIP ベースのアートを作ることができます。それぞれ微妙に違う技法を使っています。楽しんでください
<!-- **(you can use these Colab notebooks to make your own CLIP based art; just input a prompt. They each use slightly different techniques. Have fun !)** -->

- [The Big Sleep](https://colab.research.google.com/drive/1NCceX2mbiKOSlAd_o7IU7nA9UskKN5WR?usp=sharing)
- [Aleph2Image](https://colab.research.google.com/drive/1oA1fZP7N1uPBxwbGIvOEXbTsq2ORa9vb?usp=sharing)
- [Deep Daze](https://colab.research.google.com/drive/1FoHdqoqKntliaQKnMoNs3yn5EALqWtvP?usp=sharing)
- [VQ-GAN+CLIP (codebook sampling)](https://colab.research.google.com/drive/15UwYDsnNeldJFHJ9NdgYBYeo6xPmSelP)
- [VQ-GAN+CLIP (z+quantize)](https://colab.research.google.com/drive/1L8oL-vLJXVcRzCFbPwOoMkPKJ8-aYdPN)
- [VQ-GAN+CLIP (EleutherAI)](https://colab.research.google.com/drive/17AqhaKLZmmUA27aNSc6fJYMR9uypeIci?usp=sharing)

**(注: Google Colabに慣れていない場合は [このチュートリアル](https://docs.google.com/document/d/1Lu7XPRKlNhBQjcKr8k8qRzUzbBW7kzxb5Vu72GMRn2E/edit)をお勧めします)** 
<!-- *(Note: if you are unfamiliar with Google Colab, I can recommend [this tutorial](https://docs.google.com/document/d/1Lu7XPRKlNhBQjcKr8k8qRzUzbBW7kzxb5Vu72GMRn2E/edit)* -->

### 関連するツイッターアカウント
<!-- ### Relevant Twitter Accounts -->

*CLIP で生成したアートを頻繁に投稿している Twitter アカウントです。*
<!-- *(these are all twitter accounts that frequently post art generated with CLIP)* -->

- [@ak92501], [@arankomatsuzaki], [@RiversHaveWings], [@advadnoun], [@eps696],[@quasimondo], [@M_PF], [@hollyherndon], [@matdryhurst], [@erocdrahs], [@erinbeess]
, [@ganbrood], [@92C8301A], [@bokar_n], [@genekogan], [@danielrussruss], [@kialuy], [@jbusted1], [@BoneAmputee], [@eyaler]

Published 30 Jun 2021

- [Art](https://ml.berkeley.edu/blog/tag/art/), [Neural Networks](https://ml.berkeley.edu/blog/tag/neural-networks/), [CLIP](https://ml.berkeley.edu/blog/tag/clip/), [DALL-E](https://ml.berkeley.edu/blog/tag/dall-e/), [Generative Modeling](https://ml.berkeley.edu/blog/tag/generative-modeling/), [BigGAN](https://ml.berkeley.edu/blog/tag/big-gan/), [The Big Sleep](https://ml.berkeley.edu/blog/tag/the-big-sleep/), - [DeepDream](https://ml.berkeley.edu/blog/tag/deep-dream/), [AI Art](https://ml.berkeley.edu/blog/tag/ai-art/)

私たちはカリフォルニア大学バークレー校の学生団体で、キャンパス内に活気ある機械学習コミュニティを構築・育成し、より大きな機械学習コミュニティやそれ以外の場所にも貢献することを目的としています。[**Machine Learning at Berkeley** on Twitter](https://www.twitter.com/berkeleyml)
<!-- We are a student organization at UC Berkeley dedicated to building and fostering a vibrant machine learning community on campus while contributing to the greater machine learning community and beyond.[**Machine Learning at Berkeley** on Twitter](https://www.twitter.com/berkeleyml) -->

**チャーリー スネル Charlie Snell**
<img src="https://ml.berkeley.edu/blog/static/charlie-92a143a9f944ba66635018badf4ba68a.jpeg" width="14%">

私は UC Berkeley の学部生で ML@B のメンバーです。
Berkeley NLP group で NLP の研究，Sergey Levine の RAIL lab で RL 研究をしています。生成的モデリングからメタ学習，RL，オープンエンドまで，ML に関するあらゆるトピックに興味があります。
また大の音楽オタクで，暇さえあればビートを作っています (あまりうまくないけど)。
<!-- I\'m a undergraduate student at UC Berkeley and a member of ML@B. 
I also do NLP research in the Berkeley NLP group and RL research with Sergey Levine\'s RAIL lab. I\'m interested in all sorts of topics in ML, ranging from generative modeling to meta-learning to RL and open-endedness. I\'m also a huge music nerd; I make beats in my free time, although they aren\'t very good. -->

[フォローをお願いします @sea_snell](https://twitter.com/sea_snell)
