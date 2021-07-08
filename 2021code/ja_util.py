import re
import jaconv
import MeCab
import jaconv
import unicodedata
import string

def unicode_normalize(cls, s):
    pt = re.compile('([{}]+)'.format(cls))

    def norm(c):
        return unicodedata.normalize('NFKC', c) if pt.match(c) else c

    s = ''.join(norm(x) for x in re.split(pt, s))
    s = re.sub('－', '-', s)
    return s

def remove_extra_spaces(s):
    s = re.sub('[ 　]+', ' ', s)
    blocks = ''.join(('\u4E00-\u9FFF',  # CJK UNIFIED IDEOGRAPHS
                      '\u3040-\u309F',  # HIRAGANA
                      '\u30A0-\u30FF',  # KATAKANA
                      '\u3000-\u303F',  # CJK SYMBOLS AND PUNCTUATION
                      '\uFF00-\uFFEF'   # HALFWIDTH AND FULLWIDTH FORMS
                      ))
    basic_latin = '\u0000-\u007F'

    def remove_space_between(cls1, cls2, s):
        p = re.compile('([{}]) ([{}])'.format(cls1, cls2))
        while p.search(s):
            s = p.sub(r'\1\2', s)
        return s

    s = remove_space_between(blocks, blocks, s)
    s = remove_space_between(blocks, basic_latin, s)
    s = remove_space_between(basic_latin, blocks, s)
    return s

def normalize_neologd(s):
    s = s.strip()
    s = unicode_normalize('０-９Ａ-Ｚａ-ｚ｡-ﾟ', s)

    def maketrans(_from, _to):
        return {ord(x): ord(y) for x, y in zip(_from, _to)}

    s = re.sub('[˗֊‐‑‒–⁃⁻₋−]+', '-', s)  # normalize hyphens
    s = re.sub('[﹣－ｰ—―─━ー]+', 'ー', s)  # normalize choonpus
    s = re.sub('[~∼∾〜〰～]', '', s)  # remove tildes
    s = s.translate(
        maketrans('!"#$%&\'()*+,-./:;<=>?@[¥]^_`{|}~｡､･｢｣',
                  '！”＃＄％＆’（）＊＋，－．／：；＜＝＞？＠［￥］＾＿｀｛｜｝〜。、・「」'))

    s = remove_extra_spaces(s)
    # keep ＝, ・,「,」
    s = unicode_normalize('！”＃＄％＆’（）＊＋，－．／：；＜＞？＠［￥］＾＿｀｛｜｝〜', s)  
    s = re.sub('[’]', '\'', s)
    s = re.sub('[”]', '"', s)
    return s

# source https://ja.wikipedia.org/wiki/%E3%83%A2%E3%83%BC%E3%83%A9
sample='じゃじゅじょ。きゃきゅきょ。日本語の多くの方言においても同様である。\
日本語の仮名1文字が基本的に1拍である。\
ただし、捨て仮名（「ぁ」「ぃ」「ぅ」「ぇ」「ぉ」「ゃ」「ゅ」「ょ」「ゎ」といった小書きの仮名）は、\
その前の仮名と一体になって1拍である（たとえば「ちゃ」で1拍。拗音も参照）。\
一方、長音「ー」、促音「っ」、撥音「ん」は、独立して1拍に数えられる（これが「音節」と異なる主な点である）。\
音節単位で見るなら、長音は長母音の後半部分を、促音は長子音の前半部分を切り取ったものであり、\
撥音は音節末鼻音や鼻母音をモーラとしたものといえる（鼻母音は基になる母音＋「ん」の2モーラになる）。\
これらは、「語頭に現れない」「単独で音節を形成しない」「お互いに連続することが稀である」などの性質をもち、\
二重母音の第二要素も含めて特殊拍（special mora）と呼称される。\
これらを除いて、単独で音節を形成する拍は自立拍（independent mora）と呼称される。'

sample2 = '日本語の多くの方言においても同様である。日本語の仮名1文字が基本的に1拍である。\
ただし、捨て仮名（「ぁ」「ぃ」「ぅ」「ぇ」「ぉ」「ゃ」「ゅ」「ょ」「ゎ」といった\
小書きの仮名）は、その前の仮名と一体になって1拍である（たとえば「ちゃ」で1拍。拗音も参照）。\
一方、長音「ー」、促音「っ」、撥音「ん」は、独立して1拍に数えられる（これが「音節」\
と異なる主な点である）。音節単位で見るなら、長音は長母音の後半部分を、促音は長子音の前半部\
分を切り取ったものであり、撥音は音節末鼻音や鼻母音をモーラとしたものといえる（鼻母音は基に\
なる母音＋「ん」の2モーラになる）。これらは、「語頭に現れない」「単独で音節を形成しない」\
「お互いに連続することが稀である」などの性質をもち、二重母音の第二要素も含めて特殊拍\
（special mora）と呼称される。これらを除いて、単独で音節を形成する拍は自立拍\
（independent mora）と呼称される。'

class mora_wakati(object):
    """
    source <https://ja.wikipedia.org/wiki/%E3%83%A2%E3%83%BC%E3%83%A9>
    日本語の多くの方言においても同様である。日本語の仮名1文字が基本的に1拍である。
    ただし、捨て仮名（「ぁ」「ぃ」「ぅ」「ぇ」「ぉ」「ゃ」「ゅ」「ょ」「ゎ」といった小書きの仮名）は、
    その前の仮名と一体になって1拍である（たとえば「ちゃ」で 1 拍。拗音も参照）。
    一方、長音「ー」、促音「っ」、撥音「ん」は、独立して1拍に数えられる（これが「音節」と異なる主な点である）。
    音節単位で見るなら、長音は長母音の後半部分を、促音は長子音の前半部分を切り取ったものであり、
    撥音は音節末鼻音や鼻母音をモーラとしたものといえる（鼻母音は基になる母音＋「ん」の2モーラになる）。
    これらは、「語頭に現れない」「単独で音節を形成しない」「お互いに連続することが稀である」などの性質をもち、
    二重母音の第二要素も含めて特殊拍（special mora）と呼称される。
    これらを除いて、単独で音節を形成する拍は自立拍（independent mora）と呼称される。

    # 基礎日本語学，衣畑(編) 2019, ひつじ書房, page 28 より
    ## 7.2 モーラ
    モーラも音節と同様に隣り合う分節音が強く結びついた単位であるが，音節よりも小さい単位であり，
    両者には「一つの音節は必ず１つ以上のモーラを含む」という階層関係が認められる。
    たとえば「kan」「缶」のの音節数は１だが，モーラ数は２であり「ka」と「N」とに分かれる。

    モーラは 短母音 V のみあるいは子音と短母音だけ CV からなる音節１つ分の長さを基準とした時間的単位であると定義できる。
    ここでの「長さ」は話し手・聞き手にとって同じ長さと感じられる心理的な長さであり，
    音声の物理的長さとは必ずしも一致しない。モーラは，同じ長さを持つ単位と意識されることから，
    **等時性** の単位でもあると言われる。

    ## 7.3 日本語学に特有のモーラの分類
    日本語学に特有のモーラの分類に，拗音/直音，濁音/清音，自立モーラ/特殊モーラ の分類がある。
    特殊モーラは促音，撥音，長母音・二重母音の第２要素からなる。
    
    ### 7.3.1 拗音・直音
    拗音とは，文字の上では「キャ，キュ，キョ」のように大文字の後ろに小文字の「ャ，ュ，ョ」が書かれるモーラである。
    拗音ではないモーラは（特殊モーラを除き），直音と呼ばれる。
    
    ### 7.3.2 濁音・清音
    濁音とは 仮名の上では「バ」のように濁点 ゛を伴う文字が表すモーラである。一方「ハ」など
    濁点を伴わない文字が表すモーラは清音 「パ」など半濁点 ゜ を伴う文字が表すモーラは半濁音と呼ばれる。

    ### 7.3.3 促音
    促音とは 仮名の上では小文字の「ッ」によって表されるモーラである。日本語学では促音を独立の音素
    とみなし /Q/ と表記することが広く行われている。

    ### 7.3.4 撥音
    撥音とは 仮名の上では「ン」によって表されるモーラである。日本語学では 撥音を独立の音素
    とみなし /N/ と表記することが行われている。

    ### 7.3.5 特殊モーラと自立モーラ
    特殊モーラ，促音，撥音，長母音・二重母音の第２要素。１モーラの長さを持ちながら音節とし
    て独立できないモーラを特殊モーラと定義する。特殊モーラでないモーラは自立モーラである。
    促音と撥音を独立の音素とみなし /Q/, /N/ と表記する枠組みがある。
    この枠組では長母音の第２要素も同様に音素とみなされる。
    「神学校」/siNgaQkoH/.


    from <https://www.unicode.org/reports/tr44/>
    ### 5.7.1 General Category Values

    The General_Category property of a code point provides for the most general classification of that code point.
    It is usually determined based on the primary characteristic of the assigned character for that code point.
    For example, is the character a letter, a mark, a number, punctuation, or a symbol, and if so, of what type?
    Other General_Category values define the classification of code points which are not assigned to regular graphic characters, 
    including such statuses as private-use, control, surrogate code point, and reserved unassigned.
    コードポイントの General_Category プロパティは、そのコードポイントの最も一般的な分類を示すものである。
    通常、そのコードポイントに割り当てられた文字の主な特徴に基づいて決定される。
    例えば, 文字, マーク, 数字, 句読点, 記号のいずれかであり, いずれかであれば, どのような種類のものであるか。
    その他の General_Category 値は, 通常の図形文字に割り当てられていないコードポイントの分類を定義するもので, 
    private-use, control, surrogate code point, reserved unassigned などのステータスがある。

    Many characters have multiple uses, and not all such cases can be captured entirely by the General_Category value. For example, the General_Category value of Latin, Greek, or Hebrew letters does not attempt to cover (or preclude) the numerical use of such letters as Roman numerals or in other numerary systems.
    Conversely, the General_Category of ASCII digits 0..9 as Nd (decimal digit) neither attempts to cover (or preclude) the occasional use of these digits as letters in various orthographies.
    The General_Category is simply the first-order, most usual categorization of a character.
    多くの文字には複数の用途があり, General_Category の値ですべてのケースを完全に把握できるわけではありません。
    例えば, ラテン語, ギリシャ語, ヘブライ語の文字の General_Category 値は, ローマ数字や他の数字体系での数値使用をカバーしようとはしていません (または除外しています)。
    逆に ASCII 数字の 0 -9 を Nd (10進数) とする General_Category は, これらの数字が様々な書法で文字として使用されることをカバーしようとするものではありません (または除外するものでもありません)。
    General_Category は 単に文字の一次的な 最も普通の分類です。

    For more information about the General_Category property, see Chapter 4, Character Properties in [Unicode].

    <!-- The values in the General_Category field in UnicodeData.txt make use of the short, abbreviated property value aliases for General_Category.
    For convenience in reference, Table 12 lists all the abbreviated and long value aliases for General_Category values, reproduced from PropertyValueAliases.txt, along with a brief description of each category.-->
    UnicodeData.txt の General_Category フィールドの値は, General_Category に対する短くて省略されたプロパティ値のエイリアスを利用しています。
    参照に便利なように 表12 では PropertyValueAliases.txt から再現された General_Category 値のすべての省略された値と長い値のエイリアスを、各カテゴリの簡単な説明とともに示しています。

    ```markdown
    |Abbr|Long|Description|
    |:---|:---|:---|
    |Lu|Uppercase_Letter|an uppercase letter|
    |Ll|Lowercase_Letter|a lowercase letter|
    |Lt|Titlecase_Letter|a digraphic character, with first part uppercase|
    |LC|Cased_Letter|Lu \| Ll \| Lt|
    |Lm|Modifier_Letter|a modifier letter|
    |Lo|Other_Letter|other letters, including syllables and ideographs|
    |L|Letter|Lu \| Ll \| Lt \| Lm \| Lo|
    |Mn|Nonspacing_Mark|a nonspacing combining mark (zero advance width)|
    |Mc|Spacing_Mark|a spacing combining mark (positive advance width)|
    |Me|Enclosing_Mark|an enclosing combining mark|
    |M|Mark|Mn \| Mc \| Me|
    |Nd|Decimal_Number|a decimal digit|
    |Nl|Letter_Number|a letterlike numeric character|
    |No|Other_Number|a numeric character of other type|
    |N|Number|Nd \| Nl \| No|
    |Pc|Connector_Punctuation|a connecting punctuation mark, like a tie|
    |Pd|Dash_Punctuation|a dash or hyphen punctuation mark|
    |Ps|Open_Punctuation|an opening punctuation mark (of a pair)|
    |Pe|Close_Punctuation|a closing punctuation mark (of a pair)|
    |Pi|Initial_Punctuation|an initial quotation mark|
    |Pf|Final_Punctuation|a final quotation mark|
    |Po|Other_Punctuation|a punctuation mark of other type|
    |P|Punctuation|Pc \| Pd \| Ps \| Pe \| Pi \| Pf \| Po|
    |Sm|Math_Symbol|a symbol of mathematical use|
    |Sc|Currency_Symbol|a currency sign|
    |Sk|Modifier_Symbol|a non-letterlike modifier symbol|
    |So|Other_Symbol|a symbol of other type|
    |S|Symbol|Sm \| Sc \| Sk \| So|
    |Zs|Space_Separator|a space character (of various non-zero widths)|
    |Zl|Line_Separator|U+2028 LINE SEPARATOR only|
    |Zp|Paragraph_Separator|U+2029 PARAGRAPH SEPARATOR only|
    |Z|Separator|Zs \| Zl \| Zp|
    |Cc|Control|a C0 or C1 control code|
    |Cf|Format|a format control character|
    |Cs|Surrogate|a surrogate code point|
    |Co|Private_Use|a private-use character|
    |Cn|Unassigned|a reserved unassigned code point or a noncharacter|
    |C|Other|Cc \| Cf \| Cs \| Co \| Cn|
    ```
    """

    # <https://qiita.com/shimajiroxyz/items/a133d990df2bc3affc12>
    def __init__(self):
        """各条件を正規表現で表す
        source: https://qiita.com/shimajiroxyz/items/a133d990df2bc3affc12

        方針:
        考えやすくするために、入力は記号を含まない全角カタカナの文字列とします。
        また、長音で表せるところは長音に変換されているものとします。
        これは例えば「ガッキュウ」は「ガッキュー」のように表現されているという意味です。

        なお、漢字仮名交じり文を発音のカタカナ文字列に変換する方法は別記事にまとめましたので、もしよければ御覧ください。
        ただし、MeCab を使っていますので、辞書にない言葉は変換できません。

        このとき、モーラの構成条件を下記のいずれかと定義します。

        ウ段＋「ァ/ィ/ェ/ォ」
        イ段（「イ」を除く）＋「ャ/ュ/ェ/ョ」
        「テ/デ」＋「ャ/ィ/ュ/ョ」
        上記以外のカタカナ１文字
        """
        self.c1 = '[ウクスツヌフムユルグズヅブプヴ][ァィェォ]' #ウ段＋「ァ/ィ/ェ/ォ」
        #self.c2 = '[イキシシニヒミリギジヂビピ][ャュェョ]' #イ段（「イ」を除く）＋「ャ/ュ/ェ/ョ」
        self.c2 = '[イキシチニヒミリギジヂビピ][ャュェョ]' #イ段（「イ」を除く）＋「ャ/ュ/ェ/ョ」
        self.c3 = '[テデ][ィュ]' #「テ/デ」＋「ャ/ィ/ュ/ョ」
        self.c4 = '[ァ-ヴー]' #カタカナ１文字（長音含む）
        self.c5 = '[，、.。「」]'
        #self.c6 = '[ィ]' #カタカナ１文字（長音含む）
        self.cond = '('+self.c1+'|'+self.c2+'|'+self.c3+'|'+self.c4+'|'+self.c5+')'
        #self.cond = '('+self.c1+'|'+self.c2+'|'+self.c3+'|'+self.c4+'|'+self.c5+'|'+self.c6+')'
        self.re_mora = re.compile(self.cond)
        self.mecabtagger = MeCab.Tagger()

        # 基礎日本語学，衣畑 智秀(編), 2019, ひつじ書房, page 21
        ibata_p21_ = {'マメモムミミャミョミュ': ['ma', 'me', 'mo', 'mu', 'mi', 'mya', 'mju', 'mjo'],
                     'パペポプピピャピュピョ': ['pa', 'pe', 'po', 'pu', 'pi', 'pja', 'pjo', 'pju'],
                     'バベボブビビャビョビュ': ['ba', 'be', 'bo', 'bu', 'bi', 'bja', 'bjo', 'bju'],
                     'ナネノヌニニャニョニュ': ['na', 'ne', 'no', 'nu', 'ni', 'nja', 'njo', 'nju'],
                     'ラレロルリリャリョリュ': ['ra', 're', 'ro', 'ru', 'ri', 'rja', 'rjo', 'rju'],
                     'カケコクキキャキョキュ': ['ka', 'ke', 'ko', 'ku', 'ki', 'kja', 'kjo', 'kyu'],
                     'ガゲゴグギギャギョギュ': ['ga', 'ge', 'go', 'gu', 'gi', 'gja', 'gjo', 'gju'],
                     'ワ': ['wa'],
                     'ヤユヨ': ['ja', 'jo', 'ju'],
                     'サセソスシシャショシュ': ['sa', 'se', 'so', 'su', 'si', 'sja', 'sjo', 'sju'],
                     #'ザゼゾズジジャジョジュ': ['za', 'ze', 'zo', 'zu', 'zi', 'zja', 'zjo', 'zju'], 
                     'ザゼゾズジジャジョジュ': ['za', 'ze', 'zo', 'zu', 'zi', 'zja', 'zjo', 'zju'],
                     'ジェティヲファ':['zje','ti', 'o', 'ha'],
                     'タテトツチチャチョチュ': ['ta', 'te', 'to', 'tu', 'ti', 'tja', 'tjo', 'tju'],
                     'ダデドヅヂヂャヂュヂョ': ['da', 'de', 'do', 'zu', 'zi', 'zja', 'zju', 'zjo'],
                     'ハヘホフヒヒャヒョヒュ': ['ha', 'he', 'ho', 'hu', 'hi', 'hja', 'hjo', 'hju'],
                     'アエオウイィ': ['a', 'e', 'o', 'u', 'i', 'i'],   # 母音 浅川追加 2021-0129
                     #'アエオウイ': ['a', 'e', 'o', 'u', 'i'],         # 母音 浅川追加 2021-0129
                     'ッンー': ['Q', 'N', 'H'],                       # 特殊モーラ 浅川追加 2021-0129
                      'チェフィフェフォ':['tje','hji','hje','hjo'],
                     'ディデュ':['di','du'],
                     'ウェウィ':['ue', 'ui'],                         # 2021_0422 追加 wikipedai.ja で調べた結果 これは微妙だなー
                     'ヴァヴェヴィヴォ':['vja','vje','vje','vjo'],     # 2021_0422 追加 wikipedai.ja で調べた結果 これは微妙だなー
                     #'シャシェシュショ':['sya','sye','syu','syo'],
                     'ファフェフュフョ':['fya','fye','fyu','fyo'],
                     'ゥ':['u'],
                     '「」，、。．・': ['<quo>','<quo>','<pun>','<pun>','<eos>','<eos>','<etc>']
                    }

        ibata_p21 = {'マメモムミミャミョミュ': ['ma', 'me', 'mo', 'mu', 'mi', 'mya', 'mju', 'mjo'],
                    'パペポプピピャピュピョ': ['pa', 'pe', 'po', 'pu', 'pi', 'pja', 'pjo', 'pju'],
                    'バベボブビビャビョビュ': ['ba', 'be', 'bo', 'bu', 'bi', 'bja', 'bjo', 'bju'],
                    'ナネノヌニニャニョニュ': ['na', 'ne', 'no', 'nu', 'ni', 'nja', 'njo', 'nju'],
                    'ラレロルリリャリョリュ': ['ra', 're', 'ro', 'ru', 'ri', 'rja', 'rjo', 'rju'],
                    'カケコクキキャキョキュキェ': ['ka', 'ke', 'ko', 'ku', 'ki', 'kja', 'kjo', 'kyu', 'kye'],
                    'ガゲゴグギギャギョギュ': ['ga', 'ge', 'go', 'gu', 'gi', 'gja', 'gjo', 'gju'],
                    'ハヘホフヒ': ['ha', 'he', 'ho', 'hu', 'hi'],
                    #'ザゼゾズジジャジョジュ': ['za', 'ze', 'zo', 'zu', 'zi', 'zja', 'zjo', 'zju'], 
                    'サセソスシシャショシュシェ': ['sa', 'se', 'so', 'su', 'si', 'sja', 'sjo', 'sju', 'sye'],  # シェ 追加 2021_0422
                    'ザゼゾズジジャジョジュジェ': ['za', 'ze', 'zo', 'zu', 'zi', 'zja', 'zjo', 'zju', 'zje'],
                    'タテトツチチャチョチュ': ['ta', 'te', 'to', 'tu', 'ti', 'tja', 'tjo', 'tju'],
                    'ダデドヅヂヂャヂュヂョ': ['da', 'de', 'do', 'zu', 'zi', 'zja', 'zju', 'zjo'],
                    'ヒャヒュヒョヒェ':['hja', 'hju', 'hjo', 'hje'],
                    'アァエェオォウゥイィ': ['a', 'a', 'e', 'e', 'o', 'o', 'u', 'u', 'i', 'i'],   # 母音 浅川追加 2021-0129
                    #'アエオウイ': ['a', 'e', 'o', 'u', 'i'],         # 母音 浅川追加 2021-0129
                    'ッンー': ['Q', 'N', 'H'],                       # 特殊モーラ 浅川追加 2021-0129
                    'ヤユヨヱ': ['ja', 'jo', 'ju','e'],
                    'ャュョ': ['ja', 'jo', 'ju'],
                    'ワヲ': ['wa','o'],

                    'ウァウェウィウォ':['ua', 'ue', 'ui', 'wo'],         # 2021_0422 追加 wikipedai.ja で調べた結果 これは微妙だなー
                    'ヴァヴェヴィヴォヴ':['va','ve','ve','vo','vu'],     # 2021_0422 追加 wikipedai.ja で調べた結果 これは微妙だなー
                    'ファフィフェフォ':['fa','fi','fe','fo'], 
                    'ツァツィツェツォ':['tsa','tsi','tse','tso'],
                    'イェスィミェルィクィ': ['e', 'si', 'mje', 'ri', 'ki'],
                    'クァクェクォチェテュ': ['ka', 'ke', 'ko', 'te', 'tu'], #, 'ki', 'ku', 'ke', 'ko'],
                    'ズィグァギェリェ':['zi','ga', 'ge', 're'],
                    'ブァブェブィブォ':['va', 've', 'vi', 'vo'],
                    'ティディデュ':['ti', 'di','du'],
                    #'デァディデュ':['da', 'di','du'],
                    #'ツァツィ':['tsa','tsi'],
                    #'フォフョ':['fyu','fyo'],
                    'グィヌィムィビェ':['qi', 'ni', 'mi', 'be'],
                    #'ファフィフェフォフュフョ':['fa','fi','fe','fo','fyu','fyo'],
                    '「」，、。．・': ['<quo>','<quo>','<pun>','<pun>','<eos>','<eos>','<etc>']
        }
        kana2mora = {}
        for key, val in ibata_p21.items():
            moras = self.parse(key)
            #print(moras)
            for mora in moras:
                if not mora in kana2mora:
                    idx = moras.index(mora)
                    kana2mora[mora] = val[idx]
                else:
                    print('Error', mora, idx, key) #, val[i])

        self.kana2mora = kana2mora
        self.mora2kana = {v:k for k, v in kana2mora.items()}
        self.mora2idx = {k:i for i, k in enumerate(self.mora2kana.keys())}
        self.idx2mora = {i:k for i, k in enumerate(self.mora2kana.keys())}
        self.kana2idx = {k:i for i, k in enumerate(self.kana2mora.keys())}
        self.idx2kana = {i:k for i, k in enumerate(self.kana2mora.keys())}


    def format_text(self, text):
        #text = unicodedata.normalize("NFKC", text)  # 全角記号をざっくり半角へ置換（でも不完全）
        text = unicodedata.normalize("NFC", text)

        # 記号を消し去るための魔法のテーブル作成
        table = str.maketrans("", "", string.punctuation  + "「」、。・，．？")
        #table = str.maketrans("", "", string.punctuation)
        text = text.translate(table)
        return text


    def getPronunciation(self, text):
        result = self.mecabtagger.parse(text).splitlines() #mecabの解析結果の取得
        result = result[:-1] #最後の1行は不要な行なので除く

        pro = '' #発音文字列全体を格納する変数
        for v in result:
            if '\t' not in v: 
                continue
            surface = v.split('\t')[0] #表層形
            p = v.split('\t')[1].split(',')[-1] #発音を取得したいとき
            #p = v.split('\t')[1].split(',')[-2] #ルビを取得したいとき
            #発音が取得できていないとき surface で代用
            if p == '*': 
                p = surface
            pro += p

        pro = jaconv.hira2kata(pro) #ひらがなをカタカナに変換
        pro = self.format_text(pro) # 余計な記号を削除
        return pro


    def parse(self, ent):
        return self.re_mora.findall(self.getPronunciation(ent))


    def parse2romaji(self, ent):
        #return "".join(self.kana2mora[mora] for mora in self.parse(ent))
        return [self.kana2mora[mora] for mora in self.parse(ent)] # if mora in self.kana2mora else mora]

    def  __call__(self, str):
        return self.parse(str)

#print(moraWakachi().getPronunciation(sample2))
#moraWakachi().getPronunciation(sample)
#moraWakachi().getPronunciation(sample2)
