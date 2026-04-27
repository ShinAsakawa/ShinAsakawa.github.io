"""
psycholing_metrics_ja.py — 心理言語学変数の一括計算
================================================

入力: (単語, 読み) のペア
出力: ONS, PNS, OLD20, consistency, reading_entropy

全てWLSPの規則・母集団に基づいて計算する。
読みは WLSP 方式（長音→母音化, カタカナ→ひらがな）に正規化される。

ONS/PNS は重複排除（unique）でカウントする（Coltheart's N の本来の定義）。
JALEXは重複許容（with_duplicates）でカウントしている可能性があるため、
JALEXの公式値と突き合わせる場合は注意が必要。

依存: pandas
      python-Levenshtein（推奨、高速化。なくても動作する）

使い方:
    engine = PsychoLingEngine(wlsp_path)
    result = engine.compute("研究", "ケンキュウ")
    print(result)
"""

import math
import unicodedata
import pandas as pd
import numpy as np
from collections import defaultdict, Counter
from itertools import combinations, product
from typing import List, Tuple, Dict, Optional, NamedTuple
try:
    import Levenshtein as _lev
    _levenshtein = _lev.distance
except ImportError:
    def _levenshtein(a, b):
        n, m = len(a), len(b)
        dp = list(range(m + 1))
        for i in range(1, n + 1):
            prev, dp[0] = dp[0], i
            for j in range(1, m + 1):
                cost = 0 if a[i - 1] == b[j - 1] else 1
                dp[j], prev = min(dp[j] + 1, dp[j - 1] + 1, prev + cost), dp[j]
        return dp[m]

try:
    import jamorasep as _jamorasep
    _has_jamorasep = True
except ImportError:
    _has_jamorasep = False

# ╔═══════════════════════════════════════════════════════════╗
# ║  1. WLSP 読み正規化（長音規則）                            ║
# ╚═══════════════════════════════════════════════════════════╝

_VOWEL_KATA = {}
_ROWS = {
    'ア': 'アカサタナハマヤラワガザダバパァャヮ',
    'イ': 'イキシチニヒミリギジヂビピィ',
    'ウ': 'ウクスツヌフムユルグズヅブプゥュヴ',
    'エ': 'エケセテネヘメレゲゼデベペェ',
    'オ': 'オコソトノホモヨロヲゴゾドボポォョ',
}
for _v, _cs in _ROWS.items():
    for _c in _cs:
        _VOWEL_KATA[_c] = _v

_VOWEL_HIRA = {}
_V2H = {'ア': 'あ', 'イ': 'い', 'ウ': 'う', 'エ': 'え', 'オ': 'お'}
for _c, _v in _VOWEL_KATA.items():
    _cp = ord(_c)
    if 0x30A1 <= _cp <= 0x30F6:
        _VOWEL_HIRA[chr(_cp - 0x60)] = _V2H[_v]
    elif _c == 'ヴ':
        _VOWEL_HIRA['ゔ'] = 'う'
    elif _c == 'ヮ':
        _VOWEL_HIRA[chr(ord('ヮ') - 0x60)] = _V2H[_v]

VOWEL_MAP = {**_VOWEL_KATA, **_VOWEL_HIRA}
SPECIAL_MORA = frozenset('ンッーんっ')


def resolve_chouon(text: str) -> str:
    """ー を直前仮名の母音段に解決する（WLSP規則 R1-R4）。"""
    if not isinstance(text, str):
        return text
    chars = list(text)
    for i, ch in enumerate(chars):
        if ch != 'ー':
            continue
        for j in range(i - 1, -1, -1):
            v = VOWEL_MAP.get(chars[j])
            if v is not None:
                chars[i] = v
                break
            if chars[j] not in SPECIAL_MORA:
                break
    return ''.join(chars)


def kata_to_hira(text: str) -> str:
    """カタカナ → ひらがな変換（R3）。"""
    if not isinstance(text, str):
        return text
    result = []
    for ch in text:
        cp = ord(ch)
        if 0x30A1 <= cp <= 0x30F6:
            result.append(chr(cp - 0x60))
        elif ch == 'ヴ':
            result.append('ゔ')
        else:
            result.append(ch)
    return ''.join(result)


def normalize_reading(reading: str) -> str:
    """読みを WLSP 方式に正規化する。"""
    if not isinstance(reading, str):
        return reading
    reading = resolve_chouon(reading)
    reading = kata_to_hira(reading)
    return reading


def primary_yomi(y: str) -> str:
    """・含む読みから代表形（先頭）を取る。"""
    return y.split('・')[0] if '・' in y else y


# ╔═══════════════════════════════════════════════════════════╗
# ║  2. モーラ分解                                            ║
# ╚═══════════════════════════════════════════════════════════╝

_COMBO_HIRA = set('ゃゅょぁぃぅぇぉゎ')
_SPECIAL_HIRA = set('んっ')


def tokenize_mora(reading: str) -> tuple:
    """ひらがな読みをモーラ単位に分解し tuple で返す。"""
    if not isinstance(reading, str) or reading == '':
        return ()
    morae = []
    i = 0
    while i < len(reading):
        ch = reading[i]
        if (i + 1 < len(reading)
                and reading[i + 1] in _COMBO_HIRA
                and ch not in _SPECIAL_HIRA):
            morae.append(ch + reading[i + 1])
            i += 2
        else:
            morae.append(ch)
            i += 1
    return tuple(morae)


# ╔═══════════════════════════════════════════════════════════╗
# ║  3. ONS / PNS（テンプレートインデックス）                     ║
# ║     重複排除（unique）: 同一語形は1回だけカウント              ║
# ╚═══════════════════════════════════════════════════════════╝

SENTINEL = '\x00'


def _build_ons_index(words):
    """正書法テンプレートインデックス（重複排除: set）。"""
    idx = defaultdict(set)
    for w in words:
        for i in range(len(w)):
            idx[w[:i] + SENTINEL + w[i + 1:]].add(w)
    return idx


def _compute_ons(target, idx):
    nb = set()
    for i in range(len(target)):
        nb.update(idx.get(target[:i] + SENTINEL + target[i + 1:], set()))
    nb.discard(target)
    return len(nb)


def _build_pns_index(mora_tuples):
    """音韻テンプレートインデックス（重複排除: set）。"""
    idx = defaultdict(set)
    for mt in mora_tuples:
        for i in range(len(mt)):
            idx[mt[:i] + (SENTINEL,) + mt[i + 1:]].add(mt)
    return idx


def _compute_pns(mora_tuple, idx):
    nb = set()
    for i in range(len(mora_tuple)):
        nb.update(idx.get(mora_tuple[:i] + (SENTINEL,) + mora_tuple[i + 1:],
                          set()))
    nb.discard(mora_tuple)
    return len(nb)


# ╔═══════════════════════════════════════════════════════════╗
# ║  3b. セグメントレベル PNS（訓令式ローマ字）                     ║
# ║      jamorasep.parse(output_format='kunrei', phoneme=False) ║
# ╚═══════════════════════════════════════════════════════════╝

def _reading_to_segments(reading: str) -> tuple:
    """ひらがな/カタカナ読みを訓令式モーラ列に変換する。

    例: 'かき' → ('ka','ki'), 'がっこう' → ('ga','k','ko','u')
    モーラ単位の訓令式表記を保持し、CV構造を維持する。
    jamorasep が利用できない場合は空 tuple を返す。
    """
    if not _has_jamorasep or not isinstance(reading, str) or len(reading) == 0:
        return ()
    try:
        segs = _jamorasep.parse(reading, output_format='kunrei', phoneme=False)
        return tuple(segs)
    except Exception:
        return ()


def _build_segment_pns_index(segment_tuples):
    """セグメント列のテンプレートインデックス（重複排除: set）。"""
    idx = defaultdict(set)
    for st in segment_tuples:
        for i in range(len(st)):
            idx[st[:i] + (SENTINEL,) + st[i + 1:]].add(st)
    return idx


def _compute_segment_pns(segment_tuple, idx):
    """セグメントレベルの音韻近傍サイズ。1セグメント置換で到達可能な語の数。"""
    nb = set()
    for i in range(len(segment_tuple)):
        nb.update(idx.get(
            segment_tuple[:i] + (SENTINEL,) + segment_tuple[i + 1:], set()))
    nb.discard(segment_tuple)
    return len(nb)


# ╔═══════════════════════════════════════════════════════════╗
# ║  4. OLD20 / PLD20（Levenshtein距離ベース）                  ║
# ╚═══════════════════════════════════════════════════════════╝

def _compute_old20(target, lexicon_list, k=20):
    """平均 orthographic Levenshtein distance to k nearest neighbors."""
    dists = sorted(_levenshtein(target, w)
                   for w in lexicon_list if w != target)
    if len(dists) >= k:
        return sum(dists[:k]) / k
    elif dists:
        return sum(dists) / len(dists)
    else:
        return float('nan')


def _segment_levenshtein(a: tuple, b: tuple) -> int:
    """セグメント tuple 間の Levenshtein 距離。"""
    n, m = len(a), len(b)
    dp = list(range(m + 1))
    for i in range(1, n + 1):
        prev, dp[0] = dp[0], i
        for j in range(1, m + 1):
            cost = 0 if a[i - 1] == b[j - 1] else 1
            dp[j], prev = min(dp[j] + 1, dp[j - 1] + 1, prev + cost), dp[j]
    return dp[m]


def _compute_pld20(target_segments: tuple, segment_lexicon: list,
                   k: int = 20) -> float:
    """Phonological Levenshtein Distance 20.

    セグメント列（訓令式音素列）間の平均 Levenshtein 距離を
    最も近い k 語について算出する。
    """
    dists = sorted(_segment_levenshtein(target_segments, s)
                   for s in segment_lexicon if s != target_segments)
    if len(dists) >= k:
        return sum(dists[:k]) / k
    elif dists:
        return sum(dists) / len(dists)
    else:
        return float('nan')


# ╔═══════════════════════════════════════════════════════════╗
# ║  5. EM-based consistency（漢字–読み対応）                    ║
# ║     Source: EM alignment v4 (artifact 1e7534f4)           ║
# ╚═══════════════════════════════════════════════════════════╝

def _log_sum_exp(vals: List[float]) -> float:
    if not vals:
        return -float('inf')
    mx = max(vals)
    if mx == -float('inf'):
        return -float('inf')
    return mx + math.log(sum(math.exp(v - mx) for v in vals))


def _entropy(probs: List[float]) -> float:
    return -sum(p * math.log(p + 1e-30) for p in probs if p > 0)


def _confidence(probs: List[float]) -> float:
    n = len(probs)
    if n <= 1:
        return 1.0
    H = _entropy(probs)
    H_max = math.log(n)
    return max(0.0, 1.0 - H / H_max) if H_max > 1e-30 else 1.0


def _enumerate_segmentations(M: int, N: int) -> List[Tuple[int, ...]]:
    if N > M or N < 1:
        return []
    if N == 1:
        return [()]
    return [sp for sp in combinations(range(1, M), N - 1)]


def _splits_to_segments(M: int, splits: Tuple[int, ...]) -> List[Tuple[int, int]]:
    boundaries = [0] + list(splits) + [M]
    return [(boundaries[i], boundaries[i + 1])
            for i in range(len(boundaries) - 1)]


# ── Mora segmentation (original: artifact 1e7534f4) ──

SMALL_KANA = set(list("ゃゅょぁぃぅぇぉャュョァィゥェォ"))
LONG_VOWEL = "ー"
SOKUON = {"っ", "ッ"}
HATSUON = {"ん", "ン"}


def kana_to_mora(reading: str) -> List[str]:
    """カタカナ/ひらがな混在の読みをモーラ単位に分解する。"""
    morae: List[str] = []
    i = 0
    while i < len(reading):
        ch = reading[i]
        if ch in SOKUON or ch in HATSUON:
            morae.append(ch)
            i += 1
        elif i + 1 < len(reading) and reading[i + 1] in SMALL_KANA:
            morae.append(ch + reading[i + 1])
            i += 2
        elif ch == LONG_VOWEL:
            morae.append(ch)
            i += 1
        else:
            morae.append(ch)
            i += 1
    return morae


# ── EM alignment 本体 ──

def _build_entries(data: List[Tuple[str, str]]):
    """EM学習用のエントリを構築する。"""
    entries = []
    for word, yomi in data:
        kanji_chars = list(word)
        N = len(kanji_chars)
        morae = kana_to_mora(yomi)
        M = len(morae)
        if M < N:
            continue
        segs = _enumerate_segmentations(M, N)
        if not segs:
            continue
        entries.append((word, yomi, kanji_chars, morae, segs))
    return entries


def em_align(
    data: List[Tuple[str, str]],
    n_iter: int = 20,
    smoothing: float = 1e-8,
    freq_weights: Optional[Dict[str, int]] = None,
    use_confidence_weighting: bool = True,
) -> Tuple[
    Dict[str, Counter],
    Dict[Tuple[str, str], Tuple[Tuple[str, ...], ...]],
    Dict[Tuple[str, str], float],
    Dict[Tuple[str, str], float],
]:
    """EM alignment.

    Returns
    -------
    kanji_dist : Dict[str, Counter]
        P(reading|kanji) を表す分布。
    alignments : Dict[(word, yomi), Tuple[Tuple[str,...], ...]]
        MAP アライメント。
    confidences : Dict[(word, yomi), float]
        各語のアライメント信頼度。
    align_logliks : Dict[(word, yomi), float]
        各語のアライメント対数尤度（漢字あたり平均）。
    """
    entries = _build_entries(data)

    if not entries:
        return {}, {}, {}, {}

    kanji_dist = defaultdict(Counter)

    def _log_emit(kanji, reading):
        total = sum(kanji_dist[kanji].values()) + smoothing
        count = kanji_dist[kanji].get(reading, 0) + smoothing
        return math.log(count / total)

    def _seg_log_prob(kanji_chars, morae, splits):
        segments = _splits_to_segments(len(morae), splits)
        return sum(_log_emit(kanji_chars[i], tuple(morae[a:b]))
                   for i, (a, b) in enumerate(segments))

    # Initialize uniform
    for word, yomi, kanji_chars, morae, segs in entries:
        w = freq_weights.get(word, 1) if freq_weights else 1
        uniform_w = w / len(segs)
        for splits in segs:
            segments = _splits_to_segments(len(morae), splits)
            for i, (a, b) in enumerate(segments):
                kanji_dist[kanji_chars[i]][tuple(morae[a:b])] += uniform_w

    # EM iterations
    for _ in range(n_iter):
        new_dist = defaultdict(Counter)
        for word, yomi, kanji_chars, morae, segs in entries:
            w = freq_weights.get(word, 1) if freq_weights else 1
            log_probs = [_seg_log_prob(kanji_chars, morae, sp) for sp in segs]
            lse = _log_sum_exp(log_probs)
            posteriors = [math.exp(lp - lse) for lp in log_probs]
            conf = _confidence(posteriors) if use_confidence_weighting else 1.0
            for sp, post in zip(segs, posteriors):
                segments = _splits_to_segments(len(morae), sp)
                for i, (a, b) in enumerate(segments):
                    new_dist[kanji_chars[i]][tuple(morae[a:b])] += (
                        post * w * conf)
        kanji_dist = new_dist

    # MAP alignment, confidences, align_logliks
    alignments = {}
    confidences = {}
    align_logliks = {}
    for word, yomi, kanji_chars, morae, segs in entries:
        log_probs = [_seg_log_prob(kanji_chars, morae, sp) for sp in segs]
        lse = _log_sum_exp(log_probs)
        posteriors = [math.exp(lp - lse) for lp in log_probs]
        best_idx = max(range(len(segs)), key=lambda i: log_probs[i])
        best_segments = _splits_to_segments(len(morae), segs[best_idx])
        key = (word, yomi)
        alignments[key] = tuple(tuple(morae[a:b]) for a, b in best_segments)
        confidences[key] = _confidence(posteriors)
        align_logliks[key] = log_probs[best_idx] / len(kanji_chars)

    return dict(kanji_dist), alignments, confidences, align_logliks


# ── Kanji-level statistics ──

class KanjiStats(NamedTuple):
    """Per-kanji reading statistics."""
    consistency: float
    entropy: float
    n_readings: int
    top_reading: Tuple[str, ...]
    top_prob: float


def kanji_stats(kanji: str, kanji_dist: Dict[str, Counter],
                min_prob: float = 0.01) -> Optional[KanjiStats]:
    """Compute reading statistics for a single kanji."""
    if kanji not in kanji_dist:
        return None
    cnt = kanji_dist[kanji]
    total = sum(cnt.values())
    if total <= 0:
        return None
    probs = {r: c / total for r, c in cnt.items()}
    top_reading = max(probs, key=probs.get)
    top_prob = probs[top_reading]
    prob_list = list(probs.values())
    return KanjiStats(
        consistency=top_prob,
        entropy=_entropy(prob_list),
        n_readings=sum(1 for p in prob_list if p >= min_prob),
        top_reading=top_reading,
        top_prob=top_prob,
    )


def build_consistency_from_dist(kanji_dist: Dict[str, Counter]) -> Dict[str, float]:
    consistency = {}
    for kanji, cnt in kanji_dist.items():
        total = sum(cnt.values())
        consistency[kanji] = (max(cnt.values()) / total) if total > 0 else 0.0
    return consistency


def _kanji_consistency(kanji, kanji_dist):
    """consistency = max P(reading|kanji)。"""
    if kanji not in kanji_dist:
        return float('nan'), float('nan')
    cnt = kanji_dist[kanji]
    total = sum(cnt.values())
    if total <= 0:
        return float('nan'), float('nan')
    probs = [c / total for c in cnt.values()]
    return max(probs), _entropy(probs)


def word_consistency(word, kanji_dist):
    """語全体の consistency = 構成漢字の consistency の平均。"""
    cons_list = []
    ent_list = []
    for ch in word:
        c, h = _kanji_consistency(ch, kanji_dist)
        if not math.isnan(c):
            cons_list.append(c)
            ent_list.append(h)
    if not cons_list:
        return float('nan'), float('nan')
    return (sum(cons_list) / len(cons_list),
            sum(ent_list) / len(ent_list))


# ── Nonword reading prediction（sub-lexical route approximation）──

class NonwordReading(NamedTuple):
    """Prediction result for a single nonword."""
    reading: str
    per_kanji: Tuple[Tuple[str, ...], ...]
    joint_prob: float
    joint_logprob: float
    per_kanji_consistency: Tuple[float, ...]
    per_kanji_entropy: Tuple[float, ...]
    word_consistency: float
    word_entropy: float


def predict_nonword(nonword: str, kanji_dist: Dict[str, Counter],
                    smoothing: float = 1e-8) -> NonwordReading:
    """Predict the most likely reading of a nonword using per-kanji
    reading distributions learned by EM.

    Approximates the CDP sub-lexical (GPC) route:
      - Each kanji independently activates its reading distribution
      - The argmax reading per kanji is selected
      - No lexical feedback, no inter-kanji context
    """
    per_kanji_readings = []
    per_kanji_probs = []
    per_kanji_cons = []
    per_kanji_ent = []

    for k in nonword:
        cnt = kanji_dist.get(k, Counter())
        total = sum(cnt.values())
        if total <= 0:
            per_kanji_readings.append(("？",))
            per_kanji_probs.append(smoothing)
            per_kanji_cons.append(0.0)
            per_kanji_ent.append(0.0)
            continue
        probs = {r: c / total for r, c in cnt.items()}
        top_r = max(probs, key=probs.get)
        top_p = probs[top_r]
        prob_list = list(probs.values())
        per_kanji_readings.append(top_r)
        per_kanji_probs.append(top_p)
        per_kanji_cons.append(top_p)
        per_kanji_ent.append(_entropy(prob_list))

    joint_logp = sum(math.log(p + 1e-30) for p in per_kanji_probs)
    joint_p = math.exp(joint_logp)
    reading_str = "".join("".join(r) for r in per_kanji_readings)
    n = len(nonword)
    return NonwordReading(
        reading=reading_str,
        per_kanji=tuple(per_kanji_readings),
        joint_prob=joint_p,
        joint_logprob=joint_logp,
        per_kanji_consistency=tuple(per_kanji_cons),
        per_kanji_entropy=tuple(per_kanji_ent),
        word_consistency=sum(per_kanji_cons) / n if n > 0 else 0.0,
        word_entropy=sum(per_kanji_ent) / n if n > 0 else 0.0,
    )


def predict_nonword_topn(nonword: str, kanji_dist: Dict[str, Counter],
                         top_n: int = 5
                         ) -> List[Tuple[str, Tuple[Tuple[str, ...], ...], float]]:
    """Return top-N reading candidates with joint probabilities.

    Generates candidates from the Cartesian product of per-kanji top readings.
    Useful for modeling reading competition (activation overlap in GPC output).

    Returns
    -------
    list of (reading_str, per_kanji_readings, joint_prob)
        sorted by joint_prob descending.
    """
    per_kanji_tops = []
    for k in nonword:
        cnt = kanji_dist.get(k, Counter())
        total = sum(cnt.values())
        if total <= 0:
            per_kanji_tops.append([(("？",), 1.0)])
            continue
        ranked = [(r, c / total) for r, c in cnt.most_common(top_n)]
        per_kanji_tops.append(ranked)

    candidates = []
    for combo in product(*per_kanji_tops):
        readings, probs = zip(*combo)
        joint_p = 1.0
        for p in probs:
            joint_p *= p
        reading_str = "".join("".join(r) for r in readings)
        candidates.append((reading_str, readings, joint_p))

    candidates.sort(key=lambda x: -x[2])
    return candidates[:top_n]


# ── Inspection / display utilities ──

def show_kanji_readings(kanji: str, kanji_dist: Dict[str, Counter],
                        top_n: int = 10):
    """漢字の読み分布を表示する。"""
    if kanji not in kanji_dist:
        print(f"'{kanji}' not found.")
        return
    cnt = kanji_dist[kanji]
    total = sum(cnt.values())
    ranked = cnt.most_common(top_n)
    print(f"漢字 '{kanji}' — 読み分布 (total={total:.1f}):")
    for mora_tuple, count in ranked:
        reading_str = "".join(mora_tuple)
        pct = 100 * count / total
        print(f"  {reading_str:8s}  {count:8.1f}  ({pct:5.1f}%)")


def show_alignment(word, yomi, alignments, confidences=None,
                   align_logliks=None):
    """アライメント結果を表示する。"""
    key = (word, yomi)
    if key not in alignments:
        print(f"({word}, {yomi}) not found.")
        return
    segs = alignments[key]
    parts = [f"{c}→{''.join(r)}" for c, r in zip(word, segs)]
    extras = []
    if confidences and key in confidences:
        extras.append(f"conf={confidences[key]:.3f}")
    if align_logliks and key in align_logliks:
        extras.append(f"loglik={align_logliks[key]:.3f}")
    extra_str = "  " + "  ".join(extras) if extras else ""
    print(f"{'  '.join(parts)}  ({word}/{yomi}){extra_str}")


def show_nonword_prediction(nonword: str, kanji_dist: Dict[str, Counter]):
    """非単語の読み予測結果を表示する。"""
    result = predict_nonword(nonword, kanji_dist)
    print(f"非単語: {nonword}")
    print(f"  最尤読み: {result.reading}")
    print(f"  漢字別:")
    for i, k in enumerate(nonword):
        r = "".join(result.per_kanji[i])
        c = result.per_kanji_consistency[i]
        h = result.per_kanji_entropy[i]
        print(f"    {k} → {r:6s}  consistency={c:.3f}  entropy={h:.3f}")
    print(f"  結合確率:     {result.joint_prob:.6f}")
    print(f"  結合対数尤度: {result.joint_logprob:.3f}")
    print(f"  語一貫性:     {result.word_consistency:.3f}")
    print(f"  語エントロピー: {result.word_entropy:.3f}")

    topn = predict_nonword_topn(nonword, kanji_dist, top_n=5)
    print(f"  読み候補:")
    for rank, (reading, per_k, jp) in enumerate(topn, 1):
        per_k_str = " + ".join("".join(r) for r in per_k)
        print(f"    {rank}. {reading:10s} ({per_k_str})  P={jp:.6f}")
    print()


# ╔═══════════════════════════════════════════════════════════╗
# ║  6a. データ読み込みユーティリティ                              ║
# ╚═══════════════════════════════════════════════════════════╝

def load_psylex71(path: str) -> Tuple[List[Tuple[str, str]], Dict[str, int]]:
    """psylex71.txt を読み込み、(単語, 読み) リストと頻度辞書を返す。

    半角スペース区切り, Shift-JIS, ヘッダなし。
    列0: 共通ID, 列1: 独自ID, 列2: 単語, 列3: 読み, 列4: 品詞, 列5: 全体頻度。
    読みがカタカナのみのレコードに限定する。

    Parameters
    ----------
    path : str
        psylex71.txt のパス。

    Returns
    -------
    pairs : List[Tuple[str, str]]
        (単語, 読み) のリスト。読みはカタカナのまま（正規化は Engine 側で行う）。
    freq_weights : Dict[str, int]
        単語→全体頻度の辞書。
    """
    df = pd.read_csv(
        path, sep=r'\s+', header=None, encoding='shift-jis',
        usecols=[0, 1, 2, 3, 4, 5],
        names=['共通ID', '独自ID', '単語', '読み', '品詞', '全体頻度'],
    )
    # カタカナ（＋長音符号ー）のみの読みを持つレコードに限定
    df = df[df['読み'].apply(
        lambda y: isinstance(y, str) and len(y) > 0 and all(
            ('\u30A0' <= c <= '\u30FF') or c == 'ー' for c in y))]
    pairs = list(zip(df['単語'].astype(str), df['読み'].astype(str)))
    freq_weights = dict(zip(df['単語'].astype(str),
                            df['全体頻度'].astype(int)))
    return pairs, freq_weights


def _validate_pairs(pairs: List[Tuple[str, str]]) -> List[Tuple[str, str]]:
    """(単語, 読み) リストを検証し、不正なエントリを除外する。"""
    valid = []
    for item in pairs:
        if (isinstance(item, (tuple, list)) and len(item) >= 2
                and isinstance(item[0], str) and isinstance(item[1], str)
                and len(item[0]) > 0 and len(item[1]) > 0):
            valid.append((item[0], item[1]))
    return valid


# ╔═══════════════════════════════════════════════════════════╗
# ║  6b. 統合エンジン                                          ║
# ╚═══════════════════════════════════════════════════════════╝

WLSP_COL_NAMES = [
    'レコードID番号', '見出し番号', 'レコード種別', '類', '部門',
    '中項目', '分類項目', '分類番号', '段落番号', '小段落番号', '語番号',
    '見出し', '見出し本体', '読み', '逆読み'
]


class WordMetrics(NamedTuple):
    """計算結果。"""
    word: str               # 入力語
    reading_normalized: str  # WLSP方式に正規化された読み
    n_chars: int            # 文字数
    n_morae: int            # モーラ数
    ons: int                # Orthographic Neighborhood Size
    pns: int                # Phonological Neighborhood Size（モーラレベル）
    pns_segment: int        # Phonological Neighborhood Size（訓令式モーラレベル）
    old20: float            # Orthographic Levenshtein Distance 20
    pld20: float            # Phonological Levenshtein Distance 20（セグメントレベル）
    summed_neighbor_freq: int  # 書記的隣接語の頻度総和
    consistency: float      # max P(reading|kanji) の語平均
    reading_entropy: float  # H(P(reading|kanji)) の語平均


class PsychoLingEngine:
    """日本語の心理言語学変数計算エンジン。

    3つの初期化方法を提供する:

    (A) 引数なし（デフォルト）:
        engine = PsychoLingEngine()
        カレントディレクトリに psylex71.txt があれば母集団として使用。
        なければ WLSP (bunruidb.txt) にフォールバック。

    (B) WLSP パスを明示的に指定:
        engine = PsychoLingEngine(wlsp_path='bunruidb.txt')

    (C) 任意の (単語, 読み) リストを指定:
        pairs, freq = load_psylex71('psylex71.txt')
        engine = PsychoLingEngine(pairs=pairs, freq_weights=freq,
                                  corpus_name='psylex71')

    (B) の場合、WLSP 固有の前処理（B種除外, 記号除外, データ修正, NFKC）を
    自動的に適用する。
    (C) の場合、前処理は呼び出し側の責任。読みの正規化（長音→母音, カタカナ
    →ひらがな）は内部で自動的に行われる。

    ONS/PNS は重複排除（unique）でカウントする。

    Parameters
    ----------
    wlsp_path : str, optional
        bunruidb.txt のパス。pairs と排他。
    pairs : list of (str, str), optional
        (単語, 読み) のリスト。wlsp_path と排他。
    freq_weights : dict, optional
        単語→頻度の辞書。EM 学習の頻度重みとして使用。
    corpus_name : str
        母集団の名前。出力の記録用。
    em_n_iter : int
        EM alignment の反復回数。
    em_max_chars : int
        EM 学習対象とする語の最大文字数。
    """

    def __init__(self, wlsp_path: Optional[str] = None, *,
                 pairs: Optional[List[Tuple[str, str]]] = None,
                 freq_weights: Optional[Dict[str, int]] = None,
                 corpus_name: Optional[str] = None,
                 em_n_iter: int = 20,
                 em_max_chars: int = 4):
        if wlsp_path is not None and pairs is not None:
            raise ValueError("wlsp_path と pairs は排他。どちらか一方のみ指定。")

        if wlsp_path is None and pairs is None:
            # デフォルト: psylex71.txt → bunruidb.txt → ダウンロード
            import os
            if os.path.exists('psylex71.txt'):
                print("psylex71.txt を検出。母集団として使用します。")
                pairs, freq_weights = load_psylex71('psylex71.txt')
                if corpus_name is None:
                    corpus_name = 'psylex71'
            else:
                if not os.path.exists('bunruidb.txt'):
                    print("psylex71.txt, bunruidb.txt が見つかりません。"
                          "bunruidb.txt をダウンロードします...")
                    import urllib.request
                    _WLSP_URL = ("https://raw.githubusercontent.com/"
                                 "masayu-a/WLSP/master/bunruidb.txt")
                    try:
                        urllib.request.urlretrieve(_WLSP_URL, 'bunruidb.txt')
                        print("ダウンロードが正常に完了しました。")
                    except Exception as e:
                        raise RuntimeError(
                            f"bunruidb.txt のダウンロードに失敗しました: {e}\n"
                            "wlsp_path または pairs を明示的に指定してください。"
                        ) from e
                else:
                    print("psylex71.txt が見つかりません。"
                          "WLSP にフォールバックします。")
                wlsp_path = 'bunruidb.txt'
                if corpus_name is None:
                    corpus_name = 'WLSP'

        if corpus_name is None:
            corpus_name = 'custom'
        self.corpus_name = corpus_name

        if wlsp_path is not None:
            words, yomis = self._init_from_wlsp(wlsp_path)
        else:
            words, yomis = self._init_from_pairs(pairs)

        self._build_indices(words, yomis, em_n_iter, em_max_chars,
                            freq_weights)

    # ── WLSP 初期化パス ──

    def _init_from_wlsp(self, wlsp_path: str):
        """WLSP 固有の前処理を適用し、(単語リスト, 読みリスト) を返す。"""
        print(f"[1/8] WLSP 読み込み...")
        raw_df = pd.read_csv(
            wlsp_path, header=None, encoding='shift-jis',
            names=WLSP_COL_NAMES)
        print(f"       {len(raw_df)} レコード")

        # ── レコード除外 ──
        print("[2/8] レコード除外...")
        df = raw_df.copy()
        n0 = len(df)
        df = df[df['レコード種別'] != 'B']
        df = df[df['見出し本体'] != '＊']
        df = df[~df['見出し本体'].str.startswith('…', na=False)]
        df = df[~(df['見出し本体'].str.startswith('−', na=False) |
                  df['見出し本体'].str.endswith('−', na=False))]
        df = df[~df['見出し本体'].str.contains('〓', na=False)]
        print(f"       {n0} → {len(df)} レコード（{n0 - len(df)} 件除外）")

        # ── データ修正 ──
        df = df.copy()
        _YOMI_FIXES = {
            'ウォッカ': 'うぉっか',
            'だけど':   'だけど',
        }
        for midashi, correct_yomi in _YOMI_FIXES.items():
            mask = df['見出し本体'] == midashi
            if mask.any():
                df.loc[mask, '読み'] = correct_yomi

        # ── NFKC 正規化 ──
        print("[3/8] NFKC 正規化...")
        df['見出し本体'] = df['見出し本体'].apply(
            lambda x: unicodedata.normalize('NFKC', x)
            if isinstance(x, str) else x)
        df['読み'] = df['読み'].apply(
            lambda x: unicodedata.normalize('NFKC', x)
            if isinstance(x, str) else x)

        self.wlsp_df = df
        words = df['見出し本体'].dropna().tolist()
        yomis = df['読み'].dropna().tolist()
        return words, yomis

    # ── 汎用初期化パス ──

    def _init_from_pairs(self, pairs: List[Tuple[str, str]]):
        """任意の (単語, 読み) リストから初期化する。"""
        pairs = _validate_pairs(pairs)
        print(f"[1/8] データ読み込み（{self.corpus_name}）...")
        print(f"       {len(pairs)} ペア")

        # NFKC 正規化
        print("[2/8] NFKC 正規化...")
        words = [unicodedata.normalize('NFKC', w) for w, _ in pairs]
        yomis = [unicodedata.normalize('NFKC', y) for _, y in pairs]

        # 読みの正規化（長音→母音, カタカナ→ひらがな）
        print("[3/8] 読み正規化...")
        yomis = [normalize_reading(y) for y in yomis]

        self.wlsp_df = None  # WLSP固有のDataFrameは持たない
        return words, yomis

    # ── 共通のインデックス構築 + EM学習 ──

    def _build_indices(self, words, yomis, em_n_iter, em_max_chars,
                       freq_weights):
        """ONS/PNS インデックス構築と EM 学習。WLSP/汎用の両パスから呼ばれる。"""

        self._freq_weights = freq_weights or {}

        # ── ONS ──
        print("[4/8] ONS インデックス構築...")
        self.orth_set = set(words)
        self.orth_list = list(self.orth_set)
        self._ons_idx = _build_ons_index(self.orth_set)
        print(f"       ONS 母集団: {len(self.orth_set)} 語"
              f"（unique、{self.corpus_name}）")

        # ── PNS（モーラレベル）──
        print("[5/8] PNS インデックス構築（モーラレベル）...")
        self._orth2mora = {}
        phon_set = set()

        # 読みの正規化 + モーラ分解
        seen_words = set()
        for w, y in zip(words, yomis):
            yc = primary_yomi(y)
            yn = normalize_reading(yc) if yc == y else yc
            mt = tokenize_mora(yn)
            if mt:
                phon_set.add(mt)
                if w not in seen_words:
                    self._orth2mora[w] = mt
                    seen_words.add(w)

        self._pns_idx = _build_pns_index(phon_set)
        print(f"       PNS 母集団: {len(phon_set)} 語（unique 読み）")

        # ── PNS（セグメントレベル、訓令式）──
        print("[6/8] PNS インデックス構築（セグメントレベル）...")
        self._orth2segments = {}
        segment_set = set()

        if _has_jamorasep:
            seen_words_seg = set()
            for w, y in zip(words, yomis):
                yc = primary_yomi(y)
                st = _reading_to_segments(yc)
                if st:
                    segment_set.add(st)
                    if w not in seen_words_seg:
                        self._orth2segments[w] = st
                        seen_words_seg.add(w)
            self._segment_pns_idx = _build_segment_pns_index(segment_set)
            self._segment_list = list(segment_set)
            print(f"       セグメント母集団: {len(segment_set)} 語（unique）")
        else:
            self._segment_pns_idx = {}
            self._segment_list = []
            print("       jamorasep 未インストール — スキップ")

        # ── EM alignment ──
        print("[7/8] EM alignment（consistency 学習）...")
        em_data = []
        em_freq = {}
        for w, y in zip(words, yomis):
            yc = primary_yomi(y) if '・' in y else y
            if (any('\u4E00' <= c <= '\u9FFF' for c in w)
                    and all(('\u4E00' <= c <= '\u9FFF')
                            or ('\u3040' <= c <= '\u309F') for c in w)
                    and 2 <= len(w) <= em_max_chars):
                em_data.append((w, yc))
                if freq_weights and w in freq_weights:
                    em_freq[w] = freq_weights[w]

        print(f"       EM 学習対象: {len(em_data)} 語")
        (self._kanji_dist, self._alignments,
         self._confidences, self._align_logliks) = em_align(
            em_data, n_iter=em_n_iter,
            freq_weights=em_freq if em_freq else None)
        print(f"       学習済み漢字: {len(self._kanji_dist)} 字")

        print("[8/8] 準備完了")

    @property
    def kanji_dist(self) -> Dict[str, Counter]:
        """学習済み P(reading|kanji) 分布への直接アクセス。"""
        return self._kanji_dist

    @property
    def alignments(self):
        return self._alignments

    @property
    def confidences(self):
        return self._confidences

    @property
    def align_logliks(self):
        return self._align_logliks

    def compute(self, word: str, reading: str) -> WordMetrics:
        """単一の (単語, 読み) ペアの心理言語学変数を計算する。"""
        yomi = normalize_reading(reading)
        mora = tokenize_mora(yomi)

        # ONS + 隣接語頻度
        ons_neighbors = self._compute_ons_neighbors(word)
        ons = len(ons_neighbors)
        summed_nf = sum(self._freq_weights.get(n, 0) for n in ons_neighbors)

        # PNS（モーラレベル）
        pns = _compute_pns(mora, self._pns_idx) if mora else 0

        # PNS（セグメントレベル）+ PLD20
        segments = _reading_to_segments(yomi)
        if segments and self._segment_pns_idx:
            pns_seg = _compute_segment_pns(segments, self._segment_pns_idx)
            pld20 = _compute_pld20(segments, self._segment_list)
        else:
            pns_seg = 0
            pld20 = float('nan')

        # OLD20
        old20 = _compute_old20(word, self.orth_list)

        # Consistency
        cons, ent = word_consistency(word, self._kanji_dist)

        return WordMetrics(
            word=word, reading_normalized=yomi,
            n_chars=len(word), n_morae=len(mora),
            ons=ons, pns=pns, pns_segment=pns_seg,
            old20=old20, pld20=pld20,
            summed_neighbor_freq=summed_nf,
            consistency=cons, reading_entropy=ent,
        )

    def _compute_ons_neighbors(self, word: str) -> set:
        """書記的隣接語の集合を返す（ONS計算と頻度集計の両方で使う）。"""
        nb = set()
        for i in range(len(word)):
            nb.update(self._ons_idx.get(
                word[:i] + SENTINEL + word[i + 1:], set()))
        nb.discard(word)
        return nb

    def neighbor_freq_stats(self, word: str) -> dict:
        """書記的隣接語の頻度統計を返す。

        Returns
        -------
        dict with keys:
            ons: int — 隣接語数
            neighbors: list of (word, freq) — 隣接語と頻度のペア（頻度降順）
            summed_freq: int — 頻度総和
            max_freq: int — 最大頻度
            mean_freq: float — 平均頻度
            log1p_summed_freq: float — log(1 + summed_freq)
        """
        nb = self._compute_ons_neighbors(word)
        nb_with_freq = [(n, self._freq_weights.get(n, 0)) for n in nb]
        nb_with_freq.sort(key=lambda x: -x[1])
        freqs = [f for _, f in nb_with_freq if f > 0]
        summed = sum(freqs)
        return {
            'ons': len(nb),
            'neighbors': nb_with_freq,
            'summed_freq': summed,
            'max_freq': max(freqs) if freqs else 0,
            'mean_freq': np.mean(freqs) if freqs else 0.0,
            'log1p_summed_freq': math.log1p(summed),
        }

    def compute_batch(self, pairs: List[Tuple[str, str]],
                      verbose: bool = True) -> pd.DataFrame:
        """複数の (単語, 読み) ペアを一括計算して DataFrame で返す。"""
        rows = []
        total = len(pairs)
        for i, (word, reading) in enumerate(pairs):
            if verbose and (i + 1) % 100 == 0:
                print(f"  {i+1}/{total}...")
            m = self.compute(word, reading)
            rows.append(m._asdict())
        return pd.DataFrame(rows)

    def kanji_info(self, kanji: str) -> Optional[dict]:
        """単一漢字の読み分布情報を返す。"""
        if kanji not in self._kanji_dist:
            return None
        cnt = self._kanji_dist[kanji]
        total = sum(cnt.values())
        readings = [(''.join(r), c / total)
                    for r, c in cnt.most_common(10)]
        cons, ent = _kanji_consistency(kanji, self._kanji_dist)
        return {
            'kanji': kanji,
            'consistency': cons,
            'entropy': ent,
            'n_readings': len(cnt),
            'top_readings': readings,
        }


# ╔═══════════════════════════════════════════════════════════╗
# ║  Demo                                                      ║
# ╚═══════════════════════════════════════════════════════════╝

if __name__ == '__main__':
    import os, time
    import urllib.request

    PSYLEX_FILE = "psylex71.txt"
    WLSP_FILE = "bunruidb.txt"
    WLSP_URL = ("https://raw.githubusercontent.com/masayu-a/"
                "WLSP/master/bunruidb.txt")

    if os.path.exists(PSYLEX_FILE):
        # ── psylex71.txt が存在する場合（優先）──
        print(f"'{PSYLEX_FILE}' を検出。psylex71 を母集団として使用します。")
        pairs, freq = load_psylex71(PSYLEX_FILE)
        engine = PsychoLingEngine(pairs=pairs, freq_weights=freq,
                                  corpus_name='psylex71')
    else:
        # ── psylex71.txt がなければ WLSP にフォールバック ──
        print(f"'{PSYLEX_FILE}' が見つかりません。WLSP を母集団として使用します。")
        if not os.path.exists(WLSP_FILE):
            print(f"'{WLSP_FILE}' が見つかりません。ダウンロードを開始します...")
            try:
                urllib.request.urlretrieve(WLSP_URL, WLSP_FILE)
                print("ダウンロードが正常に完了しました。")
            except urllib.error.URLError as e:
                print(f"ダウンロード中にネットワークエラーが発生しました: {e}")
                raise SystemExit(1)
            except Exception as e:
                print(f"予期せぬエラーが発生しました: {e}")
                raise SystemExit(1)
        else:
            print(f"'{WLSP_FILE}' は既にカレントディレクトリに存在します。")
        engine = PsychoLingEngine(wlsp_path=WLSP_FILE)

    # ── 心理言語学変数の計算 ──
    test_pairs = [
        ('研究', 'ケンキュウ'),  ('国語', 'コクゴ'),
        ('会社', 'カイシャ'),    ('データ', 'データ'),
        ('コース', 'コース'),    ('落とす', 'オトス'),
        ('発表', 'ハッピョウ'),  ('熱心', 'ネッシン'),
        ('生活', 'セイカツ'),    ('先生', 'センセイ'),
    ]

    print("\n" + "=" * 80)
    print(f"心理言語学変数の計算結果（母集団: {engine.corpus_name}）")
    print("=" * 80)

    t0 = time.perf_counter()
    df = engine.compute_batch(test_pairs, verbose=False)
    t1 = time.perf_counter()
    print(f"\n計算時間: {t1-t0:.2f}s ({len(test_pairs)}語)")

    display_cols = ['word', 'reading_normalized', 'n_chars', 'n_morae',
                    'ons', 'pns', 'pns_segment', 'old20', 'pld20',
                    'summed_neighbor_freq', 'consistency', 'reading_entropy']
    print(df[display_cols].to_string(index=False))

    # ── 漢字別読み分布 ──
    print("\n── 漢字別読み分布 ──")
    for k in ['生', '発', '会']:
        show_kanji_readings(k, engine.kanji_dist, top_n=5)
        st = kanji_stats(k, engine.kanji_dist)
        if st:
            print(f"  consistency={st.consistency:.3f}  "
                  f"entropy={st.entropy:.3f}  "
                  f"n_readings={st.n_readings}")
        print()

    # ── アライメント表示 ──
    print("── アライメント ──")
    for word, yomi in [('発表', 'ハッピョウ'), ('生活', 'セイカツ'),
                       ('先生', 'センセイ')]:
        show_alignment(word, yomi, engine.alignments,
                       engine.confidences, engine.align_logliks)

    # ── 非単語の読み予測 ──
    print("\n── 非単語の読み予測 ──")
    for nw in ['熱校', '発求', '活端', '生追', '会熱']:
        show_nonword_prediction(nw, engine.kanji_dist)

    # ── 隣接語頻度統計 ──
    print("── 隣接語頻度統計 ──")
    for w in ['熱心', '熱校', '研究']:
        stats = engine.neighbor_freq_stats(w)
        print(f"  {w}: ONS={stats['ons']}, "
              f"summed_freq={stats['summed_freq']}, "
              f"log1p={stats['log1p_summed_freq']:.3f}")
        for n, f in stats['neighbors'][:3]:
            print(f"    {n}: {f}")
