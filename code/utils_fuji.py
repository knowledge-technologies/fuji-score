from typing import List, Union, Dict, Tuple
from tqdm import trange
import numpy as np
import scipy.stats.stats as stats

MEASURE_FUJI = "fuzzy_jaccard"
MEASURE_JACCARD = "jaccard"
MEASURE_HAMMING = "hamming"
MEASURE_POG = "pog"
MEASURE_NPOG = "npog"
MEASURE_KUNCHEVA = "kuncheva"
MEASURE_WALD = "wald"
MEASURE_LUSTGARTEN = "lustgarten"
MEASURE_KRIZEK = "krizek"
MEASURE_CWREL = "cwrel"
MEASURE_PEARSON = "pearson"
MEASURE_CORRELATION = "correlation"
MEASAURE_FUZZY_GAMMA = "fuzzy_gamma"
ALLOWED_MEASURES = [
    MEASURE_FUJI, MEASURE_JACCARD, MEASURE_HAMMING, MEASURE_POG, MEASURE_NPOG,
    MEASURE_KUNCHEVA, MEASURE_WALD, MEASURE_LUSTGARTEN, MEASURE_KRIZEK,
    MEASURE_CWREL, MEASURE_PEARSON, MEASURE_CORRELATION, MEASAURE_FUZZY_GAMMA
]

STEP_1 = 1
STEP_SQUARED = "squared"
STEP_EXP = "exp"
ALLOWED_STEPS = [STEP_SQUARED, STEP_EXP]

IMPORTANCE_HANDLER_RAISE = "raise"
IMPORTANCE_HANDLER_CORRECT = "correct"
ALLOWED_IMPORTANCE_HANDLERS = [
    IMPORTANCE_HANDLER_RAISE, IMPORTANCE_HANDLER_CORRECT
]


class Fimp:
    def __init__(self, feature_dictionary: Dict[str, Tuple[int, List[int], List[float]]]):
        """
        Creates a feature importance structure (applicable to any ranked list where
        the higher score, the better the item).
        :param feature_dictionary: A dictionary of the form
         {feature name: (feature index, feature ranks, feature importance scores), ...}.
         Feature index is the index of the column in the dataset that belongs to this feature
         (1-based).

         Feature rank is 1 + the number of features that have higher importance that the feature
         (+1 for 1-based ranks)

         The structure is able to store multiple rankings at the same time and
         the i-th element of feature ranks and feature importance scores correspond to the
         rank and the score of the feature in the i-th ranking.

         For example, given a dataset of features x1, x2, x3 and their importance scores
         [1.0, 1.7, 1.5] and [3.0, 2.0, 1.0], the corresponding dictionary is
         {
            'x1': (1, [3, 1], [1.0, 3.0]),
            'x2': (2, [1, 2], [1.7, 2.0]),
            'x3': (3, [2, 3], [1.5, 1.0])
         }.
         The three components of the triplet values read as follows:
         1. Feature indices of x1, x2, x3 are 1, 2 and 3, respectively
         2. The ranks of the features in the first ranking are 3, 1, 2, respectively, i.e.,
            in the first ranking, feature x2 has the highest score (and thus rank 1),
            feature x3 has the second highest score (and thus rank 2), etc.
            The ranks of the features in the second ranking are 1, 2, 3, i.e,
            feature x1 has the highest score, feature x2 has the second highest score, etc.
         3. The feature importance scores in the first ranking are 1.0, 1.7 and 1.5.
            The feature importance scores in the second ranking are 3.0, 2.0 and 1.0.
        """
        self.table = []  # [[dataset index, name, ranks, relevances], ...]
        self.features = {}  # {name: [dataset index, ranks, relevances], ...}
        self.features = feature_dictionary
        for attr in feature_dictionary:
            row = [
                feature_dictionary[attr][0],
                attr,
                feature_dictionary[attr][1],
                feature_dictionary[attr][2]
            ]
            self.table.append(row)

    def sort_by_feature_index(self):
        self.table.sort(key=lambda row: row[0])

    def sort_by_relevance(self, ranking_index=0):
        self.table.sort(key=lambda row: row[2][ranking_index])

    def get_feature_names(self):
        return [row[1] for row in self.table]

    def get_relevances(self, ranking_index=None):
        return [
            row[-1] if ranking_index is None else row[-1][ranking_index]
            for row in self.table
        ]

    def get_relevance(self, feature_name, ranking_index=None):
        i = 0 if ranking_index is None else ranking_index
        return self.features[feature_name][-1][i]

    def set_relevances(self, ranking_index, feature_relevances):
        """Overwrites the current feature relevances. Does not recompute the ranks."""
        assert len(feature_relevances) == len(self.table)
        for i, (_, name, _, _) in enumerate(self.table):
            self.table[i][-1][ranking_index] = feature_relevances[i]
            self.features[name][-1][ranking_index] = feature_relevances[i]

    def get_rank(self, feature_name, ranking_index=None):
        i = 0 if ranking_index is None else ranking_index
        return self.features[feature_name][-2][i]

    @staticmethod
    def create_fimp_from_relevances(
            feature_relavance_scores,
            feature_names: Union[List[str], None] = None,
            feature_indices: Union[List[int], None] = None):
        """
        Creates a Fimp object from feature relevance scores,
        feature names (optionally), and feature indices in dataset (optionally).
        The object contains a single ranking.
        :param feature_relavance_scores: array-like structure of floats
        :param feature_names: array-like structure of strings (names) or None. If None,
        generic feature names (['feature1', 'feature2', ...]) are created.
        If feature_indices are given, e.g., [3, 1, ...],
         the names are created using this indices, i.e., ['feature3', 'feature1', ...]
        :param feature_indices: array-like structure of ints (feature indices) or None.
        If None, a generic indices are created [1, 2, ...].
        :return: a Fimp
        """
        n = len(feature_relavance_scores)
        if feature_indices is None:
            feature_indices = list(range(1, n + 1))
        if feature_names is None:
            feature_names = ["feature{}".format(i) for i in feature_indices]
        # compute ranks
        ranks = [-1 for _ in range(n)]
        relevances_positions = list(zip(feature_relavance_scores, range(n)))
        relevances_positions.sort(reverse=True)
        rank = 0
        for i, relevance_position in enumerate(relevances_positions):
            relevance, position = relevance_position
            if i == 0 or abs(relevance - relevances_positions[i - 1][0]) > 10 ** -12:
                rank = i + 1
            ranks[position] = rank
        d = {
            a: (i, [rank], [relevance])
            for a, i, rank, relevance in zip(feature_names, feature_indices,
                                             ranks, feature_relavance_scores)
        }
        return Fimp(feature_dictionary=d)


def compute_similarity_helper(fimp1: Fimp, fimp2: Fimp,
                              similarity_measure: str,
                              eps: float, step: Union[str, int],
                              use_tqdm: bool,
                              negative_importances_handler: str):
    """
    Computes one of the similarity measures for the two ranked lists, stored as Fimps.
    :param fimp1: the first ranked list structure
    :param fimp2: the second ranked list structure
    :param similarity_measure: an element of ALLOWED_MEASURES
    :param eps: applicable if similarity_measure is 'fuzzy_jaccard'; in that case,
        the threshold under which the scores are considered to be 0
    :param step:  applicable if similarity_measure is 'fuzzy_jaccard'; in that case,
       the number of items to add before computing the next similarity;
       The possible values are
       any positive integer s, 'squared' and 'exp': the corresponding feature subset sizes are
       - integer s: [1, 1 + s, 1 + 2s, 1 + 3s,  ...]
       - 'squared': [1, 4, 9, 16, 25, ...]
       - 'exp': [1, 2, 4, 8, 16, 32, ...]
       It is assured that the last size always equals the number of features.
    :param use_tqdm: if set to True, a progress bar is shown.
       If set to None, the progress is shown for larger numbers of features
    :param negative_importances_handler: applicable if similarity measure is 'fuzzy_jaccard';
       in that case,  either 'raise' or 'correct'; tells how to handle negative scores:
       if set to 'raise', ValueError is raised when a negative score is encountered,
       otherwise, negative values are set to 0
    :return: A list of similarities of the feature sets of the i top ranked features, where
    i ranges over the considered feature sizes.
    """

    def fuzzy_jaccard(f1: Fimp, f2: Fimp):
        def relative_score(absolute_score, normalisation_factor):
            if normalisation_factor == 0.0:
                return 1.0
            else:
                return absolute_score / normalisation_factor

        for f in [f1, f2]:
            f.sort_by_relevance()
        attributes = [f1.get_feature_names(), f2.get_feature_names()]
        n = len(attributes[0])

        for f in [f1, f2]:
            feature_relevances = f.get_relevances(0)
            if min(feature_relevances) < 0:
                if negative_importances_handler == "raise":
                    raise ValueError(
                        "Feature importances must not be negative")
                elif negative_importances_handler == "correct":
                    non_negative = [max(0, r) for r in feature_relevances]
                    f.set_relevances(0, non_negative)
                else:
                    raise ValueError(
                        "Wrong negative importances handling: {}".format(
                            negative_importances_handler))
        if isinstance(step, str):
            feature_subset_sizes = []
            if step == STEP_EXP:
                i = 1
                while i <= n:
                    feature_subset_sizes.append(i - 1)
                    i *= 2
            elif step == STEP_SQUARED:
                i = 1
                while i ** 2 <= n:
                    feature_subset_sizes.append(i ** 2 - 1)
                    i += 1
            else:
                raise ValueError("Wrong step specification: {}".format(step))
        else:
            feature_subset_sizes = list(range(0, n, step))
        if feature_subset_sizes[-1] != n - 1:
            feature_subset_sizes.append(n - 1)

        n_evaluated_subsets = len(feature_subset_sizes)
        i_subset = 0
        results = [-1.0] * n_evaluated_subsets
        min_scores = [float("inf")] * 2
        union_set = set()  # more exactly, union - intersection
        intersection_set = set()
        iterator = trange(n) if use_tqdm else range(n)
        for i in iterator:
            for j, (attributes_ranking,
                    f) in enumerate(zip(attributes, [f1, f2])):
                feature = attributes_ranking[i]
                s = f.get_relevance(feature, 0)
                min_scores[j] = min(min_scores[j], s)
                if max(min_scores) <= eps:
                    for i1 in range(i_subset, n_evaluated_subsets):
                        results[i1] = 1.0
                    return results
                if feature in union_set:
                    union_set.remove(feature)
                    intersection_set.add(feature)
                else:
                    union_set.add(feature)
            if i != feature_subset_sizes[i_subset]:
                continue
            fuzzy_intersection = len(intersection_set)
            for feature in union_set:
                s1 = min(
                    1.0,
                    relative_score(f1.get_relevance(feature, 0),
                                   min_scores[0]))
                s2 = min(
                    1.0,
                    relative_score(f2.get_relevance(feature, 0),
                                   min_scores[1]))
                fuzzy_intersection += min(s1, s2)
            fuzzy_union = len(intersection_set) + len(union_set)
            results[i_subset] = fuzzy_intersection / fuzzy_union
            i_subset += 1
        return results

    def correlation(f1: Fimp, f2: Fimp):
        for f in [f1, f2]:
            f.sort_by_feature_index()
            scores = f.get_relevances(0)
            n = len(scores)
            finite = [s for s in scores if -float("inf") < s < float("inf")]
            min_max = [min(finite), max(finite)]
            for i, s in enumerate(scores):
                if -float("inf") < s < float("inf"):
                    continue
                elif -float("inf") == s:
                    scores[i] = min_max[0]
                elif float("inf") == s:
                    scores[i] = min_max[1]
                else:
                    print("Very special value:", s, "at the position", i)
                    scores[i] = 0  # NaN
            f.set_relevances(0, scores)
        sort_fimps(f1, f2)
        results = [1.0] * n
        attributes = [f1.get_feature_names(), f2.get_feature_names()]
        part1 = []
        part2 = []
        union = set()
        for i in range(n):
            a1, a2 = attributes[0][i], attributes[1][i]
            for a in [a1, a2]:
                if a not in union:
                    union.add(a)
                    for part, f in zip([part1, part2], [f1, f2]):
                        s = f.get_relevance(a, 0)
                        part.append(s)
            coefficient, _ = stats.pearsonr(part1, part2)
            if np.isnan(coefficient):
                coefficient = 1.0
            results[i] = coefficient
        return results

    def jaccard_hamming_pog_npog_kuncheva_lustgarten_wald_krizek_cwrel_pearson(
            f1: Fimp, f2: Fimp, measure):
        sort_fimps(f1, f2)
        attributes = [f1.get_feature_names(), f2.get_feature_names()]
        n = len(attributes[0])
        results = [-1.0] * n
        intersection = set()
        union = set()
        for i in range(n):
            a1, a2 = attributes[0][i], attributes[1][i]
            if a1 == a2:
                intersection.add(a1)
                union.add(a1)
            else:
                if a1 in union:
                    # a1 has been added as part of the attributes2 before
                    intersection.add(a1)
                if a2 in union:
                    # symmetric case
                    intersection.add(a2)
                union.add(a1)
                union.add(a2)
            k = i + 1
            if measure == "jaccard":
                results[i] = len(intersection) / len(union)
            elif measure == "hamming":
                results[i] = 1.0 - ((len(union) - len(intersection)) / n)
            elif measure == "pog":
                results[i] = len(intersection) / k
            elif measure in ["npog", "kuncheva", "wald", "pearson"]:
                if k < n:
                    results[i] = (len(intersection) - k ** 2 / n) / (k -
                                                                     k ** 2 / n)
                else:
                    results[i] = 1.0
            elif measure == "lustgarten":
                if k < n:
                    results[i] = (len(intersection) -
                                  k ** 2 / n) / (k - max(0, 2 * k - n))
                else:
                    results[i] = 1.0
            elif measure == "krizek":
                # this is what krizek boils down to when we compare two feature subsets.
                results[i] = float(len(intersection) == len(union))
            elif measure == "cwrel":
                # this is what cwrel boils down to when we compare two feature subsets.
                y = n  # keep the notation from the paper to avoid mistakes
                n_capital = len(union) + len(
                    intersection)  # sum of feature subset sizes
                d = n_capital % y
                h = n_capital % 2  # 2: number of feature subsets
                numerator = y * (n_capital - d +
                                 2 * len(intersection)) - n_capital ** 2 + d ** 2
                nominator = y * (h ** 2 + 2 *
                                 (n_capital - h) - d) - n_capital ** 2 + d ** 2
                if k < n:
                    results[i] = numerator / nominator
                else:
                    results[i] = 1.0
            else:
                raise ValueError("Wrong measure: {}".format(measure))
        return results

    def fuzzy_gamma(f1: Fimp, f2: Fimp):
        # S. Henzgen, E. Hüllermeier.
        # Weighted Rank Correlation: A Flexible Approach based on Fuzzy Order Realtions.
        # ECML/PKDD 2015.
        def distance(rank1, rank2):
            if rank1 == rank2:
                return 0.0
            else:
                return 1.0  # a.k.a max(ws[min(rank1, rank2): max(rank1, rank2)])

        # def t_function(a, b):
        #     return a * b

        def r_function(rank1, rank2):
            return 0.0 if rank1 >= rank2 else distance(rank1, rank2)

        def c_d_function(feature1, feature2):
            rank11 = f1.get_rank(feature1, 0)
            rank12 = f1.get_rank(feature2, 0)
            rank21 = f2.get_rank(feature1, 0)
            rank22 = f2.get_rank(feature2, 0)
            r11_12 = r_function(rank11, rank12)
            r12_11 = r_function(rank12, rank11)
            r21_22 = r_function(rank21, rank22)
            r22_21 = r_function(rank22, rank21)
            c = r11_12 * r21_22 + r12_11 * r22_21
            d = r11_12 * r22_21 + r12_11 * r21_22
            return c, d

        f1.sort_by_relevance(0)
        f2.sort_by_relevance(0)
        attributes = [f1.get_feature_names(), f2.get_feature_names()]
        n = len(attributes[0])
        # ws = [1.0] * n  # ws[i]: distance between rank i and i + 1, i >= 0
        results = [0.0] * n
        union = set()
        c_total = 0.0
        d_total = 0.0
        iterator = trange(n) if (
                use_tqdm or use_tqdm is None and n > 1000) else range(n)
        for i in iterator:
            a1, a2 = attributes[0][i], attributes[1][i]
            new_features = set()
            for a in [a1, a2]:
                if a not in union:
                    new_features.add(a)
            # previous and one of the new
            for a2 in new_features:
                for a1 in union:
                    c_part, d_part = c_d_function(a1, a2)
                    c_total += c_part
                    d_total += d_part
            union |= new_features
            if len(new_features) == 2:
                c_part, d_part = c_d_function(attributes[0][i],
                                              attributes[1][i])
                c_total += c_part
                d_total += d_part
            numerator = c_total - d_total
            nominator = c_total + d_total
            results[i] = numerator / nominator if nominator > 0 else 1.0
        return results

    def sort_fimps(f1: Fimp, f2: Fimp):
        for f in [f1, f2]:
            f.sort_by_relevance(0)

    # sanity check
    for fimp in [fimp1, fimp2]:
        fimp.sort_by_feature_index()
    if fimp1.get_feature_names() != fimp2.get_feature_names():
        raise ValueError("Names of the attributes are not the same")
    if similarity_measure == MEASURE_FUJI:
        return fuzzy_jaccard(fimp1, fimp2)
    elif similarity_measure == MEASURE_CORRELATION:
        return correlation(fimp1, fimp2)
    elif similarity_measure in [
        MEASURE_JACCARD, MEASURE_HAMMING, MEASURE_POG, MEASURE_NPOG, MEASURE_KUNCHEVA,
        MEASURE_WALD, MEASURE_LUSTGARTEN, MEASURE_KRIZEK, MEASURE_CWREL, MEASURE_PEARSON
    ]:
        return jaccard_hamming_pog_npog_kuncheva_lustgarten_wald_krizek_cwrel_pearson(
            fimp1, fimp2, similarity_measure)
    elif similarity_measure == MEASAURE_FUZZY_GAMMA:
        return fuzzy_gamma(fimp1, fimp2)
    else:
        raise ValueError("Wrong Error measure: {}".format(similarity_measure))


def area_under_the_curve(points):
    """
    Computes area under the curve (i, points[i]), i.e., assumes that x-values are at distance 1.
    :param points:
    :return:
    """
    n = len(points) - 1
    a = 0.0
    for i in range(n):
        a += (points[i] + points[i + 1]) / 2
    return a / n
