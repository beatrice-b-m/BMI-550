from dataclasses import dataclass
import Levenshtein
from RuleLabeler.negation import NegationChecker, sliding_window_generator
from RuleLabeler.lexicon import Lexicon
import re
import pandas as pd
import nltk

class Labeler:
    def __init__(self, base_lexicon_path, annotation_dir, TF_IDF_THRESH: float = 0.8):
        self.lexicon = Lexicon(base_lexicon_path, annotation_dir)
        self.lexicon.preprocess_tokens(self._preprocess)
        self.lexicon.refine_tokens(TF_IDF_THRESH)
        self.negation_check = NegationChecker(self._preprocess)

    def evaluate_dataframe(self, df, save_out: bool = False, LEVENSHTEIN_THRESH: float = 0.9):
        eval_list = []

        if save_out:
            out_df = df.copy()
            out_df["Symptom CUIs"] = ""
            out_df["Negation Flag"] = ""

        for i, data in df.iterrows():
            test_text = data["TEXT"]
            true_cui_list = [s for s in re.split(r'\${2,}', data["Symptom CUIs"]) if s]
            true_negation_list = [s for s in re.split(r'\${2,}', data["Negation Flag"]) if s]

            predictions = list(self.evaluate_text(test_text, THRESH=LEVENSHTEIN_THRESH))
            true_cui_list = [f"{c}-{n}" for c, n in zip(true_cui_list, true_negation_list)]
            predicted_cui_list = [f"{x.cui}-{int(x.negated)}" for x in predictions]

            # get metrics
            true = set(true_cui_list)
            pred = set(predicted_cui_list)
            true_pos = true.intersection(pred)
            false_pos = pred.difference(true)
            false_neg = true.difference(pred)
            f1 = f1_score(true_pos, false_pos, false_neg)
            eval_list.append(f1)

            if save_out:
                # get predicted cuis and negations
                cui_list = [x.cui for x in predictions]
                neg_list = [str(int(x.negated)) for x in predictions]

                # format output strings
                cui_str = f"$$${'$$$'.join(cui_list)}$$$"
                neg_str = f"$$${'$$$'.join(neg_list)}$$$"

                # save values to dataframe
                out_df.at[i, "Symptom CUIs"] = cui_str
                out_df.at[i, "Negation Flag"] = neg_str

            # print('f1 score:', f1)

        if save_out:
            out_df.to_csv("./predicted_symptoms.csv", index=False)

        # print('total f1 score:', sum(eval_list)/len(eval_list))

    def evaluate_text(self, text, THRESH: float = 0.8):
        text_tokens = self._preprocess([text])[0]
        # print(text_tokens)

        # we'll use sets to track our findings so we don't have to
        # evaluate duplicates
        init_findings_set = set()
        findings_set = set()

        # iterate over all symptoms
        for symptom_object in self.lexicon.symptom_dict.values():
            # print("symptom:", symptom_object.name)
            # print("cui:", symptom_object.cui)

            # retrieve list of expressions for current symptom
            symptom_expression_list = symptom_object.expression_list

            # iterate over expressions for the current symptom
            for single_expression_tokens in symptom_expression_list:

                ## FOR NOW PREPROCESS TOKENS HERE, IN REAL THING PREPROCESS
                # DURING LEXICON CONSTRUCTION !!!!!!!!!!!!

                # convert the current list of expression tokens
                # to a string for fuzzy matching
                single_expression_string = " ".join(single_expression_tokens)
                # print("\tcurrent expression:", single_expression_string)

                # use an adaptive window span to match the current token
                window_span = len(single_expression_tokens)

                window_iter = sliding_window_generator(text_tokens, window_span)
                for window, start_idx in window_iter:
                    # print("\t\tcurrent window:", window)
                    window_string = " ".join(window)
                    levenshtein_eval = Levenshtein.ratio(window_string, single_expression_string)

                    if levenshtein_eval > THRESH:
                        init_findings_set.add(InitialFinding(symptom_object.cui, start_idx, window_string, single_expression_string))

                    
                    # print("\t\tcurrent window string:", window_string)
                    # print("\t\tlevenshtein evaluation:", levenshtein_eval)

        for init_finding in init_findings_set:
            negated = self.negation_check.detect_negation(text_tokens, init_finding.start_idx)
            # print(text_tokens)
            # print(init_finding.start_idx)
            findings_set.add(
                Finding(
                    init_finding.cui, 
                    init_finding.start_idx, 
                    negated,
                    init_finding.flagged_window,
                    init_finding.expression,
                )
            )

        # print(findings_set)
        # return a sorted list of findings
        # return list(findings_set).sort(key=lambda x: x.start_idx)
        return findings_set

    def _preprocess(self, string_list):
        tokens = []

        # iterate over strings in list
        for s in string_list:
            # lowercase
            out_s = s.lower()
            # strip punctuation
            out_s = strip_punctuation(
                out_s,
                punctuation_str="][)(,'‘’:;%“”",
                replace_list=['-', '/', '\n']
            )
            # tokenize string
            split_s = nltk.tokenize.word_tokenize(out_s)

            # drop empty strings
            split_s = [s for s in split_s if s]

            # drop tokens containing numbers
            split_s = [s for s in split_s if not any(c.isdigit() for c in s)]

            # append to tokens list
            tokens.append(split_s)

        return tokens
    

def strip_punctuation(target_str: str, punctuation_str: str = "][)(,",
                      replace_list: list = []):
    """
    strip punctuation in punctuation_str from target_str and replace
    elements in replace_list with ' '
    i also reused this function from hw2/hw3
    """
    out_str = target_str.translate({ord(c): None for c in punctuation_str})
    for s in ["\n"] + replace_list:
        out_str = out_str.replace(s, " ")
    return out_str


def precision(tp, fp, epsilon: float = 1e-7):
    return len(tp) / (len(tp)+len(fp)+epsilon)

def recall(tp, fn, epsilon: float = 1e-7):
    return len(tp) / (len(tp)+len(fn)+epsilon)

def f1_score(tp, fp, fn, epsilon: float = 1e-7):
    prec = precision(tp, fp)
    rec = recall(tp, fn)
    return 2*((prec*rec)/(prec+rec+epsilon))


@dataclass(eq=True, frozen=True)
class InitialFinding:
    cui: str
    start_idx: int
    flagged_window: str
    expression: str


@dataclass(eq=True, frozen=True)
class Finding:
    cui: str
    start_idx: int
    negated: bool
    flagged_window: str
    expression: str
