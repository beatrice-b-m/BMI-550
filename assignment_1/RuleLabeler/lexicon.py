from dataclasses import dataclass, field
from collections import defaultdict
import re
import glob
import os
import pandas as pd

class Lexicon:
    def __init__(self, base_txt_path, annotation_dir):
        self.base_lexicon = self._load_base_lexicon(base_txt_path)
        self.symptom_dict = {}
        self._build_symptom_dict_from_base()
        self._build_symptom_dict_from_annotations(annotation_dir)

        # self.vocabulary = {}
        # self._build_vocabulary()

    def preprocess_tokens(self, preprocessing_func):
        for sympt_obj in self.symptom_dict.values():
            sympt_obj.expression_list = list(set(sympt_obj.expression_list))
            sympt_obj.expression_list = preprocessing_func(sympt_obj.expression_list)
        print("expression preprocessing complete...")

    def refine_tokens(self, TF_IDF_THRESH: float):
        total_count_dict = defaultdict(int)
        for sympt_obj in self.symptom_dict.values():
            for token_list in sympt_obj.expression_list:
                for token in token_list:
                    total_count_dict[token] += 1
                    sympt_obj.count_dict[token] += 1

        for sympt_obj in self.symptom_dict.values():
            sympt_obj.tf_idf_dict = {
                k: sympt_obj.count_dict[k]/total_count_dict[k] for k in sympt_obj.count_dict.keys()
            }
            # only keep tokens in expressions with tf_idf higher than THRESH
            sympt_obj.expression_list = [[w for w in expression if sympt_obj.tf_idf_dict[w] > TF_IDF_THRESH] for expression in sympt_obj.expression_list]
            # remove empty lists
            sympt_obj.expression_list = [expression for expression in sympt_obj.expression_list if expression]

    def _load_base_lexicon(self, lexicon_path):
        base_lexicon = pd.read_csv(lexicon_path, delimiter='\t', header=None)
        base_lexicon.columns = ['symptom', 'cui', 'expression']
        return base_lexicon

    def _build_symptom_dict_from_base(self):
        print('logging symptom expressions from base lexicon...')

        # get array of unique cuis
        cui_array = self.base_lexicon.cui.unique()

        # iterate over cui array
        for cui in cui_array:
            # get sub dataframe from the base lexicon
            symptom_df = self.base_lexicon[self.base_lexicon.cui == cui]

            # get symptom name
            symptom_name = symptom_df.symptom.unique()[0]

            # get list of unique expressions for the symptom
            expression_list = list(symptom_df.expression.unique())

            # assign symptom object to symptom dictionary with cui as key
            sympt_obj = Symptom(symptom_name, cui)
            sympt_obj.expression_list += expression_list

            # assign to symptom dict
            self.symptom_dict[cui] = sympt_obj

        # assign 'other' cui
        self.symptom_dict['C0000000'] = Symptom('other', 'C0000000')

    def _build_symptom_dict_from_annotations(self, annotation_dir):
        print('logging symptom expressions from annotation files...')
        
        annotation_file_list = glob.glob(os.path.join(annotation_dir, "*.xlsx"))

        for annotation_file_path in annotation_file_list:
            annotation_df = pd.read_excel(annotation_file_path)
            AnnotationFile(annotation_df, self.symptom_dict)

class AnnotationFile:
    def __init__(self, df, symptom_dict):
        self.df = df
        self.symptom_dict = symptom_dict
        self.extract_annotations()

    def extract_annotations(self, delim_pattern = r'\${2,}'):
        file_symptom_expression_list = self.df['Symptom Expressions']
        file_standard_symptom_list = self.df['Standard Symptom']
        file_symptom_cui_list = self.df['Symptom CUIs']
        file_negation_flag_list = self.df['Negation Flag']

        combined_file_iter = zip(
            file_symptom_expression_list, 
            file_standard_symptom_list, 
            file_symptom_cui_list, 
            file_negation_flag_list            
        )

        for expression_str, symptom_str, cui_str, negation_str in combined_file_iter:
            # strip newlines from file
            expression_str = expression_str.replace("\n", " ")
            symptom_str = symptom_str.replace("\n", " ")
            cui_str = cui_str.replace("\n", " ")
            negation_str = negation_str.replace("\n", " ")

            # split on any sequence of 2 or more '$'
            # string should be delimited by '$$$' but there
            # are some errors
            text_symptom_expression_list = [s for s in re.split(delim_pattern, expression_str) if s]
            text_standard_symptom_list = [s for s in re.split(delim_pattern, symptom_str) if s]
            text_symptom_cui_list = [s for s in re.split(delim_pattern, cui_str) if s]
            text_negation_flag_list = [s for s in re.split(delim_pattern, negation_str) if s]

            combined_text_iter = zip(
                text_symptom_expression_list,
                text_standard_symptom_list,
                text_symptom_cui_list,
                text_negation_flag_list,
            )

            for expression, symptom, cui, negation in combined_text_iter:
                # strip non-numeric characters from negation
                negation = int(re.sub(r"[^0-9]", "", negation))
                cui = cui.strip()

                # skip invalid cuis
                if len(cui) < 4:
                    continue

                # print('symptom:', symptom)
                # print('cui:', cui)
                # print('expression:', expression)
                # print('negation:', negation)
                try:
                    self._update_dict(cui, expression, negation)
                except KeyError:
                    self.symptom_dict[cui] = Symptom(symptom, cui)
                    self._update_dict(cui, expression, negation)

    def _update_dict(self, cui, expression, negation):
        # if symptom is negated, assign the expression to the neg expression list
        if negation:
            self.symptom_dict[cui].neg_expression_list += [expression]
        
        # otherwise assign the expression to the expression list
        else:
            self.symptom_dict[cui].expression_list += [expression]

def unit_test_preprocess(string_list):
    # split strings on whitespace
    split_s = [s.split(" ") for s in string_list]
    # remove any tokens containing numbers
    split_s = [[t for t in expression if not any(c.isdigit() for c in t)] for expression in split_s]
    return split_s

@dataclass
class Symptom:
    name: str
    cui: str
    expression_list: list[str] = field(default_factory=list)
    neg_expression_list: list[str] = field(default_factory=list)
    count_dict: defaultdict[int] = field(default_factory=lambda: defaultdict(int))
    tf_idf_dict: dict = field(default_factory=dict)


if __name__ == "__main__":
    test_lexicon = Lexicon('assignment_1/COVID-Twitter-Symptom-Lexicon.txt', 'assignment_1/annotations')
    test_lexicon.preprocess_tokens(unit_test_preprocess)
    # print(test_lexicon.symptom_dict)
    print(test_lexicon.refine_tokens())
    print(test_lexicon.symptom_dict)