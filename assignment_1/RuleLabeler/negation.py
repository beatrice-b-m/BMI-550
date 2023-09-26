import json

class NegationChecker:
    def __init__(self, preprocessing_func):
        self.pre_negation_token_list = []
        self.post_negation_token_list = []
        self.negation_extender_list = []
        self.preprocessing_func = preprocessing_func

    def load_tokens_from_file(self):
        file_list = []
        filename_list = [
            "./assignment_1/RuleLabeler/negation_files/pre_negations.json",
            "./assignment_1/RuleLabeler/negation_files/post_negations.json",
            "./assignment_1/RuleLabeler/negation_files/negation_extenders.json"
        ]

        # iterate over filename list and load files
        for filename in filename_list:
            with open(filename, 'r') as f:
                file_list.append(json.load(f))

        # unpack list and preprocess negation strings
        pre_negations, post_negations, extenders = file_list
        self.pre_negation_token_list = self.preprocessing_func(pre_negations)
        self.post_negation_token_list = self.preprocessing_func(post_negations)
        self.negation_extender_list = extenders
        print("negations loaded from file...")

    def detect_negation(self, target_tokens, start_idx, SEARCH_SPAN: int = 4):
        # first check for pre-negations (recursively if or/nor is found)
        if self.pre_negation_search(target_tokens, start_idx, SEARCH_SPAN):
            return True
        # otherwise check for post negations
        elif self.post_negation_search(target_tokens, start_idx, SEARCH_SPAN):
            return True
        # otherwise return false
        else:
            return False

    def pre_negation_search(self, target_tokens, start_idx, SEARCH_SPAN):
        # get the min and max indexes of the token list
        token_bounds = (0, len(target_tokens) - 1)

        # offset the start idx by the search span, then
        # clip the values within the bounds of the tokens
        min_bound = clip_value(start_idx - SEARCH_SPAN, token_bounds)
        # print(f'searching {min_bound} to {start_idx}')
        search_tokens = target_tokens[min_bound:start_idx]

        # iterate over pre-negation phrases
        for negation in self.pre_negation_token_list:
            # print('\nnegation:', negation)

            # get a sliding window of the current search token sequence
            window_span = len(negation)
            window_iter = sliding_window_generator(search_tokens, window_span)
            for window, window_start_idx in window_iter:
                # print('window:', window)

                # if the current window matches a negation
                # return True
                if window == negation:
                    return True
                
                # otherwise, check if a negation extender (like or/nor) exists
                # if it does, recursively call the pre_negation search with a
                # new start idx
                elif [token for token in window if token in self.negation_extender_list]:
                    # print(f"recursing on idx {window_start_idx + min_bound}")
                    # if we find a negation, return True immediately
                    # otherwise continue searching
                    if self.pre_negation_search(target_tokens, window_start_idx + min_bound, SEARCH_SPAN):
                        return True
                    
        # if we haven't found a negation during our search, return false
        return False

    def post_negation_search(self, target_tokens, start_idx, SEARCH_SPAN):
        negated = False

        # get the min and max indexes of the token list
        token_bounds = (0, len(target_tokens) - 1)

        # offset the start idx by the search span, then 
        # clip the values within the bounds of the tokens
        # add 1 to max bound (if within range) since :max is exclusive
        max_bound = clip_value(start_idx + SEARCH_SPAN + 1, token_bounds)
        # print(f'searching {start_idx+1} to {max_bound}')
        search_tokens = target_tokens[start_idx+1:max_bound]

        # iterate over pre-negation phrases
        for negation in self.post_negation_token_list:
            # print('\nnegation:', negation)

            # get a sliding window of the current search token sequence
            window_span = len(negation)
            window_iter = sliding_window_generator(search_tokens, window_span)
            for window, _ in window_iter:
                # print('window:', window)

                # if the current window matches a negation
                # return True
                if window == negation:
                    return True

        # if we haven't found a negation during our search, return false
        return False


def sliding_window_generator(target_token_list, window_span):
    """
    function to return a sliding window iterator
    """
    n_target_tokens = len(target_token_list)

    # if the window span is wider than the tokens,
    # use the width of the tokens instead
    window_span = min(n_target_tokens, window_span)

    # iterate over possible window start indexes in the target_token_list
    for start_idx in range(n_target_tokens - window_span + 1):
        # get the end idx of the window
        end_idx = start_idx + window_span

        # retrieve the current window between the start and end indexes
        current_window = target_token_list[start_idx:end_idx]

        # return the window for this iteration
        yield current_window, start_idx

def clip_value(val, bounds):
    """
    bounds takes the form (min, max)
    function to clip value to range (like np.clip)
    """
    return max(bounds[0], min(val, bounds[1])) 

def unit_test_preprocess(string_list):
    return [s.split(" ") for s in string_list]

if __name__ == "__main__":
    test_negation_checker = NegationChecker(unit_test_preprocess)
    test_negation_checker.load_tokens_from_file()
    
    test_string_a = "I havent had a cough but Ive been sneezing."
    test_string_b = "I havent been coughing or really even sneezing at all."
    test_string_c = "A cough has yet to develop but Ive been sneezing a lot"
    test_tokens_a = test_string_a.lower().split(" ")
    test_tokens_b = test_string_b.lower().split(" ")
    test_tokens_c = test_string_c.lower().split(" ")


    print(f"\nTest A: {'-'*30}")
    print(test_string_a)
    start_idx = 4
    print(f"Negation found for {start_idx}? {test_negation_checker.detect_negation(test_tokens_a, start_idx)}")
    start_idx = 8
    print(f"Negation found for {start_idx}? {test_negation_checker.detect_negation(test_tokens_a, start_idx)}")

    print(f"\nTest B: {'-'*30}")
    print(test_string_b)
    start_idx = 3
    print(f"Negation found for {start_idx}? {test_negation_checker.detect_negation(test_tokens_b, start_idx)}")
    start_idx = 7
    print(f"Negation found for {start_idx}? {test_negation_checker.detect_negation(test_tokens_b, start_idx)}")

    print(f"\nTest C: {'-'*30}")
    print(test_string_c)
    start_idx = 1
    print(f"Negation found for {start_idx}? {test_negation_checker.detect_negation(test_tokens_c, start_idx)}")
    start_idx = 9
    print(f"Negation found for {start_idx}? {test_negation_checker.detect_negation(test_tokens_c, start_idx)}")



    
