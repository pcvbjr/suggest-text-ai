from modules import llm_suggest

def test_llm_next_token_probs():
    """
    Checks that llm_suggest.llm_next_token_probs() returns a list of (prob, token) tuples.
    """
    
    # Setup test data
    prompt_list = [
        'The only way to do great work is to ',
        'In three words I can',
        'Success is not final, failure is not',
        'The only true wisdom is in knowing you know '
    ]
    
    for prompt in prompt_list:
        token_probs_list = llm_suggest.llm_next_token_probs(prompt)
        for tok_prob, tok in token_probs_list:
            assert isinstance(tok_prob, float)
            assert tok_prob >= 0
            assert isinstance(tok, str)
            assert len(tok) > 0
            
    top_k = 5
    for prompt in prompt_list:
        token_probs_list = llm_suggest.llm_next_token_probs(prompt, top_k=5)
        assert len(token_probs_list) <= top_k + 26 # top-k limits words, but not characters, so we add 26 to account for each letter in the alphabet
        for tok_prob, tok in token_probs_list:
            assert isinstance(tok_prob, float)
            assert tok_prob >= 0
            assert isinstance(tok, str)
            assert len(tok) > 0