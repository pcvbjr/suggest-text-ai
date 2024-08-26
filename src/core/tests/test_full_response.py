from modules import full_response


def test_llm_request():
    """
    This test checks that full_response.llm_request() generates a response to combinations of conversation_history, system_prompt, and temperature. It checks that the responses are returned as non-empty strings.
    """
    
    # Setup test data
    conversation_history_list = [
            [
                {'role': 'user', 'content': "I need you to submit your findings to me by Monday, ok?"}
            ],
            [
                {'role': 'user', 'content': "I'm going to the park. You should come to."}
            ],
            [
                {'role': 'user', 'content': "It's a bit cold in here. Do you want a blanket?"}
            ],
    ]
    system_prompts = ["Respond to the user statement.", "Translate to Spanish.", "Repeat the user text exactly."]
    temperatures = [0.1, 0.3, 0.9]
    
    for history, sys_prompt, temp in zip(conversation_history_list, system_prompts, temperatures):
        # Make the request
        completion = full_response.llm_request(history, sys_prompt, temp)
        
        # Checks
        assert isinstance(completion, str)
        assert len(completion) > 0 


def test_generate_instructions():
    """
    Checks that full_response.generate_instructions() returns a list of three single-word strings.
    """
    # Setup the test data
    conversation_history = [{'role': 'user', 'content': "I need you to submit your findings to me by Monday, ok?"}]
    
    # Make the request
    adjectives = full_response.generate_instructions(conversation_history)
    
    # Checks
    assert len(adjectives) == 3
    for a in adjectives:
        assert isinstance(a, str)
        assert len(a.strip().split()) == 1
    


def test_generate_full_responses():
    """
    This test checks that full_response.generate_full_responses() generates full responses to each conversation history. It passes a 3 conversation histories, each with just a single statement (this statement is made by the opposite party, and would be transcribed by the STT program in integration.)
    This test checks that the function returns a list of responses with the same number of elements as the conversation_history list.
    It also checks that each response is a str instance, is not empty (zero-length), and does not start or end with an asterisk * (this may denote the start or end of the response by the LLM, and should be removed in the cleaning.)
    """
    conversation_history_list = [
            [
                {'role': 'user', 'content': "I need you to submit your findings to me by Monday, ok?"}
            ],
            [
                {'role': 'user', 'content': "I'm going to the park. You should come to."}
            ],
            [
                {'role': 'user', 'content': "It's a bit cold in here. Do you want a blanket?"}
            ],
    ]
    for conversation_history in conversation_history_list:

        responses, adjectives = full_response.generate_full_responses(conversation_history)
        assert len(responses) == len(conversation_history_list)
        for response in responses:
            print('response:', response)
            assert isinstance(response, str)
            assert len(response) > 0
            assert '*' not in [response[0], response[-1]]
            




if __name__ == "__main__":
    test_generate_full_responses_unit()








