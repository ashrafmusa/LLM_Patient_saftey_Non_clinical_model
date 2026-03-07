from src.llm_interface import generate_response


def test_llm_placeholder():
    r = generate_response('Hello world')
    assert isinstance(r, dict)
    assert 'text' in r
