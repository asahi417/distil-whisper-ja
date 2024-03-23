"""
Requirement:
pip install yakinori
pip install mecab-python3
pip install unidic-lite
"""
from yakinori import Yakinori


yakinori = Yakinori()


def text_to_pronounce(text: str):
    parsed_list = yakinori.get_parsed_list(text)
    return yakinori.get_roma_sentence(parsed_list, is_hatsuon=True)

