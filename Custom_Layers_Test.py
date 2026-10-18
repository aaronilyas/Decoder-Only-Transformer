import Custom_Layers

def test_get_keys_from_dict_method():
    word_set = set()
    word_set.add("word")
    word_set.add("name")
    cp = Custom_Layers.word_embedding_layer(word_set,4)
    assert cp.get_keys_from_dict() == "word name " or cp.get_keys_from_dict() == "name word "


