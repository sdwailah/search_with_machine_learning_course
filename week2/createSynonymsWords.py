import pandas as pd
import fasttext

df = pd.read_csv("/workspace/datasets/fasttext/top_words.txt", sep=" ")
model = fasttext.load_model("/workspace/datasets/fasttext/title_model_normalize_epoch25_minCount.bin")
with open("/workspace/datasets/fasttext/top_words.txt") as top_words_file:
    words_list = top_words_file.read().splitlines()

resutl_list = []


for word in words_list:
    synonyms = []
    results = model.get_nearest_neighbors(word)
    for item in results:
        if item[0] > 0.75:
            synonyms.append(item[1])
    resutl_list.append(synonyms)

#print(resutl_list)

out_df = pd.DataFrame(resutl_list)
print(out_df)
out_df.to_csv("/workspace/datasets/fasttext/synonyms.csv", header=False, index=False)


