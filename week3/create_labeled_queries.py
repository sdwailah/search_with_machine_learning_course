import os
import argparse
import xml.etree.ElementTree as ET
import pandas as pd
import numpy as np
import csv

# Useful if you want to perform stemming.
import nltk

stemmer = nltk.stem.PorterStemmer()

categories_file_name = r'/workspace/datasets/product_data/categories/categories_0001_abcat0010000_to_pcmcat99300050000.xml'

queries_file_name = r'/workspace/datasets/train.csv'
output_file_name = r'/workspace/datasets/labeled_query_data_1000.txt'

parser = argparse.ArgumentParser(description='Process arguments.')
general = parser.add_argument_group("general")
general.add_argument("--min_queries", default=1,  help="The minimum number of queries per category label (default is 1)")
general.add_argument("--output", default=output_file_name, help="the file to output to")

args = parser.parse_args()
output_file_name = args.output

if args.min_queries:
    min_queries = int(args.min_queries)

# The root category, named Best Buy with id cat00000, doesn't have a parent.
root_category_id = 'cat00000'

tree = ET.parse(categories_file_name)
#print(tree)
root = tree.getroot()
#print(root.text)


# Parse the category XML file to map each category id to its parent category id in a dataframe.
categories = []
parents = []
for child in root:
    id = child.find('id').text
    cat_path = child.find('path')
    cat_path_ids = [cat.find('id').text for cat in cat_path]
    leaf_id = cat_path_ids[-1]
    if leaf_id != root_category_id:
        categories.append(leaf_id)
        parents.append(cat_path_ids[-2])
parents_df = pd.DataFrame(list(zip(categories, parents)), columns =['category', 'parent'])

# Read the training data into pandas, only keeping queries with non-root categories in our category tree.
df = pd.read_csv(queries_file_name)[['category', 'query']]
df = df[df['category'].isin(categories)]

# IMPLEMENT ME: Convert queries to lowercase, and optionally implement other normalization, like stemming.
#print(df["query"])
df["query"] = df["query"].str.lower()
#print(df["query"])
df["tokens"] = df["query"].str.split()
#print(df["query"])
#print(df["tokens"])
df["stem_tokens"] = df["tokens"].apply(lambda x: [stemmer.stem(y) for y in x ])
#print(df["stem_tokens"])
df["query"] = df["stem_tokens"].str.join(" ")
#print(df)


# IMPLEMENT ME: Roll up categories to ancestors to satisfy the minimum number of queries per category.
categories_counts_df = df.groupby("category").size().reset_index(name="cat_count")
print(categories_counts_df)
df_merge = df.merge(categories_counts_df, how="left", on="category").merge(parents_df, how="left", on="category")
print(df_merge)

print("Cat count : " + str(len(categories_counts_df)))
print("Cat_count let : " + str(len(categories_counts_df.cat_count)))
#print("Cat count 2 : " + (categories_counts_df[categories_counts_df]))
print("Cat count + cat : " + str(len(categories_counts_df[categories_counts_df.cat_count < 1 ])))

num_of_subthreshold_categories = len(categories_counts_df[categories_counts_df.cat_count < min_queries])
#print("Number of Subthreshold Categories : " + str(num_of_subthreshold_categories))

while num_of_subthreshold_categories > 0:
    df_merge.loc[df_merge.cat_count < min_queries, "category"] = df_merge["parent"]
    df = df_merge[["category", "query"]]
    df = df[df.category.isin(categories)]
    categories_counts_df = df.groupby("category").size().reset_index(name="cat_count")
    df_merge = df.merge(categories_counts_df, how="left", on="category").merge(parents_df, how="left", on="category")
    num_of_subthreshold_categories = len(categories_counts_df[categories_counts_df.cat_count < min_queries])
    print(str(num_of_subthreshold_categories))

# Create labels in fastText format.
df['label'] = '__label__' + df['category']
categories_counts_df = df.groupby("category").size().reset_index(name="cat_count")
print(categories_counts_df)

# Output labeled query data as a space-separated file, making sure that every category is in the taxonomy.
df = df[df['category'].isin(categories)]
df['output'] = df['label'] + ' ' + df['query']
df[['output']].to_csv(output_file_name, header=False, sep='|', escapechar='\\', quoting=csv.QUOTE_NONE, index=False)
