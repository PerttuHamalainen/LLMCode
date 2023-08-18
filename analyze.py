import pandas as pd
import os
import sys
import llmcode
import argparse
import random

# Check that OpenAI API key defined
if os.environ.get('OPENAI_API_KEY') is None:
    print("OpenAI API key not defined. Please set the OPENAI_API_KEY environment variable.")
    exit(-1)

# Parse arguments
parser = argparse.ArgumentParser()
parser.add_argument('--input', type=str, required=True, help="Path to the input data .csv")
parser.add_argument('--input_instr', type=str, required=False, help="Path to a text file containing the coding instructions")
parser.add_argument('--pruned_code', type=str, required=False, help="A code that signals the coded text should be pruned from further analysis, e.g., UNSPECIFIED")
parser.add_argument('--column', type=str, required=True, help="Name of the data column with analyzed texts")
parser.add_argument('--output', type=str, required=True, help="Output directory name")
parser.add_argument('--coding_model', type=str, required=False, default="text-curie-001", help="LLM model for coding. Currently, only OpenAI models are supported")
parser.add_argument('--embedding_model', type=str, required=False, default="text-similarity-curie-001", help="LLM model embedding. Currently, only OpenAI models are supported")
parser.add_argument('--emb_context', type=str, required=False, default="", help="A context string that is added after each code when computing code embedding vectors")
parser.add_argument('--use_cache', type=bool, required=False, default=True, help="Whether or not to use cached results for LLM operations, which can save OpenAI API usage")
parser.add_argument('--min_group_size',type=int,required=False,default=3,help="Minimum size of code groups/themes. This controls the granularity of the grouping and the number of resulting groups.")
parser.add_argument('--grouping_dim',type=int,required=False,default=2,help="Dimensionality of the code embedding vectors used for the HDBSCAN grouping/clustering (reduced using UMAP from the original high-dimensional embeddings).")
parser.add_argument('--visualize_codes',type=bool, required=False, default=True, help="Whether or not to visualize the created codes using plotly")
parser.add_argument('--random_seed',type=int, required=False, help="Random seed. This should be set (and will be set implicitly) if cached=True, because otherwise the random shuffling of the prompt examples will always create unique OpenAI API calls that cannot be cached.")
parser.add_argument('--sep',type=str,required=False,default=",", help="The separator used in the .csv input")
parser.add_argument('--encoding',type=str,required=False,default="ISO-8859-1", help="The encoding of .csv input and output")
parser.add_argument('--dimred_method',type=str,required=False,default="UMAP", help="Dimensionality reduction method (TSNE or UMAP)")
parser.add_argument('--improve_coherence',action="store_true",help="Using this flag, the coding is informed by previously created codes, which avoids codes that are synonyms, but may inflate the frequencies of codes that get created first")
args = parser.parse_args()

# Set the random seed
if args.use_cache==True and args.random_seed is None:
    print("Caching is enabled but random seed is not defined => setting the seed to 0 to allow caching.\n")
    args.random_seed=0
if args.random_seed is not None:
    random.seed(args.random_seed)


# Helpers
input_noext,input_ext=os.path.splitext(args.input)

# Read inputs
if input_ext==".csv":
    df=pd.read_csv(args.input,sep=args.sep,encoding = args.encoding)
elif input_ext==".xlsx":
    df=pd.read_excel(args.input)
else:
    print("Unrecognized input file format. Only .csv and .xlsx supported.")
    exit()

if args.input_instr is not None:
    #read the coding instructions from text file
    coding_instruction=open(args.input_instr, "r").read()
else:
    #check if the df specifies the instructions
    if "coding_instructions" in df:
        coding_instruction=df["coding_instructions"][0]
    else:
        raise Exception("Coding instructions not specified. Please specify using --input_instr or a .csv column named \"coding_instructions\"")

print(df.head())

# Deploy llmcode
results=llmcode.code_and_group(
    df=df,
    coding_instruction=coding_instruction,
    column_to_code=args.column,
    embedding_context=args.emb_context,
    coding_model=args.coding_model,
    embedding_model=args.embedding_model,
    min_group_size=args.min_group_size,
    grouping_dim=args.grouping_dim,
    use_cache=args.use_cache,
    dimred_method=args.dimred_method,
    pruned_code=args.pruned_code,
    improve_coherence=args.improve_coherence
)

#Create output directory if needed
if not os.path.exists(args.output):
    os.mkdir(args.output)

#Save all the output dataframes
out_base_name=args.output+"/"+os.path.basename(input_noext)
results["df_coded"].to_csv(out_base_name+"_coded.csv",index=False,encoding = args.encoding)
results["df_editable"].to_csv(out_base_name+"_coded_editable.csv",index=False,encoding = args.encoding)
results["df_group_summary"].to_csv(out_base_name+"_group_summary.csv",index=False,encoding = args.encoding)
results["df_validate"].to_csv(out_base_name+"_human-gpt-comparison.csv",index=False,encoding = args.encoding)
open(out_base_name+"_prompt.txt", "w").write(results["prompt"])

# Visualize code embeddings

import plotly.express as px
import textwrap
df_vis = pd.DataFrame()
df_vis["Hover"] = results["df_editable"]["code"]
df_vis.reset_index()
for i in range(df_vis.shape[0]):
    text=results["df_editable"].loc[i,"text_1"]
    text="</br>".join(textwrap.wrap(text,width=60))
    df_vis.loc[i,"Hover"]=df_vis.loc[i,"Hover"] + "</br></br>" + "\"" + text + "\""

df_vis["Size"] = (results["df_editable"]["code_count"] / results["df_editable"]["code_count"].max()).to_list()
df_vis["x"] = results["df_editable"]["code_2d_0"]
df_vis["y"] = results["df_editable"]["code_2d_1"]
fig = px.scatter(df_vis, x="x", y="y", size="Size", hover_name="Hover", width=1000, height=1000, title="")
fig.write_html(out_base_name+"_codes_visualized.html")

