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
parser.add_argument('--pruned_code', type=str, required=False, help="A code that signals the coded text should be pruned from further analysis, e.g., UNSPECIFIED")
parser.add_argument('--column', type=str, required=True, help="Name of the data column with analyzed texts")
parser.add_argument('--output', type=str, required=True, help="Output directory name")
parser.add_argument('--coding_model', type=str, required=False, default="gpt-3.5-turbo-instruct", help="LLM model for coding. Currently, only OpenAI models are supported")
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
parser.add_argument('--grouping_batch_size',type=int,required=False,default=200, help="Batch size when grouping codes to themes.")
parser.add_argument('--other_threshold',type=int,required=False,default=1, help='Themes with counts smaller or equal to this get grouped to an "Other" theme.')
args = parser.parse_args()

# Set the random seed
if args.use_cache==True and args.random_seed is None:
    print("Caching is enabled but random seed is not defined => setting the seed to 0 to allow caching.\n")
    args.random_seed=0
if args.random_seed is not None:
    random.seed(args.random_seed)

# Helpers
input_noext,input_ext=os.path.splitext(args.input)

# Set LLM query cache directory
llmcode.set_cache_directory(os.path.join(llmcode.get_cache_directory(),os.path.basename(input_noext)))

# Read inputs
if input_ext==".csv":
    df=pd.read_csv(args.input,sep=args.sep,encoding = args.encoding)
elif input_ext==".xlsx":
    df=pd.read_excel(args.input)
else:
    print("Unrecognized input file format. Only .csv and .xlsx supported.")
    exit()


print(df.head())

#Create output directory if needed
if not os.path.exists(args.output):
    os.mkdir(args.output)

# Code the data
coding_results=llmcode.code_df(
    df=df,
    column_to_code=args.column,
    coding_model=args.coding_model,
    embedding_model=args.embedding_model,
    embedding_context=args.emb_context,
    dimred_method=args.dimred_method,
    dimred_neighbors=5, #a good default that seems to work for most of the time
    use_cache=args.use_cache,
    pruned_code=args.pruned_code
)

# Save coding results
out_base_name=args.output+"/"+os.path.basename(input_noext)
open(out_base_name+"_prompt.txt", "w").write(coding_results["prompt"])
coding_results["df_coded"].to_csv(out_base_name+"_coded.csv",index=False,encoding = args.encoding)
coding_results["df_validate"].to_csv(out_base_name+"_human-llm-comparison.csv",index=False,encoding = args.encoding)
if args.pruned_code is not None:
    coding_results["df_coded_pruned"].to_csv(out_base_name+"_coded_pruned.csv",index=False,encoding = args.encoding)
    coding_results["df_codes_pruned"].to_csv(out_base_name+"_codes_pruned.csv",index=False,encoding = args.encoding)

# Group codes into themes, if the df specifies instructions for it
themes_extracted=False
if "grouping_instructions" in df:
    grouping_instructions = df["grouping_instructions"][0]
    df_to_group=coding_results["df_codes"] if args.pruned_code is None else coding_results["df_codes_pruned"]
    df_grouped=llmcode.group_codes(
        df_data=df,
        df_codes=df_to_group,
        grouping_instructions=grouping_instructions,
        grouping_model=args.coding_model,
        use_cache=args.use_cache,
        batch_size=args.grouping_batch_size,
        other_threshold=args.other_threshold
    )
    df_grouped.to_csv(out_base_name+"_codes_grouped.csv",index=False,encoding = args.encoding)
    themes_extracted=True

# Visualize code embeddings
import plotly.express as px
import textwrap

# Visualize codes and themes
vis_data=df_grouped if themes_extracted else coding_results["df_codes"]
vis_data=vis_data.rename(columns={"count":"code_count"})
hover_texts=[]
theme_names=[]
other_theme_name=""
for i in range(vis_data.shape[0]):
    text=""
    if themes_extracted:
        text+=f"{vis_data.loc[i,'theme']} ({vis_data.loc[i,'theme_count']})</br>"
        theme_names.append(text)
        if vis_data.loc[i,'theme']=="Other":
            other_theme_name=text
    text+=f"Code: {vis_data.loc[i,'code']} ({vis_data.loc[i,'code_count']})</br></br>"
    text+='"' + "</br>".join(textwrap.wrap(vis_data.loc[i,'text 0'],width=60)) + '"'
    hover_texts.append(text)
df_vis = pd.DataFrame()
df_vis["Hover"]=hover_texts
df_vis["Size"] = (vis_data["code_count"] / vis_data["code_count"].max()).to_list()
df_vis["x"] = vis_data["code_2d_0"]
df_vis["y"] = vis_data["code_2d_1"]

# Create a custom colormap where "Other" theme is semitransparent gray
if themes_extracted:
    df_vis["Theme"] = theme_names
    unique_categories=df_vis['Theme'].unique().astype(str).tolist()
    if other_theme_name in unique_categories:
        unique_categories.remove(other_theme_name)
    cmap=px.colors.sample_colorscale(px.colors.sequential.Rainbow, len(unique_categories),low=0,high=1)
    #cmap = px.colors.sequential.Viridis[:len(unique_categories)]
    color_discrete_map = dict(zip(unique_categories, cmap))
    color_discrete_map[other_theme_name] = "rgba(128,128,128,0.5)"

# Plot
fig=px.scatter(df_vis,
             width=1300, height=1000, #The codes should be approximately 1:1 aspect ratio, but need space for the color bar
             x="x",
             y="y",
             size="Size",
             hover_name="Hover",
             title=("Codes and themes: " if themes_extracted else "Codes: ")+args.input,
             color = 'Theme' if themes_extracted else None,
             color_discrete_map=color_discrete_map if themes_extracted else None)
fig.write_html(out_base_name+"_codes_visualized.html")

