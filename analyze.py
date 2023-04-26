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
parser.add_argument('--input_instr', type=str, required=True, help="Path to a text file containing the coding instructions")
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
parser.add_argument('--encoding',type=str,required=False,default="ISO-8859-1", help="The encoding of .csv input")
args = parser.parse_args()

# Set the random seed
if args.use_cache==True and args.random_seed is None:
    print("Caching is enabled but random seed is not defined => setting the seed to 0 to allow caching.\n")
    args.random_seed=0
if args.random_seed is not None:
    random.seed(args.random_seed)

# Read inputs
df=pd.read_csv(args.input,sep=args.sep,encoding = args.encoding)
coding_instruction=open(args.input_instr, "r").read()

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
    use_cache=args.use_cache)

#Create output directory if needed
if not os.path.exists(args.output):
    os.mkdir(args.output)

#Save all the outputs
out_base_name=args.output+"/"+os.path.basename(args.input[:-4])
results["df"].to_csv(out_base_name+"_coded.csv",index=False)
results["df_editable"].to_csv(out_base_name+"_coded_editable.csv",index=False)
results["df_group_summary"].to_csv(out_base_name+"_group_summary.csv",index=False)
results["df_validate"].to_csv(out_base_name+"_human-gpt-comparison.csv",index=False)
open(out_base_name+"_prompt.txt", "w").write(results["prompt"])
