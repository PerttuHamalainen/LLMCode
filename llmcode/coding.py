import random
import pandas as pd
import numpy as np
import re
from Levenshtein import distance
from collections import defaultdict
from rapidfuzz import fuzz
from .llms import query_LLM, embed, reduce_embedding_dimensionality, print_progress_bar
from .embeddings import *

#Construct a coding prompt for a single text (unit of analysis)
def construct_prompt(coding_instruction,few_shot_texts,few_shot_codes,codes_so_far,text):
    # The prompt always starts with the user-defined coding instruction
    prompt = coding_instruction + "\n\n"

    # Next, we optionally add all the codes created so far, to allow code reuse instead of creating redundant
    # new and only slightly different codes for new coded texts
    if (codes_so_far is not None) and len(codes_so_far)>0:
        if len(codes_so_far)>0:
            prompt += "Examples of codes to use. Please add new codes when needed:\n"
            # Shuffle codes to mitigate LLM recency bias
            codes_so_far=codes_so_far.copy()
            random.shuffle(codes_so_far)
            # Add each code as a new line
            for code in codes_so_far:
                prompt+=code+"\n"

    # Next, we add the few-shot examples separated by ###
    # We first shuffle the examples to mitigate LLM recency bias
    l = list(zip(few_shot_texts, few_shot_codes))
    random.shuffle(l)
    few_shot_texts, few_shot_codes = zip(*l)
    for i in range(len(few_shot_texts)):
        prompt+="\n\n###\n\nText: "+few_shot_texts[i]+"\n\n"+"Codes: "+few_shot_codes[i]

    # Finally, we add the text to code, suggesting the LLM that it should continue the prompt with the codes
    prompt+="\n\n###\n\nText: "+text+"\n\n"+"Codes:"
    return prompt


# Extract a string of codes (separated by semicolons) from LLM continuation
def continuation_to_code_string(continuation):
    stopseq="###"
    continuation = continuation.split(stopseq)[
        0]  # if the model started generating new texts, only keep the continuation
    continuation = continuation.lstrip(" \n")  # strip leading spaces and newlines
    continuation = continuation.rstrip(" ;\n")  # strip trailing spaces, semicolons and newlines
    return continuation






# Convert from a string representation of multiple codes to a list
def codes_to_string(codes):
    return "; ".join(codes)

# Convert list of codes to a single string
def string_to_codes(s):
    codes = s.split(";")
    codes = [t.strip() for t in codes]
    return codes

#Code a list of texts using an instruction string and few-shot examples of texts and codes
def code_texts(coding_instruction,
               few_shot_texts,
               few_shot_codes,
               gpt_model,
               texts,
               use_cache=True,
               verbose=False,
               max_tokens=None):
    if max_tokens is None:
        max_tokens=64 #we assume the codes are short. TODO: measure the token length of few-shot examples, adjust this accordingly (e.g., 2x)

    #If we don't care about code consistency, we can speed things up by coding batches of texts in parallel.
    #First, we construct the prompts for all coded texts
    prompts=[]
    for text in texts:
        prompts.append(construct_prompt(coding_instruction=coding_instruction,
                            few_shot_texts=few_shot_texts,
                            few_shot_codes=few_shot_codes,
                            codes_so_far=None,
                            text=text))
    if verbose:
        print(f"Constructed {len(prompts)} prompts, example: {prompts[-1]}")

    #Query the LLM
    continuations=query_LLM(model=gpt_model,
                            prompts=prompts,
                            max_tokens=max_tokens,
                            use_cache=use_cache)

    # Extract code strings from the LLM continuations
    result_codes=[continuation_to_code_string(c) for c in continuations]

    # Ensure that the few-shot examples got coded correctly
    for i,text in enumerate(texts):
        if text in few_shot_texts:
            few_shot_code=few_shot_codes[few_shot_texts.index(text)]
            print(f"Replacing code {result_codes[i]} with {few_shot_code}")
            result_codes[i]=few_shot_code.strip()

    return result_codes



def extract_single_codes(df,multiple_column="codes",single_column="code",count_column="count",id_column="code_id",merge_singular_and_plural=False):
    df=df.copy()  #will be modified to contain the contents
    # Parse individual codes
    N = df.shape[0]
    df_append = None  # replicated rows are collected in this df
    df[single_column]=df[multiple_column].copy()
    for i in range(N):
        codes = df.at[i, multiple_column]
        codes = codes.split(";")
        codes = [t.lstrip() for t in codes]

        # if there's more than a single topic, create copies of the entry
        if len(codes) > 1:
            row = df.loc[i:i].copy()  # make a temp df of the current row
            df.at[i, multiple_column] = "PRUNED"  # mark the current row for pruning
            for code in codes:
                row[single_column] = code
                # display(row)
                if df_append is None:
                    df_append = row.copy()
                else:
                    df_append = pd.concat([df_append,row.copy()],axis=0)
    df = df[df[multiple_column] != "PRUNED"]
    df = pd.concat([df, df_append], axis=0, ignore_index=True)
    df.reset_index(drop=True, inplace=True)
    codes = df[single_column].unique().tolist()
    #print(f"Added {df.shape[0] - N} duplicate text rows to handle texts with multiple codes")
    #df=df.drop(columns=[multiple_column])

    # Merge singular and plural forms of the same code
    if merge_singular_and_plural:
        for code in codes:
            if code + "s" in codes:
                df.loc[df[single_column] == code, single_column] = code + "s"
                codes.remove(code)

    # Count and sort to have the most frequent topics at the top
    df[count_column] = 0
    counts = []
    for code in codes:
        count = df[df[single_column] == code].shape[0]
        counts.append(count)
        df.loc[df[single_column] == code, count_column] = count

    df.sort_values(by=[count_column, single_column], inplace=True, ascending=False)
    return df



def group_codes_using_embeddings(df,embedding_context="",embedding_model="text-similarity-curie-001",min_group_size=3,max_group_emb_codes=100,group_desc_codes=2,group_desc_freq=True,grouping_dim=5,use_cache=True,dimred_method=None):
    """
    Combines codes to groups/themes using HDBSCAN clustering of GPT-3 embedding vectors for each code.

    Also counts how many coded texts were assigned with each code and group. Note that one text can be assigned to
    multiple codes and therefore multiple groups.

    Args:
        df                  A dataframe with a "text_id" and "codes" column with codes separated by semicolons (as
                            produced by the code() function), or a "code" column with single codes (as produced by
                            the extract_single_codes() function).
        embedding_context   A context string appended to each code when computing code embeddings. This may help
                            disambiguating code labels that might be synonyms in the general case, but have different
                            meaning in the coding context. E.g., ", in context of experiencing games as art"
        group_desc_codes    How many of the most frequent codes to use for describing each code group
        group_desc_freq     Whether to use the group frequency in the group description
        min_group_size      Minimum number of codes in each code group
        use_cache           If True, skips OpenAI API calls if results found in the cache
                            (see set_cache_directory)

    Returns:
        A dictionary of results with the following keys:
        "coded"             Same as the input DataFrame but with the following additional columns:
                            "text_id", "text", "codes", "code", "code_id", "code_count", "group_codes", "group_id",
                            "group_count", "group_desc" columns. "codes" is a comma-separated list of all codes
                            assigned to each text. "code" is a single code, and each text is duplicated to as many
                            DataFrame rows as there are codes assigned to that text. code and group id:s are integers
                            that can be used to index the embedding arrays (see below). "text_id" is an index to the
                            original list of texts.
        "embeddings"        A numpy array of shape [num_codes,num_embedding_dimensions]
        "embeddings_5d"     Dimensionality-reduced embeddings used for clustering
        "group_embeddings"  A numpy array of shape [num_groups,num_embedding_dimensions], with the (normalized)
                            count-weighted average of each group's embedding vector. Can be useful for finding the
                            closest code groups between two separately coded datasets
        "group_code_ids"    list of lists of code ids (indices) for each group. Useful for, e.g., indexing the
                            embeddings of each group

        Note that if you want to view or list subsets of the results, e.g., the codes for each original text,
        this is easy to do by using Pandas drop_duplicates() on a suitable column such as "text_id"
    """
    if dimred_method is None:
        dimred_method="UMAP"

    df=df.copy() #will be modified to contain the results

    # extract individual codes
    if "text_id" not in df.columns:
        raise Exception("The input DataFrame does not have a \"text_id\" column!")
    if "code" in df.columns:
        print("The input dataframe already has a \"code\" column => skipping the initial extraction of single codes.")
    else:
        if "codes" not in df.columns:
            raise Exception("The input DataFrame does not have a \"codes\" column!")
        df["codes"] = df["codes"].astype(str)
        df = extract_single_codes(df)

    codes = df["code"].unique().tolist()

    code_counts=df.drop_duplicates(subset=["code"])["count"].to_numpy()
    n_codes=len(codes)
    n_texts=df["text_id"].nunique()

    #compute code embeddings
    print("Calculating code embeddings...")
    codes_with_context=[code + embedding_context for code in codes]
    embeddings=embed(codes_with_context,model=embedding_model,use_cache=use_cache)
    embed_dim=embeddings.shape[1]


    print("Reducing embedding dimensionality to 2D for visualization")
    dimred_neighbors=min_group_size+2 #heuristic: for the grouping to work, dimensionality reduction needs to consider a somewhat larger neighborhood
    embeddings_2d = reduce_embedding_dimensionality(embeddings,2,use_cache=use_cache,method=dimred_method,n_neighbors=dimred_neighbors)

    #reduce dimensionality
    print("Reducing embedding dimensionality for grouping...")
    embeddings_reduced=reduce_embedding_dimensionality(embeddings, grouping_dim, method=dimred_method,use_cache=use_cache,n_neighbors=dimred_neighbors)

    #group (UMAP + HDBSCAN)
    print("Grouping...")
    clustering_method='hdbscan'

    if clustering_method=="hdbscan":
         import hdbscan
         cluster = hdbscan.HDBSCAN(min_cluster_size=min_group_size,
                                   metric='euclidean',
                                   cluster_selection_method='eom').fit(embeddings_reduced)
         if np.min(cluster.labels_)<0:
             has_outliers=True
             labels=cluster.labels_+1  #+1 because the outlier cluster has id -1, which we now map to 0 to make indexing easier
         else:
             has_outliers = False
             labels=cluster.labels_

    elif clustering_method=="k-means":
         from sklearn.cluster import KMeans

         n_groups = 20
         kmeans = KMeans(init='k-means++', n_clusters=n_groups, n_init=20)
         kmeans.fit(embeddings_reduced)

         # Predict the cluster for all the samples
         labels = kmeans.predict(embeddings_reduced)

    #Update the cluster_id column
    n_groups = np.max(labels) + 1
    print(f"Combined {len(codes)} codes to {n_groups} groups")
    groups= [''] * n_groups
    group_code_ids=[None] * n_groups
    for i in range(n_groups):
        group_code_ids[i]=[]

    df["group_id"]=0
    df["code_id"]=-1
    df["code_2d_0"]=0
    df["code_2d_1"]=0
    for code_id,code in enumerate(codes):
         #group label for this code
         group_id=labels[code_id]
         group_code_ids[group_id].append(code_id)
         if groups[group_id]=='':
              groups[group_id]=codes[code_id]
         else:
              groups[group_id]+=', '+codes[code_id]
         df.loc[df["code"] == code,"group_id"]=group_id
         df.loc[df["code"] == code,"code_id"]=code_id
         df.loc[df["code"] == code,"code_2d_0"]=embeddings_2d[code_id,0]
         df.loc[df["code"] == code,"code_2d_1"]=embeddings_2d[code_id,1]

    #Update the various group stats and descriptors
    df["group_codes"]=""
    df["group_count"]=0
    df["group_desc"]=""
    group_embeddings=np.zeros([n_groups,embed_dim])
    for i in range(n_groups):
        #update group codes and count
        rows=df["group_id"]==i
        df.loc[rows, "group_codes"]=groups[i]
        group_count=df.loc[rows,"text_id"].nunique()
        group_freq=round(100.0 * group_count / n_texts)
        df.loc[rows,"group_count"]=group_count
        df.loc[rows,"group_freq"]=group_freq
        df.loc[rows,"group_code_ids"]=",".join([str(code_id) for code_id in group_code_ids[i]])

        #group embeddings as weighted average of code embeddings
        code_ids=group_code_ids[i]
        if len(code_ids)>max_group_emb_codes:
            code_ids=code_ids[:max_group_emb_codes]
        weights=code_counts[code_ids]
        group_embeddings[i]=np.average(embeddings[code_ids],axis=0,weights=weights)
        group_embeddings[i]/=np.linalg.norm(group_embeddings[i])+1e-10

        #codes sorted by distance from the group embedding
        all_code_embeddings=embeddings[group_code_ids[i]]
        dist=np.inner(all_code_embeddings,group_embeddings[i].reshape([1,-1]))[:,0]
        dist=1.0-dist
        sorted_group_code_ids=np.array(group_code_ids[i])[np.argsort(dist)]
        sorted_group_codes=np.array(codes)[sorted_group_code_ids]
        df.loc[rows,"sorted_group_codes"]=",".join(sorted_group_codes)

        # Helper for naming a group based on most frequent codes
        def group_name_freq(codes, freq):
            codes = codes.split(',')
            result = codes[0]
            if len(codes) > 1:
                for code in codes[1:min([len(codes), group_desc_codes])]:
                    result += ", " + code
            if group_desc_freq:
                result+=f" ({freq}%)"
            return result
        df.loc[rows, "group_desc"]=group_name_freq(groups[i],group_freq)

    #Mark outliers if using hdbscan
    outliers=None
    df["outlier"]=0
    df["not_outlier"]=1 #just to make sorting easier
    if clustering_method=="hdbscan" and has_outliers:
        df.loc[df["group_id"]==0,"outlier"]=1
        df["not_outlier"] = 1-df["outlier"]
        '''
        outliers=df[df["group_id"]==0]
        df=df[df["group_id"]!=0]
        df["group_id"]=df["group_id"]-1
        group_code_ids=group_code_ids[1:]
        group_embeddings=group_embeddings[1:]
        n_groups-=1
        '''

    #Sort
    df.sort_values(by=["not_outlier","group_count","group_codes","count", "code"], inplace=True, ascending=False)

    return {"df":df,"embeddings":embeddings,"embeddings_reduced":embeddings_reduced,"embeddings_2d":embeddings_2d,"group_embeddings":group_embeddings,"group_code_ids":group_code_ids,"n_groups":n_groups,"code_counts":code_counts}


def code_df(df,
           column_to_code,
           coding_model,
           embedding_model,
           embedding_context,
           dimred_method,
           dimred_neighbors,
           use_cache=True,
           verbose=False,
           pruned_code=None):
    #Code
    print(f"Coding the {column_to_code} column of the dataframe")
    texts = df[column_to_code].astype(str).tolist()
    few_shot_codes = df.loc[df['use_as_example'] == 1, 'human_codes'].astype(str).tolist()
    few_shot_texts = df.loc[df['use_as_example'] == 1, column_to_code].astype(str).tolist()

    # check if the df specifies the instructions
    if "coding_instructions" in df:
        coding_instruction = df["coding_instructions"][0]
    else:
        raise Exception(
            "Coding instructions not specified. Please specify using the first row of a column named \"coding_instructions\"")

    codes = code_texts(coding_instruction=coding_instruction,
                       few_shot_texts=few_shot_texts,
                       few_shot_codes=few_shot_codes,
                       gpt_model=coding_model,
                       texts=texts,
                       use_cache=use_cache,
                       verbose=verbose)
    df_coded=df.copy()
    df_coded["codes"]=codes
    df_coded["text_id"]=list(range(0, len(texts)))  #for keeping track of which original texts the codes refer to

    print("\nCoding instruction:\n\n")
    print(coding_instruction)

    print("Coded data:")
    df_coded.reset_index(drop=True,inplace=True) #for cleaner printouts
    print(df_coded[[column_to_code, "codes"]].head(20))

    # Compare gpt and human codes
    print("Comparing LLM and human codes based on modified Hausdorff distance in embedding space...")
    df_cmp=df_coded[df_coded["human_codes"].notnull()].copy()
    df_cmp=df_cmp[df_cmp["human_codes"]!=""]
    df_cmp=df_cmp[df_cmp["use_as_example"]!="1"]
    df_cmp=df_cmp[[column_to_code,"human_codes","codes"]]
    df_cmp=gpt_human_code_dist(df=df_cmp,embedding_context=embedding_context,embedding_model=embedding_model)

    # Create a df with a line per code, sorted by code frequencies, with original texts/groundings in other columns
    print("Formatting output...")
    # First, construct a df with a line code, duplicated for each coded text
    df_codes = df_coded.copy()
    df_codes["codes"] = df_codes["codes"].astype(str)
    df_codes = extract_single_codes(df_codes)
    df_codes = df_codes[["code","count","text_id",column_to_code]]
    max_count=df_codes["count"].max()

    # Now, pack the duplicate rows into columns, and convert from single text id:s to lists of text ids for each code
    df_single = df_codes.drop_duplicates(subset=["code"]).copy()
    text_ids_all=[]
    for i in range(max_count):
        df_single[f"text {i}"]=None #placeholder for the texts

    for code in df_single["code"]:
        rows=df_codes[df_codes["code"]==code]
        text_ids=rows["text_id"].astype(str).to_list()
        texts=rows[column_to_code].astype(str).to_list()
        for i in range(len(texts)):
            df_single.loc[df_single["code"] == code, f"text {i}"]=f"{text_ids[i]}: {texts[i]}"
        text_ids_all.append(",".join(text_ids))
    df_single["text_ids"]=text_ids_all
    df_single=df_single.drop(columns=["text_id",column_to_code])
    df_single.reset_index(drop=True,inplace=True)
    df_codes=df_single

    # Create and print a summary of codes
    df_code_summary=df_codes[["code","count","text 0"]].rename(columns={"text 0":"example text"})
    print("Total number of codes: ",df_single.shape[0])
    print("\nCodes sorted by number of groundings:")
    print(df_code_summary.head(df_code_summary.shape[0]))

    # Calculate 2d embeddings for each code, for visualizing
    print("Embedding codes and reducing dimensionality for visualization...")
    codes_with_context=[code + embedding_context for code in df_codes["code"].astype(str).tolist()]
    embeddings=embed(codes_with_context,model=embedding_model,use_cache=use_cache)
    embeddings_2d = reduce_embedding_dimensionality(embeddings=embeddings,
                                                    num_dimensions=2,
                                                    use_cache=use_cache,
                                                    method=dimred_method,
                                                    n_neighbors=dimred_neighbors)
    df_codes["code_2d_0"]=embeddings_2d[:,0]
    df_codes["code_2d_1"]=embeddings_2d[:,1]


    # Prune codes, if a pruning code specified
    df_coded_pruned=None
    df_codes_pruned=None
    if pruned_code is not None:
        df_coded_pruned=df_coded[~df_coded["codes"].str.contains(pruned_code)].copy()
        df_coded_pruned.reset_index(drop=True, inplace=True)
        df_codes_pruned=df_codes[~df_codes["code"].str.contains(pruned_code)].copy()
        df_codes_pruned.reset_index(drop=True, inplace=True)

    # Return results as a dict
    result={}
    result["prompt"]=construct_prompt(
        coding_instruction=coding_instruction,
        few_shot_texts=few_shot_texts,
        few_shot_codes=few_shot_codes,
        codes_so_far=None,
        text=texts[0]
    )
    result["df_coded"]=df_coded
    result["df_codes"]=df_codes
    result["df_coded_pruned"]=df_coded_pruned
    result["df_codes_pruned"]=df_codes_pruned
    result["df_validate"]=df_cmp
    #result["df_code_summary"]=df_code_summary
    return result

def group_codes(
    df_codes,
    df_data,
    grouping_model,
    use_cache,
    random_seed=None,
    verbose=False):

    #extract and check instructions
    theme_elicitation_instructions=df_data["theme_elicitation_instructions"][0]
    if not "<codes>" in theme_elicitation_instructions:
        raise Exception("Theme elicitation instructions missing the <codes> placeholder which will be replaced with the list of codes in the LLM prompt.")

    code_grouping_instructions=df_data["code_grouping_instructions"][0]
    if not "<codes>" in code_grouping_instructions:
        raise Exception("Code grouping instructions missing the <codes> placeholder which will be replaced with the list of codes in the LLM prompt.")
    if not "<themes>" in code_grouping_instructions:
        raise Exception("Code grouping instructions missing the <themes> placeholder which will be replaced with the list of themes in the LLM prompt.")

    #STEP 1: Elicit a list of themes
    #If there are more codes than fit a single prompt, we elicit the themes based on a random subset.
    codes=df_codes["code"].astype(str).to_list()
    random.seed(random_seed)
    random.shuffle(codes)

    #Determine how many codes we can fit the prompt
    code_tokens=[num_tokens_from_string(code,grouping_model) for code in codes]
    overhead=num_tokens_from_string(string=theme_elicitation_instructions,model=grouping_model)\
             +token_overhead(grouping_model)+1000 #token overhead for theme list, system message and prompt start
    max_tokens_for_codes=max_llm_context_length[grouping_model]-overhead
    code_tokens_cumulative=0
    max_codes=0
    for i in range(len(code_tokens)):
        code_tokens_cumulative+=code_tokens[i]+1 #+1 because of newlines after each code in the prompt
        if code_tokens_cumulative>max_tokens_for_codes:
            break
        max_codes=i

    #Prompt the themes
    print(f"Identifying themes based on {max_codes} codes...")
    codes_as_string="\n".join(codes[:max_codes])
    prompt=theme_elicitation_instructions
    prompt=prompt.replace("<codes>",codes_as_string)
    response=query_LLM(model=grouping_model,
                       prompts=[prompt],
                       temperature=0,
                       use_cache=use_cache)[0]
    if verbose:
        print("Prompt:\n")
        print(prompt+"\n")

    #Parse response
    response=response[response.find("{"):response.find("}")+1] #extract the .json part
    def correct_gpt_quotation_escape(s):
        return s #s.replace('\\\"\"','\\\"')       #Aalto gpt-3.5-turbo sometimes does a "double" escape of quotation marks
    response=correct_gpt_quotation_escape(response)
    j = json.loads(response)
    theme_list=j["themes"]
    theme_list=list(set(theme_list))  #remove possible duplicates
    other_theme="Other" #the theme identification probably misses some minor themes => they will get grouped under this theme
    theme_list.append(other_theme)
    themes_as_string="\n".join(theme_list)
    print("Themes identified:\n")
    print(themes_as_string+"\n")

    #STEP 2: Group codes under the themes in batches
    #prompt the LLM
    batch_size=50
    remaining_codes=codes.copy()
    themes={}   #results will be added to this dict, with codes as keys
    while len(remaining_codes)>0:
        current_remaining=len(remaining_codes)
        prompt=code_grouping_instructions
        code_batch=remaining_codes[:min(batch_size,len(remaining_codes))]
        print(f"Grouping a batch of {len(code_batch)} codes, total {current_remaining} remaining...")
        codes_as_string="\n".join(code_batch)
        if verbose:
            print("Code batch:")
            print(codes_as_string)
        prompt=prompt.replace("<codes>",codes_as_string)
        prompt=prompt.replace("<themes>",themes_as_string)
        response=query_LLM(model=grouping_model,
                           prompts=[prompt],
                           use_cache=use_cache)[0]
        response=response[response.find("{"):response.find("}")+1] #extract the .json part
        response = correct_gpt_quotation_escape(response)
        try:
            j = json.loads(response)
        except json.decoder.JSONDecodeError:
            print(".json parse of the following response failed:\n")
            print(response)
            exit()
        #remove codes that were successfully grouped
        for code in code_batch:
            if code in j:
                remaining_codes.remove(code)
                themes[code]=j[code]

        #prevent getting stuck in an infinite loop in case the LLM keeps making mistakes
        if len(remaining_codes)==current_remaining:
            print("These remaining codes could not be grouped, categorizing them as 'other':")
            for code in remaining_codes:
                print(code)
                themes[code]="Other"
            break

    #Add themes to the codes dataframe
    df_grouped=df_codes.copy()
    df_grouped.reset_index(drop=True, inplace=True)
    codes_all=df_grouped["code"]
    themes_all=[themes[code] for code in codes_all]
    df_grouped["theme"]=themes_all

    #Calculate theme counts
    themes_list=df_grouped["theme"].unique()
    df_grouped["theme_count"]=0
    df_grouped["theme_index"]=0
    for i,theme in enumerate(themes_list):
        df_theme=df_grouped[df_grouped["theme"]==theme]
        df_grouped.loc[df_grouped["theme"]==theme,"theme_count"]=df_theme["count"].sum()
        df_grouped.loc[df_grouped["theme"] == theme, "theme_index"]=i

    #Sort by both theme counts, themes, code_counts
    df_grouped.sort_values(by=["theme_count", "theme","count"], inplace=True, ascending=False)
    df_grouped.reset_index(drop=True, inplace=True)

    '''
    #Replace many very small themes with "Other"
    if other_threshold is not None:
        df_other=df_grouped[df_grouped["theme_count"]<=other_threshold].copy()
        df_rest=df_grouped[df_grouped["theme_count"]>other_threshold].copy()
        df_other["theme"]="Other"
        df_other["theme_count"]=df_other["count"].sum()
        df_grouped=pd.concat([df_rest,df_other],axis=0)
    '''

    #Some final formatting
    df_grouped.rename(columns={"count":"code_count"},inplace=True)
    cols=list(df_grouped.columns.values)
    cols.remove("theme")
    cols.remove("theme_count")
    if "index" in cols:
        cols.remove("index")
    if "level_0" in cols:
        cols.remove("level_0")
    df_grouped=df_grouped[["theme","theme_count"]+cols]
    return df_grouped

def code_and_group(df,
                   column_to_code,
                   coding_model,
                   embedding_model,
                   embedding_context=None,
                   min_group_size=3,
                   grouping_dim=5,
                   use_cache=True,
                   verbose=False,
                   dimred_method=None,
                   pruned_code=None):
    if embedding_context is None:
        embedding_context=""



    # Group codes
    group_info = group_codes(df_coded,
                             embedding_context=embedding_context,
                             embedding_model=embedding_model,
                             min_group_size=min_group_size,
                             use_cache=use_cache,
                             group_desc_codes=3,
                             grouping_dim=grouping_dim,
                             group_desc_freq=False,
                             dimred_method=dimred_method)

    # Create and print a summary of groups
    df_grouped=group_info["df"].copy()
    df_grouped = df_grouped.drop_duplicates(subset=["group_codes"])  # only print one text per group
    df_grouped.reset_index(drop=True,inplace=True) #for cleaner printouts
    df_group_summary=df_grouped[[column_to_code, "group_desc","group_freq"]].copy()
    print("\nCode groups sorted by group frequency")
    print(df_group_summary.head(20))

    # Convert the text-per-row grouped df to code-per-row,
    # which allows easier editing of the results in a spreadsheet editor

    # First, create an empty dataframe with correct columns
    columns=["group_desc","code"]
    for i in range(group_info["df"]["count"].max()):
        columns.append(f"text_{i+1}")
    df_editable=None #pd.DataFrame(columns=columns)

    # loop over groups
    groups = group_info["df"]["group_id"].unique()
    for group_id in groups:
        # select this group's rows
        group_data=group_info["df"][group_info["df"]["group_id"]==group_id]
        # skip if this is the outlier group produced by HDBSCAN
        if group_data["outlier"].iloc[0]==0:
            # loop over codes
            codes=group_data["code"].unique()
            for c in codes:
                code_data=group_data[group_data["code"]==c]
                # add a line for the code
                new_row={}
                new_row["group_desc"]=code_data["group_desc"].iloc[0]
                new_row["group_count"]=code_data["group_count"].iloc[0]
                new_row["code"]=c
                new_row["code_count"]=int(code_data["count"].iloc[0])
                new_row["code_2d_0"]=code_data["code_2d_0"].iloc[0]
                new_row["code_2d_1"] = code_data["code_2d_1"].iloc[0]

                for c in range(code_data.shape[0]):
                    new_row[f"text_{c+1}"]=code_data[column_to_code].iloc[c]
                new_row=pd.Series(data=new_row).to_frame().T
                if df_editable is None:
                    df_editable=new_row
                else:
                    df_editable=pd.concat(objs=[df_editable,new_row],axis=0,ignore_index=True)
    df_editable.reset_index(drop=True, inplace=True)

    # Return a dict with all the info the caller might need for further analysis or data export
    result=group_info.copy()
    result["prompt"]=construct_prompt(
        coding_instruction=coding_instruction,
        few_shot_texts=few_shot_texts,
        few_shot_codes=few_shot_codes,
        codes_so_far=None,
        text=texts[0]
    )
    result["df_coded"]=df_coded_not_pruned
    result["df_validate"]=df_cmp
    result["df_editable"]=df_editable
    result["df_code_summary"]=df_code_summary
    result["df_group_summary"]=df_group_summary
    return result



def code_and_embed(prompt_base,df,coding_model="text-curie-001",embedding_model="text-similarity-curie-001",column_to_code="texts",embedding_context="",use_cache=True,verbose=True):
    df_coded = code(prompt_base=prompt_base,
                      gpt_model=coding_model,
                      df=df,
                      column_to_code=column_to_code,
                      use_cache=use_cache,
                      verbose=verbose)
    df_single = extract_single_codes(df_coded)
    codes = df_single["code"].unique().tolist()
    code_counts=df_single.drop_duplicates(subset=["code"])["count"].to_numpy()
    codes_with_context=[code + embedding_context for code in codes]
    embeddings=embed(codes_with_context,model=embedding_model,use_cache=use_cache)
    return {"codes":codes,"counts":code_counts,"embeddings":embeddings}


def code_inductively(texts,
                     coding_instructions,
                     few_shot_examples,
                     gpt_model,
                     use_cache=True,
                     max_tokens=None,
                     verbose=False):
    if max_tokens is None:
        # Set max_tokens dynamically based on maximum text length
        # Note that len(text) is in characters, not tokens
        max_tokens = max(max(len(text) for text in texts), 300)
    
    # Code batches of texts in parallel
    prompts = [construct_inductive_prompt(text, coding_instructions, few_shot_examples)
               for text in texts]

    # Query the LLM
    continuations = query_LLM(
        model=gpt_model,
        prompts=prompts,
        max_tokens=max_tokens,
        use_cache=use_cache
    )

    # Attempt to correct any LLM formatting errors
    coded_texts = correct_coding_errors(texts, continuations, verbose=verbose)

    return coded_texts


def code_inductively_with_code_consistency(texts,
                                           research_question,
                                           coding_instructions,
                                           few_shot_examples,
                                           gpt_model,
                                           use_cache=True,
                                           max_tokens=None,
                                           verbose=False):
    if max_tokens is None:
        # Set max_tokens dynamically based on maximum text length
        # Note that len(text) is in characters, not tokens
        max_tokens = max(max(len(text) for text in texts), 300)

    # Process texts sequentially to enforce code consistency
    coded_texts = []
    code_descriptions = {}
    for idx, text in enumerate(texts):
        # Update progress
        print_progress_bar(idx + 1, len(texts), printEnd="")

        # Find insights independent of existing codes
        insight_prompt = construct_insight_prompt(text, research_question)
        continuations = query_LLM(
            model=gpt_model,
            prompts=[insight_prompt],
            max_tokens=200,
            use_cache=use_cache
        )
        insights = continuations[0]

        # Construct prompt, including a list of existing codes
        prompt = construct_inductive_prompt(
            text=text, 
            coding_instructions=coding_instructions,
            few_shot_examples=few_shot_examples,
            code_descriptions=code_descriptions,
            insights=insights
        )

        # Query the LLM
        continuations = query_LLM(
            model=gpt_model,
            prompts=[prompt],
            max_tokens=max_tokens,
            use_cache=use_cache
        )
        
        # Attempt to correct any LLM formatting errors
        coded_text_batch = correct_coding_errors([text], continuations, verbose=verbose)

        # Add singular coded text to output
        assert len(coded_text_batch) == 1
        coded_text = coded_text_batch[0]
        coded_texts.append(coded_text)

        # For any new codes, generate description and store
        for highlight, code in parse_codes(coded_text):
            if code not in code_descriptions:
                code_descriptions[code] = generate_code_description(
                    code,
                    [highlight],
                    research_question,
                    gpt_model,
                    use_cache
                )
                
    return coded_texts, code_descriptions


def code_deductively(texts,
                     coding_instructions,
                     codebook,
                     gpt_model,
                     few_shot_examples=None,
                     use_cache=True,
                     verbose=False):
    """
    Apply deductive qualitative coding to a list of texts using a language model.

    This function uses a predefined codebook and prompts a GPT-based model to 
    generate coded versions of the input texts. It processes the texts deductively, 
    meaning the coding is guided by the provided codebook. Optionally, few-shot 
    examples can be included to guide the model's responses. The function returns 
    the coded texts and a dictionary of codes with corresponding highlights.

    Args:
        texts (list of str): A list of texts to be coded.
        coding_instructions (str): The user-defined coding instructions used to guide the language model.
        codebook (list of str or tuple): A list of code strings or (code, description) tuples representing the codebook.
        gpt_model (str): The identifier for the GPT model to be used for coding.
        few_shot_examples (pd.DataFrame, optional): DataFrame containing few-shot examples to guide the model. 
            Each row should contain a 'coded_text' column with pre-coded examples.
        use_cache (bool, optional): Whether to use cached model results if available. Defaults to True.
        verbose (bool, optional): Whether to print warnings for coding inconsistencies and errors. Defaults to False.

    Returns:
        tuple: 
            - coded_texts (list of str): The input texts with codes inserted by the language model.
    """
    # Set max_tokens dynamically based on maximum text length
    # Note that max_text_len is in characters, not tokens
    max_text_len = max(len(text) for text in texts)

    # Make codebook into a list of tuples if list of strings
    codebook = [(item,) if isinstance(item, str) else item for item in codebook]

    # Make sure codes in few_shot_examples are from codebook
    codebook_codes = [item[0] for item in codebook]
    if few_shot_examples is not None:
        for _, row in few_shot_examples.iterrows():
            for _, code in parse_codes(row.coded_text):
                if code not in codebook_codes:
                    print(f"WARNING: Few-shot examples contain code \"{code}\" that is not in the codebook")

    # Query the LLM
    prompts = [construct_deductive_prompt(text, coding_instructions, codebook, few_shot_examples)
               for text in texts]
    continuations = query_LLM(
        model=gpt_model,
        prompts=prompts,
        max_tokens=max_text_len,
        use_cache=use_cache
    )

    # Attempt to correct any LLM formatting errors
    coded_texts = correct_coding_errors(texts, continuations, verbose=verbose)

    # Check output for any hallucinated codes
    for idx in range(len(coded_texts)):
        hallucinated_codes = []
        for _, code in parse_codes(coded_texts[idx]):
            if code not in codebook_codes:
                hallucinated_codes.append(code)
                if verbose:
                    print(f"WARNING: Output contains code \"{code}\" that is not in the codebook, removing...")

        # Remove any hallucinated codes
        for code in hallucinated_codes:
            coded_texts[idx] = _remove_code(coded_texts[idx], code)

    return coded_texts


def _remove_code(text, code_to_remove):
    highlight_pattern = r"(\*\*(.*?)\*\*<sup>(.*?)<\/sup>)"
    for full_match, highlight, codes in re.findall(highlight_pattern, text):
        codes = [code.strip() for code in codes.split(";")]
        if code_to_remove in codes:
            if len(codes) == 1:
                # If it's the only code, remove the entire highlighted structure
                new_text = highlight
            else:
                # Remove the specified code and reformat
                codes = [code for code in codes if code != code_to_remove]
                new_text = "**{}**<sup>{}</sup>".format(highlight, "; ".join(codes))
            
            # Replace the first occurrence of the full match with the modified text
            text = text.replace(full_match, new_text, 1)
            break  # Only remove one occurrence per function call
    return text


def construct_inductive_prompt(text,
                               coding_instructions,
                               few_shot_examples,
                               code_descriptions=None,
                               insights=None,
                               ):
    # Optionally add guidance about given main insights into the prompt
    insight_text = " and the main insights from the text" if insights is not None else ""

    prompt = f"""You are an expert qualitative researcher. You are given a text to code inductively{insight_text}. Please carry out the following task:
- Respond by repeating the original text, but highlighting the coded statements by surrounding the statements with double asterisks, as if they were bolded text in a Markdown document.
- Include the associated code(s) immediately after the statement, separated by a semicolon and enclosed in <sup></sup> tags, as if they were superscript text in a Markdown document.
- Preserve exact formatting of the original text. Do not correct typos or remove unnecessary spaces.\n"""
    
    # Add user-defined instructions
    prompt += coding_instructions + "\n\n"

    # Optionally add existing codes into the prompt, to encourage consistency
    if code_descriptions is not None and len(code_descriptions) > 0:
        prompt += "Some examples of codes in the format \"{code}: {description}\". Please create new codes when needed:\n"
        # Shuffle codes to mitigate LLM recency bias
        code_desc_str = [f"{code}: {description}" for code, description in code_descriptions.items()]
        random.shuffle(code_desc_str)
        # Add each code as a new line
        prompt += "\n".join(code_desc_str) + "\n\n"

    prompt += "Below, I first give you examples of the output you should produce given an example input. After that, I give you the actual input to process.\n\n"

    # Add the few-shot examples in random order
    for _, row in few_shot_examples.sample(frac=1).iterrows():
        prompt += f"EXAMPLE INPUT:\n{row.text}\n\n"
        prompt += f"EXAMPLE OUTPUT:\n{row.coded_text}\n\n"

    prompt += f"ACTUAL INPUT:\n{text}"

    if insights is not None:
        prompt += f"\n\nMAIN INSIGHTS:\n{insights}"

    return prompt


def construct_deductive_prompt(text,
                               coding_instructions,
                               codebook,
                               few_shot_examples):
    prompt = """You are an expert qualitative researcher. You are given a text to code deductively using a list of codes. Please carry out the following task:
- Respond by repeating the original text, but highlighting the coded statements by surrounding the statements with double asterisks, as if they were bolded text in a Markdown document.
- Include the associated code(s) immediately after the statement, separated by a semicolon and enclosed in <sup></sup> tags, as if they were superscript text in a Markdown document.
- Preserve exact formatting of the original text. Do not correct typos or remove unnecessary spaces.\n"""

    # Add user-defined instructions
    prompt += coding_instructions + "\n\n"

    if few_shot_examples is None:
        prompt += "The following is an example of the correct output format, with fictional codes:\n\n"
        prompt += "I really enjoy walking in the park on weekends. **It helps me clear my mind**<sup>mental clarity</sup> and **feel more connected to nature**<sup>connection to nature</sup>. Sometimes, I take my dog with me, and we just wander around, **enjoying the fresh air**<sup>sensory enjoyment; relaxation</sup>.\n\n"

    prompt += "Use codes from the following list:\n" 
    # Shuffle codebook to mitigate LLM recency bias
    shuffled_codebook = codebook[:]
    random.shuffle(shuffled_codebook)
    # Add each code as a new line
    for item in shuffled_codebook:
        if len(item) == 1:
            # No description included, add only code
            prompt += f"{item[0]}\n"
        else:
            prompt += f"{item[0]}: {item[1]}\n"
    prompt += "\n"

    if few_shot_examples is not None:
        prompt += "Below, I first give you examples of the output you should produce given an example input. After that, I give you the actual input to process.\n\n"

        # Add the few-shot examples in random order
        for _, row in few_shot_examples.sample(frac=1).iterrows():
            prompt += f"EXAMPLE INPUT:\n{row.text}\n\n"
            prompt += f"EXAMPLE OUTPUT:\n{row.coded_text}\n\n"

    prompt += f"INPUT:\n{text}"
    
    return prompt


def construct_insight_prompt(text, research_question):
    prompt = f"You are an expert qualitative researcher who is given the following text to analyze:\n\n{text}\n\n"
    prompt += f"Output a single sentence summarising the most interesting insights in the text, specifically pertaining to the research question \"{research_question}\". "
    prompt += f"If there are no relevant insights, output \"The text contains no insights relevant to the research question.\""
    return prompt


def generate_code_description(code, examples, research_question, gpt_model, use_cache, counter_examples=[]):
    prompt = "Write a brief but nuanced one-sentence description for the given inductive code, based on a set of texts annotated with the code"

    if len(counter_examples) > 0:
        prompt += " and counter-examples where the code does not apply.\n"
    else:
        prompt += ".\n"

    prompt += " For example, for the code \"overcommunication\", you might generate the description: Captures instances where participants discuss feeling overwhelmed by excessive communication, such as constant emails, messages, or meetings\n"
    prompt += f" Write the description in the context of a qualitative research project with the research question: {research_question}.\n\n"

    prompt += f"CODE: {code}\n\n"

    prompt += "CODED TEXTS SEPARATED BY \"***\":\n"
    prompt += "\n***\n".join(examples)

    if len(counter_examples) > 0:
        prompt += "\n\nCOUNTER-EXAMPLES SEPARATED BY \"***\":\n"
        prompt += "\n***\n".join(examples)

    continuations = query_LLM(
        model=gpt_model,
        prompts=[prompt],
        max_tokens=200,
        use_cache=use_cache
    )
    
    return continuations[0]


def correct_coding_errors(texts, continuations, verbose=False):
    """
    Check for LLM errors and attempt to correct errors where the LLM has omitted non-coded parts of the original text.
    Outputs a list of LLM-coded texts corresponding to texts, with None in place of outputs that could not be corrected.
    """
    coded_texts = []
    n_reconstructed = 0

    for text, cont in zip(texts, continuations):
        if len(cont) == 0:
            # If the response is empty (possibly, due to model censorship), discard
            print(f"WARNING: Discarding empty LLM response for text \"{text}\"\n")
            coded_texts.append(None)
            continue

        # Check if there were any clear errors or hallucinations. In other words, does the LLM response (cont) match
        # the original text when we remove the highlight and code annotations?
        cont_text = re.sub(r"\*\*|<sup>(.*?)<\/sup>", "", cont)
        if cont_text == text:
            coded_texts.append(cont)
            continue

        # Sometimes, the LLM autocorrects typos even though we explicitly tell it not to.
        # Ignore the difference if the edit distance between the original and LLM-highlighted text is small enough.
        dist_threshold = 5
        edit_dist = distance(text, cont_text)
        if verbose:
            print(f"Warning: LLM output differs from the original text, with edit distance {edit_dist}")
            print(f"Original text: \"{text}\"")
            print(f"LLM output: \"{cont}\"")
        
        if edit_dist < dist_threshold:
            if verbose:
                print(f"Distance less than treshold {dist_threshold}, accept\n")
            coded_texts.append(cont)
            continue

        # Error detected, attempt to reconstruct by finding the annotations in the original text with fuzzy matching
        annotations = re.findall(r"\*\*(.*?)\*\*(<sup>.*?<\/sup>)", cont)
        reconstructed = text
        reconstruction_failed = False
        for highlight, codes in annotations:
            match_start, match_end, ratio = find_best_match(reconstructed, highlight)
            if ratio >= 90:
                # Add annotation to reconstruction if found a match at sufficient similarity ratio
                rec_annotation = "**" + reconstructed[match_start:match_end] + "**" + codes
                reconstructed = reconstructed[:match_start] + rec_annotation + reconstructed[match_end:]
            else:
                if verbose:
                    print(f"Could not find the LLM-annotated text \"{highlight}\" in the original text \"{text}\"")
                reconstruction_failed = True
                break

        if reconstruction_failed:
            if verbose:
                print(f"Text reconstruction failed, discard LLM response\n")
            coded_texts.append(None)
        else:
            if verbose:
                print(f"Text reconstruction successful\n")
            coded_texts.append(reconstructed)
            n_reconstructed += 1

    if n_reconstructed > 0 and verbose:
        print(f"Had to reconstruct {n_reconstructed} texts due to LLM errors")

    n_discarded = sum(t is None for t in coded_texts)
    if n_discarded > 0:
        print(f"WARNING: A total of {n_discarded} LLM outputs were discarded because of errors\n")

    return coded_texts


def find_best_match(original_text, target_substring):
    """
    Finds the substring in original_text that best matches target_substring using
    fuzzy matching, allowing for some length variance.

    Args:
        original_text (str): The text to search in.
        target_substring (str): The string to find a similar match for.

    Returns:
        tuple: (best_start, best_end, best_ratio) where best_start and best_end are
        the indices of the best matching substring, and best_ratio is the similarity score.
    """

    best_ratio = 0
    best_start = 0
    best_end = 0

    # We only search for matches within max_length_variance from len(target_substring)
    max_length_variance = int(len(target_substring) / 4)

    # Iterate over all possible window sizes
    for start in range(len(original_text)):
        end_min = start + len(target_substring) - max_length_variance
        end_max = min(start + len(target_substring) + max_length_variance, len(original_text) + 1)
        for end in range(end_min, end_max):
            window = original_text[start:end]
            similarity_ratio = fuzz.ratio(window, target_substring)
            
            # Update the best match if this window has a better match
            if similarity_ratio > best_ratio:
                best_ratio = similarity_ratio
                best_start = start
                best_end = end

    return best_start, best_end, best_ratio


def get_codes_and_highlights(coded_texts):
    code_highlights = defaultdict(list)
    for coded_text in coded_texts:
        for highlight, code in parse_codes(coded_text):
            code_highlights[code].append(highlight)
    return code_highlights


def parse_codes(coded_text):
    if coded_text is None:
        return []
    return [(highlight, code.strip()) 
            for highlight, codes in re.findall(r"\*\*(.*?)\*\*<sup>(.*?)<\/sup>", coded_text)
            for code in codes.split(";")]


def get_2d_code_embeddings(codes, embedding_context, embedding_model, verbose=False):
    # Add context to each set of codes before embedding
    codes_with_context = [code + embedding_context for code in codes]
    embeddings = embed(codes_with_context, model=embedding_model, verbose=verbose)

    # Reduce embedding dimensionality to 2
    embeddings_2d = reduce_embedding_dimensionality(embeddings=embeddings, num_dimensions=2, verbose=verbose)

    # Return results as a DataFrame
    df_em = pd.DataFrame([(code,) for code in codes], columns=["code"])
    df_em["code_2d_0"] = embeddings_2d[:,0]
    df_em["code_2d_1"] = embeddings_2d[:,1]
    return df_em
