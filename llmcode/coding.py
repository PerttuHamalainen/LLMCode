import random
import pandas as pd
import os
import openai
import numpy as np
import pickle
import hashlib
import json
import scipy
import shutil
import tiktoken
import re
import asyncio
import time
from itertools import chain
from sklearn.neighbors import NearestNeighbors


#progress bar helper
def printProgressBar (iteration, total, prefix = '', suffix = '', decimals = 1, length = 100, fill = '█', printEnd = "\r"):
    """
    Call in a loop to create terminal progress bar
    @params:
        iteration   - Required  : current iteration (Int)
        total       - Required  : total iterations (Int)
        prefix      - Optional  : prefix string (Str)
        suffix      - Optional  : suffix string (Str)
        decimals    - Optional  : positive number of decimals in percent complete (Int)
        length      - Optional  : character length of bar (Int)
        fill        - Optional  : bar fill character (Str)
        printEnd    - Optional  : end character (e.g. "\r", "\r\n") (Str)
    """
    percent = ("{0:." + str(decimals) + "f}").format(100 * (iteration / float(total)))
    filledLength = int(length * iteration // total)
    bar = fill * filledLength + '-' * (length - filledLength)
    print(f'\r{prefix} |{bar}| {percent}% {suffix}', end = printEnd)
    # Print New Line on Complete
    if iteration == total:
        print()

cache_dir = os.path.join(os.path.dirname(os.path.realpath(__file__)),"cache")

def cache_keys_equal(key1,key2):
    if (type(key1) is np.ndarray) and (type(key2) is np.ndarray):
        return np.array_equal(key1,key2)
    return key1==key2

def cache_hash(key):
    return hashlib.md5(key).hexdigest()

def load_cached(key):
    cached_name= cache_dir + "/" + cache_hash(key)
    if os.path.exists(cached_name):
        cached=pickle.load(open(cached_name,"rb"))
        if cache_keys_equal(cached["key"],key):
            #cache_copy_dir = os.path.join(os.path.dirname(os.path.realpath(__file__)), "cache_copy") #for debugging which files are actually used...
            #shutil.copy(cached_name, cache_copy_dir+"/" + cache_hash(key))
            return cached["value"]
    return None

def cache(key,value):
    if not os.path.exists(cache_dir):
        os.makedirs(cache_dir,exist_ok=True)
    cached_name= cache_dir + "/" + cache_hash(key)
    pickle.dump({"key":key,"value":value},open(cached_name,"wb"))


tiktoken_encodings = {
    "gpt-4": tiktoken.get_encoding("cl100k_base"),
    "gpt-3.5-turbo": tiktoken.get_encoding("cl100k_base"),
    "gpt-3.5-turbo-instruct": tiktoken.get_encoding("cl100k_base"),
    "gpt-3.5-turbo-16k": tiktoken.get_encoding("cl100k_base"),
    "text-davinci-003": tiktoken.get_encoding("p50k_base"),
    "text-davinci-002": tiktoken.get_encoding("p50k_base"),
    "text-davinci-001": tiktoken.get_encoding("r50k_base"),
    "text-curie-001": tiktoken.get_encoding("r50k_base"),
    "text-babbage-001": tiktoken.get_encoding("r50k_base"),
    "text-ada-001": tiktoken.get_encoding("r50k_base"),
    "davinci": tiktoken.get_encoding("r50k_base"),
    "curie": tiktoken.get_encoding("r50k_base"),
    "babbage": tiktoken.get_encoding("r50k_base"),
    "ada": tiktoken.get_encoding("r50k_base"),
}

max_llm_context_length = {
    "gpt-3.5-turbo-16k": 16384,
    "gpt-4": 8192,
    "gpt-3.5-turbo": 4096,
    "gpt-3.5-turbo-instruct": 4096,
    "text-davinci-003": 4096,
    "text-davinci-002": 4096,
    "text-davinci-001": 2049,
    "text-curie-001": 2049,
    "text-babbage-001": 2049,
    "text-ada-001": 2049,
    "davinci": 2049,
    "curie": 2049,
    "babbage": 2049,
    "ada": 2049
}

def is_chat_model(model):
    return ("gpt-4" in model) or ("gpt-3.5-turbo" in model) and ("gpt-3.5-turbo-instruct" not in model)

def token_overhead(model):
    if is_chat_model(model):
        return 300 #these models have some overhead because of the system message and chat structure
    return 0

def num_tokens_from_string(string: str, model: str) -> int:
    """Returns the number of tokens in a text string."""
    if not model in tiktoken_encodings:
        raise Exception(f"Tiktoken encoding unknown for LLM: {model}")
    encoding = tiktoken_encodings[model]
    num_tokens = len(encoding.encode(string))
    return num_tokens

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


# Queries an LLM for continuations of a batch of prompts given as a list
def query_LLM(model, prompt_batch, max_tokens, use_cache):
    openai.api_key = os.getenv("OPENAI_API_KEY")
    cache_key=model.join(prompt_batch).encode('utf-8')
    if use_cache:
        cached_result=load_cached(cache_key)
        if cached_result is not None:
            return cached_result

    #choose whether to use the chat API or the older query API (the latter is a better fit for data coding,
    #but GPT-4 series of models is only available via former)
    if is_chat_model(model):
        system_message = "You are a helpful assistant that continues the text given by the user."

        # each batch in the prompt becomes its own asynchronous chat completion request
        async def batch_request(prompt_batch):
            tasks=[]
            for prompt in prompt_batch:
                messages = [
                    {"role": "system", "content": ""},
                    {"role": "user", "content": prompt},
                ]
                tasks.append(openai.ChatCompletion.acreate(
                    model=model,
                    messages=messages,
                    temperature=0.0,
                    max_tokens=max_tokens,
                    n=1,  # one completion per prompt
                    stop="###",
                    frequency_penalty=0.0,
                    presence_penalty=0.0,
                ))
            return await asyncio.gather(*tasks)

        loop = asyncio.get_event_loop()
        responses = loop.run_until_complete(batch_request(prompt_batch))
        continuations = [response["choices"][0]["message"]["content"].strip() for response in responses]

        # before we return the continuations, ensure that we don't violate OpenAI's rate limits
        total_tokens = 0
        for prompt in prompt_batch:
            total_tokens += num_tokens_from_string(string=system_message, model=model)
            total_tokens += num_tokens_from_string(string=prompt, model=model)
        for continuation in continuations:
            total_tokens += num_tokens_from_string(string=continuation, model=model)
        max_tokens_per_minute = 90000  # currently imposed limit for ChatGPT models
        wait_seconds = (total_tokens / max_tokens_per_minute) * 60.0
        #print(f"Waiting {wait_seconds} seconds to ensure staying within rate limit")
        time.sleep(wait_seconds)

    else:
        # OpenAI API query
        openai.api_key = os.getenv("OPENAI_API_KEY")
        response = openai.Completion.create(
            model=model,
            prompt=prompt_batch,
            temperature=0.0,  # in this use case we prefer to get the single max prob continuation
            max_tokens=max_tokens,
            top_p=1.0,  # get the most probable topic
            frequency_penalty=0.0,
            presence_penalty=0.0,
            n=1  # one completion per prompt
        )
        # extract continuations
        continuations = [response["choices"][i]["text"] for i in range(len(prompt_batch))]

    if use_cache:
        cache(key=cache_key,value=continuations)
    return continuations




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

    #Query the LLM in batches
    continuations=[]
    batch_size = 10  # The exact max batch_size for each model is unknown. This seems to work for all, and provides a nice speed-up.
    N = len(texts)
    for i in range(0, N, batch_size):
        if verbose:
            print(f"Coding texts {i}:{min([N, i + batch_size])} of total {N}")
        prompt_batch=prompts[i:min([N, i + batch_size])]
        continuations+=query_LLM(model=gpt_model,
                                 prompt_batch=prompt_batch,
                                 max_tokens=max_tokens,
                                 use_cache=use_cache)
        printProgressBar(min([N, i + batch_size]), N)

    print("")

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


def embed(texts,use_cache=True,model="text-embedding-ada-002"):
    cache_key=(model+("".join(texts))).encode('utf-8')
    if use_cache:
        cached_result=load_cached(cache_key)
        if cached_result is not None:
            print("Loaded embeddings from cache, hash", cache_hash(cache_key))
            return cached_result


    #query embeddings from the API
    texts=[json.dumps(s) for s in texts]  #make sure we escape quotes in a way compatible with GPT-3 API's internal use of json
    openai.api_key = os.getenv("OPENAI_API_KEY")
    batch_size = 32
    N = len(texts)

    embed_matrix=[]
    for i in range(0, N, batch_size):
        printProgressBar(i, N)
        embed_batch=texts[i:min([N, i + batch_size])]
        embeddings = openai.Embedding.create(input=embed_batch, model=model)
        for j in range(len(embed_batch)):
            embed_matrix.append(embeddings['data'][j]['embedding'])
    print("")
    embed_matrix=np.array(embed_matrix)
    #dim = len(embeddings['data'][0]['embedding'])
    #embed_matrix = np.zeros([N, dim])
    #for i in range(N):
    #    embed_matrix[i, :] = embeddings['data'][i]['embedding']

    #update cache
    if use_cache:
        cache(cache_key,embed_matrix)

    #return results
    return embed_matrix


def reduce_embedding_dimensionality(embeddings,num_dimensions,method="UMAP",use_cache=True,n_neighbors=None):
    if isinstance(embeddings,list):
        #embeddings is a list of embedding matrices => pack all to one big matrix for joint dimensionality reduction
        all_emb = np.concatenate(embeddings, axis=0)
    else:
        all_emb = embeddings
    def unpack(x,embeddings_list):
        row = 0
        result = []
        for e in embeddings_list:
            N = e.shape[0]
            result.append(x[row:row + N])
            row += N
        return result

    cache_key=(str(all_emb.tostring())+str(num_dimensions)+method+str(n_neighbors)).encode('utf-8')
    if use_cache:
        cached_result=load_cached(cache_key)
        if cached_result is not None:
            print("Loaded dimensionality reduction results from cache, hash ", cache_hash(cache_key))
            if isinstance(embeddings, list):
                return unpack(cached_result,embeddings)
            else:
                return cached_result
    from sklearn.manifold import MDS
    from sklearn.manifold import TSNE
    import umap
    from sklearn.decomposition import PCA
    #cosine distance
    all_emb=all_emb/np.linalg.norm(all_emb,axis=1,keepdims=True)

    if method=="MDS":
        mds=MDS(n_components=num_dimensions,dissimilarity="precomputed")
        cosine_sim = np.inner(all_emb, all_emb)
        cosine_dist = 1 - cosine_sim
        x=mds.fit_transform(cosine_dist)
    elif method=="TSNE":
        tsne=TSNE(n_components=num_dimensions)
        x=tsne.fit_transform(all_emb)
    elif method=="PCA":
        pca=PCA(n_components=num_dimensions)
        x=pca.fit_transform(all_emb)
    elif method=="UMAP":
        if n_neighbors is None:
            n_neighbors=5
        reducer = umap.UMAP(n_components=num_dimensions,metric='cosine',n_neighbors=n_neighbors)
        x=reducer.fit_transform(all_emb)
    else:
        raise Exception("Invalid dimensionality reduction method!")

    if use_cache:
        cache(cache_key,x)

    if isinstance(embeddings, list):
        return unpack(x,embeddings)
    return x


def set_cache_directory(dir):
    global cache_dir
    cache_dir=dir

def get_cache_directory():
    return cache_dir

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




def gpt_human_code_dist(df,embedding_context,embedding_model):
    result=df.copy()

    # extract individual codes
    gpt_codes = df["codes"].tolist()
    human_codes = df["human_codes"].tolist()
    gpt_codes_split=[]
    human_codes_split=[]
    for i in range(df.shape[0]):
        gpt_codes_split.append([code.strip() for code in gpt_codes[i].split(";")])
        human_codes_split.append([code.strip() for code in human_codes[i].split(";")])
    gpt_codes = list(chain(*gpt_codes_split))
    human_codes = list(chain(*human_codes_split))

    # add context and generate embeddings
    gpt_codes_with_context = [code + embedding_context for code in gpt_codes]
    gpt_embeddings = embed(gpt_codes_with_context, use_cache=True, model=embedding_model)
    human_codes_with_context = [code + embedding_context for code in human_codes]
    human_embeddings = embed(human_codes_with_context, use_cache=True, model=embedding_model)

    # compute the modified Hausdorff distances between the gpt and human codes assigned to each text
    hausdorff_distances=[]
    i_gpt=0
    i_human=0
    for i in range(df.shape[0]):
        n_gpt=len(gpt_codes_split[i])
        n_human=len(human_codes_split[i])
        hausdorff_distances.append(hausdorff_embedding_distance(
            A=gpt_embeddings[i_gpt:i_gpt+n_gpt],
            B=human_embeddings[i_human:i_human+n_human]
        ))
        i_gpt+=n_gpt
        i_human+=n_human
    hausdorff_distances=np.array(hausdorff_distances)
    hausdorff_distances=np.where(hausdorff_distances < 1e-10, 0, hausdorff_distances) #clean floating point accuracy errors
    result["dist"]=hausdorff_distances
    result.rename(columns={"codes":"gpt_codes"},inplace=True)

    # sort the result based on similarity
    result_sorted = result.sort_values('dist', ascending=True)
    return result_sorted


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
    grouping_instructions,
    use_cache,
    other_threshold=None,
    batch_size=None,
    verbose=False):

    #get codes to group
    codes=df_codes["code"].astype(str).to_list()
    random.shuffle(codes) #to make the batches more representative, in case the input is somehow sorted

    #extract the few-shot code and theme examples from the original data_df
    df_ex = df_data[df_data['use_as_example'] == 1]
    df_ex = df_ex[df_ex['human_themes'].notnull()]
    print("Grouping codes into themes using the following human-defined examples:")
    print(df_ex[["human_codes","human_themes"]].head(df_ex.shape[0]))
    few_shot_code_strings = df_ex['human_codes'].astype(str).tolist()
    few_shot_theme_strings = df_ex['human_themes'].astype(str).tolist()
    n_few_shot_strings=len(few_shot_theme_strings)
    few_shot_codes=[]
    few_shot_themes=[]
    for i in range(n_few_shot_strings):
        c=string_to_codes(few_shot_code_strings[i])
        t=string_to_codes(few_shot_theme_strings[i])
        if len(c)!=len(t):
            raise Exception(f"The number of examples codes does not match the example themes. Codes: {few_shot_code_strings[i]}, themes: {few_shot_theme_strings[i]}.")
        few_shot_codes+=c
        few_shot_themes+=t
    n_few_shots=len(few_shot_codes)

    #remove the few-shot examples from the codes to group
    #print("all codes: ",codes)
    #print("few-shot codes",few_shot_codes)

    for code in few_shot_codes:
        if not (code in codes):
            raise Exception(f"Few-shot code '{code}' not in the codes to group")
        codes.remove(code)

    #with batch_size None, the user can try coding everything in a single batch,
    #if that happens to fit the LLM context
    if batch_size is None:
        batch_size=len(codes)

    #Query the LLM in batches, until all codes have been grouped
    #Note that depending on the LLM, each batch query may not yield themes for all codes in the batch
    #This is why the exact number of batches cannot be pre-calculated
    # TODO: chat mode? With OpenAI models, this would be cheaper and allow larger context
    grouped_codes=[]
    themes={}
    while len(codes)>0:
        n_batch=min([len(codes), batch_size])
        code_batch=codes[:n_batch]

        #for others than the first batch, we add a random selection of previously grouped codes
        #to the few-shot examples, to ensure consistency of grouping
        few_shot_codes_batch = few_shot_codes.copy()
        few_shot_themes_batch = few_shot_themes.copy()
        if len(grouped_codes)>0:
            extra_examples=grouped_codes.copy()
            random.shuffle(extra_examples)
            extra_examples=extra_examples[:min([len(grouped_codes),batch_size])]
            few_shot_codes_batch+=extra_examples
            few_shot_themes_batch+=[themes[x] for x in extra_examples]

        # construct prompt
        prompt = grouping_instructions + "\n\n" + "Codes:\n\n"
        for code in few_shot_codes:  # first the examples (so that they can be first in the completion)
            prompt += code + "\n"
        for code in code_batch:  # then the rest
            prompt += code + "\n"
        prompt += "\nThe codes above again in the same order, one code per row, with the corresponding theme in square brackets after each code:\n\n"
        for i in range(n_few_shots):  # first the examples (so that they can be first in the completion)
            prompt += f"{few_shot_codes[i]} [{few_shot_themes[i]}]\n"
        prompt += codes[0] + " ["

        # query
        print(f"Prompting for themes for {n_batch} codes out of {len(codes)} remaining...")
        if verbose:
            print("Using the following prompt:")
            print(prompt)
        max_tokens = max_llm_context_length[grouping_model] - num_tokens_from_string(prompt, grouping_model)
        max_tokens -= token_overhead(grouping_model)

        continuation = query_LLM(model=grouping_model,
                                 prompt_batch=[prompt],
                                 max_tokens=max_tokens,
                                 use_cache=use_cache)[0]
        if verbose:
            print(continuation)

        # extract themes from the LLM continuation
        continuation = codes[0] + " [" + continuation #add back the first code, for easier parsing
        lines=continuation.split("\n")
        n_extracted=0
        for line in lines:
            #print("Parsing line: ",line)
            themes_line = re.findall(r'\[(.*?)\]', line)
            n_themes_line = len(themes_line)
            if n_themes_line!=1:
                print("Warning: LLM returned other than 1 theme per line, skipping this line: ",line)
            else:
                theme=themes_line[0].strip()
                code=line[:line.find(theme)-1].strip()
                if not code in codes:
                    print(f"Warning: LLM returned code '{code}' that is not (anymore) in the list of remaining codes, possibly a duplicate return. Skipping this line: ", line)
                else:
                    if code in grouped_codes:
                        print(f"Warning: LLM returned code '{code}' that has already been grouped. Skipping this line: ",line)
                    else:
                        n_extracted += 1
                        grouped_codes.append(code)
                        themes[code]=theme
                        codes.remove(code)
        print(f"LLM returned valid themes for {n_extracted} / {n_batch} codes.")
    #Grouping done. For completeness, add the few-shot examples to the results
    for i in range(n_few_shots):
        themes[few_shot_codes[i]]=few_shot_themes[i]

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

    #Replace many very small themes with "Other"
    if other_threshold is not None:
        df_other=df_grouped[df_grouped["theme_count"]<=other_threshold].copy()
        df_rest=df_grouped[df_grouped["theme_count"]>other_threshold].copy()
        df_other["theme"]="Other"
        df_other["theme_count"]=df_other["count"].sum()
        df_grouped=pd.concat([df_rest,df_other],axis=0)

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

def frechet_embedding_distance(A,B,A_counts=None,B_counts=None):
    '''
    Frechet distance calculation adapted from (https://machinelearningmastery.com/how-to-implement-the-frechet-inception-distance-fid-from-scratch/)

    :param A: Numpy array of embedding vectors, shape [num_vectors,num_embedding_dimensions]
    :param B: Numpy array of embedding vectors, shape [num_vectors,num_embedding_dimensions]
    :param A_counts: Numpy array of counts (frequencies) for each A vector
    :param B_counts: Numpy array of counts (frequencies) for each B vector
    :return:
    '''

    if A_counts is None:
        A_counts=np.ones(A.shape[0])
    if B_counts is None:
        B_counts=np.ones(B.shape[0])
    # calculate mean and covariance statistics
    mu1, sigma1 = np.average(A,axis=0,weights=A_counts), np.cov(A, rowvar=False,fweights=A_counts)
    mu2, sigma2 = np.average(B,axis=0,weights=B_counts), np.cov(B, rowvar=False,fweights=B_counts)
    # calculate sum squared difference between means
    ssdiff = np.sum((mu1 - mu2) ** 2.0)
    # calculate sqrt of product between cov
    covmean = scipy.linalg.sqrtm(sigma1.dot(sigma2))
    # check and correct imaginary numbers from sqrt
    if np.iscomplexobj(covmean):
        covmean = covmean.real
    # calculate score
    fd = ssdiff + np.trace(sigma1 + sigma2 - 2.0 * covmean)
    return fd

def hausdorff_embedding_distance(A,B,A_counts=None,B_counts=None):
    '''
    Modified Hausdorff distance calculation

    :param A: Numpy array of embedding vectors, shape [num_vectors,num_embedding_dimensions]
    :param B: Numpy array of embedding vectors, shape [num_vectors,num_embedding_dimensions]
    :param A_counts: Numpy array of counts (frequencies) for each A vector
    :param B_counts: Numpy array of counts (frequencies) for each B vector
    :return:
    '''

    if A_counts is None:
        A_counts=np.ones(A.shape[0])
    if B_counts is None:
        B_counts=np.ones(B.shape[0])

    A/=np.linalg.norm(A,axis=1,keepdims=True)
    B/=np.linalg.norm(B,axis=1,keepdims=True)
    cosine_sim = np.inner(A, B)  #result shape [n_A,n_B]
    cosine_dist = 1.0 - cosine_sim  #cosine distance

    res = np.sum(A_counts*np.min(cosine_dist,axis=1))+np.sum(B_counts*np.min(cosine_dist,axis=0))
    res = res / (np.sum(A_counts)+np.sum(B_counts))

    return res
    #return np.sum(A_counts*np.min(dist,axis=1))+np.sum(B_counts*np.min(dist,axis=0))
    #return np.max(np.min(dist,axis=1))+np.max(np.min(dist,axis=0))

def KNN_precision_and_recall(generated,real,generated_counts=None,real_counts=None,k=5,knn_algorithm='brute'):
    '''
    Kynkäänniemi et al. precision and recall estimation for generated data.

    :param generated: generated data, shape [num_vectors,dim]
    :param real: real data (ground truth), shape [num_vectors,dim]
    :param generated_counts: counts (frequencies) for each generated vector
    :param real_counts: counts (frequencies) for each real vector
    :param k: number of nearest neighbors to use. Low values are noisy, but high values may provide optimistic results
    :param knn_algorithm: the nearest neighbors algorithm to use (see sklearn.neighbors.NearestNeighbors)
    :return: tuple: (precision,recall)
    '''

    def KNN_support(A, B, k, knn_algorithm):
        '''
        For each vector in A, checks whether it lies within the support of the distribution of B vectors.
        This is implemented by checking whether the vector lies within a hypersphere centered at the closest B vector,
        hypersphere radius set equal to the distance between the B vector and its k:th nearest neighbor in B.
        This is the core of the Kynkäänniemi et al. method for precision and recall estimation of generative models.
        WARNING: you shouldn't call this with a B matrix that has vectors replicated based on frequencies, as this will mess up
        the k:th nearest neighbor distance measures, which may become zero for replicated vectors.
        '''
        # Build KNN lookup structure for B
        nbrs = NearestNeighbors(n_neighbors=k, algorithm=knn_algorithm).fit(B)
        distances, _ = nbrs.kneighbors(B, n_neighbors=k)
        kth_b_dist = distances[:, -1]
        ab_distances, indices = nbrs.kneighbors(A, n_neighbors=1)
        ab_closest_dist = ab_distances[:, 0]
        b_search_ball_radiuses = kth_b_dist[indices[:, -1]]
        assert (ab_closest_dist.shape == b_search_ball_radiuses.shape)
        return ab_closest_dist <= b_search_ball_radiuses

    if generated_counts is None:
        generated_counts = np.ones(generated.shape[0])
    if real_counts is None:
        real_counts = np.ones(real.shape[0])

    precision=np.average(KNN_support(A=generated, B=real, k=k, knn_algorithm=knn_algorithm),weights=generated_counts)
    recall=np.average(KNN_support(A=real, B=generated, k=k, knn_algorithm=knn_algorithm),weights=real_counts)
    return precision,recall


def KNN_density(embeddings,counts=None,k=5,knn_algorithm="brute"):

    nbrs = NearestNeighbors(n_neighbors=k, algorithm=knn_algorithm).fit(embeddings)
    distances, indices = nbrs.kneighbors(embeddings, n_neighbors=k)
    dim=embeddings.shape[1]
    kth_distances=distances[:, -1]
    hypersphere_volumes=np.power(kth_distances,dim)
    if counts is None:
        return 1.0/hypersphere_volumes
    else:
        n=np.sum(counts[indices],axis=1)
        return n/hypersphere_volumes

def KNN_counts(embeddings,counts,k=5,knn_algorithm="brute"):
    nbrs = NearestNeighbors(n_neighbors=k, algorithm=knn_algorithm).fit(embeddings)
    _, indices = nbrs.kneighbors(embeddings, n_neighbors=k)
    return np.sum(counts[indices],axis=1)

