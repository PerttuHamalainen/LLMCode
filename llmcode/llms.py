"""

Generic LLM helpers:

- querying with a list of multiple prompts
- caching LLM responses to save API costs
- tokenization
- handling different APIs

"""


import random
import pandas as pd
import os
import openai
from openai import AzureOpenAI
import httpx
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
import json
from itertools import chain
from sklearn.neighbors import NearestNeighbors


# globals: OpenAI client instances
client = None
embed_client = None
client_async = None
API_type = None
__all__=["API_type","client","embed_client","client_async"]

'''
Set up rewriting the base path with Aalto mappings
For all endpoints see https://www.aalto.fi/en/services/azure-openai#6-available-api-s
'''
# prior to making any Aalto API requests, we will update this with the desired model's OpenAI name
current_openai_model = None
# mapping from OpenAI model names to Aalto API URLs
openai2aalto = {
    "gpt-3.5-turbo": "/v1/chat",
    "gpt-4-turbo": "/v1/openai/gpt4-turbo/chat/completions",
    "gpt-4o": "/v1/openai/gpt4o/chat/completions",
    "text-embedding-3-large": "/v1/openai/text-embedding-3-large/embeddings",
    "text-embedding-ada-002": "/v1/openai/ada-002/embeddings"
}


def update_base_url_for_aalto(request: httpx.Request) -> None:
    '''
    A callback that the Aalto OpenAI clients will use to append the base API url with model URL
    '''
    if request.url.path == "/chat/completions":
        if current_openai_model not in openai2aalto:
            raise Exception(f"Model {current_openai_model} not available via the Aalto API")
        request.url = request.url.copy_with(path=openai2aalto[current_openai_model])


def init(API):
    '''
    This must be called before calling QueryLLM or other methods that make GPT API calls.

    :param API: Either "OpenAI" or "Aalto"
    '''
    global client
    global embed_client
    global client_async
    global API_type
    API_type = API

    if API == "OpenAI":
        assert (
                "OPENAI_API_KEY" in os.environ and os.environ.get("OPENAI_API_KEY") != ""
        ), "you must set the `OPENAI_API_KEY` environment variable."
        client = openai.OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))
        embed_client = client
        client_async = openai.AsyncOpenAI(api_key=os.environ.get("OPENAI_API_KEY"))
    elif API == "Aalto":
        # create chat client
        assert (
                "AALTO_OPENAI_API_KEY" in os.environ and os.environ.get("AALTO_OPENAI_API_KEY") != ""
        ), "you must set the `AALTO_OPENAI_API_KEY` environment variable."

        client = openai.OpenAI(
            base_url="https://aalto-openai-apigw.azure-api.net",
            api_key=False,  # API key not used, and rather set below
            default_headers={
                "Ocp-Apim-Subscription-Key": os.environ.get("AALTO_OPENAI_API_KEY"),
            },
            http_client=httpx.Client(
                event_hooks={"request": [update_base_url_for_aalto]}
            ),
        )

        # create embedding client
        auth_headers = {
            'Ocp-Apim-Subscription-Key': os.environ.get("AALTO_OPENAI_API_KEY")
        }
        embed_client = AzureOpenAI(
            api_key="not_in_use",  # This attribute is required but it is not in use
            api_version="2024-06-01",
            azure_endpoint="https://aalto-openai-apigw.azure-api.net/v1/",
            default_headers=auth_headers
        )
        '''
        assert (
                "OPENAI_API_KEY" in os.environ and os.environ.get("OPENAI_API_KEY") != ""
        ), "you must set the `OPENAI_API_KEY` environment variable."


        #Aalto embedding models give an error: {'statusCode': 404, 'message': 'Resource not found'}
        #For now, we use OpenAI embeddings instead, which should be GDPR-safe because we only embed codes instead of the raw data
        assert (
                "OPENAI_API_KEY" in os.environ and os.environ.get("OPENAI_API_KEY") != ""
        ), "you must set the `OPENAI_API_KEY` environment variable."
        embed_client = openai.OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))


        #Asynchronous processing disabled for now, as it gives weird errors with Aalto models
        client_async=None

        client_async = openai.AsyncAzureOpenAI(
            base_url="https://aalto-openai-apigw.azure-api.net",
            api_key=os.environ.get("AALTO_OPENAI_API_KEY"), #False, # API key not used, and rather set below
            api_version="2023-05-15",
            default_headers = {
                "Ocp-Apim-Subscription-Key": os.environ.get("AALTO_OPENAI_API_KEY"),
            },
            http_client=httpx.AsyncClient(
                event_hooks={ "request": [update_base_url_for_aalto] }
            ),
        )
        '''
    else:
        raise Exception(f"Invalid LLM API: {API}")


# progress bar helper
def print_progress_bar(iteration, total, prefix='', suffix='', decimals=1, length=100, fill='█', printEnd="\r"):
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
    print(f'\r{prefix} |{bar}| {percent}% {suffix}', end=printEnd)
    # Print New Line on Complete
    if iteration == total:
        print()


cache_dir = "./_LLMCode_cache" #os.path.join(os.path.dirname(os.path.realpath(__file__)), "../_LLM_cache")


def cache_keys_equal(key1, key2):
    if (type(key1) is np.ndarray) and (type(key2) is np.ndarray):
        return np.array_equal(key1, key2)
    return key1 == key2


def cache_hash(key):
    return hashlib.md5(key).hexdigest()


def load_cached(key):
    cached_name = cache_dir + "/" + cache_hash(key)
    if os.path.exists(cached_name):
        cached = pickle.load(open(cached_name, "rb"))
        if cache_keys_equal(cached["key"], key):
            # cache_copy_dir = os.path.join(os.path.dirname(os.path.realpath(__file__)), "cache_copy") #for debugging which files are actually used...
            # shutil.copy(cached_name, cache_copy_dir+"/" + cache_hash(key))
            return cached["value"]
    return None


def cache(key, value):
    if not os.path.exists(cache_dir):
        os.mkdir(cache_dir)
    cached_name = cache_dir + "/" + cache_hash(key)
    pickle.dump({"key": key, "value": value}, open(cached_name, "wb"))


tiktoken_encodings = {
    "gpt-4o": tiktoken.get_encoding("cl100k_base"),
    "gpt-4o-mini": tiktoken.get_encoding("cl100k_base"),
    "gpt-4-turbo": tiktoken.get_encoding("cl100k_base"),
    "gpt-4-turbo-preview": tiktoken.get_encoding("cl100k_base"),
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
    "gpt-4o": 128000,
    "gpt-4-turbo": 16384 * 2,
    "gpt-4-turbo-preview": 16384 * 2,
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
        return 300  # these models have some overhead because of the system message and chat structure
    return 0


def num_tokens_from_string(string: str, model: str) -> int:
    """Returns the number of tokens in a text string."""
    if not model in tiktoken_encodings:
        raise Exception(f"Tiktoken encoding unknown for LLM: {model}")
    encoding = tiktoken_encodings[model]
    num_tokens = len(encoding.encode(string))
    return num_tokens




# Queries an LLM for continuations of a batch of prompts given as a list
def query_LLM_batch(model, prompt_batch, max_tokens, use_cache=None, temperature=None,system_message=None,stop=None):
    global current_openai_model
    current_openai_model = model    #needed for the Aalto Azure GPT API callbacks

    if temperature is None:
        temperature=0 #by default, operate fully deterministically

    if use_cache is None:
        use_cache=False
    cache_key=(API_type+"_"+model+"".join(prompt_batch)).encode('utf-8')
    if use_cache:
        cached_result=load_cached(cache_key)
        if cached_result is not None:
            return cached_result

    start_time = time.time()

    #choose whether to use the chat API or the older query API
    if is_chat_model(model):
        if system_message is None:
            system_message = "You are a helpful assistant."
        if API_type=="Aalto" and (client_async is None):
            # In case no async API client, fall back to running the prompts one-by-one
            continuations=[]
            for prompt in prompt_batch:
                success=False
                while not success:
                    try:
                        response = client.chat.completions.create(
                            model=model,
                            # the model variable must be set, but has no effect, model selection done in update_base_url_for_aalto
                            messages=[
                                {"role": "system", "content": system_message},
                                {"role": "user", "content": prompt},
                            ],
                        )
                        success=True
                    except openai.RateLimitError:
                        print("Rate limit error! Will retry in 5 seconds")
                        time.sleep(5)
                    if response is None or response.choices is None:
                        print("No response from API, will retry in 5 seconds. Check VPN settings if you're not in Aalto intranet.")
                        time.sleep(5)
                        success=False
                if response.choices[0].message.content is None:
                    continuations.append("")
                else:
                    continuations.append(response.choices[0].message.content.strip())
        else:
                # each batch in the prompt becomes its own asynchronous chat completion request
            async def batch_request(prompt_batch):
                tasks=[]
                for prompt in prompt_batch:
                    messages = [
                        {"role": "system", "content": system_message},
                        {"role": "user", "content": prompt},
                    ]
                    tasks.append(client_async.chat.completions.create(
                        model=model,
                        messages=messages,
                        temperature=temperature,
                        max_tokens=max_tokens,
                        n=1,  # one completion per prompt
                        stop=stop,
                        frequency_penalty=0.0,
                        presence_penalty=0.0,
                    ))
                return await asyncio.gather(*tasks)

            loop = asyncio.get_event_loop()
            responses = loop.run_until_complete(batch_request(prompt_batch))
            continuations = [response.choices[0].message.content.strip() for response in responses]

        # before we return the continuations, ensure that we don't violate OpenAI's rate limits
        total_tokens = 0
        for prompt in prompt_batch:
            total_tokens += num_tokens_from_string(string=system_message, model=model)
            total_tokens += num_tokens_from_string(string=prompt, model=model)
        for continuation in continuations:
            total_tokens += num_tokens_from_string(string=continuation, model=model)
        max_tokens_per_minute = 600000  # currently imposed limit for GPT-4o on tier 5 is 30 million TPM
        wait_seconds = (total_tokens / max_tokens_per_minute) * 60.0
        #print(f"Waiting {wait_seconds} seconds to ensure staying within rate limit")
        time_elapsed=time.time() - start_time
        if time_elapsed<wait_seconds:
            time.sleep(wait_seconds-time_elapsed)

    else:
        # The old completions API supports batched prompts out-of-the-box
        response = client.completions.create(
            model=model,
            prompt=prompt_batch,
            temperature=temperature,
            max_tokens=max_tokens,
            top_p=1.0,
            frequency_penalty=0.0,
            presence_penalty=0.0,
            stop=stop,
            n=1  # one completion per prompt
        )
        # extract continuations
        continuations = [choice.text for choice in response.choices]

        # before we return the continuations, ensure that we don't violate OpenAI's rate limits
        total_tokens = 0
        for prompt in prompt_batch:
            total_tokens += num_tokens_from_string(string=prompt, model=model)
        for continuation in continuations:
            total_tokens += num_tokens_from_string(string=continuation, model=model)
        max_tokens_per_minute = 90000  # currently imposed limit for ChatGPT models
        wait_seconds = (total_tokens / max_tokens_per_minute) * 60.0
        time_elapsed=time.time() - start_time
        if time_elapsed<wait_seconds:
            time.sleep(wait_seconds-time_elapsed)

    if use_cache:
        cache(key=cache_key,value=continuations)
    return continuations

def query_LLM(prompts, model=None, max_tokens=None, use_cache=None, temperature=None, system_message=None,stop=None):
    """
        Query a Language Model (LLM) with one or more prompts.

        This function sends prompts to a specified language model and returns the generated continuations for each prompt in batches.

        Args:
            prompts (str or list of str): The prompt or list of prompts to be sent to the LLM. If a single string is provided, it will be wrapped in a list.
            model (str, optional): The identifier for the model to be queried. Defaults to "gpt-4o".
            max_tokens (int, optional): The maximum number of tokens to generate in each completion. If None, the model's default will be used.
            use_cache (bool, optional): Whether to use cached results if available. Defaults to False.
            temperature (float, optional): Sampling temperature to use, in range (0, 1]. Higher values mean the model will take more risks. Defaults to 0.
            system_message (str, optional): A system message that can influence the generated response. Defaults to None.
            stop (str or list of str, optional): Sequences where the model will stop producing further tokens. Defaults to None.

        Returns:
            list of str: The continuations generated by the model for each input prompt.

        Examples:
            >>> responses = query_LLM("What is the capital of France?")
            >>> print(responses)
            ["The capital of France is Paris."]

            >>> prompts = ["Translate the following sentence to French: 'Hello, world!'", "Translate the following sentence to Spanish: 'Good morning!'"]
            >>> responses = query_LLM(prompts, model="gpt-3.5-turbo", max_tokens=50, temperature=0.5)
            >>> print(responses)
            ["Bonjour, le monde!", "¡Buenos días!"]
        """
    if model is None:
        model="gpt-4o"
    if use_cache is None:
        use_cache=False
    if temperature is None:
        temperature=0
    return_single=False
    if isinstance(prompts, str):
        prompts=[prompts] #the following code always expects a list of prompts
        return_single=True

    #Query the LLM in batches
    continuations=[]
    batch_size = 10  # The exact max batch_size for each model is unknown. This seems to work for all, and provides a nice speed-up.
    N = len(prompts)
    for i in range(0, N, batch_size):
        prompt_batch=prompts[i:min([N, i + batch_size])]
        continuations+=query_LLM_batch(model=model,
                                 prompt_batch=prompt_batch,
                                 max_tokens=max_tokens,
                                 use_cache=use_cache,
                                 temperature=temperature,
                                 system_message=system_message,
                                 stop=stop)
        if N>batch_size:
            print_progress_bar(min([N, i + batch_size]), N,printEnd="")
    return continuations[0] if return_single else continuations


## TODO: Integrate with other query_LLM function
def query_LLM_with_response_format(prompt, response_format, model="gpt-4o", max_tokens=None):
    """
    Queries an LLM with a specified prompt and returns the response parsed into the specified response format class.
    Useful for when you want a structured output.

    Args:
        prompt (str): The input text prompt for the model.
        response_format (Type[BaseModel]): The Pydantic model class to parse the response into.
        model (str, optional): The model to use for querying (default is 'gpt-4o').
        max_tokens (int, optional): Maximum number of tokens in the response (default is None).

    Returns:
        BaseModel: The parsed response as an instance of the specified Pydantic model class.
    """
    success = False
    while not success:
        try:
            response = client.beta.chat.completions.parse(
                model=model,
                messages=[
                    {"role": "user", "content": prompt},
                ],
                response_format=response_format,
                max_tokens=max_tokens,
            )
            success = True
        except openai.RateLimitError:
            print("Rate limit error! Will retry in 5 seconds")
            time.sleep(5)
    return response.choices[0].message.parsed


def set_cache_directory(dir):
    global cache_dir
    cache_dir=dir

def get_cache_directory():
    return cache_dir



def embed(texts,use_cache=None,model=None,verbose=True):

    if model is None:
        model = "text-embedding-ada-002"

    if use_cache is None:
        use_cache = True

    cache_key=(API_type+"_"+model+"".join(texts)).encode('utf-8')
    if use_cache:
        cached_result=load_cached(cache_key)
        if cached_result is not None:
            if verbose:
                print("Loaded embeddings from cache, hash", cache_hash(cache_key))
            return cached_result


    #query embeddings from the API
    texts=[json.dumps(s) for s in texts]  #make sure we escape quotes in a way compatible with GPT-3 API's internal use of json
    batch_size = 32
    N = len(texts)

    embed_matrix=[]
    for i in range(0, N, batch_size):
        print_progress_bar(i, N, printEnd="")
        embed_batch=texts[i:min([N, i + batch_size])]
        embeddings = embed_client.embeddings.create(input=embed_batch, model=model)
        for j in range(len(embed_batch)):
            embed_matrix.append(embeddings.data[j].embedding)
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


def reduce_embedding_dimensionality(embeddings,num_dimensions,method="UMAP",use_cache=True,n_neighbors=None,verbose=True):
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
            if verbose:
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
