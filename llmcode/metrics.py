"""

Various quality measures for extracting relevant passages of text and coding them

"""


import html
import re
import numpy as np
import pandas as pd
import scipy
from itertools import chain
from sklearn.neighbors import NearestNeighbors
from .llms import embed
from .coding import parse_codes

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


def extract_IoU(df,
                extracts_col,
                reference_col,
                verbose=True):
    # Helper for IoU calculation: remove spaces and punctuation from a string and convert to lower case, while preserving double asterisks
    def clean(text):
        # Regular expression to match sequences of double asterisks.
        preserved_pattern = r'\*\*'

        # Split the text into parts based on the preservation pattern.
        parts = re.split(f'({preserved_pattern})', text)

        # Define a regular expression pattern for spaces and punctuation.
        remove_pattern = r'\W+'  # r'[^\w\*\*]|_'

        # Process each part: remove spaces and punctuation from parts that are not "**".
        processed_parts = [part if part == "**" else re.sub(remove_pattern, '', part) for part in parts]

        # Join the processed parts back together.
        result = ''.join(processed_parts)
        return result.lower()

    # If the input data contains human ground truth extracts, calculate Intersection over Union as an extraction quality metric
    IoUs = []
    for _, row in df.iterrows():
        # get llm and human extracts
        llm = row[extracts_col]
        human = row[reference_col]

        if pd.isna(llm) or pd.isna(human):

            continue

        llm = clean(llm)
        human = clean(human)

        # construct numerical highlight arrays, one element per character
        def to_arr(s):
            s_raw = s.replace("**", "")
            result = [0] * len(s_raw)
            highlighting = 0
            pos = 0
            pos_out = 0
            l = len(s)
            while pos < l:
                if (pos < l - 1) and (s[pos:pos + 2] == "**"):
                    highlighting = 1 - highlighting
                    pos += 2
                else:
                    result[pos_out] = highlighting
                    pos_out += 1
                    pos += 1
            return np.array(result, dtype=np.int32)

        # print(f"human extract:{human}")
        # print(f"llm extract:{llm}")
        human_map = to_arr(human)
        llm_map = to_arr(llm)

        # pad to same length
        l = max([human_map.shape[0], llm_map.shape[0]])

        def pad_array(arr, l):
            if arr.shape[0] < l:
                arr = np.concatenate([arr, np.zeros(l - arr.shape[0], dtype=np.int32)])
            return arr

        human_map = pad_array(human_map, l)
        llm_map = pad_array(llm_map, l)

        # calculate IoU
        # IoU=np.sum(np.bitwise_and(human_map,llm_map))/np.sum(np.bitwise_or(human_map, llm_map))
        intersection = np.sum(np.bitwise_and(human_map, llm_map))
        union = np.sum(np.bitwise_or(human_map, llm_map))
        if intersection == 0 and union == 0:
            IoU = 1  # all correct with this one!
        else:
            IoU = intersection / union
        # print(f"IoU {IoU}")
        IoUs.append(IoU)
    # Calculate mean IoU
    IoUs = np.array(IoUs)
    if verbose:
        print(f"Average IoU: {np.mean(IoUs)}")

    # store output
    df=df.copy()
    df["IoU"] = IoUs

    # Write a .html with a table that shows the human and gpt extracts side by side, sorted by IoU
    df_html = df.copy()
    df_html = df_html[[reference_col, extracts_col, "IoU"]]
    df_html.sort_values(by='IoU', ascending=False, inplace=True)

    df_html[extracts_col] = df_html[extracts_col].map(_markdown_to_html)
    df_html[reference_col] = df_html[reference_col].map(_markdown_to_html)

    # Output the DataFrame as HTML
    html_output = df_html.to_html(escape=False)
    return IoUs,html_output


def _markdown_to_html(text):
    # Replace <sup> tags with placeholders
    text = text.replace("<sup>", "PLACEHOLDER_SUP_START").replace("</sup>", "PLACEHOLDER_SUP_END")
    # Escape the rest of the text
    text = html.escape(text)
    # Handle bold formatting
    while "**" in text:
        text = text.replace("**", "<b>", 1).replace("**", "</b>", 1)
    # Replace newlines with <br> tags
    text = text.replace("\n", "<br>")
    # Restore <sup> tags from placeholders
    text = text.replace("PLACEHOLDER_SUP_START", "<sup>").replace("PLACEHOLDER_SUP_END", "</sup>")
    return text


def _remove_codes(coded_text):
    return re.sub(r"<sup>.*?</sup>", "", coded_text)


def run_coding_eval(llm_coded_texts, human_coded_texts, embedding_context, embedding_model, sort_by="", use_cache=True, verbose=False):
    if len(llm_coded_texts) != len(human_coded_texts):
        raise ValueError("llm_coded_texts must be equal in length to human_coded_texts")
    
    llm_col = "llm_coded_text"
    reference_col = "human_coded_text"

    # Remove codes and calculate IoUs for highlights
    data_hl_only = [(_remove_codes(llm_t), _remove_codes(human_t)) for llm_t, human_t in zip(llm_coded_texts, human_coded_texts)]
    df_hl_only = pd.DataFrame(data_hl_only, columns=[llm_col, reference_col])
    IoUs, _ = extract_IoU(
        df=df_hl_only,
        extracts_col=llm_col,
        reference_col=reference_col,
        verbose=False
    )

    # Calculate hausdorff for code sets
    df_codes = pd.DataFrame(list(zip(llm_coded_texts, human_coded_texts)), columns=[llm_col, reference_col])
    df_codes["llm_codes"] = ["; ".join(code for _, code in parse_codes(coded_text)) if not pd.isna(coded_text) else None for coded_text in df_codes[llm_col]]
    df_codes["human_codes"] = ["; ".join(code for _, code in parse_codes(coded_text)) if not pd.isna(coded_text) else None for coded_text in df_codes[reference_col]]
    hausdorff_distances, _ = gpt_human_code_dist(
        df=df_codes,
        llm_col="llm_codes",
        reference_col="human_codes",
        embedding_context=embedding_context,
        embedding_model=embedding_model,
        use_cache=use_cache,
        verbose=verbose
    )

    # Collect data in output df
    data_out = list(zip(llm_coded_texts, human_coded_texts, IoUs, hausdorff_distances))
    df_out = pd.DataFrame(data_out, columns=[llm_col, reference_col, "IoU", "Hausdorff"])

    # Sort rows by metric if requested
    if sort_by.lower() == "iou":
        df_out.sort_values(by="IoU", ascending=False, inplace=True)
    elif sort_by.lower() == "hausdorff": 
        df_out.sort_values(by="Hausdorff", ascending=False, inplace=True)
    
    # Create html table output to show markdown
    df_html = df_out.copy()
    df_html[llm_col] = df_html[llm_col].map(_markdown_to_html)
    df_html[reference_col] = df_html[reference_col].map(_markdown_to_html)
    html_out = df_html.to_html(escape=False)

    return html_out, df_out


def gpt_human_code_dist(df, llm_col, reference_col, embedding_context, embedding_model, use_cache=True, verbose=True):
    # Extract individual codes split by text
    gpt_codes_split = [set(code.strip() for code in codes.split(";")) if not pd.isna(codes) else None for codes in df[llm_col]]
    human_codes_split = [set(code.strip() for code in codes.split(";")) if not pd.isna(codes) else None for codes in df[reference_col]]

    # Collect all GPT- and human-generated codes in one list, removing duplicates and None values
    all_gpt_codes = set(chain(*filter(lambda c: c is not None, gpt_codes_split)))
    all_human_codes = set(chain(*filter(lambda c: c is not None, human_codes_split)))
    all_codes = list(all_gpt_codes.union(all_human_codes))

    # Add context and generate embeddings for each code
    codes_with_context = [code + embedding_context for code in all_codes]
    embedding_matrix = embed(codes_with_context, use_cache=use_cache, model=embedding_model)
    code_embeddings = {code: embedding_matrix[idx] for idx, code in enumerate(all_codes)}

    # Compute the modified Hausdorff distances between the gpt and human codes assigned to each text
    hausdorff_distances = []
    for gpt_codes, human_codes in zip(gpt_codes_split, human_codes_split):
        if pd.isna(gpt_codes) or pd.isna(human_codes):
             # If codes for either text is None, add np.nan as distance output
            hausdorff_distances.append(np.nan)
            continue
        
        gpt_embeddings = np.array([code_embeddings[code] for code in gpt_codes])
        human_embeddings = np.array([code_embeddings[code] for code in human_codes])
        hausdorff_distances.append(hausdorff_embedding_distance(
            A=gpt_embeddings,
            B=human_embeddings
        ))
    
    # Clean floating point accuracy errors
    hausdorff_distances=np.array(hausdorff_distances)
    hausdorff_distances=np.where(hausdorff_distances < 1e-10, 0, hausdorff_distances) 

    results_df = df.copy()
    results_df["hausdorff_dist"] = hausdorff_distances

    return hausdorff_distances, results_df

def merge_codes(coded_text):
    codes = set(code for _, code in parse_codes(coded_text))
    return "; ".join(codes)

def get_llm_and_human_codes_by_text(df_input, coded_texts):
    """
    Prepare a DataFrame containing LLM and human-generated codes for the same texts for comparison by merging all codes for each text
    """
    data = []
    for idx, row in df_input.iterrows():
        text = row.text
        llm_coded_text = coded_texts[idx]
        human_coded_text = row.coded_text

        # For each text, extract all LLM and human-generated codes
        llm_codes = merge_codes(llm_coded_text) if llm_coded_text else None
        human_codes = merge_codes(human_coded_text) if human_coded_text else None
        
        data.append((text, llm_codes, human_codes))
    return pd.DataFrame(data, columns=["text", "codes", "human_codes"])