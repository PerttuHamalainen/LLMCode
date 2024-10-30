from .llms import query_LLM
from Levenshtein import distance
import html
import re

"""
TODO:

refactor so that context can be added for each analyzed item
"""

def extract_relevant(prompt_base,
                     df,
                     data_col,
                     extracts_col,
                     model=None,
                     show_prompt=None,
                     max_tokens=None,
                     min_output_length=None):
    df=df.copy()
    df.reset_index()
    if show_prompt is None:
        show_prompt=False

    if max_tokens is None:
        max_tokens=2000

    if min_output_length is None:
        min_output_length=2

    if model is None:
        model="gpt-4o"

    # construct prompts
    prompts = []
    for i in range(df.shape[0]):
        # construct the prompt
        prompt = prompt_base + df[data_col].iloc[i] + "\n\n"
        prompts.append(prompt)

    if show_prompt:
        print("Full prompt for the first data item:\n\n")
        print(prompts[0])
        print("\n\n")

    # Query the LLM
    #print("Querying LLM...")
    continuations = query_LLM(model=model,
                                      prompts=prompts,
                                      max_tokens=max_tokens,
                                      temperature=0,
                                      use_cache=True)

    # Correct a particular type of error: Sometimes, the LLM might only output the highlighted passages, omitting the non-highlighted parts.
    # Since this will affect IoU calculations, we correct this mistake by adding the double asterisks back to the original comments.
    # At the same time, this removes any statements extracted from the parent comments.
    n_uncertain = 0
    n_reconstructed = 0

    for count, cont in enumerate(continuations):
        comment = df[data_col].iloc[count]
        if len(cont) < min_output_length:
            # If the response is empty (possibly, due to model censorship), we highlight the full comment to avoid false negatives
            continuations[count] = f"**{comment}**"
            print(f"Warning: LLM response is empty, highlighting the whole comment to avoid false negatives:\n\n")
            print(continuations[count])
            n_uncertain+=1
            continue
        # Check if there were any clear errors or hallucinations. In other words, does the LLM response (cont) match
        # the original comment when we remove non-alphanumeric characters?
        comment_cmp = re.sub(r'\W+', '', comment)
        cont_cmp = re.sub(r'\W+', '', cont)
        if comment_cmp != cont_cmp:
            # Sometimes, the LLM autocorrects typos even though we explicitly tell it not to.
            # This is why we ignore the differences if the edit distance between the original and LLM-highlighted comment
            # is small enough
            dist_threshold = 5
            edit_dist = distance(comment_cmp, cont_cmp)
            if edit_dist < dist_threshold:
                print(
                    f"Warning: LLM-highlighted text differs from the original text, with edit distance {edit_dist}\n\n")
                print(f"Comment: {comment})\n\n")
                print(f"LLM-highlighted: {cont})\n\n")
                continue

            # Error detected => reconstruct by adding the "**" back to the original comment
            n_reconstructed += 1
            reconstructed = comment
            statements = re.findall('\*\*(.+?)\*\*', cont)  # get the extracts from the continuation as a string list
            reconstruction_failed = False
            for s in statements:
                if s in reconstructed:
                    reconstructed = reconstructed.replace(s, f"**{s}**")
                else:
                    # re.sub(pattern,repl,string) replaces parts matching the pattern. We replace with '', i.e., remove the parts. \W denotes non-alphanumeric characters, and the + denotes "one or many"
                    s_cmp = re.sub(r'\W+', '', s)
                    comment_cmp = re.sub(r'\W+', '', comment)
                    if s_cmp not in comment_cmp:
                        print(
                            f"Possible hallucination warning: Could not find the following LLM-highlighted statement in text {count}:\n\n{s}\n")
                        reconstruction_failed=True
                        continue
            if reconstruction_failed:
                print(f"Highlighting the whole text to avoid false negatives:\n\n")
                continuations[count] = f"**{comment}**"
                print(continuations[count])
                n_uncertain += 1
            else:
                continuations[count] = reconstructed

    print(f"Had to reconstruct {n_reconstructed} texts due to LLM errors.")
    print(f"A total of {n_uncertain} texts were highlighted fully because of LLM errors, to avoid false negatives.")
    result_df=df.copy()
    result_df[extracts_col]=continuations
    return result_df

def markdown_to_html(text):
    text=html.escape(text)
    while "**" in text:
        text = text.replace("**", "<b>", 1).replace("**", "</b>", 1)
    text = text.replace("\n", "<br>")
    text = text.replace("---","<hr>")
    return text

def extracts_to_html(texts):
    if isinstance(texts, str):
        return markdown_to_html(texts)
    else:
        html_text=""
        for text in texts:
          html_text += markdown_to_html(text)
          html_text += "<hr>"
        return html_text

def get_single_extract_per_row(df, extracts_col):
    continuations=df[extracts_col].tolist()
    statements = []
    parent_comments_out = []
    comment_out = []
    hallucinations = []
    for count, c in enumerate(continuations):
        statements = re.findall('\*\*(.+?)\*\*', c)
        comment = df["comment"][count]
        for s in statements:
            comment_out.append(comment)
            parent_comments_out.append(parent_comments)
            statements.append(s)
    df = pd.DataFrame()
    df["texts"] = statements
    df["extracts"] = comment_out
    return df
