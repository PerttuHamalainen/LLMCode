from .llms import query_LLM, query_LLM_with_response_format
from pydantic import BaseModel
from titlecase import titlecase
import random
from rapidfuzz import fuzz
import re
from itertools import chain

def get_themes(codes,
               prior_themes,
               research_question,
               gpt_model,
               code_descriptions=None,
               max_tokens=None,
               verbose=False,
               max_retries=0):
    """
    Generate and assign themes to a set of codes based on existing themes and an LLM query.

    Args:
        codes (set or list): A set or list of codes to be thematically grouped.
        prior_themes (dict or list): Existing themes to which the LLM should add new codes. 
            If a dict is provided, keys are theme names and values are sets of codes already associated with each theme.
            If a list is provided, it's assumed to be a list of theme names with no associated codes.
        research_question (str): The research question that the LLM should consider when thematically grouping the codes.
        gpt_model (str): The LLM model to be used for generating themes.
        code_descriptions (dict, optional): Optional descriptions for each code, providing additional context to the LLM.
        max_tokens (int, optional): Maximum number of tokens for the LLM output.
        verbose (bool, optional): If True, prints additional information during execution, including LLM prompts and any identified hallucinations. Defaults to False.
        max_retries (int, optional): Maximum number of retries to attempt assigning themes to all codes. Defaults to 0 (no retries).

    Returns:
        tuple: 
            - themes (dict): A dictionary where keys are theme names and values are sets of codes assigned to each theme.
            - unthemed_codes (set): A set of codes that could not be assigned to a theme.
    """
    # Handle inputs
    codes = set(codes)
    if isinstance(prior_themes, dict):
        # Codes provided for themes, ensure they are a set
        prior_themes = {titlecase(theme): set(codes) for theme, codes in prior_themes.items()}
    else:
        # No codes provided for themes, init with empty set
        prior_themes = {titlecase(theme): set() for theme in prior_themes}

    # Remove codes that are already given a theme
    themed_codes = set().union(*prior_themes.values())
    unthemed_codes = codes - themed_codes

    # Construct prompt
    prompt = construct_theme_prompt(
        codes=unthemed_codes, 
        prior_themes=prior_themes,
        research_question=research_question,
        code_descriptions=code_descriptions
    )

    if verbose:
        print(f"Prompt:\n\n{prompt}\n\n")

    # Define response format
    class ResponseFormat(BaseModel):
        class Theme(BaseModel):
            name: str
            codes: list[str]
        themes: list[Theme]

    # Query LLM
    response = query_LLM_with_response_format(
        prompt, 
        ResponseFormat, 
        model=gpt_model,
        max_tokens=max_tokens
    )

    # Parse response
    # Initialise themes as the user-defined themes
    themes = prior_themes.copy()
    for llm_theme in response.themes:
        # Exclude any hallucinations
        llm_codes = set(llm_theme.codes)
        valid_llm_codes = llm_codes & codes
        if verbose and valid_llm_codes != llm_codes:
            print(f"Hallucinated code(s), removing: {llm_codes - valid_llm_codes}")

        # Only include LLM output for unthemed codes
        valid_and_unthemed_llm_codes = valid_llm_codes & unthemed_codes
        if len(valid_and_unthemed_llm_codes) > 0:
            if llm_theme.name in themes:
                themes[llm_theme.name] = themes[llm_theme.name] | valid_and_unthemed_llm_codes
            else:
                # Check if LLM used a different capitalisation
                theme_match = next((theme for theme in themes.keys() if theme.lower() == llm_theme.name.lower()), None)
                if theme_match is not None:
                    themes[theme_match] = themes[theme_match] | valid_and_unthemed_llm_codes
                else:
                    themes[llm_theme.name] = valid_and_unthemed_llm_codes
                    if verbose:
                        print(f"Found new theme: {llm_theme.name}")

    # Ensure each code is included in exactly one theme
    unthemed_codes = set()
    for code in codes:
        code_count = sum(code in theme_codes for theme_codes in themes.values())

        if code_count == 0:
            # No theme assigned
            unthemed_codes.add(code)
            if verbose:
                print(f"Did not assign a theme for code: {code}")
        elif code_count > 1:
            # Multiple themes assigned, remove code from all themes
            unthemed_codes.add(code)
            themes = {theme: codes - {code} for theme, codes in themes.items()}
            if verbose:
                print(f"Assigned multiple themes for code, removing: {code}")

    # Remove any themes with no codes
    themes = {theme: codes for theme, codes in themes.items() if len(codes) > 0}

    if len(unthemed_codes) > 0 and max_retries > 0:
        # If there are unthemed codes and retries remaining, call the function recursively with the found themes and max_retries - 1
        if verbose:
            print("Did not find a theme for all codes, attempting again...")
        return get_themes(
            codes=codes,
            prior_themes=themes,
            research_question=research_question,
            gpt_model=gpt_model,
            code_descriptions=code_descriptions,
            max_tokens=max_tokens,
            verbose=verbose,
            max_retries=max_retries-1
        )
    else:
        return themes, unthemed_codes

def construct_theme_prompt(codes, prior_themes, research_question, code_descriptions=None):
    prompt = """You are an expert qualitative researcher. You are given a list of qualitative codes at the end of this prompt. Please carry out the following task:
- Group these codes into overearching themes that relate to the research question.
- Assign codes to the themes provided in the list of user-defined themes and generate new themes when needed.
- The theme names should be detailed and expressive.
- Output a list of Theme objects, containing the theme name and a list of codes that are included in that theme. Start this list with the user-defined themes.
- Include each of the codes under exactly one theme.\n\n"""

    # Optionally add prior themes into the prompt
    prompt += "USER-DEFINED THEMES:\n\n"
    if prior_themes is not None and len(prior_themes) > 0:
        for theme, theme_codes in prior_themes.items():
            prompt += f"THEME: {theme}\n"
            if len(theme_codes) > 0:
                prompt += "CODE EXAMPLES FOR THEME:\n"
                prompt += "\n".join(code + (f": {code_descriptions[code]}" if code_descriptions else "") for code in theme_codes)
            prompt += "\n\n"
    else:
        prompt += "No themes identified yet." + "\n\n"

    prompt += f"CODES:\n"
    prompt += "\n".join(code + (f": {code_descriptions[code]}" if code_descriptions else "") for code in codes)

    prompt += f"\n\nRESEARCH QUESTION: {research_question}"

    return prompt

def write_report(themes,
                 code_highlights,
                 research_question,
                 gpt_model,
                 max_tokens=None,
                 use_cache=False,
                 verbose=False,
                 max_themes=None,
                 fuzz_treshold=0.9,
                 max_retries=3):
    """
    Generates a qualitative research report by summarizing themes and code highlights using an LLM.

    Args:
        themes (dict): A dictionary where keys are theme names and values are sets of codes that belong to each theme.
        code_highlights (dict): A dictionary where keys are codes and values are lists of highlights (quotes or text) related to those codes.
        research_question (str): The research question that frames the qualitative analysis in the report.
        gpt_model (str): The LLM model used to generate the report.
        max_tokens (int, optional): Maximum number of tokens the LLM can generate in a single prompt. Defaults to None.
        use_cache (bool, optional): Whether to use cached results for the LLM query. Defaults to False.
        verbose (bool, optional): If True, additional output such as LLM prompts and debugging information will be printed. Defaults to False.
        max_themes (int, optional): The maximum number of themes to include in the report. If None, all themes are included. Defaults to None.
        fuzz_treshold (float, optional): The threshold for fuzzy matching used in hallucination detection. Defaults to 0.9.
        max_retries (int, optional): The maximum number of retries to attempt correcting hallucinated quotes in the LLM output. Defaults to 3.

    Returns:
        str: The generated qualitative research report as a string. If hallucinations in LLM output cannot be fixed after `max_retries` attempts, returns an error message.
    """
    report = f"# Qualitative research report: {research_question}\n"

    theme_highlight_counts = {theme: sum(len(code_highlights[code]) for code in themes[theme]) for theme in themes}
    ranked_themes = sorted(theme_highlight_counts, key=theme_highlight_counts.get, reverse=True)

    if max_themes is not None:
        ranked_themes = ranked_themes[:max_themes]

    for theme in ranked_themes:
        if theme_highlight_counts[theme] < 3:
            print(f"Less than three mentions for theme '{theme}', excluding...")
            continue

        report += f"## {theme} ({theme_highlight_counts[theme]} mentions)\n"

        theme_code_highlights = {code: highlights for code, highlights in code_highlights.items() if code in themes[theme]}

        prompt = construct_theme_report_prompt(
            theme=theme,
            code_highlights=theme_code_highlights,
            text_so_far=report,
            research_question=research_question
        )

        if verbose:
            print(f"Prompt:\n\n{prompt}\n\n")

        continuation = query_LLM(
            model=gpt_model,
            prompts=[prompt],
            max_tokens=max_tokens,
            use_cache=use_cache
        )[0]

        # Check that no quotes are hallucinated
        all_theme_highlights = list(chain(*theme_code_highlights.values())) 
        continuation = check_hallucination_and_fix(
            continuation=continuation, 
            all_theme_highlights=all_theme_highlights, 
            fuzz_treshold=fuzz_treshold, 
            max_tokens=max_tokens, 
            use_cache=use_cache, 
            gpt_model=gpt_model, 
            max_retries=max_retries,
            verbose=verbose
        )

        if continuation is None:
            return "Could not generate report due to hallucination errors."
        
        report += continuation.strip() + "\n"

    return report

def construct_theme_report_prompt(theme, code_highlights, text_so_far, research_question):
    prompt = f"You are an expert qualitative researcher writing a qualitative research report. "
    prompt += f"""You are given a theme and its associated codes, as well as examples of quotes annotated by the codes. Please carry out the following task:
- Write a section continuing the qualitative research report that describes the findings for the given theme.
- The text should be a single paragraph long and not include a conclusion, as it forms only a part of a larger report.
- The text should relate to the research question.
- Reference quotations when appropriate to illustrate the findings and connect them to the underlying data. Do not generate quotations that do not appear in the list and do not change any words in the quotations. Using substrings from quotations is allowed.
- Tell an engaging story about the data and the findings, using the themes and codes as background material. Do not talk about "themes" or "codes" explicitly as these are for research purposes.
- Follow the style used in the following example.\n\n"""
    
    prompt += f"AN EXAMPLE EXCERPT FROM AN UNRELATED QUALITATIVE RESEARCH REPORT:\n\nMany design projects discussed in the texts involved multiple interviews or documents, offering a wide range of perspectives for analysis. From these materials, three levels of analysis emerged: contextual, comparative, and holistic. Each level varies in the depth and breadth of data analyzed, as well as the aims and methods used in the analysis. Contextual analysis focuses on understanding individual texts within the context in which they were produced. Several examples emphasized the importance of reviewing full transcriptions or detailed notes to capture the complete meaning of the responses. One text highlighted this approach, stating, \"With qualitative data, it's more important to analyze responses as a whole.\" Other sources followed similar practices, preferring to retain complete transcriptions, as this helped to \"understand the interviewee as a person\" or the context behind the data. This deeper understanding was seen as critical for gaining empathy for the individuals or situations described, enabling better design solutions.\n\n"

    prompt += f"RESEARCH QUESTION: {research_question}\n\n"

    # Shuffle codes to handle recency bias
    code_highlights = list(code_highlights.items())
    random.shuffle(code_highlights)

    prompt += f"THEME: {theme}\n"
    prompt += "THEME CODES AND QUOTATIONS:\n"
    prompt += "\n".join(
        "{}: {}".format(
            code, "; ".join('"{}"'.format(hl) for hl in highlights)
        ) for code, highlights in code_highlights
    )
    prompt += "\n\n"

    prompt += f"REPORT TEXT SO FAR:\n\n{text_so_far}"

    return prompt

def check_hallucination_and_fix(continuation, all_theme_highlights, fuzz_treshold, max_tokens, use_cache, gpt_model, max_retries, verbose):
    """
    Check for hallucinated quotes and retry fixing them up to `max_retries` times.
    
    Args:
        continuation: The initial LLM response.
        all_theme_highlights: The set of highlights to match quotes against.
        fuzz_treshold: The fuzzy matching threshold.
        max_tokens: Max tokens for LLM query.
        use_cache: Whether to use cached results.
        gpt_model: The LLM model.
        max_retries: The maximum number of times to attempt fixing hallucinated quotes.
        
    Returns:
        The corrected continuation after fixing hallucinations, or None if not fixed after max_retries.
    """
    # Loop up to `max_retries` times
    for _ in range(max_retries):
        # Extract quotes from the LLM response
        quotes = _extract_quotes(continuation)

        # Identify false quotes
        false_quotes = [quote for quote in quotes if _is_false_quote(quote, all_theme_highlights, fuzz_treshold)]

        # If no false quotes, return the corrected continuation
        if not false_quotes:
            return continuation
        elif verbose:
            print(f"Found hallucinated quotes, attempting to fix text:\n\n{continuation}\n\n")

        # If false quotes exist, construct a new prompt to fix them
        prompt = construct_false_quote_removal_prompt(continuation, false_quotes)

        # Query the LLM again with the updated prompt
        continuation = query_LLM(
            model=gpt_model,
            prompts=[prompt],
            max_tokens=max_tokens,
            use_cache=use_cache
        )[0]

    # If still hallucinated quotes after max_retries, return None
    return None

def construct_false_quote_removal_prompt(text, false_quotes):
    prompt = f"Remove the false quotes from the text, including any parts of the text that refers to them or builds upon them. Otherwise, keep the text identical and do not modify any quotes that aren't listed as incorrect quotes.\n\n"
    prompt += f"TEXT:\n{text}\n\n"
    prompt += "FALSE QUOTES:\n{}".format("\n".join('"{}"'.format(q) for q in false_quotes))
    return prompt

def _extract_quotes(text):
    # Use regex to match text within double quotes
    quotes = re.findall(r'"(.*?)"', text)
    return quotes

def _is_false_quote(quote, all_highlights, fuzz_threshold):
    """Check if a quote has no fuzzy match with any theme highlight."""
    return not any(fuzzy_match_with_brackets(quote, hl, fuzz_threshold) for hl in all_highlights)

def _convert_quotes_to_italics(markdown_text):
    # Regular expression to match text enclosed in double quotes
    pattern = r'"([^"]+)"'
    # Replace with italic Markdown syntax
    return re.sub(pattern, r'*\1*', markdown_text)

def _fuzzy_match_segment(segment, text, threshold):
    """Perform fuzzy matching of a single segment within the given text."""
    # Use fuzzy partial ratio to allow for matching of substrings
    similarity = fuzz.partial_ratio(segment, text)
    return similarity >= threshold

def fuzzy_match_with_brackets(input_quote, generated_text, threshold):
    """Fuzzy match each segment of the input quote, separated by possible [...] for edited quotations, with the generated text."""
    # Use regex to split the quote into segments, retaining parts before/after and inside brackets
    segments = re.split(r'(\[.*?\])', input_quote)

    start_index = 0
    for segment in segments:
        # Check if the segment is inside square brackets (bracketed content)
        if segment.startswith("[") and segment.endswith("]"):
            # Allow any text to match for the part inside brackets
            continue
        
        # Perform fuzzy matching on non-bracketed segments
        segment_cleaned = segment.strip()
        if segment_cleaned:
            # Look for a fuzzy match of this segment in the remaining generated text
            match_found = False
            for i in range(start_index, len(generated_text)):
                match_found = _fuzzy_match_segment(segment_cleaned, generated_text[start_index:], threshold=threshold)
                if match_found:
                    # Move the start index forward for the next segment
                    start_index = i + len(segment_cleaned)
                    break

            if not match_found:
                # Return False if no match is found for any segment
                return False

    # Return True if all segments match
    return True