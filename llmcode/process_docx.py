#!/usr/bin/env python
# Given a .docx file, extract a CSV list of all texts (separated by "-----" in the input document) with integrated codes in markdown format, extracted from the document's comments
# This is version 6.0 of the script
# Date: 12 February 2020

import zipfile
from bs4 import BeautifulSoup as Soup
import re
import pandas as pd

def open_docx_and_process_codes(path=None):
    # If no path is given, make user choose a file
    if path is None:
        import tkinter as tk
        from tkinter import filedialog
        # Show file selection dialog box
        root = tk.Tk()
        root.withdraw()
        paths = filedialog.askopenfilenames()
        root.update()

        if len(paths) != 1:
            raise ValueError("Please provide a single file")

        path = paths[0]

    # .docx files are really ZIP files with a separate 'file' within them for the document
    # itself and the text of the comments. This unzips the file and parses the comments.xml
    # file within it, which contains the comment (label) text
    unzip = zipfile.ZipFile(path)
    comments = Soup(unzip.read("word/comments.xml"), features="xml")
    # The structure of the document itself is more complex and we need to do some
    # preprocessing to handle multi-paragraph and nested comments, so we unzip
    # it into a string first
    doc = unzip.read("word/document.xml").decode()

    # Find all paragraphs
    paragraphs = re.findall(r"<w:p\s[^>]*>(.*?)<\/w:p>", doc, re.DOTALL)

    # Divide paragraphs into text by assuming any paragraph with "-----" is a divider
    texts = [[]]
    for p in paragraphs:
        if "-----" in p:
            # Start new text
            texts.append([])
        else:
            # Add paragraph to latest text
            texts[-1].append(p)
    texts = ["\n".join(text_paragraphs) for text_paragraphs in texts]

    # Parse code mapping
    codes = {c.attrs['w:id']: ''.join(c.findAll(text=True)) for c in comments.find_all('w:comment')}

    # Function to replace comment tags and format text
    def process_comments(text):
        # Store uncoded text
        uncoded_text = re.sub(r"<[^>]+>", "", text)

        # Regular expression to find commentRangeStart with w:id
        comment_start = r'<w:commentRangeStart.*?w:id="(.*?)".*?>'
        comment_end = r'<w:commentRangeEnd.*?>'

        # Find all commentRangeStart matches and replace them
        while True:
            match = re.search(comment_start, text)
            if not match:
                break

            comment_id = match.group(1)
            code = codes.get(comment_id)
            
            # Locate the text between commentRangeStart and commentRangeEnd
            text_start = match.end()  # End position of <w:commentRangeStart>
            text_end_match = re.search(comment_end, text[text_start:])
            
            if text_end_match:
                text_end = text_start + text_end_match.start()
                # Extract the text between the comment tags
                commented_text = text[text_start:text_end].strip()
                # Surround text with ** and append the code in <sup>
                replacement = f"**{commented_text}**<sup>{code}</sup>"
                # Replace the comment tag and text with the formatted version
                text = text[:match.start()] + replacement + text[text_start + text_end_match.end():]
        
        # Replace all remaining tags with an empty string
        coded_text = re.sub(r"<(?!/?sup\b)[^>]+>", "", text)
        return (uncoded_text, coded_text)

    coded_texts = [process_comments(t) for t in texts]

    output_df = pd.DataFrame(coded_texts, columns=["text", "coded_text"])
    return output_df
