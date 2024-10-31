### Executive Summary
This repository contains the **LLMCode** toolkit for **AI-assisted qualitative data analysis** using Large Language Models (LLMs). This is a further development of the initial tools developed for the CHI 2023 paper [Evaluating Large Language Models in Generating Synthetic HCI Research Data: a Case Study.](https://dl.acm.org/doi/abs/10.1145/3544548.3580688)

Here's an example of the codes and themes created by LLMCode for text data about experiencing video games as art:
![](test_results/bopp_test_visualization.gif)

Currently, we support OpenAI models via both OpenAI API and Aalto University's Azure OpenAI API. The latter provides better data privacy and is GDPR-safe. Support for fully private processing with local LLMs via [LMStudio](https://lmstudio.ai/) is in the works.


### Design principles
Traditional qualitative content analysis and thematic analysis can very labor-intensive, and our goal is to both **improve the quality of life** of researchers and designers doing qualitative analysis and enable extending such analyses to **large-scale data** such as online discussions.

We believe that **AI should do the dishes while humans do art and interesting work such as research**, not the other way around. Thus, we prioritize **researcher agency and control**. Hence, we focus on defining and guiding the analysis style through concrete examples, i.e., we assume our users are willing to put in at least some initial thinking and manual coding work to provide the examples. **The more data you code manually, the more accurately LLMCode can mimic your coding style** and the more reliably you can analyze and quantify LLMCode's output reliability and quality.

Preferences about AI use vary. Hence, LLMCode supports **multiple use cases and workflows** with varying degrees of automation:

*AI for data exploration and visualization:* If you prefer to analyze data manually, you might still benefit from LLMCode's data visualization and exploration tools in the initial research phase of immersing yourself with your data. Furthermore, LLMCode can also be used to check for the consistency of your manual data coding.

*AI for relevant data highlighting or sampling:* If you prefer manual analysis but have too much data, you might try using LLMCode to highlight relevant text passages to speed up the analysis. The highlights can also be used to extract an informed random sample of the data.

*AI for inductive and deductive coding:* LLMCode can mimic your coding style based on examples manually coded by you. By providing a handful of examples (e.g., 10), you can quickly try out a coding approach and abstract and visualize your data as a distribution of codes. If you provide more manually coded data, it can be used to quantify and analyze the reliability of the AI coding.

*AI for identifying and reporting themes:* Based on the coding results, LLMCode can automatically group the codes under broader themes and produce a report with illustrative quotes. Knowing that LLMs can hallucinate such quotes, **LLMCode automatically checks and removes such hallucinations.** Note that although **academic researchers may not prefer to automate this stage** of analysis, we consider it potentially useful for 1) initial quick exploration of potential research topics and datasets, and 2) industry designers and researchers who work under time pressure and cannot engage in deeper manual analysis.  


### How to use

#### Step 1: Setup an OpenAI account if you don't have oneself
You can also use an Aalto Azure OpenAI API key and skip these steps, although the Aalto API is slower and vanilla OpenAI API is more convenient for trying out the tutorials below.
1. Create an OpenAI account at [https://platform.openai.com](https://platform.openai.com).
2. Go to the account management page
3. Choose "Billing" and buy a fixed amount of usage (at least 1€) or setup continuous billing (with a monthly quota for safety reasons)
4. Choose "API keys" and create an API key. The system gives you a private key that you should store in a safe place, e.g., a password manager.  

*Important info regarding API costs:*
* LLM API calls cost money. YOU USE LLMCODE AT YOUR OWN RISK.
* To minimize costs, it's good to first test with smaller datasets of up to few hundred paragraphs. Analyzing such data typically costs less than 1€
* It is also recommended to use the OpenAI management interface to limit API spend, either by purchasing fixed amounts of credits or defining a monthly quota
* LLMCode minimizes the costs by caching the LLM API query results. This means that if you code the exact same data again with the same model, instructions, and examples, the results will be returned from the cache. If you need to delete the cache, e.g., to save disk space, you can navigate to where you installed the LLMCode (see below), navigate to the ```_LLM_cache``` folder and delete the subfolders.  


#### Step 2: Try the notebook interface and tutorials on Google Colab.
Note: Colab needs a Google account. If you are an Aalto University student or researcher, you can either use a personal account or your Aalto Google account by signing in to Google with your Aalto email.

The notebooks are designed as a series of tutorials with default test data, but you can also swap in and process your own data.

Tutorial 1: [Data visualization and exploration](https://colab.research.google.com/github/PerttuHamalainen/LLMCode/blob/master/data_exploration_and_visualization.ipynb).

Tutorial 2: [Relevant text extraction or highlighting](https://colab.research.google.com/github/PerttuHamalainen/LLMCode/blob/master/relevant_data_extraction.ipynb).

Tutorial 3: [Inductive and deductive coding](https://colab.research.google.com/github/PerttuHamalainen/LLMCode/blob/master/inductive_and_deductive_coding.ipynb).

Tutorial 4: [From codes to themes](https://colab.research.google.com/github/PerttuHamalainen/LLMCode/blob/master/theme_generation.ipynb).

#### Step 3: Install and run the notebooks locally
Google Colab is the fastest way to try LLMCode, but if you want to avoid your data being processed by Google, you should install and run LLMCode locally.

Download and install [Anaconda](https://www.anaconda.com/) and [git](https://git-scm.com), if you don't already have them.

Open the Anaconda command prompt and run the following commands:

    conda create --name llmcode python=3.8
    activate llmcode
    git clone https://github.com/PerttuHamalainen/LLMCode
    cd LLMCode
    pip install -r requirements_notebooks.txt
    jupyter notebook

The last line should launch the Jupyter notebook interface in your browser. Note that this interface doesn't have Colab's UI functionality and instead of using the sliders and other UI elements, you have to edit the values directly in the code.


### Why not just use ChatGPT, Claude, or Gemini?
LLMs like ChatGPT can conduct a form of qualitative analysis out of the box: Just paste your data to the chat, and ask ChatGPT to identify themes. However, this has two major problems:

* All data must fit the maximum context size.
* The results may have errors. For instance, the LLM may simply neglect large parts of your data.
* Evaluating the quality and correctness of the results is hard.

LLMCode addresses the problems above:
* LLMCode only requires that a single coded text and the coding examples fit the context.
* LLMCode systematically processes the data chunk-by-chunk (e.g., paragraph of interview text), ensuring that equal attention is paid to all data.
* LLMCode allows comparing the results to human-annotated data both quantitatively and qualitatively, which helps in both evaluating the severity of the LLM errors. Furthermore, the error analysis often reveals inconsistencies and mistakes in the human-annotations, allowing one to improve one's manual coding and analysis.



### Citation 
LLMCode is developed by Perttu Hämäläinen and Joel Oksanen, with additional contributions by Mikke Tavast and Prabhav Bhatnagar.

If you use LLMCode for your research, please use cite the repository as (a full technical report coming soon, citation info will be updated):

    @software{LLMCode,
      author = {H{\"a}m{\"a}l{\"a}inen, Perttu and Oksanen, Joel and Tavast, Mikke and Bhatnagar, Prabhav},
      title = {{LLMCode: A toolkit for AI-assisted qualitative data analysis}},
      url = {https://github.com/PerttuHamalainen/LLMCode},
      year = {2024}
    }
