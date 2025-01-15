import { queryLLM } from "./LLM";
import { parseCodes } from "./Utils";

export async function codeInductivelyWithCodeConsistency(
    inputs,
    examples,
    researchQuestion,
    codingInstructions,
    gptModel,
    onProgress = () => {}
  ) {
    // Dynamically set maxTokens based on the maximum text length
    const maxTokens = Math.max(
      ...inputs.map((input) => input.text.length),
      300 // Default to 300 if no large text exists
    );
  
    const codedTexts = [];
    const codeDescriptions = {};
  
    // Process texts sequentially to enforce code consistency
    for (let idx = 0; idx < inputs.length; idx++) {
      const input = inputs[idx];
  
      // Construct prompt, including a list of existing codes
      const prompt = constructInductivePrompt({
        input,
        examples,
        codingInstructions,
        codeDescriptions,
      });
      console.log(prompt);
  
      // Query the LLM
      const continuation = await queryLLM(
        prompt,
        gptModel,
        maxTokens,
      );
      console.log(continuation);
  
      // Attempt to correct any LLM formatting errors
      const codedTextBatch = correctCodingErrors([input.text], [continuation]);
  
      const codedText = codedTextBatch[0];
      codedTexts.push(codedText);
  
      // For any new codes, generate description and store
      for (const [highlight, code] of parseCodes(codedText)) {
        if (!codeDescriptions[code]) {
          codeDescriptions[code] = await generateCodeDescription(
            code,
            [highlight],
            researchQuestion,
            gptModel,
          );
        }
      }

      // Update progress
      onProgress({
        current: idx + 1,
        max: inputs.length,
        frac: (idx + 1) / inputs.length
      });
    }
  
    return { codedTexts, codeDescriptions };
  }

  function constructInductivePrompt({
    input,
    examples,
    codingInstructions,
    codeDescriptions = null,
  }) {
    let prompt = `You are an expert qualitative researcher. You are given a text to code inductively. Please carry out the following task:
  - Respond by repeating the original text, but highlighting the coded statements by surrounding the statements with double asterisks, as if they were bolded text in a Markdown document.
  - Include the associated code(s) immediately after the statement, separated by a semicolon and enclosed in <sup></sup> tags, as if they were superscript text in a Markdown document.
  - Preserve exact formatting of the original text. Do not correct typos or remove unnecessary spaces.\n`;
  
    // Add user-defined instructions
    prompt += `${codingInstructions.trim()}\n\n`;
  
    // Optionally add existing codes into the prompt to encourage consistency
    if (codeDescriptions && Object.keys(codeDescriptions).length > 0) {
      prompt += 'Some examples of codes in the format "{code}: {description}". Please create new codes when needed:\n';
  
      // Shuffle codes to mitigate LLM recency bias
      let codeDescArr = Object.entries(codeDescriptions).map(
        ([code, description]) => `${code}: ${description}`
      );
      codeDescArr = shuffleArray(codeDescArr);
  
      // Add each code as a new line
      prompt += `${codeDescArr.join("\n")}\n\n`;
    }
  
    prompt += "Below, I first give you examples of the output you should produce given an example input. After that, I give you the actual input to process.";
    prompt += " The input may come from a thread of texts, and any preceding texts are added as context (labelled CONTEXT). Your task is to code only the last text (labelled TEXT).\n\n";
  
    // Add the few-shot examples in random order
    const shuffledExamples = shuffleArray(examples);
    shuffledExamples.forEach((example, idx) => {
      prompt += `EXAMPLE ${idx + 1} INPUT:\n`;
      if (example.ancestors) {
        example.ancestors.forEach((ancestorText) => {
          prompt += `CONTEXT: ${ancestorText}\n`;
        });
        prompt += `TEXT: ${example.text}\n\n`;
      } else {
        prompt += `${example.text}\n\n`;
      }
      prompt += `EXAMPLE ${idx + 1} OUTPUT:\n${example.codedText}\n\n`;
    });
  
    // Add the actual input text with ancestors if present
    if (input.ancestors) {
      prompt += `ACTUAL INPUT:\n`;
      input.ancestors.forEach((ancestorText) => {
        prompt += `CONTEXT: ${ancestorText}\n`;
      });
      prompt += `TEXT: ${input.text}`;
    } else {
      prompt += `ACTUAL INPUT:\n${input.text}`;
    }
  
    return prompt;
  }

  async function generateCodeDescription(
    code,
    examples,
    researchQuestion,
    gptModel,
    counterExamples = [],
  ) {
    let prompt = "Write a brief but nuanced one-sentence description for the given inductive code, based on a set of texts annotated with the code";
  
    if (counterExamples.length > 0) {
      prompt += " and counter-examples where the code does not apply.\n";
    } else {
      prompt += ".\n";
    }
  
    prompt += "For example, for the code \"overcommunication\", you might generate the description: Captures instances where participants discuss feeling overwhelmed by excessive communication, such as constant emails, messages, or meetings.\n";
    prompt += `Write the description in the context of a qualitative research project with the research question: ${researchQuestion}\n\n`;
  
    prompt += `CODE: ${code}\n\n`;
  
    prompt += 'CODED TEXTS SEPARATED BY "***":\n';
    prompt += examples.join("\n***\n");
  
    if (counterExamples.length > 0) {
      prompt += '\n\nCOUNTER-EXAMPLES SEPARATED BY "***":\n';
      prompt += counterExamples.join("\n***\n");
    }
  
    // Query the LLM
    const continuation = await queryLLM(
      prompt,
      gptModel,
      200,
    );
  
    return continuation;
  }

  function correctCodingErrors(texts, continuations) {
    /**
     * Check for errors in LLM responses and attempt to correct them.
     * Outputs an array of corrected texts corresponding to the original texts,
     * with `null` in place of outputs that could not be corrected.
     */
    const codedTexts = [];
    let nReconstructed = 0;
  
    texts.forEach((text, index) => {
      const cont = continuations[index];
  
      if (!cont || cont.length === 0) {
        // If the response is empty, discard it
        console.warn(`WARNING: Discarding empty LLM response for text "${text}"`);
        codedTexts.push(null);
        return;
      }
  
      // Remove highlights and code annotations to check for LLM errors
      const contText = cont.replace(/\*\*|<sup>(.*?)<\/sup>/g, "");
      if (contText === text) {
        codedTexts.push(cont); // No errors, accept the response
        return;
      }
  
      // Ignore small differences if edit distance is below threshold
      const distThreshold = 5;
      const editDist = calculateLevenshteinDistance(text, contText);
      if (editDist < distThreshold) {
        codedTexts.push(cont);
        return;
      }
  
      // Attempt reconstruction using fuzzy matching for annotations
      const annotations = [...cont.matchAll(/\*\*(.*?)\*\*(<sup>.*?<\/sup>)/g)];
      let reconstructed = text;
      let reconstructionFailed = false;
  
      annotations.forEach(([_, highlight, codes]) => {
        const { matchStart, matchEnd, ratio } = findBestMatch(reconstructed, highlight);
        if (ratio >= 90) {
          // Add annotation to reconstruction if a sufficient match is found
          const recAnnotation = `**${reconstructed.slice(matchStart, matchEnd)}**${codes}`;
          reconstructed =
            reconstructed.slice(0, matchStart) + recAnnotation + reconstructed.slice(matchEnd);
        } else {
          console.warn(`Could not find LLM-annotated text "${highlight}" in the original text "${text}"`);
          reconstructionFailed = true;
        }
      });
  
      if (reconstructionFailed) {
        console.warn(`Text reconstruction failed, discarding LLM response`);
        codedTexts.push(null);
      } else {
        console.log(`Text reconstruction successful`);
        codedTexts.push(reconstructed);
        nReconstructed++;
      }
    });
  
    if (nReconstructed > 0) {
      console.log(`Reconstructed ${nReconstructed} texts due to LLM errors`);
    }
  
    const nDiscarded = codedTexts.filter((t) => t === null).length;
    if (nDiscarded > 0) {
      console.warn(`WARNING: A total of ${nDiscarded} LLM outputs were discarded because of errors`);
    }
  
    return codedTexts;
  }

  function findBestMatch(originalText, targetSubstring) {
    /**
     * Finds the substring in `originalText` that best matches `targetSubstring`
     * using fuzzy matching, allowing for some length variance.
     *
     * Args:
     *   originalText (string): The text to search in.
     *   targetSubstring (string): The string to find a similar match for.
     *
     * Returns:
     *   Object: { bestStart, bestEnd, bestRatio } where:
     *     - bestStart: Starting index of the best matching substring.
     *     - bestEnd: Ending index of the best matching substring.
     *     - bestRatio: Similarity score of the best match.
     */
  
    let bestRatio = 0;
    let bestStart = 0;
    let bestEnd = 0;
  
    const maxLengthVariance = Math.floor(targetSubstring.length / 4);
  
    // Iterate over all possible window sizes
    for (let start = 0; start < originalText.length; start++) {
      const endMin = start + targetSubstring.length - maxLengthVariance;
      const endMax = Math.min(
        start + targetSubstring.length + maxLengthVariance,
        originalText.length + 1
      );
  
      for (let end = endMin; end < endMax; end++) {
        const window = originalText.slice(start, end);
        const similarityRatio = calculateSimilarity(window, targetSubstring);
  
        // Update the best match if this window has a better match
        if (similarityRatio > bestRatio) {
          bestRatio = similarityRatio;
          bestStart = start;
          bestEnd = end;
        }
      }
    }
  
    return { 
        matchStart: bestStart,
        matchEnd: bestEnd,
        ratio: bestRatio
    };
  }
  
  /**
   * Calculates similarity ratio between two strings using Levenshtein distance.
   */
  function calculateSimilarity(a, b) {
    const levenshteinDistance = calculateLevenshteinDistance(a, b);
    return (1 - levenshteinDistance / Math.max(a.length, b.length)) * 100;
  }
  
  function calculateLevenshteinDistance(a, b) {
    const dp = Array.from({ length: a.length + 1 }, () => Array(b.length + 1).fill(0));
  
    for (let i = 0; i <= a.length; i++) dp[i][0] = i;
    for (let j = 0; j <= b.length; j++) dp[0][j] = j;
  
    for (let i = 1; i <= a.length; i++) {
      for (let j = 1; j <= b.length; j++) {
        if (a[i - 1] === b[j - 1]) {
          dp[i][j] = dp[i - 1][j - 1];
        } else {
          dp[i][j] = Math.min(dp[i - 1][j - 1], dp[i - 1][j], dp[i][j - 1]) + 1;
        }
      }
    }
  
    return dp[a.length][b.length];
  }

  function shuffleArray(array) {
    const shuffled = [...array]; // Create a shallow copy to avoid modifying the original array
    for (let i = shuffled.length - 1; i > 0; i--) {
      const j = Math.floor(Math.random() * (i + 1)); // Random index from 0 to i
      [shuffled[i], shuffled[j]] = [shuffled[j], shuffled[i]]; // Swap elements
    }
    return shuffled;
  }