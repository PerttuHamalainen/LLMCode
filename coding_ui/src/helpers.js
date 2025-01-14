export const formatTextWithHighlights = (text, highlights) => {
  // Sort highlights by startIndex to process them in order
  const sortedHighlights = [...highlights].sort((a, b) => a.startIndex - b.startIndex);

  let result = "";
  let currentIndex = 0;

  sortedHighlights.forEach((hl) => {
    const { startIndex, endIndex, codes } = hl;

    // Append text before the current highlight
    if (currentIndex < startIndex) {
      result += text.slice(currentIndex, startIndex);
    }

    // Wrap the highlighted text with double asterisks and add the <sup> tag
    const highlightedText = text.slice(startIndex, endIndex);
    result += `**${highlightedText}**<sup>${codes}</sup>`;

    // Update the current index
    currentIndex = endIndex;
  });

  // Append any remaining text after the last highlight
  if (currentIndex < text.length) {
    result += text.slice(currentIndex);
  }

  return result;
};

export function nanMean(arr) {
  const filtered = arr.filter((value) => !isNaN(value)); // Filter out NaN values
  if (filtered.length === 0) {
    return NaN; // Return NaN if all values are NaN
  }
  const sum = filtered.reduce((acc, value) => acc + value, 0);
  return sum / filtered.length; // Compute mean
}

export function parseTextHighlights(text) {
  // Regular expression to match **highlight**<sup>codes</sup>
  const pattern = /\*\*([\s\S]*?)\*\*<sup>(.*?)<\/sup>/g;

  // Initialize results and plainText
  const highlights = [];
  let plainText = "";
  let currentOffset = 0;

  // Process all matches
  let match;
  while ((match = pattern.exec(text)) !== null) {
    const [fullMatch, highlight, codes] = match;

    // Compute the plain text up to the current match
    plainText += text.slice(currentOffset, match.index) + highlight;

    // Compute the startIndex and endIndex in the plain text
    const startIndex = plainText.length - highlight.length;
    const endIndex = plainText.length;

    // Add the match details to the results
    highlights.push({
      id: crypto.randomUUID(),
      startIndex,
      endIndex,
      codes,
      text: highlight,
      focused: false,
      hovered: false,
    });

    // Update the offset to continue parsing
    currentOffset = match.index + fullMatch.length;
  }

  // Append any remaining text after the last match
  plainText += text.slice(currentOffset);

  return { plainText: plainText, textHighlights: highlights };
}

/**
 * Splits an array of objects into test and validation sets.
 *
 * @param {Object[]} data - The array of objects to split.
 * @param {number} testCount - Number of samples for the test set.
 * @param {number} maxValidationCount - Maximum number of samples for the validation set.
 * @returns {{ testSet: Object[], validationSet: Object[] }}
 */
export function splitData(data, testCount, maxValidationCount) {
  // Make a shallow copy so we don't mutate the original data
  const shuffled = [...data];

  // Fisher-Yates shuffle (in-place)
  for (let i = shuffled.length - 1; i > 0; i--) {
    const j = Math.floor(Math.random() * (i + 1));
    [shuffled[i], shuffled[j]] = [shuffled[j], shuffled[i]];
  }

  // Slice out the test set
  const testSet = shuffled.slice(0, testCount);

  // Slice out the validation set from the remaining
  const validationSet = shuffled.slice(testCount, testCount + maxValidationCount);

  return {
    testSet,
    validationSet
  };
}