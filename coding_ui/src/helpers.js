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