
import { parseCodes } from "./Utils";
import { embed } from "./LLM";

export async function runCodingEval(humanCodedTexts, modelCodedTexts, embeddingContext, embeddingModel) {
  // Calculate IoU
  const IoUs = extractIoU(humanCodedTexts, modelCodedTexts);

  // Calculate Hausdorff on text level
  const humanCodes = extractCodes(humanCodedTexts);
  const modelCodes = extractCodes(modelCodedTexts);
  const hausdorffDistances = await calculateHausdorffDistances(humanCodes, modelCodes, embeddingContext, embeddingModel);

  return { IoUs, hausdorffDistances };
}

function extractIoU(humanCodedTexts, modelCodedTexts) {
  // Helper function to clean the text
  function clean(text) {
    if (!text) return "";

    // Remove code annotations
    const textWithoutCodes = text.replace(/<sup>.*?<\/sup>/g, "");

    // Split the text into parts, around **
    const parts = textWithoutCodes.split(/\*\*/);

    // Process each part: remove spaces and punctuation from parts that are not "**".
    const removePattern = /[^\w]/g;
    const processedParts = parts.map((part) => part.replace(removePattern, ""));

    // Join the processed parts back together and convert to lowercase.
    return processedParts.join("**").toLowerCase();
  }

  // Helper function to construct a numerical highlight array
  function toHighlightArray(text) {
    const rawText = text.replace(/\*\*/g, "");
    const result = Array(rawText.length).fill(0);

    let highlighting = 0;
    let pos = 0;
    let posOut = 0;

    while (pos < text.length) {
      if (text.slice(pos, pos + 2) === "**") {
        highlighting = 1 - highlighting; // Toggle highlighting
        pos += 2;
      } else {
        result[posOut] = highlighting;
        posOut++;
        pos++;
      }
    }
    return result;
  }

  // Helper function to pad arrays to the same length
  function padArray(arr, length) {
    if (arr.length < length) {
      return [...arr, ...Array(length - arr.length).fill(0)];
    }
    return arr;
  }

  const IoUs = [];
  for (let i = 0; i < humanCodedTexts.length; i++) {
    const human = humanCodedTexts[i];
    const model = modelCodedTexts[i];

    if (!human || !model) {
      IoUs.push(NaN);
      continue;
    }

    const cleanedHuman = clean(human);
    const cleanedModel = clean(model);

    const humanMap = toHighlightArray(cleanedHuman);
    const modelMap = toHighlightArray(cleanedModel);

    // Pad to same length
    const maxLength = Math.max(humanMap.length, modelMap.length);
    const paddedHumanMap = padArray(humanMap, maxLength);
    const paddedModelMap = padArray(modelMap, maxLength);

    // Calculate intersection and union
    const intersection = paddedHumanMap.reduce(
      (sum, value, idx) => sum + (value & paddedModelMap[idx]),
      0
    );

    const union = paddedHumanMap.reduce(
      (sum, value, idx) => sum + (value | paddedModelMap[idx]),
      0
    );

    // Calculate IoU
    const IoU = intersection === 0 && union === 0 ? 1 : intersection / union;
    IoUs.push(IoU);
  }

  return IoUs;
}

async function calculateHausdorffDistances(
  humanCodes,
  modelCodes,
  embeddingContext,
  embeddingModel,
) {
  // Collect all GPT- and human-generated codes, removing duplicates and null/undefined values
  const allHumanCodes = new Set(
    humanCodes.flat().filter((code) => code !== null && code !== undefined)
  );
  const allModelCodes = new Set(
    modelCodes.flat().filter((code) => code !== null && code !== undefined)
  );
  const allCodes = Array.from(new Set([...allHumanCodes, ...allModelCodes]));

  // Add context to each code and generate embeddings
  const codesWithContext = allCodes.map((code) => code + embeddingContext);
  const embeddingMatrix = await embed(codesWithContext, embeddingModel);
  const codeEmbeddings = allCodes.reduce((acc, code, idx) => {
    acc[code] = embeddingMatrix[idx];
    return acc;
  }, {});

  // Compute the Hausdorff distances between the GPT and human codes assigned to each text
  const hausdorffDistances = humanCodes.map((human, index) => {
    const model = modelCodes[index];

    // If codes for either GPT or human are null/undefined, add NaN as the distance output
    if (!human || !model) {
      return NaN;
    }

    // Fetch embeddings for GPT and human codes
    const humanEmbeddings = human.map((code) => codeEmbeddings[code]);
    const modelEmbeddings = model.map((code) => codeEmbeddings[code]);

    // Compute the Hausdorff distance
    return hausdorffEmbeddingDistance(humanEmbeddings, modelEmbeddings);
  });

  // Clean floating-point accuracy errors
  return hausdorffDistances.map((distance) =>
    distance < 1e-10 ? 0 : distance
  );
}

function extractCodes(codedTexts) {
  return codedTexts.map((codedText) => {
    if (codedText) {
      return parseCodes(codedText).map(([, code]) => code);
    } else {
      // If codedText is null/undefined, return null
      return null;
    }
  });
}

function hausdorffEmbeddingDistance(A, B, ACounts = null, BCounts = null) {
  /**
   * Modified Hausdorff distance calculation.
   *
   * @param {Array<Array<number>>} A - Array of embedding vectors, shape [num_vectors, num_embedding_dimensions].
   * @param {Array<Array<number>>} B - Array of embedding vectors, shape [num_vectors, num_embedding_dimensions].
   * @param {Array<number>} ACounts - Array of counts (frequencies) for each vector in A.
   * @param {Array<number>} BCounts - Array of counts (frequencies) for each vector in B.
   * @return {number} - Modified Hausdorff distance.
   */

  
  if (A.length === 0 && B.length === 0) {
    // If both sets have no codes, return 0
    return 0;
  } else if (A.length === 0 || B.length === 0) {
    // If one set has no codes, return NaN
    return NaN;
  }

  // If counts are not provided, default to ones
  if (!ACounts) ACounts = Array(A.length).fill(1);
  if (!BCounts) BCounts = Array(B.length).fill(1);

  // Normalize the embedding vectors
  A = normalizeRows(A);
  B = normalizeRows(B);

  // Calculate cosine similarity and distance
  const cosineSim = matrixInnerProduct(A, B); // Shape: [n_A, n_B]
  const cosineDist = cosineSim.map((row) => row.map((value) => 1.0 - value)); // Cosine distance

  // Compute the Hausdorff distance
  const resA = ACounts.reduce(
    (sum, count, i) => sum + count * Math.min(...cosineDist[i]),
    0
  );

  const resB = BCounts.reduce(
    (sum, count, j) =>
      sum +
      count *
        Math.min(...cosineDist.map((row) => row[j])),
    0
  );

  const totalRes = resA + resB;
  const totalCounts = ACounts.reduce((a, b) => a + b, 0) + BCounts.reduce((a, b) => a + b, 0);

  return totalRes / totalCounts;
}

// Helper function to normalize rows of a matrix
function normalizeRows(matrix) {
  return matrix.map((row) => {
    const norm = Math.sqrt(row.reduce((sum, value) => sum + value ** 2, 0));
    return row.map((value) => value / norm);
  });
}

// Helper function to calculate the inner product between two matrices
function matrixInnerProduct(A, B) {
  return A.map((aRow) =>
    B.map((bRow) =>
      aRow.reduce((sum, aVal, index) => sum + aVal * bRow[index], 0)
    )
  );
}