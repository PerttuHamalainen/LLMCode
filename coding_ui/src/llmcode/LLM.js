import OpenAI from "openai";

let client = null; // Holds the initialized OpenAI client

export function initializeClient(apiKey) {
  if (!apiKey) {
    client = null;
  }
  client = new OpenAI({ apiKey, dangerouslyAllowBrowser: true });
}

export async function queryLLM(prompt, model, maxTokens) {
  if (!client) {
    throw new Error(
      "OpenAI client is not initialized. Please call initializeClient with a valid API key."
    );
  }

  try {
    const completion = await client.chat.completions.create({
      model,
      messages: [
        { role: "system", content: "You are a helpful assistant." },
        { role: "user", content: prompt },
      ],
      max_tokens: maxTokens,
    });

    return completion.choices[0].message.content;
  } catch (error) {
    console.error("Error querying LLM:", error.message);
    throw error;
  }
}

export async function embed(texts, model, batchSize=32) {
  const escapedTexts = texts.map((text) => JSON.stringify(text));

  const embedMatrix = [];
  for (let i = 0; i < escapedTexts.length; i += batchSize) {
    // Get the current batch
    const embedBatch = escapedTexts.slice(i, Math.min(escapedTexts.length, i + batchSize));

    // Call the embedding API for the current batch
    const embeddings = await client.embeddings.create({
      input: embedBatch,
      model: model,
    });

    // Append embeddings to the result matrix
    embeddings.data.forEach((item) => {
      embedMatrix.push(item.embedding);
    });
  }
  return embedMatrix;
}

  