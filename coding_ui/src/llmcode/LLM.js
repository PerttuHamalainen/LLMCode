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