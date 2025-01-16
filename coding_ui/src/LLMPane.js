import React, { useState } from "react";
import {
  DARK_ACCENT_COLOR,
  NEUTRAL_MEDIUM_DARK_COLOR,
  HUMAN_HL_COLOR_ACTIVE,
} from "./colors";

const MIN_ANNOTATED = 80;
const MIN_EXAMPLES = 2;

const LLMPane = ({ texts, apiKey, prompt, setPrompt, codeWithLLM }) => {
  const [isExpanded, setIsExpanded] = useState(false);
  const toggleExpand = () => setIsExpanded((prev) => !prev);

  const annotatedCount = texts.filter((obj) => obj.isAnnotated).length;
  const exampleCount = texts.filter((obj) => obj.isExample).length;

  const isCodingDisabled =
    annotatedCount < MIN_ANNOTATED ||
    exampleCount < MIN_EXAMPLES ||
    !apiKey.submitted ||
    prompt.researchQuestion.trim() === "";

  return (
    <div
      style={{
        position: "relative",
        zIndex: 10,
        pointerEvents: "auto"
      }}
    >
      {!isExpanded ? (
        // Small round marker
        <div
          onClick={toggleExpand}
          style={{
            borderRadius: "50%",
            backgroundColor: DARK_ACCENT_COLOR,
            color: "white",
            width: "60px",
            height: "60px",
            display: "flex",
            alignItems: "center",
            justifyContent: "center",
            boxShadow: "0 4px 12px rgba(0, 0, 0, 0.08)",
            cursor: "pointer",
            transition: "transform 0.3s ease",
          }}
        >
          <b>AI</b>
        </div>
      ) : (
        // Expanded LLMPane
        <div
          style={{
            width: "350px",
            backgroundColor: "#fff",
            borderRadius: "12px",
            boxShadow: "0 4px 12px rgba(0, 0, 0, 0.08)",
            border: `1px solid ${NEUTRAL_MEDIUM_DARK_COLOR}`,
            padding: "20px",
            boxSizing: "border-box",
            overflowY: "auto",
            transition: "transform 0.3s ease",
            transform: isExpanded ? "scale(1)" : "scale(0.8)",
          }}
        >
          <div
            onClick={toggleExpand}
            style={{
              position: "absolute",
              top: "10px",
              right: "10px",
              fontSize: "14px",
              color: "#999",
              cursor: "pointer",
            }}
          >
            ✕
          </div>
          {!apiKey.submitted ? (
            <div
              style={{
                display: "flex",
                justifyContent: "center",
                alignItems: "center",
              }}
            >
              <p
                style={{
                  width: "150px",
                  color: "#333",
                  fontSize: "14px",
                  textAlign: "center",
                  lineHeight: 1.6,
                }}
              >
                Please add your OpenAI API key above to use LLM coding.
              </p>
            </div>
          ) : (
            <div>
              <div style={{ display: "flex", flexDirection: "column", marginBottom: "10px" }}>
                <label
                  style={{
                    fontSize: "11px",
                    textTransform: "uppercase",
                    marginBottom: "5px",
                    color: "#555",
                    textAlign: "left",
                  }}
                >
                  Prompt instructions
                </label>
                <textarea
                  value={prompt.instructions}
                  onChange={(e) => setPrompt((p) => ({ ...p, instructions: e.target.value }))}
                  placeholder="Enter dashed list of coding instructions (Optional)"
                  style={{
                    width: "100%",
                    fontSize: "12.5px",
                    flex: 1,
                    padding: "5px",
                    border: "1px solid #ccc",
                    borderRadius: "3px",
                    boxSizing: "border-box",
                    minHeight: "80px",
                    resize: "none",
                  }}
                />
              </div>

              <div style={{ display: "flex", flexDirection: "column" }}>
                <label
                  style={{
                    fontSize: "11px",
                    textTransform: "uppercase",
                    marginBottom: "5px",
                    color: "#555",
                    textAlign: "left",
                  }}
                >
                  Research Question
                </label>
                <input
                  type="text"
                  value={prompt.researchQuestion}
                  onChange={(e) => setPrompt((p) => ({ ...p, researchQuestion: e.target.value }))}
                  placeholder="Enter research question"
                  style={{
                    width: "100%",
                    flex: 1,
                    padding: "5px",
                    border: "1px solid #ccc",
                    borderRadius: "3px",
                    boxSizing: "border-box",
                    fontSize: "12.5px",
                  }}
                />
              </div>

              <div style={{ paddingTop: "10px" }}>
                <button
                  onClick={codeWithLLM}
                  disabled={isCodingDisabled}
                  style={{
                    padding: "10px 20px",
                    backgroundColor: isCodingDisabled ? "lightGray" : DARK_ACCENT_COLOR,
                    color: isCodingDisabled ? "gray" : "white",
                    fontWeight: "bold",
                    cursor: isCodingDisabled ? "not-allowed" : "pointer",
                    width: "100%",
                    border: "none",
                    borderRadius: "5px",
                  }}
                >
                  Code with AI
                </button>
                {isCodingDisabled && (
                  <p style={{ color: "gray", fontSize: "12px", margin: "10px 0px -5px 0px" }}>
                    You need at least 80 annotated texts, 2 examples, and a research question to use
                    this feature.
                  </p>
                )}
              </div>
            </div>
          )}
        </div>
      )}
    </div>
  );
};

export default LLMPane;