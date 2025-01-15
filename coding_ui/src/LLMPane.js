import React from "react";
import { DARK_ACCENT_COLOR } from "./colors";

const MIN_ANNOTATED = 20;
const MIN_EXAMPLES = 2;

const LLMPane = ({ texts, apiKey, researchQuestion, setResearchQuestion, codeWithLLM, evalSession }) => {
  const annotatedCount = texts.filter((obj) => obj.isAnnotated).length;
  const exampleCount = texts.filter((obj) => obj.isExample).length;

  const isCodingDisabled =
    annotatedCount < MIN_ANNOTATED ||
    exampleCount < MIN_EXAMPLES ||
    !apiKey.submitted ||
    researchQuestion.trim() === "";

  return (
    <div
      style={{
        width: "350px",
        backgroundColor: "#fff",
        borderRadius: "12px",
        boxShadow: "0 4px 12px rgba(0, 0, 0, 0.08)",
        padding: "20px",
        boxSizing: "border-box",
        overflowY: "auto",
        pointerEvents: "auto",
        zIndex: 5
      }}
    >
      { !apiKey.submitted ? (
        <div
          style={{
            display: "flex", // Enables Flexbox
            justifyContent: "center", // Centers horizontally
            alignItems: "center", // Centers vertically
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
          <div style={{ display: "flex", flexDirection: "column" }}>
            <input
              type="text"
              value={researchQuestion}
              onChange={(e) => setResearchQuestion(e.target.value)}
              placeholder="Enter research question"
              style={{
                width: "100%",
                flex: 1,
                padding: "5px",
                border: "1px solid #ccc",
                borderRadius: "3px",
                boxSizing: "border-box",
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
                borderRadius: "5px"
              }}
            >
              Code with AI
            </button>
            {isCodingDisabled && (
              <p style={{ color: "gray", fontSize: "12px", margin: "10px 0px -5px 0px" }}>
                You need at least {MIN_ANNOTATED} annotated texts,{" "}
                {MIN_EXAMPLES} examples, and a research question to use this feature.
              </p>
            )}
          </div>
        </div>
      )}
    </div>
  );
};

export default LLMPane;