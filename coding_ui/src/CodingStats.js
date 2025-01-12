import React, { useState } from "react";

const CodingStats = ({ texts, minAnnotated, minExamples, onButtonClick, apiKey, setApiKey }) => {
  const [isKeySubmitted, setIsKeySubmitted] = useState(() => apiKey.trim() !== "");
  const annotatedCount = texts.filter((obj) => obj.isAnnotated).length;
  const exampleCount = texts.filter((obj) => obj.isExample).length;

  const isButtonDisabled =
    annotatedCount < minAnnotated ||
    exampleCount < minExamples ||
    !isKeySubmitted;

  const handleApiKeySubmit = () => {
    setIsKeySubmitted(true);
  };

  const handleApiKeyDelete = () => {
    setIsKeySubmitted(false);
    setApiKey("");
  };

  const handleApiKeyChange = (e) => {
    setApiKey(e.target.value);
  };

  return (
    <div
      style={{
        display: "flex",
        flexDirection: "column",
        gap: "2px",
        paddingBottom: "15px",
      }}
    >
      <h2>Statistics</h2>
      <div style={{ display: "flex", justifyContent: "space-between" }}>
        <p style={{ margin: 0, padding: 0 }}>Annotated texts</p>
        <p style={{ margin: 0, padding: 0 }}>{annotatedCount}</p>
      </div>
      <div style={{ display: "flex", justifyContent: "space-between", paddingBottom: "20px" }}>
        <p style={{ margin: 0, padding: 0 }}>Examples</p>
        <p style={{ margin: 0, padding: 0 }}>{exampleCount}</p>
      </div>

      {/* <div style={{ paddingTop: "20px" }}>
        <input
          type="text"
          id="api-key"
          value={apiKey}
          onChange={handleApiKeyChange}
          placeholder="Enter your API key"
          style={{
            width: "100%",
            padding: "5px",
            marginTop: "5px",
            marginBottom: "10px",
            border: "1px solid #ccc",
            borderRadius: "3px",
            boxSizing: "border-box",
          }}
        />
      </div> */}
      {!isKeySubmitted ? (
        <div style={{ display: "flex", gap: "5px", marginTop: "5px" }}>
        <input
          type="text"
          id="api-key"
          value={apiKey}
          onChange={handleApiKeyChange}
          placeholder="Enter API key"
          style={{
            width: "100px",
            flex: 1,
            padding: "5px",
            border: "1px solid #ccc",
            borderRadius: "3px",
            boxSizing: "border-box",
          }}
        />
        <button
          onClick={handleApiKeySubmit}
          disabled={apiKey.trim() === ""}
          style={{
            padding: "5px 10px",
            backgroundColor: apiKey.trim() === "" ? "#ccc" : "#007bff",
            color: "white",
            border: "none",
            borderRadius: "3px",
            cursor: apiKey.trim() === "" ? "not-allowed" : "pointer",
          }}
        >
            Submit
        </button>
        </div>
    ) : (
        <div
        style={{
            marginTop: "5px",
            padding: "5px",
            border: "1px solid #ccc",
            borderRadius: "3px",
            backgroundColor: "#f9f9f9",
            color: "#555",
            fontSize: "14px",
        }}
        >
        API Key: ****{" "}
        <button
            onClick={handleApiKeyDelete}
            style={{
            marginLeft: "10px",
            padding: "3px 8px",
            backgroundColor: "#007bff",
            color: "white",
            border: "none",
            borderRadius: "3px",
            cursor: "pointer",
            fontSize: "12px",
            }}
        >
            Delete
        </button>
        </div>
    )}

      <div style={{ paddingTop: "10px" }}>
        <button
          onClick={onButtonClick}
          disabled={isButtonDisabled}
          style={{
            padding: "5px 20px",
            color: isButtonDisabled ? "gray" : "#333",
            cursor: isButtonDisabled ? "not-allowed" : "pointer",
            width: "100%",
          }}
        >
          Code with AI
        </button>
        {isButtonDisabled && (
          <p style={{ color: "gray", fontSize: "12px", marginTop: "10px" }}>
            You need at least {minAnnotated} annotated texts,{" "}
            {minExamples} examples, and a valid API key to use this feature.
          </p>
        )}
      </div>
    </div>
  );
};

export default CodingStats;