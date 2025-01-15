import React, { useState } from "react";
import FileManager from "./FileManager";

const FileIcon = () => {
  return (
    <svg
      xmlns="http://www.w3.org/2000/svg"
      width="20"
      height="20"
      fill="none"
      viewBox="0 0 24 24"
      stroke="currentColor"
      strokeWidth="2"
    >
      <path
        strokeLinecap="round"
        strokeLinejoin="round"
        d="M4 19V5a2 2 0 012-2h7l5 5v11a2 2 0 01-2 2H6a2 2 0 01-2-2z"
      />
    </svg>
  );
};

const TopBar = ({ texts, fileName, editLog, onDelete, apiKey, setApiKey, studyData }) => {
  const annotatedCount = texts.filter((obj) => obj.isAnnotated).length;
  const exampleCount = texts.filter((obj) => obj.isExample).length;

  const handleApiKeySubmit = () => {
    setApiKey((value) => ({ ...value, submitted: true }));
  };

  const handleApiKeyDelete = () => {
    setApiKey({ key: "", submitted: false });
  };

  const handleApiKeyChange = (e) => {
    setApiKey((value) => ({ ...value, key: e.target.value }));
  };
  
  return (
    <div
      style={{
        width: "100%",
        height: "50px",
        display: "flex",
        justifyContent: "space-between",
        alignItems: "center",
        padding: "0px 40px",
        color: "white",
        background: "#1c1b1a",
        userSelect: "none",
        boxSizing: "border-box",
        fontSize: "14px",
      }}
    >
      { texts.length > 0 &&
        <>
          <div style={{ display: "flex", alignItems: "center", gap: "20px"}} >
            <FileManager
              texts={texts}
              editLog={editLog}
              studyData={studyData}
              onDelete={onDelete}
            />
            <div style={{ display: "flex", alignItems: "center", gap: "5px", color: "gray" }}>
              <FileIcon />
              {fileName}
            </div>
          </div>

          <div style={{ display: "flex", alignItems: "center", gap: "20px", color: "white" }}>
            <p>
              {annotatedCount} <span style={{ color: "gray"}}>annotated</span>
            </p>
            <p>
              {exampleCount} <span style={{ color: "gray"}}>examples</span>
            </p>
          </div>

          {!apiKey.submitted ? (
            <div style={{ display: "flex", gap: "10px" }}>
              <input
                type="text"
                id="api-key"
                value={apiKey.key}
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
                disabled={apiKey.key.trim() === ""}
                style={{
                  padding: "5px 10px",
                  backgroundColor: apiKey.key.trim() === "" ? "#ccc" : "#007bff",
                  color: apiKey.key.trim() === "" ? "black" : "white",
                  border: "none",
                  borderRadius: "3px",
                  cursor: apiKey.key.trim() === "" ? "not-allowed" : "pointer",
                }}
              >
                Submit
              </button>
            </div>
          ) : (
            <div
              style={{
                color: "gray",
              }}
            >
              API key added
              <button
                onClick={handleApiKeyDelete}
                style={{
                marginLeft: "20px",
                padding: "3px 8px",
                backgroundColor: "#383838",
                color: "white",
                border: "none",
                borderRadius: "3px",
                cursor: "pointer",
                }}
              >
                Remove 
              </button>
            </div>
          )}
        </>
      }
    </div>
  );
};

export default TopBar;