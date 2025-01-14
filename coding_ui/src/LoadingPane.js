import React from "react";
import ProgressBar from "./components/ProgressBar";
import { NEUTRAL_MEDIUM_DARK_COLOR, NEUTRAL_DARK_COLOR } from "./colors";

const LoadingPane = ({ progress }) => {
  return (
    <div
      style={{
        display: "flex",
        flexDirection: "column",
        justifyContent: "center",
        alignItems: "center",
        height: "100%", // Fill the entire height
        gap: "2px", // Add spacing between elements
      }}
    >
      <p style={{ fontSize: "18px", color: "#666", margin: 0, paddingBottom: "15px" }}>Coding with AI...</p>

      { progress.current > 0 ? (
        <>
          <p style={{ color: "#666" }}>{progress.current} / {progress.max}</p>
          <ProgressBar progressFrac={progress.frac} color={NEUTRAL_DARK_COLOR} backgroundColor={NEUTRAL_MEDIUM_DARK_COLOR} />
        </>
      ) : (
        <div
          style={{
            width: "30px",
            height: "30px",
            border: `4px solid ${NEUTRAL_DARK_COLOR}`,
            borderTop: `4px solid ${NEUTRAL_MEDIUM_DARK_COLOR}`,
            borderRadius: "50%",
            animation: "spin 1s linear infinite",
          }}
        ></div>
      )}
    </div>
  );
};

export default LoadingPane;