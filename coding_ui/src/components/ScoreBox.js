import React, { useState } from "react";

const ScoreBox = ({ name, score, color, description }) => {
  const [isHovered, setIsHovered] = useState(false);
  const scoreLabel = `${Number.isNaN(score) ? "0.00" : score.toFixed(2)}`;

  return (
    <div
      onMouseEnter={() => setIsHovered(true)}
      onMouseLeave={() => setIsHovered(false)}
      style={{
        position: "relative", // Needed for the tooltip positioning
        width: "fit-content",
        color: "black",
        backgroundColor: color,
        outline: `2px solid ${color}`,
        padding: "3px 6px",
        fontSize: "12px",
        borderRadius: "3px",
        boxSizing: "border-box",
        userSelect: "none",
        cursor: "default", // Changes cursor to indicate a non-clickable element
      }}
    >
      {name} <b>{scoreLabel}</b>
      {isHovered && (
        <div
          style={{
            position: "absolute",
            top: "100%", // Position below the ScoreBox
            left: "50%",
            transform: "translateX(-50%)",
            marginTop: "5px",
            backgroundColor: "rgba(0, 0, 0, 0.8)",
            color: "white",
            padding: "5px",
            borderRadius: "4px",
            fontSize: "10px",
            whiteSpace: "nowrap",
            zIndex: 1000,
            boxShadow: "0 2px 5px rgba(0, 0, 0, 0.2)",
          }}
        >
          {description || "No description available"}
        </div>
      )}
    </div>
  );
};

export default ScoreBox;