import React from "react";

const ScoreBox = ({ name, score, color }) => {
  const scoreLabel = `${name} ${Number.isNaN(score) ? "—" : score.toFixed(2)}`

  return (
    <div
      style={{
        width: "fit-content",
        color: "black",
        backgroundColor: color,
        outline: `2px solid ${color}`,
        padding: "3px 6px",
        fontSize: "12px",
        borderRadius: "3px",
        boxSizing: "border-box",
        userSelect: "none",
      }}
    >
      {scoreLabel}
    </div>
  );
};

export default ScoreBox;