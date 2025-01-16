import React from "react";
import Tooltip from "./Tooltip";

const ScoreBox = ({ name, score, color, description }) => {
  const scoreLabel = `${score === null || Number.isNaN(score) ? "—" : score.toFixed(2)}`;

  return (
    <Tooltip description={description}>
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
        {name} <b>{scoreLabel}</b>
      </div>
    </Tooltip>
  );
};

export default ScoreBox;