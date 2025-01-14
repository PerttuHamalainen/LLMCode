import React from "react";

const ProgressBar = ({ progressFrac, color, backgroundColor }) => {
  const containerStyle = {
    backgroundColor: backgroundColor,
    borderRadius: "10px",
    height: "7px",
    width: "150px",
    overflow: "hidden",
    position: "relative",
  };

  const fillerStyle = {
    backgroundColor: color,
    width: `${progressFrac * 100}%`,
    height: "100%",
    borderRadius: "10px",
    transition: "width 0.3s ease",
  };

  return (
    <div style={containerStyle}>
      <div style={fillerStyle}></div>
    </div>
  );
};

export default ProgressBar;