import React, { useState } from "react";

const Tooltip = ({ children, description }) => {
  const [isHovered, setIsHovered] = useState(false);

  return (
    <div
      onMouseEnter={() => setIsHovered(true)}
      onMouseLeave={() => setIsHovered(false)}
      style={{ position: "relative", display: "inline-block", cursor: "default" }}
    >
      {children}
      {isHovered && (
        <div
          style={{
            position: "absolute",
            top: "100%", // Position below the element
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

export default Tooltip;