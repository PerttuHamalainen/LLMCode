import React from "react";

const ToggleButton = ({
  isActive,
  onToggle,
  activeText = "Active",
  inactiveText = "Inactive",
  activeColor = "#90ee90",
}) => {
  return (
    <div
      style={{
        width: "fit-content",
        color: isActive ? "black" : "gray",
        backgroundColor: isActive ? activeColor : null,
        outline: isActive ? `2px solid ${activeColor}` : "2px solid lightGray",
        padding: "3px 6px",
        fontSize: "12px",
        borderRadius: "3px",
        boxSizing: "border-box",
        cursor: "pointer",
        userSelect: "none",
      }}
      onClick={onToggle}
    >
      {isActive ? activeText : inactiveText}
    </div>
  );
};

export default ToggleButton;