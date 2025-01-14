import React from "react";

const Checkbox = ({ text, isChecked, setIsChecked }) => {
  const handleChange = (event) => {
    setIsChecked(event.target.checked);
  };

  return (
    <label
      style={{
        display: "flex",
        alignItems: "center",
        fontSize: "14px",
        cursor: "pointer",
        gap: "2px",
        color: "white"
      }}
    >
      <input
        type="checkbox"
        checked={isChecked}
        onChange={handleChange}
        style={{
          width: "20px", // Bigger checkbox
          height: "20px",
          cursor: "pointer", // Pointer cursor for better UX
          accentColor: "#fff",
        }}
      />
      {text}
    </label>
  );
};

export default Checkbox;