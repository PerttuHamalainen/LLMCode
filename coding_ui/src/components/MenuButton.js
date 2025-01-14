const MenuButton = ({ onClick, children, color = "black" }) => {
  const buttonStyle = {
    display: "block",
    width: "100%",
    padding: "8px",
    border: "none",
    background: "none",
    textAlign: "left",
    cursor: "pointer",
    color: color,
  };

  const handleMouseEnter = (e) => {
    e.target.style.backgroundColor = "#f0f0f0";
  };

  const handleMouseLeave = (e) => {
    e.target.style.backgroundColor = "transparent";
  };

  return (
    <button
      onClick={onClick}
      style={buttonStyle}
      onMouseEnter={handleMouseEnter}
      onMouseLeave={handleMouseLeave}
    >
      {children}
    </button>
  );
};

export default MenuButton;