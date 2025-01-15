import React, { useState, useEffect } from "react";
import { DARK_ACCENT_COLOR, HUMAN_HL_COLOR_ACTIVE } from "./colors";

const CodeList = ({ highlights, focusedOnAny }) => {
  const [codeList, setCodeList] = useState([]);
  const [isHovered, setIsHovered] = useState(false);

  useEffect(() => {
    if (!focusedOnAny) {
      setCodeList(getCodes(highlights));
    }
  }, [highlights, focusedOnAny]);

  const getCodes = (highlights) => {
    const codeCounts = {};

    // Count occurrences of each code
    highlights.forEach((hl) => {
      if (!hl.codes) return; // Skip if no codes

      // Split codes by ";" and count each code
      hl.codes.split(";").forEach((code) => {
        const trimmedCode = code.trim();
        if (trimmedCode) {
          codeCounts[trimmedCode] = (codeCounts[trimmedCode] || 0) + 1;
        }
      });
    });

    // Convert to an array of { name, count } objects and sort alphabetically
    return Object.entries(codeCounts)
      .map(([name, count]) => ({ name, count }))
      .sort((a, b) => a.name.localeCompare(b.name));
  };

  // A small helper component with the code-list rendering logic
  const renderCodeList = () => {
    return (
      <div style={{ display: "flex", flexDirection: "column", gap: "2px" }}>
        {codeList.length > 0 ? (
          codeList.map((code, idx) => (
            <div
              key={idx}
              style={{
                display: "flex",
                justifyContent: "space-between",
                gap: "2px",
              }}
            >
              <p
                style={{
                  margin: 0,
                  padding: 0,
                  whiteSpace: "nowrap",
                  overflow: "hidden",
                  textOverflow: "ellipsis",
                }}
                title={code.name}
              >
                {code.name}
              </p>
              <p style={{ margin: 0, padding: 0 }}>{code.count}</p>
            </div>
          ))
        ) : (
          <p
            style={{
              color: "#333",
              fontSize: "14px",
              lineHeight: 1.6,
              margin: 0,
              padding: 0,
            }}
          >
            Start coding by highlighting sections with your cursor and typing the
            relevant code(s) separated by a semicolon. A highlight is specific to
            a single text, so ensure that your selection does not span across
            multiple texts.
          </p>
        )}
      </div>
    );
  };

  return (
    <div
      style={{ position: "relative", display: "inline-block", pointerEvents: "auto" }}
      onMouseEnter={() => setIsHovered(true)}
      onMouseLeave={() => setIsHovered(false)}
    >
      {/* Round marker */}
      <div
        style={{
          borderRadius: "50%",
          backgroundColor: HUMAN_HL_COLOR_ACTIVE,
          color: "white",
          width: "60px",
          height: "60px",
          display: "flex",
          flexDirection: "column",
          alignItems: "center",
          justifyContent: "center",
          boxShadow: "0 4px 12px rgba(0, 0, 0, 0.08)",
          cursor: "pointer",
        }}
      >
        {/* Example inline SVG for a book icon (you can replace it with your own icon) */}
        <svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 24 24" width="25" height="25" fill="currentColor"><path d="M22.922,1.7a2.985,2.985,0,0,0-2.458-.648l-6.18,1.123A3.993,3.993,0,0,0,12,3.461,3.993,3.993,0,0,0,9.716,2.172L3.536,1.049A3,3,0,0,0,0,4V20.834l12,2.183,12-2.183V4A2.992,2.992,0,0,0,22.922,1.7ZM11,20.8,2,19.166V4a1,1,0,0,1,1.179-.983L9.358,4.14A2,2,0,0,1,11,6.108Zm11-1.636L13,20.8V6.108A2,2,0,0,1,14.642,4.14l6.179-1.123A1,1,0,0,1,22,4Z"/></svg>
        {/* <span
          style={{
            fontSize: "8px",
            marginTop: "2px",
            textTransform: "uppercase",
            letterSpacing: "1px",
          }}
        >
          Codebook
        </span> */}
      </div>

      {/* Popup code list shown only on hover */}
      {isHovered && (
        <div
          style={{
            position: "absolute",
            bottom: "0px",
            left: "-220px", // Adjust as needed to position the pop-up to the left
            transform: "translateY(100%)",
            backgroundColor: "#fff",
            border: "1px solid #ccc",
            borderRadius: "4px",
            padding: "10px",
            boxShadow: "0 2px 6px rgba(0,0,0,0.2)",
            width: "200px",
            zIndex: 9999,
          }}
        >
          {renderCodeList()}
        </div>
      )}
    </div>
  );
};

export default CodeList;