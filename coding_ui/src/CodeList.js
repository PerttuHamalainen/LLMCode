import React, { useState, useEffect } from "react";

const CodeList = ({ highlights, focusedOnAny }) => {
  const [codeList, setCodeList] = useState([]);

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

  return (
    <div style={{ display: "flex", flexDirection: "column", gap: "2px" }}>
      <h2>Codes</h2>

      {codeList.length > 0 ? (
        codeList.map((code, idx) => (
          <div key={idx} style={{ display: "flex", justifyContent: "space-between", gap: "2px" }}>
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
            <p style={{ margin: 0, padding: 0 }}>
              {code.count}
            </p>
          </div>
        ))
      ) : (
        <p style={{ color: "#333", fontSize: "14px", lineHeight: 1.6, margin: 0, padding: 0 }}>
          Start coding by highlighting sections with your cursor and typing the relevant code(s) separated by a semicolon. A highlight is specific to a single text, so ensure that your selection does not span across multiple texts.
        </p>
      )
    }
    </div>
  );
}

export default CodeList;