import TextPane from "./TextPane";
import React, { useState } from "react";

const CodingPane = ({
  texts,
  getAncestors,
  setHighlightsForId,
  focusedOnAny,
  createLog,
  setAnnotated,
  setExample,
  evalSession
}) => {
  const [sortOption, setSortOption] = useState(null); // Tracks the selected sorting option

  // Sorting logic
  const sortedTexts = React.useMemo(() => {
    if (!evalSession?.results || !sortOption || sortOption === "original") return texts;
  
    const sorted = [...texts].sort((a, b) => {
      const aData = evalSession.results[a.id] || {};
      const bData = evalSession.results[b.id] || {};
  
      const aIou = aData.iou != null ? aData.iou : Infinity; // Fallback only if null or undefined
      const bIou = bData.iou != null ? bData.iou : Infinity;
  
      const aHausdorff = aData.hausdorffDistance != null ? aData.hausdorffDistance : Infinity;
      const bHausdorff = bData.hausdorffDistance != null ? bData.hausdorffDistance : Infinity;
  
      switch (sortOption) {
        case "leastSimilarHighlights":
          return aIou - bIou;
        case "mostSimilarHighlights":
          return bIou - aIou;
        case "leastSimilarCodes":
          return bHausdorff - aHausdorff;
        case "mostSimilarCodes":
          return aHausdorff - bHausdorff;
        default:
          return 0;
      }
    });

    return sorted;
  }, [texts, evalSession, sortOption]);

  return (
    <div style={{ height: "100%", display: "flex", flexDirection: "column" }}>
      {/* Top Bar for Sorting */}
      {evalSession?.results && (
        <div
          style={{
            padding: "10px",
            backgroundColor: "#f5f5f5",
            borderBottom: "1px solid #ddd",
          }}
        >
          <label htmlFor="sort-menu" style={{ marginRight: "10px" }}>
            Sort by:
          </label>
          <select
            id="sort-menu"
            value={sortOption || ""}
            onChange={(e) => setSortOption(e.target.value)}
            style={{
              padding: "5px",
              border: "1px solid #ccc",
              borderRadius: "4px",
              cursor: "pointer",
            }}
          >
            <option value="" disabled>
              Select sorting option
            </option>
            <option value="original">Original order</option>
            <option value="leastSimilarHighlights">Least similar highlights</option>
            <option value="mostSimilarHighlights">Most similar highlights</option>
            <option value="leastSimilarCodes">Least similar codes</option>
            <option value="mostSimilarCodes">Most similar codes</option>
          </select>
        </div>
      )}

      {/* Coding Pane */}
      <div
        style={{
          height: "100%",
          overflowY: "auto",
          display: "flex",
          flexDirection: "column",
          gap: "20px",
          padding: "25px 10px",
          boxSizing: "border-box",
          scrollbarWidth: "none",
          msOverflowStyle: "none",
        }}
      >
        {sortedTexts.map((item, idx) => {
          return (
            <div
              style={{
                display: "flex",
                gap: "18px",
                marginLeft: item.isExample ? "28px" : "50px",
              }}
              key={idx}
            >
              {item.isExample && (
                <div
                  style={{
                    width: "4px",
                    borderRadius: "2px",
                    height: "100%",
                    backgroundColor: "#a2e8c5",
                  }}
                />
              )}
              <TextPane
                item={item}
                getAncestors={getAncestors}
                highlights={item.highlights}
                setHighlights={(updateFunc) => setHighlightsForId(item.id, updateFunc)}
                focusedOnAny={focusedOnAny}
                createLog={(logData) =>
                  createLog({ ...logData, textId: item.id })
                }
                setAnnotated={(isAnnotated) => setAnnotated(item.id, isAnnotated)}
                setExample={(isExample) => setExample(item.id, isExample)}
                evalData={evalSession.results ? evalSession.results[item.id] : null}
              />
            </div>
          );
        })}
      </div>
    </div>
  );
};

export default CodingPane;