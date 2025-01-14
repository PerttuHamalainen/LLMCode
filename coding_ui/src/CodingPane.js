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
  
    const sortedEval = texts.filter((item) => item.id in evalSession.results).sort((a, b) => {
      const aData = evalSession.results[a.id] || {};
      const bData = evalSession.results[b.id] || {};
  
      const aIou = aData.iou != null ? aData.iou : Infinity;
      const bIou = bData.iou != null ? bData.iou : Infinity;
  
      const aHausdorff = aData.hausdorffDistance != null && !Number.isNaN(aData.hausdorffDistance) ? aData.hausdorffDistance : Infinity;
      const bHausdorff = bData.hausdorffDistance != null && !Number.isNaN(bData.hausdorffDistance) ? bData.hausdorffDistance : Infinity;
  
      let primaryComparison = 0;
      switch (sortOption) {
        case "leastSimilarHighlights":
          primaryComparison = aIou - bIou;
          break;
        case "mostSimilarHighlights":
          primaryComparison = bIou - aIou;
          break;
        case "leastSimilarCodes":
          primaryComparison = bHausdorff - aHausdorff;
          break;
        case "mostSimilarCodes":
          primaryComparison = aHausdorff - bHausdorff;
          break;
        default:
          primaryComparison = 0;
      }

      if (primaryComparison !== 0) {
        return primaryComparison; // Use primary comparison if not equal
      }

      // Secondary comparison: Prefer items with more highlights
      const aHighlightsCount = a.highlights ? a.highlights.length : 0;
      const bHighlightsCount = b.highlights ? b.highlights.length : 0;
      return bHighlightsCount - aHighlightsCount; // Prefer more highlights
    });

    // Add remaining texts that weren't evaluated at the end
    return [...sortedEval, ...texts.filter((item) => !(item.id in evalSession.results))];
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
            <option value="leastSimilarHighlights">Highlight similarity (worst)</option>
            <option value="mostSimilarHighlights">Highlight similarity (best)</option>
            <option value="leastSimilarCodes">Code similarity (worst)</option>
            <option value="mostSimilarCodes">Code similarity (best)</option>
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
                evalData={evalSession?.results ? evalSession.results[item.id] : null}
              />
            </div>
          );
        })}
      </div>
    </div>
  );
};

export default CodingPane;