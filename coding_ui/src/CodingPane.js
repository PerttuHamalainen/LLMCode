import TextPane from "./TextPane";
import React, { useState } from "react";
import LLMPane from "./LLMPane";
import EvalTopBar from "./EvalTopBar";
import { NEUTRAL_MEDIUM_DARK_COLOR } from "./colors";

const CodingPane = ({
  texts,
  getAncestors,
  setHighlightsForId,
  focusedOnAny,
  createLog,
  setAnnotated,
  setExample,
  evalSession,
  apiKey,
  setApiKey,
  researchQuestion,
  setResearchQuestion,
  codeWithLLM
}) => {
  const [displayState, setDisplayState] = useState({
    sortOption: null,
    showInput: true,
    showExamples: true
  });

  // Sorting logic
  const sortedTexts = React.useMemo(() => {
    const filteredTexts = texts.filter((item) => {
      return !(item.isExample && !displayState.showExamples) && !(!item.isExample && !displayState.showInput)
    });

    if (!evalSession?.results || !displayState.sortOption || displayState.sortOption === "original") {
      return filteredTexts;
    }
  
    const sortedEval = filteredTexts.filter((item) => item.id in evalSession.results).sort((a, b) => {
      const aData = evalSession.results[a.id] || {};
      const bData = evalSession.results[b.id] || {};
  
      const aIou = aData.iou != null ? aData.iou : Infinity;
      const bIou = bData.iou != null ? bData.iou : Infinity;
  
      const aHausdorff = aData.hausdorffDistance != null && !Number.isNaN(aData.hausdorffDistance) ? aData.hausdorffDistance : Infinity;
      const bHausdorff = bData.hausdorffDistance != null && !Number.isNaN(bData.hausdorffDistance) ? bData.hausdorffDistance : Infinity;
  
      let primaryComparison = 0;
      switch (displayState.sortOption) {
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
    return [...sortedEval, ...filteredTexts.filter((item) => !(item.id in evalSession.results))];
  }, [texts, evalSession, displayState]);

  return (
    <div style={{ height: "100%", display: "flex", flexDirection: "column", position: "relative" }}>
      {/* Floating elements */}
      <div
        style={{
          position: "absolute",
          top: "10px",
          width: "100%",
          display: "flex",
          gap: "10px",
          justifyContent: "flex-end",
          padding: "0px 20px",
          boxSizing: "border-box",
          pointerEvents: "none",
        }}
      >
        {evalSession?.results && (
          <EvalTopBar displayState={displayState} setDisplayState={setDisplayState} />
        )}

        <LLMPane
          texts={texts}
          apiKey={apiKey}
          setApiKey={setApiKey}
          researchQuestion={researchQuestion}
          setResearchQuestion={setResearchQuestion}
          codeWithLLM={codeWithLLM}
          evalSession={evalSession}
        />
      </div>

      {/* Divider */}
      <div
        style={{
          position: "absolute",
          top: "0px",
          left: "0px",
          width: "600px",
          height: "100%",
          // width: "1px",
          backgroundColor: "white",
        }}
      >
      </div>

      {/* Coding Pane */}
      <div
        style={{
          height: "100%",
          overflowY: "auto",
          display: "flex",
          flexDirection: "column",
          gap: "40px",
          padding: `${evalSession?.results ? "80" : "20"}px 35px`,
          boxSizing: "border-box",
          scrollbarWidth: "none",
          msOverflowStyle: "none",
          zIndex: 1,
        }}
      >
        {sortedTexts.map((item, idx) => {
          return (
            <div
              style={{
                display: "flex",
                gap: "18px",
                // marginLeft: item.isExample ? "28px" : "50px",
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