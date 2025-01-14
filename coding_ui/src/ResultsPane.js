import React from "react";
import ScoreBox from "./components/ScoreBox";

const ResultsPane = ({ evalSession }) => {
  return (
    <div
      style={{
        width: "100%",
        padding: "20px",
        boxSizing: "border-box"
      }}
    >
      <div style={{
        display: "flex",
        flexDirection: "column",
        gap: "20px"
      }}>
        { evalSession.results.map((result, idx) => 
          <ResultsRow result={result} key={idx} />
        )}
      </div>
    </div>
  );
};

const ResultsRow = ({ result }) => {
  return (
    <div style={{
      display: "flex",
      flexDirection: "row",
      gap: "10px",
    }}>
      <ScoreBox score={result.iou} name={"IoU"} color={"pink"} />
      <ScoreBox score={result.hausdorffDistance} name={"Hausdorff"} color={"pink"} />
      
      <div style={{
        fontSize: "12px"
      }}>
        {result.text}
      </div>
    </div>
  )
}

export default ResultsPane;