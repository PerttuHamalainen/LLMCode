import React from "react";
import ExamplePane from "./ExamplePane";
import ResultsPane from "./ResultsPane";

const EvalPane = ({ evalSession }) => {
  return (
    <div
      style={{
        overflowY: "auto",
        width: "400px",
        height: "100%",
        borderLeft: "1px solid #ddd",
      }}
    >
      { evalSession.results ?
        <div style={{ display: "flex", flexDirection: "column" }}>
            <ExamplePane examples={evalSession.examples} />
            <ResultsPane evalSession={evalSession} />
        </div>
        :
        <p>Coding with AI...</p>
      }
    </div>
  );
};

export default EvalPane;