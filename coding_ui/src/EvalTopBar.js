import Checkbox from "./components/Checkbox";
import { DARK_ACCENT_COLOR, EXAMPLE_GREEN_COLOR, RED_COLOR } from "./colors";
import Tooltip from "./components/Tooltip";

const EvalTopBar = ({ displayState, setDisplayState, evalAverages, prevAverages }) => {
  const metricChangeSymbol = (currentValue, previousValue) => {
    if (currentValue.toFixed(2) > previousValue.toFixed(2)) {
      return <b style={{ color: EXAMPLE_GREEN_COLOR }}>↑ </b>;
    } else if (currentValue.toFixed(2) < previousValue.toFixed(2)) {
      return <b style={{ color: RED_COLOR }}>↓ </b>;
    } else {
      return <></>;
    }
  };

  return (
    <div
      style={{
        width: "100%",
        flex: 1,
        height: "25px",
        display: "flex",
        justifyContent: "space-between",
        alignItems: "center",
        backgroundColor: DARK_ACCENT_COLOR,
        borderRadius: "12px",
        boxShadow: "0 4px 12px rgba(0, 0, 0, 0.2)",
        padding: "10px 15px 10px 10px",
        boxSizing: "border-box",
        zIndex: 6,
        pointerEvents: "auto"
      }}
    >
      <div style={{ display: "flex", gap: "15px" }}>
        <select
          id="sort-menu"
          value={displayState.sortOption || ""}
          onChange={(e) => setDisplayState((s) => ({ ...s, sortOption: e.target.value }))}
          style={{
            padding: "5px",
            border: "1px solid #ccc",
            borderRadius: "4px",
            cursor: "pointer",
          }}
        >
          <option value="" disabled>
            Sort by
          </option>
          <option value="original">Original order</option>
          <option value="leastSimilarHighlights">Highlight similarity (worst)</option>
          <option value="mostSimilarHighlights">Highlight similarity (best)</option>
          <option value="leastSimilarCodes">Code similarity (worst)</option>
          <option value="mostSimilarCodes">Code similarity (best)</option>
        </select>

        <Checkbox
          text="Show input"
          isChecked={displayState.showInput}
          setIsChecked={(isChecked) => setDisplayState((s) => ({ ...s, showInput: isChecked }))}
        />
        <Checkbox
          text="Show examples"
          isChecked={displayState.showExamples}
          setIsChecked={(isChecked) => setDisplayState((s) => ({ ...s, showExamples: isChecked }))}
        />
      </div>

      <div
        style={{
          display: "flex",
          gap: "15px",
          color: "white",
          height: "25px",
          alignItems: "center",
          fontSize: "14px",
        }}
      >
        <Tooltip description="Average highlight similarity">
          <p>
            Highlights <b>{evalAverages.highlightSimilarity.toFixed(2)}</b>{" "}
            {prevAverages && (
              <>
                {metricChangeSymbol(
                  evalAverages.highlightSimilarity,
                  prevAverages.highlightSimilarity
                )}
                ({prevAverages.highlightSimilarity.toFixed(2)})
              </>
            )}
          </p>
        </Tooltip>
        <Tooltip description="Average code similarity">
          <p>
            Codes <b>{evalAverages.codeSimilarity.toFixed(2)}</b>{" "}
            {prevAverages && (
              <>
                {metricChangeSymbol(
                  evalAverages.codeSimilarity,
                  prevAverages.codeSimilarity
                )}
                ({prevAverages.codeSimilarity.toFixed(2)})
              </>
            )}
          </p>
        </Tooltip>
      </div>
    </div>
  );
};

export default EvalTopBar;