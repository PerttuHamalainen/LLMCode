import Checkbox from "./components/Checkbox";
import { DARK_ACCENT_COLOR } from "./colors";

const EvalTopBar = ({ displayState, setDisplayState }) => {
  return (
    <div
      style={{
        flex: 1,
        width: "630px",
        height: "25px",
        display: "flex",
        justifyContent: "space-between",
        alignItems: "center",
        backgroundColor: DARK_ACCENT_COLOR,
        borderRadius: "12px",
        boxShadow: "0 4px 12px rgba(0, 0, 0, 0.2)",
        padding: "10px 15px 10px 10px",
        zIndex: 5,
        pointerEvents: "auto"
      }}
    >
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

      <div style={{ display: "flex", gap: "15px" }}>
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
    </div>
  );
};

export default EvalTopBar;