import React, { useRef, useState, useEffect } from "react";
import './CodingPane.css';
import CodeLabel from "./CodeLabel";
import ToggleButton from "./components/ToggleButton";
import { HUMAN_HL_COLOR, HUMAN_HL_COLOR_ACTIVE, MODEL_HL_COLOR, MODEL_HL_COLOR_ACTIVE, NEUTRAL_MEDIUM_COLOR } from "./colors";
import ScoreBox from "./components/ScoreBox";

const TextPane = ({ item, getAncestors, highlights, setHighlights, focusedOnAny, createLog, setAnnotated, setExample, evalData }) => {
  const textRef = useRef(null);

  const [showContext, setShowContext] = useState(false);

  // Group highlights based on their position in the text for displaying in the UI
  const [groupedHighlights, setGroupedHighlights] = useState([]);
  useEffect(() => {
    function findNodeForIndex(root, targetIndex) {
      let index = targetIndex;
      const walker = document.createTreeWalker(root, NodeFilter.SHOW_TEXT);

      while (walker.nextNode()) {
        const currentNode = walker.currentNode;
        const length = currentNode.textContent.length;

        if (index < length) {
          // The character index is within this text node
          return { node: currentNode, offset: index };
        }
        // Otherwise, skip this entire text node and decrement the index
        index -= length;
      }

      // If index is out of range, return the last text node with offset at the end
      return { node: null, offset: 0 };
    }

    function getYCoordinate(charIndex, textRef) {
      const root = textRef.current;

      // Find the text node and offset where this index lands
      const { node, offset } = findNodeForIndex(root, charIndex);

      // If we couldn’t find a text node (index out of range), return 0 or handle accordingly
      if (!node) {
        return 0;
      }

      // Create a range that starts and ends at the exact character position
      const range = document.createRange();
      range.setStart(node, offset);
      range.setEnd(node, offset);

      // Measure the bounding rectangle of this zero-width range
      const rects = range.getClientRects();

      if (!rects.length) {
        return 0;
      }

      // Calculate Y coordinate relative to the top of the container element
      const containerRect = root.getBoundingClientRect();
      const charRect = rects[0]; 
      const yCoord = charRect.top - containerRect.top;

      return yCoord;
    }
    
    const groupHighlights = (highlights, textRef) => {
      // Populate y values for highlights using getYCoordinate with hl.startIndex as input
      const highlightsWithY = highlights
        .map((hl) => ({
          ...hl,
          y: getYCoordinate(hl.startIndex, textRef),
        }))
        .filter((hl) => hl.y !== null);
    
      // Group highlights by `y` using a dictionary (object)
      const groups = highlightsWithY.reduce((acc, hl) => {
        const { y } = hl;
        if (!acc[y]) {
          acc[y] = [];
        }
        acc[y].push(hl);
        return acc;
      }, {});
    
      // Sort each group by `startIndex`
      const sortedGroups = Object.values(groups).map((group) =>
        group.sort((a, b) => a.startIndex - b.startIndex)
      );
    
      // Sort the outer list by `y`
      return sortedGroups.sort((a, b) => a[0].y - b[0].y);
    };

    if (textRef.current) {
      setGroupedHighlights(groupHighlights(highlights, textRef));
    }
  }, [highlights]);

  const handleTextSelect = () => {
    const selection = window.getSelection();
    if (!selection.rangeCount) return;
  
    const range = selection.getRangeAt(0);
    const selectedString = selection.toString();
  
    // Ensure the selection is inside the text div only(textRef)
    const rootNode = textRef.current;
    if (!rootNode || !rootNode.contains(range.startContainer) || !rootNode.contains(range.endContainer)) {
      return;
    }
  
    // Calculate global start and end indices
    let currentIndex = 0;
    let startIndex = -1;
    let endIndex = -1;
  
    // Use a TreeWalker to traverse all text nodes
    const walker = document.createTreeWalker(rootNode, NodeFilter.SHOW_TEXT, null, false);
  
    while (walker.nextNode()) {
      const textNode = walker.currentNode;
      const nodeLength = textNode.textContent.length;
  
      // Check if the selection starts within the current text node
      if (textNode === range.startContainer) {
        startIndex = currentIndex + range.startOffset;
      }
  
      // Check if the selection ends within the current text node
      if (textNode === range.endContainer) {
        endIndex = currentIndex + range.endOffset;
        break;
      }
  
      // Update the cumulative index
      currentIndex += nodeLength;
    }
  
    // Validate indices
    if (selectedString && startIndex !== -1 && endIndex !== -1) {
      // Check for overlap with existing human highlights
      const isOverlapping = highlights.filter((hl) => hl.type === "human").some(
        (hl) =>
          (startIndex >= hl.startIndex && startIndex < hl.endIndex) || // New highlight starts inside an existing highlight
          (endIndex > hl.startIndex && endIndex <= hl.endIndex) ||     // New highlight ends inside an existing highlight
          (startIndex <= hl.startIndex && endIndex >= hl.endIndex)     // New highlight completely overlaps an existing highlight
      );

      if (isOverlapping) {
        alert("Overlapping highlights are not supported.");
        window.getSelection().removeAllRanges();
        return;
      }

      // Create the new highlight
      const highlight = {
        id: crypto.randomUUID(),
        startIndex,
        endIndex,
        codes: "",
        text: selectedString,
        focused: true,
        hovered: false,
        type: "human"
      };

      // Add creation entry into edit log
      createLog({
        event: "create",
        highlight: (({ id, startIndex, endIndex }) => ({ id, startIndex, endIndex }))(highlight)
      })

      console.log("New hl", highlight)

      // Update highlights state
      setHighlights((prevHighlights) => [
        ...prevHighlights.map((hl) => ({ ...hl, focused: false })),
        highlight,
      ]);
    }
  };

  // Handler to update the text of a specific code
  const updateCodes = (newText, id) => {
    setHighlights((prevHighlights) =>
      prevHighlights.map((hl) =>
        hl.id === id ? { ...hl, codes: newText } : hl
      )
    );
  };

  // Handler to update the focus of a specific code
  const updateFocus = (focused, codes, id) => {
    // Handle logging for each highlight at blur
    if (!focused) {
      if (codes === "") {
        createLog({
          event: "delete",
          highlight: { 
            id: id 
          }
        })
      } else {
        createLog({
          event: "edit",
          highlight: { 
            id: id,
            codes: codes,
          }
        })
      }
    }

    // Update focus in highlights, deleting the hl if it has empty codes at blur
    setHighlights((prevHighlights) =>
      prevHighlights
        .map((hl) => (hl.id === id ? { ...hl, focused: focused } : hl))
        .filter((hl) => !(hl.id === id && !focused && codes === ""))
    );
  };

  const updateHover = (hovered, id) => {
    setHighlights((prevHighlights) =>
      prevHighlights.map((hl) =>
        hl.id === id ? { ...hl, hovered: hovered } : hl
      )
    );
  };

  const renderHighlightedText = (text, highlights) => {
    // Function to combine styles for overlapping highlights
    const getHighlightStyle = (highlight) => {
      const styles = {};
      if (highlight.type === "human") {
        styles.backgroundColor = highlight.focused || (!focusedOnAny && highlight.hovered) ? HUMAN_HL_COLOR_ACTIVE : HUMAN_HL_COLOR;
      }
      if (highlight.type === "model") {
        styles.borderBottom = highlight.focused || (!focusedOnAny && highlight.hovered) ? `4px solid ${MODEL_HL_COLOR_ACTIVE}` : `4px solid ${MODEL_HL_COLOR}`;
      }
      return styles;
    };

    // Get all highlight changing points in ascending order, excluding 0
    const highlightChanges = [...new Set(
      highlights
        .flatMap((hl) => [hl.startIndex, hl.endIndex])
    )]
    .sort((a, b) => a - b)
    .filter((a) => a > 0);
  
    const parts = [];
    let startIndex = 0;
    highlightChanges.forEach((endIndex) => {
      // Highlight from startIdx to endIdx, as highlights are constant in this span
  
      // Find highlights in this span
      const spanHighlights = highlights.filter(
        (hl) => hl.startIndex < endIndex && hl.endIndex > startIndex
      );
  
      // Generate the combined style for overlapping highlights
      const combinedStyle = spanHighlights.reduce((acc, highlight) => {
        return { ...acc, ...getHighlightStyle(highlight) };
      }, {});
  
      // Add the highlighted text with combined styles
      parts.push(
        <span
          key={`highlight-${startIndex}`}
          style={combinedStyle}
          onMouseEnter={() => spanHighlights.forEach((hl) => updateHover(true, hl.id))}
          onMouseLeave={() => spanHighlights.forEach((hl) => updateHover(false, hl.id))}
        >
          {text.slice(startIndex, endIndex)}
        </span>
      );
  
      // Update the start index
      startIndex = endIndex;
    });
  
    // Add any remaining non-highlighted text
    if (startIndex < text.length) {
      parts.push(<span key="text-end">{text.slice(startIndex)}</span>);
    }
  
    return parts;
  };

  return (
    <div style={{ display: "flex", flexDirection: "column", gap: "10px", padding: "20px 0px 10px 0px" }}>
      <div style={{ display: "flex", alignItems: "center", gap: "10px" }}>
        { evalData &&
          <>
            <ScoreBox name={"IoU"} score={evalData.iou} color={NEUTRAL_MEDIUM_COLOR} />
            <ScoreBox name={"Hsf"} score={evalData.hausdorffDistance} color={NEUTRAL_MEDIUM_COLOR} />
          </>
        }
        <ToggleButton
          isActive={item.isAnnotated}
          onToggle={() => {
            setAnnotated(!item.isAnnotated);
            if (item.isAnnotated && item.isExample) {
              setExample(false);
            }
          }}
          activeText="✔ Annotated"
          inactiveText="Mark as annotated"
          activeColor={NEUTRAL_MEDIUM_COLOR}
        />
        <ToggleButton
          isActive={item.isExample}
          onToggle={() => {
            setExample(!item.isExample);
            if (!item.isExample && !item.isAnnotated) {
              setAnnotated(true);
            }
          }}
          activeText="Example"
          inactiveText="Make example"
          activeColor="#a2e8c5"
        />
        { item.parentId &&
          <p
            style={{
              margin: 0, // Removes default margin
              height: "auto", // Prevents it from growing unnecessarily
              color: "gray",
              fontSize: "12px",
              lineHeight: "1", // Ensures text doesn't grow vertically
              display: "flex", // Makes it align properly with flex items
              alignItems: "center", // Aligns the text vertically
              cursor: "pointer",
              userSelect: "none",
            }}
            onClick={() => setShowContext(!showContext)}
          >
            { showContext ? "Hide context" : "View context" }
          </p>
        }
      </div>

      { showContext && (
        <Ancestors ancestors={getAncestors(item.parentId)} />
      )}

      <div style={{ display: "flex", justifyContent: "center", alignItems: "stretch", gap: "30px" }} onMouseUp={handleTextSelect}>
        <div
          style={{
            width: "600px",
            textAlign: "left",
            lineHeight: "2.2"
          }}
          ref={textRef}
        >
          <p style={{ margin: 0, padding: 0 }}>
            {renderHighlightedText(item.text, highlights)}
          </p>
        </div>

        <div
          style={{
            position: "relative",
            minWidth: "400px",
          }}
        >
          {groupedHighlights.map((hlGroup, groupIndex) => (
            <div
              style={{
                position: "absolute",
                top: hlGroup[0].y - 5, // Position the group based on the Y coordinate
                display: "flex",
                gap: "6px",
              }}
              key={groupIndex}
            >
              {hlGroup.map((hl) => (
                <CodeLabel
                  highlight={hl}
                  onTextChange={(newText) => updateCodes(newText, hl.id)}
                  onFocusChange={(focused, text) => updateFocus(focused, text, hl.id)}
                  onHoverChange={(hovered) => updateHover(hovered, hl.id)}
                  focusedOnAny={focusedOnAny}
                  key={hl.id}
                />
              ))}
            </div>
          ))}
        </div>
      </div>
    </div>
  );
};

const Ancestors = ({ ancestors }) => {
  return (
    <div style={{
      width: "600px",
      display: "flex",
      flexDirection: "column",
      fontSize: "12px",
      color: "gray",
      gap: "10px",
      paddingTop: "10px",
    }}>
      {ancestors.map((ancestorText, idx) => (
        <div 
          style={{
            display: "flex",
            gap: "5px"
          }}
          key={idx}
        >
          <div style={{
              flex: "0 0 2px",
              width: "2px",
              height: "100%",
              backgroundColor: "lightGray",
          }}/>
          {ancestorText}
        </div>
      ))}
    </div>
  );
}

export default TextPane;