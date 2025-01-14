import React, { useState, useRef, useEffect } from "react";
import { HUMAN_HL_COLOR, HUMAN_HL_COLOR_ACTIVE, MODEL_HL_COLOR, MODEL_HL_COLOR_ACTIVE, HUMAN_HL_COLOR_DARK } from "./colors";

const CodeLabel = ({ highlight, onTextChange, onFocusChange, onHoverChange, focusedOnAny }) => {
  const [text, setText] = useState(highlight.codes || "");
  const [cursorPosition, setCursorPosition] = useState(highlight.codes.length || 0);
  const [deleted, setDeleted] = useState(false);
  const divRef = useRef(null);

  const baseColor = highlight.type === "human" ? HUMAN_HL_COLOR : MODEL_HL_COLOR;
  const activeColor = highlight.type === "human" ? HUMAN_HL_COLOR_ACTIVE : MODEL_HL_COLOR_ACTIVE;

  const getCursorPosition = () => {
    const selection = window.getSelection();
    const range = selection.getRangeAt(0);
    const preCaretRange = range.cloneRange();
    preCaretRange.selectNodeContents(divRef.current);
    preCaretRange.setEnd(range.endContainer, range.endOffset);
    return preCaretRange.toString().length;
  };

  const restoreCursorPosition = (position) => {
    const element = divRef.current;
    if (!element) return;

    const selection = window.getSelection();
    const range = document.createRange();
    range.selectNodeContents(element);

    let charCount = 0;
    let found = false;
    const walker = document.createTreeWalker(element, NodeFilter.SHOW_TEXT, null, false);

    while (walker.nextNode() && !found) {
      const textNode = walker.currentNode;
      const nodeLength = textNode.textContent.length;

      if (charCount + nodeLength >= position) {
        range.setStart(textNode, position - charCount);
        range.setEnd(textNode, position - charCount);
        found = true;
      } else {
        charCount += nodeLength;
      }
    }

    selection.removeAllRanges();
    selection.addRange(range);
  };

  const handleInput = () => {
    setCursorPosition(getCursorPosition());
    setText(divRef.current.textContent);
  };

  const handleKeyDown = (e) => {
    if (e.key === "Enter") {
      divRef.current.blur();
      e.preventDefault();
    }
  };

  const handleBlur = () => {
    if (!deleted) {
      onTextChange(text);
      onFocusChange(false, text);
      setCursorPosition(text.length);
    }
  };

  const handleDelete = () => {
    setDeleted(true);
    onFocusChange(false, "");  // Deletes the highlight
  }

  useEffect(() => {
    if (highlight.focused) {
      restoreCursorPosition(cursorPosition);
    }
  }, [text, highlight.focused, cursorPosition]);

  useEffect(() => {
    if (highlight.focused && divRef.current && document.activeElement !== divRef.current) {
      divRef.current.focus();
    } else if (!highlight.focused && divRef.current && document.activeElement === divRef.current) {
      divRef.current.blur();
    }
  }, [highlight.focused]);

  return (
    <div style={{ display: "flex", alignItems: "center", gap: "3px" }}>
      <div
        ref={divRef}
        style={{
          display: "inline-block",
          backgroundColor: highlight.focused || (!focusedOnAny && highlight.hovered) ? activeColor : baseColor,
          borderRadius: "8px",
          padding: "5px 8px",
          color: "black",
          position: "relative",
          whiteSpace: "nowrap",
          overflow: "hidden",
          textOverflow: "ellipsis",
        }}
        contentEditable={highlight.type === "human"}
        suppressContentEditableWarning={true}
        onInput={handleInput}
        onKeyDown={handleKeyDown}
        onFocus={() => onFocusChange(true, text)}
        onBlur={handleBlur}
        onMouseEnter={() => onHoverChange(true)}
        onMouseLeave={() => onHoverChange(false)}
        data-placeholder="Codes separated by ;"
      >
        {text}
      </div>

      { highlight.focused &&
        <button
          onPointerDown={handleDelete}
          style={{
            width: "20px",
            height: "20px",
            borderRadius: "50%",
            backgroundColor: HUMAN_HL_COLOR_ACTIVE,
            color: "black",
            border: "none",
            display: "flex",
            alignItems: "center",
            justifyContent: "center",
            cursor: "pointer",
            fontSize: "14px",
            lineHeight: "14px",
          }}
        >
          ×
        </button>
      }
    </div>
  );
};


export default CodeLabel;