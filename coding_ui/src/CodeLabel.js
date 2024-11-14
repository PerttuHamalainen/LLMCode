import React, { useState, useRef, useEffect } from "react";

const CodeLabel = ({ highlight, onTextChange, onFocusChange, onHoverChange, focusedOnAny }) => {
  const [cursorPosition, setCursorPosition] = useState(0);
  const divRef = useRef(null);

  const getCursorPosition = () => {
    const selection = window.getSelection();
    const range = selection.getRangeAt(0);
    const preCaretRange = range.cloneRange();
    preCaretRange.selectNodeContents(divRef.current);
    preCaretRange.setEnd(range.endContainer, range.endOffset);
    return preCaretRange.toString().length; // Return the cursor position
  };

  // Restore the cursor position after updating the text
  const restoreCursorPosition = (position) => {
    const element = divRef.current;
    if (!element) return;

    const selection = window.getSelection();
    const range = document.createRange();
    range.selectNodeContents(element);

    // Find the correct text node and offset
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

  // Handle input and update the text state
  const handleInput = (e) => {
    setCursorPosition(getCursorPosition());
    onTextChange(e.currentTarget.textContent);
  };

  // End focus on enter
  const handleKeyDown = (e) => {
    if (e.key === "Enter") {
      onFocusChange(false)
      e.preventDefault();
    }
  };

  useEffect(() => {
    if (highlight.focused) {
      restoreCursorPosition(cursorPosition);
    }
  }, [highlight.codes, highlight.focused, cursorPosition]);

  useEffect(() => {
    if (highlight.focused && divRef.current && document.activeElement !== divRef.current) {
      divRef.current.focus();
    } else if (!highlight.focused && divRef.current && document.activeElement === divRef.current) {
      divRef.current.blur();
    }
  }, [highlight.focused]);

  const handleUIFocusChange = (newFocused) => {
    if (newFocused !== highlight.focused) {
      onFocusChange(newFocused)
    }
  }

  return (
    <div
      ref={divRef}
      style={{
        display: "inline-block",
        backgroundColor: highlight.focused || (!focusedOnAny && highlight.hovered) ? "#c7e3ff" : "#f0f0f0",
        borderRadius: "8px",
        padding: "5px 8px",
        color: highlight.codes ? "black" : "#aaa",
        position: "relative",
        whiteSpace: "nowrap",
        overflow: "hidden",
        textOverflow: "ellipsis",
      }}
      contentEditable={true}
      suppressContentEditableWarning={true}
      onInput={handleInput}
      onKeyDown={handleKeyDown}
      onFocus={() => handleUIFocusChange(true)}
      onBlur={() => handleUIFocusChange(false)}
      onMouseEnter={() => onHoverChange(true)}
      onMouseLeave={() => onHoverChange(false)}
      data-placeholder="Codes separated by ;"
    >
      {highlight.codes}
    </div>
  );
}

export default CodeLabel;