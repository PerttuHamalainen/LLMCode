import React, { useState, useEffect } from "react";
import CodingPane from './CodingPane';
import EvalPane from "./EvalPane";
import CodeList from "./CodeList";
import FileUpload from "./FileUpload";
import './App.css';
import FileManager from "./FileManager";
import CodingStats from "./CodingStats";
import { initializeClient } from "./llmcode/LLM";
import { codeInductivelyWithCodeConsistency } from "./llmcode/Coding";
import { formatTextWithHighlights, nanMean, parseTextHighlights } from "./helpers";
import { runCodingEval } from "./llmcode/Metrics";

function App() {
  const [fileName, setFileName] = useState(() => {
    // Load file name from local storage if available
    const savedFileName = localStorage.getItem("fileName");
    return savedFileName ? JSON.parse(savedFileName) : "";
  });
  useEffect(() => {
    // Save file name to local storage whenever it changes
    localStorage.setItem("fileName", JSON.stringify(fileName));
  }, [fileName]);

  const [texts, setTexts] = useState(() => {
    // Load texts from local storage if available
    const savedTexts = localStorage.getItem("texts");
    return savedTexts ? JSON.parse(savedTexts) : [];
  });
  useEffect(() => {
    // Save texts to local storage whenever it changes
    localStorage.setItem("texts", JSON.stringify(texts));
  }, [texts]);

  const [focusedOnAny, setFocusedOnAny] = useState(false);
  useEffect(() => {
    setFocusedOnAny(texts.some((t) => t.highlights.some((h) => h.focused)));
  }, [texts]);

  const [editLog, setEditLog] = useState(() => {
    // Load file name from local storage if available
    const savedEditLog = localStorage.getItem("editLog");
    return savedEditLog ? JSON.parse(savedEditLog) : [];
  });
  useEffect(() => {
    // Save log to local storage whenever it changes
    localStorage.setItem("editLog", JSON.stringify(editLog));
  }, [editLog]);

  const [apiKey, setApiKey] = useState(() => {
    const savedKey = localStorage.getItem("apiKey");
    initializeClient(savedKey);
    return savedKey ? savedKey : "";
  });
  useEffect(() => {
    initializeClient(apiKey);
    localStorage.setItem("apiKey", apiKey);
  }, [apiKey]);

  const [researchQuestion, setResearchQuestion] = useState(() => {
    const savedRq = localStorage.getItem("researchQuestion");
    return savedRq ? savedRq : "";
  });
  useEffect(() => {
    localStorage.setItem("researchQuestion", researchQuestion);
  }, [researchQuestion]);

  const [evalSession, setEvalSession] = useState({
    examples: null,
    results: null
  });

  const getAncestors = (parentId) => {
    const parentText = texts.find((item) => item.id === parentId);
    if (!parentText.parentId) {
      return [parentText.text];
    }
    const ancestors = getAncestors(parentText.parentId, texts);
    return [...ancestors, parentText.text];
  };

  const codeWithLLM = async () => {
    // Remove any previous model highlights
    texts.forEach(({ id }) => {
      setHighlightsForId(id, (hls) => hls.filter((hl) => hl.type !== "model"));
    });

    // Construct input texts with ancestors
    const inputTexts = texts.filter(({ isAnnotated, isExample }) => isAnnotated && !isExample).slice(0, 30);  // TODO!!! For now only take first 30
    const inputs = inputTexts.map(({ text, parentId }) => ({
      text: text,
      ancestors: parentId ? getAncestors(parentId) : [],
    }));

    // Construct examples with ancestors
    const exampleTexts = texts.filter(({ isExample }) => isExample);
    const examples = exampleTexts.map(({ id, text, parentId, highlights }) => ({
      id,
      text,
      codedText: formatTextWithHighlights(text, highlights),
      ancestors: parentId ? getAncestors(parentId) : [],
    }));

    // Start new eval session
    setEvalSession({
      examples: examples,
      results: null
    });

    // Run LLM coding
    const { codedTexts: modelCodedTexts } = await codeInductivelyWithCodeConsistency(
      inputs,
      examples,
      researchQuestion,
      "gpt-4o",
    );

    console.log(modelCodedTexts);

    // Eval against human codes
    const humanCodedTexts = inputTexts.map(({ text, highlights }) => formatTextWithHighlights(text, highlights));
    console.log(humanCodedTexts);

    const embeddingContext = `, in the context of the research question: ${researchQuestion}`;
    const { ious, hausdorffDistances } = await runCodingEval(
      humanCodedTexts,
      modelCodedTexts,
      embeddingContext,
      "text-embedding-3-large"
    );

    // Add model highlights
    inputTexts.forEach(({ id }, idx) => {
      var { textHighlights: modelHighlights } = parseTextHighlights(modelCodedTexts[idx]);
      modelHighlights = modelHighlights.map((hl) => ({ ...hl, type: "model" }));
      setHighlightsForId(id, (hls) => [...hls, ...modelHighlights]);
    });

    // Store results for eval session
    setEvalSession((value) => ({
      ...value,
      results: inputTexts.reduce((acc, { id }, idx) => {
        const { textHighlights: humanHighlights } = parseTextHighlights(humanCodedTexts[idx]);
        const { textHighlights: modelHighlights } = parseTextHighlights(modelCodedTexts[idx]);
        acc[id] = {
          humanHighlights: [],  // humanHighlights.map((hl) => ({ ...hl, type: "human" }))
          modelHighlights: [],  // modelHighlights.map((hl) => ({ ...hl, type: "model" }))
          iou: ious[idx],
          hausdorffDistance: hausdorffDistances[idx]
        };
        return acc;
      }, {})
    }));

    console.log(nanMean(ious), ious);
    console.log(nanMean(hausdorffDistances), hausdorffDistances);
  }

  const setHighlightsForId = (id, updateFunc) => {
    setTexts((prevTexts) =>
      prevTexts.map((t) => (t.id === id ? { ...t, highlights: updateFunc(t.highlights) } : t))
    );
  };

  const createLog = (logData) => {
    console.log("New log: ", logData)

    setEditLog((prevEntries) => {
      // Do not log highlights that never had any codes (that were instantly deleted)
      const prevLog = prevEntries.at(-1);
      if (prevLog && logData.highlight.id === prevLog.highlight.id && logData.event === "delete" && prevLog.event === "create") {
        return prevEntries.slice(0, -1);  // Delete previous log
      }

      // Do not store edit logs where there are no changes to the codes
      const prevLogForHighlight = prevEntries.findLast((prevLog) => logData.highlight.id === prevLog.highlight.id);
      if (prevLogForHighlight && logData.event === "edit" && logData.highlight.codes === prevLogForHighlight.highlight.codes) {
        return prevEntries;
      }
      
      return [
        ...prevEntries,
        {
          ...logData,
          date: new Date()
        },
      ];
    });
  };

  const handleFileUpload = (res) => {
    // Save uploaded data to state
    setFileName(res.fileName);
    setTexts(res.data);
  }

  const handleFileDelete = () => {
    // Restore default values
    setFileName("");
    setTexts([]);
    setEditLog([]);
  }

  const setAnnotated = (id, isAnnotated) => {
    setTexts((prevEntries) => {
      return prevEntries.map((item) => item.id == id ? { ...item, isAnnotated: isAnnotated } : item )
    });
  }

  const setExample = (id, isExample) => {
    setTexts((prevEntries) => {
      return prevEntries.map((item) => item.id == id ? { ...item, isExample: isExample } : item )
    });
  }

  useEffect(() => {
    const handleBeforeUnload = (event) => {
      event.preventDefault();
      event.returnValue = "You may lose all progress on exiting the browser tab. Have you downloaded your codes and logs?";
    };

    window.addEventListener("beforeunload", handleBeforeUnload);

    // Cleanup the event listener on component unmount
    return () => {
      window.removeEventListener("beforeunload", handleBeforeUnload);
    };
  }, []);

  return (
    <div
      style={{
        height: "100vh", // Ensures the container fills the viewport height
        display: "flex", // Flex container
        boxSizing: "border-box",
        overflow: "hidden"
      }}
    >
      {/* Sidebar */}
      <div
        style={{
          width: "230px", // Fixed width
          height: "100%", // Full height of the parent
          backgroundColor: "#f7f7f7",
          borderRight: "1px solid #ddd",
          padding: "20px 30px 20px 30px",
          overflowY: "auto", // Allows scrolling independently
          boxSizing: "border-box",
        }}
      >
        <div
          style={{
            display: "flex",
            flexDirection: "column",
            gap: "2px",
            paddingBottom: "0px",
          }}
        >
          <h2>File</h2>
          {texts.length === 0 ? (
            <FileUpload onUpload={handleFileUpload} />
          ) : (
            <FileManager
              fileName={fileName}
              texts={texts}
              editLog={editLog}
              onDelete={handleFileDelete}
              researchQuestion={researchQuestion}
              setResearchQuestion={setResearchQuestion}
            />
          )}
        </div>

        {texts.length > 0 && (
          <>
            <CodingStats
              texts={texts}
              minAnnotated={40}
              minExamples={3}
              onButtonClick={codeWithLLM}
              apiKey={apiKey}
              setApiKey={setApiKey}
              researchQuestion={researchQuestion}
            />
            <CodeList highlights={texts.map((t) => t.highlights).flat().filter((hl) => hl.type === "human")} focusedOnAny={focusedOnAny} />
          </>
        )}
      </div>

      <div
        style={{
          flex: 1, // Takes up the remaining width of the parent
          display: "flex",
          flexDirection: "row",
          overflow: "hidden", // Prevents content spill
          boxSizing: "border-box",
          height: "100%",
          position: "relative", // Make the parent relative for absolute positioning
          backgroundColor: "#fcfcfa"
        }}
      >
        {/* Coding Pane */}
        <div
          style={{
            flex: 1, // Coding pane takes up all available space
            overflow: "auto", // Enable scrolling if necessary
          }}
        >
          <CodingPane
            texts={texts}
            getAncestors={getAncestors}
            setHighlightsForId={setHighlightsForId}
            focusedOnAny={focusedOnAny}
            createLog={createLog}
            setAnnotated={setAnnotated}
            setExample={setExample}
            evalSession={evalSession}
          />
        </div>
      </div>
    </div>
  );
}

export default App;
