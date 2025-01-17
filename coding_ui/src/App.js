import React, { useState, useEffect } from "react";
import CodingPane from './CodingPane';
import TopBar from "./TopBar";
import CodeList from "./CodeList";
import FileUpload from "./FileUpload";
import './App.css';
import LoadingPane from "./LoadingPane";
import { initializeClient } from "./llmcode/LLM";
import { codeInductivelyWithCodeConsistency } from "./llmcode/Coding";
import { formatTextWithHighlights, splitData, nanMean, parseTextHighlights } from "./helpers";
import { runCodingEval } from "./llmcode/Metrics";
import { NEUTRAL_LIGHT_COLOR } from "./colors";

const MAX_ANCESTORS = 3;

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
    if (savedKey?.submitted) {
      initializeClient(savedKey.key);
    }
    return savedKey ? JSON.parse(savedKey) : { key: "", submitted: false };
  });
  useEffect(() => {
    if (apiKey.submitted) {
      initializeClient(apiKey.key);
    }
    localStorage.setItem("apiKey", JSON.stringify(apiKey));
  }, [apiKey]);

  // Prompt includes research question and any model instructions
  const [prompt, setPrompt] = useState(() => {
    const savedPrompt = localStorage.getItem("prompt");
    return savedPrompt ? JSON.parse(savedPrompt) : { researchQuestion: "", instructions: "" };
  });
  useEffect(() => {
    localStorage.setItem("prompt", JSON.stringify(prompt));
  }, [prompt]);

  const [evalSession, setEvalSession] = useState(() => {
    const savedEvalSession = localStorage.getItem("evalSession");
    return savedEvalSession ? JSON.parse(savedEvalSession) : null;
  });
  useEffect(() => {
    localStorage.setItem("evalSession", JSON.stringify(evalSession));
  }, [evalSession]);

  const getAncestors = (parentId) => {
    const gatherAncestors = (id, depth) => {
      const parentText = allTexts.find((item) => item.id === id);
      if (!parentText || depth >= MAX_ANCESTORS) {
        return parentText ? [parentText.text] : [];
      }
      if (!parentText.parentId) {
        return [parentText.text];
      }
      const ancestors = gatherAncestors(parentText.parentId, depth + 1);
      return [...ancestors, parentText.text];
    };
  
    return gatherAncestors(parentId, 1);
  };

  const [prevAverages, setPrevAverages] = useState(() => {
    const savedData = localStorage.getItem("prevAverages");
    return savedData ? JSON.parse(savedData) : null;
  });
  useEffect(() => {
    localStorage.setItem("prevAverages", JSON.stringify(prevAverages));
  }, [prevAverages]);

  // For user study
  const [allTexts, setAllTexts] = useState(() => {
    const savedTexts = localStorage.getItem("allTexts");
    return savedTexts ? JSON.parse(savedTexts) : [];
  });
  useEffect(() => {
    localStorage.setItem("allTexts", JSON.stringify(allTexts));
  }, [allTexts]);

  // For user study
  const [studyData, setStudyData] = useState(() => {
    const savedData = localStorage.getItem("studyData");
    return savedData ? JSON.parse(savedData) : {};
  });
  useEffect(() => {
    localStorage.setItem("studyData", JSON.stringify(studyData));
  }, [studyData]);

  const codeWithLLM = async () => {
    var textsToCode = [];
    if (evalSession === null) {
      // If first instance of LLM coding, remove all non-annotated texts
      const annotatedTexts = texts.filter(({ isAnnotated }) => isAnnotated);
      setStudyData({
        log: []
      });
      setTexts(annotatedTexts);
      textsToCode = annotatedTexts;
    } else {
      // Save previous averages
      setPrevAverages(evalSession.averages);
      // Remove any previous model highlights
      setTexts((prevTexts) =>
        prevTexts.map((t) => ({ ...t, highlights: t.highlights.filter((hl) => hl.type !== "model") }))
      );
      textsToCode = texts;
    }

    // Construct input texts with ancestors
    const inputTexts = textsToCode.filter(({ isAnnotated, isExample }) => isAnnotated && !isExample);
    const inputs = inputTexts.map(({ text, parentId }) => ({
      text: text,
      ancestors: parentId ? getAncestors(parentId) : [],
    }));

    // Construct examples with ancestors
    const exampleTexts = textsToCode.filter(({ isExample }) => isExample);
    const examples = exampleTexts.map(({ id, text, parentId, highlights }) => ({
      id,
      text,
      codedText: formatTextWithHighlights(text, highlights.filter((hl) => hl.type === "human")),
      ancestors: parentId ? getAncestors(parentId) : [],
    }));

    // Start new eval session
    setEvalSession({
      examples: examples,
      progress: {
        current: 0,
        max: inputs.length,
        frac: 0
      },
      results: null
    });

    // Prepare instructions
    var codingInstructions = prompt.instructions ? (
      `- Ignore text that is not relevant to the research question: ${prompt.researchQuestion}\n${prompt.instructions}`
    ) : (
      `- Ignore text that is not relevant to the research question: ${prompt.researchQuestion}`
    );

    // Run LLM coding
    const { codedTexts: modelCodedTexts } = await codeInductivelyWithCodeConsistency(
      inputs,
      examples,
      prompt.researchQuestion,
      codingInstructions,
      "gpt-4o",
      (progress) => setEvalSession((s) => ({ ...s, progress: progress}))
    );

    console.log(modelCodedTexts);

    // Eval against human codes
    const humanCodedTexts = inputTexts.map(({ text, highlights }) => (
      formatTextWithHighlights(text, highlights.filter((hl) => hl.type === "human"))
    ));
    console.log(humanCodedTexts);

    const embeddingContext = `, in the context of the research question: ${prompt.researchQuestion}`;
    const { ious, hausdorffDistances } = await runCodingEval(
      humanCodedTexts,
      modelCodedTexts,
      embeddingContext,
      "text-embedding-3-large"
    );

     // Note that we use 1 - Hausdorff as code similarity
     const results = inputTexts.reduce((acc, { id }, idx) => {
      acc[id] = {
        highlightSimilarity: ious[idx],
        codeSimilarity: Number.isNaN(hausdorffDistances[idx]) ? NaN : 1 - hausdorffDistances[idx]
      };
      return acc;
    }, {});
    const avgHighlightSimilarity = nanMean(ious);
    const avgCodeSimilarity = nanMean(hausdorffDistances.map(d => Number.isNaN(d) ? NaN : 1 - d));

    // Log study data
    const inputsWithAnnotations = inputTexts.map(({ id }, idx) => {
      const modelCodedText = modelCodedTexts[idx];
      var modelHighlights = [];
      if (modelCodedText !== null) {
        var { textHighlights: modelHighlights } = parseTextHighlights(modelCodedText);
        modelHighlights = modelHighlights.map((hl) => ({ ...hl, type: "model" }));
      }
      const humanHighlights = texts.find((t) => t.id === id).highlights.filter((hl) => hl.type === "human")
      return {
        id,
        humanHighlights,
        modelHighlights
      }
    });
    setStudyData((data) => ({
      ...data,
      log: [
        ...data.log,
        {
          inputsWithAnnotations,
          examples,
          results,
          researchQuestion: prompt.researchQuestion,
          date: new Date()
        }
      ]
    }));

    // Add model highlights to UI
    inputTexts.forEach(({ id }, idx) => {
      const modelCodedText = modelCodedTexts[idx];
      if (modelCodedText !== null) {
        var { textHighlights: modelHighlights } = parseTextHighlights(modelCodedText);
        modelHighlights = modelHighlights.map((hl) => ({ ...hl, type: "model" }));
        setHighlightsForId(id, (hls) => [...hls, ...modelHighlights]);
      }
    });

    // Store results for eval session
    setEvalSession((value) => ({
      ...value,
      results: results,
      averages: {
        highlightSimilarity: avgHighlightSimilarity,
        codeSimilarity: avgCodeSimilarity
      }
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
    setAllTexts(res.data);
  }

  const handleFileDelete = () => {
    // Restore default values
    setFileName("");
    setTexts([]);
    setEditLog([]);
    setEvalSession(null);
    setPrevAverages(null);
    setStudyData({});
    setPrompt({ researchQuestion: "", instructions: "" });
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
        height: "100vh",
        display: "flex", 
        flexDirection: "column",
        boxSizing: "border-box",
        overflow: "hidden"
      }}
    >
      <TopBar
        texts={texts}
        onUpload={handleFileUpload}
        fileName={fileName}
        editLog={editLog}
        onDelete={handleFileDelete}
        apiKey={apiKey}
        setApiKey={setApiKey}
        studyData={studyData}
      />

      <div
        style={{
          height: "100vh",
          display: "flex",
          boxSizing: "border-box",
          overflow: "hidden",
          position: "relative", // Important for positioning the floating pane
        }}
      >
        {/* Central Coding View */}
        <div
          style={{
            flex: 1, // Takes up the remaining width of the parent
            display: "flex",
            flexDirection: "row",
            overflow: "hidden",
            boxSizing: "border-box",
            height: "100%",
            backgroundColor: "#fcfcfa",
          }}
        >
          {texts.length === 0 ? (
            <FileUpload onUpload={handleFileUpload} />
          ) : (
            <div
              style={{
                flex: 1, // Coding pane takes up all available space
                overflow: "auto", // Enable scrolling if necessary
                backgroundColor: NEUTRAL_LIGHT_COLOR,
              }}
            >
              {!evalSession || evalSession.results ? (
                <CodingPane
                  texts={texts}
                  getAncestors={getAncestors}
                  setHighlightsForId={setHighlightsForId}
                  focusedOnAny={focusedOnAny}
                  createLog={createLog}
                  setAnnotated={setAnnotated}
                  setExample={setExample}
                  evalSession={evalSession}
                  prevAverages={prevAverages}
                  apiKey={apiKey}
                  setApiKey={setApiKey}
                  prompt={prompt}
                  setPrompt={setPrompt}
                  codeWithLLM={codeWithLLM}
                />
              ) : (
                <LoadingPane progress={evalSession.progress} />
              )}
            </div>
          )}
        </div>
      </div>
    </div>
  );
}

export default App;
