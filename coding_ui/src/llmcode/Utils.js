export function parseCodes(codedText) {
	if (!codedText) {
		return [];
	}

	// Use a regular expression to extract highlights and codes
	const matches = [...codedText.matchAll(/\*\*(.*?)\*\*<sup>(.*?)<\/sup>/gs)];
	
	// Flatten and process matches into highlight-code pairs
	return matches.flatMap(([_, highlight, codes]) =>
		codes
			.split(";") // Split codes by semicolon
			.map((code) => code.trim()) // Remove extra spaces
			.filter((code) => code !== "") // Exclude empty codes
			.map((code) => [highlight, code]) // Pair highlight with each code
	);
}
