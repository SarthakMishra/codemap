
# Implementation Plan for CodeStruct VS Code Extension

## 1. Setup and Dependencies

```bash
# Install required tools
pip install textX[cli]
pip install git+https://github.com/IgorMaj/SyntaxColoring.git
```

## 2. Create an EasyColorLang (.eclr) grammar file for CodeStruct

Create a file `codestruct.eclr` with pattern definitions for CodeStruct syntax:

```
// Keywords pattern
#keywords:
  match: "\b(dir|file|module|namespace|class|func|lambda|attr|param|returns|var|const|type_alias|union|optional|import|doc)\b:" name: "keyword.other.codestruct"

// Short keywords pattern (for minified version)
#short_keywords:
  match: "\b(d|f|m|cl|fn|at|p|r|v|c|i|dc|t|s|rf)\b:" name: "keyword.other.codestruct.minified"

// Attribute pattern
#attributes:
  begin: "\\[" names: "punctuation.definition.attributes.begin.codestruct"
  end: "\\]" names: "punctuation.definition.attributes.end.codestruct"
  name: "meta.attributes.codestruct" (
    match: "\b(type|default|source|ref)\b:" name: "entity.other.attribute-name.codestruct"
    match: "(,)" name: "punctuation.separator.attribute.codestruct"
    include: #strings
    include: #numbers
    include: #boolean
  )

// String pattern
#strings:
  begin: "\"" names: "punctuation.definition.string.begin.codestruct"
  end: "\"" names: "punctuation.definition.string.end.codestruct"
  name: "string.quoted.double.codestruct"

  begin: "'" names: "punctuation.definition.string.begin.codestruct"
  end: "'" names: "punctuation.definition.string.end.codestruct"
  name: "string.quoted.single.codestruct"

// Number pattern
#numbers:
  match: "-?\\d+(\\.\\d+)?" name: "constant.numeric.codestruct"

// Boolean pattern
#boolean:
  match: "\b(true|false)\b" name: "constant.language.boolean.codestruct"

// Comment pattern
#comments:
  match: "#[^\n]*" name: "comment.line.number-sign.codestruct"

// Indentation pattern
#indentation:
  match: "^[ \t]+" name: "meta.whitespace.indentation.codestruct"

// Entity name pattern
#entity_names:
  match: "(?<=:)[ \t]+([a-zA-Z0-9_./\\-]+)" name: "entity.name.codestruct"

// Minified separator pattern
#minified_separators:
  match: "(;)" name: "punctuation.separator.entity.codestruct.minified"
  match: "(\\|)" name: "punctuation.separator.child.codestruct.minified"
  match: "(,)" name: "punctuation.separator.attribute.codestruct.minified"

start codestruct.source(keywords, short_keywords, attributes, strings, numbers, boolean, comments, indentation, entity_names, minified_separators)
```

### 3. Generate TextMate Grammar JSON

```bash
textx generate --target textmate codestruct.eclr --output-path codestruct.tmLanguage.json
```

### 4. Create VS Code Extension Structure

```bash
# Create extension directory
mkdir -p codestruct-vscode/syntaxes

# Copy the generated grammar
cp codestruct.tmLanguage.json codestruct-vscode/syntaxes/
```

### 5. Create package.json for the Extension

```bash
cd codestruct-vscode
# Create package.json with extension metadata
```

Create a `package.json` file:

```json
{
  "name": "codestruct",
  "displayName": "CodeStruct",
  "description": "Syntax highlighting for CodeStruct notation",
  "version": "0.1.0",
  "engines": {
    "vscode": "^1.60.0"
  },
  "categories": [
    "Programming Languages"
  ],
  "contributes": {
    "languages": [
      {
        "id": "codestruct",
        "aliases": ["CodeStruct", "codestruct"],
        "extensions": [".cs.md", ".codestruct"],
        "configuration": "./language-configuration.json"
      }
    ],
    "grammars": [
      {
        "language": "codestruct",
        "scopeName": "codestruct.source",
        "path": "./syntaxes/codestruct.tmLanguage.json"
      }
    ],
    "iconThemes": [
      {
        "id": "codestruct-icons",
        "label": "CodeStruct Icons",
        "path": "./icons/icons-theme.json"
      }
    ]
  }
}
```

### 6. Add Language Configuration

Create `language-configuration.json`:

```json
{
  "comments": {
    "lineComment": "#"
  },
  "brackets": [
    ["[", "]"]
  ],
  "autoClosingPairs": [
    { "open": "[", "close": "]" },
    { "open": "\"", "close": "\"" },
    { "open": "'", "close": "'" }
  ],
  "surroundingPairs": [
    ["[", "]"],
    ["\"", "\""],
    ["'", "'"]
  ],
  "indentationRules": {
    "increaseIndentPattern": "^\\s*\\w+:",
    "decreaseIndentPattern": "^\\s*$"
  }
}
```

### 7. Create File Icon Theme

```bash
mkdir -p icons/fileicons
```

Create an icon file (`codestruct.svg`) in the `icons/fileicons` directory.

Create `icons/icons-theme.json`:

```json
{
  "iconDefinitions": {
    "_codestruct": {
      "iconPath": "./fileicons/codestruct.svg"
    }
  },
  "fileExtensions": {
    "cs.md": "_codestruct",
    "codestruct": "_codestruct"
  }
}
```

### 8. Package and Install the Extension

```bash
# Install vsce (VS Code Extension Manager)
npm install -g @vscode/vsce

# Package the extension
vsce package

# Install the extension
code --install-extension codestruct-0.1.0.vsix
```

### Benefits of This Approach

1. **Simplicity**: Using EasyColorLang makes it easier to define TextMate grammar compared to writing JSON directly
2. **Maintainability**: The grammar is more readable and easier to update
3. **Integration**: TextMate grammars work well with VS Code's built-in syntax highlighting engine

This lightweight extension will provide syntax highlighting and file icons for CodeStruct files without requiring complex tree-sitter integration.
