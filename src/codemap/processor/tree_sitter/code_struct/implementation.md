
# Implementation Plan for CodeStruct VS Code Extension

## 1. Create an TextMate grammar file for CodeStruct

Create a file `codestruct.tmLanguage.json` with pattern definitions for CodeStruct syntax:

### 2. Create package.json for the Extension

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

### 3. Add Language Configuration

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

### 4. Create File Icon Theme

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

### 5. Package and Install the Extension

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
